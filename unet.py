"""
this script trains the model, evaluates it and predicts the original raster
"""
import json
import logging
import math
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import tensorflow as tf
from scipy.spatial.distance import cdist
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
from tensorflow.keras.callbacks import ModelCheckpoint
from tqdm import tqdm

from definitions import DATA_PATH, INTERMEDIATE_PATH, RASTER_PATH, RESULT_PATH, logger
from main import BATCH_SIZE, EPOCH, TARGET_SIZE
from model import build_model, get_generator
from utils import get_tiles, update_json


def plot_fit_progress(history) -> None:
    """This function plots the accuracies ofer epochs while training the model"""
    plt.figure()
    plt.plot(range(EPOCH), history.history["accuracy"], label="Training Accuracy")
    plt.plot(range(EPOCH), history.history["val_accuracy"], label="Test Accuracy")
    plt.legend()
    plt.title("Progress per Epoch")
    plt.tight_layout()
    plt.savefig(os.path.join(INTERMEDIATE_PATH, "Progress_per_Epoch.png"))
    plt.close()
    logger.info("Fitting progress saved in png-File.")


def determine_class_threshold(mode) -> tuple:
    """
    This function determines the optimum threshold which defines whether a pixel
        belongs to a building or not
    :param mode: whether this function is used for testing or on the validation dataset
    :return: all predicted and all actual results as a tuple of two arrays
    """
    model = build_model(target_size=TARGET_SIZE)
    model.load_weights(os.path.join(RESULT_PATH, "weights.h5"))

    _, test_gen, _, no_of_testsets = get_generator(
        batch_size=BATCH_SIZE, target_size=TARGET_SIZE, mode=f"{mode}_data"
    )
    # make usable by Tensorflow 2.0
    test_gen = (pair for pair in test_gen)

    counter_processed_files = 0
    mask_total = np.array([])
    pred_total = np.array([])

    # predict all tiles in the test dir and store the results and the actual masks in an
    # array
    for image, mask in tqdm(test_gen, total=no_of_testsets):
        predictions = model.predict(image, batch_size=1, workers=1)
        predictions = predictions.flatten()
        mask = mask.flatten()
        mask_total = np.concatenate((mask_total, mask))
        pred_total = np.concatenate((pred_total, predictions))

        counter_processed_files += 1
        if counter_processed_files >= no_of_testsets:
            break

    if mode == "test":
        # use roc_curve to determine threshold
        fpr_total, tpr_total, thresholds_batch = roc_curve(mask_total, pred_total)
        auc_keras = auc(fpr_total, tpr_total)

        # choosing point which is closest to point(0,1)
        x = np.array([[0, 1]])
        y = np.array([fpr_total, tpr_total]).transpose()
        d = cdist(x, y)
        idx = np.argmin(d)

        # store this threshold in json
        threshold = thresholds_batch[idx]
        print("\nOptimum threshold:", threshold)
        update_json("threshold", threshold)

        # plot roc curve and chosen threshold in figure
        plt.figure(1)
        plt.plot([0, 1], [0, 1], "k--")
        plt.plot(fpr_total, tpr_total, label="Keras (area = {:.3f})".format(auc_keras))
        plt.scatter(
            y[idx][0],
            y[idx][1],
            c="black",
            label=f"Optimum Threshold (= {round(threshold, 3)})",
        )
        plt.text(y[idx][0], y[idx][1], f"{round(threshold, 3)}")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title("ROC curve")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(os.path.join(INTERMEDIATE_PATH, "ROC.png"))
        plt.close()

    return mask_total, pred_total


def unet_evaluate(mode) -> None:
    """This function evaluates the training of the unet on test or validation data"""
    # get prediction results
    mask_total, pred_total = determine_class_threshold(mode)
    # load threshold from json and binarily reclassify prediction results using it
    with open(os.path.join(RESULT_PATH, "metrics.json"), "r") as file:
        threshold = json.load(file)["threshold"]
    pred_total = np.where(pred_total < threshold, 0, 1)

    # calculate confusion matrix and other precision metrics. store them in json
    cm = confusion_matrix(mask_total, pred_total)

    print(f"Confusion Matrix {mode}\n", cm)
    TP = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[1][1]
    dict_cm = {"TP": int(TP), "FP": int(FP), "FN": int(FN), "TN": int(TN)}
    update_json(f"{mode}_ConfMat", dict_cm)

    print(
        f"Classification Report {mode}\n", classification_report(mask_total, pred_total)
    )

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    f1score = 2 * TP / (2 * TP + FP + FN)

    update_json(f"{mode}_precision", precision)
    update_json(f"{mode}_recall", recall)
    update_json(f"{mode}_accuracy", accuracy)
    update_json(f"{mode}_f1score", f1score)


def predict_raster() -> None:
    """This function predicts buildings in the original raster"""
    with open(os.path.join(RESULT_PATH, "metrics.json"), "r") as file:
        threshold = json.load(file)["threshold"]
    raster = rasterio.open(RASTER_PATH)

    if os.path.exists(os.path.join(RESULT_PATH, "predicted_raster.tif")):
        os.remove(os.path.join(RESULT_PATH, "predicted_raster.tif"))
    out_meta = raster.meta.copy()
    out_meta.update(
        {
            "driver": "GTiff",
            "count": int(1),
            "dtype": "uint8",
            "compress": "lzw",
            "nodata": 0,
        }
    )

    model = build_model(target_size=TARGET_SIZE)
    model.load_weights(os.path.join(RESULT_PATH, "weights.h5"))

    tiles_to_be_predicted = math.ceil(
        (out_meta["width"] * out_meta["height"]) / (TARGET_SIZE[0] * TARGET_SIZE[1])
    )

    rasterio_logger = rasterio.logging.getLogger()
    rasterio_logger.setLevel(logging.ERROR)
    logger.setLevel(logging.ERROR)

    # getting tile of raster
    for window, transform in tqdm(
        get_tiles(raster, TARGET_SIZE[0], TARGET_SIZE[1]), total=tiles_to_be_predicted
    ):
        padding_mode = False
        tile_data = raster.read(window=window, boundless=True, fill_value=raster.nodata)
        # rescale and padd tile if it doesn't match the target size defined in main.py
        tile_data = tile_data * (1.0 / 255.0)
        if tile_data.shape[1] < TARGET_SIZE[0]:
            orig_tile_size = tile_data.shape
            t = TARGET_SIZE[0] - tile_data.shape[1]
            tile_data = np.pad(tile_data, ((0, 0), (0, t), (0, 0)), constant_values=0)
            padding_mode = True
        if tile_data.shape[2] < TARGET_SIZE[1]:
            orig_tile_size = tile_data.shape
            t = TARGET_SIZE[1] - tile_data.shape[2]
            tile_data = np.pad(tile_data, ((0, 0), (0, 0), (0, t)), constant_values=0)
            padding_mode = True

        # prepare data to be predicted by changing it's format
        tile_data = np.moveaxis(tile_data, 0, 2)
        tile_data = tf.expand_dims(tile_data, axis=0)

        # predict and classify using the threshold, change back to original format
        predicted_tile = model.predict(tile_data, batch_size=1, workers=1)
        predicted_tile = np.where(predicted_tile < threshold, 0, 1)
        predicted_tile = tf.squeeze(predicted_tile, axis=0)
        predicted_tile = np.moveaxis(predicted_tile, 2, 0)

        # if tile needed to be padded, "remove" padded area
        if padding_mode is True:
            predicted_tile = predicted_tile[
                :, 0 : orig_tile_size[1], 0 : orig_tile_size[2]
            ]

        # write predicted tile to predicted raster
        if os.path.exists(os.path.join(RESULT_PATH, "predicted_raster.tif")):
            with rasterio.open(
                os.path.join(RESULT_PATH, "predicted_raster.tif"),
                "r+",
                BIGTIFF="YES",
                **out_meta,
            ) as outds:
                outds.write(predicted_tile, window=window)
                outds.close()
        else:
            with rasterio.open(
                os.path.join(RESULT_PATH, "predicted_raster.tif"),
                "w",
                BIGTIFF="YES",
                **out_meta,
            ) as outds:
                outds.write(predicted_tile, window=window)
                outds.close()
    logger.setLevel(logging.INFO)


def unet_fit() -> None:
    """This function trains the model"""
    # Setting up generator
    train_gen, test_gen, no_of_trainsets, no_of_testsets = get_generator(
        batch_size=BATCH_SIZE, target_size=TARGET_SIZE, mode="test_data"
    )
    # make usable by Tensorflow 2.0
    train_gen = (pair for pair in train_gen)
    test_gen = (pair for pair in test_gen)

    # Build model and compile
    model = build_model(target_size=TARGET_SIZE)

    # training logging
    logdir = os.path.join(DATA_PATH, "logs") + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=logdir, histogram_freq=1
    )
    # checkpoint
    checkpoint = ModelCheckpoint(
        os.path.join(logdir, "checkpoint"),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
    )

    # train model und save weights
    logger.info("Fit model...")
    history = model.fit(
        train_gen,
        validation_data=test_gen,
        steps_per_epoch=(no_of_trainsets // BATCH_SIZE),
        epochs=EPOCH,
        validation_steps=(no_of_testsets // BATCH_SIZE),
        workers=1,
        use_multiprocessing=False,
        callbacks=[checkpoint, tensorboard_callback],
    )
    model.save_weights(os.path.join(RESULT_PATH, "weights.h5"))
    logger.info("Model weights saved!")

    # plot progess in figure
    plot_fit_progress(history)


def unet_execution() -> None:
    """ This function executes all other function in correct order"""
    unet_fit()
    unet_evaluate(mode="test")
    unet_evaluate(mode="valid")
    predict_raster()


if __name__ == "__main__":
    unet_execution()
