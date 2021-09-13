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

    for image, mask in tqdm(test_gen, total=no_of_testsets):
        predictions = model.predict(image, batch_size=1, workers=1)
        predictions = predictions.flatten()
        mask = mask.flatten()
        mask_total = np.concatenate((mask_total, mask))
        pred_total = np.concatenate((pred_total, predictions))

        counter_processed_files += 1
        if counter_processed_files >= no_of_testsets:
            break

    with open(os.path.join(RESULT_PATH, f"{mode}_ground_truth.json"), "w") as file:
        json.dump(mask_total.tolist(), file)
    with open(os.path.join(RESULT_PATH, f"{mode}_prediction.json"), "w") as file:
        json.dump(pred_total.tolist(), file)

    if mode == "test":
        fpr_total, tpr_total, thresholds_batch = roc_curve(mask_total, pred_total)
        auc_keras = auc(fpr_total, tpr_total)

        x = np.array([[0, 1]])
        y = np.array([fpr_total, tpr_total]).transpose()
        d = cdist(x, y)
        idx = np.argmin(d)

        threshold = thresholds_batch[idx]
        print("\nOptimum threshold:", threshold)
        update_json("threshold", threshold)

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
    mask_total, pred_total = determine_class_threshold(mode)
    with open(os.path.join(RESULT_PATH, "metrics.json"), "r") as file:
        threshold = json.load(file)["threshold"]
    pred_total = np.where(pred_total < threshold, 0, 1)
    cm = confusion_matrix(mask_total, pred_total)

    print(f"Confusion Matrix {mode}\n", cm)
    TP = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[1][1]

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

    for window, transform in tqdm(
        get_tiles(raster, TARGET_SIZE[0], TARGET_SIZE[1]), total=tiles_to_be_predicted
    ):
        padding_mode = False
        tile_data = raster.read(window=window, boundless=True, fill_value=raster.nodata)
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

        tile_data = np.moveaxis(tile_data, 0, 2)
        tile_data = tf.expand_dims(tile_data, axis=0)

        predicted_tile = model.predict(tile_data, batch_size=1, workers=1)
        predicted_tile = np.where(predicted_tile < threshold, 0, 1)
        predicted_tile = tf.squeeze(predicted_tile, axis=0)
        predicted_tile = np.moveaxis(predicted_tile, 2, 0)

        if padding_mode is True:
            predicted_tile = predicted_tile[
                :, 0 : orig_tile_size[1], 0 : orig_tile_size[2]
            ]

        if os.path.exists(os.path.join(RESULT_PATH, "predicted_raster.tif")):
            with rasterio.open(
                os.path.join(RESULT_PATH, "predicted_raster.tif"),
                "r+",
                BIGTIFF="YES",
                **out_meta,
            ) as outds:
                outds.write(predicted_tile, window=window)
        else:
            with rasterio.open(
                os.path.join(RESULT_PATH, "predicted_raster.tif"),
                "w",
                BIGTIFF="YES",
                **out_meta,
            ) as outds:
                outds.write(predicted_tile, window=window)
    logger.setLevel(logging.INFO)


def unet_fit() -> None:
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

    plot_fit_progress(history)


def unet_execution() -> None:
    unet_fit()
    unet_evaluate(mode="test")
    unet_evaluate(mode="valid")
    predict_raster()


if __name__ == "__main__":
    unet_execution()
