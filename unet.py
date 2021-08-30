import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from definitions import INTERMEDIATE_PATH, TEST_PATH_IMG, logger
from main import BATCH_SIZE, EPOCH, TARGET_SIZE
from model import build_model, get_generator, get_generator_test


def plot_fit_progress(history) -> None:
    plt.figure()
    plt.plot(range(EPOCH), history.history["accuracy"], label="Training Accuracy")
    plt.plot(range(EPOCH), history.history["val_accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(INTERMEDIATE_PATH, "Progress_per_Epoch.png"))
    logger.info("Fitting progress saved in png-File.")


def unet_evaluate() -> None:
    model = build_model(target_size=TARGET_SIZE)
    model.load_weights(os.path.join(INTERMEDIATE_PATH, "weights.h5"))

    no_of_testcases = len(os.listdir(TEST_PATH_IMG))

    train_gen, test_gen, no_of_trainsets, no_of_validsets = get_generator(
        batch_size=5, target_size=TARGET_SIZE
    )

    confusion_array = np.array([[0, 0], [0, 0]])

    counter = 0

    for image, mask in test_gen:
        predictions = model.predict(image, batch_size=1, workers=7)
        predictions = predictions.flatten()
        mask = mask.flatten()
        confusion_array += confusion_matrix(mask, predictions)
        counter += 1
        if counter > 10:
            break

    print(confusion_array)


def unet_fit() -> None:
    # Setting up generator
    train_gen, test_gen, no_of_trainsets, no_of_validsets = get_generator(
        batch_size=BATCH_SIZE, target_size=TARGET_SIZE
    )
    # Build model and compile
    model = build_model(target_size=TARGET_SIZE)

    logger.info("Fit model...")
    history = model.fit(
        train_gen,
        validation_data=test_gen,
        steps_per_epoch=10,#(no_of_trainsets // BATCH_SIZE),
        epochs=EPOCH,
        validation_steps=10,#(no_of_validsets // BATCH_SIZE),
        workers=7,
        use_multiprocessing=False
    )
    model.save_weights(os.path.join(INTERMEDIATE_PATH, "weights.h5"))
    logger.info("Model weights saved!")

    plot_fit_progress(history)

    unet_evaluate()


if __name__ == "__main__":
    #unet_fit()
    unet_evaluate()
