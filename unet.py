import os

import matplotlib.pyplot as plt
import tensorflow.python.keras.callbacks

from definitions import INTERMEDIATE_PATH, logger
from main import BATCH_SIZE, EPOCH, TARGET_SIZE
from model import build_model, get_generator


def plot_fit_progress(history: tensorflow.python.keras.callbacks.History) -> None:
    plt.figure()
    plt.plot(range(EPOCH), history.history["accuracy"], label="Training Accuracy")
    plt.plot(range(EPOCH), history.history["val_accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(INTERMEDIATE_PATH, "Progress_per_Epoch.png"))
    logger.info("Fitting progress saved in png-File.")


def calculate_confusion_matrix() -> None:
    print()


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
        steps_per_epoch=(no_of_trainsets // BATCH_SIZE),
        epochs=EPOCH,
        validation_steps=(no_of_validsets // BATCH_SIZE),
    )
    model.save_weights(os.path.join(INTERMEDIATE_PATH, "weights.h5"))
    logger.info("Model weights saved!")

    plot_fit_progress(history)
    calculate_confusion_matrix()


if __name__ == "__main__":
    unet_fit()