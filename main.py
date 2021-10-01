"""
This script sets preprocessing- and CNN-training variables and starts a run of either
the whole or parts of the workflow
"""
from definitions import logger

# SCRIPT SETTINGS:

# Defining whether to use the new, working or the old, faulty workflow (in order to
#  reproduce the failed experiment) for preprocessing
NEW_WORKFLOW = True
# the tile size which should be processed at once in preprocessing
TILE_WIDTH = 25600
TILE_HEIGHT = 25600


# ML VARIABLES:

EPOCH = 100
BATCH_SIZE = 10
TARGET_SIZE = [256, 256]


def main(mode: str) -> None:
    if mode == "Preprocessing":
        import preprocessing

        logger.info("Working Mode: Preprocess the data")
        preprocessing.preprocessing_data()
    elif mode == "Unet":
        import unet

        logger.info("Working Mode: Train the model and predict")
        unet.unet_execution()
    elif mode == "Complete":
        import preprocessing
        import unet

        logger.info(
            "Working Mode: Complete run, including preprocessing training and predicting"
        )

        logger.info("Doing the preprocessing")
        preprocessing.preprocessing_data()

        logger.info("Training the model and predict the raster")
        unet.unet_execution()
    else:
        raise ValueError("Working mode not correctly set")


if __name__ == "__main__":
    working_mode = "Complete"
    main(working_mode)
