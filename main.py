from definitions import logger

import preprocessing
import unet

# SCRIPT SETTINGS:

TILE_WIDTH = 25600
TILE_HEIGHT = 25600


# ML VARIABLES:

EPOCH = 15
BATCH_SIZE = 2
TARGET_SIZE = [224, 224]


def main(mode: str) -> None:
    if mode == "Preprocessing":
        logger.info("Working Mode: Preprocess the data")
        preprocessing.preprocessing_data()
    elif mode == "Unet":
        logger.info("Working Mode: Train the model and predict")
        unet.unet_execution()
    elif mode == "Complete":
        logger.info("Working Mode: Complete run, including preprocessing training and predicting")

        logger.info("Doing the preprocessing")
        preprocessing.preprocessing_data()

        logger.info("Training the model and predict the raster")
        unet.unet_execution()
    else:
        raise ValueError("Working mode not correctly set")


if __name__ == "__main__":
    #working_mode = "Preprocessing"
    #working_mode = "Unet"
    working_mode = "Complete"
    main(working_mode)
