from definitions import logger


# SCRIPT SETTINGS:

NEW_WORKFLOW = True
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
        logger.info("Working Mode: Complete run, including preprocessing training and predicting")

        logger.info("Doing the preprocessing")
        preprocessing.preprocessing_data()

        logger.info("Training the model and predict the raster")
        unet.unet_execution()
    else:
        raise ValueError("Working mode not correctly set")


if __name__ == "__main__":
    #working_mode = "Preprocessing"
    working_mode = "Unet"
    #working_mode = "Complete"
    main(working_mode)
