from definitions import logger

# SCRIPT SETTINGS:

DATA_PROCESSING = True
TILE_WIDTH = 25600
TILE_HEIGHT = 25600


# ML VARIABLES:

EPOCH = 5
BATCH_SIZE = 2
TARGET_SIZE = [224, 224]


def main(mode: str) -> None:
    if mode == "Preprocessing":
        import preprocessing

        logger.info("Working Mode: Preprocess the data")
        preprocessing.preprocessing_data()
    elif mode == "Unet":
        import unet

        logger.info("Working Mode: Train the model")
        unet.unet_fit()
    else:
        raise ValueError("Working mode not correctly set")


if __name__ == "__main__":
    working_mode = "Preprocessing"
    #working_mode = "Unet"
    main(working_mode)
