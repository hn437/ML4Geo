from preprocessing import preprocessing
from unet import unet

# SCRIPT SETTINGS:

DATA_PROCESSING = True
TILE_WIDTH = 2052
TILE_HEIGHT = 2052

# ML VARIABLES:

EPOCH = 10
BATCH_SIZE = 10
TARGET_SIZE = [224, 224]

if __name__ == "__main__":
    preprocessing()
    unet(batch_size=BATCH_SIZE, target_size=TARGET_SIZE, epoch=EPOCH)
