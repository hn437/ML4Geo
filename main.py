import os
import geopandas
import rasterio
from rasterio import mask

buildings_path = None
raster_path = None

training_path = "training_data"
test_path = "test_data"


def setup_folders():
    if not os.path.exists(training_path):
        os.mkdir(training_path)
    if not os.path.exists(test_path):
        os.mkdir(test_path)



def generate_mask(raster, vector):
    r_mask = mask.mask(raster, vector, all_touched=False, invert=False, nodata=None, filled=True, crop=False,
                       pad=False, pad_width=0.5, indexes=None)
    return r_mask


def create_ml_data(raster, r_mask, vector):
    row_count = vector.shape[0]
    train_max = round(row_count * 0.8)


    for index, row in vector.iterrows():
        cropped_raster = mask.mask(raster, row, crop=True)
        cropped_mask = mask.mask(r_mask, row, crop=True)



def main():
    setup_folders()

    buildings = geopandas.read_file(buildings_path)
    raster = rasterio.open(raster_path)


    r_mask = generate_mask(raster, buildings)
    bounds = buildings.bounds

