import os
import geopandas as gpd
import rasterio
from rasterio import mask


from definitions import TRAINING_PATH, TEST_PATH, RASTER_PATH
from utils import query
buildings_path = None

buil_def = {
    "description":"All Buildings in an Area",
    "endpoint": "elements/geometry",
    "filter": """
         building = *
    """
}

def setup_folders():
    if not os.path.exists(TRAINING_PATH):
        os.mkdir(TRAINING_PATH)
    if not os.path.exists(TEST_PATH):
        os.mkdir(TEST_PATH)


def get_building_data(raster):
    bbox = raster.bounds
    buildings = query(buil_def, bbox)
    return buildings


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
    raster = rasterio.open(RASTER_PATH)
    buildings = get_building_data(raster)
    buildings = gpd.GeoDataFrame.from_features(buildings["features"])

    r_mask = generate_mask(raster, buildings)
    bounds = buildings.bounds

main()