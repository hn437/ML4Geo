"""
This script creates training, test, and validation data based on the input raster and
OSM data
"""
import json
import logging
import math
import os

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import mask
from shapely.geometry import box
from skimage.filters import median
from skimage.morphology import disk
from tqdm import tqdm

from definitions import (
    INTERMEDIATE_PATH,
    RASTER_PATH,
    TEST_PATH,
    TEST_PATH_IMG,
    TEST_PATH_MASK,
    TRAINING_PATH,
    TRAINING_PATH_IMG,
    TRAINING_PATH_MASK,
    VALIDATION_PATH,
    logger,
)
from main import NEW_WORKFLOW, TARGET_SIZE, TILE_HEIGHT, TILE_WIDTH
from utils import get_tiles, query, write_raster_window

# defining the ohsome query
buil_def = {
    "description": "All Buildings in an Area",
    "endpoint": "elements/geometry",
    "filter": """
         building = * and geometry:polygon and building != roof
    """,
}


def get_building_data(raster) -> gpd.GeoDataFrame:
    """
    This function queries OSM buildings within the bounding box of the raster and
        reprojects them into the rasters' projection
    :param raster: the input raster which is used for ML
    :return: geopandas' dataframe of the OSM buildings
    """
    bbox = box(*raster.bounds)
    bbox = gpd.GeoSeries([bbox]).set_crs(raster.crs).to_crs(epsg=4326).__geo_interface__
    bbox = json.dumps(bbox)
    buildings = query(buil_def, bbox)
    buildings = (
        gpd.GeoDataFrame.from_features(buildings["features"])
        .set_crs(epsg=4326)
        .to_crs(crs=raster.crs)
    )
    buildings.to_file(
        os.path.join(INTERMEDIATE_PATH, "buildings.geojson"), driver="GeoJSON"
    )
    return buildings


def generate_mask(raster, vector) -> None:
    """
    This function generates a binary mask for a raster indicating where vector features
        overlay the raster
    :param raster: the raster the mask shoul be generated for
    :param vector: the vector features which should be used for mask creating
    :return: None. The mask will be saved as file to the harddrive in order to use
        tile-by-tile creation
    """
    if os.path.exists(os.path.join(INTERMEDIATE_PATH, "masked_raster.tif")):
        os.remove(os.path.join(INTERMEDIATE_PATH, "masked_raster.tif"))
    out_meta = raster.meta.copy()
    out_meta.update(
        {
            "driver": "GTiff",
            "count": int(1),
        }
    )

    tiles_needed = math.ceil(
        (out_meta["width"] * out_meta["height"]) / (TILE_WIDTH * TILE_HEIGHT)
    )
    # change logging level as rasterio always warns when using BIGTIFF option in "r+"
    rasterio_logger = rasterio.logging.getLogger()
    rasterio_logger.setLevel(logging.ERROR)
    logger.setLevel(logging.ERROR)
    for window, transform in tqdm(
        # get a tile of the original raster
        get_tiles(raster, TILE_WIDTH, TILE_HEIGHT),
        total=tiles_needed,
    ):
        meta_tile = raster.meta.copy()
        meta_tile["transform"] = transform
        meta_tile["width"], meta_tile["height"] = window.width, window.height

        tiledata = raster.read(window=window)
        with rasterio.open(
            os.path.join(INTERMEDIATE_PATH, f"tile.tif"), "w", **meta_tile
        ) as dest:
            dest.write(tiledata)
        with rasterio.open(os.path.join(INTERMEDIATE_PATH, f"tile.tif"), "r") as tile:
            out_image, out_transform = mask.mask(
                tile,
                vector,
                nodata=None,
                all_touched=False,
                invert=False,
                filled=True,
                crop=False,
                pad=False,
                pad_width=0.5,
                indexes=None,
            )
        # create single band from triple band and classify binary
        out_image = np.sum(out_image, axis=0, dtype="uint8")
        out_image = np.where(out_image > 0, 1, out_image)
        # reduce salt-n-pepper noise
        out_image = median(out_image, disk(1), mode="constant", cval=0)
        out_image = np.array([out_image])

        # write tile in resulting mask raster
        if os.path.exists(os.path.join(INTERMEDIATE_PATH, "masked_raster.tif")):
            with rasterio.open(
                os.path.join(INTERMEDIATE_PATH, "masked_raster.tif"),
                "r+",
                BIGTIFF="YES",
                **out_meta,
            ) as outds:
                outds.write(out_image, window=window)
        else:
            with rasterio.open(
                os.path.join(INTERMEDIATE_PATH, "masked_raster.tif"),
                "w",
                BIGTIFF="YES",
                **out_meta,
            ) as outds:
                outds.write(out_image, window=window)
    # set logging level back to info
    logger.setLevel(logging.INFO)


def crop_and_save(raster, bbox_feature, path, counter) -> bool:
    """Deprecated: Belongs to the old workflow"""
    cropped_raster = mask.mask(raster, bbox_feature, crop=True)
    if len(cropped_raster[0].shape) == 3 and np.sum(cropped_raster[0]) == 0:
        return False
    out_meta = raster.meta
    out_meta.update(
        {
            "height": cropped_raster[0].shape[-2],
            "width": cropped_raster[0].shape[-1],
            "transform": cropped_raster[1],
        }
    )
    with rasterio.open(
        os.path.join(path, f"sample_{counter}.tif"), "w", **out_meta
    ) as dest:
        dest.write(cropped_raster[0])
    return True


def create_ml_data(raster, r_mask, vector) -> None:
    """Deprecated: Belongs to the old workflow"""
    feature_count = len(vector)
    counter_failed_crops = 0
    for index, row in tqdm(vector.iterrows(), total=feature_count):
        bbox_feature = box(*row)
        bbox_feature = gpd.GeoSeries([bbox_feature])

        try:
            if index % 5 == 0 and index != 0:
                answer = crop_and_save(raster, bbox_feature, TEST_PATH_IMG, index)
                if answer is False:
                    continue
                crop_and_save(r_mask, bbox_feature, TEST_PATH_MASK, index)
            else:
                answer = crop_and_save(raster, bbox_feature, TRAINING_PATH_IMG, index)
                if answer is False:
                    continue
                crop_and_save(r_mask, bbox_feature, TRAINING_PATH_MASK, index)
        except:
            counter_failed_crops += 1
    logger.info(f"\n ML Data for all footprints generated!")
    if counter_failed_crops > 0:
        logger.info(f"Cropping failed for {counter_failed_crops} buildings")


def create_ml_tiles(raster, r_mask) -> None:
    """
    This function breaks up the raster and mask into tiles and writes 80% into a dir for
        training, 10% into a dir for testing and 10% into a dir for validation
    :param raster: the raster data to be used for training/testing/validation
    :param r_mask: the mask which indicates the buildings
    :return: None, as data is stored on hard drive
    """
    tiles_to_be_created = math.ceil(
        (raster.meta["width"] * raster.meta["height"])
        / (TARGET_SIZE[0] * TARGET_SIZE[1])
    )
    counter_successful = 0
    for window, transform in tqdm(
        get_tiles(raster, TARGET_SIZE[0], TARGET_SIZE[1]), total=tiles_to_be_created
    ):
        if counter_successful % 10 == 0 and counter_successful != 0:
            output_path = VALIDATION_PATH
        elif counter_successful % 5 == 0 and counter_successful != 0:
            output_path = TEST_PATH
        else:
            output_path = TRAINING_PATH
        result = write_raster_window(
            raster, r_mask, window, transform, output_path, counter_successful
        )
        if result:
            counter_successful += 1
    logger.info(
        f"\n ML Data generated! {counter_successful} of possible {tiles_to_be_created} tiles were successfully created (80% train, 10% test, 10% validation)."
    )


def preprocessing_data() -> None:
    """This function runs the other functions within this script in the correct order"""
    raster = rasterio.open(RASTER_PATH)

    logger.info("Query buildings...")
    buildings = get_building_data(raster)
    logger.info(f"Number of buildings queried: {len(buildings.index)}")

    logger.info("Generate Mask")
    generate_mask(raster, buildings["geometry"])
    logger.info("Mask written")
    r_mask = rasterio.open(os.path.join(INTERMEDIATE_PATH, "masked_raster.tif"))

    logger.info("Create data for ML")

    if not NEW_WORKFLOW:
        bounds = buildings.bounds
        create_ml_data(raster, r_mask, bounds)
    else:
        create_ml_tiles(raster, r_mask)


if __name__ == "__main__":
    preprocessing_data()
