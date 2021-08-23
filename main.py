import json
import os
import sys

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import mask
from rasterio.warp import Resampling, calculate_default_transform, reproject
from shapely.geometry import box

from definitions import (
    INTERMEDIATE_PATH,
    RASTER_PATH,
    TEST_PATH_IMG,
    TEST_PATH_MASK,
    TRAINING_PATH_IMG,
    TRAINING_PATH_MASK,
    logger,
)
from model import get_generator
from utils import query


buil_def = {
    "description": "All Buildings in an Area",
    "endpoint": "elements/geometry",
    "filter": """
         building = * and geometry:polygon
    """,
}

# SCRIPT SETTINGS:

FROM_FILE = False

# ML VARIABLES:

BATCH_SIZE = 20
TARGET_SIZE = [224, 224]


def reproject_raster():
    dst_crs = "EPSG:4326"
    with rasterio.open(RASTER_PATH) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update(
            {"crs": dst_crs, "transform": transform, "width": width, "height": height, "compress": "jpeg", "tiled": True, "blockxsize": 256, "blockysize": 256}
        )

        with rasterio.open(
            os.path.join(INTERMEDIATE_PATH, "reprojected_raster.tif"), "w", **kwargs
        ) as dst:
            for i in range(1, src.count + 1):
                logger.info(f"Reprojecting band {i} of {src.count}...")
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest,
                    num_threads=7
                )
        logger.info("Raster reprojected and saved.")


def get_building_data(raster):
    bbox = box(*raster.bounds)
    bbox = gpd.GeoSeries([bbox]).__geo_interface__
    bbox = json.dumps(bbox)
    buildings = query(buil_def, bbox)
    return buildings


def generate_mask(raster, vector):
    out_image, out_transform = mask.mask(
        raster,
        vector,
        all_touched=False,
        invert=False,
        nodata=None,
        filled=True,
        crop=False,
        pad=False,
        pad_width=0.5,
        indexes=None,
    )
    out_image = np.sum(out_image, axis=0, dtype="uint8")
    out_image = np.where(out_image > 0, 1, out_image)
    out_meta = raster.meta
    out_meta.update(
        {
            "driver": "GTiff",
            "count": int(1),
            "height": out_image.shape[0],
            "width": out_image.shape[1],
            "transform": out_transform,
        }
    )

    with rasterio.open(
        os.path.join(INTERMEDIATE_PATH, "masked_raster.tif"), "w", **out_meta
    ) as dest:
        dest.write(out_image, 1)


def crop_and_save(raster, bbox_feature, path, counter):
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


def create_ml_data(raster, r_mask, vector):
    feature_count = len(vector)
    for index, row in vector.iterrows():
        if index % 100 == 0:
            percentage = int((index + 1) / feature_count * 100)
            sys.stdout.write(f"\r Progress: {percentage} %")
            sys.stdout.flush()

        bbox_feature = box(*row)
        bbox_feature = gpd.GeoSeries([bbox_feature])

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


def main(from_file: bool, batch_size: int, target_size: list):
    if not from_file:
        reproject_raster()
    raster = rasterio.open(os.path.join(INTERMEDIATE_PATH, "reprojected_raster.tif"))

    if not from_file:
        buildings = get_building_data(raster)
        buildings = gpd.GeoDataFrame.from_features(buildings["features"])
        buildings.to_file(
            os.path.join(INTERMEDIATE_PATH, "buildings.geojson"), driver="GeoJSON"
        )
    else:
        buildings = gpd.read_file(os.path.join(INTERMEDIATE_PATH, "buildings.geojson"))
    logger.info(f"Number of buildings queried: {len(buildings.index)}")

    if not from_file:
        generate_mask(raster, buildings["geometry"])
    generate_mask(raster, buildings["geometry"])  # TODO: remove this line
    r_mask = rasterio.open(os.path.join(INTERMEDIATE_PATH, "masked_raster.tif"))
    bounds = buildings.bounds
    create_ml_data(raster, r_mask, bounds)

    #train_gen, test_gen = get_generator(batch_size=batch_size, target_size=target_size)


if __name__ == "__main__":
    main(from_file=FROM_FILE, batch_size=BATCH_SIZE, target_size=TARGET_SIZE)
