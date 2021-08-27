import json
import os
import sys
from itertools import product

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import mask, windows
from shapely.geometry import box
from skimage.filters import median
from skimage.morphology import disk

from definitions import (
    INTERMEDIATE_PATH,
    RASTER_PATH,
    TEST_PATH_IMG,
    TEST_PATH_MASK,
    TRAINING_PATH_IMG,
    TRAINING_PATH_MASK,
    logger,
)
from utils import query

buil_def = {
    "description": "All Buildings in an Area",
    "endpoint": "elements/geometry",
    "filter": """
         building = * and geometry:polygon
    """,
}


def get_building_data(raster):
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


def get_tiles(ds, width=256, height=256):
    ncols, nrows = ds.meta["width"], ds.meta["height"]
    offsets = product(range(0, ncols, width), range(0, nrows, height))
    big_window = windows.Window(col_off=0, row_off=0, width=ncols, height=nrows)
    for col_off, row_off in offsets:
        window = windows.Window(
            col_off=col_off, row_off=row_off, width=width, height=height
        ).intersection(big_window)
        transform = windows.transform(window, ds.transform)
        yield window, transform


def generate_mask(raster, vector):
    if os.path.exists(os.path.join(INTERMEDIATE_PATH, "masked_raster.tif")):
        os.remove(os.path.join(INTERMEDIATE_PATH, "masked_raster.tif"))
    out_meta = raster.meta.copy()
    out_meta.update(
        {
            "driver": "GTiff",
            "count": int(1),
        }
    )

    for window, transform in get_tiles(raster, TILE_WIDTH, TILE_HEIGHT):
        meta_tile = raster.meta.copy()
        meta_tile["transform"] = transform
        meta_tile["width"], meta_tile["height"] = window.width, window.height

        tiledata = raster.read(window=window)
        with rasterio.open(
            os.path.join(INTERMEDIATE_PATH, f"tile.tif"), "w", **meta_tile
        ) as dest:
            dest.write(tiledata)
        tile = rasterio.open(os.path.join(INTERMEDIATE_PATH, f"tile.tif"))

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
        out_image = np.sum(out_image, axis=0, dtype="uint8")
        out_image = np.where(out_image > 0, 1, out_image)
        out_image = median(out_image, disk(1), mode="constant", cval=0)
        out_image = np.array([out_image])

        if os.path.exists(os.path.join(INTERMEDIATE_PATH, "masked_raster.tif")):
            with rasterio.open(
                os.path.join(INTERMEDIATE_PATH, "masked_raster.tif"), "r+", **out_meta
            ) as outds:
                outds.write(out_image, window=window)
        else:
            with rasterio.open(
                os.path.join(INTERMEDIATE_PATH, "masked_raster.tif"), "w", **out_meta
            ) as outds:
                outds.write(out_image, window=window)


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
    counter_failed_crops = 0
    for index, row in vector.iterrows():
        if index % 100 == 0:
            percentage = int((index + 1) / feature_count * 100)
            sys.stdout.write(f"\r Progress: {percentage} %")
            sys.stdout.flush()

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


def preprocessing():
    raster = rasterio.open(RASTER_PATH)

    logger.info("Query buildings...")
    buildings = get_building_data(raster)
    logger.info(f"Number of buildings queried: {len(buildings.index)}")

    logger.info("Generate Mask")
    generate_mask(raster, buildings["geometry"])
    logger.info("Mask written")
    r_mask = rasterio.open(os.path.join(INTERMEDIATE_PATH, "masked_raster.tif"))
    bounds = buildings.bounds
    logger.info("Create data for ML")
    create_ml_data(raster, r_mask, bounds)


if __name__ == "__main__":
    preprocessing()
