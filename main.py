import os
import geopandas as gpd
from shapely.geometry import box
import shapely
import rasterio
from rasterio import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import json
import numpy as np


from definitions import TRAINING_PATH, TEST_PATH, RASTER_PATH, INTERMEDIATE_PATH
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
    if not os.path.exists(INTERMEDIATE_PATH):
        os.mkdir(INTERMEDIATE_PATH)


def reproject_raster():
    dst_crs = 'EPSG:4326'
    with rasterio.open(RASTER_PATH) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(os.path.join(INTERMEDIATE_PATH, "reprojected_raster.tif"), 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)



    return None


def get_building_data(raster):
    bbox = box(*raster.bounds)
    bbox = gpd.GeoSeries([bbox]).__geo_interface__
    bbox = json.dumps(bbox)
    buildings = query(buil_def, bbox)
    return buildings


def generate_mask(raster, vector):
    out_image, out_transform = mask.mask(raster, vector, all_touched=False, invert=False, nodata=None, filled=True, crop=False,
                       pad=False, pad_width=0.5, indexes=None)
    out_image = out_image[0] + out_image[1] + out_image[2]
    out_image = np.where(out_image > 0, 1, out_image)
    out_meta = raster.meta
    out_meta.update({"driver": "GTiff",
                     "count": int(1),
                     "height": out_image.shape[0],
                     "width": out_image.shape[1],
                     "transform": out_transform})

    with rasterio.open(os.path.join(INTERMEDIATE_PATH, "masked_raster.tif"), "w", **out_meta) as dest:
        dest.write(out_image, 1)
    return out_meta


def create_ml_data(raster, r_mask, vector):
    row_count = vector.shape[0]
    train_max = round(row_count * 0.8)


    for index, row in vector.iterrows():
        cropped_raster = mask.mask(raster, row, crop=True)
        cropped_mask = mask.mask(r_mask, row, crop=True)



def main():
    setup_folders()
    reproject_raster()
    raster = rasterio.open(os.path.join(INTERMEDIATE_PATH, "reprojected_raster.tif"))
    buildings = get_building_data(raster)
    buildings = gpd.GeoDataFrame.from_features(buildings["features"])

    r_mask = generate_mask(raster, buildings["geometry"])
    bounds = buildings.bounds

main()

"""
To-dos:
remove buildings in no data area
"""