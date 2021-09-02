import datetime
import os
from typing import Dict
import json

import requests
from itertools import product
from rasterio import windows

from definitions import OHSOME_API, logger, RESULT_PATH


def query(request: Dict, bpolys: str, properties: str = None) -> Dict:
    """Query ohsome API endpoint with filter."""
    url = OHSOME_API + request["endpoint"]
    if properties is not None:
        data = {"bpolys": bpolys, "filter": request["filter"], "properties": properties}
    else:
        data = {"bpolys": bpolys, "filter": request["filter"]}
    logger.info("Query ohsome API.")
    logger.info("Query URL: " + url)
    logger.info("Query Filter: " + request["filter"])

    response = requests.post(url, data=data)

    if response.status_code == 200:
        logger.info("Query successful!")
    elif response.status_code == 404:
        logger.info("Query failed!")
    else:
        logger.info(response.status_code)
    return response.json()


def update_json(key, val) -> None:
    if os.path.exists(os.path.join(RESULT_PATH, "metrics.json")):
        with open(os.path.join(RESULT_PATH, "metrics.json"), "r") as file:
            metrics = json.load(file)
    else:
        metrics = {}
    metrics[key] = val
    with open(os.path.join(RESULT_PATH, "metrics.json"), "w") as file:
        json.dump(metrics, file)


def get_tiles(ds, width=256, height=256) -> tuple:
    ncols, nrows = ds.meta["width"], ds.meta["height"]
    offsets = product(range(0, ncols, width), range(0, nrows, height))
    big_window = windows.Window(col_off=0, row_off=0, width=ncols, height=nrows)
    for col_off, row_off in offsets:
        window = windows.Window(
            col_off=col_off, row_off=row_off, width=width, height=height
        ).intersection(big_window)
        transform = windows.transform(window, ds.transform)
        yield window, transform
