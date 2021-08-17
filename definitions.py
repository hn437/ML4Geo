"""
Global Variables and Functions.
"""

import logging.config
import os
from pathlib import Path
import yaml

OHSOME_API = os.getenv("OHSOME_API", default="https://api.ohsome.org/v1/")
DATA_PATH = "./data"
Path(DATA_PATH).mkdir(parents=True, exist_ok=True)

TRAINING_PATH = os.path.join(DATA_PATH, "training_data")
TEST_PATH = os.path.join(DATA_PATH, "test_data")
RASTER_PATH = os.path.join(DATA_PATH, "ortho_1-1_hn_s_nm061_2020_1.sid")


def get_logger():
    logs_path = os.path.join(DATA_PATH, "logs")
    logging_file_path = os.path.join(logs_path, "ml4geo.log")
    logging_config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "logging.yaml"
    )
    Path(logs_path).mkdir(parents=True, exist_ok=True)

    with open(logging_config_path, "r") as f:
        logging_config = yaml.safe_load(f)
    logging_config["handlers"]["file"]["filename"] = logging_file_path
    logging.config.dictConfig(logging_config)

    return logging.getLogger("ml4geo")

logger = get_logger()