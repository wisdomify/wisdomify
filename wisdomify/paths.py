"""
paths to data, paths to saved models should be saved here
"""
from pathlib import Path
from os import path
# The directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent.__str__()
DATA_DIR = path.join(PROJECT_ROOT, "data")  # where lightning_logs will be stored
WANDB_DIR = path.join(PROJECT_ROOT, DATA_DIR)  # for putting all the wandb logs


# files
CONF_JSON = path.join(PROJECT_ROOT, "conf.json")