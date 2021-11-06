"""
paths to data, paths to saved models should be saved here
"""
from pathlib import Path
from os import path
# --- The directories --- #
ROOT_DIR = Path(__file__).resolve().parent.parent.__str__()
WANDB_DIR = path.join(ROOT_DIR, "wandb")  # for saving wandb logs
ARTIFACTS_DIR = path.join(WANDB_DIR, "artifacts")  # for saving wandb artifacts
# --- for configuring experiments --- #
CONF_JSON = path.join(ROOT_DIR, "conf.json")
