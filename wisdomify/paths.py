"""
paths to data, paths to saved models should be saved here
"""
from pathlib import Path
from os import path

# The directories
from wisdomify.config import RUN_VER

PROJECT_ROOT = Path(__file__).resolve().parent.parent.__str__()
DATA_DIR = path.join(PROJECT_ROOT, "data")

# path를 지정하는 것은? 일단, tsv 한 이유? -> 단순하게.
# tsv파일이 너무 불편하다! 그때는 DB로.
WISDOMDATA_DIR = path.join(DATA_DIR, "wisdomdata")
WISDOMDATA_VER_DIR = path.join(WISDOMDATA_DIR, f"version_{RUN_VER}")

LIGHTNING_LOGS_DIR = path.join(DATA_DIR, "lightning_logs")

# project - files
CONF_JSON = path.join(PROJECT_ROOT, "conf.json")

# the directories in which each version is stored
WISDOMIFIER = path.join(LIGHTNING_LOGS_DIR, f"version_{RUN_VER}")
WISDOMIFIER_CKPT = path.join(WISDOMIFIER, "checkpoints", "wisdomify.ckpt")
WISDOMIFIER_HPARAMS_YAML = path.join(WISDOMIFIER, "hparams.yaml")
