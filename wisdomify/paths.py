"""
paths to data, paths to saved models should be saved here
"""
from pathlib import Path
from os import path

# The directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent.__str__()
DATA_DIR = path.join(PROJECT_ROOT, "data")


# project - files
CONF_JSON = path.join(PROJECT_ROOT, "conf.json")

# data - files
WISDOM2DEF_TSV = path.join(DATA_DIR, "wisdom2def.tsv")
WISDOM2EG_TSV = path.join(DATA_DIR, "wisdom2eg.tsv")

# data - trained models
WISDOMIFIER_A_CKPT = path.join(DATA_DIR, "lightning_logs/version_0/checkpoints/wisdomify_def_epoch=19_train_loss=0.00.ckpt")
