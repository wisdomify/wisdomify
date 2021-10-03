"""
paths to data, paths to saved models should be saved here
"""
from pathlib import Path
from os import path

# The directories

PROJECT_ROOT = Path(__file__).resolve().parent.parent.__str__()
DATA_DIR = path.join(PROJECT_ROOT, "data")

# path를 지정하는 것은? 일단, tsv 한 이유? -> 단순하게.
# tsv파일이 너무 불편하다! 그때는 DB로.
WISDOMDATA_DIR = path.join(DATA_DIR, "wisdomdata")
# use WISDOMDATA_VER_DIR.format(ver=SOME_NUMBER) to set version num.
WISDOMDATA_VER_DIR = path.join(WISDOMDATA_DIR, "version_{ver}")


LIGHTNING_LOGS_DIR = path.join(DATA_DIR, "lightning_logs")

# project - files
CONF_JSON = path.join(PROJECT_ROOT, "conf.json")

# the directories in which each version is stored
# use WISDOMIFIER_CKPT.format(ver=SOME_NUMBER) to set version num.
# use WISDOMIFIER_HPARAMS_YAML.format(ver=SOME_NUMBER) to set version num.
VERSION_DIR = path.join(LIGHTNING_LOGS_DIR, "version_{ver}")
WISDOMIFIER_CKPT = path.join(VERSION_DIR, "checkpoints", "wisdomifier.ckpt")
WISDOMIFIER_HPARAMS_YAML = path.join(VERSION_DIR, "hparams.yaml")
WISDOMIFIER_TOKENIZER_DIR = path.join(VERSION_DIR, "tokenizer")
