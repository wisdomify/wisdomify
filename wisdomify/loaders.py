from typing import List, Tuple
import csv
import json
from wisdomify.paths import WISDOMDATA_VER_0_DIR, CONF_JSON
from dataclasses import dataclass
from os import path


def load_conf() -> dict:
    with open(CONF_JSON, 'r') as fh:
        return json.loads(fh.read())
