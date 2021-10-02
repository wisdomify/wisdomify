import json
from wisdomify.paths import CONF_JSON
import torch


# --- loaders --- #
def load_conf_json() -> dict:
    with open(CONF_JSON, 'r') as fh:
        return json.loads(fh.read())


def load_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device
