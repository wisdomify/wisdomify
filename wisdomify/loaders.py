"""
A loader loads something from local.
"""
import json
from wisdomify.constants import CONF_JSON
import torch


def load_conf() -> dict:
    with open(CONF_JSON, 'r', encoding="utf-8") as fh:
        return json.loads(fh.read())


def load_device(use_gpu: bool) -> torch.device:
    if use_gpu:
        if not torch.cuda.is_available():
            raise ValueError("cuda is unavailable")
        else:
            return torch.device("cuda")
    return torch.device("cpu")
