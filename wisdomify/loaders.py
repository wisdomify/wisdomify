"""
A loader loads something from local.
"""
import json
from wisdomify.paths import CONF_JSON
import torch


def load_conf() -> dict:
    with open(CONF_JSON, 'r', encoding="utf-8") as fh:
        return json.loads(fh.read())


def load_device(use_gpu=False) -> tuple(torch.device, int):
    if not use_gpu: 
        return torch.device("cpu"), 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device, torch.cuda.device_count()


