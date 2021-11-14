"""
A loader loads something from local.
"""
from wisdomify.constants import CONFIG_YAML
import yaml
import torch


def load_config() -> dict:
    with open(CONFIG_YAML, 'r', encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_device(use_gpu: bool = False) -> torch.device:
    if use_gpu:
        if not torch.cuda.is_available():
            raise ValueError("cuda is unavailable")
        else:
            return torch.device("cuda")
    return torch.device("cpu")
