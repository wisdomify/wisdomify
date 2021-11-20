"""
A loader loads something from local.
"""
from wisdomify.constants import CONFIG_YAML
import yaml
import torch


def load_config() -> dict:
    with open(CONFIG_YAML, 'r', encoding="utf-8") as fh:
        return yaml.safe_load(fh)

