import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch

from wisdomify.builders import InputsBuilder
from wisdomify.datamodules import WisdomifyDataModule
from wisdomify.models import RD


class Wisdomifier:
    def __init__(self, rd: RD,  builder: InputsBuilder, wisdoms: List[str]):
        self.rd = rd
        self.builder = builder
        self.wisdoms = wisdoms

    def __call__(self, sent: str) -> List[Tuple[str, float]]:
        wisdom2sent = [("", sent)]
        X = self.builder(wisdom2sent)
        probs = self.rd.P_wisdom(X).squeeze().tolist()
        wisdom2prob = [
                (wisdom, prob)
                for wisdom, prob in zip(self.wisdoms, probs)
        ]
        # sort and append
        res = list(sorted(wisdom2prob, key=lambda x: x[1], reverse=True))
        return res


class Experiment:

    def __init__(self, rd: RD, config: dict, datamodule: WisdomifyDataModule):
        self.rd = rd
        self.config = config
        self.datamodule = datamodule
        # fix seeds (always do this)
        torch.manual_seed(config['seed'])
        random.seed(config['seed'])
        np.random.seed(config['seed'])
