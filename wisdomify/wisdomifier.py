from typing import List, Tuple
from wisdomify.builders import XBuilder
from wisdomify.models import RD
from transformers import BertTokenizerFast
from wisdomify.experiment import Experiment


class Wisdomifier:
    def __init__(self, rd: RD, tokenizer: BertTokenizerFast, X_builder: XBuilder, wisdoms: List[str]):
        self.rd = rd
        self.tokenizer = tokenizer
        self.X_builder = X_builder
        self.wisdoms = wisdoms

    @staticmethod
    def from_pretrained(ver: str, device, wandb_support) -> 'Wisdomifier':
        exp = Experiment.load(ver, device, wandb_support)
        exp.rd.eval()

        return Wisdomifier(exp.rd, exp.tokenizer, exp.data_module.X_builder, exp.config['wisdoms'])

    def __call__(self, sents: List[str]) -> List[List[Tuple[str, float]]]:
        # get the X
        wisdom2sent = [("", desc) for desc in sents]
        X = self.X_builder(wisdom2sent)
        # get H_all for this.
        P_wisdom = self.rd.P_wisdom(X)
        results = list()
        for S_word_prob in P_wisdom.tolist():
            wisdom2prob = [
                (wisdom, prob)
                for wisdom, prob in zip(self.wisdoms, S_word_prob)
            ]
            # sort and append
            results.append(sorted(wisdom2prob, key=lambda x: x[1], reverse=True))

        return results
