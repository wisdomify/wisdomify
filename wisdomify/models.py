from typing import List, Tuple
from torch.nn import functional as F
from wisdomify.builders import XBuilder
from wisdomify.rds import RD
from transformers import BertTokenizer
from wisdomify.utils import Experiment


class Wisdomifier:
    def __init__(self, rd: RD, tokenizer: BertTokenizer, X_builder: XBuilder, wisdoms: List[str]):
        self.rd = rd
        self.tokenizer = tokenizer
        self.X_builder = X_builder
        self.wisdoms = wisdoms

    @staticmethod
    def from_pretrained(ver: str, device) -> 'Wisdomifier':
        exp = Experiment.from_pretrained(ver, device)
        exp.rd.eval()
        wisdomifier = Wisdomifier(exp.rd, exp.tokenizer, exp.data_module.X_builder,
                                  exp.config['wisdoms'])
        return wisdomifier

    def __call__(self, sents: List[str]) -> List[List[Tuple[str, float]]]:
        # get the X
        wisdom2sent = [("", desc) for desc in sents]
        X = self.X_builder(wisdom2sent)
        # get H_all for this.
        H_all = self.rd.forward(X)
        S_wisdom = self.rd.S_wisdom(H_all)
        P_wisdom = F.softmax(S_wisdom, dim=1)
        results = list()
        for S_word_prob in P_wisdom.tolist():
            wisdom2prob = [
                (wisdom, prob)
                for wisdom, prob in zip(self.wisdoms, S_word_prob)
            ]
            # sort and append
            results.append(sorted(wisdom2prob, key=lambda x: x[1], reverse=True))
        return results
