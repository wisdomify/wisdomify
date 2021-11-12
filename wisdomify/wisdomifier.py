from typing import List
from wisdomify.datamodules import WisdomifyDataModule
from wisdomify.models import RD


class Wisdomifier:
    def __init__(self, rd: RD,  datamodule: WisdomifyDataModule):
        self.rd = rd
        self.datamodule = datamodule

    def __call__(self, sents: List[str]):
        wisdom2sent = [("", sent) for sent in sents]
        inputs_builder, _ = self.datamodule.tensor_builders()
        X = inputs_builder(wisdom2sent)
        # get H_all for this.
        P_wisdom = self.rd.P_wisdom(X)
        results = list()
        for S_word_prob in P_wisdom.tolist():
            wisdom2prob = [
                (wisdom, prob)
                for wisdom, prob in zip(self.datamodule.wisdoms, S_word_prob)
            ]
            # sort and append
            results.append(sorted(wisdom2prob, key=lambda x: x[1], reverse=True))
        return results
