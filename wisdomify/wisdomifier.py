from typing import List, Tuple
from wisdomify.experiment import Experiment


class Wisdomifier:
    def __init__(self, exp: Experiment):
        self.exp = exp

    def __call__(self, sents: List[str]) -> List[List[Tuple[str, float]]]:
        # get the X
        wisdom2sent = [("", desc) for desc in sents]
        x_builder, _ = self.exp.datamodule.tensor_builders()
        X = x_builder(wisdom2sent)
        # get H_all for this.
        P_wisdom = self.exp.rd.P_wisdom(X)
        results = list()
        for S_word_prob in P_wisdom.tolist():
            wisdom2prob = [
                (wisdom, prob)
                for wisdom, prob in zip(self.exp.datamodule.wisdoms, S_word_prob)
            ]
            # sort and append
            results.append(sorted(wisdom2prob, key=lambda x: x[1], reverse=True))

        return results
