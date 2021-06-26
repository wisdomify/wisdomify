"""
The reverse dictionary models below are based off of: https://github.com/yhcc/BertForRD/blob/master/mono/model/bert.py
"""
from argparse import Namespace
from typing import Tuple, List
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.optim import Optimizer
from transformers import BertTokenizer
from transformers.models.bert.modeling_bert import BertForMaskedLM
from torch.nn import functional as F
from wisdomify.builders import build_X
from wisdomify.vocab import VOCAB


class RD(pl.LightningModule):
    """
    A reverse-dictionary.
    """
    # fruitify into these!
    def __init__(self, bert_mlm: BertForMaskedLM, vocab2subwords: Tensor, k: int, lr: float):
        super().__init__()
        # -- the only network we need -- #
        self.bert_mlm = bert_mlm
        # -- to be used to compute S_word -- #
        self.vocab2subwords = vocab2subwords
        # -- hyper params --- #
        # should be saved to self.hparams
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/4390#issue-730493746
        self.save_hyperparameters(Namespace(k=k, lr=lr))

    def forward(self, X: Tensor) -> Tensor:
        """
        :param X: (N, 3, L) (num samples, 0=input_ids/1=token_type_ids/2=attention_mask, the maximum length)
        :return: (N, K, |V|); (num samples, k, the size of the vocabulary of subwords)
        """
        input_ids = X[:, 0]
        token_type_ids = X[:, 1]
        attention_mask = X[:, 2]
        H_all = self.bert_mlm.bert.forward(input_ids, attention_mask, token_type_ids)[0]  # (N, 3, L) -> (N, L, 768)
        H_k = H_all[:, 1: self.hparams['k'] + 1]  # (N, L, 768) -> (N, K, 768)
        S_subword = self.bert_mlm.cls(H_k)  # (N, K, 768) ->  (N, K, |S|)
        return S_subword

    def S_word(self, S_subword: Tensor) -> Tensor:
        # pineapple -> pine, ###apple, mask, mask, mask, mask, mask
        # [ ...,
        #   ...,
        #   ...
        #   [98, 122, 103, 103]]
        # [
        word2subs = self.vocab2subwords.T.repeat(S_subword.shape[0], 1, 1)  # (|V|, K) -> (N, K, |V|)
        S_word = S_subword.gather(dim=-1, index=word2subs)  # (N, K, |S|) -> (N, K, |V|)
        S_word = S_word.sum(dim=1)  # (N, K, |V|) -> (N, |V|)
        return S_word

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """
        :param batch: A tuple of X, y, subword_ids; ((N, 3, L), (N,),
        :param batch_idx: the index of the batch
        :return: (1,); the loss for this batch
        """
        X, y = batch
        # load the batches on the device.
        X = X.to(self.device)
        y = y.to(self.device)
        S_subword = self.forward(X)  # (N, 3, L) -> (N, K, |S|)
        S_word = self.S_word(S_subword)  # (N, K, |S|) -> (N, |V|)
        loss = F.cross_entropy(S_word, y)  # (N, |V|) -> (N,)
        loss = loss.sum()  # (N,) -> scalar
        self.log('train_loss', loss.item())  # monitor this, while training.
        return loss

    def configure_optimizers(self) -> Optimizer:
        """
        Instantiates and returns the optimizer to be used for this model
        e.g. torch.optim.Adam
        """
        # The authors used Adam, so we might as well use it
        return torch.optim.AdamW(self.parameters(), lr=self.hparams['lr'])


class Wisdomifier:

    def __init__(self, rd: RD, tokenizer: BertTokenizer):
        self.rd = rd  # a trained reverse dictionary
        self.tokenizer = tokenizer

    def wisdomify(self, sents: List[str]) -> List[List[Tuple[str, float]]]:
        # get the X
        wisdom2sent = [("", desc) for desc in sents]
        X = build_X(wisdom2sent, tokenizer=self.tokenizer, k=self.rd.hparams['k']).to(self.rd.device)
        # get S_subword for this.
        S_subword = self.rd.forward(X)
        S_word_logits = self.rd.S_word(S_subword)  # (N, |V|).
        S_word_probs = F.softmax(S_word_logits, dim=1)  # (N, |V|) -> (N, |V|) softmax along |V|
        results = list()
        for S_word_prob in S_word_probs.tolist():
            wisdom2prob = [
                (wisdom, prob)
                for wisdom, prob in zip(VOCAB, S_word_prob)
            ]
            # sort and append
            results.append(sorted(wisdom2prob, key=lambda x: x[1], reverse=True))
        return results
