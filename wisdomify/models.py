"""
The reverse dictionary models below are based off of: https://github.com/yhcc/BertForRD/blob/master/mono/model/bert.py
"""
from wisdomify.metrics import RDMetric
from argparse import Namespace
from typing import Tuple, List, Optional
import pytorch_lightning as pl
from transformers.models.bert.modeling_bert import BertForMaskedLM
from torch.nn import functional as F
import torch


class RD(pl.LightningModule):
    """
    The superclass of all the reverse-dictionaries. This class houses any methods that are required by
    whatever reverse-dictionaries we define.
    """

    def __init__(self, bert_mlm: BertForMaskedLM,
                 wisdom2subwords: torch.Tensor, k: int, lr: float, device: torch.device):
        """
        :param bert_mlm: a bert model for masked language modeling
        :param wisdom2subwords: (|W|, K)
        :return: (N, K, |V|); (num samples, k, the size of the vocabulary of subwords)
        """
        super().__init__()
        # -- the only neural network we need -- #
        self.bert_mlm = bert_mlm
        # -- to be used to compute S_wisdom -- #
        self.wisdom2subwords = wisdom2subwords  # (|W|, K)
        # --- to be used for getting H_k --- #
        self.wisdom_mask: Optional[torch.Tensor] = None  # (N, L)
        # -- to be used to evaluate the model -- #
        self.rd_metric = RDMetric()
        # -- hyper params --- #
        # should be saved to self.hparams
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/4390#issue-730493746
        self.save_hyperparameters(Namespace(k=k, lr=lr))
        self.to(device)  # always make sure to do this.

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        :param X: (N, 4, L);
         (num samples, 0=input_ids/1=token_type_ids/2=attention_mask/3=wisdom_mask, the maximum length)
        :return: (N, L, H); (num samples, k, the size of the vocabulary of subwords)
        """
        input_ids = X[:, 0]  # (N, 4, L) -> (N, L)
        token_type_ids = X[:, 1]  # (N, 4, L) -> (N, L)
        attention_mask = X[:, 2]  # (N, 4, L) -> (N, L)
        self.wisdom_mask = X[:, 3]  # (N, 4, L) -> (N, L)
        H_all = self.bert_mlm.bert.forward(input_ids, attention_mask, token_type_ids)[0]  # (N, 3, L) -> (N, L, H)
        return H_all

    def H_k(self, H_all: torch.Tensor) -> torch.Tensor:
        """
        You may want to override this. (e.g. RDGamma - the k's could be anywhere)
        :param H_all (N, L, H)
        :return H_k (N, K, H)
        """
        N, _, H = H_all.size()
        # refer to: wisdomify/examples/explore_masked_select.py
        wisdom_mask = self.wisdom_mask.unsqueeze(2).expand(H_all.shape)  # (N, L) -> (N, L, 1) -> (N, L, H)
        H_k = torch.masked_select(H_all, wisdom_mask.bool())  # (N, L, H), (N, L, H) -> (N * K * H)
        H_k = H_k.reshape(N, self.hparams['k'], H)  # (N * K * H) -> (N, K, H)
        self.wisdom_mask = None  # clear wisdom_mask after it is used.
        return H_k

    def S_wisdom_literal(self, H_k: torch.Tensor) -> torch.Tensor:
        """
        To be used for both RDAlpha & RDBeta
        :param H_k: (N, K, H)
        :return: S_wisdom_literal (N, |W|)
        """
        S_vocab = self.bert_mlm.cls(H_k)  # bmm; (N, K, H) * (H, |V|) ->  (N, K, |V|)
        indices = self.wisdom2subwords.T.repeat(S_vocab.shape[0], 1, 1)  # (|W|, K) -> (N, K, |W|)
        S_wisdom_literal = S_vocab.gather(dim=-1, index=indices)  # (N, K, |V|) -> (N, K, |W|)
        S_wisdom_literal = S_wisdom_literal.sum(dim=1)  # (N, K, |W|) -> (N, |W|)
        return S_wisdom_literal

    def S_wisdom(self, H_all: torch.Tensor) -> torch.Tensor:
        """
        :param H_all: (N, L, H)
        :return S_wisdom: (N, |W|)
        """
        raise NotImplementedError("An RD class must implement S_wisdom")

    def P_wisdom(self, X: torch.Tensor) -> torch.Tensor:
        """
        :param X: (N, 3, L)
        :return P_wisdom: (N, |W|), normalized over dim 1.
        """
        H_all = self.forward(X)
        S_wisdom = self.S_wisdom(H_all)
        P_wisdom = F.softmax(S_wisdom, dim=1)
        return P_wisdom

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> dict:
        X, y = batch
        H_all = self.forward(X)  # (N, 3, L) -> (N, L, H)
        S_wisdom = self.S_wisdom(H_all)  # (N, L, H) -> (N, |W|)
        loss = F.cross_entropy(S_wisdom, y)  # (N, |W|), (N,) -> (N,)
        loss = loss.sum()  # (N,) -> (1,)
        P_wisdom = F.softmax(S_wisdom, dim=1)  # (N, |W|) -> (N, |W|)
        self.rd_metric.update(preds=P_wisdom, targets=y)
        median, var, top1, top10, top100 = self.rd_metric.compute()
        return {
            'loss': loss,
            'median': median,
            'var': var,
            'top1': top1,
            'top10': top10,
            'top100': top100
        }

    def training_epoch_end(self, outputs: List[dict]) -> None:
        # reset the metric every epoch
        self.rd_metric.reset()
        avg_train_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_median = sum([x['median'] for x in outputs]) / len(outputs)
        avg_var = sum([x['var'] for x in outputs]) / len(outputs)
        avg_top1 = sum([x['top1'] for x in outputs]) / len(outputs)
        avg_top10 = sum([x['top10'] for x in outputs]) / len(outputs)
        avg_top100 = sum([x['top100'] for x in outputs]) / len(outputs)
        # logging using tensorboard logger

        self.log("Train/Average Loss", avg_train_loss)
        self.log("Train/Average Median", avg_median)
        self.log("Train/Average Variance", avg_var)
        self.log("Train/Average Top 1 Acc", avg_top1)
        self.log("Train/Average Top 10 Acc", avg_top10)
        self.log("Train/Average Top 100 Acc", avg_top100)

    def validation_step(self, *args, **kwargs):
        # TODO: 나중에 구현하기. (이렇게 하면 워닝은 안뜨겠지)
        pass

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor],
                  batch_idx: int, *args, **kwargs):
        self.rd_metric.reset()
        X, y = batch
        P_wisdom = self.P_wisdom(X)
        self.rd_metric.update(preds=P_wisdom, targets=y)

        median, var, top1, top10, top100 = self.rd_metric.compute()

        self.log("Test Median", median)
        self.log("Test Variance", var)
        self.log("Test Top 1 Acc", top1)
        self.log("Test Top 10 Acc", top10)
        self.log("Test Top 100 Acc", top100)

        print("batch:{}".format(batch_idx), (median, var, top1, top10, top100))

    def on_test_end(self):
        median, var, top1, top10, top100 = self.rd_metric.compute()
        print("### final ###")
        print("median:", median)
        print("var:", var)
        print("top1:", top1)
        print("top10:", top10)
        print("top100:", top100)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Instantiates and returns the optimizer to be used for this model
        e.g. torch.optim.Adam
        """
        # The authors used Adam, so we might as well use it as well.
        return torch.optim.AdamW(self.parameters(), lr=self.hparams['lr'])


class RDAlpha(RD):
    """
    The first prototype.
    S_wisdom = S_wisdom_literal
    trained on: wisdom2def
    """

    def S_wisdom(self, H_all: torch.Tensor) -> torch.Tensor:
        H_k = self.H_k(H_all)  # (N, L, H) -> (N, K, H)
        S_wisdom = self.S_wisdom_literal(H_k)  # (N, K, H) -> (N, |W|)
        return S_wisdom


class RDBeta(RD):
    """
    The second prototype.
    S_wisdom = S_wisdom_literal + S_wisdom_figurative
    trained on: wisdom2def
    """

    def __init__(self, bert_mlm: BertForMaskedLM,
                 wisdom2subwords: torch.Tensor, wiskeys: torch.Tensor,
                 k: int, lr: float, device: torch.device):
        super().__init__(bert_mlm, wisdom2subwords, k, lr, device)
        self.wiskeys = wiskeys  # (|W|,)

    def S_wisdom(self, H_all: torch.Tensor) -> torch.Tensor:
        H_k = self.H_k(H_all)  # (N, L, H) -> (N, K, H
        S_wisdom = self.S_wisdom_literal(H_k) + self.S_wisdom_figurative(H_all)  # (N, |W|) + (N, |W|) -> (N, |W|)
        return S_wisdom

    def S_wisdom_figurative(self, H_all: torch.Tensor) -> torch.Tensor:
        """
        param: H_all (N, L, H)
        return: S_wisdom_figurative (N, |W|)
        """
        # 속담의 임베딩은 여기에 있습니다!
        # 참고: wisdomify/examples/explore_bert_embeddings.py
        W_embed = self.bert_mlm.bert.embeddings.word_embeddings(self.wiskeys)  # (|W|,) -> (|W|, H)

        # TODO 다음을 계산해주세요!
        S_wisdom_figurative: torch.Tensor = ...
        return S_wisdom_figurative


class RDGamma(RD):
    """
    The third prototype.
    S_wisdom = S_wisdom_literal + S_wisdom_figurative
    trained on = wisdom2def & wisdom2eg with two-stage training.
    This is to be implemented & experimented after experimenting with RDAlpha & RDBeta is done.
    """

    def S_wisdom(self, H_all: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        param X: (N, 4, L) - wisdom_mask 추가.
        return: what should this return?
        """
        raise NotImplementedError
        # input_ids = X[:, 0]
        # token_type_ids = X[:, 1]
        # attention_mask = X[:, 2]
        # wisdom_mask = X[:, 3]  # this is new.  # (N, L).
        # H_all = self.bert_mlm.bert.forward(input_ids, attention_mask, token_type_ids)[0]  # (N, 4, L) -> (N, L, 768)
        # H_cls = H_all[:, 0]  # (N, L, 768) -> (N, 768)
        # N = wisdom_mask.shape[0]  # get the batch size.
        # H = H_all.shape[2]  # get the hidden size.
        # wisdom_mask = wisdom_mask.unsqueeze(-1).expand(H_all.shape)
        # H_k = torch.masked_select(H_all, wisdom_mask.bool())
        # H_k = H_k.reshape(N, self.hparams['k'], H)  # (1, K * N) -> (N, K)
        # return H_cls, H_k

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor],
                      batch_idx: int) -> dict:
        """
        param: X (N, 4, L)
        param:
        """
        raise NotImplementedError
        # X, y = batch  # (N, 3, L), (N,)
        # H_cls, H_k = self.forward(X)  # (N, 3, L) -> (N, H), (N, K, H)
        # S_vocab = self.bert_mlm.cls(H_k)  # (N, K, 768) ->  (N, K, |V|)
        # S_wisdom_literal = self.S_wisdom_literal(S_vocab)  # (N, K, |V|) -> (N, |W|)
        # S_wisdom_figurative = self.S_wisdom_figurative(H_cls, H_k)  # (N, H),(N, K, H) -> (N, |W|)
        # S_wisdom = S_wisdom_literal + S_wisdom_figurative  # (N, |W|) + (N, |W|) -> (N, |W|).
        # loss = F.cross_entropy(S_wisdom, y)  # (N, |W|), (N,) -> (N,)
        # loss = loss.sum()  # (N,) -> (1,) (scalar)
        # return {
        #     'loss': loss
        # }
