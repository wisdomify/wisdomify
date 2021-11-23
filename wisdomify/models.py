"""
The reverse dictionary models below are based off of: https://github.com/yhcc/BertForRD/blob/master/mono/model/bert.py
"""
from argparse import Namespace
from typing import Tuple, List, Optional
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from transformers.models.bert.modeling_bert import BertForMaskedLM
from torch.nn import functional as F
from wisdomify.metrics import RDMetric


class RD(pl.LightningModule):
    """
    @eubinecto
    The superclass of all the reverse-dictionaries. This class houses any methods that are required by
    whatever reverse-dictionaries we define.
    """

    # --- boilerplate; the loaders are defined in datamodules, so we don't define them here
    # passing them to avoid warnings ---  #
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        pass

    def test_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def val_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def __init__(self, k: int, lr: float, bert_mlm: BertForMaskedLM,
                 wisdom2subwords: torch.Tensor):
        """
        :param bert_mlm: a bert model for masked language modeling
        :param wisdom2subwords: (|W|, K)
        :return: (N, K, |V|); (num samples, k, the size of the vocabulary of subwords)
        """
        super().__init__()
        # -- hyper params --- #
        # should be saved to self.hparams
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/4390#issue-730493746
        self.save_hyperparameters(Namespace(k=k, lr=lr))
        # -- the only neural network we need -- #
        self.bert_mlm = bert_mlm
        # -- to be used to compute S_wisdom -- #
        self.register_buffer("wisdom2subwords", wisdom2subwords)  # (|W|, K)
        # --- to be used for getting H_k --- #
        self.wisdom_mask: Optional[torch.Tensor] = None  # (N, L)
        # --- to be used for getting H_desc --- #
        self.desc_mask: Optional[torch.Tensor] = None  # (N, L)
        # --- to be used to evaluate the model --- #
        # have different metric objects for each phase
        self.metric_train = RDMetric()
        self.metric_val = RDMetric()
        self.metric_test = RDMetric()

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
        self.desc_mask = X[:, 4]  # (N, 4, L) -> (N, L)
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
        return H_k

    def H_desc(self, H_all: torch.Tensor) -> torch.Tensor:
        """
        :param H_all (N, L, H)
        :return H_desc (N, L - (K + 3), H)
        """
        N, L, H = H_all.size()
        desc_mask = self.desc_mask.unsqueeze(2).expand(H_all.shape)
        H_desc = torch.masked_select(H_all, desc_mask.bool())  # (N, L, H), (N, L, H) -> (N * (L - (K + 3)) * H)
        H_desc = H_desc.reshape(N, L - (self.hparams['k'] + 3), H)  # (N * (L - (K + 3)) * H) -> (N, L - (K + 3), H)
        return H_desc

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
        H_all = self.forward(X)  # (N, 3, L) -> (N, L, H)
        S_wisdom = self.S_wisdom(H_all)  # (N, L, H) -> (N, W)
        P_wisdom = F.softmax(S_wisdom, dim=1)  # (N, W) -> (N, W)
        return P_wisdom

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> dict:
        X, y = batch
        H_all = self.forward(X)  # (N, 3, L) -> (N, L, H)
        S_wisdom = self.S_wisdom(H_all)  # (N, L, H) -> (N, |W|)
        train_loss = F.cross_entropy(S_wisdom, y)  # (N, |W|), (N,) -> (N,)
        train_loss = train_loss.sum()  # (N,) -> (1,)
        P_wisdom = F.softmax(S_wisdom, dim=1)  # (N, |W|) -> (N, |W|)
        # so that the metrics accumulate over the course of this epoch
        # why dict? - just a boilerplate
        return {
            # you cannot change the keyword for the loss
            "loss": train_loss,
            # this is to update metrics after an epoch ends
            "P_wisdom": P_wisdom.detach(),
            # the same as above; for updating metrics
            "y": y.detach()
        }

    def on_train_batch_end(self, outputs: dict, *args, **kwargs) -> None:
        # watch the loss for this batch
        self.log("Train/Loss", outputs['loss'])
        self.metric_train.update(outputs['P_wisdom'], outputs['y'])

    def training_epoch_end(self, outputs: List[dict]) -> None:
        # to see an average performance over the batches in this specific epoch
        avg_loss = torch.stack([output['loss'] for output in outputs]).mean()
        self.log("Train/Average Loss", avg_loss)

    def on_train_epoch_end(self) -> None:
        rank_mean, rank_median, rank_std, top1, top3, top5 = self.metric_train.compute()
        self.metric_train.reset()
        self.log("Train/Rank Mean", rank_mean)
        self.log("Train/Rank Median", rank_median)
        self.log("Train/Rank Standard Deviation", rank_std)
        self.log("Train/Top 1 Accuracy", top1)
        self.log("Train/Top 3 Accuracy", top3)
        self.log("Train/Top 5 Accuracy", top5)

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> dict:
        return self.training_step(batch, batch_idx)

    def on_validation_batch_end(self, outputs: dict, *args, **kwargs) -> None:
        self.log("Validation/Loss", outputs['loss'])
        self.metric_val.update(outputs['P_wisdom'], outputs['y'])

    def validation_epoch_end(self, outputs: List[dict]) -> None:
        # to see an average performance over the batches in this specific epoch
        avg_loss = torch.stack([output['loss'] for output in outputs]).mean()
        self.log("Validation/Average Loss", avg_loss)

    def on_validation_epoch_end(self) -> None:
        # log the metrics
        rank_mean, rank_median, rank_std, top1, top3, top5 = self.metric_val.compute()
        self.metric_val.reset()
        self.log("Validation/Rank Mean", rank_mean)
        self.log("Validation/Rank Median", rank_median)
        self.log("Validation/Rank Standard Deviation", rank_std)
        self.log("Validation/Top 1 Accuracy", top1)
        self.log("Validation/Top 3 Accuracy", top3)
        self.log("Validation/Top 5 Accuracy", top5)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        X, y = batch
        P_wisdom = self.P_wisdom(X)
        self.metric_test.update(preds=P_wisdom, targets=y)

    def test_epoch_end(self, outputs: List[dict]) -> None:
        # log the metrics
        rank_mean, rank_median, rank_std, top1, top3, top5 = self.metric_test.compute()
        total = self.metric_test.total
        self.metric_test.reset()
        self.log("Test/Total samples", total)
        self.log("Test/Rank Mean", rank_mean)
        self.log("Test/Rank Median", rank_median)
        self.log("Test/Rank Standard Deviation", rank_std)
        self.log("Test/Top 1 Accuracy", top1)
        self.log("Test/Top 3 Accuracy", top3)
        self.log("Test/Top 5 Accuracy", top5)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Instantiates and returns the optimizer to be used for this model
        e.g. torch.optim.Adam
        """
        # The authors used Adam, so we might as well use it as well.
        return torch.optim.AdamW(self.parameters(), lr=self.hparams['lr'])


class RDAlpha(RD):
    """
    @eubinecto
    The first prototype.
    S_wisdom = S_wisdom_literal
    trained on: wisdom2def only.
    """

    def S_wisdom(self, H_all: torch.Tensor) -> torch.Tensor:
        H_k = self.H_k(H_all)  # (N, L, H) -> (N, K, H)
        S_wisdom = self.S_wisdom_literal(H_k)  # (N, K, H) -> (N, |W|)
        return S_wisdom


class FCLayer(nn.Module):
    """
    Reference:
    https://github.com/monologg/R-BERT
    """

    def __init__(self, input_dim, output_dim, dropout_rate=0.0, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

        torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)


class RDBeta(RD):
    """
    @ohsuz
    The second prototype.
    S_wisdom = S_wisdom_literal + S_wisdom_figurative
    trained on: wisdom2def only.
    """

    def __init__(self, k: int, lr: float, bert_mlm: BertForMaskedLM,
                 wisdom2subwords: torch.Tensor, wiskeys: torch.Tensor):
        super().__init__(k, lr, bert_mlm, wisdom2subwords)
        self.register_buffer("wiskeys", wiskeys)  # (|W|,)
        self.hidden_size = bert_mlm.config.hidden_size
        self.dr_rate = 0.0
        # fully- connected layers
        self.cls_fc = FCLayer(self.hidden_size, self.hidden_size // 3, self.dr_rate)
        self.wisdom_fc = FCLayer(self.hidden_size, self.hidden_size // 3, self.dr_rate)
        self.desc_fc = FCLayer(self.hidden_size, self.hidden_size // 3, self.dr_rate)
        self.final_fc = FCLayer(self.hidden_size, self.hidden_size, self.dr_rate, False)

    def S_wisdom(self, H_all: torch.Tensor) -> torch.Tensor:
        H_k = self.H_k(H_all)  # (N, L, H) -> (N, K, H)
        S_wisdom = self.S_wisdom_literal(H_k) + self.S_wisdom_figurative(H_all)  # (N, |W|) + (N, |W|) -> (N, |W|)
        return S_wisdom

    def S_wisdom_figurative(self, H_all: torch.Tensor) -> torch.Tensor:
        """
        param: H_all (N, L, H)
        return: S_wisdom_figurative (N, |W|)
        """
        W_embed = self.bert_mlm.bert.embeddings.word_embeddings(self.wiskeys)  # (|W|,) -> (|W|, H)

        H_cls = H_all[:, 0]  # (N, L, H) -> (N, H)
        H_wisdom = torch.mean(self.H_k(H_all), dim=1)  # (N, L, H) -> (N, K, H) -> (N, H)
        H_desc = torch.mean(self.H_desc(H_all), dim=1)  # (N, L, H) -> (N, L - (K + 3), H) -> (N, H)

        # Dropout -> tanh -> fc_layer
        H_cls = self.cls_fc(H_cls)  # (N, H) -> (N, H//3)
        H_wisdom = self.wisdom_fc(H_wisdom)  # (N, H) -> (N, H//3)
        H_desc = self.desc_fc(H_desc)  # (N, H) -> (N, H//3)

        # Concat -> fc_layer
        H_concat = torch.cat([H_cls, H_wisdom, H_desc], dim=-1)  # (N, H//3) X 3 -> (N, H)
        H_final = self.final_fc(H_concat)  # (N, H) -> (N, H)
        S_wisdom_figurative = torch.einsum("nh,hw->nw", H_final, W_embed.T)  # (N, H) * (H, |W|)-> (N, |W|)
        return S_wisdom_figurative


class RDGamma(RD):
    """
    @eubinecto
    S_wisdom  = S_wisdom_literal + S_wisdom_figurative
    but the way we get S_wisdom_figurative is much simplified, compared with RDBeta.
    """

    def __init__(self, k: int, lr: float, pooler_size: int, loss_func: str,
                 bert_mlm: BertForMaskedLM, wisdom2subwords: torch.Tensor):
        super().__init__(k, lr, bert_mlm, wisdom2subwords)
        # (N, K, H) -> (N, H)
        # a linear pooler
        self.save_hyperparameters(Namespace(pooler_size=pooler_size, loss_func=loss_func))
        # a pooler is a multilayer perceptron that pools wisdom_embeddings from wisdom2subwords_embeddings
        self.pooler = torch.nn.Sequential(
            torch.nn.Linear(self.hparams['k'], pooler_size),
            torch.nn.ReLU(),  # for non-linearity
            torch.nn.Linear(pooler_size, 1),
            torch.nn.ReLU()  # for another non-linearity
        )

    def S_wisdom(self, H_all: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        H_k = self.H_k(H_all)  # (N, L, H) -> (N, K, H)
        S_wisdom_literal = self.S_wisdom_literal(H_k)
        S_wisdom_figurative = self.S_wisdom_figurative(H_all)
        S_wisdom = S_wisdom_literal + S_wisdom_figurative  # (N, |W|) + (N, |W|) -> (N, |W|)
        return S_wisdom, S_wisdom_literal, S_wisdom_figurative

    def S_wisdom_figurative(self, H_all: torch.Tensor) -> torch.Tensor:
        # --- draw the embeddings for wisdoms from  the embeddings of wisdom2subwords -- #
        # this is to use as less of newly initialised weights as possible
        wisdom2subwords_embeddings_ = self.bert_mlm.bert \
            .embeddings.word_embeddings(self.wisdom2subwords)  # (W, K)  -> (W, K, H)
        wisdom2subwords_embeddings = torch.einsum('wkh->whk', wisdom2subwords_embeddings_)  # (W, K, H) -> (W, H, K)
        wisdom_embeddings_ = self.pooler(wisdom2subwords_embeddings)  # (W, H, K) * (K, 1) -> (W, H, 1)
        wisdom_embeddings = wisdom_embeddings_.squeeze()  # (W, H)
        # --- draw H_wisdom from H_desc with attention --- #
        H_k_ = self.H_k(H_all)  # (N, L, H) -> (N, K, H)
        H_k = torch.einsum("nkh->nhk", H_k_)
        H_wisdom = self.pooler(H_k).squeeze()  # (N, K, H) -pooling-> (N, H, 1) -> (N, H)
        # --- now compare H_wisdom with all the wisdoms --- #
        S_wisdom_figurative = torch.einsum("nh,wh->nw", H_wisdom, wisdom_embeddings)  # (N, H) * (W, H) -> (N, W)
        return S_wisdom_figurative

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> dict:
        X, y = batch
        H_all = self.forward(X)  # (N, 3, L) -> (N, L, H)
        S_wisdom, S_wisdom_literal, S_wisdom_figurative = self.S_wisdom(H_all)  # (N, L, H) -> (N, |W|)
        if self.hparams['loss_func'] == "cross_entropy":
            loss = F.cross_entropy(S_wisdom, y).sum()  # (N, |W|), (N,) -> (N,) -> (1,)
        elif self.hparams['loss_func'] == "cross_entropy_with_mtl":
            loss = F.cross_entropy(S_wisdom, y).sum()  # (N, |W|), (N,) -> (N,) -> (1,)
            loss += F.cross_entropy(S_wisdom_literal, y).sum()  # multi-task learning
            loss += F.cross_entropy(S_wisdom_figurative, y).sum()  # multi-task learning
            # S_wisdom_literal = torch.log_softmax(S_wisdom_literal, dim=1)
            # S_wisdom_figurative = torch.log_softmax(S_wisdom_figurative, dim=1)
            # # mse outperforms kl_div: https://arxiv.org/abs/2105.08919
            # # KD library gets use of MSE:
            # # https://github.com/SforAiDl/KD_Lib/blob/df4d9e5c0a494410cb2994e3a1d5902afdccf0d6/KD_Lib/KD/vision/vanilla/vanilla_kd.py#L69-L71
            # # you add this to the cross entropy loss
            # loss += F.mse_loss(S_wisdom_literal, S_wisdom_figurative)
        else:
            raise ValueError(f"Invalid loss_func: {self.hparams['loss_func']}")
        P_wisdom = F.softmax(S_wisdom, dim=1)  # (N, |W|) -> (N, |W|)
        return {
            # you cannot change the keyword for the loss
            "loss": loss,
            "P_wisdom": P_wisdom.detach(),
            "y": y.detach()
        }

    def P_wisdom(self, X: torch.Tensor) -> torch.Tensor:
        """
        :param X: (N, 3, L)
        :return P_wisdom: (N, |W|), normalized over dim 1.
        """
        H_all = self.forward(X)
        S_wisdom, _, _ = self.S_wisdom(H_all)
        P_wisdom = F.softmax(S_wisdom, dim=1)
        return P_wisdom
