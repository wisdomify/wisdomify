"""
The reverse dictionary models below are based off of: https://github.com/yhcc/BertForRD/blob/master/mono/model/bert.py
"""
from argparse import Namespace
from typing import Tuple, List
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.optim import Optimizer
from transformers import BertTokenizerFast, AutoModelForMaskedLM, AutoConfig, AutoTokenizer
from transformers.models.bert.modeling_bert import BertForMaskedLM
from torch.nn import functional as F
from wisdomify.builders import build_X, build_vocab2subwords
from wisdomify.loaders import load_conf
from wisdomify.metrics import RDMetric
from wisdomify.paths import WISDOMIFIER_V_0_CKPT
from wisdomify.vocab import VOCAB


class RD(pl.LightningModule):
    """
    A reverse-dictionary. The model is based on
    """
    def __init__(self, bert_mlm: BertForMaskedLM, vocab2subwords: Tensor, k: int, lr: float):
        super().__init__()
        # -- the only network we need -- #
        self.bert_mlm = bert_mlm
        # -- to be used to compute S_word -- #
        self.vocab2subwords = vocab2subwords
        # -- to be used to evaluate the model -- #
        self.rd_metric = RDMetric()
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

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> dict:
        """
        :param batch: A tuple of X, y, subword_ids; ((N, 3, L), (N,),
        :param batch_idx: the index of the batch
        :return: (1,); the train_loss for this batch
        """
        X, y = batch
        # load the batches on the device.
        X = X.to(self.device)
        y = y.to(self.device)
        S_subword = self.forward(X)
        S_word = self.S_word(S_subword)
        # compute the loss
        train_loss = F.cross_entropy(S_word, y).sum()  # (N, |V|) -> (N,) -> scalar
        S_word_probs = F.softmax(S_word, dim=1)
        self.rd_metric.update(preds=S_word_probs, targets=y)
        median, var, top1, top10, top100 = self.rd_metric.compute()
        # evaluate the model on the batch.
        # we need this to check if the model is overfitting.
        # return a batch dict.
        #
        return {
            'loss': train_loss,
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
        self.logger.experiment.add_scalar("Train/Average Loss",
                                          # y - coord
                                          avg_train_loss,
                                          # x - coord; you can choose th
                                          self.current_epoch)
        self.logger.experiment.add_scalar("Train/Average Median", avg_median, self.current_epoch)
        self.logger.experiment.add_scalar("Train/Average Variance", avg_var, self.current_epoch)
        self.logger.experiment.add_scalar("Train/Average Top 1 Acc", avg_top1, self.current_epoch)
        self.logger.experiment.add_scalar("Train/Average Top 10 Acc", avg_top10, self.current_epoch)
        self.logger.experiment.add_scalar("Train/Average Top 100 Acc", avg_top100, self.current_epoch)

    def configure_optimizers(self) -> Optimizer:
        """
        Instantiates and returns the optimizer to be used for this model
        e.g. torch.optim.Adam
        """
        # The authors used Adam, so we might as well use it
        return torch.optim.AdamW(self.parameters(), lr=self.hparams['lr'])

    def test_step(self, batch, batch_idx, *args, **kwargs):
        self.rd_metric.reset()

        X, y = batch

        S_subword = self.forward(X)
        S_word = self.S_word(S_subword)
        S_word_probs = F.softmax(S_word, dim=1)

        self.rd_metric.update(preds=S_word_probs, targets=y)
        print("\nbatch:{}".format(batch_idx), self.rd_metric.compute())

    def on_test_end(self):
        median, var, top1, top10, top100 = self.rd_metric.compute()
        print("### final ###")
        print("median:", median)
        print("var:", var)
        print("top1:", top1)
        print("top10:", top10)
        print("top100:", top100)


class Wisdomifier:
    def __init__(self, rd: RD, tokenizer: BertTokenizerFast):
        self.rd = rd  # a trained reverse dictionary
        self.tokenizer = tokenizer

    def from_pretrained(self, ver: str, device) -> 'Wisdomifier':
        if ver == "0":
            conf = load_conf()
            wisdomifier_path = WISDOMIFIER_V_0_CKPT
            k: int = conf['versions'][ver]['k']
            bert_model: str = conf['versions'][ver]['bert_model']
            bert_mlm = AutoModelForMaskedLM.from_config(AutoConfig.from_pretrained(bert_model))
            self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
            vocab2subwords = build_vocab2subwords(self.tokenizer, k, VOCAB).to(device)

            self.rd = RD.load_from_checkpoint(wisdomifier_path, bert_mlm=bert_mlm, vocab2subwords=vocab2subwords)
            self.rd.to(device)
            self.rd.eval()
            wisdomifier = Wisdomifier(self.rd, self.tokenizer)

        else:
            raise NotImplementedError

        return wisdomifier

    def wisdomify(self, sents: List[str]) -> List[List[Tuple[str, float]]]:
        # get the X
        wisdom2sent = [("", desc) for desc in sents]
        X = build_X(wisdom2sent, tokenizer=self.tokenizer, k=self.rd.hparams['k']).to(self.rd.device)
        # get S_subword for this.
        S_subword = self.rd.forward(X)
        S_word = self.rd.S_word(S_subword)
        S_word_probs = F.softmax(S_word, dim=1)
        results = list()
        for S_word_prob in S_word_probs.tolist():
            wisdom2prob = [
                (wisdom, prob)
                for wisdom, prob in zip(VOCAB, S_word_prob)
            ]
            # sort and append
            results.append(sorted(wisdom2prob, key=lambda x: x[1], reverse=True))
        return results
