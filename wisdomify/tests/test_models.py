import unittest
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM
from wisdomify.builders import build_vocab2subwords
from wisdomify.models import RD
from wisdomify.loaders import load_conf, load_wisdom2def
from wisdomify.vocab import VOCAB
from wisdomify.datasets import WisdomDataset


class TestRD(unittest.TestCase):
    rd: RD
    X: torch.Tensor
    y: torch.Tensor
    S: int  # the number of possible subwords in total
    V: int  # the number of classes (wisdom)

    @classmethod
    def setUpClass(cls) -> None:
        # set up the mono rd
        k = 11
        batch_size = 10
        lr = 0.001
        bert_model = load_conf()['bert_model']
        bert_mlm = AutoModelForMaskedLM.from_pretrained(bert_model)
        tokenizer = AutoTokenizer.from_pretrained(bert_model)
        vocab2subwords = build_vocab2subwords(tokenizer, k=k, vocab=VOCAB)
        cls.rd = RD(bert_mlm, vocab2subwords, k=k, lr=lr)
        cls.S = tokenizer.vocab_size
        # load a single batch
        dataset = WisdomDataset(load_wisdom2def(), tokenizer, k=k, vocab=VOCAB)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        X, y = next(iter(loader))  # just get the first batch.
        cls.X = X
        cls.y = y
        cls.V = len(VOCAB)

    def test_forward_dim(self):
        # (N, 3, L) -> (N, K, |S|)
        S_subword = self.rd.forward(self.X)
        self.assertEqual(S_subword.shape[0], self.X.shape[0])
        self.assertEqual(S_subword.shape[1], self.rd.hparams['k'])
        self.assertEqual(S_subword.shape[2], self.S)

    def test_S_word_dim(self):
        # (N, 3, L) -> (N, K, |S|)
        S_subword = self.rd.forward(self.X)
        # (N, K, |S|) -> (N, |V|)
        S_word = self.rd.S_word(S_subword)
        self.assertEqual(S_word.shape[0], self.X.shape[0])
        self.assertEqual(S_word.shape[1], self.V)

    def test_training_step_dim(self):
        # (N, 3, L) -> scalar
        loss = self.rd.training_step((self.X, self.y), 0)
        self.assertEqual(len(loss.shape), 0)  # should be a scalar
