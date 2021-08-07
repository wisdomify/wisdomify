import unittest
from typing import Tuple, List
from transformers import BertTokenizerFast
from wisdomify.builders import build_X, build_y, build_vocab2subwords
from wisdomify.loaders import load_conf
from wisdomify.vocab import VOCAB


class TestBuilders(unittest.TestCase):
    wisdom2sent: List[Tuple[str, str]]
    tokenizer: BertTokenizerFast
    k: int

    @staticmethod
    def get_base_data_set():
        return [('가는 날이 장날', '어떤 일을 하려고 하는데 뜻하지 않은 일을 공교롭게 당함')]

    @classmethod
    def setUpClass(cls) -> None:
        bert_model = load_conf()['versions']['0']['bert_model']
        cls.wisdom2sent = cls.get_base_data_set()
        cls.tokenizer = BertTokenizerFast.from_pretrained(bert_model)
        cls.k = 11

    def test_build_X_dim(self):
        X = build_X(self.wisdom2sent, self.tokenizer, self.k)  # should be (N, 3, L)
        self.assertEqual(len(self.wisdom2sent), X.shape[0])
        self.assertEqual(3, X.shape[1])
        self.assertEqual(27, X.shape[2])  # the max length.. what is it?

    def test_build_y_dim(self):
        y = build_y(self.wisdom2sent, VOCAB)  # should be (N,)
        self.assertEqual(len(self.wisdom2sent), y.shape[0])
        self.assertEqual(1, len(y.shape))  # should be a 1-dim vector

    def test_build_vocab2subwords_dim(self):
        vocab2subwords = build_vocab2subwords(self.tokenizer, self.k, VOCAB)  # should be (V,K)
        self.assertEqual(len(VOCAB), vocab2subwords.shape[0])
        self.assertEqual(self.k, vocab2subwords.shape[1])
