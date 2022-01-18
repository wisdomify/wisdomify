import unittest
from typing import List
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
from wisdomify.models import (
    RD,
    Alpha,
    Beta
)
from wisdomify.loaders import load_config
from wisdomify.tensors import Wisdom2SubwordsBuilder, WiskeysBuilder, InputsBuilder,\
     Wisdom2DefInputsBuilder, TargetsBuilder
import torch


class RDCommonTest:
    class Test(unittest.TestCase):
        X: torch.Tensor
        y: torch.LongTensor
        N: int
        K: int
        L: int
        H: int
        W: int
        rd: RD

        @staticmethod
        def get_wisdom2sent():
            return [
                ('가는 날이 장날', '어떤 일을 하려고 하는데 뜻하지 않은 일을 공교롭게 당함'),
                ('산 넘어 산', '어려움을 해결했더니 또 어려움에 닥침')
            ]

        @classmethod
        def initialize(cls, inputs_builder: InputsBuilder, targets_builder: TargetsBuilder,
                       wisdoms: List[str],
                       H: int, W: int, K: int):
            wisdom2sent = cls.get_wisdom2sent()
            cls.X = inputs_builder(wisdom2sent)
            cls.y = targets_builder(wisdom2sent, wisdoms)
            cls.N = len(wisdom2sent)
            cls.L = cls.X.shape[2]
            cls.K = K
            cls.H = H
            cls.W = W

        # all RD classes must implement these tests.
        def test_forward_dim(self):
            H_all = self.rd.forward(self.X)
            self.assertEqual(self.N, H_all.shape[0])  # N
            self.assertEqual(self.L, H_all.shape[1])  # L
            self.assertEqual(self.H, H_all.shape[2])  # H

        def test_S_wisdom_dim(self):
            H_all = self.rd.forward(self.X)
            S_wisdom = self.rd.S_wisdom(H_all)
            self.assertEqual(self.N, S_wisdom.shape[0])  # N
            self.assertEqual(self.W, S_wisdom.shape[1])  # |W|

        def test_H_k_dim(self):
            H_all = self.rd.forward(self.X)
            H_k = self.rd.H_k(H_all)
            self.assertEqual(self.N, H_k.shape[0])  # N
            self.assertEqual(self.K, H_k.shape[1])  # K
            self.assertEqual(self.H, H_k.shape[2])  # H

        def test_S_wisdom_literal_dim(self):
            H_all = self.rd.forward(self.X)
            H_k = self.rd.H_k(H_all)
            S_wisdom_literal = self.rd.S_wisdom_literal(H_k)
            self.assertEqual(self.N, S_wisdom_literal.shape[0])  # N
            self.assertEqual(self.W, S_wisdom_literal.shape[1])  # |W|

        def test_P_wisdom_dim(self):
            P_wisdom = self.rd.P_wisdom(self.X)
            self.assertEqual(self.N, P_wisdom.shape[0])  # N
            self.assertEqual(self.W, P_wisdom.shape[1])  # |W|

        @classmethod
        def get_wisdoms_a(cls) -> List[str]:
            return [
                "가는 날이 장날",
                "갈수록 태산",
                "꿩 대신 닭",
                "등잔 밑이 어둡다",
                "소문난 잔치에 먹을 것 없다",
                "핑계 없는 무덤 없다",
                "고래 싸움에 새우 등 터진다",
                "서당개 삼 년이면 풍월을 읊는다",
                "원숭이도 나무에서 떨어진다",
                "산 넘어 산"
            ]


class RDAlphaTest(RDCommonTest.Test):
    rd: Alpha

    @classmethod
    def setUpClass(cls):
        # version - 0 and 1
        config = load_config()['rd_alpha']['a']
        bert = config['bert']
        wisdoms = cls.get_wisdoms_a()
        k = config['k']
        lr = config['lr']
        bert_mlm = AutoModelForMaskedLM.from_config(AutoConfig.from_pretrained(bert))
        tokenizer = AutoTokenizer.from_pretrained(bert)
        wisdom2subwords = Wisdom2SubwordsBuilder(tokenizer, k)(wisdoms)
        cls.rd = Alpha(k, lr, bert_mlm, wisdom2subwords)
        cls.initialize(Wisdom2DefInputsBuilder(tokenizer, k), TargetsBuilder(),
                       wisdoms, bert_mlm.config.hidden_size, len(wisdoms), k)

    def test_forward_dim(self):
        super(RDAlphaTest, self).test_forward_dim()

    def test_H_k_dim(self):
        super(RDAlphaTest, self).test_H_k_dim()

    def test_S_wisdom_dim(self):
        super(RDAlphaTest, self).test_S_wisdom_dim()

    def test_S_wisdom_literal_dim(self):
        super(RDAlphaTest, self).test_S_wisdom_literal_dim()

    def test_P_wisdom_dim(self):
        super(RDAlphaTest, self).test_P_wisdom_dim()


class RDBetaTest(RDCommonTest.Test):
    rd: Beta

    @classmethod
    def setUpClass(cls):
        # version - 2
        config = load_config()['rd_beta']['a']
        bert = config['bert']
        wisdoms = cls.get_wisdoms_a()
        k = config['k']
        lr = config['lr']
        bert_mlm = AutoModelForMaskedLM.from_config(AutoConfig.from_pretrained(bert))
        tokenizer = AutoTokenizer.from_pretrained(bert)
        # need to add the wisdoms to the tokenizer, and resize the embedding table as well
        tokenizer.add_tokens(wisdoms)
        bert_mlm.resize_token_embeddings(len(tokenizer))
        wisdom2subwords = Wisdom2SubwordsBuilder(tokenizer, k)(wisdoms)
        wiskeys = WiskeysBuilder(tokenizer)(wisdoms)
        cls.rd = Beta(k, lr, bert_mlm, wisdom2subwords, wiskeys)
        cls.initialize(Wisdom2DefInputsBuilder(tokenizer, k), TargetsBuilder(),
                       wisdoms, bert_mlm.config.hidden_size, len(wisdoms), k)

    def test_forward_dim(self):
        super(RDBetaTest, self).test_forward_dim()

    def test_H_k_dim(self):
        super(RDBetaTest, self).test_H_k_dim()

    def test_S_wisdom_dim(self):
        super(RDBetaTest, self).test_S_wisdom_dim()

    def test_S_wisdom_literal_dim(self):
        super(RDBetaTest, self).test_S_wisdom_literal_dim()

    def test_S_wisdom_figurative_dim(self):
        H_all = self.rd.forward(self.X)
        S_wisdom_figurative = self.rd.S_wisdom_figurative(H_all)
        self.assertEqual(self.N, S_wisdom_figurative.shape[0])  # N
        self.assertEqual(self.W, S_wisdom_figurative.shape[1])  # |W|

    def test_P_wisdom_dim(self):
        super(RDBetaTest, self).test_P_wisdom_dim()

# class TestRD(unittest.TestCase):
#     rd: RD
#     X: torch.Tensor
#     y: torch.Tensor
#     S: int  # the number of possible subwords in total
#     V: int  # the number of vocab (wisdom)
#
#     @staticmethod
#     def get_wisdom2sent():
#         return [('가는 날이 장날', '어떤 일을 하려고 하는데 뜻하지 않은 일을 공교롭게 당함')]
#
#     @classmethod
#     def setUpClass(cls) -> None:
#         # set up the mono rd
#         k = 11
#         batch_size = 10
#         lr = 0.001
#         bert_model = load_conf()['versions']['0']['bert_model']
#         bert_mlm = AutoModelForMaskedLM.from_pretrained(bert_model)
#         tokenizer = AutoTokenizer.from_pretrained(bert_model)
#         wisdom2sent = cls.get_wisdom2sent()
#         vocab2subwords = build_vocab2subwords(tokenizer, k=k, vocab=WISDOMS)
#         X = build_X_with_wisdom_mask(wisdom2sent, tokenizer, k)
#         y = build_y(wisdom2sent, WISDOMS)
#         cls.rd = RD(bert_mlm, vocab2subwords, k=k, lr=lr)
#         cls.S = tokenizer.vocab_size
#         # load a single batch
#         dataset = WisdomDataset(X, y)
#         loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#         X, y = next(iter(loader))  # just get the first batch.
#         cls.X = X
#         cls.y = y
#         cls.V = len(WISDOMS)
#
#     def test_forward_dim(self):
#         # (N, 3, L) -> (N, K, |S|)
#         S_subword = self.rd.forward(self.X)
#         self.assertEqual(S_subword.shape[0], self.X.shape[0])
#         self.assertEqual(S_subword.shape[1], self.rd.hparams['k'])
#         self.assertEqual(S_subword.shape[2], self.S)
#
#     def test_S_word_dim(self):
#         S_subword = self.rd.forward(self.X)
#         # (N, 3, L) -> (N, |V|)
#         S_word = self.rd.S_word(S_subword)
#         self.assertEqual(S_word.shape[0], self.X.shape[0])
#         self.assertEqual(S_word.shape[1], self.V)
#
#     def test_training_step_dim(self):
#         # (N, 3, L) -> scalar
#         loss = self.rd.training_step((self.X, self.y), 0)['loss']
#         self.assertEqual(len(loss.shape), 0)  # should be a scalar
