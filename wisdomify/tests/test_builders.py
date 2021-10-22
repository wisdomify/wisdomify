"""
at the moment, the tests test for version_0 only.
"""
import unittest
from wisdomify.loaders import load_conf, load_device
from wisdomify.builders import (
    Wisdom2SubWordsBuilder,
    WisKeysBuilder,
    Wisdom2DefXBuilder,
    Wisdom2EgXBuilder,
    YBuilder
)
from transformers import AutoTokenizer, BertTokenizerFast
from typing import Tuple, List
import torch


class TensorBuilderTest(unittest.TestCase):

    tokenizer: BertTokenizerFast
    k: int
    device: torch.device
    wisdoms: List[str]

    @classmethod
    def setUpClass(cls):
        conf = load_conf()['versions']['0']
        cls.tokenizer = AutoTokenizer.from_pretrained('beomi/kcbert-base')
        cls.tokenizer.add_tokens(conf['wisdoms'])
        cls.k = conf['model']['k']
        cls.wisdoms = conf['wisdoms']
        cls.device = load_device()

    @classmethod
    def get_wisdom2sent(cls) -> List[Tuple[str, str]]:
        return [
            ('가는 날이 장날', '어떤 일을 하려고 하는데 뜻하지 않은 일을 공교롭게 당함'),
            ('서당개 삼 년이면 풍월을 읊는다', '아무리 무지해도 오래오래 듣거나 보게 되면 자연히 잘하게 된다'),
        ]  # data to test.


class Wisdom2SubwordsBuilderTest(TensorBuilderTest):

    wisdom2subwords_builder: Wisdom2SubWordsBuilder

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.tokenizer.add_tokens(cls.wisdoms)  # treat wisdoms as single tokens
        cls.wisdom2subwords_builder = Wisdom2SubWordsBuilder(cls.tokenizer, cls.k, cls.device)

    def test_build_wisdom2subwords_dim(self):
        wisdom2subwords = self.wisdom2subwords_builder(self.wisdoms)  # should be (V,K)
        # --- check dimensions --- #
        self.assertEqual(len(self.wisdoms), wisdom2subwords.shape[0])
        self.assertEqual(self.k, wisdom2subwords.shape[1])

    def test_build_wisdom2subwords_semantics(self):
        # -- check the semantics --- #
        wisdom2subwords = self.wisdom2subwords_builder(self.wisdoms)  # should be (V,K)
        wisdom2subwords = wisdom2subwords != self.tokenizer.mask_token_id
        # there must be at least one non-mask tensor.
        self.assertIn(torch.ones(wisdom2subwords.shape[1]), wisdom2subwords)


class WiskeysBuilderTest(TensorBuilderTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.tokenizer.add_tokens(cls.wisdoms)  # treat wisdoms as single tokens
        cls.wiskeys_builder = WisKeysBuilder(cls.tokenizer, cls.device)

    def test_build_wiskeys_dim(self):
        wiskeys = self.wiskeys_builder(self.wisdoms)
        self.assertEqual(len(self.wisdoms), wiskeys.shape[0])
        self.assertEqual(1, len(wiskeys.shape))  # should be a vector

    def test_build_wiskeys_semantics(self):
        #  the wiskeys must not contain the unknown token
        wiskeys = self.wiskeys_builder(self.wisdoms)
        wiskeys = wiskeys != self.tokenizer.unk_token_id
        self.assertTrue(torch.all(wiskeys, dim=0))


class Wisdom2DefXBuilderTest(TensorBuilderTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.wisdom2def_X_builder = Wisdom2DefXBuilder(cls.tokenizer, cls.k, cls.device)

    def test_build_X_dim(self):
        X = self.wisdom2def_X_builder(self.get_wisdom2sent())  # should be (N, 3, L)
        self.assertEqual(len(self.get_wisdom2sent()), X.shape[0])
        self.assertEqual(5, X.shape[1])  # input_ids, token_type_ids, attention_mask, wisdom_mask
        self.assertEqual(28, X.shape[2])  # the max length.. what is it?


class Wisdom2EgXBuilderTest(TensorBuilderTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.wisdom2eg_X_builder = Wisdom2EgXBuilder(cls.tokenizer, cls.k, cls.device)

    @classmethod
    def get_wisdom2sent(cls) -> List[Tuple[str, str]]:
        return [
            ('가는 날이 장날', '개발을 하러 토끼굴에 가야지라고 생각하고 있었는데, [WISDOM]이라고 토끼굴 소독이 오늘 진행돼서 출입 할 수가 없다고 한다.'),
            ('산 넘어 산', '모델 개선을 위해 코드 수정만 하면 끝날 줄 알았는데, 수정해야할 게 [WISDOM]이다. 테스트 코드, 데이터 코드 이것 저것 수정할게 많다.')
        ]  # data to test against

    def test_build_X_dim(self):
        X = self.wisdom2eg_X_builder(self.get_wisdom2sent())  # should be (N, 4, L)
        self.assertEqual(len(self.get_wisdom2sent()), X.shape[0])
        self.assertEqual(5, X.shape[1])
        self.assertEqual(41, X.shape[2])


class YBuilderTest(TensorBuilderTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.y_builder = YBuilder(cls.device)

    def test_build_y_dim(self):
        y = self.y_builder(self.get_wisdom2sent(), self.wisdoms)  # should be (N,)
        self.assertEqual(len(self.get_wisdom2sent()), y.shape[0])
        self.assertEqual(1, len(y.shape))  # should be a 1-dim vector
