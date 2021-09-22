import unittest
import torch
from typing import Tuple, List
from transformers import BertTokenizerFast
from wisdomify.builders import Builder, BuilderZero, BuilderOne
from wisdomify.loaders import load_conf
from wisdomify.vocab import VOCAB


class TestBuilder(unittest.TestCase):
    WISDOM2SENT: List[Tuple[str, str]]
    tokenizer: BertTokenizerFast
    k: int

    @classmethod
    def setUpClass(cls) -> None:
        conf = load_conf()['versions']['0']
        bert_model = conf['bert_model']
        cls.tokenizer = BertTokenizerFast.from_pretrained(bert_model)
        cls.tokenizer.add_tokens(new_tokens=VOCAB)
        cls.k = conf['k']

    @classmethod
    def get_wisdom2sent(cls) -> List[Tuple[str, str]]:
        return [
            ('가는 날이 장날', '어떤 일을 하려고 하는데 뜻하지 않은 일을 공교롭게 당함'),
            ('서당개 삼 년이면 풍월을 읊는다', '아무리 무지해도 오래오래 듣거나 보게 되면 자연히 잘하게 된다'),
        ]  # data to test.

    def test_build_wisdom2subwords_dim(self):
        wisdom2subwords = Builder.build_wisdom2subwords(self.tokenizer, self.k, VOCAB)  # should be (V,K)
        self.assertEqual(len(VOCAB), wisdom2subwords.shape[0])
        self.assertEqual(self.k, wisdom2subwords.shape[1])

    def test_build_wiskeys_dim(self):
        # this should be added someday.
        wiskeys = Builder.build_wiskeys(self.tokenizer, VOCAB)
        self.assertEqual(len(VOCAB), wiskeys.shape[0])
        self.assertEqual(1, len(wiskeys.shape))  # should be a vector

    def test_build_y_dim(self):
        y = Builder.build_y(self.get_wisdom2sent(), VOCAB)  # should be (N,)
        self.assertEqual(len(self.get_wisdom2sent()), y.shape[0])
        self.assertEqual(1, len(y.shape))  # should be a 1-dim vector


class TestBuilderZero(TestBuilder):

    @classmethod
    def setUpClass(cls) -> None:
        conf = load_conf()['versions']['0']
        bert_model = conf['bert_model']
        cls.tokenizer = BertTokenizerFast.from_pretrained(bert_model)
        cls.k = conf['k']

    def test_build_X_dim(self):
        X = BuilderZero.build_X(self.get_wisdom2sent(), self.tokenizer, self.k)  # should be (N, 3, L)
        self.assertEqual(len(self.get_wisdom2sent()), X.shape[0])
        self.assertEqual(3, X.shape[1])
        self.assertEqual(31, X.shape[2])  # the max length.. what is it?


class TestBuilderOne(TestBuilder):

    @classmethod
    def setUpClass(cls) -> None:
        conf = load_conf()['versions']['4']
        bert_model = conf['bert_model']
        cls.tokenizer = BertTokenizerFast.from_pretrained(bert_model)
        cls.k = conf['k']

    @classmethod
    def get_wisdom2sent(cls) -> List[Tuple[str, str]]:
        return [
            ('가는 날이 장날', '개발을 하러 토끼굴에 가야지라고 생각하고 있었는데, [WISDOM]이라고 토끼굴 소독이 오늘 진행돼서 출입 할 수가 없다고 한다.'),
            ('산 넘어 산', '모델 개선을 위해 코드 수정만 하면 끝날 줄 알았는데, 수정해야할 게 [WISDOM]이다. 테스트 코드, 데이터 코드 이것 저것 수정할게 많다.')
        ]  # data to test against

    def test_build_X_dim(self):
        X = BuilderOne.build_X(self.get_wisdom2sent(), self.tokenizer, self.k)  # should be (N, 4, L)
        self.assertEqual(len(self.get_wisdom2sent()), X.shape[0])
        self.assertEqual(4, X.shape[1])
        self.assertEqual(15, X.shape[2])  # the max length.. what is it?

    def test_build_wisdom_mask(self):
        X = BuilderOne.build_X(self.get_wisdom2sent(), self.tokenizer, self.k)[0]  # should be (N, 4, L)
        mask_from_input_ids = X[0] == self.tokenizer.mask_token_id
        mask_from_wisdom_mask = X[3] == 1

        is_equal = torch.all(mask_from_input_ids.eq(mask_from_wisdom_mask))

        self.assertEqual(is_equal, True)
