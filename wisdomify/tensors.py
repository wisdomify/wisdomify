"""
all the functions for building tensors are defined here.
builders must accept device as one of the parameters.
"""
import torch
from typing import List, Tuple
from transformers import BertTokenizerFast, BatchEncoding


class TensorBuilder:

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """
        whatever it does, a tensor builder outputs a Tensor.
        """
        raise NotImplementedError


class Wisdom2SubwordsBuilder(TensorBuilder):
    def __init__(self, tokenizer: BertTokenizerFast, k: int):
        self.tokenizer = tokenizer
        self.k = k

    def __call__(self, wisdoms: List[str]) -> torch.Tensor:
        mask_id = self.tokenizer.mask_token_id
        pad_id = self.tokenizer.pad_token_id
        # temporarily disable single-token status of the wisdoms
        wisdoms = [wisdom.split(" ") for wisdom in wisdoms]
        encoded = self.tokenizer(text=wisdoms,
                                 add_special_tokens=False,
                                 # should set this to True, as we already have the wisdoms split.
                                 is_split_into_words=True,
                                 padding='max_length',
                                 max_length=self.k,  # set to k
                                 return_tensors="pt")
        input_ids = encoded['input_ids']
        input_ids[input_ids == pad_id] = mask_id  # replace them with masks
        return input_ids


class WiskeysBuilder(TensorBuilder):
    def __init__(self, tokenizer: BertTokenizerFast):
        self.tokenizer = tokenizer

    def __call__(self, wisdoms: List[str]) -> torch.Tensor:
        # TODO: makes sure that the tokenizer treats each wisdom as a single token.
        encoded = self.tokenizer(text=wisdoms,
                                 add_special_tokens=False,
                                 return_tensors="pt")
        input_ids = encoded['input_ids']  # (W, 1)
        input_ids = input_ids.squeeze()  # (W, 1) -> (W,)
        return input_ids


class InputsBuilder(TensorBuilder):
    def __init__(self, tokenizer: BertTokenizerFast, k: int):
        self.tokenizer = tokenizer
        self.k = k

    def __call__(self, wisdom2desc: List[Tuple[str, str]]) -> torch.Tensor:
        encodings = self.encode(wisdom2desc)
        input_ids: torch.Tensor = encodings['input_ids']
        cls_id: int = self.tokenizer.cls_token_id
        sep_id: int = self.tokenizer.sep_token_id
        mask_id: int = self.tokenizer.mask_token_id
        
        wisdom_mask = torch.where(input_ids == mask_id, 1, 0)
        desc_mask = torch.where(((input_ids != cls_id) & (input_ids != sep_id) & (input_ids != mask_id)), 1, 0)
        
        inputs = torch.stack([input_ids,
                             encodings['token_type_ids'],
                             encodings['attention_mask'],
                             wisdom_mask,
                             desc_mask], dim=1)
        return inputs

    def encode(self, wisdom2desc: List[Tuple[str, str]]) -> BatchEncoding:
        raise NotImplementedError


class Wisdom2DefInputsBuilder(InputsBuilder):
    def encode(self, wisdom2def: List[Tuple[str, str]]) -> BatchEncoding:
        """
        param wisdom2def: (가는 날이 장날, 어떤 일을 하려고 하는데 뜻하지 않은 일을 공교롭게 당하는 것을 비유적으로 이르는 말)
        """
        rights = [sent for _, sent in wisdom2def]
        lefts = [" ".join(["[MASK]"] * self.k)] * len(rights)
        encodings = self.tokenizer(text=lefts,
                                   text_pair=rights,
                                   return_tensors="pt",
                                   add_special_tokens=True,
                                   truncation=True,
                                   padding=True,
                                   verbose=True)
        return encodings


class Wisdom2EgInputsBuilder(InputsBuilder):
    def encode(self, wisdom2eg: List[Tuple[str, str]]) -> BatchEncoding:
        """
        param wisdom2eg: (가는 날이 장날, 아이고... [WISDOM]이라더니, 오늘 하필 비가 오네.)
        return: (N, 4, L)
        (N, 1) - input_ids
        (N, 2) - token_type_ids
        (N, 3) - attention_mask
        (N, 4) - wisdom_mask
        아이고, 가는 날이 장날이라더니, 오늘 하필 비가 오네.
        -> [CLS], ... 아, ##이고, [MASK] * K, 이라더니, 오늘, ..., [SEP].
        """
        egs = [
            eg.replace("[WISDOM]", " ".join(["[MASK]"] * self.k))
            for _, eg in wisdom2eg
        ]
        encodings = self.tokenizer(text=egs,
                                   return_tensors="pt",
                                   add_special_tokens=True,
                                   truncation=True,
                                   padding=True,
                                   verbose=True)
        return encodings


class TargetsBuilder(TensorBuilder):

    def __call__(self, wisdom2desc: List[Tuple[str, str]], wisdoms: List[str]) -> torch.LongTensor:
        """
        :param wisdom2desc:
        :param wisdoms:
        :return: (N, )
        """
        targets = torch.LongTensor([
            wisdoms.index(wisdom)
            for wisdom in [wisdom for wisdom, _ in wisdom2desc]
        ])
        return targets
