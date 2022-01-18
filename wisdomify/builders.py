"""
all the functions for building tensors are defined here.
builders must accept device as one of the parameters.
"""
import torch
from typing import List, Tuple
from transformers import BertTokenizerFast, BatchEncoding


class InputsBuilder:
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


class Wisdom2DefBuilder(InputsBuilder):
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


class Wisdom2EgBuilder(InputsBuilder):
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


class TargetsBuilder:

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


# dataset builders should go under here.

# experiment builder ... should go under here should go under here

