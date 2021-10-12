"""
all the functions for building tensors are defined here.
builders must accept device as one of the parameters.
"""
from typing import List, Tuple
import torch
from transformers import BertTokenizerFast, BatchEncoding


class TensorBuilder:

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """
        whatever it does,a builder outputs a Tensor.
        """
        raise NotImplementedError


class Wisdom2SubWordsBuilder(TensorBuilder):
    def __init__(self, tokenizer: BertTokenizerFast, k: int, device: torch.device):
        self.tokenizer = tokenizer
        self.k = k
        self.device = device

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
        return input_ids.to(self.device)


class WisKeysBuilder(TensorBuilder):
    def __init__(self, tokenizer: BertTokenizerFast, device: torch.device):
        self.tokenizer = tokenizer
        self.device = device

    def __call__(self, wisdoms: List[str]) -> torch.Tensor:
        # TODO: makes sure that the tokenizer treats each wisdom as a single token.
        encoded = self.tokenizer(text=wisdoms,
                                 add_special_tokens=False,
                                 return_tensors="pt")
        input_ids = encoded['input_ids']  # (W, 1)
        input_ids = input_ids.squeeze()  # (W, 1) -> (W,)
        return input_ids.to(self.device)


class XBuilder(TensorBuilder):
    def __init__(self, tokenizer: BertTokenizerFast, k: int, device: torch.device):
        self.tokenizer = tokenizer
        self.k = k
        self.device = device

    def __call__(self, wisdom2sent: List[Tuple[str, str]]) -> torch.Tensor:
        encodings = self.encode(wisdom2sent)
        input_ids: torch.Tensor = encodings['input_ids']
        cls_id: int = self.tokenizer.cls_token_id
        sep_id: int = self.tokenizer.sep_token_id
        pad_id: int = self.tokenizer.pad_token_id
        mask_id: int = self.tokenizer.mask_token_id
        
        wisdom_mask = torch.where(input_ids == mask_id, 1, 0)
        eg_mask = torch.where(((input_ids != cls_id) & (input_ids != sep_id) &
                               (input_ids != pad_id) & (input_ids != mask_id)), 1, 0)
        
        return torch.stack([input_ids,
                            # token type for the padded tokens? -> they are masked with the
                            # attention mask anyways
                            # https://github.com/google-research/bert/issues/635#issuecomment-491485521
                            encodings['token_type_ids'],
                            encodings['attention_mask'],
                            wisdom_mask,
                            eg_mask], dim=1).to(self.device)

    def encode(self, wisdom2sent: List[Tuple[str, str]]) -> BatchEncoding:
        raise NotImplementedError


class Wisdom2DefXBuilder(XBuilder):
    def encode(self, wisdom2def: List[Tuple[str, str]]) -> BatchEncoding:
        """
        param wisdom2def: (가는 날이 장날, 어떤 일을 하려고 하는데 뜻하지 않은 일을 공교롭게 당하는 것을 비유적으로 이르는 말)
        """
        defs = [sent for _, sent in wisdom2def]
        lefts = [" ".join(["[MASK]"] * self.k)] * len(defs)
        rights = defs
        encodings = self.tokenizer(text=lefts,
                                   text_pair=rights,
                                   return_tensors="pt",
                                   add_special_tokens=True,
                                   truncation=True,
                                   padding=True,
                                   verbose=True)
        return encodings


class Wisdom2EgXBuilder(XBuilder):
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


class YBuilder(TensorBuilder):

    def __init__(self, device: torch.device):
        self.device = device

    def __call__(self, wisdom2sent: List[Tuple[str, str]], wisdoms: List[str]) -> torch.LongTensor:
        """
        :param wisdom2sent:
        :param wisdoms:
        :return: (N, )
        """
        return torch.LongTensor([
            wisdoms.index(wisdom)
            for wisdom in [wisdom for wisdom, _ in wisdom2sent]
        ]).to(self.device)
