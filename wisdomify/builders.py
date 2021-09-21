"""
all the functions for building tensors are defined here.
"""
from typing import List, Tuple
import torch
from transformers import BertTokenizerFast


class Builder:
    @staticmethod
    def build_X(*args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def build_y(wisdom2sent: List[Tuple[str, str]], vocab: List[str]) -> torch.LongTensor:
        """
        :param wisdom2sent:
        :param vocab:
        :return: (N, )
        """
        return torch.LongTensor([
            vocab.index(wisdom)
            for wisdom in [wisdom for wisdom, _ in wisdom2sent]
        ])

    @staticmethod
    def build_wisdom2subwords(tokenizer: BertTokenizerFast, k: int, wisdoms: List[str]) -> torch.Tensor:
        """
        [ ...,
          ...,
          ...
          [98, 122, 103, 103]
        ]
        :param wisdoms:
        :param tokenizer:
        :param k:
        :return:
        """
        mask_id = tokenizer.mask_token_id
        pad_id = tokenizer.pad_token_id
        encoded = tokenizer(text=wisdoms,
                            add_special_tokens=False,
                            # do not assume that the tokens have been tokenized to words.
                            is_split_into_words=False,
                            padding='max_length',
                            max_length=k,  # set to k
                            return_tensors="pt")
        input_ids = encoded['input_ids']
        input_ids[input_ids == pad_id] = mask_id  # replace them with masks
        return input_ids

    @staticmethod
    def build_wiskeys(tokenizer: BertTokenizerFast, wisdoms: List[str]) -> torch.Tensor:
        """
        wiskeys!
        """
        encoded = tokenizer(text=wisdoms,
                            add_special_tokens=False,
                            # assume that the tokens have already been tokenized to words.
                            is_split_into_words=True,
                            return_tensors="pt")
        input_ids = encoded['input_ids']  # (W, 1)
        return input_ids.squeeze()  # (W, 1) -> (W,)


class BuilderZero(Builder):
    @staticmethod
    def build_X(wisdom2sent: List[Tuple[str, str]], tokenizer: BertTokenizerFast, k: int) -> torch.Tensor:
        """
        :param wisdom2sent:
        :param tokenizer:
        :param k:
        :return: (N, 3, L)
        """
        sents = [sent for _, sent in wisdom2sent]
        lefts = [" ".join(["[MASK]"] * k)] * len(sents)
        rights = sents

        encodings = tokenizer(text=lefts,
                              text_pair=rights,
                              return_tensors="pt",
                              add_special_tokens=True,
                              truncation=True,
                              padding=True,
                              verbose=True)

        return torch.stack([encodings['input_ids'],
                            # token type for the padded tokens? -> they are masked with the
                            # attention mask anyways
                            # https://github.com/google-research/bert/issues/635#issuecomment-491485521
                            encodings['token_type_ids'],
                            encodings['attention_mask']], dim=1)


class BuilderOne(Builder):
    @staticmethod
    def build_X(wisdom2sent: List[Tuple[str, str]], tokenizer: BertTokenizerFast, k: int) -> torch.Tensor:
        """
        param wisdom2sent: 아이고... [WISDOM]이라더니, 오늘 하필 비가 오네.
        return: (N, 4, L)
        (N, 1) - input_ids
        (N, 2) - token_type_ids
        (N, 3) - attention_mask
        (N, 4) - wisdom_mask
        아이고, 가는 날이 장날이라더니, 오늘 하필 비가 오네.
        -> [CLS], ... 아, ##이고, [MASK] * K, 이라더니, 오늘, ..., [SEP].
        """
        sents = [sent for _, sent in wisdom2sent]
        sents = [
            sent.replace("[WISDOM]", " ".join(["[MASK]"] * k))
            for sent in sents
        ]

        encodings = tokenizer(text=sents,
                              return_tensors="pt",
                              add_special_tokens=True,
                              truncation=True,
                              padding=True,
                              verbose=True)
        mask_id: int = tokenizer.mask_token_id
        input_ids: torch.Tensor = encodings['input_ids']
        token_type_ids: torch.Tensor = encodings['token_type_ids']
        attention_mask: torch.Tensor = encodings['attention_mask']
        # another dimension is added.
        wisdom_mask: torch.Tensor = BuilderOne.build_wisdom_mask(input_ids, mask_id)

        return torch.stack([input_ids,
                            token_type_ids,
                            attention_mask,
                            wisdom_mask], dim=1)

    @staticmethod
    def build_wisdom_mask(input_ids: torch.Tensor, mask_id: int) -> torch.Tensor:
        return torch.where(input_ids == mask_id, 1, 0)
