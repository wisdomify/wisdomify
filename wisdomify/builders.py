"""
all the functions for building tensors are defined here.
"""
from typing import List, Tuple
import torch
from transformers import BertTokenizerFast


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


def build_vocab2subwords(tokenizer: BertTokenizerFast, k: int, vocab: List[str]) -> torch.Tensor:
    """
    [ ...,
      ...,
      ...
      [98, 122, 103, 103]
    ]
    :param vocab:
    :param tokenizer:
    :param k:
    :return:
    """
    mask_id = tokenizer.convert_tokens_to_ids("[MASK]")
    pad_id = tokenizer.convert_tokens_to_ids("[PAD]")
    encoded = tokenizer(text=vocab,
                        add_special_tokens=False,
                        padding='max_length',
                        max_length=k,  # set to k
                        return_tensors="pt")
    input_ids = encoded['input_ids']
    input_ids[input_ids == pad_id] = mask_id  # replace them with masks
    return input_ids
