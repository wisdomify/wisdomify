"""
define any functions that output a tensor here.
"""

import torch
from typing import List
from transformers import BertTokenizerFast, BertTokenizer


def wisdom2subwords(tokenizer: BertTokenizer, wisdoms: List[str], k: int) -> torch.Tensor:
    # temporarily disable single-token status of the wisdoms
    wisdoms = [wisdom.split(" ") for wisdom in wisdoms]
    encoded = tokenizer(text=wisdoms,
                        add_special_tokens=False,
                        # should set this to True, as we already have the wisdoms split.
                        is_split_into_words=True,
                        padding='max_length',
                        max_length=k,  # set to k
                        return_tensors="pt")
    input_ids = encoded['input_ids']
    input_ids[input_ids == tokenizer.pad_token_id] = tokenizer.mask_token_id  # replace them with masks
    return input_ids


def wiskeys(tokenizer: BertTokenizer,  wisdoms: List[str]) -> torch.Tensor:
    encoded = tokenizer(text=wisdoms,
                        add_special_tokens=False,
                        return_tensors="pt")
    input_ids = encoded['input_ids']  # (W, 1)
    input_ids = input_ids.squeeze()  # (W, 1) -> (W,)
    return input_ids
