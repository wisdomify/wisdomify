from typing import List, Tuple
from torch.utils.data import Dataset
from transformers import BertTokenizerFast
import torch
from wisdomify.builders import build_X, build_y


class WisdomDataset(Dataset):
    def __init__(self,
                 wisdom2sent: List[Tuple[str, str]],
                 tokenizer: BertTokenizerFast,
                 k: int,
                 vocab: List[str]):
        # (N, 3, L)
        self.X = build_X(wisdom2sent, tokenizer, k)
        # (N,)
        self.y = build_y(wisdom2sent, vocab)

    def __len__(self) -> int:
        """
        Returning the size of the dataset
        :return:
        """
        return self.y.shape[0]

    def upsample(self, repeat: int):
        """
        this is to try upsampling the batch by simply repeating the instances.
        https://github.com/eubinecto/fruitify/issues/7#issuecomment-860603350
        :return:
        """
        self.X = self.X.repeat(repeat, 1, 1)  # (N, 3, L) -> (N * repeat, 3, L)
        self.y = self.y.repeat(repeat)  # (N,) -> (N * repeat, )

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.LongTensor]:
        """
        Returns features & the label
        :param idx:
        :return:
        """
        return self.X[idx], self.y[idx]

