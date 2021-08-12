import csv

import torch

from datasets import load_dataset
from datasets.dataset_dict import DatasetDict
from datasets.dataset_dict import Dataset as hg_dataset
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
from pytorch_lightning import LightningDataModule

from wisdomify.builders import build_X, build_y


class WisdomDataset(Dataset):
    def __init__(self,
                 X: torch.Tensor,
                 y: torch.Tensor):
        # (N, 3, L)
        self.X = X
        # (N,)
        self.y = y

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


class WisdomDataModule(LightningDataModule):
    def __init__(self,
                 data_version: str,
                 data_name: str,
                 k: int = None,
                 device=None,
                 vocab=None,
                 tokenizer=None,
                 batch_size: int = None,
                 num_workers: int = None,
                 train_ratio: float = None,
                 test_ratio: float = None,
                 shuffle: bool = None,
                 repeat: bool = None):
        super().__init__()
        self.data_version = data_version
        self.data_name = data_name
        self.k = k
        self.device = device
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.shuffle = shuffle
        self.repeat = repeat
        self.dataset_raw: Optional[DatasetDict] = None
        self.dataset_train: Optional[WisdomDataset] = None
        self.dataset_val: Optional[WisdomDataset] = None
        self.dataset_test: Optional[WisdomDataset] = None
        self.seed = 42

    def prepare_data(self):
        """
        prepare the data needed. (eg. downloading)
        """
        self.dataset_raw = load_dataset(path="DicoTiar/story",
                                        name=self.data_name,
                                        script_version=f"version_{self.data_version}")

        if self.dataset_raw['train'].version.version_str != self.data_version:
            raise NotImplementedError(f"This version is not valid: {self.data_version}")

    def setup(self, stage: Optional[str] = None) -> None:
        """
        convert dataset to desired form. (eg. pre-process)
        """
        def _convert_raw_to_embed(raw_data: hg_dataset, x_col: str, y_col: str) -> WisdomDataset:
            """
            This function convert
            raw data from huggingface api (which has form of DatasetDict)
            to WisdomDataset.

            :param raw_data: raw dataset from huggingface api
            :param x_col: name of x column
            :param y_col: name of y column
            :return:
            """
            wisdom2sent = list(map(lambda row: (row[x_col], row[y_col]), raw_data))

            X = build_X(wisdom2sent, self.tokenizer, self.k).to(self.device)
            y = build_y(wisdom2sent, self.vocab).to(self.device)

            data = WisdomDataset(X, y)
            data.upsample(self.repeat)

            return data

        x_col = 'wisdom'
        y_col = 'def' if self.data_name == 'definition' else 'eg' if self.data_name == 'example' else None

        if y_col is None:
            raise ValueError("Wrong data name")

        self.dataset_train = _convert_raw_to_embed(self.dataset_raw['train'], x_col, y_col)
        self.dataset_train = _convert_raw_to_embed(self.dataset_raw['validation'], x_col, y_col)
        self.dataset_train = _convert_raw_to_embed(self.dataset_raw['test'], x_col, y_col)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size,
                          shuffle=self.shuffle, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size,
                          shuffle=self.shuffle, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size,
                          shuffle=self.shuffle, num_workers=self.num_workers)

    @staticmethod
    def load_wisdom2sent(tsv_path: str) -> List[Tuple[str, str]]:
        """
        wisdom2def인 경우: 가는 날이 장날 -> 어떤 일을 하려고 하는데 뜻하지 않은 일을 공교롭게 당함.
        wisdom2eg인 경우: 가는 날이 장날 -> 한 달 전부터 기대하던 소풍날 아침에 굵은 빗방울이 떨어지기 시작했다.
        """
        with open(tsv_path, 'r') as fh:
            tsv_reader = csv.reader(fh, delimiter="\t")
            next(tsv_reader)
            return [
                (row[0], row[1])  # wisdom, sent pair
                for row in tsv_reader
            ]
