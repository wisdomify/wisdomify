import csv

import torch

import pandas as pd

from typing import List, Tuple, Optional
from datasets.dataset_dict import DatasetDict
from datasets.dataset_dict import Dataset as HuggingFaceDataset
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

from wisdomify.builders import build_X, build_y
from wisdomify.utils import get_wandb_artifact


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
                 dtype: str,
                 k: int = None,
                 device=None,
                 vocab=None,
                 tokenizer=None,
                 batch_size: int = None,
                 num_workers: int = None,
                 shuffle: bool = None,
                 repeat: int = None):

        super().__init__()
        self.data_version = data_version
        self.data_name = data_name
        self.dtype = dtype
        self.k = k
        self.device = device
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.repeat = repeat
        self.story: Optional[DatasetDict] = None
        self.dataset_train: Optional[WisdomDataset] = None
        self.dataset_val: Optional[WisdomDataset] = None
        self.dataset_test: Optional[WisdomDataset] = None

    def prepare_data(self):
        """
        prepare the data needed. (eg. downloading)
        """

        wandb_artifact_dir = get_wandb_artifact(
            name=self.data_name,
            ver=self.data_version,
            dtype='dataset'
        )

        self.story = {
            'train': self.read_wandb_artifact(wandb_artifact_dir, 'training.tsv'),
            'validation': self.read_wandb_artifact(wandb_artifact_dir, 'validation.tsv'),
            'test': self.read_wandb_artifact(wandb_artifact_dir, 'test.tsv'),
        }

        # self.story = load_dataset(path="wisdomify/story",
        #                           name=self.data_name,
        #                           script_version=f"version_{self.data_version}")
        #
        # if self.story['train'].version.version_str != self.data_version:
        #     raise NotImplementedError(f"This version is not valid: {self.data_version}")

    def setup(self, stage: Optional[str] = None) -> None:
        """
        convert dataset to desired form. (eg. pre-process)
        """
        x_col = 'wisdom'
        y_col = 'def' if self.dtype == 'definition' else 'eg' if self.dtype == 'example' else None

        if y_col is None:
            raise ValueError("Wrong data name")

        self.dataset_train = self.convert_raw_to_embed(self.story['train'], x_col, y_col)
        self.dataset_val = self.convert_raw_to_embed(self.story['validation'], x_col, y_col)
        self.dataset_test = self.convert_raw_to_embed(self.story['test'], x_col, y_col)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size,
                          shuffle=self.shuffle, num_workers=self.num_workers)

    def val_dataloader(self):
        # why should we set shuffle for val & test to False?
        # 오히려 셔플을 하면 지표가 자꾸 바뀌겠다. 비교를 해야하는데.. 그러면 안되겠지.
        return DataLoader(self.dataset_val, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    # --- user-defined methods --- #
    @staticmethod
    def load_wisdom2sent(tsv_path: str) -> List[Tuple[str, str]]:
        """
        wisdom2def인 경우: 가는 날이 장날 -> 어떤 일을 하려고 하는데 뜻하지 않은 일을 공교롭게 당함.
        wisdom2eg인 경우: 가는 날이 장날 -> 한 달 전부터 기대하던 소풍날 아침에 굵은 빗방울이 떨어지기 시작했다.
        """
        with open(tsv_path, 'r') as fh:
            tsv_reader = csv.reader(fh, delimiter="\t")
            next(tsv_reader)  # skip the header
            return [
                (row[0], row[1])  # wisdom, sent pair
                for row in tsv_reader
            ]

    def convert_raw_to_embed(self, raw_data: HuggingFaceDataset, x_col: str, y_col: str) -> WisdomDataset:
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

    @staticmethod
    def read_wandb_artifact(dl_dir: str, file: str):
        return pd\
            .read_csv(f"{dl_dir}/{file}", sep='\t' if file.split('.')[-1] == 'tsv' else ',')\
            .to_dict(orient='records')
