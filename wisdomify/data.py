import torch
from typing import Tuple, Optional, List
from datasets import load_dataset
from datasets.dataset_dict import DatasetDict
from datasets.dataset_dict import Dataset as HuggingFaceDataset
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from transformers import BertTokenizerFast
from wisdomify.builders import XBuilder, YBuilder


class WisdomDataset(Dataset):
    def __init__(self,
                 X: torch.Tensor,
                 y: torch.Tensor):
        self.X = X
        self.y = y

    def __len__(self) -> int:
        """
        Returning the size of the dataset
        :return:
        """
        return self.y.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.LongTensor]:
        """
        Returns features & the label
        :param idx:
        :return:
        """
        return self.X[idx], self.y[idx]

    def to(self, device: torch.device):
        self.X.to(device)
        self.y.to(device)


class WisdomDataModule(LightningDataModule):
    def __init__(self,
                 config: dict,
                 X_builder: XBuilder,
                 y_builder: YBuilder,
                 tokenizer: BertTokenizerFast,
                 device: torch.device):
      
        super().__init__()
        self.data_version: str = config['data_version']
        self.data_name: str = config['data_name']
        self.k: int = config['k']
        self.wisdoms: List[str] = config['wisdoms']
        self.batch_size: int = config['batch_size']
        self.num_workers: int = config['num_workers']
        self.shuffle: bool = config['shuffle']
        self.X_builder = X_builder
        self.y_builder = y_builder
        self.tokenizer = tokenizer
        self.device = device
        self.story: Optional[DatasetDict] = None
        self.dataset_train: Optional[WisdomDataset] = None
        self.dataset_val: Optional[WisdomDataset] = None
        self.dataset_test: Optional[WisdomDataset] = None

    def prepare_data(self):
        """
        prepare the data needed. (eg. downloading)
        """
        self.story = load_dataset(path="wisdomify/story",
                                  name=self.data_name,
                                  script_version=f"version_{self.data_version}")

    def setup(self, stage: Optional[str] = None) -> None:
        """
        convert dataset to desired form. (eg. pre-process)
        """
        x_col = 'wisdom'
        y_col = 'def' if self.data_name == 'definition' else 'eg' if self.data_name == 'example' else None

        if y_col is None:
            raise ValueError("Invalid data_name")
        self.dataset_train = self.build_dataset(self.story['train'], x_col, y_col)
        self.dataset_val = self.build_dataset(self.story['validation'], x_col, y_col)
        self.dataset_test = self.build_dataset(self.story['test'], x_col, y_col)

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

    def build_dataset(self, raw_data: HuggingFaceDataset, x_col: str, y_col: str) -> WisdomDataset:
        """
        This function convert
        raw dataset from huggingface api (which has form of DatasetDict)
        to WisdomDataset.

        :param raw_data: raw dataset from huggingface api
        :param x_col: name of x column
        :param y_col: name of y column
        :return:
        """
        wisdom2sent: List[Tuple[str, str]] = [
            (row[x_col], row[y_col])
            for row in raw_data
        ]
        X = self.X_builder(wisdom2sent)
        y = self.y_builder(wisdom2sent, self.wisdoms)
        dataset = WisdomDataset(X, y)
        return dataset
