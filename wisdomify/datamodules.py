import torch
from typing import Tuple, Optional, List
from wandb.wandb_run import Run
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from transformers import BertTokenizerFast
from wisdomify import flows
from wisdomify.builders import (
    InputsBuilder,
    TargetsBuilder,
    Wisdom2DefBuilder,
    Wisdom2EgBuilder
)
from wisdomify.fetchers import fetch_wisdom2query, fetch_wisdom2def, fetch_wisdom2eg


class WisdomifyDataset(Dataset):
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


class WisdomifyDataModule(LightningDataModule):

    def __init__(self,
                 config: dict,
                 tokenizer: BertTokenizerFast,
                 wisdoms: List[str]):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.wisdoms = wisdoms
        self.k = config["k"]
        self.batch_size = config['batch_size']
        self.shuffle = config['shuffle']
        self.num_workers = config['num_workers']
        # --- to be downloaded & built --- #
        self.train: Optional[List[Tuple[str, str]]] = None
        self.val: Optional[List[Tuple[str, str]]] = None
        self.test: Optional[List[Tuple[str, str]]] = None
        self.train_dataset: Optional[WisdomifyDataset] = None
        self.val_dataset: Optional[WisdomifyDataset] = None
        self.test_dataset: Optional[WisdomifyDataset] = None

    def prepare_data(self):
        """
        prepare: download all data needed for this from wandb to local.
        """
        self.train = self.prepare_train()
        self.val, self.test = fetch_wisdom2query(self.config['val_test_ver'])

    def prepare_train(self) -> List[Tuple[str, str]]:
        raise NotImplementedError

    def prepare_builders(self) -> Tuple[InputsBuilder, TargetsBuilder]:
        raise NotImplementedError

    def setup(self, stage: Optional[str] = None) -> None:
        """
        setup the builders.
        """
        self.train_dataset = self.build_dataset(self.train, self.wisdoms)
        self.val_dataset = self.build_dataset(self.val, self.wisdoms)
        self.test_dataset = self.build_dataset(self.test, self.wisdoms)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=self.shuffle, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def predict_dataloader(self):
        # TODO: what is this? never seen this before...?
        # has this been added?
        pass

    def build_dataset(self, wisdom2desc: List[Tuple[str, str]], wisdoms: List[str]) -> WisdomifyDataset:
        """
        """
        inputs_builder, targets_builder = self.prepare_builders()
        X = inputs_builder(wisdom2desc)
        y = targets_builder(wisdom2desc, wisdoms)
        return WisdomifyDataset(X, y)


class Wisdom2DefDataModule(WisdomifyDataModule):

    def prepare_train(self) -> List[Tuple[str, str]]:
        return fetch_wisdom2def(self.config['train_ver'])

    def prepare_builders(self) -> Tuple[InputsBuilder, TargetsBuilder]:
        return Wisdom2DefBuilder(self.tokenizer, self.k), TargetsBuilder()


class Wisdom2EgDataModule(WisdomifyDataModule):

    def prepare_train(self) -> List[Tuple[str, str]]:
        return fetch_wisdom2eg(self.config['train_ver'])

    def prepare_builders(self) -> Tuple[InputsBuilder, TargetsBuilder]:
        return Wisdom2EgBuilder(self.tokenizer, self.k), TargetsBuilder()
