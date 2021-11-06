import torch
from typing import Tuple, Optional, List
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from transformers import BertTokenizerFast
from wisdomify.builders import XBuilder, YBuilder, Wisdom2DefXBuilder, Wisdom2EgXBuilder
from wandb.wandb_run import Run
from wisdomify.downloaders import (
    Wisdom2TestDownloader,
    Wisdom2DefDownloader,
    Wisdom2EgDownloader, Wisdom2DescDownloader
)


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


class Wisdom2DescDataModule(LightningDataModule):
    def __init__(self,
                 config: dict,
                 tokenizer: BertTokenizerFast,
                 wisdoms: List[str],
                 run: Run,
                 device: torch.device):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.wisdoms = wisdoms
        self.run = run
        self.device = device
        self.wisdom2desc_type = config['wisdom2desc_type']
        self.k = config["k"]
        self.batch_size = config['batch_size']
        self.shuffle = config['shuffle']
        self.num_workers = config['num_workers']
        # --- to be downloaded & built --- #
        self.train_dataset: Optional[WisdomifyDataset] = None
        self.val_dataset: Optional[WisdomifyDataset] = None
        self.test_dataset: Optional[WisdomifyDataset] = None

    def prepare_data(self):
        """
        prepare: download all data needed for this from wandb to local.
        """
        self.wisdom2desc_downloader()(self.config["wisdom2desc_ver"])
        Wisdom2TestDownloader(self.run)(self.config["wisdom2test_ver"])

    def setup(self, stage: Optional[str] = None) -> None:
        """
        setup the builders.
        """
        # --- set up the builders --- #
        # build the datasets
        train, val = self.wisdom2desc_downloader()(self.config["wisdom2desc_ver"])
        test = Wisdom2TestDownloader(self.run)(self.config["wisdom2test_ver"])
        self.train_dataset = self.build_dataset(train, self.wisdoms)
        self.val_dataset = self.build_dataset(val, self.wisdoms)
        self.test_dataset = self.build_dataset(test, self.wisdoms)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=self.shuffle, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def build_dataset(self, wisdom2desc: List[Tuple[str, str]], wisdoms: List[str]) -> WisdomifyDataset:
        """
        """
        X = self.x_builder()(wisdom2desc)
        y = YBuilder(self.device)(wisdom2desc, wisdoms)
        return WisdomifyDataset(X, y)

    # --- to be implemented --- #
    def wisdom2desc_downloader(self) -> Wisdom2DescDownloader:
        raise NotImplementedError

    def x_builder(self) -> XBuilder:
        raise NotImplementedError


class Wisdom2DefDataModule(Wisdom2DescDataModule):
    def wisdom2desc_downloader(self) -> Wisdom2DescDownloader:
        return Wisdom2DefDownloader(self.run)

    def x_builder(self) -> XBuilder:
        return Wisdom2DefXBuilder(self.tokenizer, self.k, self.device)


class Wisdom2EgDataModule(Wisdom2DescDataModule):

    def wisdom2desc_downloader(self) -> Wisdom2DescDownloader:
        return Wisdom2EgDownloader(self.run)

    def x_builder(self) -> XBuilder:
        return Wisdom2EgXBuilder(self.tokenizer, self.k, self.device)
