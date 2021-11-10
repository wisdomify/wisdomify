import torch
from typing import Tuple, Optional, List
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from transformers import BertTokenizerFast
from wisdomify.builders import (
    XBuilder,
    YBuilder,
    Wisdom2DefXBuilder,
    Wisdom2EgXBuilder
)
from wandb.wandb_run import Run
from wisdomify.artifacts import (
    Wisdom2QueryLoader,
    Wisdom2DefLoader,
    Wisdom2EgLoader,
    ArtifactLoader
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


class WisdomifyDataModule(LightningDataModule):
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
        self.train_type = config['train_type']
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
        self.train_downloader()(self.config["train_ver"])
        self.val_test_downloader()(self.config["val_test_ver"])

    def setup(self, stage: Optional[str] = None) -> None:
        """
        setup the builders.
        """
        # --- set up the builders --- #
        # build the datasets
        train = self.train_downloader()(self.config["train_ver"])
        val, test = self.val_test_downloader()(self.config["val_test_ver"])
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
        x_builder, y_builder = self.tensor_builders()
        X = x_builder(wisdom2desc)
        y = y_builder(wisdom2desc, wisdoms)
        return WisdomifyDataset(X, y)

    def train_downloader(self) -> ArtifactLoader:
        raise NotImplementedError

    def val_test_downloader(self) -> ArtifactLoader:
        return Wisdom2QueryLoader(self.run)

    def tensor_builders(self) -> Tuple[XBuilder, YBuilder]:
        raise NotImplementedError


class Wisdom2DefDataModule(WisdomifyDataModule):

    def train_downloader(self) -> ArtifactLoader:
        return Wisdom2DefLoader(self.run)

    def tensor_builders(self) -> Tuple[XBuilder, YBuilder]:
        return Wisdom2DefXBuilder(self.tokenizer, self.k, self.device), YBuilder(self.device)


class Wisdom2EgDataModule(WisdomifyDataModule):

    def train_downloader(self) -> ArtifactLoader:
        return Wisdom2EgLoader(self.run)

    def tensor_builders(self) -> Tuple[XBuilder, YBuilder]:
        return Wisdom2EgXBuilder(self.tokenizer, self.k, self.device), YBuilder(self.device)
