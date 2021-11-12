import torch
from typing import Tuple, Optional, List
from wandb.wandb_run import Run
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from transformers import BertTokenizerFast
from wisdomify import flows
from wisdomify.tensors import (
    InputsBuilder,
    TargetsBuilder,
    Wisdom2DefInputsBuilder,
    Wisdom2EgInputsBuilder
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
        self.train_flow()("d", self.config)
        self.val_test_flow()("d", self.config)

    def setup(self, stage: Optional[str] = None) -> None:
        """
        setup the builders.
        """
        # --- set up the builders --- #
        # build the datasets
        train_flow = self.train_flow()("d", self.config)
        val_test_flow = self.val_test_flow()("d", self.config)
        train = [(row[0], row[1]) for row in train_flow.all_table.data]
        val, test = [(row[0], row[1]) for row in val_test_flow.val_table.data], \
                    [(row[0], row[1]) for row in val_test_flow.test_table.data]
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

    def predict_dataloader(self):
        # TODO: what is this? never seen this before...?
        # has this been added?
        pass

    def build_dataset(self, wisdom2desc: List[Tuple[str, str]], wisdoms: List[str]) -> WisdomifyDataset:
        """
        """
        inputs_builder, targets_builder = self.tensor_builders()
        X = inputs_builder(wisdom2desc)
        y = targets_builder(wisdom2desc, wisdoms)
        return WisdomifyDataset(X, y)

    def train_flow(self) -> flows.DatasetFlow:
        raise NotImplementedError

    def val_test_flow(self) -> flows.Wisdom2QueryFlow:
        return flows.Wisdom2QueryFlow(self.run, self.config['val_test_ver'])

    def tensor_builders(self) -> Tuple[InputsBuilder, TargetsBuilder]:
        raise NotImplementedError


class Wisdom2DefDataModule(WisdomifyDataModule):

    def train_flow(self) -> flows.Wisdom2DefFlow:
        return flows.Wisdom2DefFlow(self.run, self.config['train_ver'])

    def tensor_builders(self) -> Tuple[InputsBuilder, TargetsBuilder]:
        return Wisdom2DefInputsBuilder(self.tokenizer, self.k, self.device), TargetsBuilder(self.device)


class Wisdom2EgDataModule(WisdomifyDataModule):

    def train_flow(self) -> flows.Wisdom2EgFlow:
        return flows.Wisdom2EgFlow(self.run, self.config['train_ver'])

    def tensor_builders(self) -> Tuple[InputsBuilder, TargetsBuilder]:
        return Wisdom2EgInputsBuilder(self.tokenizer, self.k, self.device), TargetsBuilder(self.device)
