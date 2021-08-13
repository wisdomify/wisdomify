from typing import List, Tuple, Optional
from pytorch_lightning import LightningDataModule
from wisdomify.builders import build_X, build_y
from torch.utils.data import Dataset, DataLoader
import csv
import torch
from os import path
from wisdomify.paths import WISDOMDATA_DIR


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
                 data_train_path: str,
                 data_test_path: str,
                 k: int,
                 device,
                 vocab,
                 tokenizer,
                 batch_size: int,
                 num_workers: int,
                 shuffle: bool,
                 repeat: bool):
        super().__init__()
        self.data_train_path = data_train_path
        self.data_test_path = data_test_path
        self.k = k
        self.device = device
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.repeat = repeat
        self.wisdom2sent_train: Optional[List[Tuple[str, str]]] = None
        self.wisdom2sent_test: Optional[List[Tuple[str, str]]] = None
        self.dataset_train: Optional[WisdomDataset] = None
        self.dataset_test: Optional[WisdomDataset] = None

    def prepare_data(self):
        """
        prepare the data needed. (download, etc)
        """
        wisdom2sent_train_tsv_path = path.join(WISDOMDATA_DIR, self.data_train_path)
        self.wisdom2sent_train = self.load_wisdom2sent(wisdom2sent_train_tsv_path)
        if self.data_test_path:
            wisdom2sent_test_tsv_path = path.join(WISDOMDATA_DIR, self.data_test_path)
            self.wisdom2sent_test = self.load_wisdom2sent(wisdom2sent_test_tsv_path)

    def setup(self, stage: Optional[str] = None) -> None:
        """
        역할?
        """
        if stage == "fit":
            X_train = build_X(self.wisdom2sent_train, self.tokenizer, self.k).to(self.device)
            y_train = build_y(self.wisdom2sent_train, self.vocab).to(self.device)
            self.dataset_train = WisdomDataset(X_train, y_train)
            self.dataset_train.upsample(self.repeat)
        elif stage == "test":
            X_test = build_X(self.wisdom2sent_test, self.tokenizer, self.k).to(self.device)
            y_test = build_y(self.wisdom2sent_test, self.vocab).to(self.device)
            self.dataset_test = WisdomDataset(X_test, y_test)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size,
                          shuffle=self.shuffle, num_workers=self.num_workers)

    def val_dataloader(self):
        """
        implement this later.
        """
        raise NotImplementedError

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
            next(tsv_reader)  # skip the header
            return [
                (row[0], row[1])  # wisdom, sent pair
                for row in tsv_reader
            ]
