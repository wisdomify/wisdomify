from typing import List, Tuple, Optional
from pytorch_lightning import LightningDataModule
from wisdomify.builders import build_X, build_y
from torch.utils.data import Dataset, DataLoader, random_split
import csv
import torch
from os import path
from wisdomify.paths import WISDOMDATA_VER_DIR, WISDOMDATA_DIR


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
                 k: int = None,
                 device = None,
                 vocab = None,
                 tokenizer = None,
                 batch_size: int = None,
                 num_workers: int = None,
                 train_ratio: float = None,
                 test_ratio: float = None,
                 shuffle: bool = None,
                 repeat: bool = None):
        super().__init__()
        self.data_version = data_version
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
        self.dataset_all: Optional[WisdomDataset] = None
        self.dataset_train: Optional[WisdomDataset] = None
        self.dataset_val: Optional[WisdomDataset] = None
        self.dataset_test: Optional[WisdomDataset] = None

    def prepare_data(self):
        """
        prepare the data needed.
        """

        if not path.isdir(path.join(WISDOMDATA_DIR, f'version_{self.data_version}')):
            raise NotImplementedError(f"The version {self.data_version}'s data is not prepared.")

        version_dir = WISDOMDATA_VER_DIR
        wisdom2def_tsv_path = path.join(version_dir, "wisdom2def.tsv")
        wisdom2sent = self.load_wisdom2sent(wisdom2def_tsv_path)

        X = build_X(wisdom2sent, self.tokenizer, self.k).to(self.device)
        y = build_y(wisdom2sent, self.vocab).to(self.device)
        self.dataset_all = WisdomDataset(X, y)
        self.dataset_all.upsample(self.repeat)

    def setup(self, stage: Optional[str] = None) -> None:
        """
        역할?
        """
        n_total_data = len(self.dataset_all)
        n_train = int(n_total_data * self.train_ratio)
        n_test = int(n_total_data * self.test_ratio)
        n_val = n_total_data - (n_train + n_test)
        self.dataset_train, self.dataset_val, self.dataset_test = random_split(self.dataset_all,
                                                                               [n_train, n_val, n_test])

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
