from typing import List, Tuple
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizerFast, AutoTokenizer
import torch
from wisdomify.builders import build_X, build_y
from wisdomify.loaders import load_wisdom2eg

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
                 k: int, 
                 device,
                 VOCAB,
                 tokenizer, 
                 batch_size: int,
                 num_workers: int):
        super().__init__()
        self.k = k
        self.device = device
        self.VOCAB = VOCAB
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.wisdom_data = load_wisdom2eg()
        self.dataset = None

    def prepare_data(self):
        X = build_X(self.wisdom_data, self.tokenizer, self.k).to(self.device)
        y = build_y(self.wisdom_data, self.VOCAB).to(self.device)

        self.dataset = WisdomDataset(X, y)

    def setup(self):
        n_total_data = len(self.dataset)
        n_train = int(n_total_data * 0.8)
        n_val = int(n_total_data * 0.1)
        n_test = n_total_data - n_train - n_val

        self.wisdom_train, self.wisdom_val, self.wisdom_test = random_split(self.dataset, [n_train, n_val, n_test])

    def train_dataloader(self):
        return DataLoader(self.wisdom_train, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def valid_dataloader(self):
        return DataLoader(self.wisdom_val, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.wisdom_test, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)