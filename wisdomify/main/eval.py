import pytorch_lightning as pl
import torch
import os
import argparse

import wandb
from pytorch_lightning.loggers import WandbLogger

from wisdomify.datasets import WisdomDataModule
from wisdomify.loaders import load_conf

from wisdomify.vocab import VOCAB
from wisdomify.models import Wisdomifier

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("--ver", type=str,
                        default="2")

    args = parser.parse_args()
    ver: str = args.ver

    conf = load_conf()
    vers = conf['versions']
    if ver not in vers.keys():
        raise NotImplementedError("Invalid version provided".format(ver))

    selected_ver = vers[ver]
    data_version: str = selected_ver['data_version']
    # TODO: should enable to load both example and definition on one dataset
    data_name: str = selected_ver['data_name'][0]
    batch_size: int = selected_ver['batch_size']
    repeat: bool = selected_ver['repeat']
    num_workers: int = selected_ver['num_workers']
    k: int = selected_ver['k']
    dtype: str = selected_ver['dtype'][0]

    wisdomifier = Wisdomifier.from_pretrained(ver, device)

    data_module = WisdomDataModule(data_version=data_version,
                                   data_name=data_name,
                                   dtype=dtype,
                                   k=k,
                                   device=device,
                                   vocab=VOCAB,
                                   tokenizer=wisdomifier.tokenizer,
                                   batch_size=batch_size,
                                   num_workers=num_workers,
                                   shuffle=False,
                                   repeat=repeat)

    wandb_logger = WandbLogger(project='wisdomify', entity='wisdomify', name='evaluation_log')

    trainer = pl.Trainer(gpus=torch.cuda.device_count(),
                         # do not save checkpoints to a file.
                         logger=wandb_logger)

    trainer.test(model=wisdomifier.rd, datamodule=data_module, verbose=False)

    wandb.finish()


if __name__ == '__main__':
    main()
