import pytorch_lightning as pl
import torch
import argparse
from wisdomify.datasets import WisdomDataModule
from wisdomify.loaders import load_conf

from wisdomify.classes import WISDOMS
from wisdomify.models import Wisdomifier


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("--ver", type=str,
                        default="0")

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

    wisdomifier = Wisdomifier.from_pretrained(ver, device)

    data_module = WisdomDataModule(data_version=data_version,
                                   data_name=data_name,
                                   k=k,
                                   device=device,
                                   vocab=WISDOMS,
                                   tokenizer=wisdomifier.tokenizer,
                                   batch_size=batch_size,
                                   num_workers=num_workers,
                                   shuffle=False,
                                   repeat=repeat)

    trainer = pl.Trainer(gpus=torch.cuda.device_count(),
                         # do not save checkpoints to a file.
                         logger=False)

    trainer.test(model=wisdomifier.rd, datamodule=data_module, verbose=False)


if __name__ == '__main__':
    main()
