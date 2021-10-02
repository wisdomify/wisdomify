import pytorch_lightning as pl
import torch
import argparse
from wisdomify.loaders import load_device
from wisdomify.utils import Experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ver", type=str,
                        default="0")
    device = load_device()
    args = parser.parse_args()
    ver: str = args.ver
    exp = Experiment.load(ver, device)
    trainer = pl.Trainer(gpus=torch.cuda.device_count(),
                         # do not save checkpoints to a file.
                         logger=False)

    trainer.test(model=exp.rd, datamodule=exp.data_module, verbose=True)


if __name__ == '__main__':
    main()
