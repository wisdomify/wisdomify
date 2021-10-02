import pytorch_lightning as pl
import torch
import argparse
from wisdomify.utils import Experiment


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument("--ver", type=str,
                        default="0")

    args = parser.parse_args()
    ver: str = args.ver
    exp = Experiment.from_pretrained(ver, device)
    trainer = pl.Trainer(gpus=torch.cuda.device_count(),
                         # do not save checkpoints to a file.
                         logger=False)

    trainer.test(model=exp.rd, datamodule=exp.data_module, verbose=True)


if __name__ == '__main__':
    main()
