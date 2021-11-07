import pytorch_lightning as pl
import torch
import argparse
import wandb
from pytorch_lightning.loggers import WandbLogger
from wisdomify.loaders import load_device
from wisdomify.paths import ROOT_DIR
from wisdomify.utils import Experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default="rd_alpha")
    parser.add_argument("--ver", type=str,
                        default="latest")
    device = load_device()
    args = parser.parse_args()
    model: str = args.model
    ver: str = args.ver
    # --- init a run instance --- #
    run = wandb.init(name="wisdomify.main.eval",
                     tags=[f"{model}:{ver}"],
                     dir=ROOT_DIR,
                     project="wisdomify",
                     entity="wisdomify")
    # --- load a pre-trained experiment --- #
    exp = Experiment.load(model, ver, run, device)
    exp.rd.eval()
    # --- instantiate the training logger --- #
    logger = WandbLogger(log_model=False)
    trainer = pl.Trainer(gpus=torch.cuda.device_count(),
                         # do not save checkpoints to a file.
                         logger=logger)
    # --- run the test on the test set! --- #
    trainer.test(model=exp.rd, datamodule=exp.datamodule, verbose=True)


if __name__ == '__main__':
    main()
