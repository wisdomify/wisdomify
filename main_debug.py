"""
https://pytorch-lightning.readthedocs.io/en/latest/common/debugging.html
"""

import torch.cuda
import argparse
import pytorch_lightning as pl
from wisdomify.connectors import connect_to_wandb
from wisdomify.loaders import load_config
from wisdomify.flows import ExperimentFlow


def main():
    # --- prep the arguments --- #
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", "--model", type=str, default="rd_alpha")
    parser.add_argument("--v", "--ver", type=str, default="a")
    parser.add_argument("--g", "--gpu", dest="g", action='store_true', default=False)
    args = parser.parse_args()
    model: str = args.m
    ver: str = args.v
    gpu: bool = args.g
    # --- set up the device to train the model with --- #
    gpus = torch.cuda.device_count() if gpu else 0
    # --- init a run  --- #
    config = load_config()[model][ver]
    with connect_to_wandb(job_type="debug", config=config) as run:
        # --- build an experiment --- #
        flow = ExperimentFlow(run, model, ver)(mode="b", config=config)
        # a trainer should be within the context of wandb, so that the
        # data can be pulled from wandb on training
        trainer = pl.Trainer(fast_dev_run=True,
                             gpus=gpus,
                             profiler="simple")
        # --- start training with validation --- #
        trainer.fit(model=flow.rd_flow.rd, datamodule=flow.datamodule)


if __name__ == '__main__':
    main()
