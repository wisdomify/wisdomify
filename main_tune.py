"""
Finds an optimal value of lr & batch_size.
"""
import torch.cuda
import argparse
import pytorch_lightning as pl
from wisdomify.connectors import connect_to_wandb
from wisdomify.loaders import load_config
from wisdomify.flows import ExperimentFlow
import os


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
    with connect_to_wandb(job_type="tune", config=config) as run:
        # --- build an experiment --- #
        flow = ExperimentFlow(run, model, ver)(mode="b", config=config)
        # Run learning rate finder
        trainer = pl.Trainer(auto_lr_find=True,
                             gpus=gpus,
                             enable_checkpointing=False,
                             logger=False)
        try:
            results = trainer.tune(flow.rd_flow.rd, datamodule=flow.datamodule)
        except Exception as e:
            # whatever exception occurs, make sure to delete the cash
            os.system("rm lr_find*")  # just remove the file
            raise e
        else:
            lr_finder = results['lr_find']
            # results is a dictionary, so logging it should work
            run.log(lr_finder.results)


if __name__ == '__main__':
    main()
