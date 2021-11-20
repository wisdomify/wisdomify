"""
This script is to be used for hyper parameter tuning
"""
import torch
import argparse
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.loggers import WandbLogger
from wisdomify.connectors import connect_to_wandb
from wisdomify.constants import ROOT_DIR
from wisdomify.flows import ExperimentFlow

"""
bert: beomi/kcbert-base
desc: trained on `wisdom2def:a`
seed: 410
train_type: wisdom2def
train_ver: a
wisdoms_ver: a
val_test_ver: a
k: 11
lr: 0.00001
max_epochs: 50
batch_size: 30
num_workers: 4
shuffle: true
log_every_n_steps: 1
"""


def main():
    # parse the parameters here
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--ver", type=str)
    parser.add_argument("--bert", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--wisdom2def", type=str)
    parser.add_argument("--train_type", type=str)
    parser.add_argument("--train_ver", type=str)
    parser.add_argument("--wisdoms_ver", type=str)
    parser.add_argument("--val_test_ver", type=str)
    parser.add_argument("--k", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--max_epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--shuffle", type=int)
    parser.add_argument("--log_every_n_steps", type=int)
    parser.add_argument("--num_sanity_val_steps", type=int)
    args = parser.parse_args()
    config = vars(args)
    print(config)
    # --- set up the device to train the model with --- #
    # --- init a run  --- #
    with connect_to_wandb(job_type="sweep", config=config) as run:
        # --- build an experiment --- #
        flow = ExperimentFlow(run, config['model'], config['ver'])(mode="b", config=config)
        # --- instantiate the training logger --- #
        # A new W&B run will be created when training starts
        # if you have not created one manually before with wandb.init().
        logger = WandbLogger(log_model=False)
        # --- instantiate the trainer --- #
        # -- trainer는 flow로 만들지 않는다. 계속 옵션을 바꾸고 싶을때가 많을거라서, 그냥 이대로 두는게 좋다.
        trainer = Trainer(max_epochs=config['max_epochs'],
                          # wisdomify now supports multi-gpu training! (hardware-agnostic)
                          gpus=torch.cuda.device_count(),
                          # lightning_logs will be saved under this directory
                          default_root_dir=ROOT_DIR,
                          # each step means each batch. Don't set this too low
                          # https://youtu.be/3JgpG4K6HxA
                          log_every_n_steps=config["log_every_n_steps"],
                          # do not save anything, all we want is just logging metrics
                          # to find the best combination
                          enable_checkpointing=False,
                          logger=logger)
        # --- start training with validation --- #
        trainer.fit(model=flow.rd_flow.rd, datamodule=flow.datamodule)


if __name__ == '__main__':
    main()
