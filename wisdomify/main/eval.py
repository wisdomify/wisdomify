import pytorch_lightning as pl
import torch
import argparse
from wisdomify.loaders import load_device
from wisdomify.experiment import Experiment
from wisdomify.utils import WandBSupport


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ver", type=str,
                        default="0")
    device = load_device()
    args = parser.parse_args()
    ver: str = args.ver

    # --- W&B support object init --- #
    wandb_support = WandBSupport(ver=ver, run_type='eval')

    exp = Experiment.load(ver, device, wandb_support)

    # --- instantiate the training logger --- #
    logger = wandb_support.get_model_logger('eval_log')

    trainer = pl.Trainer(gpus=torch.cuda.device_count(),
                         # do not save checkpoints to a file.
                         logger=logger)

    trainer.test(model=exp.rd, datamodule=exp.data_module, verbose=True)

    wandb_support.push()


if __name__ == '__main__':
    main()
