import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from wisdomify.connectors import connect_to_wandb
from wisdomify.loaders import load_config
from wisdomify.flows import ExperimentFlow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default="rd_gamma")
    parser.add_argument("--ver", type=str,
                        default="b")
    parser.add_argument("--gpu", dest="gpu",
                        action='store_true', default=False)
    args = parser.parse_args()
    model: str = args.model
    ver: str = args.ver
    gpu: bool = args.gpu
    config = load_config()[model][ver]
    gpus = torch.cuda.device_count() if gpu else 0
    # --- init a run instance --- #
    with connect_to_wandb(job_type="eval", config=config) as run:
        # --- load a pre-trained experiment --- #
        flow = ExperimentFlow(run, model, ver)(mode="d", config=config)
        flow.rd_flow.rd.eval()
        # --- instantiate the training logger --- #
        logger = WandbLogger(log_model=False)
        trainer = pl.Trainer(gpus=gpus,
                             # do not save checkpoints to a file.
                             logger=logger)
        # --- run the test on the test set! --- #
        trainer.test(model=flow.rd_flow.rd,
                     datamodule=flow.datamodule,
                     verbose=True)


if __name__ == '__main__':
    main()
