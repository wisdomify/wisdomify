import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from wisdomify.connectors import connect_to_wandb
from wisdomify.loaders import load_device, load_config
from wisdomify.flows import ExperimentFlow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default="rd_alpha")
    parser.add_argument("--ver", type=str,
                        default="latest")
    parser.add_argument("--use_gpu", dest="use_gpu",
                        action='store_true', default=False)
    args = parser.parse_args()
    model: str = args.model
    ver: str = args.ver
    use_gpu: bool = args.use_gpu
    device = load_device(use_gpu)
    config = load_config()[model][ver]
    # --- init a run instance --- #
    with connect_to_wandb(job_type="eval", config=config) as run:
        # --- load a pre-trained experiment --- #
        flow = ExperimentFlow(run, model, ver, device)(mode="d", config=config)
        flow.rd_flow.rd.eval()
        # --- instantiate the training logger --- #
        logger = WandbLogger(log_model=False)
        trainer = pl.Trainer(gpus=1 if device.type == "cuda" else 0,
                             # do not save checkpoints to a file.
                             logger=logger)
        # --- run the test on the test set! --- #
        trainer.test(model=flow.rd_flow.rd,
                     datamodule=flow.datamodule,
                     verbose=True)


if __name__ == '__main__':
    main()
