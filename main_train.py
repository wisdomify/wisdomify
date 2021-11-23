import torch.cuda
import wandb
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from wisdomify.connectors import connect_to_wandb
from wisdomify.loaders import load_config
from wisdomify.constants import ROOT_DIR
from wisdomify.flows import ExperimentFlow
from termcolor import colored


def main():
    # --- prep the arguments --- #
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", "--model", type=str, default="rd_alpha")
    parser.add_argument("--v", "--ver", type=str, default="a")
    parser.add_argument("--g", "--gpu", dest="g", action='store_true', default=False)
    parser.add_argument("--u", "--upload", dest='u', action='store_true', default=False)
    args = parser.parse_args()
    model: str = args.m
    ver: str = args.v
    gpu: bool = args.g
    upload: bool = args.u
    if not upload:
        print(colored("WARNING: YOU CHOSE NOT TO UPLOAD. NOTHING BUT LOGS WILL BE SAVED TO WANDB", color="red"))
    # --- set up the device to train the model with --- #
    gpus = torch.cuda.device_count() if gpu else 0
    # --- init a run  --- #
    config = load_config()[model][ver]
    with connect_to_wandb(job_type="train", config=config) as run:
        # --- build an experiment --- #
        flow = ExperimentFlow(run, model, ver)(mode="b", config=config)
        # --- instantiate the training logger --- #
        # A new W&B run will be created when training starts
        # if you have not created one manually before with wandb.init().
        logger = WandbLogger(log_model=False)
        # --- instantiate the trainer --- #
        # -- trainer는 flow로 만들지 않는다. 계속 옵션을 바꾸고 싶을때가 많을거라서, 그냥 이대로 두는게 좋다.
        trainer = pl.Trainer(max_epochs=config['max_epochs'],
                             # wisdomify now supports multi-gpu training! (hardware-agnostic)
                             gpus=gpus,
                             # lightning_logs will be saved under this directory
                             default_root_dir=ROOT_DIR,
                             # each step means each batch. Don't set this too low
                             # https://youtu.be/3JgpG4K6HxA
                             log_every_n_steps=config["log_every_n_steps"],
                             # do not save checkpoints every epoch - we need this especially for sweeping
                             # https://github.com/PyTorchLightning/pytorch-lightning/issues/5867#issuecomment-775223087
                             enable_checkpointing=config["enable_checkpointing"],
                             # set this to zero to prevent "calling metric.compute before metric.update" error
                             # https://forums.pytorchlightning.ai/t/validation-sanity-check/174/6
                             num_sanity_val_steps=config["num_sanity_val_steps"],
                             logger=logger)
        # --- start training with validation --- #
        trainer.fit(model=flow.rd_flow.rd, datamodule=flow.datamodule)
        # --- upload the model as an artifact to wandb, after training is done --- #
        if upload:
            # save the rd & the tokenizer locally, first
            trainer.save_checkpoint(flow.rd_flow.rd_ckpt_path)
            flow.rd_flow.tokenizer.save_pretrained(flow.rd_flow.tok_dir_path)
            artifact = wandb.Artifact(model, type="model")
            # add the paths to the artifact
            artifact.add_file(flow.rd_flow.rd_ckpt_path)
            artifact.add_dir(flow.rd_flow.tok_dir_path, "tokenizer")
            # add the config
            artifact.metadata = config
            #  upload both the model and the tokenizer to wandb
            run.log_artifact(artifact, aliases=[ver, "latest"])


if __name__ == '__main__':
    main()
