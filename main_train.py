import wandb
import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from wisdomify.connectors import connect_to_wandb
from wisdomify.loaders import load_device, load_config
from wisdomify.constants import ROOT_DIR
from wisdomify.flows import ExperimentFlow


def main():
    # --- prep the arguments --- #
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="rd_alpha")
    parser.add_argument("--ver", type=str, default="a")
    parser.add_argument("--use_gpu", dest="use_gpu", action='store_true', default=False)
    parser.add_argument("--upload", dest='upload', action='store_true',
                        default=False)  # set this flag up if you want to save the logs & the model
    args = parser.parse_args()
    model: str = args.model
    ver: str = args.ver
    use_gpu: bool = args.use_gpu
    upload: bool = args.upload
    if not upload:
        print("########## WARNING: YOU CHOSE NOT TO UPLOAD. NOTHING BUT LOGS WILL BE SAVED TO WANDB #################")
    # --- set up the device to train the model with --- #
    device = load_device(use_gpu)
    # --- init a run  --- #
    config = load_config()
    with connect_to_wandb("train", config) as run:
        # --- build an experiment --- #
        flow = ExperimentFlow(run, model, ver, config,  "build", device)
        # --- instantiate the training logger --- #
        # A new W&B run will be created when training starts
        # if you have not created one manually before with wandb.init().
        logger = WandbLogger(log_model=False)
        # --- instantiate the trainer --- #
        # -- trainer는 flow로 만들지 않는다. 계속 옵션을 바꾸고 싶을때가 많을거라서, 그냥 이대로 두는게 좋다.
        trainer = pl.Trainer(max_epochs=flow.rd_flow.config['max_epochs'],
                             # wisdomify does not support multi-gpu training, as of right now
                             gpus=1 if device.type == "cuda" else 0,
                             # lightning_logs will be saved under this directory
                             default_root_dir=ROOT_DIR,
                             # each step means each batch. Don't set this too low
                             # https://youtu.be/3JgpG4K6HxA
                             log_every_n_steps=flow.rd_flow.config["log_every_n_steps"],
                             # do not save checkpoints every epoch - we need this especially for sweeping
                             # https://github.com/PyTorchLightning/pytorch-lightning/issues/5867#issuecomment-775223087
                             enable_checkpointing=flow.rd_flow.config["enable_checkpointing"],
                             # set this to zero to prevent "calling metric.compute before metric.update" error
                             # https://forums.pytorchlightning.ai/t/validation-sanity-check/174/6
                             num_sanity_val_steps=flow.rd_flow.config["num_sanity_val_steps"],
                             logger=logger)
        # --- start training with validation --- #
        trainer.fit(model=flow.rd_flow.rd, datamodule=flow.datamodule)
        # --- upload the model as an artifact to wandb, after training is done --- #
        if upload:
            # save the rd & the tokenizer locally
            torch.save(flow.rd_flow.rd.state_dict(), flow.rd_flow.rd_bin_path)  # saving stat_dict only
            flow.rd_flow.tokenizer.save_pretrained(flow.rd_flow.tok_dir_path)  # save the tokenizer as well
            artifact = wandb.Artifact(model, type="model")
            # add the paths to the artifact
            artifact.add_file(flow.rd_flow.rd_bin_path)
            artifact.add_dir(flow.rd_flow.tok_dir_path, "tokenizer")
            # add the config
            artifact.metadata = flow.rd_flow.config
            #  upload to wandb
            run.log_artifact(artifact, aliases=[ver, "latest"])


if __name__ == '__main__':
    main()
