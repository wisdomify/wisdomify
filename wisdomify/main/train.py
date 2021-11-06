import torch
import argparse
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from wisdomify.builders import RDArtifactBuilder
from wisdomify.loaders import load_device
from wisdomify.paths import ROOT_DIR
from wisdomify.utils import Experiment


def main():
    # --- setup the device --- #
    device = load_device()
    # --- prep the arguments --- #
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="rd_beta")
    parser.add_argument("--ver", type=str, default="v0")
    parser.add_argument("--upload", type=int, default=1)  # set this to 0 if you don't want to upload the models.
    args = parser.parse_args()
    model: str = args.model
    ver: str = args.ver
    upload: int = args.upload
    # --- init a run  --- #
    run = wandb.init(name="wisdomify.main.train",
                     tags=[f"{model}:{ver}"],
                     dir=ROOT_DIR,
                     project="wisdomify",
                     entity="wisdomify")
    # --- build an experiment instance --- #
    exp = Experiment.build(model, ver, run, device)
    # --- instantiate the training logger --- #
    # A new W&B run will be created when training starts if you have not created one manually before with wandb.init().
    logger = WandbLogger(log_model=False)
    # --- instantiate the trainer --- #
    trainer = pl.Trainer(gpus=torch.cuda.device_count(),
                         max_epochs=exp.config['max_epochs'],
                         default_root_dir=ROOT_DIR,
                         # do not save checkpoints every epoch - we need this especially for sweeping
                         # https://github.com/PyTorchLightning/pytorch-lightning/issues/5867#issuecomment-775223087
                         checkpoint_callback=False,
                         callbacks=[],  # TODO: implement early stopping
                         logger=logger)
    # --- start training with validation --- #
    trainer.fit(model=exp.rd, datamodule=exp.datamodule)
    # --- upload the model as an artifact to wandb, after training is done --- #
    if upload:
        builder = RDArtifactBuilder(model, ver, exp.config)
        # save rd & tokenizer to local
        torch.save(exp.rd.state_dict(), builder.rd_bin_path)  # save the state dict only.
        exp.tokenizer.save_pretrained(builder.tok_dir_path)
        # upload them to wandb
        rd_artifact = builder()
        run.log_artifact(rd_artifact)


if __name__ == '__main__':
    main()
