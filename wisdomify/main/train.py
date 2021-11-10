import torch
import argparse
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from wisdomify.artifacts import RDBuilder
from wisdomify.loaders import load_device
from wisdomify.paths import ROOT_DIR
from wisdomify.utils import Experiment


def main():
    # --- prep the arguments --- #
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="rd_alpha")
    parser.add_argument("--ver", type=str, default="a")
    parser.add_argument("--upload", dest='upload', action='store_true',
                        default=False)  # set this flag up if you want to save the logs & the model
    parser.add_argument("--use_gpu", type=bool, default=False)
    parser.add_argument("--online", dest="online", action="store_true",
                        default=False)
    args = parser.parse_args()
    model: str = args.model
    ver: str = args.ver
    upload: bool = args.upload
    # --- setup the device --- #
    device, n_gpus = load_device(args.use_gpu)
    online: bool = args.online
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
    logger = WandbLogger(log_model=False, offline=not online)
    # --- instantiate the trainer --- #
    trainer = pl.Trainer(gpus=n_gpus,
                         max_epochs=exp.config['max_epochs'],
                         default_root_dir=ROOT_DIR,
                         # do not save checkpoints every epoch - we need this especially for sweeping
                         # https://github.com/PyTorchLightning/pytorch-lightning/issues/5867#issuecomment-775223087
                         checkpoint_callback=False,
                         callbacks=[],
                         logger=logger)
    # --- start training with validation --- #
    trainer.fit(model=exp.rd, datamodule=exp.datamodule)
    # --- upload the model as an artifact to wandb, after training is done --- #
    if upload:
        builder = RDBuilder(model, ver)
        torch.save(exp.rd.state_dict(), builder.rd_bin_path)  # saving stat_dict only
        exp.tokenizer.save_pretrained(builder.tok_dir_path)  # save the tokenizer as well
        rd_artifact = builder(exp.rd, exp.tokenizer, exp.config)
        run.log_artifact(rd_artifact, aliases=[ver, "latest"])


if __name__ == '__main__':
    main()
