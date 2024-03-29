import torch.cuda
import wandb
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from wisdomify.connectors import connect_to_wandb
from wisdomify.loaders import load_config
from wisdomify.constants import ROOT_DIR
from wisdomify.flows import ExperimentFlow
from termcolor import colored


def main():
    # --- prep the arguments --- #
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="rd_alpha")
    parser.add_argument("--ver", type=str, default="a")
    parser.add_argument("--gpu", dest="gpu", action='store_true', default=False)
    parser.add_argument("--upload", dest='upload', action='store_true', default=False)
    # --- trainer arguments --- #
    parser.add_argument("--log_every_n_steps", type=int, default=1)
    parser.add_argument("--num_sanity_val_steps", type=int, default=1)
    args = parser.parse_args()
    config = load_config()[args.model][args.ver]
    config.update(vars(args))
    if not config['upload']:
        print(colored("WARNING: YOU CHOSE NOT TO UPLOAD. NOTHING BUT LOGS WILL BE SAVED TO WANDB", color="red"))
    # --- set up the device to train the model with --- #
    gpus = torch.cuda.device_count() if config['gpu'] else 0
    # --- init a run  --- #
    with connect_to_wandb(job_type="train", config=config) as run:
        # --- build an experiment --- #
        flow = ExperimentFlow(run, config['model'], config['ver'])(mode="b", config=config)
        # --- instantiate the training logger --- #
        # A new W&B run will be created when training starts
        # if you have not created one manually before with wandb.init().
        logger = WandbLogger(log_model=False)

        if config['upload']:
            # --- configure callbacks  --- #
            monitor = "Validation/Top 1 Accuracy"
            checkpoint_callback = ModelCheckpoint(dirpath=flow.rd_flow.artifact_path,
                                                  filename="rd", verbose=True, monitor=monitor,
                                                  save_top_k=1, mode="max", auto_insert_metric_name=False, every_n_epochs=1)
            # --- instantiate the trainer --- #
            # -- trainer는 flow로 만들지 않는다. 계속 옵션을 바꾸고 싶을때가 많을거라서, 그냥 이대로 두는게 좋다.
            trainer = pl.Trainer(max_epochs=config['max_epochs'],
                                 # wisdomify now supports multi-gpu training! (hardware-agnostic)
                                 gpus=gpus,
                                 # stop training when the validation accuracy does not increase anymore
                                 callbacks=[checkpoint_callback],
                                 # lightning_logs will be saved under this directory
                                 default_root_dir=ROOT_DIR,
                                 # each step means each batch. Don't set this too low
                                 # https://youtu.be/3JgpG4K6HxA
                                 log_every_n_steps=config["log_every_n_steps"],
                                 num_sanity_val_steps=config["num_sanity_val_steps"],
                                 stochastic_weight_avg=config["stochastic_weight_avg"],
                                 logger=logger)
            # --- start training with validation --- #
            trainer.fit(model=flow.rd_flow.rd, datamodule=flow.datamodule)
            # --- log the best model's score --- #
            run.log({"best_model_score": checkpoint_callback.best_model_score})
            # rd.ckpt is already saved as the best checkpoint
            # so all you need to do now is logging the best score, and saving the tokenizer
            flow.rd_flow.tokenizer.save_pretrained(flow.rd_flow.tok_dir_path)
            artifact = wandb.Artifact(config['model'], type="model")
            # add the paths to the artifact
            artifact.add_file(flow.rd_flow.rd_ckpt_path)
            artifact.add_dir(flow.rd_flow.tok_dir_path, "tokenizer")
            # add the config
            artifact.metadata = config
            #  upload both the model and the tokenizer to wandb
            run.log_artifact(artifact, aliases=[config['ver'], "latest"])

        else:
            # --- instantiate the trainer --- #
            # -- trainer는 flow로 만들지 않는다. 계속 옵션을 바꾸고 싶을때가 많을거라서, 그냥 이대로 두는게 좋다.
            trainer = pl.Trainer(max_epochs=config['max_epochs'],
                                 # wisdomify now supports multi-gpu training! (hardware-agnostic)
                                 gpus=gpus,
                                 # stop training when the validation accuracy does not increase anymore
                                 # lightning_logs will be saved under this directory
                                 default_root_dir=ROOT_DIR,
                                 # each step means each batch. Don't set this too low
                                 # https://youtu.be/3JgpG4K6HxA
                                 log_every_n_steps=config["log_every_n_steps"],
                                 num_sanity_val_steps=config["num_sanity_val_steps"],
                                 stochastic_weight_avg=config["stochastic_weight_avg"],
                                 enable_checkpointing=False,
                                 logger=logger)
            trainer.fit(flow.rd_flow.rd, datamodule=flow.datamodule)


if __name__ == '__main__':
    main()
