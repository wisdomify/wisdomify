import pytorch_lightning as pl
import torch
import argparse
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from wisdomify.loaders import load_device
from wisdomify.wisdomifier import Experiment
from wisdomify.paths import DATA_DIR, WISDOMIFIER_TOKENIZER_DIR
from wisdomify.utils import WandBSupport


def main():
    # --- setup the device --- #
    device = load_device()

    # --- prep the arguments --- #
    parser = argparse.ArgumentParser()
    parser.add_argument("--ver", type=str,
                        default="0")
    args = parser.parse_args()
    ver: str = args.ver

    # --- initialise WandB object --- #
    wandb_support = WandBSupport(job_type=job_name, notes=desc)

    # --- build an experiment instance --- #
    exp = Experiment.build(ver, device)
    model_name = "wisdomifier"

    # --- init callbacks --- #
    checkpoint_callback = ModelCheckpoint(
        filename=model_name,
        verbose=True
    )

    # --- instantiate the training logger --- #
    wandb_logger = wandb_support.get_model_logger('training_log')

    # --- instantiate the trainer --- #
    trainer = pl.Trainer(gpus=torch.cuda.device_count(),
                         max_epochs=exp.config['max_epochs'],
                         callbacks=[checkpoint_callback],
                         default_root_dir=DATA_DIR,
                         logger=wandb_logger)

    # --- start training --- #
    trainer.fit(model=exp.rd, datamodule=exp.data_module)

    # --- save the tokenizer --- #
    exp.tokenizer.save_pretrained(WISDOMIFIER_TOKENIZER_DIR.format(ver=ver))

    # --- saving and logging model --- #
    wandb_support.models.push_mlm(model=rd.bert_mlm,
                                  name=out_mlm_name,
                                  desc=out_mlm_desc)

    wandb_support.models.push_tokenizer(model=tokenizer,
                                        name=out_tokenizer_name,
                                        desc=out_tokenizer_desc)

    wandb_support.push()


if __name__ == '__main__':
    main()
