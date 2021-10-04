import pytorch_lightning as pl
import torch
import argparse
from pytorch_lightning.callbacks import ModelCheckpoint

from wisdomify.loaders import load_device
from wisdomify.wisdomifier import Experiment
from wisdomify.paths import DATA_DIR
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

    # --- W&B support object init --- #
    wandb_support = WandBSupport(ver)

    # --- build an experiment instance --- #
    exp = Experiment.build(ver, device, wandb_support)
    model_name = "wisdomifier"

    # --- init callbacks --- #
    checkpoint_callback = ModelCheckpoint(
        filename=model_name,
        save_last=True,
        verbose=True
    )

    # --- instantiate the training logger --- #
    logger = wandb_support.get_model_logger('training_log')

    # --- instantiate the trainer --- #
    trainer = pl.Trainer(gpus=torch.cuda.device_count(),
                         max_epochs=exp.config['model']['max_epochs'],
                         callbacks=[checkpoint_callback],
                         default_root_dir=DATA_DIR,
                         logger=logger)

    # --- start training --- #
    trainer.fit(model=exp.rd, datamodule=exp.data_module)

    # --- saving model --- #
    wandb_support.models.push_mlm(exp.rd.bert_mlm)      # TODO: 유빈님 이거 저장하는 거 맞아요?
    wandb_support.models.push_tokenizer(exp.tokenizer)  # TODO: tokenizer는 이게 맞는 거 같은데
    wandb_support.models.push_rd_ckpt()

    # --- uploading wandb logs and other files --- #
    wandb_support.push()


if __name__ == '__main__':
    main()
