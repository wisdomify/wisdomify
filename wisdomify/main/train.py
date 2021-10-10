import pytorch_lightning as pl
import torch
import argparse
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from wisdomify.loaders import load_device
from wisdomify.wisdomifier import Experiment
from wisdomify.paths import DATA_DIR, WISDOMIFIER_TOKENIZER_DIR, WISDOMIFIER_CKPT


def main():
    # --- setup the device --- #
    device = load_device()

    # --- prep the arguments --- #
    parser = argparse.ArgumentParser()
    parser.add_argument("--ver", type=str, default="2")
    args = parser.parse_args()
    ver: str = args.ver

    # --- build an experiment instance --- #
    exp = Experiment.build(ver, device)
    model_name = "wisdomifier"

    # --- init callbacks --- #
    checkpoint_callback = ModelCheckpoint(
        filename=model_name,
        verbose=True
    )
    
    # --- instantiate the logger --- #
    logger = TensorBoardLogger(save_dir=DATA_DIR,
                               name="lightning_logs")

    # --- instantiate the trainer --- #
    trainer = pl.Trainer(gpus=torch.cuda.device_count(),
                         max_epochs=exp.config['max_epochs'],
                         callbacks=[checkpoint_callback],
                         default_root_dir=DATA_DIR,
                         logger=logger
                        )

    # --- start training --- #
    trainer.fit(model=exp.rd, datamodule=exp.datamodule)
    trainer.save_checkpoint(WISDOMIFIER_CKPT.format(ver=ver))

    # --- save the tokenizer --- #
    exp.tokenizer.save_pretrained(WISDOMIFIER_TOKENIZER_DIR.format(ver=ver))


if __name__ == '__main__':
    main()
