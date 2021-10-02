import pytorch_lightning as pl
import torch
import argparse
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from wisdomify.loaders import load_device
from wisdomify.models import Experiment
from wisdomify.paths import DATA_DIR
from wisdomify.utils import TrainerFileSupport


def main():
    # --- setup the device --- #
    device = load_device()

    # --- prep the arguments --- #
    parser = argparse.ArgumentParser()
    parser.add_argument("--ver", type=str,
                        default="0")
    args = parser.parse_args()
    ver: str = args.ver

    # --- load an experiment instance --- #
    exp = Experiment.build(ver, device)
    model_name = "wisdomifier"

    # --- init callbacks --- #
    checkpoint_callback = ModelCheckpoint(
        filename=model_name,
        verbose=True,
    )
    # --- instantiate the logger --- #
    logger = TensorBoardLogger(save_dir=DATA_DIR,
                               name="lightning_logs")

    # --- version check logic --- #
    trainerFileSupporter = TrainerFileSupport(version=ver,
                                              logger=logger,
                                              data_dir=DATA_DIR)

    # --- instantiate the trainer --- #
    trainer = pl.Trainer(gpus=torch.cuda.device_count(),
                         max_epochs=exp.config['max_epochs'],
                         callbacks=[checkpoint_callback],
                         default_root_dir=DATA_DIR,
                         logger=logger)

    # --- start training --- #
    trainer.fit(model=exp.rd, datamodule=exp.data_module)
    trainerFileSupporter.save_training_result()


if __name__ == '__main__':
    main()
