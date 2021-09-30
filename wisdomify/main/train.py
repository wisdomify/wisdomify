import pytorch_lightning as pl
import torch
import argparse

import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import AutoModelForMaskedLM, AutoTokenizer
from wisdomify.loaders import load_conf
from wisdomify.models import RD
from wisdomify.builders import build_vocab2subwords
from wisdomify.paths import DATA_DIR
from wisdomify.utils import TrainerFileSupport
from wisdomify.vocab import VOCAB
from wisdomify.datasets import WisdomDataModule
from pytorch_lightning.loggers import WandbLogger  # newline 1
from pytorch_lightning import Trainer

import os


def main():
    # --- setup the device --- #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # --- prep the arguments --- #
    parser = argparse.ArgumentParser()
    parser.add_argument("--ver", type=str,
                        default="2")

    args = parser.parse_args()
    ver: str = args.ver
    # parameters from conf

    conf = load_conf()
    vers = conf['versions']
    if ver not in vers.keys():
        raise NotImplementedError(f"Cannot find version {ver}.\nWrite your setting and version properly on conf.json")

    selected_ver = vers[ver]
    bert_model: str = selected_ver['bert_model']
    k: int = selected_ver['k']
    lr: float = selected_ver['lr']
    max_epochs: int = selected_ver['max_epochs']
    batch_size: int = selected_ver['batch_size']
    repeat: int = selected_ver['repeat']
    shuffle: bool = selected_ver['shuffle']
    num_workers: int = selected_ver['num_workers']
    # TODO: should enable to load both example and definition on one dataset
    data_name: str = selected_ver['data_name']
    data_version: str = selected_ver['data_version']
    dtype: str = selected_ver['dtype'][0]
    model_name = "wisdomifier"

    # --- instantiate the model --- #
    kcbert_mlm = AutoModelForMaskedLM.from_pretrained(bert_model)
    tokenizer = AutoTokenizer.from_pretrained(bert_model)


    vocab2subwords = build_vocab2subwords(tokenizer, k, VOCAB).to(device)
    rd = RD(kcbert_mlm, vocab2subwords, k, lr)  # mono rd
    rd.to(device)
    # --- setup a dataloader --- #
    data_module = WisdomDataModule(data_version=data_version,
                                   data_name=data_name,
                                   dtype=dtype,
                                   k=k,
                                   device=device,
                                   vocab=VOCAB,
                                   tokenizer=tokenizer,
                                   batch_size=batch_size,
                                   num_workers=num_workers,
                                   shuffle=shuffle,
                                   repeat=repeat)

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
    wandb_logger = WandbLogger(project='wisdomify', entity='wisdomify', name='training_log')

    trainer = pl.Trainer(gpus=torch.cuda.device_count(),
                         max_epochs=max_epochs,
                         callbacks=[checkpoint_callback],
                         default_root_dir=DATA_DIR,
                         logger=wandb_logger)

    # --- start training --- #
    trainer.fit(model=rd, datamodule=data_module)

    tokenizer_path = os.path.join(DATA_DIR, f'lightning_logs/version_{ver}/checkpoints/')
    tokenizer.save_pretrained(tokenizer_path)

    wandb.finish()
    # trainerFileSupporter.save_training_result()

    # TODO: validate every epoch and test model after training
    '''
    trainer.validate(model=rd,
                     valid_loader=valid_loader)

    trainer.test(model=rd,
                 test_loader=test_loader)
    '''


if __name__ == '__main__':
    main()
