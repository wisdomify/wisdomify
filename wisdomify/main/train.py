import pytorch_lightning as pl
import torch
import argparse
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import AutoModelForMaskedLM, AutoTokenizer
from wisdomify.loaders import load_conf
from wisdomify.models import RD
from wisdomify.builders import build_vocab2subwords
from wisdomify.paths import DATA_DIR
from wisdomify.vocab import VOCAB
from wisdomify.datasets import WisdomDataModule


def main():
    # --- setup the device --- #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- prep the arguments --- #
    parser = argparse.ArgumentParser()
    parser.add_argument("--ver", type=str,
                        default="1")

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
    data_name: str = selected_ver['data_name'][0]
    data_version: str = selected_ver['data_version']
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
        monitor='train_loss',
        filename=model_name
    )
    # --- instantiate the logger --- #
    logger = TensorBoardLogger(save_dir=DATA_DIR,
                               name="lightning_logs")

    # --- instantiate the trainer --- #
    trainer = pl.Trainer(gpus=torch.cuda.device_count(),
                         max_epochs=max_epochs,
                         callbacks=[checkpoint_callback],
                         default_root_dir=DATA_DIR,
                         logger=logger)
    # --- start training --- #
    # data_module.prepare_data()
    # data_module.setup(stage='fit')

    trainer.fit(model=rd, datamodule=data_module)

    # TODO: validate every epoch and test model after training
    '''
    trainer.validate(model=rd,
                     valid_loader=valid_loader)

    trainer.test(model=rd,
                 test_loader=test_loader)
    '''


if __name__ == '__main__':
    main()
