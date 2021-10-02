import torch
import argparse

from pytorch_lightning.callbacks import ModelCheckpoint
from wisdomify.loaders import load_conf
from wisdomify.models import RD
from wisdomify.builders import build_vocab2subwords
from wisdomify.paths import DATA_DIR
from wisdomify.utils import WandBSupport
from wisdomify.vocab import VOCAB
from wisdomify.datasets import WisdomDataModule
from pytorch_lightning import Trainer


def main():
    # --- setup the device --- #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # --- prep the arguments --- #
    parser = argparse.ArgumentParser()
    parser.add_argument("--ver", type=str, default="2")
    parser.add_argument("--wandb_name", type=str, required=True)

    args = parser.parse_args()
    ver: str = args.ver
    job_name: str = args.wandb_name

    # parameters from conf
    conf = load_conf()
    vers = conf['versions']
    if ver not in vers.keys():
        raise NotImplementedError(f"Cannot find version {ver}.\nWrite your setting and version properly on conf.json")

    selected_ver = vers[ver]

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

    desc: str = selected_ver['desc']

    in_mlm_name: str = selected_ver['model']['load']['mlm_name']
    in_mlm_ver: str = selected_ver['model']['load']['mlm_ver']
    in_tokenizer_name: str = selected_ver['model']['load']['tokenizer_name']
    in_tokenizer_ver: str = selected_ver['model']['load']['tokenizer_ver']

    out_mlm_name: str = selected_ver['model']['save']['mlm_name']
    out_mlm_desc: str = selected_ver['model']['save']['mlm_desc']
    out_tokenizer_name: str = selected_ver['model']['save']['tokenizer_name']
    out_tokenizer_desc: str = selected_ver['model']['save']['tokenizer_desc']

    # --- initialise WandB object --- #
    wandb_support = WandBSupport(job_type=job_name, notes=desc)

    # --- instantiate the model --- #
    kcbert_mlm = wandb_support.models.get_mlm(name=in_mlm_name, ver=in_mlm_ver)
    tokenizer = wandb_support.models.get_tokenizer(name=in_tokenizer_name, ver=in_tokenizer_ver)

    vocab2subwords = build_vocab2subwords(tokenizer, k, VOCAB).to(device)
    rd = RD(kcbert_mlm, vocab2subwords, k, lr)  # mono rd
    rd.to(device)

    # --- setup a dataloader --- #
    data_module = WisdomDataModule(wandb_support=wandb_support,
                                   data_version=data_version,
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

    # --- instantiate the training logger --- #
    wandb_logger = wandb_support.get_model_logger('training_log')

    # --- instantiate the trainer --- #
    trainer = Trainer(gpus=torch.cuda.device_count(),
                      max_epochs=max_epochs,
                      callbacks=[checkpoint_callback],
                      default_root_dir=DATA_DIR,
                      logger=wandb_logger)

    # --- start training --- #
    trainer.fit(model=rd, datamodule=data_module)

    # --- saving and logging model --- #
    wandb_support.models.push_mlm(model=rd.bert_mlm,
                                  name=out_mlm_name,
                                  desc=out_mlm_desc)

    wandb_support.models.push_tokenizer(model=tokenizer,
                                        name=out_tokenizer_name,
                                        desc=out_tokenizer_desc)

    wandb_support.push()

    # TODO: validate every epoch and test model after training
    '''
    trainer.validate(model=rd,
                     valid_loader=valid_loader)

    trainer.test(model=rd,
                 test_loader=test_loader)
    '''


if __name__ == '__main__':
    main()
