import pytorch_lightning as pl
import torch
import argparse
import yaml
from transformers import AutoModelForMaskedLM, AutoConfig, AutoTokenizer
from wisdomify.builders import build_vocab2subwords
from wisdomify.datasets import WisdomDataModule
from wisdomify.loaders import load_conf
from wisdomify.models import RD
from wisdomify.paths import WISDOMIFIER_V_0_CKPT, WISDOMIFIER_V_0_HPARAMS_YAML
from wisdomify.vocab import VOCAB


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("--ver", type=str,
                        default="0")

    args = parser.parse_args()
    ver: str = args.ver

    conf = load_conf()
    vers = conf['versions']
    if ver not in vers.keys():
        raise NotImplementedError("Invalid version provided".format(ver))

    selected_ver = vers[ver]
    bert_model: str = selected_ver['bert_model']
    data_version: str = selected_ver['data_version']
    # TODO: should enable to load both example and definition on one dataset
    data_name: str = selected_ver['data_name'][0]
    batch_size: int = selected_ver['batch_size']
    repeat: bool = selected_ver['repeat']
    num_workers: int = selected_ver['num_workers']
    train_ratio: float = selected_ver['train_ratio']
    test_ratio: float = selected_ver['test_ratio']

    wisdomifier_path = WISDOMIFIER_V_0_CKPT.format(ver=ver)
    with open(WISDOMIFIER_V_0_HPARAMS_YAML.format(ver=ver), 'r') as fh:
        wisdomifier_hparams = yaml.safe_load(fh)
    k = wisdomifier_hparams['k']

    bert_mlm = AutoModelForMaskedLM.from_config(AutoConfig.from_pretrained(bert_model))
    tokenizer = AutoTokenizer.from_pretrained(bert_model)
    vocab2subwords = build_vocab2subwords(tokenizer, k, VOCAB).to(device)
    rd = RD.load_from_checkpoint(wisdomifier_path, bert_mlm=bert_mlm, vocab2subwords=vocab2subwords)
    rd.eval()  # otherwise, the model will output different results with the same inputs
    rd = rd.to(device)  # otherwise, you can not run the inference process on GPUs.

    data_module = WisdomDataModule(data_version=data_version,
                                   data_name=data_name,
                                   k=k,
                                   device=device,
                                   vocab=VOCAB,
                                   tokenizer=tokenizer,
                                   batch_size=batch_size,
                                   num_workers=num_workers,
                                   train_ratio=train_ratio,
                                   test_ratio=test_ratio,
                                   shuffle=False,
                                   repeat=repeat)

    trainer = pl.Trainer(gpus=torch.cuda.device_count())

    trainer.test(model=rd, datamodule=data_module, verbose=False)


if __name__ == '__main__':
    main()
