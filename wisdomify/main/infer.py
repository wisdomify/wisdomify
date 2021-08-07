"""
load a pre-trained wisdomify, and play with it.
"""
import argparse
import torch
import yaml
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
from wisdomify.loaders import load_conf
from wisdomify.models import RD, Wisdomifier
from wisdomify.paths import WISDOMIFIER_CKPT, WISDOMIFIER_HPARAMS_YAML
from wisdomify.vocab import VOCAB
from wisdomify.builders import build_vocab2subwords


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("--ver", type=str,
                        default="0")
    parser.add_argument("--desc", type=str,
                        default="왜 하필이면 오늘이야")
    args = parser.parse_args()
    ver: str = args.ver
    desc: str = args.desc

    conf = load_conf()
    vers = conf['versions']
    if ver not in vers.keys():
        raise NotImplementedError("Invalid version provided".format(ver))

    selected_ver = vers[ver]
    bert_model: str = selected_ver['bert_model']

    wisdomifier_path = WISDOMIFIER_CKPT
    with open(WISDOMIFIER_HPARAMS_YAML, 'r') as fh:
        wisdomifier_hparams = yaml.safe_load(fh)
    k = wisdomifier_hparams['k']

    bert_mlm = AutoModelForMaskedLM.from_config(AutoConfig.from_pretrained(bert_model))
    tokenizer = AutoTokenizer.from_pretrained(bert_model)
    vocab2subwords = build_vocab2subwords(tokenizer, k, VOCAB).to(device)
    rd = RD.load_from_checkpoint(wisdomifier_path, bert_mlm=bert_mlm, vocab2subwords=vocab2subwords)
    rd.eval()  # otherwise, the model will output different results with the same inputs
    rd = rd.to(device)  # otherwise, you can not run the inference process on GPUs.
    wisdomifier = Wisdomifier(rd, tokenizer)

    print("### desc: {} ###".format(desc))
    for results in wisdomifier.wisdomify(sents=[desc]):
        for idx, res in enumerate(results):
            print("{}: ({}, {:.4f})".format(idx, res[0], res[1]))


if __name__ == '__main__':
    main()
