"""
load a pre-trained wisdomify, and play with it.
"""
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from wisdomify.loaders import load_conf
from wisdomify.models import RD, Wisdomifier
from wisdomify.paths import WISDOMIFIER_A_CKPT
from wisdomify.vocab import VOCAB
from wisdomify.builders import build_vocab2subwords


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("--ver", type=str,
                        default="a")
    parser.add_argument("--desc", type=str,
                        default="왜 하필이면 오늘이야")
    args = parser.parse_args()
    ver: str = args.ver
    desc: str = args.desc
    conf = load_conf()
    bert_model: str = conf['bert_model']

    if ver == "a":
        wisdomifier_path = WISDOMIFIER_A_CKPT
        k = conf['exps']['a']['k']
    else:
        raise NotImplementedError

    bert_mlm = AutoModelForMaskedLM.from_pretrained(bert_model)
    tokenizer = AutoTokenizer.from_pretrained(bert_model)
    vocab2subwords = build_vocab2subwords(tokenizer, k, VOCAB)  # this is something I don't really like...
    rd = RD.load_from_checkpoint(wisdomifier_path, bert_mlm=bert_mlm, vocab2subwords=vocab2subwords)
    rd.eval()  # this is necessary
    rd = rd.to(device)
    wisdomifier = Wisdomifier(rd, tokenizer)

    print("### desc: {} ###".format(desc))
    for results in wisdomifier.wisdomify(sents=[desc]):
        for idx, res in enumerate(results):
            print("{}:".format(idx), res)


if __name__ == '__main__':
    main()



