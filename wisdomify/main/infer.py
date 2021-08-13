"""
load a pre-trained wisdomify, and play with it.
"""
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
from wisdomify.loaders import load_conf
from wisdomify.models import RD, Wisdomifier


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
    # a one-liner for loading a pre-trained wisdomifier
    wisdomifier = Wisdomifier.from_pretrained(ver, device)

    print("### desc: {} ###".format(desc))
    for results in wisdomifier.wisdomify(sents=[desc]):
        for idx, res in enumerate(results):
            print("{}: ({}, {:.4f})".format(idx, res[0], res[1]))


if __name__ == '__main__':
    main()
