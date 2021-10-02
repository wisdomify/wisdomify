"""
load a pre-trained wisdomify, and play with it.
"""
import argparse
import torch

from wisdomify.loaders import load_device
from wisdomify.wisdomifier import Wisdomifier


def main():
    device = load_device()
    parser = argparse.ArgumentParser()
    parser.add_argument("--ver", type=str,
                        default="0")
    parser.add_argument("--desc", type=str,
                        default="왜 하필이면 오늘이야")
    args = parser.parse_args()
    ver: str = args.ver
    desc: str = args.desc
    wisdomifier = Wisdomifier.from_pretrained(ver, device)
    print("### desc: {} ###".format(desc))
    for results in wisdomifier(sents=[desc]):
        for idx, res in enumerate(results):
            print("{}: ({}, {:.4f})".format(idx, res[0], res[1]))


if __name__ == '__main__':
    main()
