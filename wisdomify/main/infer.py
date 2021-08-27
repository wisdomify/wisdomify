"""
load a pre-trained wisdomify, and play with it.
"""
import argparse
import torch
from wisdomify.loaders import load_conf
from wisdomify.models import Wisdomifier


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("--ver", type=str,
                        default="3")
    parser.add_argument("--desc", type=str,
                        default="소문이랑 다르네")
    args = parser.parse_args()
    ver: str = args.ver
    desc: str = args.desc

    conf = load_conf()
    wisdomifier = Wisdomifier.from_pretrained(ver, device)

    print("### desc: {} ###".format(desc))
    for results in wisdomifier.wisdomify(sents=[desc]):
        for idx, res in enumerate(results):
            print("{}: ({}, {:.4f})".format(idx, res[0], res[1]))


if __name__ == '__main__':
    main()

