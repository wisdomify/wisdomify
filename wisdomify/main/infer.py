"""
load a pre-trained wisdomify, and play with it.
"""
import argparse
from wisdomify.loaders import load_device
from wisdomify.utils import WandBSupport
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

    # --- W&B support object init --- #
    wandb_support = WandBSupport(ver=ver, run_type='infer')

    wisdomifier = Wisdomifier.from_pretrained(ver, device, wandb_support)

    print("### desc: {} ###".format(desc))
    for results in wisdomifier(sents=[desc]):
        for idx, res in enumerate(results):
            print("{}: ({}, {:.4f})".format(idx, res[0], res[1]))

    wandb_support.push()


if __name__ == '__main__':
    main()
