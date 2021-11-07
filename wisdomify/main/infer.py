"""
load a pre-trained wisdomify, and play with it.
"""
import argparse
import wandb
from wisdomify.loaders import load_device
from wisdomify.paths import ROOT_DIR
from wisdomify.utils import Experiment, Wisdomifier


def main():
    device = load_device()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default="rd_beta")
    parser.add_argument("--ver", type=str,
                        default="v1")
    parser.add_argument("--desc", type=str,
                        default="오전 내내 비가 안오길래 산책하러 밖을 나왔더니 갑자기 비가 쏟아지기 시작했다")
    args = parser.parse_args()
    model: str = args.model
    ver: str = args.ver
    desc: str = args.desc
    # --- init a run  --- #
    run = wandb.init(name="wisdomify.main.train",
                     tags=[f"{model}:{ver}"],
                     dir=ROOT_DIR,
                     project="wisdomify",
                     entity="wisdomify")
    # --- init a wisdomifier --- #
    exp = Experiment.load(model, ver, run, device)
    wisdomifier = Wisdomifier(exp)

    # --- inference --- #
    print("### desc: {} ###".format(desc))
    for results in wisdomifier(sents=[desc]):
        for idx, res in enumerate(results):
            print("{}: ({}, {:.4f})".format(idx, res[0], res[1]))


if __name__ == '__main__':
    main()
