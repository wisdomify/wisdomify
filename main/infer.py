"""
load a pre-trained wisdomify, and play with it.
"""
import argparse
from wisdomify.connectors import connect_to_wandb
from wisdomify.loaders import load_device
from wisdomify import flows


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
    with connect_to_wandb() as run:
        # --- init a wisdomifier --- #
        flow = flows.WisdomifyFlow(run, model, ver, [desc], device)
        # --- inference --- #
        print("### desc: {} ###".format(desc))
        for result in flow.results:
            for idx, entry in enumerate(result):
                print("{}: ({}, {:.4f})".format(idx, entry[0], entry[1]))


if __name__ == '__main__':
    main()
