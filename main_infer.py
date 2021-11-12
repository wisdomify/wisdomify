"""
load a pre-trained wisdomify, and play with it.
"""
import argparse
from wisdomify.connectors import connect_to_wandb
from wisdomify.loaders import load_device, load_config
from wisdomify import flows
from wisdomify.wisdomifier import Wisdomifier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default="rd_alpha")
    parser.add_argument("--ver", type=str,
                        default="a")
    parser.add_argument("--desc", type=str,
                        default="오전 내내 비가 안오길래 산책하러 밖을 나왔더니 갑자기 비가 쏟아지기 시작했다")
    parser.add_argument("--use_gpu", dest="use_gpu", action="store_true", default=False)
    args = parser.parse_args()
    model: str = args.model
    ver: str = args.ver
    desc: str = args.desc
    use_gpu: bool = args.use_gpu
    device = load_device(use_gpu)
    config = load_config()[model][ver]
    with connect_to_wandb(job_type="infer", config=config) as run:
        # --- init a wisdomifier --- #
        flow = flows.ExperimentFlow(run, model, ver, device)("d", config)
    # --- wisdomifier is independent of wandb run  --- #
    wisdomifier = Wisdomifier(flow.rd_flow.rd, flow.datamodule)
    # --- inference --- #
    print("### desc: {} ###".format(desc))
    results = wisdomifier(sents=[desc])
    for result in results:
        for idx, entry in enumerate(result):
            print("{}: ({}, {:.4f})".format(idx, entry[0], entry[1]))


if __name__ == '__main__':
    main()
