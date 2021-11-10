import argparse
from wisdomify.connectors import connect_to_wandb
from wisdomify.flows_datasets import (
    WisdomsFlow,
    Wisdom2QueryFlow,
    Wisdom2DefFlow,
    Wisdom2EgFlow
)


def main():
    # --- so that we don't get "module level import should be on top" error --- #
    parser = argparse.ArgumentParser()
    # build what?
    parser.add_argument("--what", type=str,
                        default="wisdoms")
    parser.add_argument("--ver", type=str,
                        default="a")
    parser.add_argument("--val_ratio", type=float,
                        default=0.2)
    parser.add_argument("--seed", type=int,
                        default=410)
    args = parser.parse_args()
    what: str = args.what
    ver: str = args.ver
    val_ratio: float = args.val_ratio
    seed: int = args.seed
    with connect_to_wandb() as run:
        if what == "wisdoms":
            WisdomsFlow(run, ver, "upload")
        elif what == "wisdom2query":
            Wisdom2QueryFlow(run, ver, "upload", val_ratio, seed)
        elif what == "wisdom2def":
            Wisdom2DefFlow(run, ver, "upload")
        elif what == "wisdom2eg":
            Wisdom2EgFlow(run, ver, "upload")
        else:
            raise ValueError


if __name__ == '__main__':
    main()
