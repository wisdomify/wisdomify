import argparse
from wisdomify.connectors import connect_to_wandb
from wisdomify.flows import (
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
    parser.add_argument("--upload", dest='upload', action='store_true',
                        default=False)  # set this flag up if you want to save the logs & the model
    args = parser.parse_args()
    what: str = args.what
    ver: str = args.ver
    val_ratio: float = args.val_ratio
    seed: int = args.seed
    upload: bool = args.upload
    # flows, flows, oh I love flows
    with connect_to_wandb() as run:
        if what == "wisdoms":
            flow = WisdomsFlow(run, ver, "build")
        elif what == "wisdom2query":
            flow = Wisdom2QueryFlow(run, ver, "upload", val_ratio, seed)
        elif what == "wisdom2def":
            flow = Wisdom2DefFlow(run, ver, "upload")
        elif what == "wisdom2eg":
            flow = Wisdom2EgFlow(run, ver, "upload")
        else:
            raise ValueError
        if upload:
            run.log_artifact(flow.artifact, aliases=[ver, "latest"])


if __name__ == '__main__':
    main()
