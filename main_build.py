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
    parser.add_argument("--data", type=str,
                        default="wisdoms")
    parser.add_argument("--ver", type=str,
                        default="a")
    parser.add_argument("--val_ratio", type=float,
                        default=None)
    parser.add_argument("--seed", type=int,
                        default=None)
    parser.add_argument("--upload", dest='upload', action='store_true',
                        default=False)  # set this flag up if you want to save the logs & the model
    args = parser.parse_args()
    data: str = args.data
    ver: str = args.ver
    val_ratio: float = args.val_ratio
    seed: int = args.seed
    upload: bool = args.upload
    if not upload:
        print("########## WARNING: YOU CHOSE NOT TO UPLOAD. NOTHING BUT LOGS WILL BE SAVED TO WANDB #################")
    # flows, flows, oh I love flows
    with connect_to_wandb(job_type="build", config=vars(args)) as run:
        if data == "wisdoms":
            flow = WisdomsFlow(run, ver)(mode="b")
        elif data == "wisdom2query":
            flow = Wisdom2QueryFlow(run, ver)(mode="b", val_ratio=val_ratio, seed=seed)
        elif data == "wisdom2def":
            flow = Wisdom2DefFlow(run, ver)(mode="b")
        elif data == "wisdom2eg":
            flow = Wisdom2EgFlow(run, ver)(mode="b")
        else:
            raise ValueError
        if upload:
            run.log_artifact(flow.artifact, aliases=[ver, "latest"])


if __name__ == '__main__':
    main()
