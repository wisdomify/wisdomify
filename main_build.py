import argparse
from wisdomify.connectors import connect_to_wandb
from wisdomify import flows


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
                        default=410)
    parser.add_argument("--upload", dest='upload', action='store_true',
                        default=False)  # set this flag up if you want to save the logs & the model
    args = parser.parse_args()
    data: str = args.data
    ver: str = args.ver
    val_ratio: float = args.val_ratio
    seed: int = args.seed
    upload: bool = args.upload
    config = vars(args)
    if not upload:
        print("########## WARNING: YOU CHOSE NOT TO UPLOAD. NOTHING BUT LOGS WILL BE SAVED TO WANDB #################")
    # flows, flows, oh I love flows
    with connect_to_wandb(job_type="build", config=config) as run:
        if data == "wisdoms":
            flow = flows.WisdomsFlow(run, ver)(mode="b", config=config)
        elif data == "wisdom2query":
            flow = flows.Wisdom2QueryFlow(run, ver)(mode="b", config=config)
        elif data == "wisdom2def":
            flow = flows.Wisdom2DefFlow(run, ver)(mode="b", config=config)
        elif data == "wisdom2eg":
            flow = flows.Wisdom2EgFlow(run, ver)(mode="b", config=config)
        else:
            raise ValueError
        if upload:
            run.log_artifact(flow.artifact, aliases=[ver, "latest"])


if __name__ == '__main__':
    main()
