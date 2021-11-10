import argparse
from wisdomify.connectors import connect_to_wandb
from wisdomify.flows import BuildWisdomsFlow, BuildWisdom2QueryFlow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--what", type=str,
                        default="wisdoms")
    parser.add_argument("--ver", type=str,
                        default="a")
    parser.add_argument("-val_ratio", type=float,
                        default=0.2)
    parser.add_argument("seed", type=int,
                        default=410)
    args = parser.parse_args()
    what: str = args.what
    ver: str = args.ver
    val_ratio: float = args.val_ratio
    seed: int = args.seed
    with connect_to_wandb as run:
        if what == "wisdoms":
            flow = BuildWisdomsFlow(run, ver)
        elif what == "wisdom2query":
            flow = BuildWisdom2QueryFlow(run, ver, val_ratio, seed)
        else:
            raise ValueError
        # execute the flow
        flow()
