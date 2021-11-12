import wandb
import os
from wisdomify.constants import ROOT_DIR, ARTIFACTS_DIR
import types


def main():
    run = wandb.init(dir=ROOT_DIR,  # this is for logging
                     project="wisdomify",
                     entity="wisdomify")
    artifact = run.use_artifact("wisdoms:a")
    # yeah, a bit messy, but it does what it is supposed to do
    # I might want to do a PR on this, later
    artifact._default_root = types.MethodType(lambda a, b=True:
                                              os.path.join(ARTIFACTS_DIR, a.name),
                                              artifact)
    table = artifact.get("wisdoms")
    # this is how you access wisdoms here
    print(table)


if __name__ == '__main__':
    main()
