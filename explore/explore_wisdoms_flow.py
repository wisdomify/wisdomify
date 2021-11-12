from wisdomify.connectors import connect_to_wandb
from wisdomify.flows import WisdomsFlow

if __name__ == '__main__':
    config = dict()
    run = connect_to_wandb("eplore", config=config)
    table = WisdomsFlow(run, "a")(mode="d", config=config).raw_table
    wisdoms = list(table.iterrows())
    print(table)
    print(wisdoms)
