from wisdomify.connectors import connect_to_wandb
from wisdomify.flows import WisdomsFlow

if __name__ == '__main__':
    run = connect_to_wandb()
    table = WisdomsFlow(run, "a", "download").raw_table
    wisdoms = list(table.iterrows())
    print(table)
    print(wisdoms)
