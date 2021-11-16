from wisdomify import flows
from wisdomify.connectors import connect_to_wandb
from wisdomify.flows import Wisdom2DefFlow


def wisdom2def():
    config = {'data': 'wisdoms', 'ver': 'a', 'val_ratio': None, 'seed': 410, 'upload': False}

    with connect_to_wandb(job_type="download", config=config) as run:
        wisdom2DefFlow = Wisdom2DefFlow(run, config['ver'])
        wisdom2DefFlow.download_raw_df()

        print(wisdom2DefFlow.raw_df)


if __name__ == '__main__':
    wisdom2def()