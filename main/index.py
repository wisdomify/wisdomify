"""
index a pre-downloaded corpus into elasticsearch.
"""
import argparse
from wisdomify.connectors import connect_to_es
from wisdomify.flows_elastic import IndexFlow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_name", type=str,
                        default="gk_story")
    parser.add_argument("--batch_size", type=int,
                        default=1000)
    args = parser.parse_args()
    index_name: str = args.index_name
    batch_size: int = args.batch_size

    # run indexing
    with connect_to_es() as client:
        IndexFlow(client, index_name, batch_size)


if __name__ == '__main__':
    main()
