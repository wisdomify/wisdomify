"""
index a pre-downloaded corpus into elasticsearch.
"""
import argparse
from wisdomify.connectors import connect_to_es
from wisdomify.elastic.indexer import Indexer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=str,
                        default="gk_story")
    parser.add_argument("--batch_size", type=int,
                        default=1000)
    args = parser.parse_args()
    index: str = args.index
    batch_size: int = args.batch_size
    # run indexing
    with connect_to_es() as es:
        indexer = Indexer(es, batch_size)
        indexer(index)


if __name__ == '__main__':
    main()
