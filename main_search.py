"""
searches for a wisdom on elasticsearch
"""
import argparse
from wisdomify.connectors import connect_to_es
from wisdomify.elastic.searcher import Searcher
from wisdomify.elastic.docs import Story


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wisdom", type=str,
                        default="꿩 대신 닭")
    parser.add_argument("--index", type=str,
                        default=",".join(name for name in Story.all_names()))
    parser.add_argument("--size", type=int,
                        default="10000")
    args = parser.parse_args()
    wisdom: str = args.wisdom
    index: str = args.index
    size: int = args.size
    with connect_to_es() as es:
        searcher = Searcher(es, index, size)
        res = searcher(wisdom)
    print(f"total hits:{res['hits']['total']}")
    parsed = [
        f"index: {hit['_index']}, highlight:{hit['highlight']['sents'][0]}"
        for hit in res['hits']['hits']
    ]
    for entry in parsed:
        print(entry)


if __name__ == '__main__':
    main()
