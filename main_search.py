"""
searches for
"""
from storyteller.connectors import connect_to_es
from storyteller.elastic.searcher import Searcher
from storyteller.elastic.docs import Story
from metaflow import FlowSpec, step, Parameter
from pprint import pprint


class SearchFlow(FlowSpec):
    wisdom = Parameter("wisdom", type=str,
                       default="가는 날이 장날")
    index_name = Parameter("index_name", type=str,
                           default=",".join(Story.all_indices()))
    size = Parameter("size", type=int,
                     default=10000)

    res: dict
    @step
    def start(self):
        print(self.__dict__)
        self.next(self.search)

    @step
    def search(self):
        self.wisdom: str
        self.index_name: str
        self.size: int
        with connect_to_es() as es:
            searcher = Searcher(es)
            self.res = searcher(self.wisdom, self.index_name, self.size)
        self.next(self.end)

    @step
    def end(self):
        print(f"total hits:{self.res['hits']['total']}")
        parsed = [
            f"index: {hit['_index']}, highlight:{hit['highlight']['sents'][0]}"
            for hit in self.res['hits']['hits']
        ]
        pprint(parsed)


if __name__ == '__main__':
    SearchFlow()
