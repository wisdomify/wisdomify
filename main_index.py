"""
index a pre-downloaded corpus into elasticsearch.
"""
from typing import Dict
from elasticsearch_dsl import Document
from metaflow import FlowSpec, step, Parameter
from storyteller.connectors import connect_to_es
from storyteller.elastic.indexer import Indexer
from storyteller.elastic.docs import (
    GK, SC, MR, BS, DS,
    SFC, KESS, KJ, KCSS,
    SFKE, KSNS, KC, KETS,
    KEPT, News, KUNIV
)


class IndexFlow(FlowSpec):
    index_name = Parameter("index_name", type=str,
                           default="mr_story")
    batch_size = Parameter("batch_size", type=int,
                           default=1000)

    name2doc: Dict[str, Document]

    @step
    def start(self):
        """
        prints out the parameters
        """
        print(self.__dict__)
        self.next(self.update)

    @step
    def update(self):
        """
        check if an index with given name already exists.
        If it does exists, delete it so that we overwrite the index in the following steps
        """
        with connect_to_es() as es:
            if es.indices.exists(index=self.index_name):
                r = es.indices.delete(index=self.index_name)
                print(f"Deleted {self.index_name} - {r}")
        self.next(self.validate)

    @step
    def validate(self):
        """
        validate index_name
        """
        self.name2doc = {
            GK.Index.name: GK,
            SC.Index.name: SC,
            MR.Index.name: MR,
            BS.Index.name: BS,
            DS.Index.name: DS,
            SFC.Index.name: SFC,
            KESS.Index.name: KESS,
            KJ.Index.name: KJ,
            KCSS.Index.name: KCSS,
            SFKE.Index.name: SFKE,
            KSNS.Index.name: KSNS,
            KC.Index.name: KC,
            KETS.Index.name: KETS,
            KEPT.Index.name: KEPT,
            News.Index.name: News,
            KUNIV.Index.name: KUNIV,
        }
        if self.index_name not in self.name2doc.keys():
            raise ValueError(f"Invalid index: {self.index_name}")
        self.next(self.end)

    @step
    def end(self):
        """
        Index Stories.
        """
        self.index_name: str
        self.batch_size: int
        stories = self.name2doc[self.index_name].stream_from_corpus()
        with connect_to_es() as es:
            self.name2doc[self.index_name].init(using=es)
            # --- init an indexer with the Stories --- #
            indexer = Indexer(es, stories, self.index_name, self.batch_size)
            # --- index the corpus --- #
            indexer()
        print(f"indexing {self.index_name} finished")


if __name__ == '__main__':
    IndexFlow()
