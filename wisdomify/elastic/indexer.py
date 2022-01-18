from typing import List
from tqdm import tqdm
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from elasticsearch_dsl import Document
from more_itertools import chunked
from wisdomify.elastic.docs import GK, SC, MR, BS, DS, SFC, KESS, KJ, KCSS, SFKE, KSNS, KC, KETS, KEPT, News, KUNIV


class Indexer:

    def __init__(self, es: Elasticsearch, batch_size: int):
        self.es = es
        self.batch_size = batch_size

    def __call__(self, index: str):
        if self.es.indices.exists(index=index):
            r = self.es.indices.delete(index=index)
            print(f"Deleted {index} - {r}")

        name2story = {
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
                KUNIV.Index.name: KUNIV
        }
        try:
            stories = name2story[index].stream_from_corpus()
        except KeyError:
            raise KeyError(f"Invalid index: {index}")

        for batch in tqdm(chunked(stories, self.batch_size),
                          desc=f"indexing {index}..."):
            batch: List[Document]  # a batch is a list of Document
            # must make sure include_meta is set to true, otherwise the helper won't be
            # aware of the name of the index that= we are indexing the corpus into
            actions = (doc.to_dict(include_meta=True) for doc in batch)
            r = bulk(self.es, actions)
            print(f"successful count: {r[0]}, error messages: {r[1]}")

