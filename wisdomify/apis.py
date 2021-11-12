from typing import List, Callable, Optional, Dict, Generator

from more_itertools import chunked
from termcolor import colored
from tqdm import tqdm
from flask import jsonify, request
from flask_classful import FlaskView, route
from elasticsearch_dsl import Document
from elasticsearch.helpers import bulk

from elasticsearch import Elasticsearch
from wisdomify.constants import (
    ES_USERNAME,
    ES_PASSWORD,
    ES_CLOUD_ID,
)

# === elastic flows === #
from wisdomify.docs import (
    Story, GK, SC, MR, BS, DS,
    SFC, KESS, KJ, KCSS,
    SFKE, KSNS, KC, KETS,
    KEPT, News, KUNIV
)  # noqa


# ==== the superclass of all flows ==== #
class Flow:
    def __call__(self, *args, **kwargs):
        for step in self.steps():
            step()
            print(f"{type(self).__name__}:{colored(step.__name__, color='cyan')}")

    def steps(self) -> List[Callable]:
        raise NotImplementedError

    def __str__(self) -> str:
        """
        you might want to do some formatting later...
        """
        return "\n".join([step.__name__ for step in self.steps()])


class SearchFlow(Flow):

    def __init__(self, es: Elasticsearch, index_name: str, size: int):
        self.es = es
        self.index_name = index_name
        self.size = size
        # to be built
        self.query: Optional[dict] = None
        self.highlight: Optional[dict] = None
        self.res: Optional[dict] = None

    def steps(self):
        return [
            self.build,
            self.search
        ]

    def __call__(self, wisdom: str):
        self.wisdom = wisdom
        super(SearchFlow, self).__call__()
        return self

    def build(self):
        self.query = {
            'match_phrase': {
                'sents': {
                    'query': self.wisdom
                }
            }
        }
        self.highlight = {
            'fields': {
                'sents': {
                    'type': 'plain',
                    'fragment_size': 100,
                    'number_of_fragments': 2,
                    'fragmenter': 'span'
                }
            }
        }

    def search(self):
        self.res = self.es.search(index=self.index_name,
                                  query=self.query,
                                  highlight=self.highlight,
                                  size=self.size)


class IndexFlow(Flow):

    def __init__(self, es: Elasticsearch, index_name: str, batch_size: int):
        self.es = es
        self.index_name = index_name
        self.batch_size = batch_size
        self.name2story: Optional[Dict[str, Story]] = None
        self.stories: Optional[Generator[Story, None, None]] = None

    def steps(self) -> List[Callable]:
        # skip the steps
        return [
            self.update,
            self.validate,
            self.index
        ]

    def update(self):
        """
        check if an index with given name already exists.
        If it does exists, delete it so that we overwrite the index in the following steps
        """
        if self.es.indices.exists(index=self.index_name):
            r = self.es.indices.delete(index=self.index_name)
            print(f"Deleted {self.index_name} - {r}")

    def validate(self):
        """
        validate index_name
        """
        self.name2story = {
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
        try:
            self.stories = self.name2story[self.index_name].stream_from_corpus()
        except KeyError:
            raise KeyError(f"Invalid index: {self.index_name}")

    def index(self):
        """
        Index Stories.
        """
        for batch in tqdm(chunked(self.stories, self.batch_size),
                          desc=f"indexing {self.index}..."):
            batch: List[Document]  # a batch is a list of Document
            # must make sure include_meta is set to true, otherwise the helper won't be
            # aware of the name of the index that= we are indexing the corpus into
            actions = (doc.to_dict(include_meta=True) for doc in batch)
            r = bulk(self.es, actions)
            print(f"successful count: {r[0]}, error messages: {r[1]}")


class WisdomifyView(FlaskView):
    """
    sents -> wisdoms
    """

    def index(self):
        return "Index"

    @route('/search')
    def search(self):
        return "search"


class StorytellerView(FlaskView):
    """
    wisdoms -> examples & definitions
    """
    es_connect = Elasticsearch(ES_CLOUD_ID, http_auth=(ES_USERNAME, ES_PASSWORD))

    def index(self):
        form = request.json
        wisdom = form['wisdom']
        index_name = ",".join(Story.all_names())
        size = 10000

        flow = SearchFlow(self.es_connect, index_name, size)(wisdom)
        res = flow.res

        parsed = [
            f"index: {hit['_index']}, highlight:{hit['highlight']['sents'][0]}"
            for hit in res['hits']['hits']
        ]
        return jsonify(parsed)
