from elasticsearch import Elasticsearch


class Searcher:

    def __init__(self, es: Elasticsearch, index: str, size: int):
        self.es = es
        self.index = index
        self.size = size

    def __call__(self, wisdom: str) -> dict:
        query = {
            'match_phrase': {
                'sents': {
                    'query': wisdom
                }
            }
        }
        highlight = {
            'fields': {
                'sents': {
                    'type': 'plain',
                    'fragment_size': 100,
                    'number_of_fragments': 2,
                    'fragmenter': 'span'
                }
            }
        }

        res = self.es.search(index=self.index,
                             query=query,
                             highlight=highlight,
                             size=self.size)
        return res
