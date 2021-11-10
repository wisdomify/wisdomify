from elasticsearch import Elasticsearch


class Searcher:
    """
    supports storyteller/main/search.py
    """
    def __init__(self, client: Elasticsearch):
        self.client = client

    def __call__(self, wisdom: str, indices: str,  size: int) -> dict:
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
        return self.client.search(index=indices, query=query, highlight=highlight, size=size)
