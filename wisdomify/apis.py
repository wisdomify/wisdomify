import json
import os
from functools import reduce

import requests as requests
from elasticsearch import Elasticsearch
from flask import jsonify
from transformers import BertTokenizerFast, AutoModelForMaskedLM

from wisdomify.constants import (
    ES_USERNAME,
    ES_PASSWORD,
    ES_CLOUD_ID, ROOT_DIR,
)
from wisdomify.docs import Story
from wisdomify.loaders import load_config, load_device
from wisdomify.models import RDBeta, RD
from wisdomify.tensors import WiskeysBuilder, Wisdom2SubwordsBuilder, Wisdom2EgInputsBuilder


def get_wisdom():
    byte_json = requests.get(
        'https://api.wandb.ai/artifactsV2/gcp-us/wisdomify/QXJ0aWZhY3Q6MzcwMjQ4MTQ=/2b3ca62907826459a957675957540776')
    return list(reduce(lambda i, j: i + j, json.loads(byte_json.content)['data']))


class WisdomifierAPI:
    def __init__(self):
        self.config = load_config()['rd_beta']['a']
        self.device = load_device(False)
        self.wisdoms = get_wisdom()

        self.tok_dir = os.path.join(ROOT_DIR, 'resource', 'tokenizer')
        self.rd_ckpt_path = os.path.join(ROOT_DIR, 'resource', 'rd.ckpt')

        self.bert_mlm = AutoModelForMaskedLM.from_pretrained(self.config['bert'])
        self.tokenizer = BertTokenizerFast.from_pretrained(self.tok_dir)  # path

        self.bert_mlm.resize_token_embeddings(len(self.tokenizer))
        self.wiskeys = WiskeysBuilder(self.tokenizer, self.device)(self.wisdoms)
        self.wisdom2subwords = Wisdom2SubwordsBuilder(self.tokenizer, self.config['k'], self.device)(self.wisdoms)
        self.rd = RDBeta.load_from_checkpoint(self.rd_ckpt_path,  # path
                                              bert_mlm=self.bert_mlm,
                                              wisdom2subwords=self.wisdom2subwords,
                                              wiskeys=self.wiskeys,
                                              device=self.device)

        self.inputs_builder = Wisdom2EgInputsBuilder(self.tokenizer, self.config['k'], self.device)

    def infer(self, sent):
        sents = [sent]
        wisdom2sent = [("", sent) for sent in sents]
        X = self.inputs_builder(wisdom2sent)

        P_wisdom = self.rd.P_wisdom(X)
        results = list()
        for S_word_prob in P_wisdom.tolist():
            wisdom2prob = [
                (wisdom, prob)
                for wisdom, prob in zip(self.wisdoms, S_word_prob)
            ]
            # sort and append
            results.append(sorted(wisdom2prob, key=lambda x: x[1], reverse=True))

        return results


class StorytellerAPI:
    def __init__(self):
        self.es = Elasticsearch(ES_CLOUD_ID, http_auth=(ES_USERNAME, ES_PASSWORD))

    def infer(self, wisdom):
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

        index_name = ",".join(Story.all_names())
        size = 10000

        res = self.es.search(index=index_name,
                             query=query,
                             highlight=highlight,
                             size=size)

        parsed = [
            f"index: {hit['_index']}, highlight:{hit['highlight']['sents'][0]}"
            for hit in res['hits']['hits']
        ]

        return jsonify(parsed)
