import json

import requests as requests
import torch

from wisdomify.constants import (
    ES_USERNAME,
    ES_PASSWORD,
    ES_CLOUD_ID,
    WANDB_PROJECT, ROOT_DIR,
)

from elasticsearch import Elasticsearch

from flask import jsonify, request
from flask_classful import FlaskView, route

from transformers import BertTokenizerFast, AutoConfig, AutoModelForMaskedLM, BertForMaskedLM, AutoTokenizer

from wisdomify.docs import Story
from wisdomify.loaders import load_config, load_device
from wisdomify.models import RDBeta
from wisdomify.tensors import WiskeysBuilder, Wisdom2SubwordsBuilder
from wisdomify.wisdomifier import Wisdomifier


def get_wisdom():
    byte_json = requests.get(
        'https://api.wandb.ai/artifactsV2/gcp-us/wisdomify/QXJ0aWZhY3Q6MzcwMjQ4NDQ=/36b09d5f21a92db67d6c099e93b07d8c')
    return json.loads(byte_json.content)['data']


class WisdomifyView(FlaskView):
    """
    sents -> wisdoms
    """
    config = load_config()['rd_beta']['b']
    device = load_device(False)
    wisdoms = get_wisdom()

    def index(self):
        form = request.json
        sent = form['sent']

        bert_mlm = AutoModelForMaskedLM.from_pretrained(self.config['bert'])
        tokenizer = BertTokenizerFast.from_pretrained(self.tok_dir_path)

        bert_mlm.resize_token_embeddings(len(tokenizer))
        wiskeys = WiskeysBuilder(tokenizer, self.device)(self.wisdoms)
        wisdom2subwords = Wisdom2SubwordsBuilder(tokenizer, self.config['k'], self.device)(self.wisdoms)
        rd = RDBeta(bert_mlm, wisdom2subwords, wiskeys,
                    self.config['k'], self.config['lr'], self.device)

        rd.load_state_dict(torch.load(self.rd_bin_path))

        datamodule = Wisdom2DefDataModule(self.config,
                                          tokenizer,
                                          self.wisdoms,
                                          None,
                                          self.device)

        wisdomifier = Wisdomifier(rd, datamodule)

        result = wisdomifier(sents=[sent])

        return "Index"


class StorytellView(FlaskView):
    """
    wisdoms -> examples & definitions
    """

    def index(self):
        es = Elasticsearch(ES_CLOUD_ID, http_auth=(ES_USERNAME, ES_PASSWORD))

        form = request.json
        wisdom = form['wisdom']

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

        res = es.search(index=index_name,
                        query=query,
                        highlight=highlight,
                        size=size)

        parsed = [
            f"index: {hit['_index']}, highlight:{hit['highlight']['sents'][0]}"
            for hit in res['hits']['hits']
        ]

        return jsonify(parsed)
