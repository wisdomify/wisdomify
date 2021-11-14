import argparse
import json

import requests as requests
from elasticsearch import Elasticsearch
from flask import jsonify, request
from flask_classful import FlaskView

from wisdomify import flows
from wisdomify.connectors import connect_to_wandb
from wisdomify.constants import (
    ES_USERNAME,
    ES_PASSWORD,
    ES_CLOUD_ID,
)
from wisdomify.docs import Story
from wisdomify.loaders import load_config, load_device
from wisdomify.wisdomifier import Wisdomifier


def get_wisdom():
    byte_json = requests.get(
        'https://api.wandb.ai/artifactsV2/gcp-us/wisdomify/QXJ0aWZhY3Q6MzcwMjQ4MTQ=/2b3ca62907826459a957675957540776')
    return json.loads(byte_json.content)['data']


class WisdomifyView(FlaskView):
    """
    sents -> wisdoms
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--ver", type=str,
                        default='a',
                        required=True,
                        help="This parameter is used for wisdomifyAPI deployment."
                             "The parameter should be the model version described on WandB.")
    parser.add_argument("--model", type=str,
                        default='rd_beta',
                        required=True,
                        help="This parameter is used for wisdomifyAPI deployment."
                             "The parameter should be the model name described on WandB.")

    args = parser.parse_args()
    ver = args.ver
    model = args.model

    config = load_config()[model][ver]
    device = load_device(False)
    with connect_to_wandb(job_type="infer", config=config) as run:
        # --- init a wisdomifier --- #
        flow = flows.ExperimentFlow(run, model, ver, device)("d", config)
    # --- wisdomifier is independent of wandb run  --- #
    wisdomifier = Wisdomifier(flow.rd_flow.rd, flow.datamodule)

    def index(self):
        form = request.json
        sent = form['sent']

        results = self.wisdomifier(sents=[sent])

        return jsonify(results)


class StorytellView(FlaskView):
    """
    wisdoms -> examples & definitions
    """
    es = Elasticsearch(ES_CLOUD_ID, http_auth=(ES_USERNAME, ES_PASSWORD))

    def index(self):
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

        res = self.es.search(index=index_name,
                             query=query,
                             highlight=highlight,
                             size=size)

        parsed = [
            f"index: {hit['_index']}, highlight:{hit['highlight']['sents'][0]}"
            for hit in res['hits']['hits']
        ]

        return jsonify(parsed)
