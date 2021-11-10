"""
Anything that has to do with instantiating some sort of  a client via acessing .env,
goes into here.
"""
import wandb
from elasticsearch import Elasticsearch
from wandb.wandb_run import Run
from wisdomify.constants import (
    ES_USERNAME,
    ES_PASSWORD,
    ES_CLOUD_ID,
    WANDB_PROJECT,
)


# --- elasticsearch --- #
def connect_to_es():
    return Elasticsearch(ES_CLOUD_ID, http_auth=(ES_USERNAME, ES_PASSWORD))


# --- wandb  --- #
# note: if you need a decorator with parameters -  https://stackoverflow.com/a/5929165
def connect_to_wandb() -> Run:
    return wandb.init(project=WANDB_PROJECT)
