"""
Anything that has to do with instantiating some sort of  a es via accessing .env,
goes into here.
"""
import os
import wandb
from dotenv import find_dotenv, load_dotenv
from elasticsearch import Elasticsearch
from wandb.wandb_run import Run
from wisdomify.paths import ROOT_DIR
load_dotenv(find_dotenv())


# --- elasticsearch --- #
def connect_to_es() -> Elasticsearch:
    return Elasticsearch(os.getenv("ES_CLOUD_ID"), http_auth=(os.getenv("ES_USERNAME"), os.getenv("ES_PASSWORD")))


# --- wandb  --- #
# note: if you need a decorator with parameters -  https://stackoverflow.com/a/5929165
def connect_to_wandb(job_type: str, config: dict) -> Run:
    """
    job_type: either build, eval, infer or train.
    config: the hyper parameters for this experiment (needed for visualising the results on wandb)
    """
    return wandb.init(dir=ROOT_DIR, entity="wisdomify", project="wisdomify", job_type=job_type, config=config)