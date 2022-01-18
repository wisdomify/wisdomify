"""
It is often better to stick to simple stuff.
Fetchers are one-off-called functions, so there is really no need to define them as classes.
just define them as functions.
"""

import io
import json
import wandb
import requests
import yaml
import pandas as pd
from tqdm import tqdm
from typing import Tuple, List
from transformers import BertTokenizer, AutoModelForMaskedLM, AutoConfig
from wisdomify.connectors import connect_to_es
from wisdomify.datamodules import Wisdom2DefDataModule, Wisdom2EgDataModule
from wisdomify.paths import CONFIG_YAML
from wisdomify import tensors as T
from wisdomify.elastic.docs import Story
from wisdomify.elastic.searcher import Searcher
from wisdomify.models import Alpha, Gamma, Beta, RD
from wisdomify.utils import Experiment


def fetch_url(url: str) -> str:
    r = requests.get(url)
    r.raise_for_status()
    r.encoding = 'utf-8'
    return r.text


def fetch_wisdoms_raw(ver: str) -> pd.DataFrame:
    if ver == "a":
        text = fetch_url("https://docs.google.com/spreadsheets/d/1--hzu43sd8nk8-R_Qf2jTZ0iuGQCIgRo3bwQ63SYVbo/export?format=tsv&gid=0")  # noqa
    elif ver == "b":
        text = fetch_url("https://docs.google.com/spreadsheets/d/1--hzu43sd8nk8-R_Qf2jTZ0iuGQCIgRo3bwQ63SYVbo/export?format=tsv&gid=822745026")  # noqa
    else:
        raise ValueError
    df = pd.read_csv(io.StringIO(text), delimiter="\t")
    return df


def fetch_wisdom2def_raw(ver: str) -> pd.DataFrame:
    if ver == "a":
        text = fetch_url("https://docs.google.com/spreadsheets/d/1n550JrAYnyy2j1CQAeXPjeuw0zD5RYNbpR4wKFTq8DI/export?format=tsv&gid=0")  # noqa
    elif ver == "b":
        text = fetch_url("https://docs.google.com/spreadsheets/d/1n550JrAYnyy2j1CQAeXPjeuw0zD5RYNbpR4wKFTq8DI/export?format=tsv&gid=1300142415")  # noqa
    else:
        raise ValueError
    df = pd.read_csv(io.StringIO(text), delimiter="\t")
    return df


def fetch_wisdom2eg_raw(ver: str) -> pd.DataFrame:
    wisdoms = fetch_wisdoms(ver)
    rows = list()
    with connect_to_es() as es:
        for wisdom in tqdm(wisdoms, desc="searching for wisdoms on stories...",
                           total=len(wisdoms)):
            searcher = Searcher(es, ",".join(Story.all_names()), size=10000)
            res = searcher(wisdom)
            # encoding korean as json files https://stackoverflow.com/a/18337754
            raw = json.dumps(res, ensure_ascii=False)
            rows.append((wisdom, raw))
    df = pd.DataFrame(data=rows, columns=["wisdom", "eg"])
    return df


def fetch_wisdom2query_raw(ver: str) -> pd.DataFrame:
    if ver == "a":
        text = fetch_url("https://docs.google.com/spreadsheets/d/17t-WFD9e8a9VUu2nda56I_akeyYJ9RYilUCIVTra7Es/export?format=tsv&gid=1307694002")  # noqa
    else:
        raise ValueError
    df = pd.read_csv(io.StringIO(text), delimiter="\t")
    return df

# # TODO: implement this later ...
# def fetch_tokenizer(ver: str) -> BertTokenizer:
#     artifact = wandb.Api().artifact(f"{WANDB_ENTITY}/{WANDB_PROJECT}/tokenizer:{ver}", type="other")
#     artifact_path = ...
#     tokenizer_path = artifact_path / "tokenizer"  # a dir
#     tokenizer = ...
#     return tokenizer


def fetch_wisdoms(ver: str) -> List[str]:
    artifact = wandb.Api().artifact(f"wisdomify/wisdomify/wisdoms:{ver}", type="dataset")
    table = artifact.get("raw")
    wisdoms = [row[0] for row in table.data]
    return wisdoms


def fetch_wisdom2def(ver: str) -> List[Tuple[str, str]]:
    artifact = wandb.Api().artifact(f"wisdomify/wisdomify/wisdom2def:{ver}", type="dataset")
    train = [(row[0], row[1]) for row in artifact.get("train").data]
    return train


def fetch_wisdom2eg(ver: str) -> List[Tuple[str, str]]:
    artifact = wandb.Api().artifact(f"wisdomify/wisdomify/wisdom2eg:{ver}", type="dataset")
    train = [(row[0], row[1]) for row in artifact.get("train").data]
    return train


def fetch_wisdom2query(ver: str) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    artifact = wandb.Api().artifact(f"wisdomify/wisdomify/wisdom2query:{ver}", type="dataset")
    val = [(row[0], row[1]) for row in artifact.get("val").data]
    test = [(row[0], row[1]) for row in artifact.get("test").data]
    return val, test


def fetch_experiment(model: str, ver: str) -> Experiment:
    # fetching rd... just fetching one would be enough?
    artifact = wandb.Api().artifact(f"wisdomify/wisdomify/{model}:{ver}", type="model")
    config = artifact.metadata
    artifact_path = artifact.checkout()
    ckpt_path = artifact_path / "rd.ckpt"
    tok_path = artifact_path / "tokenizer"
    mlm = AutoModelForMaskedLM.from_config(AutoConfig.from_pretrained(config['bert']))
    tokenizer = BertTokenizer.from_pretrained(str(tok_path))
    wisdoms = fetch_wisdoms(config['wisdoms_ver'])
    wisdom2subwords = T.wisdom2subwords(tokenizer, wisdoms, config['k'])
    # choose the rd
    if model == Alpha.name():
        rd = Alpha.load_from_checkpoint(ckpt_path, mlm=mlm, wisdom2subwords=wisdom2subwords)
    elif model == Beta.name():
        mlm.resize_token_embeddings(len(tokenizer))
        wiskeys = T.wiskeys(tokenizer, wisdoms)
        rd = Beta.load_from_checkpoint(ckpt_path, mlm=mlm,
                                       wisdom2subwords=wisdom2subwords, wiskeys=wiskeys)
    elif model == Gamma.name():
        rd = Gamma.load_from_checkpoint(ckpt_path, mlm=mlm, wisdom2subwords=wisdom2subwords)
    else:
        raise ValueError
    # choose the datamodule
    if config["train_type"] == "wisdom2def":
        datamodule = Wisdom2DefDataModule(config,
                                          tokenizer,
                                          wisdoms)
    elif config["train_type"] == "wisdom2eg":
        datamodule = Wisdom2EgDataModule(config,
                                         tokenizer,
                                         wisdoms)
    else:
        raise ValueError
    # build an experiment instance.
    return Experiment(rd, config, datamodule)


# a fetcher for fetching config, which contains every set up for this project.
def fetch_config() -> dict:
    with open(CONFIG_YAML, 'r', encoding="utf-8") as fh:
        return yaml.safe_load(fh)
