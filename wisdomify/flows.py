import os
import random
import numpy as np
import io
import json
import pandas as pd
import requests
import torch
import wandb
from os import path
from typing import Callable, List, cast, Dict, Generator, Optional
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from elasticsearch_dsl import Document
from more_itertools import chunked
from tqdm import tqdm
from transformers import BertTokenizerFast, AutoConfig, AutoModelForMaskedLM, BertForMaskedLM, AutoTokenizer
from wandb.sdk.wandb_run import Run
from wisdomify.models import RD, RDAlpha, RDBeta, RDGamma, RDGammaSync
from wisdomify.tensors import Wisdom2SubwordsBuilder, WiskeysBuilder
from wisdomify.connectors import connect_to_es
from wisdomify.constants import WISDOMS_A, WISDOMS_B, WISDOM2QUERY_RAW_A, WISDOM2DEF_RAW_A, WISDOM2DEF_RAW_B, \
    ARTIFACTS_DIR
from wisdomify.preprocess import parse, cleanse, normalise, augment, upsample, stratified_split
from termcolor import colored


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


# === elastic flows === #
from wisdomify.docs import (
    Story, GK, SC, MR, BS, DS,
    SFC, KESS, KJ, KCSS,
    SFKE, KSNS, KC, KETS,
    KEPT, News, KUNIV
)   # noqa


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


class TwoWayFlow(Flow):

    def __init__(self, run: Run, ver: str):
        self.run = run
        self.ver = ver
        # --- to be filled later --- #
        self.mode: Optional[str] = None
        self.config: Optional[dict] = None
        self.artifact: Optional[wandb.Artifact] = None

    def __call__(self, mode: str, config: dict):
        self.mode = mode
        self.config = config
        super(TwoWayFlow, self).__call__()
        return self

    def steps(self) -> List[Callable]:
        if self.mode == "d":
            return self.download_steps()
        elif self.mode == "b":
            return self.build_steps()
        else:
            raise ValueError

    def download_steps(self):
        raise NotImplementedError

    def build_steps(self):
        raise NotImplementedError

    # --- common steps to reuse --- #
    def use_artifact(self):
        self.artifact = self.run.use_artifact(f"{self.name}:{self.ver}")

    @property
    def name(self):
        raise NotImplementedError


# ==== dataset flows ==== #
class DatasetFlow(TwoWayFlow):

    def __init__(self, run: Run, ver: str):
        super().__init__(run, ver)
        # --- to build --- #
        self.raw_df: Optional[pd.DataFrame] = None
        self.all_df: Optional[pd.DataFrame] = None
        self.val_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        self.raw_table: Optional[wandb.Table] = None
        self.all_table: Optional[wandb.Table] = None
        self.val_table: Optional[wandb.Table] = None
        self.test_table: Optional[wandb.Table] = None
        self.artifact_path: Optional[str] = None

    def download_steps(self):
        return [
            self.use_artifact,
            self.download_tables
        ]

    def build_steps(self):
        return [
            self.download_raw_df,
            self.preprocess,
            self.val_test_split,
            self.build_artifact
        ]

    def download_tables(self):
        raise NotImplementedError

    def download_raw_df(self):
        raise NotImplementedError

    def preprocess(self):
        raise NotImplementedError

    def val_test_split(self):
        raise NotImplementedError

    def build_artifact(self):
        raise NotImplementedError

    @property
    def name(self):
        raise NotImplementedError

    @staticmethod
    def get(url: str) -> str:
        r = requests.get(url)
        r.raise_for_status()
        r.encoding = 'utf-8'
        return r.text


class WisdomsFlow(DatasetFlow):

    def download_tables(self):
        # download any tables you need from the given artifact
        self.raw_table = cast(wandb.Table, self.artifact.get("raw"))

    def download_raw_df(self):
        if self.ver == "a":
            text = self.get(WISDOMS_A)
        elif self.ver == "b":
            text = self.get(WISDOMS_B)
        else:
            raise ValueError
        self.raw_df = pd.read_csv(io.StringIO(text), delimiter="\t")

    def preprocess(self):
        # we do not preprocess wisdoms
        pass

    def val_test_split(self):
        # we do not split wisdoms
        pass

    def build_artifact(self):
        self.artifact = wandb.Artifact("wisdoms", type="dataset")
        raw_table = wandb.Table(dataframe=self.raw_df)
        self.artifact.add(raw_table, "raw")

    @property
    def name(self):
        return "wisdoms"


class Wisdom2QueryFlow(DatasetFlow):

    def __init__(self, run: Run, ver: str):
        super().__init__(run, ver)

    def download_tables(self):
        self.raw_table = cast(wandb.Table, self.artifact.get("raw"))
        self.all_table = cast(wandb.Table, self.artifact.get("all"))
        self.val_table = cast(wandb.Table, self.artifact.get("val"))
        self.test_table = cast(wandb.Table, self.artifact.get("test"))

    def download_raw_df(self):
        if self.ver == "a":
            text = self.get(WISDOM2QUERY_RAW_A)
        else:
            raise ValueError
        self.raw_df = pd.read_csv(io.StringIO(text), delimiter="\t")

    def preprocess(self):
        self.all_df = self.raw_df \
            .pipe(cleanse) \
            .pipe(normalise)

    def val_test_split(self):
        self.val_df, self.test_df = stratified_split(self.raw_df, self.config['val_ratio'], self.config['seed'])

    def build_artifact(self):
        self.artifact = wandb.Artifact("wisdom2query", type="dataset")
        table2name = (
            (wandb.Table(dataframe=self.raw_df), "raw"),
            (wandb.Table(dataframe=self.all_df), "all"),
            (wandb.Table(dataframe=self.val_df), "val"),
            (wandb.Table(dataframe=self.test_df), "test")
        )
        for table, name in table2name:
            self.artifact.add(table, name)

    @property
    def name(self):
        return "wisdom2query"


class Wisdom2DefFlow(DatasetFlow):

    def download_tables(self):
        self.raw_table = cast(wandb.Table, self.artifact.get("raw"))
        self.all_table = cast(wandb.Table, self.artifact.get("all"))

    def download_raw_df(self):
        if self.ver == "a":
            text = self.get(WISDOM2DEF_RAW_A)
        elif self.ver == "b":
            text = self.get(WISDOM2DEF_RAW_B)
        else:
            raise ValueError
        self.raw_df = pd.read_csv(io.StringIO(text), delimiter="\t")

    def preprocess(self):
        self.all_df = self.raw_df \
                          .pipe(cleanse) \
                          .pipe(normalise) \
                          .pipe(augment) \
                          .pipe(upsample)

    def val_test_split(self):
        # we do not split wisdom2def
        pass

    def build_artifact(self):
        self.artifact = wandb.Artifact("wisdom2def", type="dataset")
        raw_table = wandb.Table(dataframe=self.raw_df)
        all_table = wandb.Table(dataframe=self.all_df)
        self.artifact.add(raw_table, "raw")
        self.artifact.add(all_table, "all")

    @property
    def name(self):
        return "wisdom2def"


class Wisdom2EgFlow(DatasetFlow):

    def download_tables(self):
        self.raw_table = cast(wandb.Table, self.artifact.get("raw"))
        self.all_table = cast(wandb.Table, self.artifact.get("all"))

    def download_raw_df(self):
        """
        search on elasticsearch indices!
        """
        table = WisdomsFlow(self.run, self.ver)(mode="d", config=self.config).raw_table
        wisdoms = [row[0] for row in table.data]
        rows = list()
        with connect_to_es() as es:
            for wisdom in tqdm(wisdoms, desc="searching for wisdoms on stories...",
                               total=len(wisdoms)):
                flow = SearchFlow(es, ",".join(Story.all_names()), size=10000)(wisdom)
                # encoding korean as json files https://stackoverflow.com/a/18337754
                raw = json.dumps(flow.res, ensure_ascii=False)
                rows.append((wisdom, raw))
        self.raw_df = pd.DataFrame(data=rows, columns=["wisdom", "eg"])

    def preprocess(self):
        self.all_df = self.raw_df \
                          .pipe(parse) \
                          .pipe(cleanse) \
                          .pipe(normalise) \
                          .pipe(augment) \
                          .pipe(upsample)

    def val_test_split(self):
        # we do not split wisdom2eg
        pass

    def build_artifact(self):
        self.artifact = wandb.Artifact("wisdom2eg", type="dataset")
        raw_table = wandb.Table(dataframe=self.raw_df)
        all_table = wandb.Table(dataframe=self.all_df)
        self.artifact.add(raw_table, "raw")
        self.artifact.add(all_table, "all")

    @property
    def name(self):
        return "wisdom2eg"


# === model flows === #
class RDFlow(TwoWayFlow):

    def __init__(self, run: Run, ver: str):
        super().__init__(run, ver)
        self.rd: Optional[RD] = None
        self.tokenizer: Optional[BertTokenizerFast] = None
        self.artifact: Optional[wandb.Artifact] = None
        self.bert_mlm: Optional[BertForMaskedLM] = None
        self.wisdom2subwords: Optional[torch.Tensor] = None
        self.artifact_path: Optional[str] = None
        self.rd_ckpt_path: Optional[str] = None
        self.tok_dir_path: Optional[str] = None
        self.config: Optional[dict] = None
        self.mode: Optional[str] = None
        self.wisdoms: Optional[List[str]] = None
        self.artifact: Optional[wandb.Artifact] = None

    def download_steps(self):
        return [
                self.use_artifact,
                self.checkout_artifact,
                self.save_paths,
                self.save_config,  # at download, you don't have access to config
                self.load_tokenizer,
                self.build_bert_mlm,
                self.download_wisdoms,
                self.build_wisdom2subwords,
                self.load_rd  # load rd from checkpoint
            ]

    def build_steps(self):
        return [
                self.save_paths,
                self.download_bert_mlm,
                self.download_tokenizer,
                self.download_wisdoms,
                self.build_wisdom2subwords,
                self.build_rd  # build a new rd instance
            ]

    def checkout_artifact(self):
        self.artifact_path = self.artifact.checkout()

    def save_paths(self):
        if not self.artifact_path:
            self.artifact_path = path.join(ARTIFACTS_DIR, self.name)
        self.rd_ckpt_path = path.join(self.artifact_path, "rd.ckpt")
        self.tok_dir_path = path.join(self.artifact_path, "tokenizer")
        # and make directories as well, if they don't exist
        os.makedirs(self.artifact_path, exist_ok=True)
        os.makedirs(self.tok_dir_path, exist_ok=True)

    def save_config(self):
        self.config = self.artifact.metadata

    def download_wisdoms(self):
        table = WisdomsFlow(self.run, self.config["wisdoms_ver"])(mode="d", config=self.config).raw_table
        self.wisdoms = [row[0] for row in table.data]

    def download_bert_mlm(self):
        self.bert_mlm = AutoModelForMaskedLM.from_pretrained(self.config['bert'])

    def download_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['bert'])

    def build_wisdom2subwords(self):
        self.wisdom2subwords = Wisdom2SubwordsBuilder(self.tokenizer, self.config['k'])(self.wisdoms)

    def build_rd(self):
        raise NotImplementedError

    def build_bert_mlm(self):
        self.bert_mlm = AutoModelForMaskedLM.from_config(AutoConfig.from_pretrained(self.config['bert']))

    def load_tokenizer(self):
        self.tokenizer = BertTokenizerFast.from_pretrained(self.tok_dir_path)

    def load_rd(self):
        raise NotImplementedError

    @property
    def name(self):
        raise NotImplementedError

    def __str__(self) -> str:
        """
        you might want to do some formatting later...
        """
        return "\n".join([step.__name__ for step in self.steps()])


class RDAlphaFlow(RDFlow):

    def build_rd(self):
        self.rd = RDAlpha(self.config['k'], self.config['lr'],
                          self.bert_mlm, self.wisdom2subwords)

    def load_rd(self):
        self.rd = RDAlpha.load_from_checkpoint(self.rd_ckpt_path,
                                               wisdom2subwords=self.wisdom2subwords,
                                               bert_mlm=self.bert_mlm)

    @property
    def name(self):
        return "rd_alpha"


class RDBetaFlow(RDFlow):

    def build_rd(self):
        # add tokens only if you are building it.
        # if you are downloading a tokenizer from wand Artifact (i.e. mode == d), you don't need to do this
        # as the tokens would have been already added
        self.tokenizer.add_tokens(self.wisdoms)
        self.bert_mlm.resize_token_embeddings(len(self.tokenizer))
        wiskeys = WiskeysBuilder(self.tokenizer)(self.wisdoms)
        self.rd = RDBeta(self.config['k'], self.config['lr'],
                         self.bert_mlm, self.wisdom2subwords, wiskeys)

    def load_rd(self):
        self.bert_mlm.resize_token_embeddings(len(self.tokenizer))
        wiskeys = WiskeysBuilder(self.tokenizer)(self.wisdoms)
        self.rd = RDBeta.load_from_checkpoint(self.rd_ckpt_path,
                                              bert_mlm=self.bert_mlm,
                                              wisdom2subwords=self.wisdom2subwords,
                                              wiskeys=wiskeys)

    @property
    def name(self):
        return "rd_beta"


class RDGammaFlow(RDFlow):
    """
    TODO: a new rd flow
    """
    def build_rd(self):
        """
        at this point, you have access to:
        self.bert_mlm
        self.tokenizer
        self.device
        self.wisdoms
        self.wisdom2subwords
        self.config
        get use of these data to build your rd, and save to:
        self.rd <= RDSomething(...)
        """
        self.rd = RDGamma(self.config['k'], self.config['lr'],
                          self.bert_mlm, self.wisdom2subwords)

    def load_rd(self):
        self.rd = RDGamma.load_from_checkpoint(self.rd_ckpt_path,
                                               bert_mlm=self.bert_mlm,
                                               wisdom2subwords=self.wisdom2subwords)

    @property
    def name(self):
        return "rd_gamma"


class RDGammaSyncFlow(RDFlow):
    """
    TODO: a new rd flow
    """
    def build_rd(self):
        """
        at this point, you have access to:
        self.bert_mlm
        self.tokenizer
        self.device
        self.wisdoms
        self.wisdom2subwords
        self.config
        get use of these data to build your rd, and save to:
        self.rd <= RDSomething(...)
        """
        self.rd = RDGammaSync(self.config['k'], self.config['lr'],
                              self.bert_mlm, self.wisdom2subwords)

    def load_rd(self):
        self.rd = RDGammaSync.load_from_checkpoint(self.rd_ckpt_path,
                                                   bert_mlm=self.bert_mlm,
                                                   wisdom2subwords=self.wisdom2subwords)

    @property
    def name(self):
        return "rd_gamma_sync"


# ======= experiment flows ======= #
# this is here to prevent circular import error
from wisdomify.datamodules import WisdomifyDataModule, Wisdom2EgDataModule, Wisdom2DefDataModule # noqa


class ExperimentFlow(TwoWayFlow):

    def __init__(self, run: Run, model: str, ver: str):
        super().__init__(run, ver)
        self.model = model
        # top be filled
        self.config: Optional[dict] = None
        self.mode: Optional[str] = None
        self.rd_flow: Optional[RDFlow] = None
        self.datamodule: Optional[WisdomifyDataModule] = None

    def download_steps(self) -> List[Callable]:
        return [
                self.fix_seeds,  # an experiment must be preceded by fixing seeds
                self.choose_rd_flow,
                self.run_download,
                self.build_datamodule

            ]

    def build_steps(self) -> List[Callable]:
        return [
                self.fix_seeds,  # an experiment must be preceded by fixing seeds
                self.choose_rd_flow,
                self.run_build,
                self.build_datamodule
            ]

    def choose_rd_flow(self):
        if self.model == "rd_alpha":
            self.rd_flow = RDAlphaFlow(self.run, self.ver)
        elif self.model == "rd_beta":
            self.rd_flow = RDBetaFlow(self.run, self.ver)
        elif self.model == "rd_gamma":
            self.rd_flow = RDGammaFlow(self.run, self.ver)
        elif self.model == "rd_gamma_sync":
            self.rd_flow = RDGammaSyncFlow(self.run, self.ver)
        else:
            raise ValueError(f"Invalid model:{self.model}")

    def run_download(self):
        self.rd_flow(mode="d", config=self.config)

    def run_build(self):
        self.rd_flow(mode="b", config=self.config)

    def build_datamodule(self):
        if self.rd_flow.config["train_type"] == "wisdom2def":
            self.datamodule = Wisdom2DefDataModule(self.rd_flow.config,
                                                   self.rd_flow.tokenizer,
                                                   self.rd_flow.wisdoms,
                                                   self.run)
        elif self.rd_flow.config["train_type"] == "wisdom2eg":
            self.datamodule = Wisdom2EgDataModule(self.rd_flow.config,
                                                  self.rd_flow.tokenizer,
                                                  self.rd_flow.wisdoms,
                                                  self.run)
        else:
            raise ValueError

    def fix_seeds(self):
        """
        https://pytorch.org/docs/stable/notes/randomness.html
        """
        seed = self.config['seed']
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    @property
    def name(self):
        # otherwise, you would get:
        return None
