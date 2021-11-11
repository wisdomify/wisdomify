import random
import numpy as np
import io
import json
import pandas as pd
import requests
import torch
import wandb
from os import path
from typing import Callable, List, cast, Tuple, Dict, Generator
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from elasticsearch_dsl import Document
from more_itertools import chunked
from tqdm import tqdm
from transformers import BertTokenizerFast, AutoConfig, AutoModelForMaskedLM, BertForMaskedLM, AutoTokenizer
from wandb.sdk.wandb_run import Run
from wisdomify.loaders import load_conf
from wisdomify.models import RD, RDAlpha, RDBeta
from wisdomify.tensors import Wisdom2SubwordsBuilder, WiskeysBuilder
from wisdomify.connectors import connect_to_es
from wisdomify.constants import WISDOMS_A, WISDOMS_B, WISDOM2QUERY_RAW_A, WISDOM2DEF_RAW_A, WISDOM2DEF_RAW_B, \
    ARTIFACTS_DIR
from wisdomify.preprocess import parse, cleanse, normalise, augment, upsample, stratified_split


# ==== the superclass of all flows ==== #
class Flow:

    def __init__(self, on_init: bool):
        if on_init:
            self.__call__()

    def __call__(self, *args, **kwargs):
        for idx, step in enumerate(self.steps()):
            step()

    def steps(self) -> List[Callable]:
        raise NotImplementedError


# === elastic flows === #
from wisdomify.docs import (
    Story, GK, SC, MR, BS, DS,
    SFC, KESS, KJ, KCSS,
    SFKE, KSNS, KC, KETS,
    KEPT, News, KUNIV
)   # noqa


class SearchFlow(Flow):

    query: dict
    highlight: dict
    res: dict

    def __init__(self, es: Elasticsearch, wisdom: str, index: str, size: int, on_init: bool = True):
        self.es = es
        self.wisdom = wisdom
        self.index = index
        self.size = size
        super().__init__(on_init)

    def steps(self):
        return [
            self.build,
            self.search
        ]

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
        self.res = self.es.search(index=self.index,
                                  query=self.query,
                                  highlight=self.highlight,
                                  size=self.size)


class IndexFlow(Flow):

    name2story: Dict[str, Story]
    stories: Generator[Story, None, None]

    def __init__(self, es: Elasticsearch, index_name: str, batch_size: int,  on_init: bool = True):
        """
        :param es:
        """
        self.es = es
        self.index_name = index_name
        self.batch_size = batch_size
        super().__init__(on_init)

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


# ==== dataset flows ==== #
class DatasetFlow(Flow):

    artifact: wandb.Artifact
    raw_df: pd.DataFrame
    all_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    raw_table: wandb.Table
    all_table: wandb.Table
    val_table: wandb.Table
    test_table: wandb.Table

    def __init__(self, run: Run, ver: str, mode: str, on_init):
        self.run = run
        self.ver = ver
        self.mode = mode
        super().__init__(on_init)

    def steps(self) -> List[Callable]:
        # a flow executes the steps, defined by steps
        if self.mode == "download":
            return [
                self.download_artifact,
                self.download_tables
            ]
        elif self.mode == "build":
            return [
                self.download_raw_df,
                self.preprocess,
                self.val_test_split,
                self.build_artifact
            ]
        else:
            raise ValueError

    def download_artifact(self):
        self.artifact = self.run.use_artifact(f"{self.name}:{self.ver}")

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

    def __str__(self) -> str:
        """
        you might want to do some formatting later...
        """
        return "\n".join([step.__name__ for step in self.steps()])


class WisdomsFlow(DatasetFlow):

    def __init__(self, run: Run, ver: str, mode: str, on_init: bool = True):
        super().__init__(run, ver, mode, on_init)

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
        pass

    def preprocess(self):
        # we do not preprocess wisdoms
        pass

    def val_test_split(self):
        # we do not split wisdoms
        pass

    def build_artifact(self):
        artifact = wandb.Artifact("wisdoms", type="dataset")
        raw_table = wandb.Table(dataframe=self.raw_df)
        artifact.add(raw_table, "raw")
        self.artifact = artifact

    @property
    def name(self):
        return "wisdoms"


class Wisdom2QueryFlow(DatasetFlow):

    def __init__(self, run: Run, ver: str, mode: str, val_ratio: float = None, seed: int = None, on_init: bool = True):
        # if you are downloading it, you don't need to provide these
        self.val_ratio = val_ratio
        self.seed = seed
        super().__init__(run, ver, mode, on_init)

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
        self.val_df, self.test_df = stratified_split(self.raw_df, self.val_ratio, self.seed)

    def build_artifact(self):
        artifact = wandb.Artifact("wisdom2query", type="dataset")
        table2name = (
            (wandb.Table(dataframe=self.raw_df), "raw"),
            (wandb.Table(dataframe=self.all_df), "all"),
            (wandb.Table(dataframe=self.all_df), "val"),
            (wandb.Table(dataframe=self.all_df), "test")
        )
        for table, name in table2name:
            artifact.add(table, name)
        self.artifact = artifact

    @property
    def name(self):
        return "wisdom2query"


class Wisdom2DefFlow(DatasetFlow):

    def __init__(self, run: Run, ver: str, mode: str, on_init=True):
        super().__init__(run, ver, mode, on_init)

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
        artifact = wandb.Artifact("wisdom2def", type="dataset")
        raw_table = wandb.Table(dataframe=self.raw_df)
        all_table = wandb.Table(dataframe=self.all_df)
        artifact.add(raw_table, "raw")
        artifact.add(all_table, "all")
        self.artifact = artifact

    @property
    def name(self):
        return "wisdom2def"


class Wisdom2EgFlow(DatasetFlow):

    def __init__(self, run: Run, ver: str, mode: str, on_init=True):
        super().__init__(run, ver, mode, on_init)

    def download_tables(self):
        self.raw_table = cast(wandb.Table, self.artifact.get("raw"))
        self.all_table = cast(wandb.Table, self.artifact.get("all"))

    def download_raw_df(self):
        """
        search on elasticsearch indices!
        """
        table = WisdomsFlow(self.run, self.ver, mode="download").raw_table
        wisdoms = [row[0] for _, row in table.iterrows()]
        rows = list()
        with connect_to_es() as es:
            for wisdom in tqdm(wisdoms, desc="searching for wisdoms on stories...",
                               total=len(wisdoms)):
                flow = SearchFlow(es, wisdom, ",".join(Story.all_names()), size=10000)
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
        artifact = wandb.Artifact("wisdom2eg", type="dataset")
        raw_table = wandb.Table(dataframe=self.raw_df)
        all_table = wandb.Table(dataframe=self.all_df)
        artifact.add(raw_table, "raw")
        artifact.add(all_table, "all")
        self.artifact = artifact

    @property
    def name(self):
        return "wisdom2eg"


# === model flows === #
class RDFlow(Flow):

    rd: RD
    tokenizer: BertTokenizerFast
    artifact: wandb.Artifact
    bert_mlm: BertForMaskedLM
    wisdom2subwords: torch.Tensor
    artifact_path: str
    rd_bin_path: str
    tok_dir_path: str
    config: dict
    mode: str
    wisdoms: List[str]
    artifact: wandb.Artifact

    def __init__(self, run: Run, ver: str, device: torch.device):
        self.run = run
        self.ver = ver
        self.device = device
        super().__init__(on_init=False)

    def __call__(self, mode: str):
        self.mode = mode
        super(RDFlow, self).__call__()

    def steps(self) -> List[Callable]:
        # a flow executes the steps, defined by steps
        if self.mode == "download":
            return [
                self.download_artifact,
                self.save_paths,
                self.save_config,
                self.download_wisdoms,
                self.load_tokenizer,
                self.load_bert_mlm,
                self.build_wisdom2subwords,
                self.build_rd,
                self.load_rd
            ]
        elif self.mode == "build":
            return [
                self.load_config,
                self.download_wisdoms,
                self.download_bert_mlm,
                self.download_tokenizer,
                self.build_wisdom2subwords,
                self.build_rd,
                # to be used when saving
                self.save_paths,
            ]
        else:
            raise ValueError

    def download_artifact(self):
        self.run.use_artifact(f"{self.name}:{self.ver}").download()

    def save_paths(self):
        self.artifact_path = path.join(ARTIFACTS_DIR, f"{self.name}:{self.ver}")
        self.rd_bin_path = path.join(self.artifact_path, "rd.bin")
        self.tok_dir_path = path.join(self.artifact_path, "tokenizer")

    def save_config(self):
        self.config = self.artifact.metadata

    def download_wisdoms(self):
        table = WisdomsFlow(self.run, self.config["wisdoms_ver"], "download").raw_table
        self.wisdoms = [row[0] for _, row in table.iterrows()]

    def download_bert_mlm(self):
        self.bert_mlm = AutoModelForMaskedLM.from_pretrained(self.config['bert'])

    def download_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['bert'])

    def build_wisdom2subwords(self):
        self.wisdom2subwords = Wisdom2SubwordsBuilder(self.tokenizer, self.config['k'], self.device)(self.wisdoms)

    def build_rd(self):
        raise NotImplementedError

    def load_config(self):
        self.config = load_conf()[self.name][self.ver]

    def load_tokenizer(self):
        self.tokenizer = BertTokenizerFast.from_pretrained(self.tok_dir_path)

    def load_bert_mlm(self):
        self.bert_mlm = AutoModelForMaskedLM.from_config(AutoConfig.from_pretrained(self.config['bert']))

    def load_rd(self):
        self.rd.load_state_dict(torch.load(self.rd_bin_path))

    @property
    def name(self):
        raise NotImplementedError

    def __str__(self) -> str:
        """
        you might want to do some formatting later...
        """
        return "\n".join([step.__name__ for step in self.steps()])


class RDAlphaFlow(RDFlow):

    def __init__(self, run: Run, ver: str, device: torch.device):
        super().__init__(run, ver, device)

    def build_rd(self):
        self.rd = RDAlpha(self.bert_mlm, self.wisdom2subwords,
                          self.config['k'], self.config['lr'], self.device)

    @property
    def name(self):
        return "rd_alpha"


class RDBetaFlow(RDFlow):

    def __init__(self, run: Run, ver: str, device: torch.device):
        super().__init__(run, ver, device)

    def build_rd(self):
        self.tokenizer.add_tokens(self.wisdoms)
        self.bert_mlm.resize_token_embeddings(len(self.tokenizer))
        wiskeys = WiskeysBuilder(self.tokenizer, self.device)(self.wisdoms)
        rd = RDBeta(self.bert_mlm, self.wisdom2subwords, wiskeys,
                    self.config['k'], self.config['lr'], self.device)
        return rd

    @property
    def name(self):
        return "rd_beta"

# TODO implement your new RDModelFow here


# ======= experiment flows ======= #
from wisdomify import datamodules  # noqa


class ExperimentFlow(Flow):
    rd_flow: RDFlow
    datamodule: datamodules.WisdomifyDataModule

    def __init__(self, run: Run, model: str, ver: str, mode: str, device: torch.device, on_init: bool = True):
        self.run = run
        self.model = model
        self.ver = ver
        self.mode = mode
        self.device = device
        super().__init__(on_init)

    def steps(self) -> List[Callable]:
        if self.mode == "download":
            return [
                self.choose_rd_flow,
                self.run_download,
                self.build_datamodule,
                self.fix_seeds

            ]
        elif self.mode == "build":
            return [
                self.choose_rd_flow,
                self.run_build,
                self.build_datamodule,
                self.fix_seeds
            ]

    def choose_rd_flow(self):
        if self.model == "rd_alpha":
            self.rd_flow = RDAlphaFlow(self.run, self.ver, self.device)
        elif self.model == "rd_beta":
            self.rd_flow = RDBetaFlow(self.run, self.ver, self.device)
        else:
            # TODO: ADD your new flow here
            raise ValueError

    def fix_seeds(self):
        """
        https://pytorch.org/docs/stable/notes/randomness.html
        """
        seed = self.datamodule.config['seed']
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    def run_download(self):
        self.rd_flow(mode="download")

    def run_build(self):
        self.rd_flow(mode="build")

    def build_datamodule(self):
        if self.rd_flow.config["train_type"] == "wisdom2def":
            self.datamodule = datamodules.Wisdom2DefDataModule(self.rd_flow.config,
                                                               self.rd_flow.tokenizer,
                                                               self.rd_flow.wisdoms,
                                                               self.run,
                                                               self.device)
        elif self.rd_flow.config["train_type"] == "wisdom2eg":
            self.datamodule = datamodules.Wisdom2EgDataModule(self.rd_flow.config,
                                                              self.rd_flow.tokenizer,
                                                              self.rd_flow.wisdoms,
                                                              self.run,
                                                              self.device)
        else:
            raise ValueError


# ===== for deploying ==== #

class WisdomifyFlow(Flow):

    exp_flow: ExperimentFlow
    results: List[List[Tuple[str, float]]]

    def __init__(self, run: Run, model: str, ver: str, sents: List[str], device: torch.device, on_init: bool = True):
        self.run = run
        self.model = model
        self.ver = ver
        self.sents = sents
        self.device = device
        super().__init__(on_init)

    def steps(self) -> List[Callable]:
        return [
            self.download_experiment,
            self.switch_to_eval,
            self.wisdomify
        ]

    def download_experiment(self):
        self.exp_flow = ExperimentFlow(self.run, self.model, self.ver,
                                       "download", self.device)

    def switch_to_eval(self):
        self.exp_flow.rd_flow.rd.eval()

    def wisdomify(self):
        # get the X
        wisdom2sent = [("", desc) for desc in self.sents]
        inputs_builder, _ = self.exp_flow.datamodule.tensor_builders()
        X = inputs_builder(wisdom2sent)
        # get H_all for this.
        P_wisdom = self.exp_flow.rd_flow.rd.P_wisdom(X)
        results = list()
        for S_word_prob in P_wisdom.tolist():
            wisdom2prob = [
                (wisdom, prob)
                for wisdom, prob in zip(self.exp_flow.datamodule.wisdoms, S_word_prob)
            ]
            # sort and append
            results.append(sorted(wisdom2prob, key=lambda x: x[1], reverse=True))
        self.results = results
