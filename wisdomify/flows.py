import io
import json
from abc import ABC
from typing import List, Callable
import pandas as pd
import wandb
from tqdm import tqdm
from wandb.sdk.wandb_run import Run

from wisdomify.connectors import connect_to_es
from wisdomify.constants import WISDOMS_A, WISDOMS_B, WISDOM2QUERY_RAW_A, WISDOM2DEF_RAW_A, WISDOM2DEF_RAW_B
from wisdomify.elastic_docs import Story
from wisdomify.elastic_searcher import Searcher
from wisdomify.preprocess import cleanse, normalise, stratified_split, augment, upsample, parse
from wisdomify.wandb_downloaders import dl_wisdoms


class Flow:
    def __call__(self, *args, **kwargs):
        for step in self.steps():
            step()

    def steps(self) -> List[Callable]:
        raise NotImplementedError


class BuildFlow(Flow, ABC):

    def __init__(self, run: Run, ver: str):
        self.run = run
        self.ver = ver


class BuildWisdomsFlow(BuildFlow):
    # --- to be saved locally --- #
    all_df: pd.DataFrame

    def steps(self) -> List[Callable]:
        return [
            self.download,
            self.upload
        ]

    def download(self):
        """
        ver  -> all_df
        """
        if self.ver == "a":
            text = get_url(WISDOMS_A)
        elif self.ver == "b":
            text = get_url(WISDOMS_B)
        else:
            raise ValueError
        self.all_df = pd.read_csv(io.StringIO(text), delimiter="\t")

    def upload(self):
        """
        package the dataframes into an artifact
        """
        artifact = wandb.Artifact(name="wisdoms", type="dataset")
        table = wandb.Table(dataframe=self.all_df)
        artifact.add(table, name="all")
        self.run.log_artifact(artifact, aliases=[self.ver, "latest"])


class BuildWisdom2QueryFlow(BuildFlow):

    def __init__(self, run: Run, ver: str, val_ratio: float, seed: int):
        super().__init__(run, ver)
        self.val_ratio = val_ratio
        self.seed = seed

    # --- to be saved locally --- #
    raw_df: pd.DataFrame
    all_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame

    def steps(self) -> List[Callable]:
        return [
            self.download,
            self.preprocess,
            self.upload
        ]

    def download(self):
        """
        ver  -> raw_df
        """
        if self.ver == "a":
            text = get_url(WISDOM2QUERY_RAW_A)
        else:
            raise ValueError
        self.raw_df = pd.read_csv(io.StringIO(text), delimiter="\t")

    def preprocess(self):
        """
        raw_df -> all_df
        """
        self.all_df = self.raw_df \
                          .pipe(cleanse) \
                          .pipe(normalise)

    def val_test_split(self):
        """
        all_df -> val_df, test_df
        """
        self.val_df, self.test_df = stratified_split(self.raw_df, self.val_ratio, self.seed)

    def upload(self):
        """
        raw_df, all_df, val_df, test_df
        -> raw_table, all_table, val_table, test_table
        -> artifact: upload this
        """
        artifact = wandb.Artifact("wisdom2query", type="dataset")
        artifact.metadata = {"ver": self.ver, "seed": self.seed}
        raw_table = wandb.Table(dataframe=self.raw_df)
        all_table = wandb.Table(dataframe=self.all_df)
        val_table = wandb.Table(dataframe=self.val_df)
        test_table = wandb.Table(dataframe=self.test_df)
        # add the tables to the artifact
        artifact.add(raw_table, "raw")
        artifact.add(all_table, "all")
        artifact.add(val_table, "val")
        artifact.add(test_table, "test")
        self.run.log_artifact(artifact, aliases=[self.ver, "latest"])


class BuildWisdom2DefFlow(BuildFlow):



    raw_df: pd.DataFrame
    all_df: pd.DataFrame

    def steps(self) -> List[Callable]:
        return [
            self.download,
            self.preprocess,
            self.upload
        ]

    def download(self):
        """
        ver  -> raw_df
        """
        if self.ver == "a":
            text = get_url(WISDOM2DEF_RAW_A)
        elif self.ver == "b":
            text = get_url(WISDOM2DEF_RAW_B)
        else:
            raise ValueError
        self.raw_df = pd.read_csv(io.StringIO(text), delimiter="\t")

    def preprocess(self):
        """
        raw_df -> all_df
        """
        self.all_df = self.raw_df \
                          .pipe(cleanse) \
                          .pipe(normalise) \
                          .pipe(augment) \
                          .pipe(upsample)

    def upload(self):
        """
        raw_df, all_df
        -> raw_table, all_table
        -> artifact: upload this
        """
        artifact = wandb.Artifact("wisdom2def", type="dataset")
        raw_table = wandb.Table(dataframe=self.raw_df)
        all_table = wandb.Table(dataframe=self.all_df)
        # add the tables to the artifact
        artifact.add(raw_table, "raw")
        artifact.add(all_table, "all")
        self.run.log_artifact(artifact, aliases=[self.ver, "latest"])


class BuildWisdom2EgFlow(BuildFlow):

    wisdoms: List[str]
    raw_df: pd.DataFrame
    all_df: pd.DataFrame

    def steps(self) -> List[Callable]:
        return [
            self.download,
            self.search,
            self.preprocess,
            self.upload
        ]

    def download(self):
        self.wisdoms = dl_wisdoms(self.ver)

    def search(self):
        """
        ver -> raw_df
        """
        # ---
        rows = list()
        with connect_to_es() as es:
            searcher = Searcher(es)
            for wisdom in tqdm(self.wisdoms, desc="searching for wisdoms on stories...",
                               total=len(self.wisdoms)):
                raw = searcher(wisdom, ",".join(Story.all_indices()), size=10000)
                # https://stackoverflow.com/a/18337754
                raw = json.dumps(raw, ensure_ascii=False)
                rows.append((wisdom, raw))
        self.raw_df = pd.DataFrame(data=rows, columns=["wisdom", "eg"])

    def preprocess(self):
        """
        raw_df -> all_df
        """
        self.all_df = self.raw_df \
                          .pipe(parse) \
                          .pipe(cleanse) \
                          .pipe(normalise) \
                          .pipe(augment) \
                          .pipe(upsample)

    def upload(self):
        """
        raw_df, all_df
        -> raw_table, all_table
        -> wisdom2eg_artifact
        """
        artifact = wandb.Artifact("wisdom2eg", type="dataset")
        raw_table = wandb.Table(dataframe=self.raw_df)
        all_table = wandb.Table(dataframe=self.all_df)
        # add the tables to the artifact
        artifact.add(raw_table, "raw")
        artifact.add(all_table, "all")
        self.run.log_artifact(artifact, aliases=[self.ver, "latest"])
