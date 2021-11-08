import os
import types
import torch
import wandb
from os import path
from abc import ABC
from typing import List, cast, Tuple
from transformers import BertTokenizerFast, AutoConfig, AutoModelForMaskedLM, BertForMaskedLM
from wandb.wandb_run import Run
from wisdomify.tensors import Wisdom2SubwordsTensor, WisKeysTensor
from wisdomify.models import RD, RDAlpha, RDBeta
from wisdomify.paths import ARTIFACTS_DIR


class ArtifactLoader:

    def __init__(self, run: Run):
        self.run = run

    def __call__(self, ver: str):
        raise NotImplementedError

    def use_artifact(self, ver: str) -> wandb.Artifact:
        """
        whenever you use an artifact, you must change the default root directory.
        """
        artifact = self.run.use_artifact(f"{self.name}:{ver}")
        # https://stackoverflow.com/a/42154067
        # terribly messy, but I must do this, if I were to save artifacts to local
        artifact._default_root = types.MethodType(lambda a, b=True: path.join(ARTIFACTS_DIR, a.name), artifact)
        return artifact

    @property
    def name(self) -> str:
        raise NotImplementedError


# --- datasets --- #
class WisdomsLoader(ArtifactLoader):

    def __call__(self, ver: str) -> List[str]:
        artifact = self.use_artifact(ver)
        table = cast(wandb.Table, artifact.get("wisdoms"))
        return [row[0] for _, row in table.iterrows()]

    @property
    def name(self) -> str:
        return "wisdoms"


class Wisdom2QueryLoader(ArtifactLoader):

    def __call__(self, ver: str) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        artifact = self.use_artifact(ver)
        val_table = cast(wandb.Table, artifact.get("val"))
        test_table = cast(wandb.Table, artifact.get("test"))
        val = [(row[0], row[1]) for _, row in val_table.iterrows()]
        test = [(row[0], row[1]) for _, row in test_table.iterrows()]
        return val, test

    @property
    def name(self):
        return "wisdom2query"


class Wisdom2DefLoader(ArtifactLoader):

    def __call__(self, ver: str):
        artifact = self.use_artifact(ver)
        all_table = cast(wandb.Table, artifact.get("all"))
        return [(row[0], row[1]) for _, row in all_table.iterrows()]

    @property
    def name(self):
        return "wisdom2def"


class Wisdom2EgLoader(ArtifactLoader):

    def __call__(self, ver: str):
        artifact = self.use_artifact(ver)
        all_table = cast(wandb.Table, artifact.get("all"))
        return [(row[0], row[1]) for _, row in all_table.iterrows()]

    @property
    def name(self):
        return "wisdom2eg"


# --- model loaders --- #
class RDLoader(ArtifactLoader, ABC):

    def __init__(self, run: Run, device: torch.device):
        super().__init__(run)
        self.device = device

    def __call__(self, ver: str) -> Tuple[RD, BertTokenizerFast, List[str]]:
        artifact = self.use_artifact(ver)
        wisdoms = WisdomsLoader(self.run)(artifact.metadata['wisdoms_ver'])
        bert = artifact.metadata['bert']
        k = artifact.metadata['k']
        lr = artifact.metadata['lr']
        artifact_path = artifact.download()
        rd_bin_path = path.join(artifact_path, "rd.bin")
        tok_dir_path = path.join(artifact_path, "tokenizer")
        bert_mlm = AutoModelForMaskedLM.from_config(AutoConfig.from_pretrained(bert))  # just the skeleton
        tokenizer = BertTokenizerFast.from_pretrained(tok_dir_path)
        rd = self.rd(bert_mlm, tokenizer, k, lr, wisdoms)
        rd.load_state_dict(torch.load(rd_bin_path))
        return rd, tokenizer, wisdoms

    def rd(self, bert_mlm: BertForMaskedLM, tokenizer: BertTokenizerFast, k: int, lr: float, wisdoms: List[str]) -> RD:
        raise NotImplementedError


class RDAlphaLoader(RDLoader):

    def rd(self, bert_mlm: BertForMaskedLM, tokenizer: BertTokenizerFast, k: int, lr: float, wisdoms: List[str]) -> RD:
        wisdom2subwords = Wisdom2SubwordsTensor(tokenizer, k, self.device)(wisdoms)
        rd = RDAlpha(bert_mlm, wisdom2subwords, k, lr, self.device)
        return rd

    @property
    def name(self) -> str:
        return "rd_alpha"


class RDBetaLoader(RDLoader):

    def rd(self, bert_mlm: BertForMaskedLM, tokenizer: BertTokenizerFast, k: int, lr: float, wisdoms: List[str]) -> RD:
        wisdom2subwords = Wisdom2SubwordsTensor(tokenizer, k, self.device)(wisdoms)
        bert_mlm.resize_token_embeddings(len(tokenizer))
        wiskeys = WisKeysTensor(tokenizer, self.device)(wisdoms)
        rd = RDBeta(bert_mlm, wisdom2subwords, wiskeys, k, lr, self.device)
        return rd

    @property
    def name(self) -> str:
        return "rd_beta"


# --- model builders  --- #
class RDBuilder:

    def __init__(self, model: str, ver: str):
        self.model = model
        self.ver = ver

    def __call__(self, rd: RD, tokenizer: BertTokenizerFast, config: dict) -> wandb.Artifact:
        # first, save them
        torch.save(rd.state_dict(), self.rd_bin_path)  # saving stat_dict only
        tokenizer.save_pretrained(self.tok_dir_path)  # save the tokenizer as well
        artifact = wandb.Artifact(f"{self.model}:{self.ver}", metadata=config, type="model")
        artifact.add_file(self.rd_bin_path)
        artifact.add_dir(self.tok_dir_path, name="tokenizer")
        return artifact

    @property
    def artifact_dir_path(self) -> str:
        dir_path = os.path.join(ARTIFACTS_DIR, self.name)
        os.makedirs(dir_path, exist_ok=True)
        return dir_path

    @property
    def rd_bin_path(self) -> str:
        return os.path.join(self.artifact_dir_path, "rd.bin")

    @property
    def tok_dir_path(self) -> str:
        tok_dir_path = os.path.join(self.artifact_dir_path, "tokenizer")
        os.makedirs(tok_dir_path, exist_ok=True)
        return tok_dir_path

    @property
    def name(self) -> str:
        raise NotImplementedError
