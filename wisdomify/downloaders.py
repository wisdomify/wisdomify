"""
A Downloader downloads table(s) from wandb.
"""
import torch
import wandb
import types
from os import path
from typing import Tuple, cast, List
from transformers import BertTokenizerFast, AutoModelForMaskedLM, AutoConfig
from wandb.sdk.wandb_run import Run
from wisdomify.builders import Wisdom2SubwordsBuilder, WisKeysBuilder
from wisdomify.models import RD, RDAlpha, RDBeta
from wisdomify.paths import ARTIFACTS_DIR


class Downloader:
    def __init__(self, run: Run):
        self.run = run

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def use_artifact(self, name: str) -> wandb.Artifact:
        artifact = self.run.use_artifact(name)
        # https://stackoverflow.com/a/42154067
        # terribly messy, but I must do this, if I were to save artifacts to local
        artifact._default_root = types.MethodType(lambda a, b=True: path.join(ARTIFACTS_DIR, a.name), artifact)
        return artifact


class WisdomsDownloader(Downloader):

    def __call__(self, ver: str) -> List[str]:
        artifact = self.use_artifact(f"wisdoms:{ver}")
        table = cast(wandb.Table, artifact.get("wisdoms"))
        return [
            row[0]
            for _, row in table.iterrows()
        ]


class Wisdom2QueryDownloader(Downloader):

    def __call__(self, ver: str) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        artifact = self.use_artifact(f"wisdom2query:{ver}")
        val_table = cast(wandb.Table, artifact.get("val"))
        test_table = cast(wandb.Table, artifact.get("test"))
        val = [(row[0], row[1]) for _, row in val_table.iterrows()]
        test = [(row[0], row[1]) for _, row in test_table.iterrows()]
        return val, test


class Wisdom2DefDownloader(Downloader):

    def __call__(self, ver: str) -> List[Tuple[str, str]]:
        artifact = self.use_artifact(f"wisdom2def:{ver}")
        all_table = cast(wandb.Table, artifact.get("all"))
        return [(row[0], row[1]) for _, row in all_table.iterrows()]


class Wisdom2EgDownloader(Downloader):

    def __call__(self, ver: str) -> List[Tuple[str, str]]:
        artifact = self.use_artifact(f"wisdom2eg:{ver}")
        all_table = cast(wandb.Table, artifact.get("all"))
        return [(row[0], row[1]) for _, row in all_table.iterrows()]


# --- should be used with an experiment instance --- #
class RDDownloader(Downloader):

    def __init__(self, run: Run, wisdoms: List[str], device: torch.device):
        super().__init__(run)
        self.wisdoms = wisdoms
        self.device = device

    def __call__(self, ver: str) -> Tuple[RD, BertTokenizerFast]:
        raise NotImplementedError


class RDAlphaDownloader(RDDownloader):

    def __call__(self, ver: str) -> Tuple[RD, BertTokenizerFast]:
        artifact = self.use_artifact(f"rd_alpha:{ver}")
        artifact_path = artifact.download()
        rd_bin_path = path.join(artifact_path, "rd.bin")
        tok_dir_path = path.join(artifact_path, "tokenizer")
        bert = artifact.metadata['bert']
        k = artifact.metadata['k']
        lr = artifact.metadata['lr']
        bert_mlm = AutoModelForMaskedLM.from_config(AutoConfig.from_pretrained(bert))  # just the skeleton
        tokenizer = BertTokenizerFast.from_pretrained(tok_dir_path)
        wisdom2subwords = Wisdom2SubwordsBuilder(tokenizer, k, self.device)(self.wisdoms)
        rd = RDAlpha(bert_mlm, wisdom2subwords, k, lr, self.device)
        rd.load_state_dict(torch.load(rd_bin_path))
        return rd, tokenizer


class RDBetaDownloader(RDDownloader):

    def __call__(self, ver: str) -> Tuple[RD, BertTokenizerFast]:
        artifact = self.use_artifact(f"rd_beta:{ver}")
        artifact_path = artifact.download()
        rd_bin_path = path.join(artifact_path, "rd.bin")
        tok_dir_path = path.join(artifact_path, "tokenizer")
        bert = artifact.metadata['bert']
        k = artifact.metadata['k']
        lr = artifact.metadata['lr']
        bert_mlm = AutoModelForMaskedLM.from_config(AutoConfig.from_pretrained(bert))  # just the skeleton
        tokenizer = BertTokenizerFast.from_pretrained(tok_dir_path)
        wisdom2subwords = Wisdom2SubwordsBuilder(tokenizer, k, self.device)(self.wisdoms)
        bert_mlm.resize_token_embeddings(len(tokenizer))
        wiskeys = WisKeysBuilder(tokenizer, self.device)(self.wisdoms)
        rd = RDBeta(bert_mlm, wisdom2subwords, wiskeys, k, lr, self.device)
        rd.load_state_dict(torch.load(rd_bin_path))
        return rd, tokenizer
