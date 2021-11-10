import random
from typing import List, Tuple
import torch
import numpy as np
from wisdomify.downloaders import WisdomsDownloader, RDAlphaDownloader, RDBetaDownloader, RDBetaAttnDownloader
from wisdomify.models import RD, RDAlpha, RDBeta, RDBetaAttention
from wisdomify.datamodules import Wisdom2DefDataModule, Wisdom2EgDataModule, WisdomifyDataModule
from wisdomify.loaders import load_conf
from wisdomify.builders import Wisdom2SubwordsBuilder, WisKeysBuilder
from transformers import BertTokenizerFast, BertForMaskedLM
from wandb.wandb_run import Run


# --- an experiment --- #
class Experiment:
    def __init__(self, ver: str, config: dict, rd: RD,
                 tokenizer: BertTokenizerFast, datamodule: WisdomifyDataModule):
        self.ver = ver
        self.config = config
        self.rd = rd
        self.tokenizer = tokenizer
        self.datamodule = datamodule

    @classmethod
    def load(cls, model: str, ver: str, run: Run, device: torch.device) -> 'Experiment':
        """
        load an experiment that has already been done.
        """
        config = load_conf()[model][ver]
        seed = config["seed"]
        wisdoms_ver = config["wisdoms_ver"]
        train_type = config["train_type"]
        # --- for reproducibility --- #
        cls.fix_seeds(seed)
        wisdoms = WisdomsDownloader(run)(wisdoms_ver)
        # --- choose an appropriate rd version --- #
        if model == "rd_alpha":
            rd, tokenizer = RDAlphaDownloader(run, wisdoms, device)(ver)
        elif model == "rd_beta":
            rd, tokenizer = RDBetaDownloader(run, wisdoms, device)(ver)
        elif model == "rd_beta_attn":
            rd, tokenizer = RDBetaAttnDownloader(run, wisdoms, device)(ver)
        else:
            raise ValueError
        # --- load a datamodule --- #
        if train_type == "wisdom2def":
            datamodule = Wisdom2DefDataModule(config, tokenizer, wisdoms, run, device)
        elif train_type == "wisdom2eg":
            datamodule = Wisdom2EgDataModule(config, tokenizer, wisdoms, run, device)
        else:
            raise ValueError

        return Experiment(ver, config, rd, tokenizer, datamodule)

    @classmethod
    def build(cls, model: str, ver: str, run: Run, device: torch.device) -> 'Experiment':
        """
        build an experiment for training a RD.
        """
        # --- access the settings --- #
        config = load_conf()[model][ver]
        seed = config["seed"]
        bert = config["bert"]
        k = config["k"]
        lr = config["lr"]
        wisdoms_ver = config["wisdoms_ver"]
        train_type = config["train_type"]
        # --- this is to ensure reproducibility, although not perfect --- #
        cls.fix_seeds(seed)
        # --- get the pretrained model with the pretrained tokenizer --- #
        bert_mlm = BertForMaskedLM.from_pretrained(bert)
        tokenizer = BertTokenizerFast.from_pretrained(bert)
        # --- wisdom-related data --- #
        wisdoms = WisdomsDownloader(run)(wisdoms_ver)
        wisdom2subwords = Wisdom2SubwordsBuilder(tokenizer, k, device)(wisdoms)
        # --- choose an appropriate rd version --- #
        if model == "rd_alpha":
            rd = RDAlpha(bert_mlm, wisdom2subwords, k, lr, device)
        elif model == "rd_beta":
            tokenizer.add_tokens(wisdoms)
            bert_mlm.resize_token_embeddings(len(tokenizer))
            wiskeys = WisKeysBuilder(tokenizer, device)(wisdoms)
            rd = RDBeta(bert_mlm, wisdom2subwords, wiskeys, k, lr, device)
        elif model == "rd_beta_attn":
            tokenizer.add_tokens(wisdoms)
            bert_mlm.resize_token_embeddings(len(tokenizer))
            wiskeys = WisKeysBuilder(tokenizer, device)(wisdoms)
            rd = RDBetaAttention(bert_mlm, wisdom2subwords, wiskeys, k, lr, device)
        else:
            raise NotImplementedError
        # --- load a datamodule --- #
        if train_type == "wisdom2def":
            datamodule = Wisdom2DefDataModule(config, tokenizer, wisdoms, run, device)
        elif train_type == "wisdom2eg":
            datamodule = Wisdom2EgDataModule(config, tokenizer, wisdoms, run, device)
        else:
            raise ValueError
        # --- hook wandb onto rd to watch gradients --- #
        run.watch(rd)
        # --- package them into an experiment instance, and return --- #
        return Experiment(ver, config, rd, tokenizer, datamodule)

    @staticmethod
    def fix_seeds(seed: int):
        """
        https://pytorch.org/docs/stable/notes/randomness.html
        """
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)


class Wisdomifier:
    def __init__(self, exp: Experiment):
        self.exp = exp

    def __call__(self, sents: List[str]) -> List[List[Tuple[str, float]]]:
        # get the X
        wisdom2sent = [("", desc) for desc in sents]
        x_builder, _ = self.exp.datamodule.tensor_builders()
        X = x_builder(wisdom2sent)
        # get H_all for this.
        P_wisdom = self.exp.rd.P_wisdom(X)
        results = list()
        for S_word_prob in P_wisdom.tolist():
            wisdom2prob = [
                (wisdom, prob)
                for wisdom, prob in zip(self.exp.datamodule.wisdoms, S_word_prob)
            ]
            # sort and append
            results.append(sorted(wisdom2prob, key=lambda x: x[1], reverse=True))

        return results
