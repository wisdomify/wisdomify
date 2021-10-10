
from wisdomify.models import RD, RDAlpha, RDBeta
from wisdomify.data import WisdomDataModule
from wisdomify.paths import WISDOMIFIER_CKPT, WISDOMIFIER_TOKENIZER_DIR
from wisdomify.loaders import load_conf
from wisdomify.builders import (
    Wisdom2SubWordsBuilder, Wisdom2EgXBuilder,
    YBuilder, WisKeysBuilder, Wisdom2DefXBuilder
)
from transformers import AutoTokenizer, AutoConfig, AutoModelForMaskedLM, BertTokenizerFast
import torch


# --- an experiment --- #
class Experiment:
    def __init__(self, ver: str, config: dict, rd: RD, tokenizer: BertTokenizerFast, datamodule: WisdomDataModule):
        self.ver = ver
        self.config = config
        self.rd = rd
        self.tokenizer = tokenizer
        self.datamodule = datamodule

    @classmethod
    def load(cls, ver: str, device: torch.device) -> 'Experiment':
        """
        load an experiment that has already been done.
        """
        conf_json = load_conf()
        config = conf_json['versions'][ver]
        bert_model = config['bert_model']
        k = config['k']
        wisdoms = config['wisdoms']
        rd_model = config['rd_model']
        data_name = config['data_name']
        bert_mlm = AutoModelForMaskedLM.from_config(AutoConfig.from_pretrained(bert_model))  # loading the skeleton
        tokenizer = AutoTokenizer.from_pretrained(WISDOMIFIER_TOKENIZER_DIR.format(ver=ver))  # from local
        wisdom2subwords = Wisdom2SubWordsBuilder(tokenizer, k, device)(wisdoms)
        # --- choose an appropriate rd version --- #
        if rd_model == "RDAlpha":
            rd = RDAlpha.load_from_checkpoint(WISDOMIFIER_CKPT.format(ver=ver),
                                              bert_mlm=bert_mlm, wisdom2subwords=wisdom2subwords,
                                              device=device)
        elif rd_model == "RDBeta":
            tokenizer.add_tokens(wisdoms)
            bert_mlm.resize_token_embeddings(len(tokenizer))
            wiskeys = WisKeysBuilder(tokenizer, device)(wisdoms)
            rd = RDBeta.load_from_checkpoint(WISDOMIFIER_CKPT.format(ver=ver),
                                             bert_mlm=bert_mlm, wisdom2subwords=wisdom2subwords,
                                             wiskeys=wiskeys, device=device)
        else:
            raise NotImplementedError
        data_module = cls.build_datamodule(config, data_name, tokenizer, k, device)
        return Experiment(ver, config, rd, tokenizer, data_module)

    @classmethod
    def build(cls, ver: str, device: torch.device) -> 'Experiment':
        """
        build an experiment for training a RD.
        """
        conf_json = load_conf()
        config = conf_json['versions'][ver]
        bert_model = config['bert_model']
        wisdoms = config['wisdoms']
        k = config['k']
        lr = config['lr']
        rd_model = config['rd_model']
        data_name = config['data_name']
        # --- load a bert_mlm model --- #
        bert_mlm = AutoModelForMaskedLM.from_pretrained(bert_model)
        tokenizer = AutoTokenizer.from_pretrained(bert_model)
        wisdom2subwords = Wisdom2SubWordsBuilder(tokenizer, k, device)(wisdoms)
        # --- choose an appropriate rd version --- #
        if rd_model == "RDAlpha":
            rd = RDAlpha(bert_mlm, wisdom2subwords, k, lr, device)
        elif rd_model == "RDBeta":
            tokenizer.add_tokens(wisdoms)
            bert_mlm.resize_token_embeddings(len(tokenizer))
            wiskeys = WisKeysBuilder(tokenizer, device)(wisdoms)
            rd = RDBeta(bert_mlm, wisdom2subwords, wiskeys, k, lr, device)
        else:
            raise NotImplementedError
        # --- load a data module --- #
        data_module = cls.build_datamodule(config, data_name, tokenizer, k, device)
        return Experiment(ver, config, rd, tokenizer, data_module)

    @classmethod
    def build_datamodule(cls, config: dict, data_name: str,
                         tokenizer: BertTokenizerFast, k: int, device: torch.optim) -> WisdomDataModule:
        if data_name == "definition":
            X_builder = Wisdom2DefXBuilder(tokenizer, k, device)
        elif data_name == "example":
            X_builder = Wisdom2EgXBuilder(tokenizer, k, device)
        else:
            raise ValueError(f"Invalid data_name: {data_name}")
        y_builder = YBuilder(device)
        return WisdomDataModule(config, X_builder, y_builder, tokenizer, device)
