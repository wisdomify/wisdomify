from wisdomify.models import RD, RDAlpha, RDBeta
from wisdomify.data import WisdomDataModule
from wisdomify.loaders import load_conf
from wisdomify.builders import (
    Wisdom2SubWordsBuilder, Wisdom2EgXBuilder,
    YBuilder, WisKeysBuilder, Wisdom2DefXBuilder
)
from transformers import BertTokenizerFast
import torch


# --- an experiment --- #
from wisdomify.utils import WandBSupport


class Experiment:
    def __init__(self, ver: str, config: dict, rd: RD, tokenizer: BertTokenizerFast, datamodule: WisdomDataModule):
        self.ver = ver
        self.config = config
        self.rd = rd
        self.tokenizer = tokenizer
        self.datamodule = datamodule

    @classmethod
    def load(cls, ver: str, device: torch.device, wandb_support: WandBSupport) -> 'Experiment':
        """
        load an experiment that has already been done.
        """
        conf_json = load_conf()
        config = conf_json['versions'][ver]
        wisdoms = config['wisdoms']

        model_configs = config['model']
        k = model_configs['k']
        rd_model = model_configs['rd_model']
        data_type = config['wandb']['load']['data_type']
        # --- load a bert_mlm model (from W&B) --- #
        bert_mlm = wandb_support.models.get_mlm()
        tokenizer = wandb_support.models.get_tokenizer()

        wisdom2subwords = Wisdom2SubWordsBuilder(tokenizer, k, device)(wisdoms)

        # --- RD .ckpt download (from W&B) --- #
        rd_dl_path = wandb_support.models.get_rd_ckpt_path()

        # --- choose an appropriate rd version --- #
        if rd_model == "RDAlpha":
            rd = RDAlpha.load_from_checkpoint(rd_dl_path,
                                              bert_mlm=bert_mlm, wisdom2subwords=wisdom2subwords,
                                              device=device)
        elif rd_model == "RDBeta":
            wiskeys = WisKeysBuilder(tokenizer, device)(wisdoms)
            rd = RDBeta.load_from_checkpoint(rd_dl_path,
                                             bert_mlm=bert_mlm, wisdom2subwords=wisdom2subwords,
                                             wiskeys=wiskeys, device=device)
        else:
            raise NotImplementedError

        data_module = cls.build_datamodule(config, data_type, tokenizer, k, device, wandb_support)
        return Experiment(ver, config, rd, tokenizer, data_module)

    @classmethod
    def build(cls, ver: str, device: torch.device, wandb_support: WandBSupport) -> 'Experiment':
        """
        build an experiment for training a RD.
        """
        conf_json = load_conf()
        config = conf_json['versions'][ver]
        wisdoms = config['wisdoms']

        model_configs = config['model']
        k = model_configs['k']
        lr = model_configs['lr']
        rd_model = model_configs['rd_model']
        data_type = config['wandb']['load']['data_type']

        # --- load a bert_mlm model (from W&B) --- #
        bert_mlm = wandb_support.models.get_mlm()
        tokenizer = wandb_support.models.get_tokenizer()

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
        data_module = cls.build_datamodule(config, data_type, tokenizer, k, device, wandb_support)
        return Experiment(ver, config, rd, tokenizer, data_module)

    @classmethod
    def build_datamodule(cls, config: dict, data_name: str,
                         tokenizer: BertTokenizerFast, k: int, device: torch.optim,
                         wandb_support) -> WisdomDataModule:
        if data_name == "definition":
            X_builder = Wisdom2DefXBuilder(tokenizer, k, device)
        elif data_name == "example":
            X_builder = Wisdom2EgXBuilder(tokenizer, k, device)
        else:
            raise ValueError(f"Invalid data_name: {data_name}")
        y_builder = YBuilder(device)
        return WisdomDataModule(config, X_builder, y_builder, tokenizer, device, wandb_support)
