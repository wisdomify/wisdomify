from wisdomify.builders import Wisdom2SubWordsBuilder, XBuilder, XWithWisdomMaskBuilder, YBuilder, WisKeysBuilder
from wisdomify.models import RD, RDAlpha, RDBeta
from wisdomify.data import WisdomDataModule
from wisdomify.paths import WISDOMIFIER_CKPT, WISDOMIFIER_TOKENIZER_DIR
from wisdomify.loaders import load_conf
from transformers import AutoTokenizer, AutoConfig, AutoModelForMaskedLM, BertTokenizerFast
import torch


# --- an experiment --- #
from wisdomify.utils import WandBSupport


class Experiment:
    def __init__(self, ver: str, config: dict, rd: RD, tokenizer: BertTokenizerFast, data_module: WisdomDataModule):
        self.ver = ver
        self.config = config
        self.rd = rd
        self.tokenizer = tokenizer
        self.data_module = data_module

    @classmethod
    def load(cls, ver: str, device: torch.device, wandb_support: WandBSupport) -> 'Experiment':
        """
        load an experiment that has already been done.
        """
        conf_json = load_conf()
        config = conf_json['versions'][ver]
        wisdoms = config['wisdoms']

        model_configs = config['model']
        X_mode = model_configs['X_mode']
        y_mode = model_configs['y_mode']
        k = model_configs['k']
        rd_model = model_configs['rd_model']

        # --- load a bert_mlm model (from W&B) --- #
        bert_mlm = wandb_support.models.get_mlm()
        tokenizer = wandb_support.models.get_tokenizer()

        wisdom2subwords = Wisdom2SubWordsBuilder(tokenizer, k, device)(wisdoms)

        # --- RD .ckpt download (from W&B) --- #
        rd_dl_path = wandb_support.models.get_rd_ckpt()

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

        data_module = cls.get_data_module(config, X_mode, y_mode, tokenizer, k, device, wandb_support)

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
        X_mode = model_configs['X_mode']
        y_mode = model_configs['y_mode']
        k = model_configs['k']
        lr = model_configs['lr']
        rd_model = model_configs['rd_model']

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
        data_module = cls.get_data_module(config, X_mode, y_mode, tokenizer, k, device, wandb_support)

        return Experiment(ver, config, rd, tokenizer, data_module)

    @classmethod
    def get_data_module(cls, config: dict, X_mode: str, y_mode: str,
                        tokenizer: BertTokenizerFast, k: int, device: torch.optim,
                        wandb_support: WandBSupport) -> WisdomDataModule:
        X_builder = cls.get_X_builder(X_mode, tokenizer, k, device)
        y_builder = cls.get_y_builder(y_mode, device)
        return WisdomDataModule(config, X_builder, y_builder, tokenizer, device, wandb_support)

    @staticmethod
    def get_X_builder(X_mode: str, tokenizer: BertTokenizerFast, k: int, device: torch.device) -> XBuilder:
        # --- choose an appropriate builder for X --- #
        if X_mode == "XBuilder":
            X_builder = XBuilder(tokenizer, k, device)
        elif X_mode == "XWithWisdomMaskBuilder":
            X_builder = XWithWisdomMaskBuilder(tokenizer, k, device)
        else:
            raise ValueError(f"Invalid X_mode: {X_mode}")
        return X_builder

    @staticmethod
    def get_y_builder(y_mode: str, device: torch.device) -> YBuilder:
        # --- choose an appropriate builder for y --- #
        if y_mode == "YBuilder":
            y_builder = YBuilder(device)
        else:
            raise ValueError(f"Invalid y_mode: {y_mode}")
        return y_builder
