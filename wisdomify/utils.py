import shutil
from os import path
from os.path import join
from pytorch_lightning.loggers import TensorBoardLogger
from wisdomify.builders import Wisdom2SubWordsBuilder, XBuilder, XWithWisdomMaskBuilder, YBuilder
from wisdomify.paths import CONF_JSON
from wisdomify.rds import RD, RDAlpha, RDBeta
from wisdomify.datasets import WisdomDataModule
from wisdomify.paths import WISDOMIFIER_CKPT
from transformers import BertTokenizer, AutoModelForMaskedLM, AutoTokenizer, AutoConfig
import torch
import json


# --- loaders --- #
def load_conf_json() -> dict:
    with open(CONF_JSON, 'r') as fh:
        return json.loads(fh.read())


# --- an experiment --- #
class Experiment:
    def __init__(self, config: dict, rd: RD, tokenizer: BertTokenizer, data_module: WisdomDataModule):
        self.config = config
        self.rd = rd
        self.tokenizer = tokenizer
        self.data_module = data_module

    @classmethod
    def from_pretrained(cls, ver: str, device: torch.device) -> 'Experiment':
        conf_json = load_conf_json()
        wisdomifier_path = WISDOMIFIER_CKPT.format(ver=ver)
        config = conf_json['versions'][ver]
        bert_model = config['bert_model']
        k = config['k']
        wisdoms = config['wisdoms']
        rd_model = config['rd_model']
        X_mode = config['X_mode']
        y_mode = config['y_mode']
        bert_mlm = AutoModelForMaskedLM.from_config(AutoConfig.from_pretrained(bert_model))
        tokenizer = AutoTokenizer.from_pretrained(bert_model)
        wisdom2subwords_builder = Wisdom2SubWordsBuilder(tokenizer, k, device)
        wisdom2subwords = wisdom2subwords_builder(wisdoms)
        # --- choose an appropriate rd version --- #
        if rd_model == "RDAlpha":
            rd = RDAlpha.load_from_checkpoint(wisdomifier_path,
                                              bert_mlm=bert_mlm, wisdom2subwords=wisdom2subwords,
                                              device=device)
        elif rd_model == "RDBeta":
            rd = RDBeta.load_from_checkpoint(wisdomifier_path,
                                             bert_mlm=bert_mlm, wisdom2subwords=wisdom2subwords,
                                             device=device)
        else:
            raise NotImplementedError
        data_module = cls.get_data_module(config, X_mode, y_mode, tokenizer, k, device)
        return Experiment(config, rd, tokenizer, data_module)

    @classmethod
    def build(cls, ver: str, device: torch.device) -> 'Experiment':
        """
        for training.
        """
        conf_json = load_conf_json()
        config = conf_json['versions'][ver]
        bert_model = config['bert_model']
        wisdoms = config['wisdoms']
        X_mode = config['X_mode']
        y_mode = config['y_mode']
        k = config['k']
        lr = config['lr']
        rd_model = config['rd_model']
        # --- load a bert_mlm model --- #
        bert_mlm = AutoModelForMaskedLM.from_pretrained(bert_model)
        tokenizer = AutoTokenizer.from_pretrained(bert_model)
        wisdom2subwords_builder = Wisdom2SubWordsBuilder(tokenizer, k, device)
        wisdom2subwords = wisdom2subwords_builder(wisdoms)

        # --- choose an appropriate rd version --- #
        if rd_model == "RDAlpha":
            rd = RDAlpha(bert_mlm, wisdom2subwords, k, lr, device)
        elif rd_model == "RDBeta":
            rd = RDBeta(bert_mlm, wisdom2subwords, k, lr, device)
        else:
            raise NotImplementedError
        # --- load a data module --- #
        data_module = cls.get_data_module(config, X_mode, y_mode, tokenizer, k, device)
        return Experiment(config, rd, tokenizer, data_module)

    @classmethod
    def get_data_module(cls, config: dict, X_mode: str, y_mode: str,
                        tokenizer: BertTokenizer, k: int, device: torch.optim) -> WisdomDataModule:
        X_builder = cls.get_X_builder(X_mode, tokenizer, k, device)
        y_builder = cls.get_y_builder(y_mode, device)
        return WisdomDataModule(config, X_builder, y_builder, tokenizer, device)

    @staticmethod
    def get_X_builder(X_mode: str, tokenizer: BertTokenizer, k: int, device: torch.device) -> XBuilder:
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


class TrainerFileSupport:
    def __init__(self, version: str, logger: TensorBoardLogger, data_dir: str):
        self.version = int(version)
        self.logger = logger
        self.data_dir = data_dir

        self._is_overwrite = False
        self._is_mv_dir = False

        self.check_overwrite_and_mv_dir()

    def check_overwrite_and_mv_dir(self):
        # Input version is just next version on data directory.
        if self.version == self.logger.version:
            self._is_overwrite = False
            self._is_mv_dir = False

        # Input version is smaller than next version on data directory.
        # The input version directory should be removed and the directory after training should be moved to the version.

        # If user does not want to make the file be overwritten,
        # then the whole training will be terminated with warning.
        elif self.version < self.logger.version:
            _is_path_exist = path.exists(join(self.data_dir, 'lightning_logs', f'version_{self.version}'))

            # now checks cases such as
            # 0, 1, 2, 8 and trying to add version_6
            # this case overwrite does not have to be checked and the new directory just need to be moved.
            if not _is_path_exist:
                self._is_overwrite = False
                self._is_mv_dir = True

            else:
                _user_input_overwrite = ""

                while _user_input_overwrite not in ['y', 'n']:
                    _user_input_overwrite = str(
                        input(f"Your version setting (version: {self.version}) "
                              f"conflicts with local data directory.\n"
                              f"Do you want to overwrite "
                              f"{self.data_dir}/lightning_logs/version_{self.version}? (Y/n):")
                    ).lower()

                if _user_input_overwrite == 'y':
                    self._is_overwrite = True
                    self._is_mv_dir = True

                # Training terminates.
                # Let user to know input version is wrong.
                elif _user_input_overwrite == 'n':
                    raise FileExistsError(f"Check your version setting on run command. Your input was {self.version}.")

        # Input version is larger than next version on data directory.
        # The directory after training should be moved.
        elif self.version > self.logger.version:
            self._is_mv_dir = True

    def save_training_result(self):
        if self._is_overwrite:
            shutil.rmtree(join(self.data_dir, 'lightning_logs', f'version_{self.version}'))

        if self._is_mv_dir:
            shutil.move(
                src=join(self.data_dir, 'lightning_logs', f'version_{self.logger.version}'),
                dst=join(self.data_dir, 'lightning_logs', f'version_{self.version}')
            )
