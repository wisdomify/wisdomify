import shutil

from os import rmdir
from os.path import join

from pytorch_lightning.loggers import TensorBoardLogger


class TrainerFileSupporter:
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

        # Input version is smaller than next version on data directory.
        # The input version directory should be removed and the directory after training should be moved to the version.

        # If user does not want to make the file be overwritten,
        # then the whole training will be terminated with warning.
        elif self.version < self.logger.version:
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
