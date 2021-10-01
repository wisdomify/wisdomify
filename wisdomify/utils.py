import os
import shutil
import wandb

from os import path
from os.path import join

from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger


def get_wandb_artifact(name: str, ver: str, dtype: str):
    with wandb.init() as run:
        artifact = run.use_artifact(f'wisdomify/wisdomify/{name}:{ver}', type=dtype)
        artifact_dir = artifact.download()

        return artifact_dir


class WandBSupport:
    def __init__(self,
                 job_type: str):
        # initialise wandb connection object.
        self.wandb_obj = wandb.init(
            entity='wisdomify',
            project='wisdomfiy',
            job_type=job_type
        )

    def _get_artifact(self,
                      name: str,
                      dtype: str,
                      ver: str = 'latest'):
        # this function returns existing artifact object.
        return self.wandb_obj.use_artifact(f"{name}:{ver}", type=dtype)

    def download_artifact(self,
                          name: str,
                          dtype: str,
                          ver: str = 'latest'):
        # this function returns download path and downloaded files
        """
        :return:
        {
            'downloaded_dir': downloaded directory
            'downloaded_files': file names downloaded
        }
        """
        dl_dir = self._get_artifact(name=name, ver=ver, dtype=dtype).download()
        return {
            'downloaded_dir': dl_dir,
            'downloaded_files': list(filter(lambda s: s != '.DS_Store', os.listdir(dl_dir)))
        }

    @staticmethod
    def create_artifact(name: str,
                        dtype: str,
                        desc: str,
                        meta: str = None):
        # this function creates and returns new artifact
        return wandb.Artifact(name, type=dtype, description=desc, metadata=meta)

    def write_artifact(self,
                       artifact: wandb.Artifact,
                       file_name: str,
                       scripts):
        # This function was intended to make user to write file on the desired artifact.
        # However, file saving script may be different depending on the data type.
        """
        >>> with artifact.new_file(file_name, mode="wb") as file:
        >>>     Write_data_saving_script
        """
        raise NotImplementedError

    def add_artifact(self,
                     artifact: wandb.Artifact,
                     file_path: str):
        # This function is used when user trying to add already saved file directly with file path
        return artifact.add_file(file_path)

    @staticmethod
    def get_model_logger(log_name: str):
        # this function returns model logger
        return WandbLogger(project='wisdomify', entity='wisdomify', name=log_name)

    @staticmethod
    def push() -> None:
        wandb.finish()


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
