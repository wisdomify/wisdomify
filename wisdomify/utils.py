import shutil
import wandb

from os import path
from os.path import join

from pytorch_lightning.loggers import TensorBoardLogger


def get_wandb_artifact(name: str, ver: str, dtype: str):
    with wandb.init() as run:
        artifact = run.use_artifact(f'wisdomify/wisdomify/{name}:{ver}', type=dtype)
        artifact_dir = artifact.download()

        return artifact_dir


def upload_wandb_artifact(job_type: str, ):
    with wandb.init(project='wisdomify',
                    entity='wisdomify',
                    job_type='remove empty rows') as run:
        # 새롭게 저장할 아티펙트
        processed_data = wandb.Artifact(
            artifact_name, type="dataset",
            description="Unintended rows, empty information is removed.",
        )

        # ✔️ declare which artifact we'll be using (기존에 있던 데이터셋)
        raw_data_artifact = run.use_artifact(f'{artifact_name}:latest')

        # 📥 if need be, download the artifact
        raw_dataset = raw_data_artifact.download()

        for split in ["training", "validation", "test"]:
            raw_split = _read(raw_dataset, split)
            processed_dataset = _preprocess(raw_split)

            with processed_data.new_file(split + ".tsv", mode="wb") as file:
                processed_dataset.to_csv(file, sep='\t', index=False)

        run.log_artifact(processed_data)
    pass


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
