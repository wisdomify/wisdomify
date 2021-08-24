import json
import os
import shutil
from os import mkdir, path

from wisdomify.paths import DATA_DIR, LIGHTNING_LOGS_DIR, PROJECT_ROOT


def save_config(config: dict, save_path: str):
    with open(save_path, 'w') as fp:
        json.dump(config, fp, indent=4)


def save_wisdomify_package():
    zip_loc = PROJECT_ROOT
    zip_dest = path.join(PROJECT_ROOT, 'wisdomify')
    shutil.make_archive(base_dir=zip_loc, root_dir=zip_loc, format='zip', base_name=zip_dest)


def build_archive_model(checkpoint_file_name: str, version: int, max_length: int):
    torch_serve_model_dir = path.join(DATA_DIR, "torchServeModels")
    if not path.isdir(torch_serve_model_dir):
        mkdir(torch_serve_model_dir)

    dict_config = {
        "model_name": "wisdomifier",
        "bert_model": "beomi/kcbert-base",
        "desc": "proverb_from_sentence",
        "model_mode": "torchscript",
        "checkpoint_filename": checkpoint_file_name,
        "version": version,
        "max_length": max_length
    }
    current_version_dir = path.join(torch_serve_model_dir, f"version_{version}")
    if not path.isdir(current_version_dir):
        mkdir(current_version_dir)

    save_config(dict_config, path.join(current_version_dir, "./build_setting.json"))

    package_zip_dir = path.join(PROJECT_ROOT, 'wisdomify.zip')
    if path.isfile(package_zip_dir):
        os.remove(package_zip_dir)

    save_wisdomify_package()

    os.system(f"""
    torch-model-archiver \
    --model-name wisdomifier \
    --version {version} \
    --serialized-file {LIGHTNING_LOGS_DIR}/version_{version}/checkpoints/{checkpoint_file_name} \
    --handler {PROJECT_ROOT}/wisdomify/torchserve_handler.py \
    --export-path {current_version_dir} \
    --extra-files "{path.join(current_version_dir, "./build_setting.json")},{PROJECT_ROOT}/wisdomify.zip"
    """)
