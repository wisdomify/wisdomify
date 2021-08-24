import argparse
from torch.utils.data import Dataset, DataLoader, random_split
from wisdomify.dataloaders import WisdomDataModule
from wisdomify.loaders import load_conf
from wisdomify.paths import WISDOMDATA_VER_1_DIR
from os import path
import csv
import random

TRAIN_RATIO = 0.8
random.seed(318)


def main():
    wisdom2eg_tsv_path = path.join(WISDOMDATA_VER_1_DIR, "wisdom2eg.tsv")
    wisdom2sent = WisdomDataModule.load_wisdom2sent(wisdom2eg_tsv_path)
    n_total_data = len(wisdom2sent)
    n_train = int(n_total_data * TRAIN_RATIO)
    random.shuffle(wisdom2sent)
    wisdom2sent_train = wisdom2sent[:n_train]
    wisdom2sent_test = wisdom2sent[n_train:]

    # save them
    with open(path.join(WISDOMDATA_VER_1_DIR, "wisdom2eg_train.tsv"), 'w') as fh:
        tsv_writer = csv.writer(fh, delimiter="\t")
        tsv_writer.writerow(["wisdom", "eg"])
        for row in wisdom2sent_train:
            tsv_writer.writerow(row)

    with open(path.join(WISDOMDATA_VER_1_DIR, "wisdom2eg_test.tsv"), 'w') as fh:
        tsv_writer = csv.writer(fh, delimiter="\t")
        tsv_writer.writerow(["wisdom", "eg"])
        for row in wisdom2sent_test:
            tsv_writer.writerow(row)


if __name__ == '__main__':
    main()
