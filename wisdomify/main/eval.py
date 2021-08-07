import torch
import argparse
from wisdomify.datasets import WisdomDataModule
from wisdomify.loaders import load_conf
from wisdomify.metrics import RDMetric
from wisdomify.models import Wisdomifier


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("--ver", type=str,
                        default="0")

    args = parser.parse_args()
    ver: str = args.ver
    conf = load_conf()
    data_version: str = conf['versions'][ver]['data_version']
    batch_size: int = conf['versions'][ver]['batch_size']
    shuffle: bool = conf['versions'][ver]['shuffle']
    k: int = conf['versions'][ver]['k']
    num_workers: int = conf['versions'][ver]['num_workers']

    wisdomifier = Wisdomifier.from_pretrained(ver, device)

    data_module = WisdomDataModule(data_version=data_version,
                                   batch_size=batch_size,
                                   shuffle=shuffle,
                                   num_workers=num_workers)
    data_module.prepare_data()
    data_module.setup()
    test_loader = data_module.test_dataloader()

    # the metric
    rd_metric = RDMetric()
    for idx, batch in enumerate(test_loader):
        X, y = batch
        S_word_probs = wisdomifier.rd.S_word_probs(X)
        rd_metric.update(preds=S_word_probs, targets=y)
        print("batch:{}".format(idx), rd_metric.compute())
        # top1 , top10, top100 -- should be 1.0?
    median, var, top1, top10, top100 = rd_metric.compute()
    print("### final ###")
    print("data_version:", data_version)
    print("median:", median)
    print("var:", var)
    print("top1:", top1)
    print("top10:", top10)
    print("top100:", top100)


if __name__ == '__main__':
    main()
