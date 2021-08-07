import torch
import argparse
import yaml
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoConfig, AutoTokenizer
from wisdomify.builders import build_vocab2subwords, build_X, build_y
from wisdomify.datasets import WisdomDataset
from wisdomify.loaders import load_conf, WisdomDataLoader
from wisdomify.metrics import RDMetric
from wisdomify.models import RD
from wisdomify.paths import WISDOMIFIER_V_0_CKPT, WISDOMIFIER_V_0_HPARAMS_YAML
from wisdomify.vocab import VOCAB


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("--ver", type=str,
                        default="0")

    args = parser.parse_args()
    ver: str = args.ver
    conf = load_conf()
    bert_model: str = conf['versions'][ver]['bert_model']
    data_version: str = conf['versions'][ver]['data_version']
    batch_size: int = conf['versions'][ver]['batch_size']
    shuffle: bool = conf['versions'][ver]['shuffle']
    num_workers: int = conf['versions'][ver]['num_workers']

    if ver == "0":
        wisdomifier_path = WISDOMIFIER_V_0_CKPT
        with open(WISDOMIFIER_V_0_HPARAMS_YAML, 'r') as fh:
            wisdomifier_hparams = yaml.safe_load(fh)
        k = wisdomifier_hparams['k']
    else:
        # this version is not supported yet.
        raise NotImplementedError("Invalid version provided".format(ver))

    wisdomdata = WisdomDataLoader(data_version).__call__()
    if ver == "0":
        wisdom2sent = wisdomdata.wisdom2def
    else:
        raise NotImplementedError

    bert_mlm = AutoModelForMaskedLM.from_config(AutoConfig.from_pretrained(bert_model))
    tokenizer = AutoTokenizer.from_pretrained(bert_model)
    vocab2subwords = build_vocab2subwords(tokenizer, k, VOCAB).to(device)
    X = build_X(wisdom2sent, tokenizer, k).to(device)
    y = build_y(wisdom2sent, VOCAB).to(device)
    rd = RD.load_from_checkpoint(wisdomifier_path, bert_mlm=bert_mlm, vocab2subwords=vocab2subwords)
    rd.eval()  # otherwise, the model will output different results with the same inputs
    rd = rd.to(device)  # otherwise, you can not run the inference process on GPUs.
    dataset = WisdomDataset(X, y)
    dataloader = DataLoader(dataset, batch_size, shuffle, num_workers=num_workers)

    # the metric
    rd_metric = RDMetric()

    for idx, batch in enumerate(dataloader):
        X, y = batch
        S_word_probs = rd.S_word_probs(X)
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
