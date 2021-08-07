import torch
import argparse
import yaml
from transformers import AutoModelForMaskedLM, AutoConfig, AutoTokenizer
from wisdomify.builders import build_vocab2subwords
from wisdomify.datasets import WisdomDataModule
from wisdomify.loaders import load_conf
from wisdomify.metrics import RDMetric
from wisdomify.models import RD
from wisdomify.paths import WISDOMIFIER_CKPT, WISDOMIFIER_HPARAMS_YAML
from wisdomify.vocab import VOCAB


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("--ver", type=str,
                        default="0")

    args = parser.parse_args()
    ver: str = args.ver

    conf = load_conf()
    vers = conf['versions']
    if ver not in vers.keys():
        raise NotImplementedError("Invalid version provided".format(ver))

    selected_ver = vers[ver]
    bert_model: str = selected_ver['bert_model']
    data_version: str = selected_ver['data_version']
    batch_size: int = selected_ver['batch_size']
    shuffle: bool = selected_ver['shuffle']
    num_workers: int = selected_ver['num_workers']

    wisdomifier_path = WISDOMIFIER_CKPT
    with open(WISDOMIFIER_HPARAMS_YAML, 'r') as fh:
        wisdomifier_hparams = yaml.safe_load(fh)
    k = wisdomifier_hparams['k']

    data_module = WisdomDataModule(data_version=data_version,
                                   batch_size=batch_size,
                                   shuffle=shuffle,
                                   num_workers=num_workers)
    data_module.prepare_data()
    data_module.setup()

    bert_mlm = AutoModelForMaskedLM.from_config(AutoConfig.from_pretrained(bert_model))
    tokenizer = AutoTokenizer.from_pretrained(bert_model)
    vocab2subwords = build_vocab2subwords(tokenizer, k, VOCAB).to(device)
    rd = RD.load_from_checkpoint(wisdomifier_path, bert_mlm=bert_mlm, vocab2subwords=vocab2subwords)
    rd.eval()  # otherwise, the model will output different results with the same inputs
    rd = rd.to(device)  # otherwise, you can not run the inference process on GPUs.
    test_loader = data_module.test_dataloader()

    # the metric
    rd_metric = RDMetric()
    for idx, batch in enumerate(test_loader):
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
