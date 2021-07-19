"""
load a pre-trained wisdomify, and play with it.
"""
import argparse
import torch
import yaml
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
from wisdomify.loaders import load_conf
from wisdomify.models import RD, Wisdomifier
from wisdomify.paths import WISDOMIFIER_V_0_CKPT, WISDOMIFIER_V_0_HPARAMS_YAML
from wisdomify.vocab import VOCAB
from wisdomify.builders import build_vocab2subwords

import time


def main():
    start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("--ver", type=str,
                        default="version_0")
    parser.add_argument("--desc", type=str,
                        default="왜 하필이면 오늘이야")
    args = parser.parse_args()
    ver: str = args.ver
    desc: str = args.desc
    conf = load_conf()
    bert_model: str = conf['bert_model']
    print('parser variable load', time.time() - start)

    if ver == "version_0":
        wisdomifier_path = WISDOMIFIER_V_0_CKPT
        with open(WISDOMIFIER_V_0_HPARAMS_YAML, 'r') as fh:
            wisdomifier_hparams = yaml.safe_load(fh)
        k = wisdomifier_hparams['k']
    else:
        # this version is not supported yet.
        raise NotImplementedError("Invalid version provided".format(ver))
    print('hparams load', time.time() - start)

    bert_mlm = AutoModelForMaskedLM.from_config(AutoConfig.from_pretrained(bert_model))
    print('bert mlm config load', time.time() - start)
    tokenizer = AutoTokenizer.from_pretrained(bert_model)
    print('tokenizer instance load', time.time() - start)
    vocab2subwords = build_vocab2subwords(tokenizer, k, VOCAB).to(device)
    print('vocab2subwords instance load', time.time() - start)
    rd = RD.load_from_checkpoint(wisdomifier_path, bert_mlm=bert_mlm, vocab2subwords=vocab2subwords)
    print('rd(load from checkpoint) instance load', time.time() - start)
    rd.eval()  # otherwise, the model will output different results with the same inputs
    print('rd instance evaluation', time.time() - start)
    rd = rd.to(device)  # otherwise, you can not run the inference process on GPUs.
    print('rd device setting', time.time() - start)
    wisdomifier = Wisdomifier(rd, tokenizer)
    print('wisdomifier load', time.time() - start)

    for desc in ["왜 하필이면 오늘이야", "까불지말고 침착하여라", "근처에 있을 것이라고는 전혀 예상하지 못했다", "쓸데없는 변명은 그만 둬", "커피가 없으니 홍차라도 마시자",
                 "결과가 좋아서 기쁘다", "실제로는 별거없네", "코카콜라?? 펩시 마시자"]:
        print("### desc: {} ###".format(desc))
        for results in wisdomifier.wisdomify(sents=[desc]):
            for idx, res in enumerate(results):
                print("{}: ({}, {:.4f})".format(idx, res[0], res[1]))

        print('wisdomify infer time', time.time() - start)


if __name__ == '__main__':
    main()
