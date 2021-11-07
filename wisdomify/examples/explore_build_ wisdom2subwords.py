
from wisdomify.builders import Wisdom2SubwordsBuilder
from wisdomify.loaders import load_device, load_conf
from transformers import AutoTokenizer


def main():
    conf = load_conf()['versions']['0']
    device = load_device()
    bert_model = conf['bert_model']
    k = conf['k']
    wisdoms = conf['wisdoms']
    tokenizer = AutoTokenizer.from_pretrained(bert_model)
    tokenizer.add_tokens(new_tokens=wisdoms)  # this is the expected input coming in.
    wisdom2subwords = Wisdom2SubwordsBuilder(tokenizer, k, device)(wisdoms)
    decoded = [
        [tokenizer.decode(idx) for idx in row]
        for row in wisdom2subwords.tolist()
    ]
    for row in decoded:
        print(row)


if __name__ == '__main__':
    main()
