
from wisdomify.builders import TensorBuilder
from wisdomify.loaders import load_conf
from wisdomify.classes import WISDOMS
from transformers import AutoTokenizer
from pprint import pprint


def main():
    conf = load_conf()['versions']['0']
    bert_model = conf['bert_model']
    k = conf['k']
    tokenizer = AutoTokenizer.from_pretrained(bert_model)
    tokenizer.add_tokens(new_tokens=WISDOMS)  # this is the expected input coming in.
    wisdom2subwords = TensorBuilder.build_wisdom2subwords(tokenizer, k, WISDOMS)
    decoded = [
        [tokenizer.decode(idx) for idx in row]
        for row in wisdom2subwords.tolist()
    ]
    for row in decoded:
        print(row)


if __name__ == '__main__':
    main()
