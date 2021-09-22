from wisdomify.loaders import load_conf
from transformers import AutoTokenizer
from wisdomify.vocab import VOCAB


def main():
    conf = load_conf()['versions']['0']
    bert_model = conf['bert_model']
    k = conf['k']
    tokenizer = AutoTokenizer.from_pretrained(bert_model)
    tokenizer.add_tokens(VOCAB)


if __name__ == '__main__':
    main()
