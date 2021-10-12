from transformers import AutoTokenizer
from wisdomify.loaders import load_conf, load_device
from wisdomify.builders import Wisdom2SubWordsBuilder


def main():
    conf = load_conf()['versions']['0']
    device = load_device()
    bert_model = conf['bert_model']
    wisdoms = conf['wisdoms']
    tokenizer = AutoTokenizer.from_pretrained(bert_model)
    k = 0
    for val in range(50):
        k += 1
        try:
            wisdom2subwords = Wisdom2SubWordsBuilder(tokenizer, k, device)(wisdoms)
        except ValueError as ve:
            print(ve, k)
        else:
            print("minimum k:", k)
            print(wisdom2subwords)
            break


if __name__ == '__main__':
    main()

