from transformers import AutoTokenizer
from wisdomify.loaders import load_config, load_device
from wisdomify.tensors import Wisdom2SubwordsBuilder


def main():
    conf = load_config()['versions']['0']
    device = load_device(gpu=False)
    bert_model = conf['bert_model']
    wisdoms = conf['wisdoms']
    tokenizer = AutoTokenizer.from_pretrained(bert_model)
    k = 0
    for val in range(50):
        k += 1
        try:
            wisdom2subwords = Wisdom2SubwordsBuilder(tokenizer, k, device)(wisdoms)
        except ValueError as ve:
            print(ve, k)
        else:
            print("minimum k:", k)
            print(wisdom2subwords)
            break


if __name__ == '__main__':
    main()

