from transformers import AutoTokenizer
from wisdomify.loaders import load_conf
from wisdomify.builders import Builder
from wisdomify.classes import WISDOMS


def main():
    bert_model = load_conf()['versions']['0']['bert_model']
    tokenizer = AutoTokenizer.from_pretrained(bert_model)
    k = 0
    for val in range(50):
        k += 1
        try:
            wisdom2subwords = Builder.build_wisdom2subwords(tokenizer, k, WISDOMS)
        except ValueError as ve:
            print(ve, k)
        else:
            print("minimum k:", k)
            print(wisdom2subwords)
            break
    # the minimum possible k is 11.


if __name__ == '__main__':
    main()

