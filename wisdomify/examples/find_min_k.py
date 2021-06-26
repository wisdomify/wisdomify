from transformers import AutoTokenizer
from wisdomify.loaders import load_conf
from wisdomify.builders import build_vocab2subwords
from wisdomify.vocab import VOCAB


def main():
    bert_model = load_conf()['bert_model']
    tokenizer = AutoTokenizer.from_pretrained(bert_model)
    k = 0
    for val in range(50):
        k += 1
        try:
            vocab2subwords = build_vocab2subwords(tokenizer, k, VOCAB)
        except ValueError as ve:
            print(ve, k)
        else:
            print("minimum k:", k)
            print(vocab2subwords)
            break
    # the minimum possible k is 11.


if __name__ == '__main__':
    main()

