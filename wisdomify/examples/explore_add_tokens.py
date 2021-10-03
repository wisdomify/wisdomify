
from wisdomify.loaders import load_conf
from transformers import AutoTokenizer


def main():
    conf = load_conf()['versions']['0']
    bert_model = conf['bert_model']
    k = conf['k']
    wisdoms = conf['wisdoms']
    tokenizer = AutoTokenizer.from_pretrained(bert_model)
    # why does the vocab size not increase after adding new tokens?
    # https://github.com/huggingface/transformers/issues/12632
    # apparently, I should access the vocab size by len(tokenizer), not tokenizer.vocab_size
    # print(tokenizer.vocab_size)
    print(len(tokenizer))
    tokenizer.add_tokens(wisdoms)
    print(len(tokenizer))
    # print(tokenizer.vocab_size)


if __name__ == '__main__':
    main()