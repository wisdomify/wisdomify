
from transformers import BertTokenizer
from wisdomify.paths import DATA_DIR
from os import path


def main():
    tokenizer = BertTokenizer.from_pretrained("beomi/kcbert-base")
    tokenizer_path = path.join(DATA_DIR, "tokenizer_eg")
    tokenizer.save_pretrained(tokenizer_path)  # save
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)  # and then load it


if __name__ == '__main__':
    main()
