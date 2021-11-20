from transformers import BertTokenizer, AutoTokenizer


def main():
    # the
    tokenizer = BertTokenizer.from_pretrained('kykim/bert-kor-base')
    encoded = tokenizer("서당개 삼 년이면 풍월을 읊는다")
    print(encoded['input_ids'])

    # does autotokenizer work with  albert? -> No, it does not work with it.
    tokenizer = AutoTokenizer.from_pretrained('kykim/albert-kor-base')
    encoded = tokenizer("서당개 삼 년이면 풍월을 읊는다")
    print(encoded['input_ids'])


if __name__ == '__main__':
    main()
