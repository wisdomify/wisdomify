from transformers import AutoTokenizer


def main():
    print("---- experiment 1 ---")
    sent = "안전상의 이유로"
    tokenizer_1 = AutoTokenizer.from_pretrained("monologg/kobert") # kobert는 subword tokenizer를 사용하지 않았다
    tokenizer_2 = AutoTokenizer.from_pretrained('beomi/kcbert-base')  # kcbert는 subword tokenizer를 사용했다 (WordPiece)
    encoded_1 = tokenizer_1(sent)
    encoded_2 = tokenizer_2(sent)
    # 그래서 교착어인 한국어의 경우, 형태소로 미리 쪼개주지 않으면, OOV토큰으로 분류된다
    print([tokenizer_1.decode(tok_id) for tok_id in encoded_1['input_ids']])  # ['[CLS]', '[UNK]', '[UNK]', '[SEP]']
    # WordPiece 알고리즘의 경우, 형태소로 쪼개주지 않아도, 알아서 데이터로부터 형태소의 개념을 학습한다
    print([tokenizer_2.decode(tok_id) for tok_id in encoded_2['input_ids']])  # ['[CLS]', '안전', '##상의', '이유로', '[SEP]']

    # 그렇다면 kobert를 사용하고 싶다면 어떻게 해야할까?
    # konlpy등을 활용해서, 미리 학습했을만한  형태소로 쪼개줘야한다.
    print("---- experiment 2 ---")
    sent = "안전 상 의 이유 로"
    encoded_1 = tokenizer_1(sent)
    encoded_2 = tokenizer_2(sent)
    print([tokenizer_1.decode(tok_id) for tok_id in encoded_1['input_ids']])  # ['[CLS]', '안전', '상', '의', '[UNK]', '[SEP]']
    print([tokenizer_2.decode(tok_id) for tok_id in encoded_2['input_ids']])  # ['[CLS]', '안전', '상', '의', '이유로', '[SEP]']

    # 결론:
    # kobert를 사용하고 싶다면, 미리 형태소로 쪼개줘야한다.
    # 다른 SubowrdTokenizer를 사용하여 학습한 모델을 사용하고 싶다면, 미리 형태소로 쪼갤필요가 없다.


if __name__ == '__main__':
    main()
