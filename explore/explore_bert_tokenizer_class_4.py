
# pip3 install transformers.
from transformers import BertTokenizer

BATCH = [
    ("나는 테슬라의 비전이 좋아.", "하하, 그거 좋은 [MASK]네."),  # pun
    ("[MASK]는 원숭이가 가장 좋아하는 과일이다.", None)  # bananas,
]


def main():
    # 이미 학습이 된 토크나이저를 로드할 수 있다.
    kor_tokenizer = BertTokenizer.from_pretrained("beomi/kcbert-base")
    # eng_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')  # 모델 별로 토크나이저 클래스가 존재한다.
    lefts = [left for left, _ in BATCH]
    rights = [right for _, right in BATCH]

    encoded = kor_tokenizer(text=lefts,
                            text_pair=rights,
                            # 알아서, e.g. [CLS] 제일 앞에 인코딩, [SEP]
                            add_special_tokens=True,
                            # return_tensors="tf"
                            return_tensors="pt",
                            # 가장 길이가 긴 문장에 맞추어서 padding을 진행
                            padding=True)

    print("input_ids\n", encoded['input_ids'])
    print("token_type_ids\n", encoded['token_type_ids'])
    print("attention_mask\n", encoded['attention_mask'])

    print("--- decoded ---")
    decoded = [
        [kor_tokenizer.decode(token_id) for token_id in sequence]
        for sequence in encoded['input_ids']
    ]
    for tokens in decoded:
        print(tokens)
    # the model will not attend to the padded tokens, thanks to the attention mask.


if __name__ == '__main__':
    main()
