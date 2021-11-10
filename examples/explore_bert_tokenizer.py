from transformers import BertTokenizer

# you can just use special tokens in the sentences, e.g. [MASK]
BATCH = [
    ("나는 테슬라의 비전이 좋아.", "하하, 그거 좋은 [MASK]네."),  # pun
    ("[MASK]는 원숭이가 가장 좋아하는 과일이다.", None)  # bananas,
]
# 네이버 뉴스 기사 댓글을 사전학습한 모델.
BERT_MODEL = "beomi/kcbert-base"


def main():
    global BATCH, BERT_MODEL

    # encode the batch into input_ids, token_type_ids and attention_mask
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
    lefts = [left for left, _ in BATCH]
    rights = [right for _, right in BATCH]
    encoded = tokenizer(text=lefts,
                        text_pair=rights,
                        # return them as pytorch tensors
                        # return_tensors='tf'
                        return_tensors="pt",
                        # 알아서 special token을 insert
                        add_special_tokens=True,
                        # align the lengths by padding the short ones with [PAD] tokens.
                        truncation=True,
                        padding=True)

    print(type(encoded))
    print("--- encoded ---")
    print(encoded['input_ids'])  # the id of each subword
    print(encoded['token_type_ids'])  # 0 = first sentence, 1 = second sentence (used for NSP)
    print(encoded['attention_mask'])  # 0 = do not compute attentions for (e.g. auto-regressive decoding)
    # note: positional encodings are optional; if they are not given, BertModel will automatically generate
    # one.
    print("--- decoded ---")
    decoded = [
        [tokenizer.decode(token_id) for token_id in sequence]
        for sequence in encoded['input_ids']
    ]
    for tokens in decoded:
        print(tokens)
    # the model will not attend to the padded tokens, thanks to the attention mask.


if __name__ == '__main__':
    main()
