from transformers import AutoModel, AutoTokenizer


def main():
    # --- ko bigbird --- #
    model = AutoModel.from_pretrained("monologg/kobigbird-bert-base")  # BigBirdModel
    tokenizer = AutoTokenizer.from_pretrained("monologg/kobigbird-bert-base")
    text = "아 한국돌아가서 불고기 먹고싶다"
    encoded = tokenizer(text, return_tensors="pt")
    outputs = model(**encoded)
    H_all = outputs['last_hidden_state']  # 제일 마지먹 레이어의 출력
    print(H_all.shape)  # (1, 12, 768)

    # ---  kcbert --- #
    tokenizer = AutoTokenizer.from_pretrained("beomi/kcbert-base")
    model = AutoModel.from_pretrained("beomi/kcbert-base")
    text = "아 한국돌아가서 불고기 먹고싶다"
    encoded = tokenizer(text, return_tensors="pt")
    outputs = model(**encoded)
    H_all = outputs['last_hidden_state']  # 제일 마지먹 레이어의 출력
    print(H_all.shape)  # (1, 12, 768)


if __name__ == '__main__':
    main()
