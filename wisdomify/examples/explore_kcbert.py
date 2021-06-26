from transformers import AutoModelForMaskedLM, AutoTokenizer

BERT_MODEL = "beomi/kcbert-base"


def main():
    # just for downloading the models.
    kcbert_mlm = AutoModelForMaskedLM.from_pretrained(BERT_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    # the classification layer, used for masked language modeling task.
    print(kcbert_mlm.cls)


if __name__ == '__main__':
    main()
