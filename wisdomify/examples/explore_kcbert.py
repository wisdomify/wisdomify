from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers.models.bert import BertModel
BERT_MODEL = "beomi/kcbert-base"


def main():
    # just for downloading the models.
    kcbert_mlm = AutoModelForMaskedLM.from_pretrained(BERT_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    # the classification layer, used for masked language modeling task.
    print(kcbert_mlm.cls)
    # auto models are for automatically inferring the model vocab from the model name.
    # https://huggingface.co/transformers/model_doc/auto.html#auto-classes
    print(type(kcbert_mlm))  # BertForMaskedLM
    print(type(tokenizer))  # BertTokenizerFast


if __name__ == '__main__':
    main()
