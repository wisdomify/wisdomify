from wisdomify.loaders import load_config, load_device
from wisdomify.tensors import WiskeysBuilder
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch


def main():
    conf = load_config()['versions']['0']
    device = load_device(gpu=False)
    bert_model = conf['bert_model']
    wisdoms = conf['wisdoms']
    tokenizer = AutoTokenizer.from_pretrained(bert_model)
    model = AutoModelForMaskedLM.from_pretrained(bert_model)
    # add wisdoms, and sync model
    # https://huggingface.co/transformers/internal/tokenization_utils.html#transformers.tokenization_utils_base.SpecialTokensMixin.add_tokens
    tokenizer.add_tokens(wisdoms)
    # https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.resize_token_embeddings
    model.resize_token_embeddings(len(tokenizer))
    # https://huggingface.co/transformers/_modules/transformers/modeling_bert.html
    word_embeddings: torch.nn.Embedding = model.bert.embeddings.word_embeddings
    wiskeys = WiskeysBuilder(tokenizer, device)(wisdoms)  # (|W|,)
    print(wiskeys)
    print(word_embeddings)
    W = word_embeddings(wiskeys)  # (|W|, E=H)
    print(W)
    print(W.shape)


if __name__ == '__main__':
    main()
