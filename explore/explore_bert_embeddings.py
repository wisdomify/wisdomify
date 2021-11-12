from wisdomify.loaders import load_config, load_device
from wisdomify.builders import WisKeysBuilder
from transformers import BertTokenizer, BertModel
import torch

# you can just use special tokens in the sentences, e.g. [MASK]
BATCH = [
    ("나는 테슬라의 비전이 좋아.", "하하, 그거 좋은 [MASK]네."),  # pun
    ("[MASK]는 원숭이가 가장 좋아하는 과일이다.", None)  # bananas,
]


BERT_MODEL = "beomi/kcbert-base"


def main():
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
    bert = BertModel.from_pretrained(BERT_MODEL)
    # add wisdoms, and sync model
    # https://huggingface.co/transformers/_modules/transformers/modeling_bert.html
    word_embeddings: torch.nn.Embedding = bert.embeddings.word_embeddings  # "token embeddings"
    token_type_embeddings: torch.nn.Embedding = bert.embeddings.token_type_embeddings  # "segment embeddings"
    position_embeddings: torch.nn.Embedding = bert.embeddings.position_embeddings # "posiiton embeddings"
    print(word_embeddings.weight.shape)  # (?=V, ?=768)
    print(token_type_embeddings.weight.shape)  # (?=2, 768)
    print(position_embeddings.weight.shape)  # (?=L, 768)











    lefts = [left for left, _ in BATCH]
    rights = [right for _, right in BATCH]

    encoded = tokenizer(text=lefts,
                        text_pair=rights,
                        # return them as pytorch tensors
                        return_tensors="pt",
                        add_special_tokens=True,
                        # align the lengths by padding the short ones with [PAD] tokens.
                        truncation=True,
                        padding=True)
    N, L = encoded['input_ids'].size()
    # 30000이 무엇을 의미하는가? - 어휘의 크기/
    inputs = word_embeddings(encoded['input_ids'])  # (N, L) -> (N, L, E=H)
    token_types = token_type_embeddings(encoded['token_type_ids'])  # (N, L) -> (N, L, E=H)
    positions = position_embeddings(torch.arange(L).expand(N, L))
    fused = inputs + token_types + positions
    print(fused)  # 이것이 bert의 입력으로 들어간다.


if __name__ == '__main__':
    main()

