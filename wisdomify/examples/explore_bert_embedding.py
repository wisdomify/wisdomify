import torch
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEmbeddings


def main():
    config = BertConfig('beomi/kcbert-base')
    bertEmbedding = BertEmbeddings(config)
    X = torch.Tensor([[[2, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                        4, 4, 3, 8374, 9231, 10962, 8498, 1099, 7998, 8860,
                        9231, 314, 4267, 10940, 802, 4421, 3],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1]]]
                     )
    bertEmbedding.forward(
        input_ids=X[0],
        token_type_ids=X[2]
    )


if __name__ == '__main__':
    main()

