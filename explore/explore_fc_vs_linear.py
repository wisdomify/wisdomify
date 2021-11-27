
import torch
from torch.nn import functional as F


def main():
    W, K, H = (10, 33, 768)
    # 목표는 subwords embeddings 을 풀링해서, 최종적으로 하나의 속담에 대응하는 벡터를 얻어내는 것.
    wisdom2subwords_embeddings = torch.randn(size=(W, K, H))

    # A fully connected layer
    fc_layer = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(in_features=K * H, out_features=H)
    )  # (W, K, H) -> (W, K * H) -> (W, H)
    wisdom_embeddings = fc_layer(wisdom2subwords_embeddings)

    linear_layer = torch.nn.Linear(in_features=K, out_features=H)
    wisdom_embeddings = linear_layer(torch.einsum("wkh->whk", wisdom_embeddings)).squeeze()
    # (W, K, H) -> (W, H, K) -> (W, H, 1) -> (W, H)


if __name__ == '__main__':
    main()
