import torch
K = 3


def main():
    mask = torch.Tensor([[0, 1, 1, 1, 0],
                         [1, 1, 1, 0, 0],
                         [0, 0, 1, 1, 1],
                         [0, 1, 1, 1, 0]])
    H_all = torch.randn(size=(4, 5, 768))
    n = mask.shape[0]
    h = H_all.shape[2]
    # H_k = H_all[mask.bool()]  # (N, L, 768) -> (
    # print(H_k.shape)  # (K
    # # reshape 문서 - https://pytorch.org/docs/stable/generated/torch.reshape.html
    # # 원소의 개수만 맞추기.
    # H_k = H_k.reshape(H_k.shape[0] // k, k)  # (1, K * N) -> (N, K)
    # print(H_k)

    mask = mask.unsqueeze(-1).expand(H_all.shape)
    H_k = torch.masked_select(H_all, mask.bool())
    H_k = H_k.reshape(n, K, h)  # (1, K * N) -> (N, K)
    print(H_k.shape)  # (4, 3, 768)


if __name__ == '__main__':
    main()
