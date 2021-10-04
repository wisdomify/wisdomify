import torch
K = 3


def main():
    wisdom_mask = torch.Tensor([[0, 1, 1, 1, 0],
                                [1, 1, 1, 0, 0],
                                [0, 0, 1, 1, 1],
                                [0, 1, 1, 1, 0]])  # (N, L)
    H_all = torch.Tensor([[[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]],
                          [[2, 2], [2, 2], [2, 2], [0, 0], [0, 0]],
                          [[0, 0], [0, 0], [3, 3], [3, 3], [3, 3]],
                          [[0, 0], [4, 4], [4, 4], [4, 4], [0, 0]]])  # (N, L, H)
    N, _, H = H_all.size()
    # H_k = H_all[wisdom_mask.bool()]  # (N, L, 768) -> (
    # print(H_k.shape)  # (K
    # # reshape 문서 - https://pytorch.org/docs/stable/generated/torch.reshape.html
    # # 원소의 개수만 맞추기.
    # H_k = H_k.reshape(H_k.shape[0] // k, k)  # (1, K * N) -> (N, K)
    # print(H_k)
    wisdom_mask = wisdom_mask.unsqueeze(2).expand(H_all.shape)
    H_k = torch.masked_select(H_all, wisdom_mask.bool())
    print(H_k.shape)
    print(H_k)
    H_k = H_k.reshape(N, K, H)  # (N * K * H) -> (N, K, H)
    print(H_k.shape)
    print(H_k)


if __name__ == '__main__':
    main()
