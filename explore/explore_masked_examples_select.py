import torch
K = 2


def main():
    wisdom_mask = torch.Tensor([[0, 1, 1, 0, 0],
                                [1, 1, 0, 0, 0],
                                [0, 0, 0, 1, 1],
                                [0, 1, 1, 0, 0]])  # (N, L)
    H_all = torch.Tensor([[[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]],
                          [[2, 2], [2, 2], [2, 2], [0, 0], [0, 0]],
                          [[0, 0], [0, 0], [3, 3], [3, 3], [3, 3]],
                          [[0, 0], [4, 4], [4, 4], [4, 4], [0, 0]]])  # (N, L, H)
    N, _, H = H_all.size()
   
    print("="*101)
    H_copy = H_all.clone().detach() 
    wisdom_positions = torch.where(wisdom_mask == 1)
    H_copy[wisdom_positions] = 0
    ex_positions = torch.where((H_copy != 0) & (H_copy != 2) & (H_copy != 3))
    ex_mask = torch.zeros_like(H_copy)
    ex_mask[ex_positions] = 1
    print(ex_mask)
    print(ex_mask.shape)  # (N, L, H)
    
    print("="*101)
    H_ex = H_all.clone().detach()
    H_ex[ex_mask == 0] = 0
    print(H_ex)
    print(H_ex.shape)
    print("="*101)


if __name__ == '__main__':
    main()
