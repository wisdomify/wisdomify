"""
https://stackoverflow.com/a/47867513
"""

import torch


def main():
    t = torch.Tensor([[1, 2, 3],
                      [2, 3, 1],
                      [1, 3, 2]])
    t = t == 2
    print(t)
    t = t.nonzero(as_tuple=True)  # get the indices of nonzero values.
    print(t[0])  # dim 0
    print(t[1])  # dim 1 <- this is what I need


if __name__ == '__main__':
    main()
