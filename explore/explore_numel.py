"""
https://pytorch.org/docs/stable/generated/torch.numel.html
"""

import torch


def main():
    a = torch.randn(1, 2, 3, 4, 5)
    print(a.numel())  # total number of elements = 1 * 2 * 3 * 4 * 5 = 120

    b = torch.zeros(4, 4)
    print(b.numel())  # total number of elements = 4 * 4 = 16


if __name__ == '__main__':
    main()
