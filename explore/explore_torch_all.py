"""
https://pytorch.org/docs/stable/generated/torch.all.html
"""
import torch


def main():
    a = torch.rand(4, 2).bool()
    print(torch.all(a, dim=1))
    print(torch.all(a, dim=0))


if __name__ == '__main__':
    main()
