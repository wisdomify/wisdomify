import torch


def main():
    B = torch.ones(4)
    print(B)
    A = torch.ones(size=(10, 4))
    print(B in A)


if __name__ == '__main__':
    main()