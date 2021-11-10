
import torch


def main():
    X = torch.Tensor([[0.6, 0.3, 0.1],
                      [0.3, 0.1, 0.6],
                      [0.1, 0.6, 0.3]])  # should be normalised
    argsorted = torch.argsort(X, dim=1, descending=True)
    y = torch.LongTensor([0, 1, 2])  # should be a long tensor
    y = y.repeat(argsorted.shape[1], 1).T
    print(argsorted)
    print(y)
    print(argsorted == y)


if __name__ == '__main__':
    main()
