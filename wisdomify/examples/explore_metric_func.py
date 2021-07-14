"""
https://torchmetrics.readthedocs.io/en/latest/#functional-metrics
"""
import torch
import torchmetrics


def main():

    # simulate a classification problem
    preds = torch.randn(10, 5).softmax(dim=-1)
    target = torch.randint(5, (10,))
    acc = torchmetrics.functional.accuracy(preds, target)

    print(acc)


if __name__ == '__main__':
    main()
