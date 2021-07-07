"""
https://torchmetrics.readthedocs.io/en/latest/?_ga=2.105837638.439472567.1625635601-345348531.1625635601
"""


def main():
    import torch
    import torchmetrics

    # initialize metric
    metric = torchmetrics.Accuracy()

    n_batches = 10
    for i in range(n_batches):
        # simulate a classification problem
        preds = torch.randn(10, 5).softmax(dim=-1)
        target = torch.randint(5, (10,))
        # metric on current batch
        acc = metric(preds, target)  # this will get use of metric.compute() function.
        print(f"Accuracy on batch {i}: {acc}")

    # metric on all batches using custom accumulation
    acc = metric.compute()  # evaluate on all the pairs that have been accumulated.
    print(f"Accuracy on all data: {acc}")


if __name__ == '__main__':
    main()
