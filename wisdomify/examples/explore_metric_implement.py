import torch
from torchmetrics import Metric


class MyAccuracy(Metric):
    def __init__(self, dist_sync_on_step=False):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    # noinspection PyInitNewSignature
    # "noqa" is for suppressing an unnecessary warning
    # https://www.jetbrains.com/help/pycharm/disabling-and-enabling-inspections.html#suppress-inspections
    def update(self, preds: torch.Tensor, targets: torch.Tensor):  # noqa
        # update metric states
        preds = torch.argmax(preds, dim=1)
        assert preds.shape == targets.shape
        self.correct += torch.sum(torch.eq(targets, preds))
        self.total += targets.numel()

    def compute(self):
        # compute final result
        return self.correct.float() / self.total


def main():
    metric = MyAccuracy()
    n_batches = 10
    for i in range(n_batches):
        # simulate a classification problem
        preds = torch.randn(10, 5).softmax(dim=-1)
        target = torch.randint(5, (10,))
        # metric on current batch
        acc = metric(preds, target)  # this will accumulate the pairs.
        print(f"Accuracy on batch {i}: {acc}")

    # metric on all batches using custom accumulation
    acc = metric.compute()  # evaluate on all the pairs that have been accumulated.
    print(f"Accuracy on all data: {acc}")


if __name__ == '__main__':
    main()
