from typing import Tuple
import numpy as np
from torchmetrics import Metric
import torch


class RDMetric(Metric):

    def __init__(self, dist_sync_on_step=False):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        # define the states
        self.add_state("ranks", default=[], dist_reduce_fx='cat')  # concat is the reduce function
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("correct_top1", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("correct_top10", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("correct_top100", default=torch.tensor(0), dist_reduce_fx="sum")


    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:  # noqa
        # so, how do I get the ranks?
        argsorted = torch.argsort(preds, dim=1, descending=True)
        targets = targets.repeat(argsorted.shape[1], 1).T
        self.ranks += torch.eq(argsorted, targets).nonzero(as_tuple=True)[1].tolist()  # noqa, see examples/explore_nonzero
        self.correct_top1 += torch.eq(argsorted[:, :1], targets[:, :1]).sum()  # noqa
        self.correct_top10 += torch.eq(argsorted[:, :10], targets[:, :10]).sum()  # noqa
        self.correct_top100 += torch.eq(argsorted[:, :100], targets[:, :100]).sum()  # noqa
        self.total += preds.shape[0]

    def compute(self) -> Tuple[float, float, float, float, float]:
        """
        returns: median, var, top1 acc, top10 acc, top10 acc.
        """
        median = np.median(self.ranks)
        var = np.var(self.ranks)
        top1_acc = self.correct_top1.float() / self.total
        top10_acc = self.correct_top10.float() / self.total
        top100_acc = self.correct_top100.float() / self.total
        return median, var, top1_acc.item(), top10_acc.item(), top100_acc.item()

