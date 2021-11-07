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
        self.add_state("correct_top3", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("correct_top5", default=torch.tensor(0), dist_reduce_fx="sum")


    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:  # noqa
        # so, how do I get the ranks?
        argsorted = torch.argsort(preds, dim=1, descending=True)
        targets = targets.repeat(argsorted.shape[1], 1).T
        self.ranks += torch.eq(argsorted, targets).nonzero(as_tuple=True)[1].tolist()  # noqa, see examples/explore_nonzero
        self.correct_top1 += torch.eq(argsorted[:, :1], targets[:, :1]).sum()  # noqa
        self.correct_top3 += torch.eq(argsorted[:, :3], targets[:, :3]).sum()  # noqa
        self.correct_top5 += torch.eq(argsorted[:, :5], targets[:, :5]).sum()  # noqa
        self.total += preds.shape[0]

    def compute(self) -> Tuple[float, float, float, float, float, float]:
        """
        returns: median, var, top1 acc, top10 acc, top10 acc.
        """
        mean = np.mean(self.ranks)
        median = np.median(self.ranks)
        std = np.std(self.ranks)
        top1 = self.correct_top1.float() / self.total
        top3 = self.correct_top3.float() / self.total
        top5 = self.correct_top5.float() / self.total
        return mean.item(), median.item(), std.item(), top1.item(), top3.item(), top5.item()
