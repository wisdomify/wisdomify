from typing import Tuple
from torchmetrics import Metric
import torch


class RDMetrics(Metric):

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
        preds = torch.argmax(preds, dim=1)
        assert preds.shape == targets.shape
        # so, how do I get the ranks?
        argsorted = torch.argsort(preds, dim=1)
        self.ranks = (argsorted == targets).nonzero(as_tuple=True)[1].tolist()  # noqa, see examples/explore_nonzero
        self.correct_top1 = torch.eq(argsorted[:, :1], targets[:, :1]).sum()  # noqa
        self.correct_top10 = torch.eq(argsorted[:, :10], targets[:, :10]).sum()  # noqa
        self.correct_top100 = torch.eq(argsorted[:, :100], targets[:, :100]).sum()  # noqa
        self.total += targets.numel()

    def compute(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        returns: median, var, top1 acc, top10 acc, top10 acc.
        """
        median = torch.median(self.ranks)
        var = torch.var(self.ranks)
        top1_acc = self.correct_top1.float() / self.total
        top10_acc = self.correct_top10.float() / self.total
        top100_acc = self.correct_top100.float() / self.total
        return median, var, top1_acc, top10_acc, top100_acc
