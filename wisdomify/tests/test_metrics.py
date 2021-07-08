import unittest
from typing import List
import numpy as np
import torch
from wisdomify.metrics import RDMetric


class TestRDMetric(unittest.TestCase):

    preds: torch.Tensor  # predictions
    targets: torch.LongTensor  # targets (labels)
    ranks: List[int]

    @classmethod
    def setUpClass(cls) -> None:
        cls.preds = torch.Tensor([[0.6, 0.3, 0.1],
                                  [0.3, 0.1, 0.6],
                                  [0.1, 0.6, 0.3]])  # should be normalised
        cls.targets = torch.LongTensor([1, 1, 1])  # should be a long tensor
        cls.ranks = [1, 2, 0]

    def test_update(self):
        rd_metric = RDMetric()
        rd_metric.update(self.preds, self.targets)
        self.assertEqual(self.ranks, rd_metric.ranks)
        self.assertEqual(1, rd_metric.correct_top1)
        self.assertEqual(3, rd_metric.correct_top10)
        self.assertEqual(3, rd_metric.correct_top100)
        self.assertEqual(self.targets.numel(), rd_metric.total)

    def test_compute(self):
        rd_metric = RDMetric()
        rd_metric.update(self.preds, self.targets)
        median, var, top1, top10, top100 = rd_metric.compute()
        self.assertEqual(np.median(self.ranks), median)
        self.assertEqual(np.var(self.ranks), var)

