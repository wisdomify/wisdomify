import unittest
from typing import List
import numpy as np
import torch
from wisdomify.metrics import RDMetric


class TestRDMetric(unittest.TestCase):

    Y_hat: torch.Tensor  # predictions
    y: torch.LongTensor  # targets (labels)
    ranks: List[int]

    @classmethod
    def setUpClass(cls) -> None:
        cls.Y_hat = torch.Tensor([[0.6, 0.3, 0.1, 0.2],
                                  [0.3, 0.1, 0.6, 0.2],
                                  [0.1, 0.6, 0.3, 0.4]])  # should be normalised
        cls.y = torch.LongTensor([0, 1, 2])  # should be a long tensor
        cls.ranks = [0, 3, 2]  # 0.6 is 0th, 0.1 is 3th, 0.3 is 2th.

    def test_update(self):
        rd_metric = RDMetric()
        rd_metric.update(self.Y_hat, self.y)
        self.assertEqual(self.ranks, rd_metric.ranks)
        self.assertEqual(1, rd_metric.correct_top1)  # 0
        self.assertEqual(2, rd_metric.correct_top3)  # 0 & 2
        self.assertEqual(3, rd_metric.correct_top5)  # 0 & 2 & 3
        self.assertEqual(self.Y_hat.shape[0], rd_metric.total)

    def test_compute(self):
        rd_metric = RDMetric()
        rd_metric.update(self.Y_hat, self.y)
        mean, median, std, top1, top3, top5 = rd_metric.compute()
        self.assertEqual(np.mean(self.ranks), mean)
        self.assertEqual(np.median(self.ranks), median)
        self.assertEqual(np.std(self.ranks), std)
        self.assertAlmostEqual(1 / 3, top1, 6)
        self.assertAlmostEqual(2 / 3, top3, 6)
        self.assertAlmostEqual(3 / 3, top5, 6)
