import torch
import unittest

from common.metrics.turn_metrics import EndOfTurnMetric
from common.metrics.f1 import F1
from data.aligned_process import TurnEndType


class TestF1Score(unittest.TestCase):
    def test_binary_f1(self):
        metric = EndOfTurnMetric(
            token_ids=torch.tensor([0], device="cuda:0"),
            config=TurnEndType.NONE,
            device="cuda:0",
            metric=F1,
        )
        metric.add(
            torch.tensor(
                [[[0.1, 0.8, 0.1], [0.8, 0.1, 0.1], [0.5, 0.1, 0.1], [0.8, 0.1, 0.1]]],
                device="cuda:0",
            ),
            torch.tensor([[0, 1, 2, 0]], device="cuda:0"),
        )

        f1 = metric.compute()

    def test_binary_multitoken_f1(self):
        metric = EndOfTurnMetric(
            token_ids=torch.tensor([0, 1], device="cuda:0"),
            config=TurnEndType.NONE,
            device="cuda:0",
            metric=F1,
        )
        metric.add(
            torch.tensor(
                [[[0.1, 0.8, 0.1], [0.8, 0.1, 0.1], [0.5, 0.1, 0.1], [0.8, 0.1, 0.1]]],
                device="cuda:0",
            ),
            torch.tensor([[0, 1, 2, 0]], device="cuda:0"),
        )

        f1 = metric.compute()

    def test_multiclass_multitoken_f1(self):
        metric = EndOfTurnMetric(
            token_ids=torch.tensor([0, 1], device="cuda:0"),
            config=TurnEndType.NONE,
            device="cuda:0",
            task="multiclass",
            metric=F1,
        )
        metric.add(
            torch.tensor(
                [[[0.1, 0.8, 0.1], [0.8, 0.1, 0.1], [0.5, 0.1, 0.1], [0.8, 0.1, 0.1]]],
                device="cuda:0",
            ),
            torch.tensor([[0, 1, 2, 0]], device="cuda:0"),
        )

        f1 = metric.compute()
