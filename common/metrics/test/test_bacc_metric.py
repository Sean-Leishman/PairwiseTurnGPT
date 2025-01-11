import torch
import unittest

from common.metrics.turn_metrics import EndOfTurnMetric
from common.metrics.bacc import BACC, BinaryBACC, MultiClassBACC
from data.aligned_process import TurnEndType


class TestBACC(unittest.TestCase):
    def test_binary_bacc_token(self):
        metric = EndOfTurnMetric(
            token_ids=torch.tensor([1], device="cuda:0"),
            config=TurnEndType.NONE,
            device="cuda:0",
            metric=BACC,
            thresholds=torch.linspace(0, 1, 9),
        )
        metric.add(
            torch.tensor(
                [[[0.1, 0.7, 0.2], [0.1, 0.2, 0.7], [0.4, 0.1, 0.5], [0.7, 0.2, 0.1]]],
                device="cuda:0",
            ),
            torch.tensor([[0, 1, 2, 0]], device="cuda:0"),
        )

        output = metric.compute()
        bacc_scores = metric.compute_raw()

        self.assertTrue(isinstance(metric.metric_thresholds[0].bacc, BinaryBACC))
        self.assertEqual(bacc_scores, [0.5, 0.75, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
        self.assertEqual(output[0], 1.0)
        self.assertEqual(output[2], metric.thresholds[2])

    def test_multiclass_multiple_bacc_token(self):
        metric = EndOfTurnMetric(
            token_ids=torch.tensor([1, 2], device="cuda:0"),
            task="multiclass",
            config=TurnEndType.NONE,
            device="cuda:0",
            metric=BACC,
            thresholds=torch.linspace(0, 1, 9),
        )
        metric.add(
            torch.tensor(
                [[[0.1, 0.7, 0.2], [0.1, 0.2, 0.7], [0.4, 0.1, 0.5], [0.7, 0.2, 0.1]]],
                device="cuda:0",
            ),
            torch.tensor([[0, 1, 2, 0]], device="cuda:0"),
        )

        output = metric.compute()
        bacc_scores = metric.compute_raw()

        self.assertTrue(isinstance(metric.metric_thresholds[0].bacc, MultiClassBACC))
        self.assertTrue(metric.num_classes == 2)
        self.assertEqual(
            bacc_scores, [0.5, 0.625, 0.875, 0.875, 0.875, 1.0, 0.5, 0.5, 0.5]
        )
        self.assertEqual(output[0], 1.0)
        self.assertEqual(output[2], metric.thresholds[5])

    def test_multiclass_num_mismatch_token(self):
        self.assertRaises(
            AssertionError,
            EndOfTurnMetric,
            token_ids=torch.tensor([1, 2], device="cuda:0"),
            task="multiclass",
            config=TurnEndType.NONE,
            device="cuda:0",
            num_classes=3,
            metric=BACC,
            thresholds=torch.linspace(0, 1, 9),
        )

    def test_multiclass_single_class_bacc_token(self):
        metric = EndOfTurnMetric(
            token_ids=torch.tensor([1], device="cuda:0"),
            task="multiclass",
            config=TurnEndType.NONE,
            device="cuda:0",
            num_classes=1,
            metric=BACC,
            thresholds=torch.linspace(0, 1, 9),
        )
        metric.add(
            torch.tensor(
                [[[0.1, 0.7, 0.2], [0.1, 0.2, 0.7], [0.4, 0.1, 0.5], [0.7, 0.2, 0.1]]],
                device="cuda:0",
            ),
            torch.tensor([[0, 1, 2, 0]], device="cuda:0"),
        )

        output = metric.compute()
        bacc_scores = metric.compute_raw()

        self.assertTrue(isinstance(metric.metric_thresholds[0].bacc, MultiClassBACC))
        self.assertEqual(bacc_scores, [0.5, 0.75, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5])
        self.assertEqual(output[0], 1.0)
        self.assertEqual(output[2], metric.thresholds[2])

    def test_binary_multiple_class_bacc_token(self):
        metric = EndOfTurnMetric(
            token_ids=torch.tensor([1, 2], device="cuda:0"),
            task="binary",
            config=TurnEndType.NONE,
            device="cuda:0",
            num_classes=1,
            metric=BACC,
            thresholds=torch.linspace(0, 1, 9),
        )
        metric.add(
            torch.tensor(
                [[[0.1, 0.7, 0.2], [0.1, 0.2, 0.7], [0.4, 0.1, 0.5], [0.7, 0.2, 0.1]]],
                device="cuda:0",
            ),
            torch.tensor([[0, 1, 2, 0]], device="cuda:0"),
        )

        output = metric.compute()
        bacc_scores = metric.compute_raw()

        self.assertTrue(isinstance(metric.metric_thresholds[0].bacc, BinaryBACC))
        self.assertEqual(bacc_scores, [0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5])
        self.assertEqual(output[0], 1.0)
        self.assertEqual(output[2], metric.thresholds[5])
