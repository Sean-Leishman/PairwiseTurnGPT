import torch
import logging

from typing import Union

from data import TurnType, TurnEndType
from common.metrics.bacc import BACC
from common.metrics.utils import get_mask, get_interrupt_mask

logger = logging.getLogger(__name__)


class ClassificationMetric:
    def __init__(
        self,
        token_id=-1,
        tokens="all",
        task="binary",
        num_classes=None,
        device="cuda:0",
        metric=BACC,
        thresholds=torch.linspace(0, 1, 20),
        average=None,
    ):
        self.device = device
        self.thresholds = thresholds.to(self.device)
        self.num_classes = num_classes
        self.task = task
        self.average = average

        assert (
            task == "binary" or task == "multiclass"
        ), f"Task must be binary or multiclass not {task}"
        self.metric_thresholds = [
            metric(
                task=self.task,
                num_classes=self.num_classes,
                threshold=float(thresh),
                average=self.average,
            ).to(self.device)
            for thresh in self.thresholds
        ]

        self.token_id = token_id
        self.label = tokens

        if self.task == "binary":
            assert (
                self.token_id != -1
            ), "Token ID must be specified for binary classification"

    def __str__(self):
        return f"{self.task}-{self.label}-{str(self.metric_thresholds[0])}"

    def add(self, probs, labels, **kwargs):
        mask = get_mask(TurnType.NONE, labels, self.device, **kwargs).to(self.device)
        labels = labels[..., 1:][mask]

        if self.task == "binary":
            probs = probs[:, :-1, :][mask, :][..., self.token_id]
            labels = torch.eq(labels, self.token_id).int()
        else:
            probs = probs[:, :-1, :][mask, :]

        for metric in self.metric_thresholds:
            metric.update(probs, labels)

    def compute(self, params=None):
        if params is None:
            params = {}

        thresh = params.get(str(self), -1)
        results = [metric.compute() for metric in self.metric_thresholds]
        self.log(results)

        if thresh == -1:
            idx = results.index(max(results))
        else:
            idx = self.thresholds.tolist().index(thresh)

        output = results[idx].item()
        count = self.metric_thresholds[idx].count()
        return output, count, self.thresholds[idx]

    def compute_raw(self):
        return [metric.compute() for metric in self.metric_thresholds]

    def reset(self):
        for metric in self.metric_thresholds:
            metric.reset()

    def plot(self):
        return None

    def log(self, results):
        def string_result(idx, result):
            return f"Threshold={self.thresholds[idx].item()}, Result={result}"

        logger.info(
            f"Results for {self}: {[string_result(idx,result) for idx,result in enumerate(results)]}"
        )


class EndOfTurnMetric:
    def __init__(
        self,
        token_ids: torch.Tensor,
        config: Union[TurnType, TurnEndType],
        tokens="",
        task="binary",
        num_classes=None,
        device: str = "cuda:0",
        metric=BACC,
        thresholds=torch.linspace(0, 1, 20),
        average=None,
    ):
        self.device = device
        self.thresholds = thresholds.to(self.device)
        self.config = config

        self.token_ids = token_ids.to(self.device)
        self.tokens = tokens
        assert (
            len(self.token_ids.shape) == 1
        ), f"Token IDs must be a 0D tensor {self.token_ids.shape}"

        self.num_classes = num_classes
        if task == "multiclass":
            if self.num_classes is None:
                self.num_classes = self.token_ids.shape[0]
            assert (
                self.num_classes >= 1
            ), "Number of classes must be greater than 1 for multiclass task"
            assert (
                self.num_classes == self.token_ids.shape[0]
            ), "Should have the same number of classes as token IDs"

        assert (
            task == "binary" or task == "multiclass"
        ), f"Task must be binary or multiclass not {task}"
        self.task = task

        self.average = average
        self.metric_thresholds = [
            metric(
                task=self.task,
                num_classes=self.num_classes,
                threshold=float(thresh),
                device=self.device,
            )
            for thresh in self.thresholds
        ]

    def __str__(self):
        return f"{self.tokens}-{self.task}-{self.config.name}-{str(self.metric_thresholds[0])}"

    def add(self, probs, labels, **kwargs):
        mask = get_mask(self.config, labels, self.device, **kwargs).to(self.device)
        labels = labels[..., 1:][mask]

        nlabels = torch.zeros_like(labels)
        if self.task == "binary":
            token_probs, _ = torch.max(probs[..., self.token_ids], dim=-1)
            nlabels = torch.isin(labels, self.token_ids)
            token_probs = token_probs[..., :-1][mask]
        elif self.task == "multiclass":
            token_probs = probs[..., self.token_ids]
            nlabels = torch.zeros(
                self.token_ids.shape[0], labels.shape[0], device=self.device
            )
            token_probs = token_probs[:, :-1, :]
            token_probs = token_probs[mask, :]
            for idx, token in enumerate(self.token_ids):
                nlabels[idx] = torch.eq(labels, token)
            nlabels = torch.transpose(nlabels, 1, -2)

        for metric in self.metric_thresholds:
            metric.update(token_probs, nlabels)

    def compute(self, params=None):
        """
        Returns a list of the computed metrics if at_max is False, otherwise returns the metric at the maximum threshold
        params: output metric at a specific threshold
        """
        if params is None:
            params = {}

        thresh = params.get(str(self), -1)
        results = [metric.compute() for metric in self.metric_thresholds]
        self.log(results)

        if thresh == -1:
            idx = results.index(max(results))
        else:
            idx = self.thresholds.tolist().index(thresh)

        output = results[idx].item()

        count = self.metric_thresholds[idx].count()
        return output, count, self.thresholds[idx]

    def compute_raw(self):
        return [metric.compute() for metric in self.metric_thresholds]

    def reset(self):
        for metric in self.metric_thresholds:
            metric.reset()

    def plot(self):
        return None

    def log(self, results):
        def string_result(idx, result):
            return f"Threshold={self.thresholds[idx].item()}, Result={result}"

        logger.info(
            f"Results for {self}: {[string_result(idx,result) for idx,result in enumerate(results)]}"
        )


class StartOfTurnMetric(EndOfTurnMetric):
    def __init__(
        self,
        token_ids: torch.Tensor,
        config: Union[TurnType, TurnEndType],
        device: str = "cuda:0",
        metric=BACC,
        thresholds=torch.linspace(0, 1, 20),
        average=None,
    ):
        super().__init__(
            token_ids=token_ids,
            config=config,
            device=device,
            metric=metric,
            thresholds=thresholds,
            average=average,
        )

    def add(self, probs, labels, **kwargs):
        interrupt_mask = get_interrupt_mask(self.config, labels, **kwargs).to(
            self.device
        )

        tokens_probs = probs[..., self.token_ids[0]].clone()
        new_labels = torch.eq(labels, self.token_ids[0]).clone()

        if len(self.token_ids) > 1:
            for token_id in self.token_ids[1:]:
                tokens_probs += probs[..., token_id]
                new_labels += torch.eq(labels, token_id)

        tokens_probs = 1 - tokens_probs
        labels = torch.logical_not(new_labels)

        tokens_probs = tokens_probs[..., :-1][interrupt_mask]
        labels = labels[..., 1:][interrupt_mask]

        for metric in self.metric_thresholds:
            metric.update(tokens_probs, labels)

    def plot(self):
        return None

    def __str__(self):
        return f"turn-start-{self.task}-{self.config.name}-{str(self.metric_thresholds[0])}"
