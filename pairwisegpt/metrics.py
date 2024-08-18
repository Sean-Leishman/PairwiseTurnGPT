import logging
from enum import IntEnum
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

from torch import Tensor
from torchmetrics import PrecisionRecallCurve, F1Score, ROC, Accuracy
from torchmetrics.functional.text.perplexity import _check_shape_and_type_consistency
from torchmetrics.text import Perplexity
from torchmetrics.utilities.compute import _auc_compute_without_check
from torchmetrics.utilities.plot import plot_curve
from torcheval.metrics.aggregation.auc import AUC


import torch
import logging

from pairwise_generation_dm import TurnType
from torchmetrics.metric import Metric as TorchMetric, Metric

logger = logging.getLogger(__name__)


class MetricType(IntEnum):
    BACC = 0
    PR_AUC = 1
    ROC_AUC = 2
    PERPLEXITY = 3
    F1_SCORE = 4
    ACC = 5
    NRR = 6
    BR = 7


class MeasureType(IntEnum):
    PERPLEXITY = 0
    CATEGORICAL = 1
    REGRESSION = 2


def _perplexity_update(predsA, targetA, predsB, targetB, ignore_index):
    """
    Adapted from https://github.com/Lightning-AI/torchmetrics/blob/v1.3.2/src/torchmetrics/functional/text/perplexity.py#L65#
    to be used with DualPerplexity
    """
    _check_shape_and_type_consistency(predsA, targetA)
    _check_shape_and_type_consistency(predsB, targetB)

    probsA = torch.nn.functional.softmax(
        predsA.reshape(-1, predsA.shape[-1]), dim=1)
    targetA = targetA.reshape(-1)

    probsB = torch.nn.functional.softmax(
        predsB.reshape(-1, predsB.shape[-1]), dim=1)
    targetB = targetB.reshape(-1)

    if ignore_index is not None:
        maskA = targetA.ne(ignore_index)
        targetA = targetA.where(targetA != ignore_index,
                                torch.tensor(0, device=targetA.device))

        maskB = targetB.ne(ignore_index)
        targetB = targetB.where(targetB != ignore_index,
                                torch.tensor(0, device=targetB.device))
    else:
        maskA = torch.ones_like(targetA, dtype=torch.bool)
        maskB = torch.ones_like(targetB, dtype=torch.bool)

    probsA1 = probsA[torch.arange(targetA.numel()), targetA][maskA]
    probsB1 = probsB[torch.arange(targetB.numel()), targetB][maskB]

    log_probsA = -probsA1.log()
    log_probsB = -probsB1.log()
    total_log_probs = log_probsA.sum() + log_probsB.sum()

    count = (maskA.sum() + maskB.sum()) / 2

    return total_log_probs, count


def _perplexity_compute(total: Tensor, count: Tensor):
    return torch.exp(total / count)


class DualPerplexity(Metric):
    """
    Adapted from https://github.com/Lightning-AI/torchmetrics/blob/v1.3.2/src/torchmetrics/text/perplexity.py#L28-L131
    for use with dual-channel model
    """

    def __init__(
            self,
            ignore_index: Optional[int] = None,
            **kwargs: Dict[str, Any],
    ) -> None:
        super().__init__(**kwargs)
        if ignore_index is not None and not isinstance(ignore_index, int):
            raise ValueError(
                f"Argument `ignore_index` expected to either be `None` or an `int` but got {ignore_index}")
        self.ignore_index = ignore_index
        self.add_state("total_log_probs", default=torch.tensor(
            0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(
            0.0), dist_reduce_fx="sum")

    def update(self, predsA: Tensor, predsB: Tensor, targetA: Tensor, targetB: Tensor) -> None:
        """Update state with predictions and targets."""
        total_log_probs, count = _perplexity_update(
            predsA, targetA, predsB, targetB, self.ignore_index)
        self.total_log_probs += total_log_probs
        self.count += count

    def compute(self) -> Tensor:
        """Compute the Perplexity."""
        return _perplexity_compute(self.total_log_probs, self.count)


class Metric():
    def __init__(self, metric_type: MetricType, token_id: int, token: str, nthresh=50, turn_type=TurnType.NORMAL, tokens_dict: dict = {}, device="cuda:0"):
        self.metric_type = metric_type
        self.token_id = token_id
        self.token = token
        self.nthresh = nthresh
        self.turn_type = turn_type
        self.device = device

        self.special_tokens = tokens_dict

        self.thresholds = torch.linspace(0.00, 1, 20)

    @abstractmethod
    def calculate(self, params=None):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def is_graphical(self):
        pass

    @abstractmethod
    def graph(self):
        pass


class Perplexity_Score(Metric):
    def __init__(self, device="cuda:0", turn_type=TurnType.NONE, rule="none", filter_bc_overlap_token=False, **kwargs):
        self.emp_index = -100
        self.turn_type = turn_type
        self.filter_bc_overlap_token = False
        self.perplexity = Perplexity(
            ignore_index=self.emp_index).to(device=device)

        self.rule = rule
        self.filter_bc_overlap_token = filter_bc_overlap_token
        if rule == "joint":
            self.perplexity = DualPerplexity(
                ignore_index=self.emp_index).to(device=device)

    def add(self, preds, labels, yield_mask=None, non_yield_mask=None, ignore_mask=None, overlap_mask=None, non_overlap_mask=None, mask_special=None, rule="none", **kwargs):
        if overlap_mask is None:
            overlap_mask = torch.zeros_like(labels)
        if non_overlap_mask is None:
            non_overlap_mask = torch.ones_like(labels)
        if ignore_mask is None:
            ignore_mask = torch.ones_like(labels)
        if yield_mask is None:
            yield_mask = torch.zeros_like(labels)
        if non_yield_mask is None:
            non_yield_mask = torch.zeros_like(labels)
        if mask_special is None:
            mask_special = torch.zeros_like(labels)

        preds = preds[:, :-1, :]
        labels = labels[:, 1:]

        if self.turn_type == TurnType.NORMAL:
            overlap_mask = torch.logical_not(overlap_mask)
            mask = torch.logical_and(overlap_mask, ignore_mask)
        elif self.turn_type == TurnType.NON_OVERLAP:
            mask = torch.logical_and(non_overlap_mask, ignore_mask)
        elif self.turn_type == TurnType.OVERLAP:
            mask = torch.logical_and(overlap_mask, ignore_mask)
        elif self.turn_type == TurnType.YIELD:
            mask = torch.logical_and(ignore_mask, yield_mask)
        elif self.turn_type == TurnType.NON_YIELD:
            mask = torch.logical_and(ignore_mask, non_yield_mask)
        else:
            mask = torch.logical_and(ignore_mask, ignore_mask)

        if self.filter_bc_overlap_token:
            mask = torch.logical_and(mask, mask_special)

        mask = mask[..., 1:]

        labels = torch.where(mask, labels, torch.tensor(-100, device="cuda:0"))
        self.perplexity.update(preds, labels)

    def add_joint(self,
                  predsA, predsB,
                  labelsA, labelsB,
                  rule="none", **kwargs):
        assert self.rule == "joint", f"ERROR: Will be wrong perplexity type with {self.rule}"

        overlap_maskA = kwargs.get('overlap_maskA', torch.zeros_like(labelsA))
        overlap_maskB = kwargs.get('overlap_maskB', torch.zeros_like(labelsB))

        ignore_maskA = kwargs.get('ignore_maskA', torch.ones_like(labelsA))
        ignore_maskB = kwargs.get('ignore_maskB', torch.ones_like(labelsB))

        predsA = predsA[:, :-1, :]
        predsB = predsB[:, :-1, :]

        labelsA = labelsA[:, 1:]
        labelsB = labelsB[:, 1:]

        if self.turn_type == TurnType.NORMAL:
            overlap_maskA = torch.logical_not(overlap_maskA)
            maskA = torch.logical_and(overlap_maskA, ignore_maskA)
            maskA = maskA[..., 1:]

            overlap_maskB = torch.logical_not(overlap_maskB)
            maskB = torch.logical_and(overlap_maskB, ignore_maskB)
            maskB = maskB[..., 1:]
        elif self.turn_type == TurnType.OVERLAP:
            maskA = torch.logical_and(overlap_maskA, ignore_maskA)
            maskA = maskA[..., 1:]

            maskB = torch.logical_and(overlap_maskB, ignore_maskB)
            maskB = maskB[..., 1:]
        else:
            maskA = torch.logical_and(ignore_maskA, ignore_maskA)
            maskA = maskA[..., 1:]

            maskB = torch.logical_and(ignore_maskB, ignore_maskB)
            maskB = maskB[..., 1:]

        labelsA = torch.where(
            maskA, labelsA, torch.tensor(-100, device="cuda:0"))
        labelsB = torch.where(
            maskB, labelsB, torch.tensor(-100, device="cuda:0"))
        self.perplexity.update(predsA, predsB, labelsA, labelsB)

    def calculate(self, params=None):
        self.output = self.perplexity.compute()
        return self.output.item(), None, None

    def reset(self):
        self.perplexity.reset()

    def is_graphical(self):
        return False

    def graph(self):
        return None

    def __str__(self):
        rule = "joint" if self.rule == "joint" else ""
        filter = f"filter_" if self.filter_bc_overlap_token else "_"
        if rule + filter == "__":
            rule = ""
        return f"{rule}{filter}PPL_{self.turn_type.name}"


class ACC(Metric):
    def __init__(self, token_id: int, token: str, nthresh=50, turn_type=TurnType.NONE, tokens_dict: dict = {}, device="cuda:0"):
        super().__init__(MetricType.ACC, token_id, token,
                         turn_type, tokens_dict, nthresh, device)

        self.metric = Accuracy(
            task='multiclass', thresholds=self.thresholds).to(self.device)

    def calculate(self, params=None):
        self.output = self.metric.compute()
        self.max_output = self.output
        return self.max_output.item(), None, None

    def add(self, probs, labels, ignore_mask=None, overlap_mask=None, ** kwargs):
        if self.token_id != -1:
            token_prob = probs[..., self.token_id]
            token_prob = token_prob[..., :-1]

            labels = labels[..., 1:]
            labels = (labels == self.token_id)
        else:
            token_prob = probs
            labels = labels

        self.metric.update(token_prob, labels)

    def reset(self):
        self.metric.reset()

    def is_graphical(self):
        return False

    def graph(self):
        return None

    def __str__(self):
        return f"{self.token}_{self.metric_type.name.lower()}_{self.turn_type.name}"


class BACC(Metric):
    def __init__(self, token_id: int, token: str, rule='none', nthresh=50, turn_type=TurnType.NONE, tokens_dict: dict = {}, filter_bc_overlap_token=False, device="cuda:0"):
        super().__init__(MetricType.BACC, token_id, token,
                         nthresh, turn_type, tokens_dict, device)

        self.thresholds = torch.linspace(0, 1, steps=20, device=device)
        # self.metric = [Accuracy(
        #    task='multiclass', num_classes=2, average='macro', threshold=threshold.item()).to(self.device) for threshold in self.thresholds]
        self.tp = torch.zeros_like(self.thresholds, device=device)
        self.tn = torch.zeros_like(self.thresholds, device=device)
        self.fn = torch.zeros_like(self.thresholds, device=device)
        self.fp = torch.zeros_like(self.thresholds, device=device)

        self.count = 0

        self.rule = rule
        self.device = device

        self.filter_bc_overlap_token = filter_bc_overlap_token
        if isinstance(self.token_id, str):
            return

        if not isinstance(self.token_id, list):
            token_id = [self.token_id]
        if not isinstance(token_id, torch.Tensor):
            token_id = torch.Tensor(token_id).long().to(self.device)
        self.tokens = token_id

    def calculate(self, params=None, prefix=""):
        recall = torch.div(self.tp, self.tp + self.fn, out=torch.zeros_like(
            self.tp, dtype=float), rounding_mode=None)
        specificity = torch.div(self.tn, self.tn + self.fp, out=torch.zeros_like(
            self.tn, dtype=float), rounding_mode=None)

        bacc = (recall + specificity) / 2
        self.output = bacc
        self.max_output = torch.max(bacc)
        idx = torch.argmax(bacc)
        thresh = self.thresholds[idx].item()
        count = self.tp[idx] + self.fp[idx] + self.fn[idx] + self.tn[idx]

        if prefix[-1] != "_":
            prefix += "_"

        for i in range(len(self.thresholds)):
            logger.info(
                f"{prefix}{self.__str__()}({self.thresholds[i]}) = {bacc[i]}")

        logger.info(f"{prefix}{self.__str__()}(tp) = {self.tp[idx]}")
        logger.info(f"{prefix}{self.__str__()}(fp) = {self.fp[idx]}")
        logger.info(f"{prefix}{self.__str__()}(fn) = {self.fn[idx]}")
        logger.info(f"{prefix}{self.__str__()}(tn) = {self.tn[idx]}")
        logger.info(f"{prefix}{self.__str__()}(total) = {count}")

        if params is not None:
            logger.info(
                f"{prefix}{self.__str__()}({self.thresholds[i]}: Output for Val Params) = {bacc[i]}")
            thresh = params["val_" + str(self)]
            idx = self.thresholds.tolist().index(thresh)
            bacc = bacc[idx]
            return bacc, self.count, thresh

        return self.max_output.item(), self.count, thresh

    def add(self,
            probs,
            labels,
            yield_mask=None,
            non_yield_mask=None,
            ignore_mask=None,
            mask_special=None,
            overlap_mask=None,
            bc_mask=None,
            interrupt_mask=None,
            interrupt_mask_normal=None,
            interrupt_mask_overlap=None,
            interrupt_mask_bc=None,
            interrupt_mask_yield=None,
            turn_mask=None,
            **kwargs):
        if self.rule == "rule":
            self.add_joint(labels, ignore_mask, overlap_mask,
                           yield_mask, turn_mask, **kwargs)
            return

        if overlap_mask is None:
            overlap_mask = torch.zeros_like(labels)
        if ignore_mask is None:
            ignore_mask = torch.ones_like(labels)
        if yield_mask is None:
            yield_mask = torch.zeros_like(labels)
        if non_yield_mask is None:
            non_yield_mask = torch.zeros_like(labels)
        if interrupt_mask is None:
            interrupt_mask = torch.zeros_like(labels)
        if interrupt_mask_normal is None:
            interrupt_mask_normal = torch.zeros_like(labels)
        if interrupt_mask_bc is None:
            interrupt_mask_bc = torch.zeros_like(labels)
        if interrupt_mask_yield is None:
            interrupt_mask_yield = torch.zeros_like(labels)
        if interrupt_mask_overlap is None:
            interrupt_mask_overlap = torch.zeros_like(labels)
        if bc_mask is None:
            bc_mask = torch.zeros_like(labels)

        if self.turn_type == TurnType.NORMAL:
            mask = torch.logical_and(torch.logical_not(
                overlap_mask), ignore_mask)
        elif self.turn_type == TurnType.OVERLAP:
            mask = torch.logical_and(overlap_mask, ignore_mask)
        elif self.turn_type == TurnType.YIELD:
            mask = torch.logical_and(ignore_mask, yield_mask)
        elif self.turn_type == TurnType.NON_YIELD:
            mask = torch.logical_and(ignore_mask, non_yield_mask)
        elif self.turn_type == TurnType.BACKCHANNEL:
            mask = torch.logical_and(ignore_mask, bc_mask)
        else:
            mask = torch.logical_and(ignore_mask, ignore_mask)

        if self.filter_bc_overlap_token:
            mask = torch.logical_and(mask, mask_special)
        mask = mask[..., 1:]

        if self.token_id == 'all':
            interrupt_mask = interrupt_mask[..., 1:]
            if self.turn_type == TurnType.NORMAL:
                interrupt_mask = interrupt_mask_normal[..., 1:]
            elif self.turn_type == TurnType.OVERLAP:
                interrupt_mask = interrupt_mask_overlap[..., 1:]
            elif self.turn_type == TurnType.YIELD:
                interrupt_mask = interrupt_mask_yield[..., 1:]
            elif self.turn_type == TurnType.BACKCHANNEL:
                interrupt_mask = interrupt_mask_bc[..., 1:]

            tokens_prob = None
            new_labels = None
            token_id_handled = set()
            for token, token_id in self.special_tokens.items():
                if token_id in token_id_handled:
                    continue

                token_id_handled.add(token_id)
                if tokens_prob is None:
                    tokens_prob = probs[..., token_id].detach().clone()
                    new_labels = labels == token_id
                else:
                    tokens_prob += probs[..., token_id]
                    new_labels += labels == token_id

            tokens_prob = 1 - tokens_prob
            preds = tokens_prob[..., :-1][interrupt_mask]

            labels = torch.logical_not((new_labels > 0))
            labels = labels[..., 1:][interrupt_mask]

            for idx, threshold in enumerate(self.thresholds):
                self.tp[idx] += torch.sum((preds >= threshold) & (labels == 1))
                self.fp[idx] += torch.sum((preds >= threshold) & (labels == 0))
                self.fn[idx] += torch.sum((preds < threshold) & (labels == 1))
                self.tn[idx] += torch.sum((preds < threshold) & (labels == 0))

            return

        labels = labels[..., 1:]
        labels = labels[mask]

        token_probs, token_probs_max = torch.max(
            probs[..., self.tokens], dim=-1)
        token_prediction = self.tokens[token_probs_max]

        token_prediction = token_prediction[..., :-1]
        token_prediction = token_prediction[mask]
        token_probs = token_probs[..., :-1]
        token_probs = token_probs[mask]

        is_token = torch.eq(labels, token_prediction)
        not_token = torch.ne(labels, token_prediction)

        self.count += is_token.long().sum()

        for idx, threshold in enumerate(self.thresholds):
            self.tp[idx] += torch.sum((token_probs >= threshold) & (is_token))
            self.fp[idx] += torch.sum((token_probs >= threshold) & (not_token))
            self.fn[idx] += torch.sum((token_probs < threshold) & (is_token))
            self.tn[idx] += torch.sum((token_probs < threshold) & (not_token))

    def reset(self):
        device = self.device

        self.tp = torch.zeros_like(self.thresholds, device=device)
        self.tn = torch.zeros_like(self.thresholds, device=device)
        self.fn = torch.zeros_like(self.thresholds, device=device)
        self.fp = torch.zeros_like(self.thresholds, device=device)

        self.count = 0

    def is_graphical(self):
        return False

    def graph(self):
        return None

    def __str__(self):
        rule = f"_{self.rule}" if self.rule == "rule" else "_"
        filter = f"filter_" if self.filter_bc_overlap_token else "_"
        if rule + filter == "__":
            rule = ""
        return f"{self.token}{rule}{filter}{self.metric_type.name.lower()}_{self.turn_type.name}"

    def add_joint(self, labels, ignore_mask=None, overlap_mask=None, yield_mask=None, turn_mask=None, **kwargs):
        """
        Implement rule-based approach for predicting any turn shift, so at any point there is an
        overlap (two speakers speaking), whoever has the turn, predict a <yield>
        """
        if turn_mask is None:
            turn_mask = torch.ones_like(labels)
        if overlap_mask is None:
            overlap_mask = torch.zeros_Like(labels)
        if ignore_mask is None:
            ignore_mask = torch.ones_like(labels)
        if turn_mask is None:
            turn_mask = torch.ones_like(labels)

        # All overlaps in turns of speaker
        mask = torch.logical_and(overlap_mask, turn_mask)
        mask = torch.logical_and(mask, yield_mask)

        preds = torch.zeros_like(labels)
        preds = torch.where(mask, torch.tensor(1, device="cuda"), preds)

        labels = labels[ignore_mask.bool()]
        preds = preds[ignore_mask.bool()]

        tokens = self.token_id
        if not isinstance(self.token_id, list):
            tokens = [self.token_id]

        for token in tokens:
            is_token = (labels == token)
            not_token = (labels != token)

            for idx, threshold in enumerate(self.thresholds):
                self.tp[idx] += torch.sum((preds >= threshold) & (is_token))
                self.fp[idx] += torch.sum((preds >= threshold) & (not_token))
                self.fn[idx] += torch.sum((preds < threshold) & (is_token))
                self.tn[idx] += torch.sum((preds < threshold) & (not_token))


class ROC_AUC(Metric):
    def __init__(self, token_id: int, token: str, nthresh=50, turn_type=TurnType.NONE, tokens_dict: dict = {}, device="cuda:0"):
        super().__init__(MetricType.ROC_AUC, token_id,
                         token, nthresh, turn_type, tokens_dict, device)

        self.metric = ROC(
            task='multiclass', thresholds=self.thresholds, num_classes=1).to(self.device)
        self.auc = AUC()

    def calculate(self, params=None):
        fpr, tpr, thresholds = self.metric.compute()

        sort_indices = torch.argsort(fpr, descending=True)
        sorted_fpr = fpr[sort_indices]
        sorted_tpr = tpr[sort_indices]

        # Calculate PR AUC
        self.auc.update(sorted_fpr, sorted_tpr)
        roc_auc = self.auc.compute()

        self.output = roc_auc
        self.max_output = roc_auc
        return self.max_output.item(), None, None

    def add(self, probs, labels, yield_mask=None, non_yield_mask=None, ignore_mask=None, overlap_mask=None, interrupt_mask=None, **kwargs):
        if overlap_mask is None:
            overlap_mask = torch.zeros_like(labels)
        if ignore_mask is None:
            ignore_mask = torch.ones_like(labels)
        if yield_mask is None:
            yield_mask = torch.zeros_like(labels)
        if non_yield_mask is None:
            non_yield_mask = torch.zeros_like(labels)
        if interrupt_mask is None:
            interrupt_mask = torch.zeros_like(labels)

        if self.turn_type == TurnType.NORMAL:
            mask = torch.logical_and(torch.logical_not(
                overlap_mask), ignore_mask)
            mask = mask[..., 1:]
        elif self.turn_type == TurnType.OVERLAP:
            mask = torch.logical_and(overlap_mask, ignore_mask)
            mask = mask[..., 1:]
        elif self.turn_type == TurnType.YIELD:
            mask = torch.logical_and(ignore_mask, yield_mask)
            mask = mask[..., 1:]
        elif self.turn_type == TurnType.NON_YIELD:
            mask = torch.logical_and(ignore_mask, non_yield_mask)
            mask = mask[..., 1:]
        else:
            mask = torch.logical_and(ignore_mask, ignore_mask)
            mask = mask[..., 1:]
        tokens = self.token_id

        if not isinstance(tokens, list):
            tokens = [self.token_id]

        # Predicting the start of a new turn
        if self.token_id == 'all':
            interrupt_mask = interrupt_mask[..., 1:]
            tokens_prob = None
            new_labels = None
            for token, token_id in self.special_tokens.items():
                if tokens_prob is None:
                    tokens_prob = probs[..., token_id].detach().clone()
                    new_labels = labels == token_id
                else:
                    tokens_prob += probs[..., token_id]
                    new_labels += labels == token_id

            tokens_prob = 1 - tokens_prob
            tokens_prob = tokens_prob[..., :-1][interrupt_mask]

            labels = torch.logical_not((new_labels > 0))
            labels = labels[..., 1:][interrupt_mask]

            self.metric.update(tokens_prob, labels)
            return

        labels = labels[..., 1:]
        labels = (labels == self.token_id)
        labels = labels[mask]

        if mask.sum() == 0:
            return

        for token in tokens:
            token_prob = probs[..., token]
            token_prob = token_prob[..., :-1]

            token_prob = token_prob[mask]

            self.metric.update(token_prob, labels)

    def reset(self):
        self.metric.reset()
        self.auc.reset()

    def is_graphical(self):
        return True

    def graph(self):
        if not self.is_graphical():
            return None

        a, b, t = self.metric.compute()
        computed = (b, a, None)

        optimal_idxs = torch.where(
            torch.concatenate(
                [torch.tensor([True], device=self.device), torch.logical_or(torch.diff(
                    computed[0][:-1]), torch.diff(computed[0][1:])), torch.tensor([True], device=self.device)]
            )
        )[0]

        labels = ("FPR", "TPR")
        computed = (computed[1][optimal_idxs], computed[0][optimal_idxs], None)

        score = _auc_compute_without_check(computed[0], computed[1], 1.0)

        fig, axes = plot_curve(
            curve=computed, score=score, label_names=labels, name=str(self)
        )
        fig.suptitle(str(self))

        return fig, axes

    def __str__(self):
        return f"{self.token}_{self.metric_type.name.lower()}_{self.turn_type.name}"


class F1_Score(Metric):
    def __init__(self, token_id: int, token: str, nthresh=50, turn_type=TurnType.NONE, rule="none", tokens_dict: dict = {}, device="cuda:0"):
        super().__init__(MetricType.F1_SCORE, token_id,
                         token, turn_type, nthresh, tokens_dict, device)

        self.metric = F1Score(task='binary').to(self.device)
        self.auc = AUC()

        self.rule = rule

    def calculate(self, params=None):
        self.output = self.metric.compute()
        self.max_output = self.output
        return self.max_output.item(), None, None

    def add(self, probs, labels, yield_mask=None, non_yield_mask=None, overlap_mask=None, ignore_mask=None, interrupt_mask=None, **kwargs):
        if overlap_mask is None:
            overlap_mask = torch.zeros_like(labels)
        if ignore_mask is None:
            ignore_mask = torch.ones_like(labels)
        if yield_mask is None:
            yield_mask = torch.zeros_like(labels)
        if non_yield_mask is None:
            non_yield_mask = torch.zeros_like(labels)
        if interrupt_mask is None:
            interrupt_mask = torch.zeros_like(labels)

        labels = labels[..., 1:]
        labels = (labels == self.token_id)

        if self.turn_type == TurnType.NORMAL:
            mask = torch.logical_and(torch.logical_not(
                overlap_mask), ignore_mask)
            mask = mask[..., 1:]
        elif self.turn_type == TurnType.OVERLAP:
            mask = torch.logical_and(overlap_mask, ignore_mask)
            mask = mask[..., 1:]
        elif self.turn_type == TurnType.NON_YIELD:
            mask = torch.logical_and(ignore_mask, non_yield_mask)
            mask = mask[..., 1:]
        elif self.turn_type == TurnType.YIELD:
            mask = torch.logical_and(ignore_mask, yield_mask)
            mask = mask[..., 1:]
        else:
            mask = torch.logical_and(ignore_mask, ignore_mask)
            mask = mask[..., 1:]
        tokens = self.token_id

        if not isinstance(tokens, list):
            tokens = [self.token_id]

        labels = labels[mask]

        if self.token_id == 'all':
            interrupt_mask = interrupt_mask[..., 1:]
            tokens_prob = None
            new_labels = None
            for token, token_id in self.special_tokens.items():
                if tokens_prob is None:
                    tokens_prob = probs[..., token_id].detach().clone()
                    new_labels = labels == token_id
                else:
                    tokens_prob += probs[..., token_id]
                    new_labels += labels == token_id

            tokens_prob = 1 - tokens_prob
            tokens_prob = tokens_prob[..., :-1][interrupt_mask]

            labels = torch.logical_not((new_labels > 0))
            labels = labels[..., 1:][interrupt_mask]

            self.metric.update(tokens_prob, labels)
            return

        if mask.sum() == 0:
            return

        for token in tokens:
            token_prob = probs[..., token]
            preds = token_prob[..., :-1]

            preds = preds[mask]

            self.metric.update(preds, labels)

    def reset(self):
        self.metric.reset()

    def is_graphical(self):
        return False

    def graph(self):
        return None

    def __str__(self):
        return f"{self.token}_{self.metric_type.name.lower()}"


class PR_AUC(Metric):
    def __init__(self, token_id: int, token: str, rule="none", nthresh=50, turn_type=TurnType.NONE, tokens_dict: dict = {}, device="cuda:0"):
        super().__init__(MetricType.PR_AUC, token_id, token,
                         nthresh, turn_type, tokens_dict, device)

        self.metric = PrecisionRecallCurve(
            task='binary', thresholds=self.thresholds).to(self.device)
        self.auc = AUC()

        self.rule = rule

    def calculate(self, params=None):
        precision, recall, thresholds = self.metric.compute()

        sort_indices = torch.argsort(recall, descending=False)
        sorted_recall = recall[sort_indices]
        sorted_precision = precision[sort_indices]

        # Calculate PR AUC
        self.auc.update(sorted_recall, sorted_precision)
        pr_auc = self.auc.compute()

        self.output = pr_auc
        self.max_output = pr_auc
        return self.max_output.item(), None, None

    def add(self, probs, labels, yield_mask=None, non_yield_mask=None, overlap_mask=None, ignore_mask=None, interrupt_mask=None, ** kwargs):
        if overlap_mask is None:
            overlap_mask = torch.zeros_like(labels)
        if ignore_mask is None:
            ignore_mask = torch.ones_like(labels)
        if yield_mask is None:
            yield_mask = torch.zeros_like(labels)
        if non_yield_mask is None:
            non_yield_mask = torch.ones_like(labels)
        if interrupt_mask is None:
            interrupt_mask = torch.zeros_like(labels)

        if self.turn_type == TurnType.NORMAL:
            # NORMAL: Non-Overlap
            mask = torch.logical_and(torch.logical_not(
                overlap_mask), ignore_mask)
            mask = mask[..., 1:]
        elif self.turn_type == TurnType.NON_YIELD:
            mask = torch.logical_and(ignore_mask, non_yield_mask)
            mask = mask[..., 1:]
        elif self.turn_type == TurnType.OVERLAP:
            mask = torch.logical_and(overlap_mask, ignore_mask)
            mask = mask[..., 1:]
        elif self.turn_type == TurnType.YIELD:
            mask = torch.logical_and(ignore_mask, yield_mask)
            mask = mask[..., 1:]
        else:
            mask = torch.logical_and(ignore_mask, ignore_mask)
            mask = mask[..., 1:]
        tokens = self.token_id

        if self.token_id == 'all':
            interrupt_mask = interrupt_mask[..., 1:]
            tokens_prob = None
            new_labels = None

            token_id_handled = set()
            for token, token_id in self.special_tokens.items():
                if token_id in token_id_handled:
                    continue

                token_id_handled.add(token_id)
                if tokens_prob is None:
                    tokens_prob = probs[..., token_id].detach().clone()
                    new_labels = labels == token_id
                else:
                    tokens_prob += probs[..., token_id]
                    new_labels += labels == token_id

            tokens_prob = 1 - tokens_prob
            tokens_prob = tokens_prob[..., :-1][interrupt_mask]

            labels = torch.logical_not((new_labels > 0))
            labels = labels[..., 1:][interrupt_mask]

            self.metric.update(tokens_prob, labels)
            return

        if mask.sum() == 0:
            return

        labels = labels[..., 1:]
        labels = (labels == self.token_id)
        labels = labels[mask]
        if not isinstance(tokens, list):
            tokens = [self.token_id]

        for token in tokens:
            token_prob = probs[..., token]
            token_prob = token_prob[..., :-1]

            token_prob = token_prob[mask]

            self.metric.update(token_prob, labels)

    def reset(self):
        self.metric.reset()
        self.auc.reset()

    def is_graphical(self):
        return True

    def graph(self):
        if not self.is_graphical():
            return None

        a, b, t = self.metric.compute()
        computed = (b, a, None)

        optimal_idxs = torch.where(
            torch.concatenate(
                [torch.tensor([True], device=self.device), torch.logical_or(torch.diff(
                    computed[0][:-1]), torch.diff(computed[0][1:])), torch.tensor([True], device=self.device)]
            )
        )[0]

        labels = ("Recall", "Precision")
        computed = (computed[0][optimal_idxs], computed[1][optimal_idxs], None)

        score = _auc_compute_without_check(computed[0], computed[1], 1.0)

        fig, axes = plot_curve(
            curve=computed, score=score, label_names=labels, name=str(self)
        )
        fig.suptitle(str(self))

        return fig, axes

    def __str__(self):
        return f"{self.token}_{self.metric_type.name.lower()}_{self.turn_type.name}"

    def add_joint(self, labelsA, labelsB, maskA, maskB):
        pass


class BargeRate(Metric):
    """
    Barge-in rate metric so where threshold for eot is breached prior to actual 
    turn end 
    """

    def __init__(self, token_id: int, token: str, rule="none", nthresh=50, turn_type=TurnType.NONE,
                 tokens_dict: dict = {}, device="cuda:0"):
        super().__init__(MetricType.BR, token_id, token,
                         nthresh, turn_type, tokens_dict, device)

        self.total_turns = 0
        self.barge_in = torch.zeros_like(self.thresholds)
        self.non_barge_in = torch.zeros_like(self.thresholds)

        self.rule = rule

    def add(self, probs, labels, yield_mask=None, non_yield_mask=None, overlap_mask=None, ignore_mask=None, interrupt_mask=None, turn_mask=None, ** kwargs):
        """
        Find all turns from labels and identify if probs within that utterance are all greater then thresholds
        At all points in turn aside from the end

        Arguments:
            turn_mask: torch.Tensor
                Mask of all turns that are uttered by the speaker
        """
        if self.turn_type == TurnType.YIELD:
            mask = torch.logical_and(yield_mask, turn_mask)
        elif self.turn_type == TurnType.NON_YIELD:
            mask = torch.logical_and(non_yield_mask, turn_mask)
        else:
            mask = torch.ones_like(turn_mask)

        max_probs = []

        probs = probs[..., self.token_id]
        probs = probs[..., :-1]
        mask = mask[..., 1:]
        labels = labels[..., 1:]

        seq_len = labels.size()[-1]

        batch_labels = torch.cumsum(torch.ones_like(labels), dim=0) - 1

        turn_mask = turn_mask[..., 1:].long()

        turn_changes = torch.cat((turn_mask[:, :1], torch.abs(
            turn_mask[:, 1:] - turn_mask[:, :-1])), dim=1)
        turn_ids = torch.cumsum(turn_changes, dim=1)

        global_turn_ids = (turn_ids + seq_len * batch_labels) * turn_mask
        unique_ids = torch.unique(global_turn_ids)

        probs = probs[mask]
        labels = labels[mask]
        global_turn_ids = global_turn_ids[mask]

        if mask.sum() == 0:
            return

        for turn_id in unique_ids:
            single_turn_mask = global_turn_ids == turn_id
            if single_turn_mask.sum() == 0 or turn_id == 0 or self.token_id != labels[single_turn_mask][-1]:
                continue

            single_turn_mask = torch.logical_and(
                single_turn_mask, self.token_id != labels)
            if single_turn_mask.sum() == 0:
                continue

            max_prob = torch.max(probs[single_turn_mask])
            max_probs.append(max_prob.item())

        for idx, threshold in enumerate(self.thresholds):
            self.barge_in[idx] += sum(x >= threshold for x in max_probs)
            self.non_barge_in[idx] += sum(x < threshold for x in max_probs)
        self.total_turns += len(max_probs)

    def calculate(self, params=None):
        if params is None:
            params = {}

        # Use best performing thresh for BACC
        # Fallback if not available
        thresh = params.get("val_eot_bacc_NONE", 2)
        if thresh != 2:
            idx = self.thresholds.tolist().index(thresh)
        else:
            idx = thresh

        for thresh in self.thresholds:
            i = self.thresholds.tolist().index(thresh)
            logger.info(
                f"BR: {thresh} = {self.barge_in[i] / self.total_turns}")

        logger.info(f"BR: using thresh {self.thresholds[idx]}")
        barge_in = self.barge_in[idx] / \
            (self.non_barge_in[idx] + self.barge_in[idx])

        """
        assert barge_in == self.barge_in[idx] / \
            self.total_turns, f"ERROR: Barge-In calculation is wrong {barge_in} != {self.barge_in[idx] / self.total_turns}"
        """

        return barge_in, self.total_turns, self.thresholds[idx]

    def reset(self):
        self.total_turns = 0
        self.barge_in = torch.zeros_like(self.thresholds)
        self.non_barge_in = torch.zeros_like(self.thresholds)

    def __str__(self):
        return f"{self.token}_{self.turn_type}_BR"


class NRR(Metric):
    """
    No response rate metric
    """

    def __init__(self, token_id: int, token: str, rule="none", nthresh=50, turn_type=TurnType.NONE,
                 tokens_dict: dict = {}, device="cuda:0"):
        super().__init__(MetricType.NRR, token_id, token,
                         nthresh, turn_type, tokens_dict, device)

        self.total_turns = 0
        self.no_response_turns = torch.zeros_like(self.thresholds)
        self.response_turns = torch.zeros_like(self.thresholds)

        self.rule = rule

    def add(self, probs, labels, yield_mask=None, non_yield_mask=None, overlap_mask=None, ignore_mask=None, interrupt_mask=None, turn_mask=None, ** kwargs):
        """
        Find all turns from labels and identify if probs within that utterance are all less then thresholds 

        Arguments:
            turn_mask: torch.Tensor
                Mask of all turns that are uttered by the speaker
        """
        if self.turn_type == TurnType.YIELD:
            mask = torch.logical_and(yield_mask, turn_mask)
        elif self.turn_type == TurnType.NON_YIELD:
            mask = torch.logical_and(non_yield_mask, turn_mask)
        else:
            mask = torch.ones_like(turn_mask)

        max_probs = []

        probs = probs[..., self.token_id]
        probs = probs[..., :-1]
        mask = mask[..., 1:]
        labels = labels[..., 1:]

        seq_len = labels.size()[-1]

        batch_labels = torch.cumsum(torch.ones_like(labels), dim=0) - 1

        turn_mask = turn_mask[..., 1:].long()

        turn_changes = torch.cat((turn_mask[:, :1], torch.abs(
            turn_mask[:, 1:] - turn_mask[:, :-1])), dim=1)
        turn_ids = torch.cumsum(turn_changes, dim=1)

        global_turn_ids = (turn_ids + seq_len * batch_labels) * turn_mask
        unique_ids = torch.unique(global_turn_ids)

        probs = probs[mask]
        labels = labels[mask]
        global_turn_ids = global_turn_ids[mask]

        if mask.sum() == 0:
            return

        for turn_id in unique_ids:
            single_turn_mask = global_turn_ids == turn_id
            if single_turn_mask.sum() == 0 or turn_id == 0 or self.token_id != labels[single_turn_mask][-1]:
                continue

            max_prob = torch.max(probs[single_turn_mask])
            max_probs.append(max_prob.item())

        for idx, threshold in enumerate(self.thresholds):
            self.no_response_turns[idx] += sum(x <
                                               threshold for x in max_probs)
            self.response_turns[idx] += sum(x >= threshold for x in max_probs)

        self.total_turns += len(max_probs)

        return

    def calculate(self, params=None):
        if params is None:
            params = {}

        # Use best performing thresh for BACC
        # Fallback if not available
        thresh = params.get("val_eot_bacc_NONE", 2)
        if thresh != 2:
            idx = self.thresholds.tolist().index(thresh)
        else:
            idx = thresh

        for thresh in self.thresholds:
            i = self.thresholds.tolist().index(thresh)
            logger.info(
                f"NRR: {thresh} = {self.no_response_turns[i] / self.total_turns}")

        logger.info(f"NRR: using thresh {self.thresholds[idx]}")
        nrr = self.no_response_turns[idx] / \
            (self.response_turns[idx] + self.no_response_turns[idx])

        """
        assert abs(nrr - self.no_response_turns[idx]) < 0.01 or torch.isnan(nrr) / \
            self.total_turns, f"ERROR: NRR calculation is wrong {nrr} != {self.no_response_turns[idx] / self.total_turns}"
        """

        return nrr, self.total_turns, self.thresholds[idx]

    def reset(self):
        self.total_turns = 0
        self.no_response_turns = torch.zeros_like(self.thresholds)
        self.response_turns = torch.zeros_like(self.thresholds)

    def __str__(self):
        return f"{self.token}_{self.turn_type}_NRR"


class Metrics:
    def __init__(self, metric_config: list, tokens_dict: dict = {}, type='test', device="cuda:0"):
        metrics = []
        output = {}
        parameters = {}
        counts = {}

        self.type = type + "_"
        filter_bc_overlap_token = False

        for metric_conf in metric_config:
            if len(metric_conf) == 6:
                filter_bc_overlap_token = metric_conf[-1]

            if len(metric_conf) == 5:
                # Rule-based System
                rule, token_id, token, metric_list, turn_type = metric_conf
                for metric_item in metric_list:
                    if metric_item == MetricType.PR_AUC:
                        metric = PR_AUC(
                            token_id, token, rule=rule, turn_type=turn_type, tokens_dict=tokens_dict, device=device)
                    elif metric_item == MetricType.BACC:
                        metric = BACC(token_id, token, rule=rule, turn_type=turn_type,
                                      tokens_dict=tokens_dict, device=device)
                    elif metric_item == MetricType.PERPLEXITY:
                        metric = Perplexity_Score(
                            rule=rule, turn_type=turn_type, device=device)
                    metrics.append(metric)
                continue

            token_id, token, metric_list, turn_type = metric_conf[:4]
            for metric_item in metric_list:
                if metric_item == MetricType.PR_AUC:
                    metric = PR_AUC(token_id, token,
                                    turn_type=turn_type, tokens_dict=tokens_dict, device=device)
                elif metric_item == MetricType.ROC_AUC:
                    metric = ROC_AUC(
                        token_id, token, turn_type=turn_type, tokens_dict=tokens_dict, device=device)
                elif metric_item == MetricType.F1_SCORE:
                    metric = F1_Score(
                        token_id, token, turn_type=turn_type, tokens_dict=tokens_dict, device=device)
                elif metric_item == MetricType.BACC:
                    metric = BACC(token_id, token,
                                  turn_type=turn_type, tokens_dict=tokens_dict, filter_bc_overlap_token=filter_bc_overlap_token, device=device)
                elif metric_item == MetricType.PERPLEXITY:
                    metric = Perplexity_Score(
                        turn_type=turn_type, filter_bc_overlap_token=filter_bc_overlap_token, device=device)
                elif metric_item == MetricType.NRR:
                    metric = NRR(token_id, token,
                                 turn_type=turn_type, device=device)
                elif metric_item == MetricType.BR:
                    metric = BargeRate(token_id, token,
                                       turn_type=turn_type, device=device)
                else:
                    raise NameError(
                        f"NO METRIC GATHERED for: {metric_item.name}")
                metrics.append(metric)
                filter_bc_overlap_token = False

            output[self.type+str(metric)] = None
            parameters[str(metric)] = None
            counts[self.type+str(metric)] = None

        self.metrics = metrics
        self.output = output
        self.params = parameters
        self.counts = counts

    def add(self, probs, labels, logits=None, input_idsA=None, input_idsB=None, overlap_maskA=None, overlap_maskB=None, **kwargs):
        for metric in self.metrics:
            if metric.rule == "joint":
                continue

            if isinstance(metric, Perplexity_Score):
                metric.add(logits, labels.detach().clone(), **kwargs)
            else:
                metric.add(probs, labels, **kwargs)

    def add_joint(self, logitsA, logitsB, rule="none", input_idsA=None, input_idsB=None, ignore_maskA=None, ignore_maskB=None, overlap_maskA=None, overlap_maskB=None, **kwargs):
        for metric in self.metrics:
            if rule == "joint" and metric.rule == "joint" and input_idsA is not None and input_idsB is not None:
                metric.add_joint(predsA=logitsA,
                                 predsB=logitsB,
                                 labelsA=input_idsA,
                                 labelsB=input_idsB,
                                 ignore_maskA=ignore_maskA,
                                 ignore_maskB=ignore_maskB,
                                 overlap_maskA=overlap_maskA,
                                 overlap_maskB=overlap_maskB)

    def calculate(self, params=None, **kwargs):
        for metric in self.metrics:
            output, count, thresh = metric.calculate(params, **kwargs)
            self.output[self.type+str(metric)] = output
            self.counts[self.type+str(metric)] = count
            self.params[self.type+str(metric)] = thresh

        return self.output, self.counts, self.params

    def get_graphs(self):
        graphs = {}
        for metric in self.metrics:
            if metric.is_graphical():
                if metric.token not in graphs:
                    graphs[metric.token] = {}
                graphs[metric.token][self.type+str(metric)] = metric.graph()

        return graphs

    def reset(self):
        for metric in self.metrics:
            metric.reset()
