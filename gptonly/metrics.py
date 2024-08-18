import logging
from enum import IntEnum
from abc import ABC, abstractmethod

from torchmetrics import PrecisionRecallCurve, F1Score, ROC, Accuracy
from torchmetrics.text import Perplexity
from torchmetrics.utilities.compute import _auc_compute_without_check
from torchmetrics.utilities.plot import plot_curve
from torcheval.metrics.aggregation.auc import AUC

import torch

from pairwise_generation_dm import TurnType

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


class Metric():
    def __init__(self, metric_type: MetricType, token_id, token: str, nthresh=50, device="cuda:0"):
        self.metric_type = metric_type
        self.token_id = token_id
        self.token = token
        self.nthresh = nthresh
        self.device = device

        self.thresholds = torch.linspace(0, 1, 20, device=device)

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
    def __init__(self, device="cuda:0", turn_type=TurnType.NONE, pad_token_id=-100):
        self.emp_index = pad_token_id
        self.turn_type = turn_type
        self.perplexity = Perplexity(
            ignore_index=self.emp_index).to(device=device)

    def add(self, preds, labels, overlap_mask=None, yield_mask=None, ignore_mask=None, type='token', **kwargs):
        if overlap_mask is None:
            overlap_mask = torch.zeros_like(labels)
        if ignore_mask is None:
            ignore_mask = torch.ones_like(labels)
        if yield_mask is None:
            yield_mask = torch.zeros_like(labels)

        if type != 'token':
            return

        preds = preds[:, :-1, :]
        labels = labels[:, 1:]

        if self.turn_type == TurnType.NORMAL:
            mask = torch.logical_and(
                torch.logical_not(overlap_mask), ignore_mask)
        elif self.turn_type == TurnType.OVERLAP:
            mask = torch.logical_and(
                overlap_mask, ignore_mask)
        elif self.turn_type == TurnType.YIELD:
            mask = torch.logical_and(ignore_mask, yield_mask)
        else:
            mask = torch.logical_and(ignore_mask, ignore_mask)
        mask = mask[:, 1:]

        labels = torch.where(mask.bool(), labels,
                             torch.tensor(-100, device="cuda:0"))
        self.perplexity.update(preds, labels)

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
        return f"PPL_{self.turn_type.name}"


class ACC(Metric):
    def __init__(self, token_id: int, token: str, turn_type=TurnType.NONE, nthresh=50, bins=20, device="cuda:0"):
        super().__init__(MetricType.ACC, token_id, token, nthresh, device)

        self.bins = bins
        self.turn_type = turn_type

        self.count = 0

        task = 'multiclass' if self.bins > 1 else 'binary'
        self.metric = Accuracy(
            task=task, num_classes=self.bins, threshold=self.thresholds).to(self.device)

    def calculate(self, params=None):
        self.output = self.metric.compute()
        self.max_output = self.output
        return self.max_output.item(), self.count, 0

    def add(self, probs, labels, overlap_mask=None, ignore_mask=None, yield_mask=None, type='token', **kwargs):
        if overlap_mask is None:
            overlap_mask = torch.zeros_like(labels)
        if ignore_mask is None:
            ignore_mask = torch.ones_like(labels)
        if yield_mask is None:
            yield_mask = torch.zeros_like(labels)

        if self.token_id != -1 and type == 'token':
            labels = labels[..., 1:]
            labels = (labels == self.token_id)

            self.count += labels.sum()

            if self.turn_type == TurnType.NORMAL:
                mask = torch.logical_and(ignore_mask,
                                         torch.logical_not(overlap_mask))
            elif self.turn_type == TurnType.OVERLAP:
                mask = torch.logical_and(overlap_mask, ignore_mask)
            elif self.turn_type == TurnType.YIELD:
                mask = torch.logical_and(ignore_mask, yield_mask)
            else:
                mask = torch.logical_and(ignore_mask, ignore_mask)
            mask = mask[..., 1:]
            labels = labels[mask]

            tokens = self.token_id
            if not isinstance(self.token, list):
                tokens = [self.token_id]

            if len(tokens) == 2:
                token_prob = torch.where(probs[..., tokens[0]] > probs[..., tokens[1]], probs[..., tokens[0]],
                                         probs[..., tokens[1]])
            else:
                token_prob = probs[..., tokens[0]]

            token_prob = token_prob[..., :-1]

            token_prob = token_prob[mask]
            self.metric.update(token_prob, labels)

        elif type == 'projection' and self.bins == probs.shape[-1]:
            # MAX Category ATM
            labels[labels == -100] = 19

            token_prob = probs.reshape((probs.shape[0] * probs.shape[1], -1))
            labels = labels.reshape(-1)
            self.metric.update(token_prob, labels)
        else:
            return

    def reset(self):
        self.metric.reset()

    def is_graphical(self):
        return False

    def graph(self):
        return None

    def __str__(self):
        return f"{self.token}_{self.metric_type.name.lower()}_{self.turn_type.name}"


class BACC(Metric):
    def __init__(self, token_id: int, token: str, turn_type=TurnType.NONE, nthresh=50, device="cuda:0"):
        super().__init__(MetricType.BACC, token_id, token, nthresh, device)

        self.logger = logging.getLogger(__name__)
        self.turn_type = turn_type

        self.thresholds = torch.linspace(0, 1, steps=20, device=device)
        # self.metric = [Accuracy(
        #    task='multiclass', num_classes=2, average='macro', threshold=threshold.item()).to(self.device) for threshold in self.thresholds]
        self.tp = torch.zeros_like(self.thresholds, device=device)
        self.tn = torch.zeros_like(self.thresholds, device=device)
        self.fn = torch.zeros_like(self.thresholds, device=device)
        self.fp = torch.zeros_like(self.thresholds, device=device)

        self.count = 0

        self.device = device

    def calculate(self, params=None):
        recall = torch.div(self.tp, self.tp + self.fn, out=torch.zeros_like(
            self.tp, dtype=float), rounding_mode=None)
        specificity = torch.div(self.tn, self.tn + self.fp, out=torch.zeros_like(
            self.tn, dtype=float), rounding_mode=None)

        bacc = (recall + specificity) / 2
        self.output = bacc
        self.max_output = torch.max(bacc)
        idx = torch.argmax(bacc)

        count = self.tp[idx] + self.fp[idx] + self.fn[idx] + self.tn[idx]

        for i in range(len(self.thresholds)):
            logger.info(f"{self.__str__()}({self.thresholds[i]}) = {bacc[i]}")

        self.logger.info(f"{self.__str__()}(tp) = {self.tp[idx]}")
        self.logger.info(f"{self.__str__()}(fp) = {self.fp[idx]}")
        self.logger.info(f"{self.__str__()}(fn) = {self.fn[idx]}")
        self.logger.info(f"{self.__str__()}(tn) = {self.tn[idx]}")
        self.logger.info(f"{self.__str__()}(total) = {count}")

        if params is not None:
            thresh = params["val_" + str(self)]
            idx = self.thresholds.tolist().index(thresh)
            bacc = bacc[idx]
            return bacc, self.count, thresh

        return self.max_output.item(), self.count, self.thresholds[idx].item()

    def add(self, probs, labels, ignore_mask=None, overlap_mask=None, yield_mask=None, type='token', ignore_index=-100,
            **kwargs):
        if overlap_mask is None:
            overlap_mask = torch.zeros_like(labels)
        if yield_mask is None:
            yield_mask = torch.zeros_like(labels)
        if ignore_mask is None:
            ignore_mask = torch.ones_like(labels)

        if self.token_id != -1 and type == 'token':
            if self.turn_type == TurnType.NORMAL:
                mask = torch.logical_and(
                    torch.logical_not(overlap_mask), ignore_mask)
            elif self.turn_type == TurnType.OVERLAP:
                mask = torch.logical_and(
                    overlap_mask, ignore_mask)
            elif self.turn_type == TurnType.YIELD:
                mask = torch.logical_and(ignore_mask, yield_mask)
            else:
                mask = torch.logical_and(ignore_mask, ignore_mask)
            mask = mask[..., 1:]

            tokens = self.token_id
            if not isinstance(tokens, list):
                tokens = [self.token_id]

            if isinstance(labels, bool):
                print(labels)
                print(self)

            labels = labels[..., 1:]
            probs = probs[:, :-1, :]
            if len(tokens) == 2:
                token_prob = torch.where(probs[..., tokens[0]] > probs[..., tokens[1]], probs[..., tokens[0]],
                                         probs[..., tokens[1]])
                is_token = torch.where(probs[..., tokens[0]] > probs[..., tokens[1]],
                                       labels == tokens[0],
                                       labels == tokens[1])
                not_token = torch.logical_not(is_token)
            else:
                token_prob = probs[..., tokens[0]]
                is_token = (labels == tokens[0])
                not_token = (labels != tokens[0])

            preds = token_prob[mask]
            is_token = is_token[mask]
            not_token = not_token[mask]

            if isinstance(is_token, bool):
                print(labels)
                print(preds)
                print(labels.shape, preds.shape, token_prob.shape, mask.shape)
                print(is_token)
                print(self.token_id)
            self.count += is_token.sum()

            for idx, threshold in enumerate(self.thresholds):
                self.tp[idx] += torch.sum((preds >= threshold) & (is_token))
                self.fp[idx] += torch.sum((preds >= threshold) & (not_token))
                self.fn[idx] += torch.sum((preds < threshold) & (is_token))
                self.tn[idx] += torch.sum((preds < threshold) & (not_token))
        else:
            # Projection cannot be used as bacc is only for binary case
            return

    def reset(self):
        device = self.device

        self.tp = torch.zeros_like(self.thresholds, device=device)
        self.tn = torch.zeros_like(self.thresholds, device=device)
        self.fn = torch.zeros_like(self.thresholds, device=device)
        self.fp = torch.zeros_like(self.thresholds, device=device)

    def is_graphical(self):
        return False

    def graph(self):
        return None

    def __str__(self):
        return f"{self.token}_{self.metric_type.name.lower()}_{self.turn_type.name}"


class ROC_AUC(Metric):
    def __init__(self, token_id: int, token: str, turn_type=TurnType.NONE, nthresh=50, bins=1, device="cuda:0"):
        super().__init__(MetricType.ROC_AUC, token_id, token, nthresh, device)

        self.bins = bins
        self.turn_type = turn_type

        task = 'multiclass' if self.bins > 1 else 'binary'
        self.metric = ROC(task=task, num_classes=bins,
                          thresholds=self.thresholds, ignore_index=-100).to(self.device)
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

    def add(self, probs, labels, ignore_mask=None, yield_mask=None, overlap_mask=None, type='token', **kwargs):
        if overlap_mask is None:
            overlap_mask = torch.zeros_like(labels)
        if ignore_mask is None:
            ignore_mask = torch.ones_like(labels)
        if yield_mask is None:
            yield_mask = torch.zeros_like(labels)

        if self.token_id != -1 and type == 'token':
            labels = labels[..., 1:]
            labels = (labels == self.token_id)

            if self.turn_type == TurnType.NORMAL:
                mask = torch.logical_and(ignore_mask,
                                         torch.logical_not(overlap_mask))
            elif self.turn_type == TurnType.OVERLAP:
                mask = torch.logical_and(overlap_mask, ignore_mask)
            elif self.turn_type == TurnType.YIELD:
                mask = torch.logical_and(ignore_mask, yield_mask)
            else:
                mask = torch.logical_and(ignore_mask, ignore_mask)
            mask = mask[..., 1:]
            labels = labels[mask]

            tokens = self.token_id
            if not isinstance(tokens, list):
                tokens = [self.token_id]

            for token in tokens:
                token_prob = probs[..., token]
                token_prob = token_prob[..., :-1]

                token_prob = token_prob[mask]
                self.metric.update(token_prob, labels)

        elif type == 'projection' and self.bins == probs.shape[-1]:
            token_prob = probs.reshape((probs.shape[0] * probs.shape[1], -1))
            labels = labels.reshape(-1)
            self.metric.update(token_prob, labels)
        else:
            return

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
    def __init__(self, token_id: int, token: str, turn_type=TurnType.NONE, nthresh=50, bins=1, device="cuda:0"):
        super().__init__(MetricType.F1_SCORE, token_id, token, nthresh, device)

        self.turn_type = turn_type
        self.bins = bins
        task = 'multiclass' if self.bins > 1 else 'binary'
        self.metric = F1Score(task=task, num_classes=bins, average='macro',
                              ignore_index=-100).to(self.device)
        self.auc = AUC()

    def calculate(self, params=None):
        self.output = self.metric.compute()
        self.max_output = self.output
        return self.max_output.item(), None, None

    def add(self, probs, labels, ignore_mask=None, yield_mask=None, overlap_mask=None, type='token', **kwargs):
        if overlap_mask is None:
            overlap_mask = torch.zeros_like(labels)
        if yield_mask is None:
            yield_mask = torch.ones_like(labels)
        if ignore_mask is None:
            ignore_mask = torch.zeros_like(labels)

        if self.token_id != -1 and type == 'token':

            if self.turn_type == TurnType.NORMAL:
                mask = torch.logical_and(ignore_mask,
                                         torch.logical_not(overlap_mask))
            elif self.turn_type == TurnType.OVERLAP:
                mask = torch.logical_and(overlap_mask, ignore_mask)
            elif self.turn_type == TurnType.YIELD:
                mask = torch.logical_and(ignore_mask, yield_mask)
            else:
                mask = torch.logical_and(ignore_mask, ignore_mask)
            mask = mask[..., 1:]

            labels = labels[..., 1:]
            labels = labels[mask]

            tokens = self.token_id
            if not isinstance(tokens, list):
                tokens = [self.token_id]

            for token in tokens:
                labels = (labels == token)

                token_prob = probs[..., token]
                token_prob = token_prob[..., :-1]

                token_prob = token_prob[mask]
                self.metric.update(token_prob, labels)

        elif type == 'projection' and self.bins == probs.shape[-1]:
            token_prob = probs.reshape((probs.shape[0] * probs.shape[1], -1))
            labels = labels.reshape(-1)
            self.metric.update(token_prob, labels)
        else:
            return

    def reset(self):
        self.metric.reset()

    def is_graphical(self):
        return False

    def graph(self):
        return None

    def __str__(self):
        return f"{self.token}_{self.metric_type.name.lower()}_{self.turn_type.name}"


class PR_AUC(Metric):
    def __init__(self, token_id: int, token: str, turn_type=TurnType.NONE, nthresh=50, bins=1, device="cuda:0"):
        super().__init__(MetricType.PR_AUC, token_id, token, nthresh, device)

        self.turn_type = turn_type
        self.bins = bins

        task = 'multiclass' if self.bins > 1 else 'binary'
        self.metric = PrecisionRecallCurve(
            task='binary', thresholds=self.thresholds, ignore_index=-100).to(self.device)
        self.auc = AUC()

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

    def add(self, probs, labels, type='token', yield_mask=None, ignore_mask=None, overlap_mask=None, **kwargs):
        if overlap_mask is None:
            overlap_mask = torch.zeros_like(labels)
        if yield_mask is None:
            yield_mask = torch.ones_like(labels)
        if ignore_mask is None:
            ignore_mask = torch.ones_like(labels)

        if self.token_id != -1 and type == 'token':
            labels = labels[..., 1:]

            if self.turn_type == TurnType.NORMAL:
                mask = torch.logical_and(ignore_mask,
                                         torch.logical_not(overlap_mask))
            elif self.turn_type == TurnType.OVERLAP:
                mask = torch.logical_and(overlap_mask, ignore_mask)
            elif self.turn_type == TurnType.YIELD:
                mask = torch.logical_and(ignore_mask, yield_mask)
            else:
                mask = torch.logical_and(ignore_mask, ignore_mask)

            mask = mask[..., 1:]
            labels = labels[mask]

            tokens = self.token_id
            if not isinstance(tokens, list):
                tokens = [self.token_id]

            if len(tokens) == 2:
                token_prob = torch.where(probs[..., tokens[0]] > probs[..., tokens[1]], probs[..., tokens[0]],
                                         probs[..., tokens[1]])
            else:
                token_prob = probs[..., tokens[0]]

            token_prob = token_prob[..., :-1]

            labels = torch.eq(labels, tokens[0])

            token_prob = token_prob[mask]
            self.metric.update(token_prob, labels)

        elif type == 'projection' and self.bins == probs.shape[-1]:
            token_prob = probs.reshape((probs.shape[0] * probs.shape[1], -1))
            labels = labels.reshape(-1)
            self.metric.update(token_prob, labels)
        else:
            return

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


class BargeRate(Metric):
    """
    Barge-in rate metric so where threshold for eot is breached prior to actual
    turn end
    """

    def __init__(self, token_id: int, token: str, rule="none", nthresh=50, turn_type=TurnType.NONE,
                 tokens_dict: dict = {}, device="cuda:0"):
        super().__init__(MetricType.BR, token_id, token,
                         nthresh, device)

        self.total_turns = 0
        self.barge_in = torch.zeros_like(self.thresholds)
        self.non_barge_in = torch.zeros_like(self.thresholds)

        self.rule = rule
        self.turn_type = turn_type

    def add(self, probs, labels, type='token', yield_mask=None, non_yield_mask=None, overlap_mask=None,
            ignore_mask=None,
            interrupt_mask=None, turn_mask=None, **kwargs):
        """
        Find all turns from labels and identify if probs within that utterance are all greater then thresholds
        At all points in turn aside from the end

        Arguments:
            turn_mask: torch.Tensor
                Mask of all turns that are uttered by the speaker
        """
        if type != 'token':
            return

        if self.turn_type == TurnType.YIELD:
            maskA = torch.logical_and(yield_mask, turn_mask[0])
            maskB = torch.logical_and(yield_mask, turn_mask[1])
            mask = torch.logical_or(maskA, maskB)
        elif self.turn_type == TurnType.NON_YIELD:
            maskA = torch.logical_and(non_yield_mask, turn_mask[0])
            maskB = torch.logical_and(non_yield_mask, turn_mask[1])
            mask = torch.logical_or(maskA, maskB)
        else:
            mask = torch.ones_like(turn_mask[0])

        max_probs = []

        seq_len = labels.size()[-1]

        all_turn_masks = turn_mask

        tokens = self.token_id
        if not isinstance(tokens, list):
            tokens = [self.token_id]

        mask = mask[..., 1:]
        labels = labels[..., 1:]

        batch_labels = torch.cumsum(torch.ones_like(labels), dim=0) - 1
        labels = labels[mask]

        for idx in range(2):
            token_prob = probs[..., tokens[0]]
            if len(tokens) == 2:
                token_prob = torch.where(probs[..., tokens[0]] > probs[..., tokens[1]], probs[..., tokens[0]],
                                         probs[..., tokens[1]])
            token_prob = token_prob[..., :-1]

            turn_mask = all_turn_masks[idx]
            turn_mask = turn_mask[..., 1:].long()

            turn_changes = torch.cat((turn_mask[:, :1], torch.abs(
                turn_mask[:, 1:] - turn_mask[:, :-1])), dim=1)
            turn_ids = torch.cumsum(turn_changes, dim=1)

            global_turn_ids = (turn_ids + seq_len * batch_labels) * turn_mask
            unique_ids = torch.unique(global_turn_ids)

            token_prob = token_prob[mask]
            global_turn_ids = global_turn_ids[mask]

            for turn_id in unique_ids:
                single_turn_mask = global_turn_ids == turn_id
                if single_turn_mask.sum() == 0 or turn_id == 0 or torch.isin(labels[single_turn_mask][-1],
                                                                             torch.tensor(tokens, device="cuda")):
                    continue

                single_turn_mask = torch.logical_and(
                    single_turn_mask, torch.logical_not(torch.isin(labels, torch.tensor(tokens, device="cuda"))))
                if sum(single_turn_mask) == 0:
                    max_probs.append(0)
                    continue

                max_prob = torch.max(token_prob[single_turn_mask])
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
        if isinstance(thresh, float):
            if thresh in self.thresholds.tolist():
                idx = self.thresholds.tolist().index(thresh)
            else:
                idx = 2
        elif isinstance(thresh, torch.Tensor):
            idx = thresh.item()
        else:
            idx = 2

        barge_in = self.barge_in[idx] / \
                   (self.non_barge_in[idx] + self.barge_in[idx])

        for thresh in self.thresholds:
            i = self.thresholds.tolist().index(thresh)
            logger.info(
                f"BR: {thresh} = {self.barge_in[i] / (self.non_barge_in[idx] + self.barge_in[idx])}")

        # assert barge_in == self.barge_in[idx] / \
        #    self.total_turns, f"ERROR: Barge-In calculation is wrong {nrr} != {self.barge_in[idx] / self.total_turns}"

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
                         nthresh, device)

        self.total_turns = 0
        self.no_response_turns = torch.zeros_like(self.thresholds)
        self.response_turns = torch.zeros_like(self.thresholds)

        self.rule = rule
        self.turn_type = turn_type

    def add(self, probs, labels, type='token', yield_mask=None, non_yield_mask=None, overlap_mask=None,
            ignore_mask=None,
            interrupt_mask=None, turn_mask=None, **kwargs):
        """
        Find all turns from labels and identify if probs within that utterance are all less then thresholds

        Arguments:
            turn_mask: torch.Tensor
                Mask of all turns that are uttered by the speaker
        """
        if type != 'token':
            return

        if self.turn_type == TurnType.YIELD:
            maskA = torch.logical_and(yield_mask, turn_mask[0])
            maskB = torch.logical_and(yield_mask, turn_mask[1])
            mask = torch.logical_or(maskA, maskB)
        elif self.turn_type == TurnType.NON_YIELD:
            maskA = torch.logical_and(non_yield_mask, turn_mask[0])
            maskB = torch.logical_and(non_yield_mask, turn_mask[1])
            mask = torch.logical_or(maskA, maskB)
        else:
            mask = torch.ones_like(turn_mask[0])

        max_probs = []

        seq_len = labels.size()[-1]

        all_turn_masks = turn_mask

        tokens = self.token_id
        if not isinstance(tokens, list):
            tokens = [self.token_id]

        mask = mask[..., 1:]
        labels = labels[..., 1:]

        batch_labels = torch.cumsum(torch.ones_like(labels), dim=0) - 1
        labels = labels[mask]

        for idx in range(2):
            token_prob = probs[..., tokens[0]]
            if len(tokens) == 2:
                token_prob = torch.where(probs[..., tokens[0]] > probs[..., tokens[1]], probs[..., tokens[0]],
                                         probs[..., tokens[1]])

            token_prob = token_prob[..., :-1]

            turn_mask = all_turn_masks[idx]
            turn_mask = turn_mask[..., 1:].long()

            turn_changes = torch.cat((turn_mask[:, :1], torch.abs(
                turn_mask[:, 1:] - turn_mask[:, :-1])), dim=1)
            turn_ids = torch.cumsum(turn_changes, dim=1)

            global_turn_ids = (turn_ids + seq_len * batch_labels) * turn_mask
            unique_ids = torch.unique(global_turn_ids)

            token_prob = token_prob[mask]
            global_turn_ids = global_turn_ids[mask]

            for turn_id in unique_ids:
                single_turn_mask = global_turn_ids == turn_id
                if single_turn_mask.sum() == 0 or turn_id == 0 or torch.isin(labels[single_turn_mask][-1],
                                                                             torch.tensor(tokens, device="cuda")):
                    max_probs.append(0)
                    continue

                max_prob = torch.max(token_prob[single_turn_mask])
                max_probs.append(max_prob.item())

            for idx, threshold in enumerate(self.thresholds):
                self.no_response_turns[idx] += sum(x <
                                                   threshold for x in max_probs)
                self.response_turns[idx] += sum(x >=
                                                threshold for x in max_probs)

            self.total_turns += len(max_probs)

        return

    def calculate(self, params=None):
        if params is None:
            params = {}

        # Use best performing thresh for BACC
        # Fallback if not available
        thresh = params.get("val_eot_bacc_NONE", 2)
        print(thresh)
        if isinstance(thresh, float):
            if thresh in self.thresholds.tolist():
                idx = self.thresholds.tolist().index(thresh)
            else:
                idx = 2

        elif isinstance(thresh, torch.Tensor):
            idx = thresh.item()
        else:
            idx = 2

        print(thresh)
        print(self.thresholds.tolist())
        print(idx)

        nrr = self.no_response_turns[idx] / \
              (self.response_turns[idx] + self.no_response_turns[idx])

        for thresh in self.thresholds:
            i = self.thresholds.tolist().index(thresh)
            logger.info(
                f"NRR: {thresh} = {self.no_response_turns[i] / (self.no_response_turns[idx] + self.response_turns[idx])}")

        # assert nrr == self.no_response_turns[idx] / \
        #    self.total_turns, f"ERROR: NRR calculation is wrong {nrr} != {self.no_response_turns[idx] / self.total_turns}"

        return nrr, self.total_turns, self.thresholds[idx]

    def reset(self):
        self.total_turns = 0
        self.no_response_turns = torch.zeros_like(self.thresholds)
        self.response_turns = torch.zeros_like(self.thresholds)

    def __str__(self):
        return f"{self.token}_{self.turn_type}_NRR"


class Metrics:
    def __init__(self, metric_config: dict, type='test', device="cuda:0"):
        metrics = []
        output = {}
        parameters = {}
        counts = {}

        self.type = type + "_"

        metric_config = filter(lambda x: x is not None, metric_config)
        for (token_id, token, metric_list, turn_type, measure_type, bins) in metric_config:

            for metric_item in metric_list:
                if metric_item == MetricType.PR_AUC:
                    metric = PR_AUC(
                        token_id, token, turn_type=turn_type, device=device, bins=bins)
                elif metric_item == MetricType.ROC_AUC:
                    metric = ROC_AUC(
                        token_id, token, turn_type=turn_type, device=device, bins=bins)
                elif metric_item == MetricType.F1_SCORE:
                    metric = F1_Score(
                        token_id, token, turn_type=turn_type, device=device, bins=bins)
                elif metric_item == MetricType.BACC:
                    metric = BACC(token_id, token,
                                  turn_type=turn_type, device=device)
                elif metric_item == MetricType.PERPLEXITY:
                    metric = Perplexity_Score(device=device)
                elif metric_item == MetricType.ACC:
                    metric = ACC(token_id, token, device=device,
                                 turn_type=turn_type, bins=bins)
                elif metric_item == MetricType.NRR:
                    metric = NRR(token_id, token, turn_type=turn_type,
                                 device=device)
                elif metric_item == MetricType.BR:
                    metric = BargeRate(token_id, token, turn_type=turn_type,
                                       device=device)
                else:
                    raise NameError(
                        f"NO METRIC GATHERED for: {metric_item.name}")
                metrics.append(metric)

            output[self.type + str(metric)] = None
            parameters[self.type + str(metric)] = None
            counts[self.type + str(metric)] = None

        self.metrics = metrics
        self.parameters = parameters
        self.counts = counts
        self.output = output

    def add(self, probs, labels, logits=None, type='token', **kwargs):
        for metric in self.metrics:
            if isinstance(metric, Perplexity_Score):
                metric.add(logits, labels, type=type, **kwargs)
            else:
                metric.add(probs, labels, type=type, **kwargs)

    def calculate(self, params=None):
        for metric in self.metrics:
            output, count, thresh = metric.calculate(params)
            self.output[self.type + str(metric)] = output
            self.counts[self.type + str(metric)] = count
            self.parameters[self.type + str(metric)] = thresh

        return self.output, self.counts, self.parameters

    def get_graphs(self):
        graphs = {}
        for metric in self.metrics:
            if metric.is_graphical():
                if metric.token not in graphs:
                    graphs[metric.token] = {}
                graphs[metric.token][str(metric)] = metric.graph()

        return graphs

    def reset(self):
        for metric in self.metrics:
            metric.reset()