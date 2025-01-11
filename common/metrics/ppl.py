import torch
import logging

from torchmetrics.text import Perplexity

from data import TurnType
from common.metrics.utils import get_mask

logger = logging.getLogger(__name__)


class PPL:
    def __init__(self, device="cuda:0", turn_type=TurnType.NONE):
        self.emp_index = -100
        self.device = device
        self.turn_type = turn_type
        self.perplexity = Perplexity(ignore_index=self.emp_index).to(self.device)

    def add(self, preds, labels, **kwargs):
        preds = preds[:, :-1, :]
        labels = labels[:, 1:]

        mask = get_mask(self.turn_type, preds, labels, **kwargs).bool()

        labels = torch.where(
            mask, labels, torch.tensor(self.emp_index, device=self.device)
        )
        self.perplexity.update(preds, labels)

    def compute(self, **kwargs):
        result = self.perplexity.compute()
        self.log(result)

        if isinstance(result, torch.Tensor):
            return result.item(), None, None

        return result, None, None

    def plot(self):
        return None

    def reset(self):
        self.perplexity.reset()

    def __str__(self):
        return f"PPL-{self.turn_type.name}"

    def log(self, results):
        def string_result(idx, result):
            return f"Result={result}"

        logger.info(f"Results for {self}: {results}")
