import torch
import logging

from common.metrics.ppl import PPL
from data.utils import get_logger


logger = get_logger(__name__, level=logging.DEBUG)


class MetricContainer:
    def __init__(self, metrics, prefix=""):
        self.metrics = metrics
        self.N = 0

        self.prefix = prefix
        self.params = {}

        logger.info(f"Metric container created with {self.metrics}")

    def __str__(self):
        return f"MetricContainer({self.metrics})"

    def add(self, probs, labels, logits=None, **kwargs):
        for metric in self.metrics:
            if isinstance(metric, PPL):
                metric.add(logits, labels, **kwargs)
            else:
                metric.add(probs, labels, **kwargs)

    def calculate(self, set_param=False, use_param=False, **kwargs):
        output = {}
        counts = {}

        prefix = ""
        if self.prefix != "":
            prefix = self.prefix + "-"

        for idx, metric in enumerate(self.metrics):
            result, count, thresh = metric.compute(
                params=self.params if use_param else None
            )
            if isinstance(result, torch.Tensor):
                result = result.item()
            if (
                not isinstance(result, str)
                and not isinstance(result, int)
                and not isinstance(result, float)
            ):
                logger.error(
                    f"Result {result} is not a valid type ({type(result)}). Skipping ..."
                )
                continue

            if prefix + str(metric) in output:
                logger.info(
                    f"Metric {prefix + str(metric)} already exists in output={output}, overwriting..."
                )

            output[prefix + str(metric)] = result
            counts[prefix + str(metric)] = count
            if set_param:
                self.params[str(metric)] = thresh

        return output, counts

    def reset(self, reset_params=False):
        for metric in self.metrics:
            metric.reset()

        if reset_params:
            self.params = {}

    def set_prefix(self, prefix):
        self.prefix = prefix

    def get_params(self):
        return self.params

    def output(self, metrics):
        output = ""
        for metric, value in metrics.items():
            if isinstance(value, torch.Tensor):
                output += f"{metric}={round(value.item(),3)}"
            else:
                output += f"{metric}={round(value,3)}"

    def plot(self):
        graphs = {}
        prefix = ""
        if self.prefix != "":
            prefix = self.prefix + "-"

        for metric in self.metrics:
            graph = metric.plot()
            if graph is not None:
                graphs[prefix + str(metric)] = graph

        return graphs


DefaultMetrics = MetricContainer([])
