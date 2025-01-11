from common.metrics.metrics import MetricContainer


def DefaultMetricBuilder(*args, **kwargs):
    return MetricContainer([])
