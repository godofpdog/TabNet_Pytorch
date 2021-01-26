""" Implementations of builtin metrics. """

from sklearn import metrics
from ..core.metrics import MetricBase


class AccuracyMetric(MetricBase):
    def __init__(self):
        super(AccuracyMetric, self).__init__()

    def __repr__(self):
        return 'accuracy'

    def score_func(self, preds, targets):
        return metrics.accuracy_score(y_pred=preds, y_true=targets)


class MeanSquaredErrorMetric(MetricBase):
    def __init__(self):
        super(MeanSquaredErrorMetric, self).__init__()

    def __repr__(self):
        return 'mean_squared_error'

    def score_func(self, preds, targets):
        return metrics.mean_squared_error(y_pred=preds, y_true=targets)


SUPPORTED_METRICS = {
    'acc': AccuracyMetric,
    'mse': MeanSquaredErrorMetric
}


def create_metrics(code):
    """
    Create metrics objects.

    Arguments:
        code (st):
            Code of evaluation metric.

    Return:
        metric (subclass of `MetricBase`):
            An evaluation metric object.

    """
    metric = SUPPORTED_METRICS.get(code)

    if metric is not None:
        return metric()
    else:
        raise ValueError('Not supported metric code.')
