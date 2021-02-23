""" Implementations of regression metrics. """

from ._base import Metric
from ..mixin import RegressionTaskMixin
from sklearn import metrics


class MSEMetric(Metric, RegressionTaskMixin):
    def __init__(self):
        super(MSEMetric, self).__init__()

    def __repr__(self):
        return 'mean_squared_error'

    def score_func(self, preds, targets):
        return metrics.mean_absolute_error(y_pred=preds, y_true=targets)


class MAEMetric(Metric, RegressionTaskMixin):
    def __init__(self):
        super(MAEMetric, self).__init__()

    def __repr__(self):
        return 'mean_absolute_error'

    def score_func(self, preds, targets):
        return metrics.mean_absolute_error(y_pred=preds, y_true=targets)


class MAPEMetric(Metric, RegressionTaskMixin):
    def __init__(self):
        super(MAPEMetric, self).__init__()

    def __repr__(self):
        return 'mean_absolute_percentage_error'

    def score_func(self, preds, targets):
        return metrics.mean_absolute_percentage_error(y_pred=preds, y_true=targets)


class R2Metric(Metric, RegressionTaskMixin):
    def __init__(self):
        super(R2Metric, self).__init__()

    def __repr__(self):
        return 'r_square'

    def score_func(self, preds, targets):
        return metrics.r2_score(y_pred=preds, y_true=targets)

