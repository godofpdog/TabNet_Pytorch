""" Implementations of classification metrics. """

from ._base import Metric
from ..mixin import ClassificationTaskMixin
from sklearn import metrics


class AccMetric(Metric, ClassificationTaskMixin):
    def __init__(self):
        super(AccMetric, self).__init__()

    def __repr__(self):
        return 'accuracy'

    def score_func(self, preds, targets):
        return metrics.accuracy_score(y_pred=preds, y_true=targets)


class PrecisionMetric(Metric, ClassificationTaskMixin):
    def __init__(self):
        super(PrecisionMetric, self).__init__()

    def __repr__(self):
        return 'precision'

    def score_func(self, preds, targets, **kwargs):
        return metrics.precision_score(y_pred=preds, y_true=targets, **kwargs)


class RecallMetric(Metric, ClassificationTaskMixin):
    def __init__(self):
        super(RecallMetric, self).__init__()

    def __repr__(self):
        return 'recall'

    def score_func(self, preds, targets, **kwargs):
        return metrics.recall_score(y_pred=preds, y_true=targets, **kwargs)


class F1Metric(Metric, ClassificationTaskMixin):
    def __init__(self):
        super(F1Metric, self).__init__()

    def __repr__(self):
        return 'f1_score'

    def score_func(self, preds, targets, **kwargs):
        return metrics.f1_score(y_pred=preds, y_true=targets, **kwargs)
