"""
The module `tabnet.metrics` includes the regression and classification metrics.
"""
from ._base import Metric
from ._regression import MSEMetric
from ._regression import MAEMetric
from ._regression import MAPEMetric
from ._regression import R2Metric

from ._classfication import AccMetric
from ._classfication import PrecisionMetric
from ._classfication import RecallMetric
from ._classfication import F1Metric

from ._builder import SUPPORTED_METRICS
from ._builder import get_metric


# TODO alias


__all__ = [
    'MSEMetric',
    'MAEMetric',
    'MAPEMetric',
    'R2Metric',

    'AccMetric',
    'PrecisionMetric',
    'RecallMetric',
    'F1Metric'
]
