"""
To create supported metric objects.
"""

from ._classfication import (
    AccMetric,
    PrecisionMetric,
    RecallMetric,
    F1Metric
)

from ._regression import (
    MSEMetric,
    MAEMetric,
    MAPEMetric,
    R2Metric
)


SUPPORTED_METRICS = {
    'acc': AccMetric,
    'precision': PrecisionMetric,
    'recall': RecallMetric,
    'f1': F1Metric,

    'mse': MSEMetric,
    'mae': MAEMetric,
    'mape': MAPEMetric,
    'r2': R2Metric
}


def get_metric(metric):
    """
    Get a metric scorer.

    Arguments:
        metrics (str):
            Metric scoring method as string.

    Returns:
        scorer (subclass of `tabnet.metrics.Metric`)
            Metric scorer.
        
    """

    scorer = SUPPORTED_METRICS.get(metric)

    if scorer is not None:
        return scorer()

    else:
        raise ValueError(
            '%r is not a valid metric value.'
            'Use `sorted(tabnet.metrics.SUPPORTED_METRICS.keys())` '
            'to get valid options.' % metric
            )
