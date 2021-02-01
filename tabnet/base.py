"""
Utilities base classes and functions for this repo.

"""


class ClassificationTaskMixin:
    """
    Mixin class for all classification criterions and metrics.
    """
    _task_type = 'classification'


class RegressionTaskMixin:
    """
    Mixin class for all regression criterions and metrics.
    """
    _task_type = 'regression'

