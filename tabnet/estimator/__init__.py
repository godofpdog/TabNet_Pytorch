"""
The module `tabnet.estimatior` i

"""

from ._regression import TabNetRegressor
from ._classifier import TabNetClassifier
from ._customized import CustomizedEstimator
from ._base import BaseTabNet, BasePostProcessor


# TODO alias


__all__ = [
    'TabNetRegressor',
    'TabNetClassifier',
    'CustomizedEstimator',
    'BaseTabNet',
    'BasePostProcessor'
]