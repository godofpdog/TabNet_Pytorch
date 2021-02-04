"""
The module `tabnet.estimatior` i

"""

from ._regression import TabNetRegressor
from ._classifier import TabNetClassifier


# TODO alias


__all__ = [
    'TabNetRegressor',
    'TabNetClassifier'
]