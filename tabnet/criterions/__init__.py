"""
The module `tabnet.criterions` includes the base class of loss function module 
and some built-in loss modules.

"""

from ._base import Loss, _Criterion
from ._builder import get_loss, create_criterion
from ._regression import MSELoss
from ._classification import BinaryCrossEntropyLoss, MutiClassCrossEntropyLoss


# TODO alias


__all__ = [
    'Loss',
    '_Criterion',
    'get_loss',
    'create_criterion',
    'MSELoss',
    'BinaryCrossEntropyLoss',
    'MutiClassCrossEntropyLoss'
]
