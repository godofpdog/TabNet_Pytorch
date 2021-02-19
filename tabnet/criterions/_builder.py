"""
To create supported loss and `_Criterion` objects.
"""

from ._classification import (
    BinaryCrossEntropyLoss,
    MutiClassCrossEntropyLoss
)

from ._regression import (
    MSELoss,
    MAELoss
)

from ._pretraining import (
    TabNetPretrainingLoss
)

from ._base import _Criterion


SUPPORTED_LOSSES = {
    'bce': BinaryCrossEntropyLoss,
    'ce': MutiClassCrossEntropyLoss,

    'mse': MSELoss,
    'mae': MAELoss,

    'tabnet_pretraining': TabNetPretrainingLoss
}


def get_loss(loss):
    """
    Get a loss scorer.

    Arguments:
        loss (str):
            Loss function as string.

    Returns:
        scorer (subclass of `tabnet.criterion.Loss`)
            Loss function scorer.

    """   

    scorer = SUPPORTED_LOSSES.get(loss)

    if scorer is not None:
        return scorer()

    else:
        raise ValueError(
            '%r is not a valid loss value.'
            'Use sorted(tabnet.criterions.SUPPORTED_LOSSES.keys()) '
            'to get valid options.' % loss
            )


def create_criterion(losses, weights):
    """
    Create a `tabnet.criterion._Criterion` object.

    Arguments:
        losses (subclass of `tabnet.criterion.Loss` of list of them):
            Loss function scorer.

        weights (int or float or list of them):
            Weights of losses.

    Returns:
        criterion (tabnet.criterion._Criterion):
            Criterion object.

    """
    return _Criterion(losses, weights)
    