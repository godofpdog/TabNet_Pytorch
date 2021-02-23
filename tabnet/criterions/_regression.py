""" Implementations of regression losses. """

import torch
from ._base import Loss
from ..mixin import RegressionTaskMixin


class MSELoss(Loss, RegressionTaskMixin):
    def __init__(self):
        super(MSELoss, self).__init__()
        self._loss_func = torch.nn.MSELoss()

    def score_func(self, preds, targets):
        return self._loss_func(preds, targets)
    

class MAELoss(Loss, RegressionTaskMixin):
    def __init__(self):
        super(MAELoss, self).__init__()
        self._loss_func = torch.nn.L1Loss()

    def score_func(self, preds, targets):
        return self._loss_func(preds, targets)

