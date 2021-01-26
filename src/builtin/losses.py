""" Implementations of builtin losses. """

import torch
from ..core.criterion import CustomizedLoss


# TODO hard label classfication loss as a base class


class HardLabelClassificarionLoss:
    def __init__(self):
        raise NotImplementedError


class RegressionLoss:
    def __init__(self):
        raise NotImplementedError


class BinaryCrossEntropyLoss(CustomizedLoss):
    def __init__(self):
        super(BinaryCrossEntropyLoss, self).__init__()
        self._loss_fn = torch.nn.BCELoss()
    
    def forward(self, predictions, targets):
        targets = targets.to(predictions.device, dtype=torch.int64)

        return self._loss_fn(predictions, targets)


class MutiClassCrossEntropyLoss(CustomizedLoss):
    def __init__(self):
        super(MutiClassCrossEntropyLoss, self).__init__()
        self._loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, predictions, targets):
        targets = targets.to(predictions.device, dtype=torch.int64)

        return self._loss_fn(predictions, targets)




