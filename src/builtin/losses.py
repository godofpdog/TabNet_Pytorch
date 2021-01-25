""" Implementations of builtin losses. """

import torch
from ..core.base import CustomizedLoss


# TODO hard label classfication loss as a base class


class HardLabelClassificarionLoss:
    def __init__(self):
        raise NotImplementedError


class RegressionLoss:
    def __init__(self):
        raise NotImplementedError


class BinaryCrossEntropyLoss(CustomizedLoss):
    def __init__(self, device):
        super(BinaryCrossEntropyLoss, self).__init__()
        self._loss_fn = torch.nn.BCELoss().to(device)
    
    def forward(self, predictions, targets):
        targets = targets.to(predictions.device, dtype=torch.int64)

        return self._loss_fn(predictions, targets)


class MutiClassCrossEntropyLoss(CustomizedLoss):
    def __init__(self, device):
        super(MutiClassCrossEntropyLoss, self).__init__()
        self._loss_fn = torch.nn.CrossEntropyLoss().to(device)

    def forward(self, predictions, targets):
        targets = targets.to(predictions.device, dtype=torch.int64)

        return self._loss_fn(predictions, targets)



