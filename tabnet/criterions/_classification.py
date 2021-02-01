""" Implementations of classification losses. """

import torch
from ._base import Loss
from ..base import ClassificationTaskMixin


class BinaryCrossEntropyLoss(Loss, ClassificationTaskMixin):
    def __init__(self):
        super(BinaryCrossEntropyLoss, self).__init__()
        self._loss_fn = torch.nn.BCEWithLogitsLoss()
    
    def score_func(self, preds, targets):
        targets = targets.to(preds.device, dtype=torch.int64)

        return self._loss_fn(preds, targets)


class MutiClassCrossEntropyLoss(Loss, ClassificationTaskMixin):
    def __init__(self):
        super(MutiClassCrossEntropyLoss, self).__init__()
        self._loss_fn = torch.nn.CrossEntropyLoss()

    def score_func(self, preds, targets):
        targets = targets.to(predictions.device, dtype=torch.int64)

        return self._loss_fn(preds, targets) 
        