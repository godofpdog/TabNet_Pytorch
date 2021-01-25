""" Implementations of trainig operations """

import time
import torch 
from torch.utils.data import DataLoader
from .model import InferenceModel, PretrainModel
from ..utils import Meter


__all__ = [
    'train_epoch', 'eval_epoch'
]


def train_epoch(
    model, data_loader, epoch, criterion, optimizer, metrics=None, logger=None 

):
    """
    Train one epoch.

    Arguments:
        model (core.model.InferenceModel or core.model.PretrainModel):
            Model object.

        data_loader (torch.utils.data.DataLoader):
            A Pytorch DataLoader created from `core.data.create_data_loader`.

        epoch (int): Epoch index.

        criterion (core.criterion.Criterion):
            A `Criterion` object to compute the final loss value.

        optimizer:
            A Pytorch optimizer.

        metrics:

        logger (logging.Logger):

    Returns:
        meter (utils.Meter):
            Training information recorder.

    """
    if model.__class__ not in (InferenceModel, PretrainModel):
        raise TypeError(
            'Invalid model type, input argument `model` must be an `InferenceModel` or `PretrainModel` but got `{}`'\
                .format(model.__class__)
        )

    meter = Meter()

    torch.cuda.empty_cache()
    model.train()

    total_loss = 0

    for i, data in enumerate(data_loader):
        # NOTE must consider multi-task 

        start = time.time()


def eval_epoch():
    pass

