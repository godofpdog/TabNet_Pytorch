""" Implementations of trainig operations """

import time
import torch 
from torch.utils.data import DataLoader
from ._models import InferenceModel, PretrainModel
from ..utils.utils import Meter
from ..utils.logger import show_message


__all__ = [
    'train_epoch', 'eval_epoch'
]

# TODO train/eval batch


def train_epoch(
    model, data_loader, epoch, post_processor, criterion, optimizer, metrics=None, logger=None, device='cpu'

):
    """
    Train on one epoch.

    Arguments:
        model (core.model.InferenceModel or core.model.PretrainModel):
            Model object.

        data_loader (torch.utils.data.DataLoader):
            A Pytorch DataLoader created from `core.data.create_data_loader`.

        epoch (int): Epoch index.

        post_processor (subclass of `core.estimator_base.ProcessorBase`):
            Post processor for final result computation.

        criterion (core.criterion.Criterion):
            A `Criterion` object to compute the final loss value.

        optimizer:
            A Pytorch optimizer.

        metrics:
            pass 

        logger (logging.Logger):
            pass 

        device (str):
            Computation device.

    Returns:
        meter (utils.Meter):
            Training information recorder.

    """

    if model.__class__ not in (InferenceModel, PretrainModel):
        raise TypeError(
            'Invalid model type, input argument `model` must be an `InferenceModel` or `PretrainModel` but got `{}`'\
                .format(model.__class__)
        )

    if metrics is not None:
        assert criterion.num_tasks == len(metrics)

    meter = Meter()

    torch.cuda.empty_cache()
    model.train()

    total_loss = 0

    for b, data in enumerate(data_loader):
        show_message(
            '[Train] ==================== batch : {} ===================='.format(b),
            logger=logger, level='DEBUG'
        )

        # clear gradient
        optimizer.zero_grad()

        # init time
        start = time.time()

        # split data 
        feats, targets = data
        feats = feats.to(device)
        targets = targets.to(device)

        # forward
        outputs, mask_loss = model(feats)
    
        # calc loss
        task_loss = criterion(outputs, targets)
        total_loss = task_loss - mask_loss * 1e-3  # TODO as argument

        # update params
        total_loss.backward()
        optimizer.step()

        # training info
        updates = {
            'total_loss': total_loss.item(),
            'task_loss': task_loss.item(),
            'mask_loss': mask_loss.item(),
            'time_cost': time.time() - start
        }

        # calc metrics
        if metrics is not None:
            preds = post_processor(outputs)

            for t, metric in enumerate(metrics):
                updates[repr(metric)] = metric(
                    preds[t], targets[..., t].view(-1, 1)
                )

        # update meter
        meter.update(updates)

        # show info
        show_message(
            '[Train] *** Solver Info ***',
            logger=logger, level='DEBUG'
        )

        for name in meter.names:
            show_message(
            '[Train] current {} : {}, mean {} : {}.'.format(
                name, meter[name][-1], name, meter.get_statistics(name)[name]
            ),
            logger=logger, level='DEBUG'
        )

    return meter


def eval_epoch(
     model, data_loader, epoch, post_processor, criterion, metrics=None, logger=None, device='cpu'
):
    """
    Evaluate on one epoch.

    Arguments:
        model (core.model.InferenceModel or core.model.PretrainModel):
            Model object.

        data_loader (torch.utils.data.DataLoader):
            A Pytorch DataLoader created from `core.data.create_data_loader`.

        epoch (int): Epoch index.

        post_processor (subclass of `core.estimator_base.ProcessorBase`):
            Post processor for final result computation.

        criterion (core.criterion.Criterion):
            A `Criterion` object to compute the final loss value.

        metrics:

        logger (logging.Logger):

        device (str):
            Computation device.

    Returns:
        meter (utils.Meter):
            Evaluation information recorder. 
            
    """

    if model.__class__ not in (InferenceModel, PretrainModel):
        raise TypeError(
            'Invalid model type, input argument `model` must be an `InferenceModel` or `PretrainModel` but got `{}`'\
                .format(model.__class__)
        )

    if metrics is not None:
        assert criterion.num_tasks == len(metrics)

    meter = Meter()

    torch.cuda.empty_cache()
    model.eval()

    total_loss = 0

    with torch.no_grad():

        for b, data in enumerate(data_loader):
            show_message(
                '[Train] ==================== batch : {} ===================='.format(b),
                logger=logger, level='DEBUG'
            )

            # init time
            start = time.time()

            # split data 
            feats, targets = data
            feats = feats.to(device)
            targets = targets.to(device)

            # forward
            outputs, mask_loss = model(feats)
        
            # calc loss
            task_loss = criterion(outputs, targets)
            total_loss = task_loss - mask_loss * 1e-3  # TODO as argument

            # training info
            updates = {
                'total_loss': total_loss.item(),
                'task_loss': task_loss.item(),
                'mask_loss': mask_loss.item(),
                'time_cost': time.time() - start
            }

            # calc metrics
            if metrics is not None:
                preds = post_processor(outputs)

                for t, metric in enumerate(metrics):
                    updates[repr(metric)] = metric(
                        preds[t], targets[..., t].view(-1, 1)
                    )

            # update meter
            meter.update(updates)

            # show info
            show_message(
                '[Train] *** Solver Info ***',
                logger=logger, level='DEBUG'
            )

            for name in meter.names:
                show_message(
                '[Train] current {} : {}, mean {} : {}.'.format(
                    name, meter[name][-1], name, meter.get_statistics(name)[name]
                ),
                logger=logger, level='DEBUG'
            )

    return meter

