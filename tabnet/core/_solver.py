""" Implementations of trainig operations """

import abc
import time
import torch 
from torch.utils.data import DataLoader
from ._models import InferenceModel, PretrainModel
from ..utils.utils import Meter
from ..utils.logger import show_message


__all__ = [
    'train_epoch', 'eval_epoch'
]


DIGITS = 6

# TODO train/eval batch

class _BaseTrainer(abc.ABC):
    """
    Base class of Trainers.
    """
    
    def train_epoch(self, model, data_loader, criterion, optimizer, metrics=None, logger=None, device='cpu', **kwargs):
        """
        Train on one epoch.

        Arguments:
            model (tabnet.core.model.InferenceModel or tabnet.core.model.PretrainModel):
                A model object.

            data_loader (torch.utils.data.DataLoader):
                A Pytorch DataLoader created from `core.data.create_data_loader`.

            criterion (tabnet.criterions._Criterion):
                A `_Criterion` object to compute the final loss value.

            optimizer:
                A Pytorch optimizer.

            metrics (list of valid metric objects):  # TODO check??
                List of metric scorers.

            logger (logging.Logger):
                The system logger object.

            device (str):
                The computation device.

        Returns:
            meter (tabnet.utils.utils.Meter).
                A `Meter` object contains the training / evaluation info.

        """
        return self._run_epoch(model, data_loader, criterion, optimizer, metrics, logger, device, True, **kwargs)
    
    def eval_epoch(self, model, data_loader, criterion, metrics=None, logger=None, device='cpu', **kwargs):
        """
        Evaluate on one epoch.

        Arguments:
            model (tabnet.core.model.InferenceModel or tabnet.core.model.PretrainModel):
                A model object.

            data_loader (torch.utils.data.DataLoader):
                A Pytorch DataLoader created from `core.data.create_data_loader`.

            criterion (tabnet.criterions._Criterion):
                A `_Criterion` object to compute the final loss value.

            metrics (list of valid metric objects):  # TODO check??
                List of metric scorers.

            logger (logging.Logger):
                The system logger object.

            device (str):
                The computation device.

        Returns:
            meter (tabnet.utils.utils.Meter).
                A `Meter` object contains the training / evaluation info.

        """
        return self._run_epoch(model, data_loader, criterion, metrics, logger, device, False, **kwargs)
    
    @classmethod
    def _run_epoch(self, model, data_loader, criterion, optimizer=None, metrics=None, logger=None, device='cpu', is_train=True, **kwargs):
        
        print('model : ', type(model))
        print('data_loader : ', type(data_loader))
        print('criterion : ', type(criterion))
        print('optimizer : ', type(optimizer))
        print('metrics : ', type(metrics))
        print('logger : ', type(logger))
        print('device : ', type(device))
        print('is_train : ', type(is_train))
        print('kwargs = ', kwargs)

        """
        Run one epoch.

        If is_train, use `train_batch` for each iteration. Otherwise, use `eval_batch`.

        * Note:
            - Must to implement the abstract method `train_batch` and `eval_batch` for the specified training / evaluation strategy.

        """
        self._check_model(model)

        if metrics is not None:
            assert criterion.num_tasks == len(metrics)

        if is_train:
            model.train()
        else:
            model.eval()

        meter = Meter()

        for data in data_loader:
            start = time.time()

            if is_train:
                updates = self.train_batch(
                    model, data, criterion, 
                    optimizer=optimizer, metrics=None, logger=None, device='cpu', **kwargs
                )
            else:
                updates = self.eval_batch( model, data, criterion, **kwargs)

            updates['time_cost'] = round(time.time() - start, DIGITS)

            meter.update(updates=updates)

            if logger is not None:
                self.show_info(meter, logger)

        return meter
            
    @abc.abstractmethod
    def train_batch(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def eval_batch(self, *args, **kwargs):
        raise NotImplementedError
    
    @abc.abstractmethod
    def _check_model(self):
        raise NotImplementedError

    @classmethod
    def update_metrics(cls, preds, targets, metrics, updates=None):
        """
        Update metrics.

        Arguments:
            preds (torch.Tensor):
                The model prediction.

            targets (torch.Tensor):
                The ground truth tensor.

            metrics (list of valid metric objects):  # TODO check??
                List of metric scorers.

            updates (dict):
                Update info.

        Returns:
            None

        """
        if metrics is None:
            return 

        if updates is None:
            updates = dict()
        elif not isinstance(updates, dict):
            raise TypeError('Argument `updates` must be a `dict` object, but got `{}`'.format(type(updates)))

        for t, metric in enumerate(metrics):
            updates[repr(metric)] = metric(
                preds[t], targets[..., t].view(-1, 1)
            )

        return None
        
    @classmethod
    def show_info(cls, meter, logger=None):
        """
        Show training / evaluation info.

        Arguments:
            meters (tabnet.utils.utils.Meter).
                A `Meter` object.

            logger (logging.Logger):
                The system logger object.

        Returns:
            None
            
        """
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

        return None


# def train_epoch(
#     model, data_loader, epoch, post_processor, criterion, optimizer, metrics=None, logger=None, device='cpu'
# ):
#     """
#     Train on one epoch.

#     Arguments:
#         model (tabnet.core.model.InferenceModel or tabnet.core.model.PretrainModel):
#             Model object.

#         data_loader (torch.utils.data.DataLoader):
#             A Pytorch DataLoader created from `core.data.create_data_loader`.

#         epoch (int): Epoch index.

#         post_processor (subclass of `core.estimator_base.ProcessorBase`):
#             Post processor for final result computation.

#         criterion (core.criterion.Criterion):
#             A `Criterion` object to compute the final loss value.

#         optimizer:
#             A Pytorch optimizer.

#         metrics:
#             pass 

#         logger (logging.Logger):
#             pass 

#         device (str):
#             Computation device.

#     Returns:
#         meter (utils.Meter):
#             Training information recorder.

#     """

#     if model.__class__ not in (InferenceModel, PretrainModel):
#         raise TypeError(
#             'Invalid model type, input argument `model` must be an `InferenceModel` or `PretrainModel` but got `{}`'\
#                 .format(model.__class__)
#         )

#     if metrics is not None:
#         assert criterion.num_tasks == len(metrics)

#     meter = Meter()

#     torch.cuda.empty_cache()
#     model.train()

#     total_loss = 0

#     for b, data in enumerate(data_loader):
#         show_message(
#             '[Train] ==================== batch : {} ===================='.format(b),
#             logger=logger, level='DEBUG'
#         )

#         # clear gradient
#         optimizer.zero_grad()

#         # init time
#         start = time.time()

#         # split data 
#         feats, targets = data
#         feats = feats.to(device)
#         targets = targets.to(device)

#         # forward
#         outputs, mask_loss = model(feats)
    
#         # calc loss
#         task_loss = criterion(outputs, targets)
#         total_loss = task_loss - mask_loss * 1e-3  # TODO as argument

#         # update params
#         total_loss.backward()
#         optimizer.step()

#         # training info
#         updates = {
#             'total_loss': total_loss.item(),
#             'task_loss': task_loss.item(),
#             'mask_loss': mask_loss.item(),
#             'time_cost': time.time() - start
#         }

#         # calc metrics
#         if metrics is not None:
#             preds = post_processor(outputs)

#             for t, metric in enumerate(metrics):
#                 updates[repr(metric)] = metric(
#                     preds[t], targets[..., t].view(-1, 1)
#                 )

#         # update meter
#         meter.update(updates)

#         # show info
#         show_message(
#             '[Train] *** Solver Info ***',
#             logger=logger, level='DEBUG'
#         )

#         for name in meter.names:
#             show_message(
#             '[Train] current {} : {}, mean {} : {}.'.format(
#                 name, meter[name][-1], name, meter.get_statistics(name)[name]
#             ),
#             logger=logger, level='DEBUG'
#         )

#     return meter


# def eval_epoch(
#      model, data_loader, epoch, post_processor, criterion, metrics=None, logger=None, device='cpu'
# ):
#     """
#     Evaluate on one epoch.

#     Arguments:
#         model (core.model.InferenceModel or core.model.PretrainModel):
#             Model object.

#         data_loader (torch.utils.data.DataLoader):
#             A Pytorch DataLoader created from `core.data.create_data_loader`.

#         epoch (int): Epoch index.

#         post_processor (subclass of `core.estimator_base.ProcessorBase`):
#             Post processor for final result computation.

#         criterion (core.criterion.Criterion):
#             A `Criterion` object to compute the final loss value.

#         metrics:

#         logger (logging.Logger):

#         device (str):
#             Computation device.

#     Returns:
#         meter (utils.Meter):
#             Evaluation information recorder. 
            
#     """

#     if model.__class__ not in (InferenceModel, PretrainModel):
#         raise TypeError(
#             'Invalid model type, input argument `model` must be an `InferenceModel` or `PretrainModel` but got `{}`'\
#                 .format(model.__class__)
#         )

#     if metrics is not None:
#         assert criterion.num_tasks == len(metrics)

#     meter = Meter()

#     torch.cuda.empty_cache()
#     model.eval()

#     total_loss = 0

#     with torch.no_grad():

#         for b, data in enumerate(data_loader):
#             show_message(
#                 '[Train] ==================== batch : {} ===================='.format(b),
#                 logger=logger, level='DEBUG'
#             )

#             # init time
#             start = time.time()

#             # split data 
#             feats, targets = data
#             feats = feats.to(device)
#             targets = targets.to(device)

#             # forward
#             outputs, mask_loss = model(feats)
        
#             # calc loss
#             task_loss = criterion(outputs, targets)
#             total_loss = task_loss - mask_loss * 1e-3  # TODO as argument

#             # training info
#             updates = {
#                 'total_loss': total_loss.item(),
#                 'task_loss': task_loss.item(),
#                 'mask_loss': mask_loss.item(),
#                 'time_cost': time.time() - start
#             }

#             # calc metrics
#             if metrics is not None:
#                 preds = post_processor(outputs)

#                 for t, metric in enumerate(metrics):
#                     updates[repr(metric)] = metric(
#                         preds[t], targets[..., t].view(-1, 1)
#                     )

#             # update meter
#             meter.update(updates)

#             # show info
#             show_message(
#                 '[Train] *** Solver Info ***',
#                 logger=logger, level='DEBUG'
#             )

#             for name in meter.names:
#                 show_message(
#                 '[Train] current {} : {}, mean {} : {}.'.format(
#                     name, meter[name][-1], name, meter.get_statistics(name)[name]
#                 ),
#                 logger=logger, level='DEBUG'
#             )

#     return meter


class TabNetTrainer(_BaseTrainer):

    @classmethod
    def _check_model(cls, model):
        if model.__class__ != InferenceModel:
            raise TypeError(
                'Invalid model type, input argument `model` must be an `InferenceModel` object, but got `{}`'\
                    .format(model.__class__)
            )

        return None

    def train_batch(self, model, data, criterion, optimizer=None, post_processor=None, metrics=None, logger=None, device='cpu'):
        """
        Train on one batch.

        Arguments:
            model (tabnet.core.model.InferenceModel or tabnet.core.model.PretrainModel):
                A model object.

            data_loader (torch.utils.data.DataLoader):
                A Pytorch DataLoader created from `core.data.create_data_loader`.

            criterion (tabnet.criterions._Criterion):
                A `_Criterion` object to compute the final loss value.

            post_processor (subclass of `core.estimator_base.ProcessorBase`):
                Post processor for final result computation.

            optimizer:
                A Pytorch optimizer.

            metrics (list of valid metric objects):  # TODO check??
                List of metric scorers.

            logger (logging.Logger):
                The system logger object.

            device (str):
                The computation device.

        Returns:
            updates (dict):
                The update info.

        """
        # clear gradient
        optimizer.zero_grad()

        # process data
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
        }

        # calc metrics
        if metrics is not None:
            preds = post_processor(outputs)
            self.update_metrics(preds, targets, metrics, updates)

        return updates

    def eval_batch(self, model, data, criterion, post_processor=None, metrics=None, logger=None, device='cpu'):
        """
        Evaluate on one batch.

        Arguments:
            model (tabnet.core.model.InferenceModel or tabnet.core.model.PretrainModel):
                A model object.

            data_loader (torch.utils.data.DataLoader):
                A Pytorch DataLoader created from `core.data.create_data_loader`.

            criterion (tabnet.criterions._Criterion):
                A `_Criterion` object to compute the final loss value.

            post_processor (subclass of `core.estimator_base.ProcessorBase`):
                Post processor for final result computation.

            metrics (list of valid metric objects):  # TODO check??
                List of metric scorers.

            logger (logging.Logger):
                The system logger object.

            device (str):
                The computation device.

        Returns:
            updates (dict):
                The update info.

        """
        with torch.no_grad():

            # proess data
            feats, targets = data
            feats = feats.to(device)
            targets = targets.to(device)

            # forward
            outputs, mask_loss = model(feats)
        
            # calc loss
            task_loss = criterion(outputs, targets)
            total_loss = task_loss - mask_loss * 1e-3  # TODO as argument

            # evaluation info
            updates = {
                'total_loss': total_loss.item(),
                'task_loss': task_loss.item(),
                'mask_loss': mask_loss.item(),
            }

            # calc metrics
            if metrics is not None:
                preds = post_processor(outputs)
                self.update_metrics(preds, targets, metrics, updates)

        return updates


def get_trainer(trainer_type='tabnet_trainer'):
    SUPPORTED_TRAINER = {
        'tabnet_trainer': TabNetTrainer
    }

    trainer = SUPPORTED_TRAINER.get(trainer_type)

    if trainer is not None:
        return trainer()
    else:
        raise ValueError(
            '%r is not a valid trainer type.'
            'Use `sorted(tabnet.core.SUPPORTED_TRAINER.keys())` '
            'to get valid options.' % trainer_type
            )