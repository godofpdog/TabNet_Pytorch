""" Implementations of trainig operations """

import abc
import time
import torch 
from torch.utils.data import DataLoader
from ._data import create_data_loader
from ._models import InferenceModel, PretrainModel
from ..utils.utils import Meter
from ..utils.logger import show_message


# __all__ = [
#     'train_epoch', 'eval_epoch'
# ]


DIGITS = 6


class _BaseTrainer(abc.ABC):
    """
    Base class of Trainers.
    """
    def __init__(self):
        pass
    
    def train_epoch(self, model, data_loader, criterion, optimizer, post_processor=None, metrics=None, device='cpu', **kwargs):
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

            post_processor (subclsss of tabnet.estimator.BasePostProcessor):
                A computation module to calc the final result.

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
        return self._run_epoch(model, data_loader, criterion, optimizer, post_processor, metrics, device, True, **kwargs)
    
    def eval_epoch(self, model, data_loader, criterion, post_processor=None, metrics=None, device='cpu', **kwargs):
        """
        Evaluate on one epoch.

        Arguments:
            model (tabnet.core.model.InferenceModel or tabnet.core.model.PretrainModel):
                A model object.

            data_loader (torch.utils.data.DataLoader):
                A Pytorch DataLoader created from `core.data.create_data_loader`.

            criterion (tabnet.criterions._Criterion):
                A `_Criterion` object to compute the final loss value.
            
            post_processor (subclsss of tabnet.estimator.BasePostProcessor):
                A computation module to calc the final result.

            metrics (list of valid metric objects):  # TODO check??
                List of metric scorers.

            device (str):
                The computation device.

        Returns:
            meter (tabnet.utils.utils.Meter).
                A `Meter` object contains the training / evaluation info.

        """
        return self._run_epoch(model, data_loader, criterion, None, post_processor, metrics, device, False, **kwargs)
    
    def _run_epoch(self, model, data_loader, criterion, optimizer, post_processor=None, metrics=None, device='cpu', is_train=True, **kwargs):
        """
        Run one epoch.

        If is_train, use `train_batch` for each iteration. Otherwise, use `eval_batch`.

        * Note:
            - Must to implement the abstract method `train_batch` and `eval_batch` for the specified training / evaluation strategy.

        """
        torch.cuda.empty_cache()

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
                    model, data, criterion, optimizer, post_processor,
                    metrics=metrics, device=device, **kwargs
                )
            else:
                pass
                updates = self.eval_batch(
                    model, data, criterion, post_processor,
                    metrics=metrics, device=device, **kwargs
                )

            updates['time_cost'] = round(time.time() - start, DIGITS)

            meter.update(updates=updates)

            # if logger is not None:
            #     self.show_info(meter, logger)

        return meter
            
    @abc.abstractmethod
    def train_batch(self, model, data, criterion, optimizer, post_processor, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def eval_batch(self, model, data, criterion, post_processor, **kwargs):
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


class TabNetTrainer(_BaseTrainer):
    def __init__(self):
        super(TabNetTrainer, self).__init__()

    def train_batch(self, model, data, criterion, optimizer, post_processor, metrics=None, device='cpu'):
        """
        Train on one batch.

        Arguments:
            model (tabnet.core.model.InferenceModel):
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

    def eval_batch(self, model, data, criterion, post_processor=None, metrics=None, device='cpu'):
        """
        Evaluate on one batch.

        Arguments:
            model (tabnet.core.model.InferenceModel):
                A model object.

            data_loader (torch.utils.data.DataLoader):
                A Pytorch DataLoader created from `core.data.create_data_loader`.

            criterion (tabnet.criterions._Criterion):
                A `_Criterion` object to compute the final loss value.

            post_processor (subclass of `core.estimator_base.ProcessorBase`):
                Post processor for final result computation.

            metrics (list of valid metric objects):  # TODO check??
                List of metric scorers.

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


class TabNetPretrainer(_BaseTrainer):
    def __init__(self):
        super(TabNetPretrainer, self).__init__()

    def train_batch(self, model, data, criterion, optimizer, post_processor=None, metrics=None, device='cpu'):
        """
        Train on one batch.

        Arguments:
            model (tabnet.core.model.PretrainModel):
                A model object.

            data (torch.Tensor):
                Batch data.

            criterion (tabnet.criterions._Criterion):
                A `_Criterion` object to compute the final loss value.

            optimizer:
                A Pytorch optimizer.

            device (str):
                The computation device.

        Returns:
            updates (dict):
                The update info.

        """
        # clear gradient
        optimizer.zero_grad()

        # process data
        data = data.to(device)

        # forward
        outputs, mask_loss, binary_mask = model(data)

        # calc loss
        task_loss = criterion(outputs, data, mask=binary_mask)
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

        return updates

    def eval_batch(self):
        """ Not needed. """
        pass 


class SwapDAEPreTrainer(_BaseTrainer):
    """
    Implementation of SwapDAE pre-training algorithm described in:
    https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629#250927
    """
    def  __init__(self):
        super(SwapDAEPreTrainer, self).__init__()

    def train_epoch(self, model, dataset, criterion, optimizer,
                    post_processor=None, metrics=None, device='cpu', 
                    batch_size=512, shuffle=True, pin_memory=True, num_workers=2, **kwargs):

        return self._run_epoch(
            model, dataset, criterion, optimizer,
            post_processor, metrics, device, True,
            batch_size, shuffle, pin_memory, num_workers, **kwargs
        )

    def _run_epoch(self, model, dataset, criterion, optimizer,
                   post_processor=None, metrics=None, device='cpu', is_train=True, 
                   batch_size=512, shuffle=True, pin_memory=True, num_workers=2, **kwargs):
        """
        Run one epoch, override from `_BaseTrainer`. Apply swap augmentation every epoch.

        Arguments:
            dataset (tabnet.core._data.SwapNoiseDataset): 
                Swap dataser for the pre-training algorithm.
            
            swap_rate (float):
                Swap rate on feature dimension.
 
        """
        torch.cuda.empty_cache()

        if metrics is not None:
            assert criterion.num_tasks == len(metrics)

        if is_train:
            model.train()
        else:
            model.eval()

        meter = Meter()

        # prepare data 
        dataset.swap()
        data_loader =  DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, 
            pin_memory=pin_memory, num_workers=num_workers
        )

        # TODO num_workers

        for data in data_loader:
            start = time.time()

            if is_train:
                updates = self.train_batch(
                    model, data, criterion, optimizer, post_processor,
                    metrics=metrics, device=device, **kwargs
                )
            else:
                pass
                updates = self.eval_batch(
                    model, data, criterion, post_processor,
                    metrics=metrics, device=device, **kwargs
                )

            updates['time_cost'] = round(time.time() - start, DIGITS)

            meter.update(updates=updates)

            # if logger is not None:
            #     self.show_info(meter, logger)

        return meter

    def train_batch(self, model, data, criterion, optimizer, 
                    post_processor=None, metrics=None, device='cpu', **kwargs):
        """
        Train on one batch.

        Arguments:
            model (tabnet.core.model.PretrainModel):
                A model object.

            data (torch.Tensor):
                Batch data.

            criterion (tabnet.criterions._Criterion):
                A `_Criterion` object to compute the final loss value.

            optimizer:
                A Pytorch optimizer.

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
        feats = feats.to(device).float()
        targets = targets.to(device).float()

        # forward
        preds, mask_loss, _ = model(feats)

        # calc loss
        task_loss = criterion(preds, targets)
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

        return updates

    def eval_batch(self):
        """ Not needed. """
        pass 


def get_trainer(training_type='tabnet_training'):
    SUPPORTED_TRAINER = {
        'tabnet_training': TabNetTrainer,
        'tabnet_pretraining': TabNetPretrainer,
        'swap_dae_pretraining': SwapDAEPreTrainer,
    }

    trainer = SUPPORTED_TRAINER.get(training_type)

    if trainer is not None:
        return trainer()
    else:
        raise ValueError(
            '%r is not a valid trainer type.'
            'Use `sorted(tabnet.core.SUPPORTED_TRAINER.keys())` '
            'to get valid options.' % training_type
            )