""" Base classes of the module `tabnet.estimator`. """

import os 
import abc
import json
import torch
import numpy as np 
import pandas as pd
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from sklearn.base import BaseEstimator

from ..utils.logger import show_message
from ..utils.utils import Meter, mkdir
from ..utils.validation import is_metric, check_input_data
from ..metrics import get_metric, Metric
from ..criterions import get_loss
from ..core._data import SwapNoiseDataset
from ..core import (
    build_model, load_weights, create_data_loader, InferenceModel, get_trainer, ModelConverter
)

# TODO 
# (1) the inference batch size issue (V)
# (2) pre-train model supported explain


class BaseTabNet(BaseEstimator, abc.ABC):
    def __init__(
        self, input_dims, output_dims, reprs_dims=8, atten_dims=8, num_steps=3, num_indep=2, num_shared=2, gamma=1.3, 
        cate_indices=None, cate_dims=None, cate_embed_dims=1, batch_size=1024, virtual_batch_size=128, momentum=0.03,
        mask_type='sparsemax', is_shuffle=True, num_workers=4, pin_memory=True, is_cuda=False, logger=None):
        """
        Initialization of `BaseTabNet`.

        Arguments:
            input_dims (int):
                Dimension of input features.

            output_dims (int or list of int): 
                Dimension of output logits (list for muti-task).

            reprs_dims (int): 
                Dimension of decision representation.  

            atten_dims (int): 
                Dimension of attentive features. 

            num_steps (int): 
                Number of decision steps. 

            num_indep (int): 
                Number of step-specified `GLUBlock` in each `FeatureTransformer`. 

            num_shared (int): 
                Number of shared fully-connected layers cross all steps. 

            gamma (float):  
                Scaling factor for attention updates.

            cate_indices (list of int or int): 
                Indices of categorical features. 

            cate_dims (list of int or int): 
                Number of categories in each categorical features. 

            cate_embed_dims (list of int or int): 
                Dimensions of representation of embedding layer. 

            batch_size (int): 
                Sample size of one batch data. 

            virtual_batch_size (int): 
                Virtual batch size in `GhostBatchNorm` module. 

            momentum (float): 
                Momentum parameters in `GhostBatchNorm` module. 

            is_shuffle (bool): 
                Flag of shuffle on epoch end or not. 

            num_workers (int):
                Number of thread for data loader. 

            pin_memory (bool):
                If True, will automatically put the fetched data Tensors in pinned memory, 
                and thus enables faster data transfer to CUDA-enabled GPUs.

            is_cuda (bool): 
                Use GPU or not.

            logger (logging.Logger): 
                system logger object.

        Returns:
            None

        """ 
        # arguments
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.reprs_dims = reprs_dims
        self.atten_dims = atten_dims
        self.num_steps = num_steps
        self.num_indep = num_indep
        self.num_shared = num_shared
        self.gamma = gamma
        self.cate_indices = cate_indices
        self.cate_dims = cate_dims
        self.cate_embed_dims = cate_embed_dims
        self.batch_size = batch_size
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        self.mask_type = mask_type
        self.is_shuffle = is_shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.is_cuda = is_cuda
        self.logger = logger
        self.device = 'cuda' if self.is_cuda else 'cpu'

        # model configs
        self._get_model_configs()

        # sub modules
        self._model = None 
        self._optimizer = None 
        self._schedulers = None
        self._criterion = None 
        self._post_processor = None
        self._meters = {'train': Meter(), 'valid': Meter()}

        self.INFER_BATCH_SIZE = 512

    def _get_model_configs(self):
        """
        Model configurations to build network architecture.
        """
        self._model_configs = {
            'input_dims': self.input_dims,
            'output_dims': self.output_dims,
            'cate_indices': self.cate_indices,
            'cate_dims': self.cate_dims,
            'cate_embed_dims': self.cate_embed_dims,
            'reprs_dims': self.reprs_dims,
            'atten_dims': self.atten_dims,
            'num_steps': self.num_steps,
            'gamma': self.gamma,
            'num_indep': self.num_indep,
            'num_shared': self.num_shared,
            'virtual_batch_size': self.virtual_batch_size,
            'momentum': self.momentum,
            'mask_type': self.mask_type
        }

        return None 

    def build(self, path):
        """
        Build network architecture. 

        Arguments
            path (str): 
                Model path (dir contains model configuraton and weights).

        Returns:
            self

        """
        if self._model is not None:
            show_message(
                '[TabNet] Model architecture has already been built.',
                logger=self.logger, level='WARNING'
            )
            return

        try:
            with open(os.path.join(path, 'model_config.json')) as f:
                _model_configs = json.load(f)

            self._model = build_model(
                model_type='inference_model', weights_path=os.path.join(path, 'weights.pt'), is_cuda=self.is_cuda, **_model_configs
            )

        except Exception as e:
            show_message(
                '[TabNet] Failed to load saved model. Will build model architecture by given setting. \n{}'.format(e),
                logger=self.logger, level='WARNING'
            )

            self._model = build_model(
                model_type='inference_model', weights_path=None, is_cuda=self.is_cuda, **self._model_configs
            )
        
        return self  

    def save(self, path):
        """
        Save weights and estimator configurations.

        Arguments:
            path (str):
                Save path.

        Returns:
            None

        """
        if self._model is not None:
            self._model = ModelConverter.to_inference(self._model, self._model_configs, self.device)
            show_message('[TabNet] Convert to inference model.', logger=self.logger, level='INFO')
        else:
            raise RuntimeError('There is no model to save. Must to build and fit model before call `save`.')

        try:
            mkdir(path)
            torch.save(self._model.state_dict(), os.path.join(path, 'weights.pt'))
            
            with open(os.path.join(path, 'model_config.json'), 'w') as f:
                json.dump(self._model_configs, f) 

            show_message(
                '[TabNet] Successfully save model configuration and weights to {}.'.format(path),
                logger=self.logger, level='INFO'
            )
        
        except Exception as e:
            show_message(
                '[TabNet] Failed to save model. \n{}'.format(e),
                logger=self.logger, level='WARNING'
            )
        
        return 

    def show_model(self):
        if self._model is not None:
            show_message(
                '[TabNet] Show model architecture.',
                logger=self.logger, level='INFO'
            )

            print(self._model)

        else:
            show_message(
                '[TabNet] Must to build model first.',
                logger=self.logger, level='INFO'
            )

    def _init_optimizer(self, optimizer, optimizer_params):
        """
        Initialization of optimizer,

        Arguments:
            optimizer (subclass of torch.optim.optimizer.Optimizer):
                Pytorch optimizer. If None, use tne default optimizer `Adam` with learning rate = 1e-3.

            optimizer_params (dict):
                Parameters of the optimizer. If None, apply default setting.

        Returns:
            An optimizer object.

        """
        if self._model is None:
            raise RuntimeError('Must build model before init optimizer.')

        if optimizer is None:
            show_message(
                '[TabNet] use default optimizer',
                logger=self.logger, level='INFO'
            )
            optimizer = Adam

        if optimizer_params is None:
            show_message(
                '[TabNet] use default optimizer params',
                logger=self.logger, level='INFO'
            )
            optimizer_params = {
                'lr': 1e-3
            }
        
        if not issubclass(optimizer, Optimizer):
            raise TypeError('Invalid optimizer.')  # TODO support `str` as config 

        if not isinstance(optimizer_params, dict):
            raise TypeError('Invalid params type.')

        return optimizer(
            filter(lambda p: p.requires_grad, self._model.parameters()), **optimizer_params
        )

    def _init_schedulers(self, schedulers, scheduler_params):
        """
        Initialization of learning rate schedulers.

        Arguments:
            schedulers (subclass of `torch.optim.lr_scheduler._LRScheduler` or list of them):
                Pytorch learning rate schedulers. If multiple inputs, will call `step()` method 
                sequentially in training phase. If None, train without scheduler.

             scheduler_params (dict or list of dict):
                Parameters of the schedulers.

        Returns:
            List of scheduler objects.

        """
        # TODO check inputs 

        if self._model is None:
            raise RuntimeError('Must build model before init optimizer.')

        if self._optimizer is None:
            raise RuntimeError('Must init optimizer berfore init schedulers.')
        
        if schedulers is None:
            return None 

        if not isinstance(schedulers, list) :
            schedulers = [schedulers]

        if not isinstance(scheduler_params, list) and isinstance(scheduler_params, dict):
            scheduler_params = [scheduler_params]

        scheduler_objects = []

        for scheduler, params in zip(schedulers, scheduler_params):
            scheduler_objects.append(
                scheduler(self._optimizer, **params)
            )
        
        return scheduler_objects

    def pretrain(self, feats, batch_size=None, max_epochs=2000, 
                 optimizer=None, optimizer_params=None, schedulers=None, 
                 scheduler_params=None, algorithm_params=None):

        """
        SSL pre-training on un-labeled dataset.

        Arguments:
            feats (numpy.ndarray or pandas.DataFrame):
                Input features.

            batch_size (int):
                Batch size.

            max_epochs (int):
                Maximum of training epochs.

            optimizer:
                A Pytorch optimizer.

            schedulers:
                Pytorch scheduler(s).

            scheduler_params (dict or list of dict):
                Parameters of the training scheduler(s).

            algorithm_params (dict):
                Parameters of the training algorithm (including the `traning algorithm`).

                - Example:
                    algorithm_params = {
                        'algorithm': 'tabnet_pretraining',
                        'mask_rate': 0.2
                    }

        """
        # TODO get citeriion 

        # the default algorithm
        if algorithm_params is None:
            algorithm_params = {'algorithm': 'tabnet_pretraining', 'mask_rate': 0.2}

        # build model
        if self._model is not None:
            self._model = ModelConverter.to_pretrain(self._model, algorithm_params, self._model_configs, self.device)
            show_message('[TabNet] Convert to pretrain model.', logger=self.logger, level='INFO')
        else:
            raise RuntimeError('Must to build model before call `pretrain`.')
        
        if batch_size is None:
            batch_size = self.batch_size

        self._metrics = self.set_metrics(None)
        self._optimizer = self._init_optimizer(optimizer, optimizer_params)
        self._schedulers = self._init_schedulers(schedulers, scheduler_params)

        # setup trainer and criterion
        algorithm_name = algorithm_params['algorithm']

        if algorithm_name in ('tabnet', 'tabnet_pretraining'):
            algorithm = 'tabnet_pretraining'

        elif algorithm_name in ('swap_dae', 'swapdae', 'swap_dae_pretraining'):
            algorithm = 'swap_dae_pretraining'

        else:
            raise ValueError('Not supported pre-training algorithm.')
        
        trainer = get_trainer(training_type=algorithm)
        criterion = get_loss(algorithm).to(self.device)

        if algorithm == 'swap_dae_pretraining':
            if isinstance(feats, np.ndarray):
                feats = pd.DataFrame(feats)

            swap_rate = algorithm_params.get('swap_rate')
            if swap_rate is None:
                swap_rate = 0.15

            # dummy 
            train_loader = SwapNoiseDataset(feats, swap_rate=swap_rate)

        else:
            train_loader = create_data_loader(
                feats, None, batch_size, self.is_shuffle, self.num_workers, self.pin_memory
            )

        # start training
        show_message('[TabNet] start training.', logger=self.logger, level='INFO')

        for epoch in range(max_epochs):
            show_message(
                '[TabNet] ******************** epoch : {} ********************'.format(epoch),
                logger=self.logger, level='INFO'
            )

            train_meter = trainer.train_epoch(
                self._model, train_loader, criterion, self._optimizer, 
                self._post_processor, self._metrics, self.device, 
                batch_size=batch_size, shuffle=self.is_shuffle, 
                pin_memory=self.pin_memory, num_workers=self.num_workers
            )

            self._update_meters(train_meter, 'train')
            
            # update schedulers
            if self._schedulers is not None:
                self._schedulers_step()

        show_message(
            '[TabNet] training complete.', 
            logger=self.logger, level='INFO'
        )

        show_message(
            '[TabNet] ******************** Summary Info ********************',
            logger=self.logger, level='INFO'
        )

        del trainer
        return self

    def fit(self, feats, targets, batch_size=None, max_epochs=2000, 
            optimizer=None, optimizer_params=None, schedulers=None, scheduler_params=None,
            metrics=None, valid_feats=None, valid_targets=None, valid_metrics=None):

        """
        Fit TabNet model on specified tasks.
        """

        # build model
        if self._model is not None:
            self._model = ModelConverter.to_inference(self._model, self._model_configs, self.device)
            show_message('[TabNet] Convert to inference model.', logger=self.logger, level='INFO')
        else:
            raise RuntimeError('Must to build model before call `fit`.')

        if batch_size is None:
            batch_size = self.batch_size

        # setup metrics
        self._metrics = self.set_metrics(metrics)

        # init optimizer
        self._optimizer = self._init_optimizer(optimizer, optimizer_params)

        # init schedulers
        self._schedulers = self._init_schedulers(schedulers, scheduler_params)

        # create data loaders
        train_loader = create_data_loader(
            feats, targets, batch_size, self.is_shuffle, self.num_workers, self.pin_memory
        )

        if valid_feats is not None and valid_targets is not None:
            valid_loader = create_data_loader(
                valid_feats, valid_targets, batch_size, self.is_shuffle, self.num_workers, self.pin_memory
            )
        else:
            valid_loader = None
        
        # init trainer
        trainer = get_trainer(training_type='tabnet_training')

        # start training
        show_message('[TabNet] start training.', logger=self.logger, level='INFO')

        for epoch in range(max_epochs):
            show_message(
                '[TabNet] ******************** epoch : {} ********************'.format(epoch),
                logger=self.logger, level='INFO'
            )

            train_meter = trainer.train_epoch(
                self._model, train_loader, self._criterion, self._optimizer, 
                self._post_processor, self._metrics, self.device
            )

            self._update_meters(train_meter, 'train')
            
            # validation
            if valid_loader is not None:
                valid_meter = trainer.eval_epoch(
                    self._model, valid_loader, self._criterion, 
                    self._post_processor, self._metrics, self.device
                )

                self._update_meters(valid_meter, 'valid')

            # update schedulers
            if self._schedulers is not None:
                self._schedulers_step()

        show_message(
            '[TabNet] training complete.', 
            logger=self.logger, level='INFO'
        )

        show_message(
            '[TabNet] ******************** Summary Info ********************',
            logger=self.logger, level='INFO'
        )

        del trainer
        return self
        
    def predict(self, feats, **kwargs):
        """
        Inference by given input features.

        Arguments:
            feats (numpy.ndarray, pd.DataFrame):
                Input features.

        Returns:
            predictions (dict of numpy.ndarray)

        """
        # convert model
        if self._model is not None:
            self._model = ModelConverter.to_inference(self._model, self._model_configs, self.device)
            show_message('[TabNet] Convert to inference model.', logger=self.logger, level='INFO')
        else:
            raise RuntimeError('Must to build model before call `fit`.')

        self._check_post_processor(self._post_processor)
        check_input_data(feats)

        data_loader = create_data_loader(
            feats, None, self.batch_size, False, self.num_workers, self.pin_memory, is_drop_last=False
        )

        self._model.eval()
        predictions = dict()

        with torch.no_grad():

            for i, data in enumerate(data_loader):
                outputs, _ = self._model(data.to(self.device))
                processed_outouts = self._post_processor(outputs)
                
                for t in range(len(self.output_dims)):
                    pred = processed_outouts[t].cpu().numpy()

                    if i == 0:
                        predictions[t] = pred
                    else:
                        predictions[t] = np.vstack((predictions[t], pred))

        return predictions
    
    def explain(self, feats, **kwargs):
        """
        Calculate the instance-wise explanation.

        Arguments:
            feats (numpy.ndarray or pandas.DataFrame):
                Input features.

        Returns:
            instance_importances (numpy.ndarray):
                Instance-wise feature importances matrix with shape = (num_rows, num_cols).

            output_masks (dict contains numpy.ndarray):
                Attentive masks in each decision step.

        """
        # TODO for embedding encoding
        # TODO global importances

        # convert model
        if self._model is not None:
            self._model = ModelConverter.to_inference(self._model, self._model_configs, self.device)
            show_message('[TabNet] Convert to inference model.', logger=self.logger, level='INFO')
        else:
            raise RuntimeError('Must to build model before call `fit`.')

        data_loader = create_data_loader(
            feats, None, self.batch_size, False, self.num_workers, self.pin_memory, is_drop_last=False
        )

        self._model.eval()
        
        output_masks = dict()
        instance_importances = None 

        with torch.no_grad():

            for i, data in enumerate(data_loader):
                importances, masks = self._model.explain(data.to(self.device))
                importances = importances.cpu().numpy()

                if i == 0:
                    instance_importances = importances

                    for k, v in masks.items():
                        output_masks[k] = v.cpu().numpy()

                else:
                    instance_importances = np.vstack((instance_importances, importances))
                    
                    for k, v in masks.items():
                        output_masks[k] = np.vstack((output_masks[k], v.cpu().numpy()))

        return instance_importances, output_masks

    def extract(self, feats, **kwargs):
        """
        Extract featgure representation.

        Arguments:
            feats (numpy.ndarray or pandas.DataFrame):
                Input features.

        Returns:
            reprs (numpy.ndarray):
                Feature representation extracted from TabNetEncoder.

        """
        if self._model is None:
            raise ValueError('Must to load or fit model before call `extract`.')

        check_input_data(feats)

        data_loader = create_data_loader(
            feats, None, self.batch_size, False, self.num_workers, self.pin_memory, is_drop_last=False
        )

        self._model.eval()

        reprs = None 

        with torch.no_grad():
            
            for i, data in enumerate(data_loader):
                outputs, _ = self._model.tabnet_encoder(data.to(self.device))
                outputs = torch.sum(torch.stack(outputs, dim=0), dim=0)
                
                if i == 0:
                    reprs = outputs.cpu().numpy()
                else:
                    reprs = np.vstack((reprs, outputs.cpu().numpy()))

        return reprs
  
    def _schedulers_step(self):
        # TODO  ReduceLROnPlateau wrapper to monitor other criterions

        for scheduler in self._schedulers:
            if isinstance(scheduler, ReduceLROnPlateau):
                if len(self._meters['eval']) != 0:
                    loss = self._meters['eval']['total_loss'][-1]
                else:
                    loss = self._meters['train']['total_loss'][-1]
                scheduler.step(loss)
            else:
                scheduler.step()
        
        return None

    def _update_meters(self, meter, meter_name='train'):
        updates = {}

        show_message(
            '[TabNet] -------------------- {} info --------------------'.format(meter_name),
            logger=self.logger, level='INFO'
        )

        for name in meter.names:
            stat_type = 'sum' if name == 'time_cost' else 'mean'
            stat = meter.get_statistics(name, stat_type)[name]

            show_message(
                '[TabNet] {} : {}'.format(name, stat),
                logger=self.logger, level='INFO'
            )

            updates[name] = stat 

        self._meters[meter_name].update(updates)

    @classmethod
    def set_metrics(cls, metrics=None):
        """
        Setup metrics.

        Arguments:
            metrics (str or subclass of Metric or list of them):
                Metrics to be used.
        
        Returns:
            scorers (list contains subclasses of `tabnet.metrics.Metric`)
                List of metric scorers.

        """

        def _set_func(_metric):
            """ Inner function for setting one metric scorer. """

            if isinstance(_metric, str):
                scorer = get_metric(_metric)

            elif is_metric(_metric):
                scorer = _metric

            else:
                raise TypeError(
                    '%r is not a valid metric.'
                    'Support metrics code by `sorted(tabnet.metrics.SUPPORTED_METRICS.keys())` '
                    'or valid metric scorer (subclass of tabnet.metrics.Metric).' % _metric
                    )

            return scorer

        scorers = []

        if metrics is None:
            return 

        if not isinstance(metrics, list):
            scorers.append(_set_func(metrics))

        else:
            for metric in metrics:
                scorers.append(_set_func(metric))

        return scorers

    @classmethod
    def _check_post_processor(cla, post_processor):
        """
        Post processor verification.
        """
        if post_processor is None:
            raise RuntimeError(
                'Must to define the post processor to get the final prediction.'
            )
        else:
            if not issubclass(post_processor.__class__, BasePostProcessor):
                raise TypeError(
                    'Argument `post_processor` must be the subclass of `PostProcessorBase`.'
                )

        return None

    @abc.abstractmethod
    def _build_post_processor(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _build_criterion(self):
        raise NotImplementedError


class BasePostProcessor(abc.ABC, torch.nn.Module):
    """
    Base class of post processor.
    """
    def __init__(self, num_tasks, is_cuda=False, **kwargs):
        """
        Initialization of `PostProcessorBase` module.

        Arguments:
            num_tasks (int):
                Number of task.

            is_cuda (bool): 
                Use GPU or not.

        Retuens:
            None 

        """
        super(BasePostProcessor, self).__init__()
        self._processors = []

        if not isinstance(num_tasks, int):
            raise TypeError(
                'Argument `num_tasks` must be an `int` object, bot got `{}`.'\
                    .format(type(num_tasks))
                )
        elif num_tasks < 1:
            raise ValueError(
                'Number of tasks must be greater than 1, but got {}'\
                    .format(num_tasks)
                )

        self._build(num_tasks, **kwargs)

        if is_cuda and len(self._processors) > 0:
            for processor in self._processors:
                processor.cuda()

    @abc.abstractmethod
    def _build(self, num_tasks, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self, x, **kwargs):
        raise NotImplementedError
