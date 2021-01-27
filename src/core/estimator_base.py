""" Implementation of base classes for the estimator. """

import abc 
import torch
import numpy as np 
from collections import defaultdict
from sklearn.base import BaseEstimator
from torch.optim.optimizer import Optimizer
from torch.optim import Adam
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau, 

from .model import InferenceModel, PretrainModel
from .data import create_data_loader
from .model_builder import load_weights, build_model
from .solver import train_epoch, eval_epoch
from .metrics import MetricBase
from ..builtin.metrics import create_metrics
from ..utils import show_message, Meter


class PostProcessorBase(abc.ABC, torch.nn.Module):
    """
    Base class of post processor.
    """
    _module_type = 'post_processor'

    def __init__(self, num_tasks, is_cuda=False, **kwargs):
        super(PostProcessorBase, self).__init__()
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
    def _build(self, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self, x, **kwargs):
        raise NotImplementedError


class TabNetBase(abc.ABC, BaseEstimator):
    """
    Implementation of tabnet base class.
    """
    _module_type = 'estimator'

    def __init__(
        self, input_dims, output_dims, reprs_dims=8, atten_dims=8, num_steps=3, num_indep=2, num_shared=2, gamma=1.3, 
        cate_indices=None, cate_dims=None, cate_embed_dims=1, batch_size=1024, virtual_batch_size=128, momentum=0.03,
        mask_type='sparsemax', is_shuffle=True, num_workers=4, pin_memory=True, is_cuda=False, logger=None):
        """
        Initialization of `TabNetBase`.

        Arguments:
            input_dims (int): Dimension of input features. 
            output_dims (int or list of int): Dimension of output logits (list for muti-task). 
            reprs_dims (int): Dimension of decision representation.  
            atten_dims (int): Dimension of attentive features. 
            num_steps (int): Number of decision steps. 
            num_indep (int): Number of step-specified `GLUBlock` in each `FeatureTransformer`. 
            num_shared (int): Number of shared fully-connected layers cross all steps. 
            gamma (float): Scaling factor for attention updates 
            cate_indices (list of int or int): Indices of categorical features. 
            cate_dims (list of int or int): Number of categories in each categorical features. 
            cate_embed_dims (list of int or int): Dimensions of representation of embedding layer. 
            batch_size (int): Sample size of one batch data. 
            virtual_batch_size (int): Virtual batch size in `GhostBatchNorm` module. 
            momentum (float): Momentum parameters in `GhostBatchNorm` module. 
            is_shuffle (bool): Flag of shuffle on epoch end or not. 
            num_workers (int): Number of thread for data loader. 
            pin_memory:
            is_cuda (bool): Use GPU or not.
            logger (logging.Logger): System logger object.

        """
        # TODO check inputs

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

        self._check_arguments()

        if isinstance(self.output_dims, int):
            self.num_tasks = 1
        else:
            self.num_tasks = len(self.output_dims)

        self._model_configs = {}

        model_configs = {
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

        self.model_confgs_.update(**model_configs)
        
        # sub modules
        self._model = None 
        self._optimizer = None 
        self._schedulers = None
        self._criterion = None 
        self._post_processor = None
        self._meters = {'train': Meter(), 'eval': Meter()}

    @abc.abstractmethod
    def _create_criterion(self, **kwargs):
        raise NotImplementedError
    
    @property
    def model_confgs_(self):
        return self._model_configs

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
            self._show_message(
                '[TabNet] use default optimizer',
                logger=self.logger, level='WARNING'
            )
            optimizer = Adam

        if optimizer_params is None:
            self._show_message(
                '[TabNet] use default optimizer params',
                logger=self.logger, level='WARNING'
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
        if self._model is None:
            raise RuntimeError('Must build model before init optimizer.')

        if self._optimizer is None:
            raise RuntimeError('Must init optimizer berfore init schedulers.')
        
        if schedulers is None:
            return None 

        if not isinstance(schedulers, list) and issubclass(schedulers, _LRScheduler):
            schedulers = [schedulers]
        else:
            raise TypeError(
                'Invalid type of the argument `schedulers`, expect subclsss of `_LRScheduler` or list of them but got {}'\
                    .format(type(schedulers))
                )

        if not isinstance(scheduler_params, list) and isinstance(scheduler_params, dict):
            scheduler_params = [scheduler_params]
        else:
            raise TypeError(
                'Invalid type of the argument `scheduler_params`, expect `dict` or list of dict but got {}'\
                    .format(type(scheduler_params))
                )

        if not all(isinstance(obj, _LRScheduler) for obj in schedulers):
            raise TypeError(
                'All elements of list `schedulers` must be the subclass of `_LRScheduler`.'
                )

        if not all(isinstance(obj, dict) for obj in scheduler_params):
            raise TypeError(
                'All elements of list `scheduler_params` must be a `dict` object.'
                )

        scheduler_objects = []

        for scheduler, params in zip(schedulers, scheduler_params):
            scheduler_objects.append(
                scheduler(self._optimizer, **params)
            )
        
        return scheduler_objects

    def build(self, path, model_type='inference_model'):
        """
        Build network architecture. 

        Arguments
            path (str): Weights path.
            model_type (str): Model type.

        Returns:
            self

        """

        if not model_type in ('inference_model', 'pretrain_model'):
            raise ValueError('Not supported model type (`inference` or `pretrain`)')
        
        if self._model is None:
            self._model = build_model(
                model_type=model_type, weights_path=path, is_cuda=self.is_cuda, **self._model_configs
            )

        try:
            load_weights(self._model, path)

            if self.logger is not None:
                self.logger.info(
                    '[TabNet] Sucessfully load weights from {}'.format(path)
                )

        except Exception as e:

            if self.logger is not None:
                self.logger.warning(
                    '[TabNet] Failed to load model from {}'.format(path)
                )
                self.logger.debug(e)
        
        return self

    def seve_weights(self, path):
        try:
            torch.save(self._model.state_dict(), path)
            self._show_message(
                '[TabNet] Sucessfully save weights to {}.'.format(path),
                logger=self.logger, level='INFO'
            )
        except Exception as e:
            self._show_message(
                '[TabNet] Failed to save weights. \n {}'.format(e),
                logger=self.logger, level='WARNING'
            )

        return None 
     
    def fit(
        self, feats, targets, batch_size=1024, max_epochs=2000, optimizer=None, optimizer_params=None, 
        metrics=None, schedulers=None, scheduler_params=None, valid_feats=None, valid_targets=None, valid_metrics=None
    ):
        """
        Fit TabNet model.

        Arguments:
            feats (np.ndarray or pd.DataFrame): Training features. 
            targets (np.ndarray or pd.DataFrame): Training targets. 
            batch_size (int): Batch size. 
            max_epochs (int): Max training epochs. 
            optimizer (subclass of torch.optim.optimizer.Optimizer): Optimizer. 
            optimizer_params (dict): Parameters of optimizer. 
            metrics (str or function?): Evaluate metrics. 
            schedulers (): Training schedulers. 
            scheduler_params
            valid_feats (np.ndarray or pd.DataFrame): Validation features. 
            valid_targets (np.ndarray or pd.DataFrame): Validation targets.
            valid_metrics (str or ??): Evaluation metrics for validation set. 

        Returns:
            self

        """
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.metrics = metrics
        self.schedulers = schedulers
        self.valid_metrics = valid_metrics

        self._show_message(
            '[TabNet] set metrics.',
            logger=self.logger, level='DEBUG'
        )

        self._metrics = self._set_metrics(self.metrics)

        self._show_message(
            '[TabNet] init optimizer.',
            logger=self.logger, level='DEBUG'
        )

        self._optimizer = self._init_optimizer(optimizer, optimizer_params)

        self._show_message(
            '[TabNet] init schedulers.',
            logger=self.logger, level='DEBUG'
        )

        self._schedulers = self._init_schedulers(schedulers, scheduler_params)

        self._show_message(
            '[TabNet] create data loaders.',
            logger=self.logger, level='DEBUG'
        )

        train_loader = self._create_data_loader(
            feats, targets, self.batch_size, self.is_shuffle, self.num_workers, self.pin_memory
        )

        if valid_feats is not None and valid_targets is not None:
            valid_loader = self._create_data_loader(
                valid_feats, valid_targets, self.batch_size, self.is_shuffle, self.num_workers, self.pin_memory
            )
        else:
            valid_loader = None 

        self._show_message(
            '[TabNet] start training.',
            logger=self.logger, level='INFO'
        )

        for epoch in range(1, self.max_epochs + 1):
            self._show_message(
                '[TabNet] ******************** epoch : {} ********************'.format(epoch),
                logger=self.logger, level='INFO'
            )

            train_meter = train_epoch(
                self._model, train_loader, epoch, self._post_processor, self._criterion, 
                self._optimizer, self._metrics, self.logger, self.device
            )

            self._update_meters(train_meter, 'train')

            if valid_loader is not None:
                valid_meter = eval_epoch()
            else:
                valid_meter = None

            if self._schedulers is not None:
                self._schedulers_step()

        self._show_message(
            '[TabNet] training complete.', 
            logger=self.logger, level='INFO'
        )

        self._show_message(
            '[TabNet] ******************** Summary Info ********************',
            logger=self.logger, level='INFO'
        )

    def pretrain(self):
        pass 

    def show_model(self):
        """
        Show model architecture.
        """
        if self._model is not None:
            if self.logger is not None:
                self.logger.info('[TabNet] show model architecture.')
            print(self._model)
        else:
            if self.logger is not None:
                self.logger.info('[TabNet] build model first.')
    
    def predict(self, feats, **kwargs):
        # TODO update params

        self._check_eval_model(self._model)
        self._check_post_processor(self._post_processor)
        
        if len(feats) < self.batch_size:
            self.batch_size = len(feats)

        data_loader = self._create_data_loader(
            feats, None, self.batch_size, False, self.num_workers, self.pin_memory
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
        # TODO update params
        # TODO for embedding encoding

        self._check_eval_model(self._model)

        if len(feats) < self.batch_size:
            self.batch_size = len(feats)

        data_loader = self._create_data_loader(
            feats, None, self.batch_size, False, self.num_workers, self.pin_memory
        )

        self._model.eval()
        
        predictions = dict()

        with torch.no_grad():

            for i, data in enumerate(data_loader):
                m_explain, masks = self._model.explain(data.to(self.device))

                return m_explain, masks

                print(m_explain.size())
                print(masks)
        #         processed_outouts = self._post_processor(outputs)
                
        #         for t in range(len(self.output_dims)):
        #             pred = processed_outouts[t].cpu().numpy()

        #             if i == 0:
        #                 predictions[t] = pred
        #             else:
        #                 predictions[t] = np.vstack((predictions[t], pred))

        # return predictions

    def _schedulers_step(self, ):
        # TODO  ReduceLROnPlateau wrapper to monitor other criterions

        for scheduler in self._schedulers:
            
            if isinstance(scheduler, ReduceLROnPlateau):
                
                if self._meters.get('eval').get('')
                scheduler.step(ev)


    def _update_meters(self, meter, meter_name='train'):
        updates = {}

        self._show_message(
            '[TabNet] -------------------- {} info --------------------'.format(meter_name),
            logger=self.logger, level='INFO'
        )

        for name in meter.names:
            stat_type = 'sum' if name == 'time_cost' else 'mean'
            stat = meter.get_statistics(name, stat_type)[name]

            self._show_message(
                '[TabNet] {} : {}'.format(name, stat),
                logger=self.logger, level='INFO'
            )

            updates[name] = stat 

        self._meters[meter_name].update(updates)

        return None

    @classmethod
    def _show_message(cls, msg, logger=None, level='DEBUG'):
        """
        Show estimator message, print message if `logger` is None.

        Arguments:
            msg (str):
                Estimator message.

            logger (logging.Logger, or None):
                A Python logger object.

            level (str):
                Logger level.

        Returns:
            None

        """
        show_message(msg, logger, level)
        return 

    def _check_arguments(self):
        pass 

    @classmethod
    def _set_metrics(cls, metrics):

        if metrics is None:
            return None 

        _metrics = []

        if not isinstance(metrics, list):
            if issubclass(metrics, MetricBase):
                _metrics.append(metrics)
                return _metrics

            elif isinstance(metrics, str):
                metrics = [metrics]
            
            else:
                raise TypeError('Invalid input type : `{}`'.format(type(metrics)))

        for metric in metrics:
            if isinstance(metric, str):
                _metrics.append(create_metrics(metric))
            
            elif issubclass(metric, MetricBase):
                _metrics.append(metric)
            
            else:
                raise TypeError('Invalid input type : `{}`'.format(type(metric)))

        return _metrics

    @classmethod
    def _create_data_loader(cls, feats, targets, batch_size, is_shuffle, num_workers, pin_memory):
        return create_data_loader(
            feats, targets, batch_size, is_shuffle, num_workers, pin_memory
        )

    @classmethod
    def _check_eval_model(cls, model):
        """
        Eval model verification.
        """
        if model is None:
            raise RuntimeError('Must to build model first.')

        elif not isinstance(model, InferenceModel):
            raise TypeError(
                    'Invalid model type, use `convert_to_inference_model`.'
            )

        return None

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
            if not issubclass(post_processor.__class__, PostProcessorBase):
                raise TypeError(
                    'Argument `post_processor` must be the subclass of `PostProcessorBase`.'
                )

        return None


        