""" Implementation of base classes in this repo. """

import abc 
import torch
import numpy as np 
from collections import defaultdict
from sklearn.base import BaseEstimator
from torch.optim.optimizer import Optimizer

from .model import InferenceModel, PretrainModel
from .data import create_data_loader
from .model_builder import load_weights, build_model
from .solver import train_epoch, eval_epoch


class PostProcessorBase(abc.ABC, torch.nn.Module):
    """
    Base class of post processor.
    """
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

        if is_cuda:
            for processor in self._processors:
                processor.cuda()

    @abc.abstractmethod
    def _build(self, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def forward(self, x, **kwargs):
        raise NotImplementedError


class CustomizedLoss(abc.ABC, torch.nn.Module):
    """
    Base class for customized loss. 
    """
    def __init__(self):
        super(CustomizedLoss, self).__init__()
    
    @abc.abstractmethod
    def forward(self, x, **kwargs):
        raise NotImplementedError


class TabNetBase(abc.ABC, BaseEstimator):
    """
    Implementation of tabnet base class.
    """
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
        self._criterion = None 
        self._post_processor = None

    @abc.abstractmethod
    def _create_criterion(self, **kwargs):
        raise NotImplementedError
    
    @property
    def model_confgs_(self):
        return self._model_configs

    def _init_optimizer(self, optimizer, optimizer_params):
        if self._model is None:
            raise RuntimeError('Must build model before set optimizer.')
        
        if not issubclass(optimizer.__class__, Optimizer):
            raise TypeError('Invalid optimizer.')  # TODO support `str` as config 

        if not isinstance(optimizer_params, dict):
            raise TypeError('Invalid params type.')

        return optimizer(
            filter(lambda p: p.requires_grad, self._model.parameters()), **optimizer_params
        )

    def load_weights(self, path, model_type='inference_model'):
        """
        Load model weights. 
        Build model architecture if `self._model` is None. 
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
     
    def _create_data_loaders(self, feats, targets, valid_feats, valid_targets):
        train_loader = create_data_loader(
            feats, targets, self.batch_size, self.is_shuffle, self.num_workers, self.pin_memory
        )

        if valid_feats is not None and valid_targets is not None:
            valid_loader = create_data_loader(
                valid_feats, valid_targets, self.batch_size, self.is_shuffle, self.num_workers, self.pin_memory
            )
        else:
            valid_loader = None

        return train_loader, valid_loader

    def fit(
        self, feats, targets, batch_size=1024, max_epochs=2000, optimizer=None, optimizer_params=None, 
        metrics=None, scheduler=None, valid_feats=None, valid_targets=None, valid_metrics=None
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
            scheduler (str or ??): Training scheduler. 
            valid_feats (np.ndarray or pd.DataFrame): Validation features. 
            valid_targets (np.ndarray or pd.DataFrame): Validation targets.
            valid_metrics (str or ??): Evaluation metrics for validation set. 

        Returns:
            self

        """
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.metrics = metrics
        self.scheduler = scheduler
        self.valid_metrics = valid_metrics

        if self.logger is not None:
            self.logger.debug('[TabNet] create criterion.')

        self._criterion = self._create_criterion()

        if self.logger is not None:
            self.logger.debug('[TabNet] init optimizer.')

        self._optimizer = self._init_optimizer(optimizer, optimizer_params)

        if self.logger is not None:
            self.logger.debug('[TabNet] create data loaders.')

        train_loader, valid_loader = self._create_data_loaders(
            feats, targets, valid_feats, valid_targets
        )

        if self.logger is not None:
            self.logger.info('[TabNet] start training.')

        for epoch in range(1, self.max_epochs + 1):
            if self.logger is not None:
                self.logger.info(
                    '[TabNet] ==================== epoch : {} ===================='\
                    .format(epoch)
                )

            train_meter = train_epoch(
                self._model, train_loader, epoch, self._criterion, 
                self._optimizer, self._metrics, self.logger
            )

            if valid_loader is not None:
                valid_meter = eval_epoch()

        if self.logger is not None:
            self.logger.info('[TabNet] training complete.')

            self.logger.info(
                '[TabNet] ==================== Summary of training info ===================='
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

        self.set_params(**kwargs)

        _check_eval_model(self._model)
        _check_post_processor(self._post_processor)
        
        if len(feats) < self.batch_size:
            self.batch_size = len(feats)

        data_loader = create_data_loader(
            feats, None, self.batch_size, self.is_shuffle, self.num_workers, self.pin_memory
        )

        self._model.eval()
        predictions = dict()

        with torch.no_grad():

            for i, data in enumerate(data_loader):
                outputs, _ = self._model(data)
                processed_outouts = self._post_processor(outputs)
                
                for t in range(len(self.output_dims)):
                    pred = processed_outouts[t].cpu().numpy()

                    if i == 0:
                        predictions[t] = pred
                    else:
                        predictions[t] = np.vstack((predictions[t], pred))

        return predictions

    def _check_arguments(self):
        pass 


def _check_eval_model(model):
    """
    Eval model verification.
    """
    if model is None:
        raise RuntimeError('Must to build model before call `predict`.')

    elif not isinstance(model, InferenceModel):
        raise TypeError(
                'Invalid model type, use `convert_to_inference_model` before call `predict`.'
        )

    return None


def _check_post_processor(post_processor):
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


        