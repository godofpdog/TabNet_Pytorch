""" Implementation of base classes in this repo. """

import abc 
from sklearn.base import BaseEstimator
from .model import TabNetHead, TabNetEncoder, TabNetDecoder
from .data import create_data_loader


class TabNetBase(abc.ABC):
    """
    Implementation of tabnet base class.
    """
    def __init__(
        self, input_dims, output_dims, reprs_dims=8, atten_dims=8, num_steps=3, num_indep=2, num_shared=2, gamma=1.3, 
        cate_indices=None, cate_dims=None, cate_embed_dims=None, batch_size=1024, virtual_batch_size=128, momentum=0.03,
        is_shuffle=True, num_workers=4, pin_memory=True, device=None, mask_type='sparsemax', verbose=0):
        """
        Initialization of `TabNetBase`.
        :params input_dims: Dimension of input features. (int)
        :params output_dims: Dimension of output logits (list for muti-task). (int or list)
        :params reprs_dims: Dimension of decision representation. (int) 
        :params atten_dims: Dimension of attentive features. (int)
        :params num_steps: Number of decision steps. (int)
        :params num_indep: Number of step-specified `GLUBlock` in each `FeatureTransformer`. (int)
        :params num_shared: Number of shared fully-connected layers cross all steps. (int)
        :params gamma: Scaling factor for attention updates (float) 
        :params cate_indices: Indices of categorical features. (list of int or int)
        :params cate_dims: Number of categories in each categorical features. (list of int or int)
        :params cate_embed_dims: Dimensions of representation of embedding layer. (list of int or int)
        :params batch_size: Sample size of one batch data. (int)
        :params virtual_batch_size: Virtual batch size in `GhostBatchNorm` module. (int)
        :params momentum: Momentum parameters in `GhostBatchNorm` module. (float)
        :params is_shuffle: Flag of shuffle on epoch end or not. (bool)
        :params num_workers: Number of thread for data loader. (int)
        :params pin_memory:
        :params device: Usage decvice. (str or list of int)
        :params verbose:
        """
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

        # other training params
        self.is_shuffle = is_shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.device = device

        self._model = None 

    def load_weights(self, path, model_type):
        pass 

    def set_params(self):
        pass 

    def _update_fit_params(self):
        pass
     
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


    def fit(self, feats, targets, batch_size=1024, optimizer=None, optimizer_params=None, scheduler=None, valid_feats=None, valid_targets=None):
        self._update_fit_params()
        train_loader, valid_loader = self._create_data_loaders(feats, targets, valid_feats, valid_targets)

        # main loop
        self.max_epochs = 1000

        for epoch in range(self.max_epochs):
            # train

            for data in train_loader:
                feats, targets = data 
                feats = feats.to(self.device)
                targets = targets.to(self.device)

    def pretrain(self):
        pass 
    
    @abc.abstractmethod
    def _compute_losses(self):
        raise NotImplementedError 

class TabNetRegressor(TabNetBase):
    def __init__(
        self, input_dims, output_dims, reprs_dims=8, atten_dims=8, num_steps=3, num_indep=2, num_shared=2, gamma=1.3,
        cate_indices=None, cate_dims=None, cate_embed_dims=None, batch_size=1024, virtual_batch_size=128, momentum=0.03,
        is_shuffle=True, num_workers=4, pin_memory=True, device=None, verbose=0):
        """
        Implementation of tabnet regreesor.
        """
        pass 