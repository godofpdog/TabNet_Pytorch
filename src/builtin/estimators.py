""" Implementations of estimators. """

import torch

from ..core.base import TabNetBase, PostProcessorBase
from ..core.criterion import create_criterion


class IdentityPostProcessor(PostProcessorBase):
    def __init__(self, num_tasks, is_cuda):
        super(IdentityPostProcessor, self).__init__(num_tasks)

    def _build(self, num_tasks):
        for _ in range(num_tasks):
            self._processors.append(
                torch.nn.Identity()
            )

    def forward(self, x):
        """
        Define forward computation of `IdentityPostProcessor`.

        Arguments:
            x (list of Tensor):
                Outputs from `TabNetHead`.
        
        Returns:
            outputs (list of numpy.ndarray)

        """
        assert len(x) == len(self._processors)
        res = []

        for i, processor in enumerate(self._processors):
            res.append(
                processor(x[i])
            )

        return res


class TabNetRegressor(TabNetBase):
    def __init__(
        self, input_dims, output_dims, reprs_dims=8, atten_dims=8, num_steps=3, num_indep=2, num_shared=2, gamma=1.3,
        cate_indices=None, cate_dims=None, cate_embed_dims=1, batch_size=1024, virtual_batch_size=128, momentum=0.03,
        mask_type='sparsemax', task_weights=1, is_shuffle=True, num_workers=4, pin_memory=True, is_cuda=False, logger=None
    ):
        super(TabNetRegressor, self).__init__(
            input_dims, output_dims, reprs_dims, atten_dims, num_steps, num_indep, num_shared, gamma, 
            cate_indices, cate_dims, cate_embed_dims, batch_size, virtual_batch_size, momentum,
            mask_type, is_shuffle, num_workers, pin_memory, is_cuda, logger
        )
        """
        Initialization of `TabNetRegressor`.

        Arguments:
            input_dims (int): Dimension of input features. 
            output_dims (int or list): Dimension of output logits (list for muti-task). 
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
            task_weights (int or list of int): Loss weights of tasks.
            is_shuffle (bool): Flag of shuffle on epoch end or not. 
            num_workers (int): Number of thread for data loader. 
            pin_memory:
            is_cuda (bool): Use GPU or not. 
            logger (logging.Logger): System logger object.

        Returns:
            None

        """
        self.task_weights = task_weights 
        self._criterion = self._create_criterion()
        self._post_processor = IdentityPostProcessor(self.num_tasks, self.is_cuda)

    def _create_criterion(self):
        """
        Create default criterions of regression task.
        """
        if not isinstance(self.output_dims, (int, list)):
            raise TypeError(
                'Argument `output_dim` must be a `int` or `list of int` object, but got `{}`'\
                .format(type(self.output_dims))
            )

        return create_criterion(
            task_types='regression', logits_dims=self.output_dims, weights=self.task_weights, is_cuda=self.is_cuda
        )
        

class TabNetClassifier(TabNetBase):
    def __init__(
        self, input_dims, output_dims, reprs_dims=8, atten_dims=8, num_steps=3, num_indep=2, num_shared=2, gamma=1.3,
        cate_indices=None, cate_dims=None, cate_embed_dims=1, batch_size=1024, virtual_batch_size=128, momentum=0.03,
        mask_type='sparsemax', task_weights=1, is_shuffle=True, num_workers=4, pin_memory=True, is_cuda=False, logger=None
    ):
        super(TabNetClassifier, self).__init__(
            input_dims, output_dims, reprs_dims, atten_dims, num_steps, num_indep, num_shared, gamma, 
            cate_indices, cate_dims, cate_embed_dims, batch_size, virtual_batch_size, momentum,
            mask_type, is_shuffle, num_workers, pin_memory, is_cuda, logger
        )
        """
        Initialization of `TabNetClassifier`.

        Arguments:
            input_dims (int): Dimension of input features. 
            output_dims (int or list): Dimension of output logits (list for muti-task). 
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
            task_weights (int or list of int): Loss weights of tasks.
            is_shuffle (bool): Flag of shuffle on epoch end or not. 
            num_workers (int): Number of thread for data loader. 
            pin_memory:
            is_cuda (str or list of int): Use GPU or not. 
            logger (logging.Logger): System logger object.

        Returns:
            None

        """
        self.task_weights = task_weights
        self._criterion = self._create_criterion()

    def _create_criterion(self):
        """
        Create default criterions of classification task.
        """
        if not isinstance(self.output_dims, (int, list)):
            raise TypeError(
                'Argument `output_dim` must be a `int` or `list of int` object, but got `{}`'\
                .format(type(self.output_dims))
            )

        return create_criterion(
            task_types='classification', logits_dims=self.output_dims, weights=self.task_weights, is_cuda=self.is_cuda
        )