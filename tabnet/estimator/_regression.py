""" Implementation of TabNetRegressor. """

import torch 
import numpy as np 

from ._base import BaseTabNet, BasePostProcessor
from ..criterions import get_loss, create_criterion, Loss


class IdentityPostProcessor(BasePostProcessor):
    """
    Implementation of `IdentityPostProcessor`
    The deafult post processor for regression task.
    """
    def __init__(self, num_tasks, is_cuda):
        super(IdentityPostProcessor, self).__init__(
            num_tasks=num_tasks, is_cuda=is_cuda
        )

    def _build(self, num_tasks):
        for _ in range(num_tasks):
            self._processors.append(torch.nn.Identity())
        return None 

    def forward(self, x):
        """
        Define forward computation of `IdentityPostProcessor`.

        Arguments:
            x (list of Tensor):
                Outputs from `TabNetHead`.
        
        Returns:
            outputs (list of Tensor)

        """
        assert len(x) == len(self._processors)
        outputs = []

        for i, processor in enumerate(self._processors):
            outputs.append(
                processor(x[i])
            )

        return outputs


class TabNetRegressor(BaseTabNet):
    """
    Implementation of TabNetRegressor.
    """
    def __init__(
        self, input_dims, output_dims, reprs_dims=8, atten_dims=8, num_steps=3, num_indep=2, num_shared=2, gamma=1.3,
        cate_indices=None, cate_dims=None, cate_embed_dims=1, batch_size=1024, virtual_batch_size=128, momentum=0.03,
        mask_type='sparsemax', task_weights=1, criterions=None, is_shuffle=True, num_workers=4, pin_memory=True, is_cuda=False, logger=None):
        
        super(TabNetRegressor, self).__init__(
            input_dims, output_dims, reprs_dims, atten_dims, num_steps, num_indep, num_shared, gamma, 
            cate_indices, cate_dims, cate_embed_dims, batch_size, virtual_batch_size, momentum,
            mask_type, is_shuffle, num_workers, pin_memory, is_cuda, logger
        )

        """
        Initialization of `TabNetRegressor`.

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

            task_weights (int or list of int): 
                Loss weights of tasks.

            criterions (str or subclasses of `tabnet.criterions.Loss` of list of them):
                Loss functions of training criterions.

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
        self.task_weights = task_weights 
        self.criterions = criterions

        # build criterion
        self._criterion = self._build_criterion()

        # build post processor
        self.num_tasks = len(output_dims) if isinstance(output_dims, list) else 1
        self._build_post_processor()

    def _build_criterion(self):
        """
        Create default criterions of regression task.
        """
        # TODO check regression type
        losses = []

        # defult setting 
        if self.criterions is None:
            self.criterions = 'mse'

        if not isinstance(self.criterions, list):
            criterions = [self.criterions]

        for criterion in criterions:

            if isinstance(criterion, str):
                loss = get_loss(criterion)
            else:
                loss = criterion

            losses.append(loss)

        return create_criterion(losses, self.task_weights)

    def _build_post_processor(self):
        self._post_processor = IdentityPostProcessor(self.num_tasks, self.is_cuda)