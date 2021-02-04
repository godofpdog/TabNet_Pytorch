""" Implementation of TabNetClassifier. """

import torch 
import numpy as np 

from ._base import BaseTabNet, BasePostProcessor
from ..criterions import get_loss, create_criterion, Loss
from ..utils.validation import is_cls_score 


class ClassificationPostProcessor(BasePostProcessor):
    """
    Implementation of `ClassificationPostProcessor`
    default post prcocessor for classification task.
    """
    def __init__(self, output_dims, is_cuda):
        """
        Initialization of `ClassificationPostProcessor` object.
        
        Arguments:
            output_dims (int or list of int)
                Dimension of output logits (list for muti-task).

        Returns:
            None 

        """
        if not isinstance(output_dims, (int, list)):
            raise TypeError(
                    'Argument `output_dims` must be a `int` or `list of int` object, but got `{}`'\
                        .format(type(output_dims))
                    )
        
        if isinstance(output_dims, int):
            output_dims = [output_dims]

        if not all(isinstance(x, int) for x in output_dims):
            raise TypeError('Argument `output_dims` must be a `int` or `list of int` object')
        
        super(ClassificationPostProcessor, self).__init__(
            num_tasks=len(output_dims), is_cuda=is_cuda, output_dims=output_dims
        )

    def _build(self, num_tasks, output_dims):
        for dims in output_dims:

            if dims == 1:
                processor = torch.nn.Sigmoid()
            else:
                processor = torch.nn.Softmax()

            self._processors.append(processor)

        return None

    def forward(self, x, is_return_proba=False):
        """
        Define forward computation of `ClassificationPostProcessor`.

        Arguments:
            x (list of Tensor):
                Outputs from `TabNetHead`.

            is_return_proba (bool):
                If True, return both labels and the probabilities of all classes.
        
        Returns:
            labels (list of Tensor)
            probs (list of Tensor)

        """
        assert len(x) == len(self._processors)
        labels = []
        probs = []

        for i, processor in enumerate(self._processors):
            outputs = processor(x[i])

            if isinstance(processor, torch.nn.Sigmoid):
                label = (outputs > 0.5) * 1 
            else:
                label = outputs.argmax(dim=-1)

            labels.append(label)

            if is_return_proba:
                probs.append(
                    outputs
                )

        if is_return_proba:
            return labels, probs
        else:
            return labels


class TabNetClassifier(BaseTabNet):
    """
    Implementation of TabNetClassifier.
    """
    def __init__(
        self, input_dims, output_dims, reprs_dims=8, atten_dims=8, num_steps=3, num_indep=2, num_shared=2, gamma=1.3,
        cate_indices=None, cate_dims=None, cate_embed_dims=1, batch_size=1024, virtual_batch_size=128, momentum=0.03,
        mask_type='sparsemax', task_weights=1, criterions=None,  is_shuffle=True, num_workers=4, pin_memory=True, is_cuda=False, logger=None):
        
        super(TabNetClassifier, self).__init__(
            input_dims, output_dims, reprs_dims, atten_dims, num_steps, num_indep, num_shared, gamma, 
            cate_indices, cate_dims, cate_embed_dims, batch_size, virtual_batch_size, momentum,
            mask_type, is_shuffle, num_workers, pin_memory, is_cuda, logger
        )
        """
        Initialization of TabNetClassifier.

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
        Create default criterions of classification task.
        """
        losses = []

        # defult setting 
        if self.criterions is None:
            criterions = []

            if isinstance(self.output_dims, int):
                self.output_dims = [self.output_dims]

            elif not isinstance(self.output_dims, list):
                raise TypeError('Argument `self.output_dims` must be int or list of int.')

            for dims in self.output_dims:

                if dims < 1:
                    raise ValueError('Invalid output dimenssionb (must > 0)')

                if dims == 1:
                    criterions.append('bce')
                else:
                    criterions.append('ce')

        for criterion in criterions:

            if isinstance(criterion, str):
                loss = get_loss(criterion)

            elif is_cls_score(criterion):
                loss = criterion

            else:
                raise TypeError('Not supported criterion.')

            losses.append(loss)

        return create_criterion(losses, self.task_weights)

    def _build_post_processor(self):
        self._post_processor = ClassificationPostProcessor(self.output_dims, self.is_cuda)

        