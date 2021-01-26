""" Implementations of estimators. """

import torch
import numpy as np 

from ..core.estimator_base import TabNetBase, PostProcessorBase
from ..core.criterion import create_criterion


class IdentityPostProcessor(PostProcessorBase):
    """
    Implementation of `IdentityPostProcessor`, 
        deafult post processor for regression task.
    """
    def __init__(self, num_tasks, is_cuda):
        super(IdentityPostProcessor, self).__init__(
            num_tasks=num_tasks, is_cuda=is_cuda
        )

    def _build(self, num_tasks):
        for _ in range(num_tasks):
            self._processors.append(
                torch.nn.Identity()
            )
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


class ClassificationPostProcessor(PostProcessorBase):
    """
    Implementation of `ClassificationPostProcessor`, 
        deafult post processor for classification task.
    """
    def __init__(self, num_classes, is_cuda):
        if not isinstance(num_classes, (int, list)):
            raise TypeError(
                    'Argument `num_classes` must be a `int` or `list of int` object, but got `{}`'\
                        .format(type(num_classes))
                    )
        
        if isinstance(num_classes, int):
            num_classes = [num_classes]

        if not all(isinstance(x, int) for x in num_classes):
            raise TypeError('Argument `num_classes` must be a `int` or `list of int` object')
        
        super(ClassificationPostProcessor, self).__init__(
            num_tasks=len(num_classes), is_cuda=is_cuda, num_classes=num_classes
        )

    def _build(self, num_tasks, num_classes):
        for classes in num_classes:

            if classes == 1:
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
        self._post_processor = ClassificationPostProcessor(self.output_dims, self.is_cuda)

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

    def predict_proba(self, feats, is_return_max=False):
        self._check_eval_model(self._model)
        self._check_post_processor(self._post_processor)
        
        if len(feats) < self.batch_size:
            self.batch_size = len(feats)

        data_loader = self._create_data_loader(
            feats, None, self.batch_size, self.is_shuffle, self.num_workers, self.pin_memory
        )

        self._model.eval()
        predictions = dict()

        with torch.no_grad():

            for i, data in enumerate(data_loader):
                outputs, _ = self._model(data)
                _, probs = self._post_processor(outputs, is_return_proba=True)
                
                for t in range(len(self.output_dims)):
                    pred = probs[t].cpu().numpy().reshape(-1, self.output_dims[t])

                    if i == 0:
                        predictions[t] = pred
                    else:
                        predictions[t] = np.vstack((predictions[t], pred))

        return predictions
