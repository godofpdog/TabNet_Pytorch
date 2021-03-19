""" Implementation of `customized model`. """

from ._base import BaseTabNet, BasePostProcessor
from ..criterions import create_criterion


class CustomizedEstimator(BaseTabNet):
    """
    The interface to define a customized estimator by registering:
    (1) Valid loss object.
    (2) Valid post-processor.

    * Example:

        # init `CostumizedEstimator` object.

        my_model = CustomizedEstimator(
            input_dims=39, output_dims=[1], reprs_dims=8, atten_dims=8, 
            num_steps=4, num_indep=2, num_shared=1, virtual_batch_size=256
        )

        #  register customized loss & post-processor

        my_model.register_loss(MyLoss())
        my_model.register_postprocessor(MyPostProcessor())

    """

    def __init__(self, input_dims, output_dims, **kwargs):
        """
        Initialization of `CustomizedEstimator`.

        Arguments:
            input_dims (int):
                Dimension of input features.

            output_dims (int or list of int): 
                Dimension of output logits (list for muti-task).

        Returns: 
            None

        """
        super(CustomizedEstimator, self).__init__(input_dims, output_dims, **kwargs)

    def register_loss(self, losses, weights=1):
        criterion = create_criterion(losses, weights)
        num_tasks = 1 if isinstance(self.output_dims, int) else len(self.output_dims)

        if criterion.num_tasks != num_tasks:
            raise ValueError('Number of loss functions must match the network architecture (output_dms).')
        
        self._criterion = criterion
        return self

    def register_postprocessor(self, post_processor):
        if not issubclass(post_processor.__class__, BasePostProcessor):
            raise TypeError('Type of `post_processor` must be a subclass of `BasePostProcessor`.')
        
        self._post_processor = post_processor
        return self

    def _build_criterion(self):
        return 

    def _build_post_processor(self):
        return
        