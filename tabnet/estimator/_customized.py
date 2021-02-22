""" Implementation of `customized model`. """

from ._base import BaseTabNet, BasePostProcessor
from ..criterions import create_criterion


class CostumizedEstimator(BaseTabNet):
    """
    The interface to define a customized estimator by registering:
    (1) Valid loss object.
    (2) Valid post-processor.

    * Example:

        # init `CostumizedEstimator` object.

        my_model = CostumizedEstimator(
            input_dims=39, output_dims=[1], reprs_dims=8, atten_dims=8, 
            num_steps=4, num_indep=2, num_shared=1, virtual_batch_size=256
        )

        #  register customized loss & post-processor

        my_model.register_loss(MyLoss())
        my_model.register_postprocessor(MyPostProcessor())

    """

    def __init__(self, input_dims, output_dims, **kwargs):
        """
        Initialization of `CostumizedEstimator`.

        Arguments:
            input_dims (int):
                Dimension of input features.

            output_dims (int or list of int): 
                Dimension of output logits (list for muti-task).

        Returns:
            None

        """
        super(CostumizedEstimator, self).__init__(input_dims, output_dims, **kwargs):

    def register_loss(self, losses, weights=1):
        criterion = create_criterion(losses, weights)
        num_tasks = 1 if isinstance(self.output_dims, int) else len(self.output_dims)

        if criterion.num_tasks != num_tasks:
            raise ValueError('Number of loss functions must match the network architecture (output_dms).')

        return self

    def register_postprocessor(self, post_processors):
        if not issubclass(post_processors.__class__, BasePostProcessor):
            raise TypeError('Type of `post_processors` must be a subclass of `BasePostProcessor`.')
        
        self._post_processor = post_processors
        return self
        