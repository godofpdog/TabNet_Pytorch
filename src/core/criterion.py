""" Implementations of criterion modules. """

import abc 
import torch 
import torch.nn as nn 
from torch.nn.modules.loss import _Loss, _WeightedLoss

# TODO test code


class CustomizedLoss(abc.ABC, nn.Module):
    """
    Base class for customized loss. 
    """
    def __init__(self):
        super(CustomizedLoss, self).__init__()
        
    def forward(self, x):
        return self._forward(x)

    @abc.abstractmethod
    def _forward(self, x, **kwargs):
        raise NotImplementedError


class Criterion(nn.Module):
    """
    Compute the final loss by weighted sum of given losses.
    """
    def __init__(self, loss_layers, weights):
        super(Criterion, self).__init__()
        """
        Initialization of `Criterion` module.

        Arguments:
            loss_layers (supported loss module or list of them):
                loss modules.

            weights (int or float or list of them):
                weights of losses.

        Returns:
            None

        """
        self._check_inputs(loss_layers, weights)

    def forward(self, x):
        loss = 0
        
        for loss_layer, weight in zip(self._loss_layers, self._weights):
            loss += torch.mul(loss_layer(x), weight) 

        return loss      

    def _check_inputs(self, layers, weights):

        def _is_loss_layer(module):
            return issubclass(
                module, (CustomizedLoss, _Loss, _WeightedLoss)
            ) 

        if _is_loss_layer(layers):
            layers = [layers]

        for layer in layers:
            if not _is_loss_layer(layer):
                raise TypeError(
                    'Not supported loss layer, only support the subclass of {}, {} and {}.'\
                        .format('`CustomizedLoss`', '`_Loss`', '`_WeightedLoss`')
                )
        
        if isinstance(weights, list):
            if len(layers) != len(weights):
                raise ValueError(
                    'Number of elements in `layers` and `weights` must be the same.'
                )

        if isinstance(weights, (int, float)):
            weights = [weights] * len(layers)

        self._loss_layers = layers 
        self._weights = weights 

        return None


def create_criterion(task_types, logits_dims):
    """
    Create default criterion.

    Arguments:
        task_types (str or list of str):
            Task types, MSE loss for regression and Cross Entropy loss for classification.
            (TODO: BCE for binary classification.)

    Returns:
        logits_dims (int or list of int):
            Dimension of output logits from the model.

    """

    reg_types = ('reg', 'regression')
    cls_types = ('cls', 'classification')

    

    