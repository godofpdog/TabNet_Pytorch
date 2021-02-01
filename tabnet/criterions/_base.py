""" Base classes of the module `tabnet.criterions`. """

import abc 
import torch 
import torch.nn as nn 
from torch.nn.modules.loss import _Loss, _WeightedLoss


class Loss(nn.Module, abc.ABC):
    """
    Base class as a wrapper for loss modules in this repo. 
    """
    def __init__(self):
        super(Loss, self).__init__()
        self._loss_func = None 
    
    @abc.abstractmethod
    def score_func(self, preds, targets):
        raise NotImplementedError

    def forward(self, preds, targets, **kwargs):
        return self.score_func(preds, targets, **kwargs)


class _Criterion(nn.Module):
    """
    Compute the final loss by weighted sum of given losses.
    """
    def __init__(self, loss_layers, weights):
        super(_Criterion, self).__init__()
        """
        Initialization of `_Criterion` module.

        Arguments:
            loss_layers (supported loss module or list of them):
                loss modules.

            weights (int or float or list of them):
                weights of losses.

        Returns:
            None

        """
        self._build(loss_layers, weights)
        
    @property
    def num_tasks(self):
        return len(self._loss_layers)

    def forward(self, X, y):
        """
        Arguments:
            X (list of Tensor):
                Otuput tensors from `TabNetHead`
            
            y (Tensor):
                Ground truth of the task (shape = (batch_size, num_tasks))

        Returns:
            loss (Tensor):
                The final loss.

        """
        loss = 0
        
        for i, (loss_layer, weight) in enumerate(zip(self._loss_layers, self._weights)):
            loss += torch.mul(
                loss_layer(X[i], y[..., i].view(-1, 1)), 
                weight
            ) 

        return loss      

    def _build(self, layers, weights):

        def _is_loss_layer(module):
            return issubclass(
                module.__class__, (Loss, _Loss, _WeightedLoss)
            ) 

        if _is_loss_layer(layers):
            layers = [layers]

        elif isinstance(layers, list):
            if not all(_is_loss_layer(layer) for layer in layers):
                raise TypeError(
                        'All elements of argument `layers` must be subclass of {}, {} and {}.'\
                            .format('`Loss`', '`_Loss`', '`_WeightedLoss`')
                        )

        else:
            raise TypeError(
                    'Argument `layers` must be a supported loss module (subclass of {}, {} and {}) or list of them.'\
                        .format('`Loss`', '`_Loss`', '`_WeightedLoss`')
                    )
        
        if isinstance(weights, (int, float)):
            weights = [weights] * len(layers)
        
        elif isinstance(weights, list):
            if not all(isinstance(weight, (int, float)) for weight in weights):
                raise TypeError(
                        'All elements of argument `weight` must be  a `int` or `float` object'
                        )

            if len(layers) != len(weights):
                raise ValueError(
                    'Number of elements in `layers` and `weights` must be the same.'
                    )

        else:
            raise TypeError(
                    'Argument `weights` must be a `int` or `float` object or list of them.'
                    )

        self._loss_layers = nn.ModuleList(layers) 
        self._weights = weights 

        return None
