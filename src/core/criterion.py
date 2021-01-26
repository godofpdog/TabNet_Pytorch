""" Implementations of criterion modules. """

import abc 
import torch 
import torch.nn as nn 
from torch.nn.modules.loss import _Loss, _WeightedLoss

# TODO test code, check len


class CustomizedLoss(abc.ABC, nn.Module):
    """
    Base class for customized loss. 
    """
    _module_type = 'loss'

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

    @property
    def num_tasks(self):
        return len(self._loss_layers)

    def forward(self, x, y):
        """
        Arguments:
            x (list of Tensor):
                Otuput tensors from `TabNetHead`
            
            y (Tensor):
                Ground truth of the task (shape = (batch_size, num_tasks))

        """
        loss = 0
        
        for i, (loss_layer, weight) in enumerate(zip(self._loss_layers, self._weights)):
            loss += torch.mul(
                loss_layer(x[i], y[..., i].view(-1, 1)), 
                weight
            ) 

        return loss      

    def _check_inputs(self, layers, weights):

        def _is_loss_layer(module):
            return issubclass(
                module.__class__, (CustomizedLoss, _Loss, _WeightedLoss)
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

        self._loss_layers = nn.ModuleList(layers) 
        self._weights = weights 

        return None


_REG_TYPES = ('reg', 'regression')
_CLS_TYPES = ('cls', 'classification')


def create_criterion(task_types, logits_dims, weights, is_cuda):
    """
    Create default criterion.

    Arguments:
        task_types (str or list of str):
            Task types, MSE loss for regression and Cross Entropy loss for classification.
            (TODO: BCE for binary classification.)

        logits_dims (int or list of int):
            Dimension of output logits from the model.

        weights (int or list of int):
            Loss weights of multi-task training.

        is_cuda (bool):
            Use GPUs or not 

    Returns:
        criterion (Criterion): 
            An `Criterion` object.

    """
    losses = []

    if not isinstance(task_types, (str, list)):
        raise TypeError(
            'Argument `task_types` must br a `str` or `list or str` object, but got `{}`'\
                .format(type(task_types))
            )
        
    if not isinstance(logits_dims, (int, list)):
        raise TypeError(
            'Argument `logits_dims` must br a `int` or `list or int` object, but got `{}`'\
                .format(type(logits_dims))
            )

    if isinstance(task_types, str):
        task_types = [task_types]

    if isinstance(logits_dims, int):
        logits_dims = [logits_dims]

    if len(task_types) != len(logits_dims):
        raise ValueError(
            'Number of elements in `task_types` and `logits_dims` must be the same.'
            )

    if isinstance(weights, list):
        if len(task_types) != len(weights):
            raise ValueError(
                'Number of elements in `task_types` and `weights` must be the same.'
                )

    elif isinstance(weights, (int, float)):
        weights = [weights]

    else:
        raise TypeError()
    
    # build loss func
    for i in range(len(task_types)):

        if task_types[i] in _REG_TYPES and logits_dims[i] == 1:
            losses.append(
                nn.MSELoss()
            )

        elif task_types[i] in _CLS_TYPES and logits_dims[i] == 1:
            losses.append(
                nn.BCEWithLogitsLoss()
            )
        
        elif task_types[i] in _CLS_TYPES and logits_dims[i] >= 2:
            losses.append(
                nn.CrossEntropyLoss()
            )
        
        else:
            raise ValueError('Invalid combination of arguments.')

    return Criterion(losses, weights)


if __name__ == '__main__':
    task_types = ['reg', 'cls', 'cls']
    logits_dims  = [1, 1, 3]
    weights = [0.2, 0.3, 0.5]

    criterion = create_criterion(
        task_types, logits_dims, weights
    )

    x1 = torch.rand((32, 1))
    x2 = torch.rand((32, 1))
    x3 = torch.rand((32, 3)) 


    y1 = torch.rand((32, 1))
    y2 = torch.rand((32, 1))
    y3 = torch.rand((32))

    print(x1)

    print(criterion([x1, x2.float(), x3], [y1, y2.float(), y3.float()]))

    nn.CrossEntropyLoss()(
        torch.Tensor([0.9, 0.1]).unsqueeze(0), 
        torch.Tensor([1]).long()
    )    
     