""" Base classes of the module `tabnet.metrics`. """

import abc 
import torch 
import numpy as np 


class Metric(abc.ABC):

    def __call__(self, preds, targets):
        preds, targets = \
            self._to_numpy(preds), self._to_numpy(targets)

        return self.score_func(preds, targets)

    @abc.abstractmethod
    def __repr__(self):
        raise NotImplementedError

    @abc.abstractmethod
    def score_func(self, preds, targets):
        raise NotImplementedError
    
    @classmethod
    def _to_numpy(cls, inputs):

        with torch.no_grad():

            if isinstance(inputs, torch.Tensor):
                inputs = inputs.data.cpu().numpy()

            elif not isinstance(inputs, np.ndarray):
                raise TypeError('Invalid input type : `{}`'.format(type(inputs)))

        return inputs
