""" Implementations of model builders. """

import os 
import abc 
import torch


class _BaseBuilder(abc.ABC):
    def __init__(self, weights_path=None):
        self.weights_path = weights_path

    def build(self, is_cuda=False):
        model = self._build()
        self._load_weights(model)

        if is_cuda:
            model = model.cuda()

        return model 

    def _load_weights(self, model):

        if self.weights_path is not None:
            try:
                model.load_state_dict(
                    torch.load(self.weights_path)
                )

            except Exception as e:
                print(e)
                self._init_weights(model)
        else:
            self._init_weights(model)

    @abc.abstractmethod
    def _build(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _init_weights(self, model):
        raise NotImplementedError


class TabNetEncoderBuilder(_BaseBuilder):
    def __init__(self):
        pass 

