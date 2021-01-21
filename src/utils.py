""" Utilities for this repo. """

import abc
import copy
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder

# TODO
# handlers.py

class _BasePreprocessor(abc.ABC):
    """
    Implementation of preprocessor base class.
    """
    def __init__(self):
        pass 

    def _check_data(self, data):
        if isinstance(data, (np.ndarray, pd.DataFrame, pd.Series)):
            raise TypeError(
                'Type of input data must be `np.ndarray` or `pd.DataFrame` or pd.Series.'
            )

    @abc.abstractmethod
    def fit(self, x, *args, **kwargs):
        raise NotImplementedError
    
    @abc.abstractmethod
    def infer(self, *args, **kwargs):
        raise NotImplementedError

    def fit_infer(self, x):
        return self.fit(x).infer(x)


class CatePreprocessor(_BasePreprocessor):
    """
    Implementation of `CatePreprocessor` module which is enable to 
    transform the categorical features to the embedding dimension.
    """
    def __init__(self):
        super(CatePreprocessor, self).__init__()
        self._label_encoders = []

    def fit(self, input_features, cate_indices):
        """
        Fit CatePreprocessor.
        :params input_features: Input raw features. (np.ndarray, pd.DataFrame)
        :params cate_indices: Indices of categorical features. (int or list of int)
        :return self
        """
        self._check_data(input_features)

        if not isinstance(cate_indices, (int, list)):
            raise TypeError('Type of argument `cate_indices` must be `int` or `list`')

        if isinstance(cate_indices, int):
            cate_indices = [cate_indices]
        self.cate_indices = cate_indices

        for _ in self.cate_indices:
            self._label_encoders.append(
                LabelEncoder().fit(input_features)
            )

        return self 

    def infer(self, input_features, is_check=True):
        """
        Return transformed features and cate_dims.
        :params input_features: Input raw features. (np.ndarray, pd.DataFrame)
        :params is_check: Flag for inference data verification or not. (bool)
        :return trans: Transformed features. (np.ndarray)
        """
        self._check_data(input_features)
        
        if is_check:
            self.check_infer_data(input_features)

        if isinstance(input_features, pd.DataFrame):
            input_features = input_features.values

        trans = copy.deepcopy(input_features)
        cate_dims = []

        for i, index in self.cate_indices:
            feats = input_features[..., index].reshape(-1)
            trans[..., index] = self._label_encoders[i].transform(feats)
            cate_dims.append(len(self._label_encoders[i].classes_))

        return trans, cate_dims

    def check_infer_data(self, x):
        """
        Inference data verification.
        :params x: input raw features
        :return None
        """
        # TODO return indices 

        for i, index in self.cate_indices:
            feats = x[..., index]

            if not set(np.unique(feats)).issubset(set(self._label_encoders[i].classes_)):
                raise ValueError('Unseen category in fitting phase.')
        
        return None 


        





        

