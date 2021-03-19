
import os
import abc 
import copy
import numpy as np  
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return 


class _BasePreprocessor(abc.ABC):
    """
    Implementation of preprocessor base class.
    """
    def __init__(self):
        pass 

    def _check_data(self, data):
        if not isinstance(data, (np.ndarray, pd.DataFrame, pd.Series)):
            raise TypeError(
                'Type of input data must be `np.ndarray` or `pd.DataFrame` or pd.Series.'
            )

    @abc.abstractmethod
    def fit(self, x, *args, **kwargs):
        raise NotImplementedError
    
    @abc.abstractmethod
    def infer(self, *args, **kwargs):
        raise NotImplementedError

    def fit_infer(self, x, **kwargs):
        return self.fit(x, **kwargs).infer(x)


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

        Arguments:
            input_features (np.ndarray, pd.DataFrame): Input raw features. 
            cate_indices (int or list of int): Indices of categorical features. 

        Returns:
            self

        """
        # self.check_infer_data(input_features)

        if not isinstance(cate_indices, (int, list)):
            raise TypeError('Type of argument `cate_indices` must be `int` or `list`')

        if isinstance(cate_indices, int):
            self.cate_indices = [cate_indices]
        else:
            self.cate_indices = cate_indices

        for i in self.cate_indices:
            self._label_encoders.append(
                LabelEncoder().fit(input_features[..., i].reshape(-1))
            )

        return self 

    def infer(self, input_features, is_check=True):
        """
        Return transformed features and cate_dims.

        Arguments:
            input_features (np.ndarray, pd.DataFrame): 
                Input raw features. 

            is_check (bool): 
                Flag for inference data verification or not. 
        
        Returns:
            trans (np.ndarray): 
                Transformed features. 

        """
        self._check_data(input_features)
        
        if is_check:
            self.check_infer_data(input_features)

        if isinstance(input_features, pd.DataFrame):
            input_features = input_features.values

        trans = copy.deepcopy(input_features)
        cate_dims = []

        for i, index in enumerate(self.cate_indices):
            feats = input_features[..., index].reshape(-1)
            trans[..., index] = self._label_encoders[i].transform(feats)
            cate_dims.append(len(self._label_encoders[i].classes_))

        return trans, cate_dims

    def check_infer_data(self, x):
        """
        Inference data verification.
        
        Arguments:
            x: input raw features

        Returns:
            None

        """
        # TODO return indices 

        for i, index in enumerate(self.cate_indices):
            feats = x[..., index]

            if not set(np.unique(feats)).issubset(set(self._label_encoders[i].classes_)):
                raise ValueError('Unseen category in fitting phase.')
        
        return None 


class Meter:
    """
    Record training/evaluation history.
    """
    def __init__(self):
        self._history = defaultdict(list)

    @property
    def names(self):
        return [k for k in self._history.keys()]

    def update(self, updates):
        """
        Update Meter.

        Arguments:
            updates (dict): Updates information. 
                (key: variable name, val: values)

        Returns:
            self

        """
        if not isinstance(updates, dict):
            raise TypeError(
                'Argument `updates` must be a `dict` object, but got `{}`'\
                    .format(type(updates))
            )
            
        for key, val in updates.items():
            self._history[key].append(val)

        return self

    def __getitem__(self, name):
        """
        Get recordings.

        Arguments:
            names (str or list of str): Variable names in `_history`.

        Returns:
            (list of scalers): recording valuse.

        """
        return self._history[name]

    def get_statistics(self, names, stat_type='mean'):
        """
        Get statistics of recordings.

        Arguments:
            names (str or list of str): Variable names in `_history`.
            stat_type (str): Statistics type.

        Retuens:
            stat (int, float): Statistics. 

        """
        # TODO ckeck inputs

        _SUPPORTED_STATS = {
            'mean': np.mean,
            'median': np.median,
            'std': np.std,
            'max': np.max,
            'min': np.min,
            'sum': np.sum
        }

        if isinstance(names, str):
            names = [names]

        stat_func = _SUPPORTED_STATS.get(stat_type)
        
        if stat_func is None:
            raise ValueError('Not supported `stat_type`.')

        res = dict()

        for name in names:
            res[name] = stat_func(self._history[name])

        return res