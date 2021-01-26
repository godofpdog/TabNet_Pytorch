""" Utilities for this repo. """

import os
import abc
import copy
import numpy as np 
import pandas as pd 
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder

# TODO
# handlers.py


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return 


def show_message(msg, logger=None, level='DEBUG'):
        """
        Show estimator message, print message if `logger` is None.

        Arguments:
            msg (str):
                Estimator message.

            logger (logging.Logger, or None):
                A Python logger object.

            level (str):
                Logger level.

        Returns:
            None

        """
        if logger is not None:
            if level == 'DEBUG':
                logger.debug(msg)
            elif level == 'INFO':
                logger.info(msg)
            elif level == 'WARNING':
                logger.warning(msg)
            elif level == 'ERROR':
                logger.error(msg)
            elif level == 'CRITICAL':
                logger.critical(msg)
            else:
                raise ValueError('Invalid level.')
        else:
            print('[{}]'.format(level) + msg)
        
        return 
    

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

        Arguments:
            input_features (np.ndarray, pd.DataFrame): Input raw features. 
            cate_indices (int or list of int): Indices of categorical features. 

        Returns:
            self

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
        
        Arguments:
            x: input raw features

        Returns:
            None

        """
        # TODO return indices 

        for i, index in self.cate_indices:
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
            
            # if not isinstance(val, (int, float, bool)):
            #     raise TypeError(
            #         'Not supported val type, only support `int`, `float` and `bool`.'
            #     )

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



if __name__ == "__main__":
    meter = Meter()
    
    for i in range(10):
        updates = {
            'a': i,
            'b': 1
        }

        meter.update(updates)
    
    print(meter['a'])
    print(meter['b'])

    print(meter.get_statistics('a', stat_type='mean'))

        





        

