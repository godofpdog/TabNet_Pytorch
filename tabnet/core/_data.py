""" Utilities for data smapling. """

import numpy as np 
import pandas as pd
from torch.utils.data import DataLoader, Dataset


def _check_data(feats, targets):
    """
    input data verification.
    """

    if not isinstance(feats, (np.ndarray, pd.DataFrame, pd.Series)):
        raise TypeError('Type of input data `feats` must be `np.ndarray` or `pd.DataFrame`.') 
    
    if targets is not None:

        if not isinstance(targets, (np.ndarray, pd.DataFrame, pd.Series)):
            raise TypeError('Type of input data `targets` must be `np.ndarray` or `pd.DataFrame` or `pd.Series`.')
        
        if len(feats) != len(targets):
            raise ValueError('Sample size of input data `feats` must be equal to `targets`.')

        if isinstance(targets, np.ndarray):
            if np.isnan(targets).any():
                raise ValueError('Missing values in targets.')
        else:
            if targets.isnull().sum().sum() > 0:
                raise ValueError('Missing values in targets.')

    return


class TabularDataset(Dataset):
    def __init__(self, feats, targets=None):
        _check_data(feats, targets)
        
        if isinstance(feats, (pd.DataFrame, pd.Series)):
            feats = feats.values

        if isinstance(targets, (pd.DataFrame, pd.Series)):
            targets = targets.values

        self._feats = feats.astype(np.float32)

        if targets is not None:
            self._targets = targets.astype(np.float32)
        else:
            self._targets = None
        
    def __getitem__(self, i):
        if self._targets is not None:
            return self._feats[i, ...], self._targets[i, ...]
        else:
            return self._feats[i, ...]
        
    def __len__(self):
        return len(self._feats)


def create_data_loader(feats, targets, batch_size=1024, is_shuffle=True, num_workers=2, pin_memory=True, is_drop_last=True):
    dataset = TabularDataset(feats, targets)
    data_loader = DataLoader(
        dataset, batch_size, shuffle=is_shuffle, drop_last=is_drop_last,  # NOTE drop_last is for gbn
        num_workers=num_workers, pin_memory=pin_memory
    )

    return data_loader
    