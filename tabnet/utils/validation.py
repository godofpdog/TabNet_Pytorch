""" Utilities for input validation """

import numpy as np 
import pandas as pd 
from sklearn.utils import assert_all_finite

from ..metrics import Metric
from ..criterions import Loss


def check_input_data(input_data):
    """
    Check if the input data is valid data type.

    Arguments:
        input_data (object):
            Input data to verify.

    Returns:
        None 

    """
    if is_has_na(input_data):
        raise ValueError('Input data contain missing value.') 

    if not is_finite(input_data):
        raise ValueError('Input data contain infiite value.')

    return None 
        

def is_finite(input_data):
    """
    Check if the input data has no infinite value.

    Arguments:
        input_data (object):
            Input data to verify.

    Returns:
        (bool):
            Is all value finite or not.

    """
    return assert_all_finite(input_data)


def is_numpy(input_data):
    """
    Check if the input data is a numpy.ndarray.

    Arguments:
        input_data (object):
            Input data to verify.

    Returns:
        (bool):
            Is a numpy.ndarray or not.

    """
    return isinstance(input_data, np.ndarray)


def is_pandas(input_data):
    """
    Check if the input data is a pandas.DataFrame.

    Arguments:
        input_data (object):
            Input data to verify.

    Returns:
        (bool):
            Is pandas.DataFrame or not.

    """
    return isinstance(input_data, pd.DataFrame)


def is_has_na(input_data):
    """
    Check if the input data has missing values.

    Arguments:
        input_data (object):
            Input data to verify.

    Returns:
        (bool):
            Has missing values or not.

    """
    if is_numpy(input_data) or is_pandas(input_data):

        if isinstance(input_data, np.ndarray):
            return np.isnan(input_data).any()
        else:
            return input_data.isnull().sum().sum() > 0

    else:
        raise TypeError(
                'Type of input data must be '
                '`np.ndarray` or `pd.DataFrame`.'
                )


def is_metric(input_object):
    """
    Check if the input object is a valid metric scorer

    Arguments:
        input_object (object):
            Input object to verify.

    Returns:
        (bool):
            Is an valid metric scorer or not.

    """
    return issubclass(input_object.__class__, Metric)


def is_loss(input_object):
    """
    Check if the input object is a valid loss scorer

    Arguments:
        input_object (object):
            Input object to verify.

    Returns:
        (bool):
            Is an valid loss scorer or not.

    """
    return issubclass(input_object.__class__, Loss)


def is_scoring_object(input_object):
    """
    Check if the input object is a valid scorer 
    subclass of `tabnet.metrics.Metric` or `tabnet.criterions.Loss`.

    Arguments:
        input_object (object):
            Input object to verify.

    Returns:
        (bool):
            Is an valid scorer or not.

    """
    return is_metric(input_object) and is_loss(input_object)
    

def is_cls_score(scorer):
    """
    Check if the scorer is for the classification task.

    Arguments:
        scorer (subclass of `tabnet.metrics.Metric` or `tabnet.criterions.Loss`)
            Input scorer.

    Return:
        (bool):
            Is for the classification task or not.

    """
    if not is_scoring_object(scorer):
        raise TypeError(
            'Invalid scorer (not subclass of `tabnet.metrics.Metric` or `tabnet.criterions.Loss`).'
            )

    return getattr(scorer, '_task_type', None) == 'classification'
    

def is_reg_score(scorer):
    """
    Check if the scorer is for the regression task.

    Arguments:
        scoring_object (subclass of `tabnet.metrics.Metric` or `tabnet.criterions.Loss`)
            Input scorer.

    Return:
        (bool):
            Is for the regression task or not.

    """
    if not is_scoring_object(scorer):
        raise TypeError(
            'Invalid scoring object (not subclass of `tabnet.metrics.Metric` or `tabnet.criterions.Loss`).'
            )

    return getattr(scorer, '_task_type', None) == 'regression'
