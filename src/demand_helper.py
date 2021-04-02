import numpy as np
import pandas as pd
from Demand_class import Demand


def unscale_y(y_array, scaler, n_lag_variables):
    """Reshapes target matrix so that it can be inverse
    scaled to extract predictions in original scale.

    Args:
        y_array (arr): target matrix; either predictions 
        or y_test
        scaler (scaler obj): instance of scaler, e.g. MinMaxScaler()
        n_lag_variables (int): number of lag variables used in the 
        feature matrix. Will add this many columns of zeros so that the
        y matrix matches the shape of the scaler

    Returns:
        [arr]: unscaled y_matrix 
    """
    zeros = np.zeros((len(y_array), n_lag_variables))
    y_with_zeros = np.concatenate([y_array, zeros], axis=1)
    unscaled_y = scaler.inverse_transform(y_with_zeros)[:, 0]
    return unscaled_y


def mean_abs_percent_error(y_test, y_pred):
    """Calculates mean absolute percent error between y_test
    matrix and predictions from a supervised learning model.

    Args:
        y_test (arr): matrix of actual target values
        y_pred (arr): matrix of predicted target values

    Returns:
        float: error metric
    """
    y_test, y_pred = np.array(y_test), np.array(y_pred)
    return np.mean(np.abs((y_test - y_pred)/y_test)) * 100

    
if __name__ == '__main__':
    pass