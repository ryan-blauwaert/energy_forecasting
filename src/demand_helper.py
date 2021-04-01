import numpy as np
import pandas as pd
from Demand_class import Demand


def unscale_y(y_array, scaler, n_lag_variables):
    zeros = np.zeros((len(y_array), n_lag_variables))
    y_with_zeros = np.concatenate([y_array, zeros], axis=1)
    unscaled_y = scaler.inverse_transform(y_with_zeros)[:, 0]
    return unscaled_y


def mean_abs_percent_error(y_test, y_pred):
    y_test, y_pred = np.array(y_test), np.array(y_pred)
    return np.mean(np.abs((y_test - y_pred)/y_test)) * 100

    
if __name__ == '__main__':
