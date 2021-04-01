import numpy as np
import pandas as pd
from Demand_class import Demand


def train_test(X, y, split):
    if isinstance(split, str):
        idx = X.index.get_loc(split)
    else:
        idx = split
    X_train, X_test = X.iloc[:idx], X.iloc[idx:]
    y_train, y_test = y.iloc[:idx], y.iloc[idx:]
    return X_train, X_test, y_train, y_test 



if __name__ == '__main__':
