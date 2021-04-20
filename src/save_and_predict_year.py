import numpy as np
import pandas as pd
from Demand_class import Demand
from XGBoost_helper import find_gridsearch_best_params, fit_best_model, predict_year_future
import pickle

def prep_data_year(region):
    demand = Demand(region)
    demand.load_data()
    demand.extend_time(8760)
    demand.create_time_features()
    df = demand.time_features_df
    X_train, X_test, y_train, _ = demand.scale_split(df, demand.split_idx)
    return X_train, X_test, y_train

def create_year_model(X_train, y_train, grid, filepath):
    params = find_gridsearch_best_params(X_train, y_train, grid)
    model = fit_best_model(X_train, y_train, params)
    with open(filepath, 'wb') as f:
       pickle.dump(model, f) 

def load_pickle_model(filepath):
    with open(filepath, 'rb') as f:
        loaded_model = pickle.load(f)
    return loaded_model


if __name__ == '__main__':

    regions = ['US48', 'CAL', 'CAR', 'CENT', 'FLA', 'MIDA', 'MIDW', 'NE',
                 'NY', 'NW', 'SE', 'SW', 'TEN', 'TEX']

    predictions = []
    for region in regions[:2]:

        X_train, X_test, y_train = prep_data_year(region)
        grid = {'learning_rate': [.01, .1, .2],
                'max_depth': [2, 4, 8],
                'lambda': [.01, .1, 1],
                'alpha': [.01, .1, 1],
                'n_estimators': [500, 1000, 1500]}
        filepath = '../models/' + region + '_year.pkl'
        create_year_model(X_train, y_train, grid, filepath)

        loaded_model = load_pickle_model(filepath)
        preds = predict_year_future(loaded_model, X_test)
        predictions.append(preds)

    print(predictions)
    print(len(predictions))