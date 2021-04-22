import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Demand_class import Demand
from XGBoost_helper import find_gridsearch_best_params, fit_best_model, predict_year_future
import pickle
from eda_plotting import plot_timeseries
plt.style.use('seaborn-darkgrid')

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

def plot_and_save_year(region, preds):
    demand = Demand(region)
    demand.load_data()
    demand.extend_time(8760)
    x_vals = demand.dataframe.iloc[-8760:, 1].values
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x_vals, preds)
    path = './static/imgs/' + region + '_year.png'
    ax.set_xlabel('Date', size=16)
    ax.set_ylabel('Megawatthours', size=16)
    fig.tight_layout()
    fig.savefig(path, dpi=500)


if __name__ == '__main__':

    regions = ['US48', 'CAL', 'CAR', 'CENT', 'FLA', 'MIDA', 'MIDW', 'NE',
                 'NY', 'NW', 'SE', 'SW', 'TEN', 'TEX']

    predictions = []
    for region in regions:
    
        X_train, X_test, y_train = prep_data_year(region)
        print(len(X_train))
        print(len(X_test))
        grid = {'learning_rate': [.01],
                'max_depth': [2, 4, 8],
                'lambda': [.01],
                'alpha': [.01],
                'n_estimators': [500, 1000]}
        filepath = '../models/' + region + '_year.pkl'
        create_year_model(X_train, y_train, grid, filepath)

        loaded_model = load_pickle_model(filepath)
        preds = predict_year_future(loaded_model, X_test)
        predictions.append(preds)

    print(predictions)
    print(len(predictions))


    
 