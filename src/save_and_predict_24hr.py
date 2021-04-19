import numpy as np
import pandas as pd
import json
import pickle
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from Demand_class import Demand
from demand_helper import unscale_y
from rnn import create_layers_SimpleRNN, compile_model, predict_future
from keras.models import load_model

def prep_data(region, scaler, n_lag=24, n_ahead=24):
    demand = Demand(region)
    demand.load_data()
    demand.extend_time(hours=n_ahead)
    lag_df = demand.create_lag_variables(demand.dataframe, n_lag, n_ahead)
    X_train, X_test, y_train, _ = demand.scale_split(lag_df, demand.split_idx, scaler)
    X_train, X_test, y_train, _ = demand.reshape_for_rnn(X_train, X_test, y_train, _)
    return X_train, X_test, y_train

def create_24hr_model(X_train, y_train, filepath):
    model = Sequential()
    create_layers_SimpleRNN(model, (X_train.shape[1], 1))
    compile_model(model, X_train, y_train)
    model.save(filepath)

def load_saved_model(filepath):
    loaded_model = load_model(filepath)
    return loaded_model

def predict_future_demand(model, X_test, scaler, n_lag=24):
    preds = model.predict(X_test)
    unscaled_preds = unscale_y(preds, sclr, n_lag)
    return unscaled_preds

if __name__ == '__main__':

    sclr = MinMaxScaler()
    regions = ['US48', 'CAL', 'CAR', 'CENT', 'FLA', 'MIDA', 'MIDW', 'NE',
                 'NY', 'NW', 'SE', 'SW', 'TEN', 'TEX']
    predictions = []
    for region in regions:
        filepath = '../models/' + region + '_24.h5'
        X_train, X_test, y_train = prep_data(region, sclr)
        # create_24hr_model(X_train, y_train, filepath)  
        loaded_model = load_saved_model(filepath)
        preds = predict_future_demand(loaded_model, X_test, sclr)
        predictions.append(preds)

    print(len(predictions))
    print(predictions)