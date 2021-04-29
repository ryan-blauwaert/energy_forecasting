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
    """Accesses data from EIA site, loads into a class instance, 
    featurizes with lag variables, scales, and splits data into 
    X_train, X_test, and y_train arrays  

    Args:
        region (str): string abbreviation of region of interest
        scaler (obj): instance of sklearn MinMaxScaler
        n_lag (int, optional): Number of hours to be predicted.
        Defaults to 24.
        n_ahead (int, optional): Number of hours in advance the 
        predictions will be made. Defaults to 24.

    Returns:
        [obj, arr, arr, arr]: MinMaxScaler, X_train, X_test, and y_train
        arrays to be used by the RNN model
    """
    demand = Demand(region)
    demand.load_data()
    demand.extend_time(hours=n_ahead)
    lag_df = demand.create_lag_variables(demand.dataframe, n_lag, n_ahead)
    scaler, X_train, X_test, y_train, _ = demand.scale_split(lag_df, demand.split_idx, scaler)
    X_train, X_test, y_train, _ = demand.reshape_for_rnn(X_train, X_test, y_train, _)
    return scaler, X_train, X_test, y_train

def create_24hr_model(X_train, y_train, filepath):
    """Generates and saves to file RNN model which has been 
    trained using the X_train and y_train arrays

    Args:
        X_train (arr): Feature array of lag variables for training
        y_train (arr): Target array of actual MWH measurements
        filepath (str): path to file location where model will be saved
    """
    model = Sequential()
    create_layers_SimpleRNN(model, (X_train.shape[1], 1))
    compile_model(model, X_train, y_train)
    model.save(filepath)

def load_saved_model(filepath):
    """Loads a saved model from file for use in future 
    predictions

    Args:
        filepath (str): file location of the model to be loaded

    Returns:
        obj: The loaded, compiled model
    """
    loaded_model = load_model(filepath)
    return loaded_model

def predict_future_demand(model, X_test, scaler, n_lag=24):
    """Uses a pre-trained RNN model to generate predicted target values
    based on the feature array passed to it.

    Args:
        model (obj): pre-compiled RNN model
        X_test (arr): feature array of lag variables
        scaler (obj): Same scaler used to scale the data initially
        n_lag (int, optional): Number of lag variables used in feature
        array. Defaults to 24.

    Returns:
        arr: Array of predicted target variables
    """
    preds = model.predict(X_test)
    return preds

def create_24hr_list(region, n_ahead=24):
    """Creates a list of datetime strings to be used in displaying 
    predicted MWH values.

    Args:
        region (str): string abbreviation of region of interest
        n_ahead (int, optional): Number of hours in advance the 
        predictions will be made. Defaults to 24.

    Returns:
        [list]: List of datetime strings with length equal to the
        number of predictions made by the model
    """
    demand = Demand(region)
    demand.load_data()
    demand.extend_time(hours=n_ahead)
    hours = demand.dataframe.iloc[-24:, 1].astype('str').tolist()
    return hours

if __name__ == '__main__':

    sclr = MinMaxScaler()
    regions = ['US48', 'CAL', 'CAR', 'CENT', 'FLA', 'MIDA', 'MIDW', 'NE',
                 'NY', 'NW', 'SE', 'SW', 'TEN', 'TEX']
    
    predictions = []
    for region in regions:
        filepath = '../models/' + region + '_24.h5'
        sclr, X_train, X_test, y_train = prep_data(region, sclr)
        # create_24hr_model(X_train, y_train, filepath)  
        loaded_model = load_saved_model(filepath)
        preds = predict_future_demand(loaded_model, X_test, sclr)
        predictions.append(preds)

    print(len(predictions))
    print(predictions)

    