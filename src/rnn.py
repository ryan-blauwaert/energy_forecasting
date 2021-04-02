import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from keras.layers import Dense, Dropout, SimpleRNN
from keras.models import Sequential
from Demand_class import Demand
from eda_plotting import plot_predictions_vs_true
from demand_helper import unscale_y, mean_abs_percent_error

def create_layers_SimpleRNN(model, input_shape, units=200, activation='tanh', dropout=0.15):
    """Creates layers of the RNN model passed in. 

    Args:
        model (RNN obj): instance of a neural network model.
        input_shape (tup): shape of the input layer
        units (int, optional): Number of units in each layer; Defaults to 200. 
        activation (str, optional): Activation function.; Defaults to 'tanh'.
        dropout (float, optional): Dropout ratio; Defaults to 0.15.
    """
    model.add(SimpleRNN(units, activation=activation, return_sequences=True, 
                                input_shape=input_shape))
    model.add(Dropout(dropout))

    model.add(SimpleRNN(units, activation=activation, return_sequences=True))
    model.add(Dropout(dropout))

    model.add(SimpleRNN(units, activation=activation, return_sequences=False))
    model.add(Dropout(dropout))

    model.add(Dense(1))

def compile_and_predict(model, X_train, y_train, X_test, optimizer='adam', 
                            loss='MSE', epochs=10, batch_size=1000):
    """Compiles the RNN model passed to it using X_train and y_Train; predicts
    target variable from X_test.

    Args:
        model (RNN obj): Built RNN to be compiled and used to predict
        target variables
        X_train (arr): Training feature matrix
        y_train (arr): Training target matrix
        X_test (arr): Test feature matrix
        optimizer (str, optional): type of optimization used to compile 
        the RNN model. Defaults to 'adam'.
        loss (str, optional): Loss metric to be used to compile the 
        RNN model. Defaults to 'MSE'.
        epochs (int, optional): Number of epochs over which to train 
        RNN model. Defaults to 10.
        batch_size (int, optional): Batch size to be used in each step 
        during RNN training. Defaults to 1000.

    Returns:
        arr: Array of target variable predictions based on X_test
        matrix.
    """
    model.compile(optimizer=optimizer, loss=loss)

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    
    predictions = model.predict(X_test)
    return predictions



if __name__ == '__main__':

    plt.style.use('seaborn-darkgrid')

    mod = Sequential()
    create_layers_SimpleRNN(mod, input_shape=(24, 1))
    print(mod.summary())