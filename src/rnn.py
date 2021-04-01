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
    model.compile(optimizer=optimizer, loss=loss)

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    
    predictions = model.predict(X_test)
    return predictions



if __name__ == '__main__':

    plt.style.use('seaborn-darkgrid')

    mod = Sequential()
    create_layers_SimpleRNN(mod, input_shape=(24, 1))
    print(mod.summary())