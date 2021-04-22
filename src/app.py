# Demonstrates Bootstrap version 3.3 Starter Template
# available here: https://getbootstrap.com/docs/3.3/getting-started/#examples


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from XGBoost_helper import predict_year_future
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, render_template, redirect, url_for, request
from save_and_predict_year import prep_data_year, load_pickle_model, plot_and_save_year
from save_and_predict_24hr import prep_data, load_saved_model, predict_future_demand, create_24hr_list
from keras.models import load_model
app = Flask(__name__)

# home page
REGION = ['US48', 'CAL', 'CAR', 'CENT', 'FLA', 'MIDA', 'MIDW', 'NE',
                 'NY', 'NW', 'SE', 'SW', 'TEN', 'TEX']
@app.route('/')
def index():
    
    return render_template('index.html', region=REGION)

@app.route('/forecast')
def forecast():
    reg = request.args.get("region")
    _, X_test, _ = prep_data_year(reg)
    filepath_1 = '../models/' + reg + '_year.pkl'
    loaded_model = load_pickle_model(filepath_1)
    preds = predict_year_future(loaded_model, X_test)
    plot_and_save_year(reg, preds)
    img_path = '/static/imgs/' + reg + '_year.png'

    sclr = MinMaxScaler()
    _, X_test2, _ = prep_data(reg, sclr)
    filepath_2 = '../models/' + reg + '_24.h5'
    hr_model = load_model(filepath_2)
    preds = predict_future_demand(hr_model, X_test2, sclr)
    hr24 = create_24hr_list(reg)
    return render_template('forecast.html', 
                            region=REGION, 
                            reg=reg, 
                            img_path=img_path,
                            hr_model=hr_model, 
                            preds=preds,
                            hr24=hr24)

@app.route('/data')
def data():
    return render_template('data.html')

@app.route('/about_me')
def about_me():
    return render_template('about_me.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=True, debug=False)
