import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from XGBoost_helper import predict_year_future
from flask import Flask, render_template, redirect, url_for, request
from save_and_predict_year import prep_data_year, load_pickle_model, plot_and_save_year


if __name__ == '__main__':
    reg = 'CAL'
    filepath1 = '../models' + reg + '_year.pkl'
    loaded_model = load_pickle_model(filepath1)
    print(loaded_model)