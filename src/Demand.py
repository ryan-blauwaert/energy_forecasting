import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class Demand():

    def __init__(self, target='Megawatthours'):

        self.dataframe = None
        self.target = target
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_and_clean_data(self, filepath):
        df = pd.read_csv(filepath)
        df['Time'] = df['Time'].apply(lambda x: x[:-6])
        df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
        df = df.loc[::-1]
        df = df[1:]
        df['Year'] = df['Time'].dt.year
        df['Month'] = df['Time'].dt.month
        df['Hour'] = df['Time'].dt.hour
        df['Day_of_week'] = df['Time'].dt.dayofweek
        df['Day_of_month'] = df['Time'].dt.day
        df['Day_of_year'] = df['Time'].dt.dayofyear
        # df['Week_of_year'] = df['Time'].dt.isocalendar().week
        df.set_index('Time', inplace=True, drop=True)
        df = df.astype('int')
        self.dataframe = df

    def split(self, split_date=None):
        X = self.dataframe.copy()
        y = X.pop(self.target)
        if split_date:
            self.X_train, self.X_test = X.loc[:split_date], X.loc[split_date:]
            self.y_train, self.y_test = y.loc[:split_date], y.loc[split_date:]
        else:
            self.X_train, self.X_test, self.y_train, self._test = train_test_split(X, y)
    
    


if __name__ == '__main__':

    path = '../data/demand_lower_48'

    national_demand = Demand()
    national_demand.load_and_clean_data(path)
    national_demand.split(split_date='2020-07-01 00:00:00')
    print(len(national_demand.X_test))