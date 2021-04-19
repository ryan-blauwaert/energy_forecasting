import numpy as np
import pandas as pd
import json
import requests

class Demand():
    """
    Class which houses electricity demand data
    and its associated methods and attributes
    """
    def __init__(self, target='Megawatthours'):
        """Initializes an instance of the Demand 
        class with a target variable

        Args:
            target (str, optional): Target variable in the dataset
            for future modeling. Defaults to 'Megawatthours'.
        """
        self.target = target
        self.dataframe = None
        self.time_features_df = None
        self.trig_df = None
        
        
    def load_data(self, region):
        """Loads electricity demand data into self.dataframe
        and performs some preliminary data cleaning operations.

        Args:
            region (str): relative path of the data file to be 
            loaded into Demand object.
        """
        
        url_plus_key = 'http://api.eia.gov/series/?api_key=bc8c4348f7c30988e817d0b1b54441c5&series_id=EBA.'
        url_tail = '-ALL.D.HL'
        url = url_plus_key + region +url_tail
        r = requests.get(url)
        pull = r.json()
        hourly_data = pull['series'][0]['data']
        df = pd.DataFrame(hourly_data, columns=['Time', 'Megawatthours'])
        df['Time'] = df['Time'].apply(lambda x: x[:-3])
        df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
        df = df.loc[::-1]
        df = df[1:]
        df.reset_index(inplace=True, drop=True)
        self.dataframe = df

    def create_time_features(self):
        """Creates several time features from self.dataframe and
        stores the resulting dataframe in self.time_features_df
        """
        df = self.dataframe.copy()
        df['Year'] = df['Time'].dt.year
        df['Month'] = df['Time'].dt.month
        df['Hour'] = df['Time'].dt.hour
        df['Day_of_week'] = df['Time'].dt.dayofweek
        df['Day_of_month'] = df['Time'].dt.day
        df['Day_of_year'] = df['Time'].dt.dayofyear
        df.set_index('Time', inplace=True, drop=True)
        df = df.astype('int')
        df.reset_index(inplace=True)
        self.time_features_df = df

    def create_trig_df(self):
        """Creates trigonometric sin and cos time features to 
        capture cyclical patterns in data. Stores resulting 
        dataframe in df.trig_df
        """
        df = self.time_features_df.copy()
        df = df.drop(columns=['Day_of_week', 
                                'Day_of_month', 
                                'Day_of_year'])
        df['sin_day'] = [np.sin(x * (2*np.pi/24)) for x in df['Hour']]
        df['cos_day'] = [np.cos(x * (2*np.pi/24)) for x in df['Hour']]
        df['Timestamp'] = [x.timestamp() for x in df['Time']]
        s_in_year = 365.25*24*60*60
        df["sin_month"] = [np.sin((x) * (2 * np.pi / s_in_year)) for x in df["Timestamp"]]
        df["cos_month"] = [np.cos((x) * (2 * np.pi / s_in_year)) for x in df["Timestamp"]]
        df = df[['Time', 'Megawatthours', 'sin_day', 'cos_day', 'sin_month', 'cos_month']]
        self.trig_df = df

    def create_lag_variables(self, df, n_lag, n_ahead=0):
        """Creates lag variables from 'Megawatthours' column of 
        input dataframe. 

        Args:
            df (DataFrame): DataFrame in which to create lag variables.
            n_lag (int): Number of lag variables to be created in each row.
            n_ahead (int, optional): Number of hours between last lag
            variable and the target variable. Increase to predict further
            into future. Value of 0 will predict the next hour. Defaults to 0.

        Returns:
            [DataFrame]: DataFrame with lag variables included as features
        """
        lag_rows = []
        for i in range(n_lag+n_ahead, len(df)):
            lag_rows.append(df.loc[i-(n_lag+n_ahead):i-n_ahead-1, 'Megawatthours'].values)
        lag_df = pd.DataFrame(lag_rows, index=df.index[n_lag+n_ahead:])
        df = pd.concat([df.loc[n_lag+n_ahead:], lag_df], axis=1)
        return df

    def scale_split(self, df, train_test_idx, scaler=None):
        """Scales and splits a dataframe into X_train, X_test, y_train, and
        y_test arrays for machine learning. Assumes target variable is in 
        first column. 

        Args:
            df (DataFrame): DataFrame from which to extract X_train, 
            X_test, y_train, and y_test objects
            train_test_idx (int or str): Index on which to split train
            from test sets.
            scaler (scaler object, optional): Instance of scaler object.
            e.g. MinMaxScaler(). Defaults to None.

        Returns:
            [arr, arr, arr, arr]: Four arrays suited for input 
            into supervised machine learning models.
        """
        df = df.copy()
        df = df.set_index('Time', drop=True)
        if isinstance(train_test_idx, str):
            idx = df.index.get_loc(train_test_idx)
        else:
            idx = train_test_idx
        train, test = df[:idx], df[idx:]    
        if scaler:
            sclr = scaler
            train = sclr.fit_transform(train)
            test = sclr.transform(test)
        else:
            train, test = train.values, test.values
        X_train = train[:, 1:]
        y_train = train[:, 0]
        X_test = test[:, 1:]
        y_test = test[:, 0]
        return X_train, X_test, y_train, y_test
            
    def reshape_for_rnn(self, X_train, X_test, y_train, y_test):
        """Adds an additional dimension to X_train, X_test, y_train, y_test
        arrays to adapt them to recurrent neural network models.

        Args:
            X_train (arr): 2-dimensional feature matrix 
            X_test (arr)): 2-dimensional feature matrix
            y_train (arr): 1-dimensional target matrix
            y_test (arr): 1-dimensional target matrix

        Returns:
            [arr, arr, arr, arr]: Four arrays reshaped for input 
            neural network model
        """
        X_train = np.expand_dims(X_train, axis=2)
        X_test = np.expand_dims(X_test, axis=2)
        y_train = np.expand_dims(y_train, axis=1)
        y_test = np.expand_dims(y_test, axis=1)
        return X_train, X_test, y_train, y_test


    
if __name__ == '__main__':

    ny = Demand()
    ny.load_data('NY')
    print(ny.dataframe.head())
    ny.create_time_features()
    ny_time_df = ny.time_features_df
    ny_time
   