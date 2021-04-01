import numpy as np
import pandas as pd


class Demand():

    def __init__(self, target='Megawatthours'):

        self.target = target
        self.dataframe = None
        self.time_features_df = None
        self.trig_df = None
        
        
    def load_data(self, filepath):
        
        df = pd.read_csv(filepath)
        df['Time'] = df['Time'].apply(lambda x: x[:-6])
        df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
        df = df.loc[::-1]
        df = df[1:]
        df.reset_index(inplace=True, drop=True)
        self.dataframe = df

    def create_time_featues(self):

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
        
        lag_rows = []
        for i in range(n_lag+n_ahead, len(df)):
            lag_rows.append(df.loc[i-(n_lag+n_ahead):i-n_ahead-1, 'Megawatthours'].values)
        lag_df = pd.DataFrame(lag_rows, index=df.index[n_lag+n_ahead:])
        df = pd.concat([df.loc[n_lag+n_ahead:], lag_df], axis=1)
        return df

    def scale_split(self, df, train_test_idx=None, scaler=None):
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
        X_train = np.expand_dims(X_train, axis=2)
        X_test = np.expand_dims(X_test, axis=2)
        y_train = np.expand_dims(y_train, axis=1)
        y_test = np.expand_dims(y_test, axis=1)
        return X_train, X_test, y_train, y_test


    
if __name__ == '__main__':

    path = '../data/demand_lower_48'

    nat_dem = Demand()
    nat_dem.load_data(path)
    # df = nat_dem.dataframe
    nat_dem.create_time_featues()
    nat_dem.create_trig_df()
    print(nat_dem.dataframe)
    print(nat_dem.time_features_df)
    print(nat_dem.trig_df)
    # print(nat_dem.time_features_df)
    
   