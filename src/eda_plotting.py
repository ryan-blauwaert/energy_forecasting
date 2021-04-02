import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Demand_class import Demand

def plot_resampled_trend(ax, df, resample, label=None):
    mwh = pd.Series(df['Megawatthours'])
    resamp = mwh.resample(resample).mean()
    ax.plot(resamp, label=label)
    ax.legend(fontsize=12)
    ax.set_xlabel('Date', size=16)
    ax.set_ylabel('Megawatthours', size=16)

def plot_timeseries(ax, df, start_date, end_date, label=None):
    df = df[start_date:end_date]
    ax.plot(df, label=label)
    ax.plot(df.index, [np.mean(df)]*len(df), label='Mean Demand')
    ax.legend(fontsize=12)
    ax.set_xlabel('Date', size=16)
    ax.set_ylabel('Megawatthours', size=16)

def plot_predictions_vs_true(ax, y_test, predictions, date_index):
    ax.plot(date_index, y_test, label='Actual')
    ax.plot(preds_index, predictions, alpha=0.5, label='Predicted')
    ax.legend(fontsize=12)
    ax.set_xlabel('Date', size=16)
    ax.set_ylabel('Megawatthours', size=16)

    
if __name__ == '__main__':

    plt.style.use('seaborn-darkgrid')
    nat_dem = Demand()
    nat_dem.load_data('../data/demand_lower_48')
    df = nat_dem.dataframe
    df = df.set_index('Time', drop=True)
    print(df.head())
    nat_dem.create_time_featues()
    time_feat_df = nat_dem.time_features_df
    # print(time_feat_df.head())


    # fig, ax = plt.subplots(figsize=(12, 4))
    # plot_resampled_trend(ax, df, 'Q-JUL', 'Quarterly Mean Demand')
    # ax.set_title('Quarterly Mean MWH Demand', size=24)
    # ax.set_ylabel('Megawatthours', size=16)
    # ax.set_xlabel('Date', size=16)
    # plt.savefig('../images/eda/quarterly_means.png', dpi=500)
    # plt.show()

    # fig, ax = plt.subplots(figsize=(12, 4))
    # plot_timeseries(ax, df, '2017-07-01 00:00:00', '2017-07-31 23:00:00', 'Actual Demand')
    # ax.set_title('July 2017 Hourly Demand', size=24)
    # ax.set_xlabel('Date', size=16)
    # ax.set_ylabel('Megawatthours', size=16)
    # plt.savefig('../images/eda/july_2017_demand.png', dpi=500)
    # plt.show()

    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.violinplot(time_feat_df['Hour'], time_feat_df['Megawatthours'])
    ax.set_title('Demand by Hour', size=24)
    plt.yticks(fontsize=12)
    plt.xticks(fontsize=12)
    ax.set_ylabel('Megawatthours', size=16)
    ax.set_xlabel('Hour', size=16)
    plt.savefig('../images/eda/hourly_agg.png', dpi=500)
    plt.show()

    # fig, ax = plt.subplots(figsize=(12, 4))
    # plot_timeseries(ax, df, df.index[0], df.index[-1], label='Hourly Demand')
    # ax.set_title('Hourly Electricity Demand', size=24)
    # plt.savefig('../images/hourly_elec_demand.png', dpi=500)
    # plt.show()

