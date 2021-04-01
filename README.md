# Forecasting Electricity Demand
### Ryan Blauwaert

![header](./images/power-transmission.jpg)

| Time                | Megawatthours |
|---------------------|---------------|
| 2015-07-01 02:00:00 | 335153        |
| 2015-07-01 03:00:00 | 333837        |
| 2015-07-01 04:00:00 | 398386        |
| 2015-07-01 05:00:00 | 388954        |
| 2015-07-01 06:00:00 | 392487        |


| Time                | Megawatthours | Year | Month | Hour | Day of Week | Day of Month | Day of Year |
|---------------------|---------------|------|-------|------|-------------|--------------|-------------|
| 2015-07-01 02:00:00 | 335153        | 2015 | 7     | 2    | 2           | 1            | 182         |
| 2015-07-01 03:00:00 | 333837        | 2015 | 7     | 3    | 2           | 1            | 182         |
| 2015-07-01 04:00:00 | 398386        | 2015 | 7     | 4    | 2           | 1            | 182         |
| 2015-07-01 05:00:00 | 388954        | 2015 | 7     | 5    | 2           | 1            | 182         |
| 2015-07-01 06:00:00 | 392487        | 2015 | 7     | 6    | 2           | 1            | 182         |

![hourly elec demand](./images/hourly_elec_demand.png)
![quarterly elec demand](./images/eda/quarterly_means.png)
![weekly elec demand](./images/eda/weekly_means.png)
![july 2017 hourly elec demand](./images/eda/july_2017_demand.png)
![Polynomial trendline](./images/eda/poly_trend.png)
![monthly agg](./images/eda/monthly_agg.png)
![daily agg](./images/eda/daily_agg.png)
![hourly agg](./images/eda/hourly_agg.png)


![xgboost](./images/xgb_best_model.png)

| Time                | n-23   | n-22   | ... | n-2    | n-1    | Megawatthours |
|---------------------|--------|--------|-----|--------|--------|---------------|
| 2015-07-01 02:00:00 | 335153 | 333837 | ... | 485722 | 453284 | 429199        |
| 2015-07-01 03:00:00 | 333837 | 398386 | ... | 453284 | 429199 | 407007        |
| 2015-07-01 04:00:00 | 398386 | 388954 | ... | 429199 | 407007 | 395194        |
| 2015-07-01 05:00:00 | 388954 | 392487 | ... | 407007 | 395194 | 387654        |
| 2015-07-01 06:00:00 | 392487 | 404647 | ... | 395194 | 387654 | 390157        |

![RNN next hour](./images/next_hour_rnn.png)
![RNN day ahead](./images/day_ahead_rnn.png)
![RNN year ahead](./images/year_ahead_rnn.png)
