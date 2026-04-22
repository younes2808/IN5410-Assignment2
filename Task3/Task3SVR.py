# Importing packages and classes
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
# RMSE
from sklearn.metrics import mean_squared_error
# For manipulating CSVs
import pandas as pd
# Target file
target_file = 'TrainData.csv'
df = pd.read_csv(target_file)
# Dropping columns U10, WS10, U100, V100, WS100 per Task 3's requests
df.drop(columns=['U10', 'WS10', 'U100', 'V100', 'WS100', 'V10'], inplace=True)
# Converting timestamp and sorting by it
df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
df = df.sort_values('TIMESTAMP')
df['time_index'] = np.arange(len(df))
# Separating into target and feature
X = df[['time_index']]
y = df['POWER']
# Scaling features for SVR
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()
# Training Model and fitting
model = SVR(kernel='rbf')
model.fit(X_scaled, y_scaled)
# Forecasting November 2013
future_steps = 720  # 720 Hours in November
last_index = df['time_index'].iloc[-1]
future_index = np.arange(last_index + 1, last_index + 1 + future_steps)
future_df = pd.DataFrame(future_index, columns=['time_index'])
future_df_scaled = scaler_X.transform(future_df)
future_pred_scaled = model.predict(future_df_scaled)
future_pred = scaler_y.inverse_transform(future_pred_scaled.reshape(-1, 1)).ravel()
# Saving to CSV
nov_timestamps = pd.date_range(
    start="2013-11-01 01:00:00",
    periods=future_steps,
    freq="h"
)
timestamp_str = nov_timestamps.strftime("%Y%m%d %H:%M")
# Removing 0 before to match timestamp format in solution
timestamp_str = timestamp_str.str.replace(" 0", " ")
forecast_df = pd.DataFrame({
    'TIMESTAMP': timestamp_str,
    'FORECAST': future_pred
})
forecast_df.to_csv("ForecastTemplate3-SVR.csv", index=False)
# Comparing with Solution.csv
solution_df = pd.read_csv("Solution.csv")
merged_df = pd.merge(forecast_df, solution_df, on='TIMESTAMP')
# Calculating RMSE
rmse = np.sqrt(mean_squared_error(merged_df['POWER'], merged_df['FORECAST']))
print("RMSE:", rmse)