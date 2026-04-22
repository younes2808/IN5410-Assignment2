# Importing packages and classes
import numpy as np
import pandas as pd

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

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
# Feature Scaling
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()
# Training Model and fitting
model = MLPRegressor(
    hidden_layer_sizes=(64, 64),
    activation='relu',
    max_iter=500,
    random_state=42
)
model.fit(X_scaled, y_scaled)
# Forecasting November 2013
future_steps = 720
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
timestamp_str = timestamp_str.str.replace(" 0", " ")
forecast_df = pd.DataFrame({
    'TIMESTAMP': timestamp_str,
    'FORECAST': future_pred
})
forecast_df.to_csv("ForecastTemplate3-ANN.csv", index=False)
# Comparing with Solution.csv
solution_df = pd.read_csv("Solution.csv")
merged_df = pd.merge(forecast_df, solution_df, on='TIMESTAMP')
rmse = np.sqrt(mean_squared_error(merged_df['POWER'], merged_df['FORECAST']))
print("RMSE:", rmse)