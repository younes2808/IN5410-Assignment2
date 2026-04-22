# Importing packages and classes
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore

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
X = df[['time_index']].values
y = df['POWER'].values.reshape(-1, 1)

# Scaling (important for RNN)
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Reshaping for LSTM [samples, timesteps, features]
X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, 1))

# Training Model and fitting
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(1, 1)))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(X_scaled, y_scaled, epochs=10, batch_size=32, verbose=0)

# Forecasting November 2013
future_steps = 720
last_index = df['time_index'].iloc[-1]

future_index = np.arange(last_index + 1, last_index + 1 + future_steps)
future_scaled = scaler_X.transform(future_index.reshape(-1, 1))
future_scaled = future_scaled.reshape((future_scaled.shape[0], 1, 1))

future_pred_scaled = model.predict(future_scaled)
future_pred = scaler_y.inverse_transform(future_pred_scaled)

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
    'FORECAST': future_pred.flatten()
})

forecast_df.to_csv("ForecastTemplate3-RNN.csv", index=False)

# Comparing with Solution.csv
solution_df = pd.read_csv("Solution.csv")
merged_df = pd.merge(forecast_df, solution_df, on='TIMESTAMP')
rmse = np.sqrt(mean_squared_error(merged_df['POWER'], merged_df['FORECAST']))
print("RMSE:", rmse)