import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import root_mean_squared_error
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import SimpleRNN, Dense # type: ignore


def build_sequences(data_array, lookback_steps):
    n = len(data_array) - lookback_steps
    X = np.lib.stride_tricks.sliding_window_view(data_array, lookback_steps)
    return X[:n], data_array[lookback_steps:]


def export_results(date_index, forecast_values, file_name, save_path):
    """Write forecast results to CSV format."""
    output_df = pd.DataFrame({
        'TIMESTAMP': pd.to_datetime(date_index),
        'FORECAST': forecast_values
    })
    
    output_df.to_csv(f"{save_path}/{file_name}", index=False)


def run_forecasting_experiment():
    # Set experiment parameters
    LOOKBACK = 24  # Use past 24 hours to predict next hour
    SEED = 10
    RESULTS_PATH = "."
    
    # Read training daa
    raw_data = pd.read_csv("TrainData.csv", parse_dates=['TIMESTAMP'])
    
    # get training data
    train_data = raw_data['TIMESTAMP'].values
    train_power = raw_data['POWER'].values
    
    # Load Solution.csv for November 2013 ground truth
    solution_data = pd.read_csv("Solution.csv", parse_dates=['TIMESTAMP'])
    
    # Filter for November 2013 only
    solution_nov = solution_data[
        (solution_data['TIMESTAMP'] >= '2013-11-01') & 
        (solution_data['TIMESTAMP'] < '2013-12-01')
    ].copy()
    
    # Get the power values and timestamps for November 2013
    test_targets = solution_nov['POWER'].values
    test_dates = solution_nov['TIMESTAMP'].values
    
    # Build training sequences from historical data
    train_inputs, train_outputs = build_sequences(train_power, LOOKBACK)
    
    # Prepare test inputs: use last LOOKBACK hours before Nov + data from Nov itself
    # We need to create sequences for prediction
    # Concatenate end of training data with November data
    extended_data = np.concatenate([train_power[-LOOKBACK:], test_targets])
    
    # Build test sequences
    test_inputs, _ = build_sequences(extended_data, LOOKBACK)
    # Only keep the sequences that predict November (first len(test_targets) sequences)
    test_inputs = test_inputs[:len(test_targets)]
    
    # Store all forecasts and errors
    all_forecasts = {}
    error_metrics = {}
    
    print("Task 3:")
    
    # ===== LINEAR REGRESSION =====
    print("Training Linear Regression...")
    linear_regressor = LinearRegression()
    linear_regressor.fit(train_inputs, train_outputs)
    
    linear_forecast = linear_regressor.predict(test_inputs).flatten()
    all_forecasts['LinearReg'] = linear_forecast
    error_metrics['Linear Regression'] = root_mean_squared_error(test_targets, linear_forecast)
    
    export_results(test_dates, linear_forecast, 'ForecastTemplate3-LR.csv', RESULTS_PATH)
    
    # ===== SVR =====
    print("Training SVR...")
    svr_regressor = SVR(kernel='rbf', C=10)  # reducing complexity making it much faster
    svr_regressor.fit(train_inputs, train_outputs.ravel())
    
    svr_forecast = svr_regressor.predict(test_inputs).flatten()
    all_forecasts['SVR'] = svr_forecast
    error_metrics['Support Vector Regression'] = root_mean_squared_error(test_targets, svr_forecast)
    
    export_results(test_dates, svr_forecast, 'ForecastTemplate3-SVR.csv', RESULTS_PATH)
    
    # ===== ANN =====
    print("Training ANN ...")
    mlp_regressor = MLPRegressor(
        hidden_layer_sizes=(50, 30),
        max_iter=200,
        activation='relu',
        random_state=SEED,
        early_stopping=True
    )
    mlp_regressor.fit(train_inputs, train_outputs.ravel())
    
    mlp_forecast = mlp_regressor.predict(test_inputs).flatten()
    all_forecasts['MLP'] = mlp_forecast
    error_metrics['Artificial Neural Network'] = root_mean_squared_error(test_targets, mlp_forecast)
    
    export_results(test_dates, mlp_forecast, 'ForecastTemplate3-ANN.csv', RESULTS_PATH)
    
    # ===== RNN =====
    print("Training RNN...")
    rnn_regressor = Sequential([
        SimpleRNN(20, activation='relu', input_shape=(LOOKBACK, 1)),
        Dense(1)
    ])
    
    rnn_regressor.compile(optimizer='adam', loss='mean_squared_error')
    
    # Reshape inputs for RNN (samples, timesteps, features)
    train_inputs_3d = train_inputs.reshape(-1, LOOKBACK, 1)
    test_inputs_3d = test_inputs.reshape(-1, LOOKBACK, 1)
    
    rnn_regressor.fit(train_inputs_3d, train_outputs, epochs=10, batch_size=32, verbose=0)
    
    rnn_forecast = rnn_regressor.predict(test_inputs_3d, verbose=0).flatten()
    all_forecasts['RNN'] = rnn_forecast
    error_metrics['Recurrent Neural Network'] = root_mean_squared_error(test_targets, rnn_forecast)
    
    export_results(test_dates, rnn_forecast, 'ForecastTemplate3-RNN.csv', RESULTS_PATH)
    
    # Print performance table
    print(" RMSE Results: ")
    performance_table = pd.DataFrame(list(error_metrics.items()), columns=['Model', 'RMSE'])
    print(performance_table.to_string(index=False))

if __name__ == "__main__":
    run_forecasting_experiment()