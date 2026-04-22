import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import root_mean_squared_error
from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import SimpleRNN, Dense #type: ignore


def build_sequences(data_array, lookback_steps):
    """Transform sequential data into input-output pairs for supervised learning."""
    # Initialize empty lists for input sequences and output values
    input_sequences = []
    output_values = []
    
    # Iterate through data starting after lookback window
    for idx in range(lookback_steps, len(data_array)):
        # Extract previous timesteps as input features
        input_sequences.append(data_array[idx - lookback_steps:idx])
        # Extract current timestep as target
        output_values.append(data_array[idx])
    
    return np.array(input_sequences), np.array(output_values)


def export_results(date_index, forecast_values, file_name, save_path):
    """Write forecast results to CSV format."""
    # Construct dataframe with dates and forecasts
    output_df = pd.DataFrame({
        'TIMESTAMP': pd.to_datetime(date_index),
        'FORECAST': forecast_values
    })
    
    output_df.to_csv(f"{save_path}/{file_name}", index=False)


def visualize_forecasts(date_index, true_values, forecast_a, forecast_b, label_a, label_b, save_path):
    """Generate comparison plot showing actual vs forecasted values."""
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot ground truth
    ax.plot(date_index, true_values, label='Ground Truth', color='navy', linewidth=2.5, alpha=0.8)
    # Plot first forecast
    ax.plot(date_index, forecast_a, label=f'{label_a} Forecast', color='crimson', linestyle='-.', linewidth=1.5)
    # Plot second forecast
    ax.plot(date_index, forecast_b, label=f'{label_b} Forecast', color='forestgreen', linestyle=':', linewidth=1.5)
    
    # Configure axes and labels
    ax.set_xlabel('Time Period', fontsize=11)
    ax.set_ylabel('Power Output', fontsize=11)
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    
    # Format date axis
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Save to disk
    plt.tight_layout()
    plt.savefig(f"{save_path}/comparison_{label_a}_{label_b}.png", dpi=200)
    plt.close(fig)


def run_forecasting_experiment():
    # Set experiment parameters
    LOOKBACK = 1
    SEED = 42
    RESULTS_PATH = "."
    START_DATE = '2013-11-01'
    
    # Read CSV data
    raw_data = pd.read_csv("TrainData.csv")
    power_series = raw_data['POWER'].values
    
    # Build training sequences
    train_inputs, train_targets = build_sequences(power_series, LOOKBACK)
    
    # Build testing sequences (using same data for demonstration)
    test_inputs, test_targets = build_sequences(power_series, LOOKBACK)
    
    # Create datetime index for test period
    test_dates = pd.date_range(start=START_DATE, periods=len(test_inputs), freq='H')
    
    # Store all forecasts and errors
    all_forecasts = {}
    error_metrics = {}
    
    # Initialize and train linear model
    linear_regressor = LinearRegression()
    linear_regressor.fit(train_inputs, train_targets)
    
    # Generate forecasts
    linear_forecast = linear_regressor.predict(test_inputs).flatten()
    all_forecasts['LinearReg'] = linear_forecast
    
    # Calculate error metric
    error_metrics['Linear Regression'] = root_mean_squared_error(test_targets, linear_forecast)
    
    # Export to file
    export_results(test_dates, linear_forecast, 'ForecastTemplate3-LR.csv', RESULTS_PATH)
    
    # Initialize and train SVR model
    svr_regressor = SVR()
    svr_regressor.fit(train_inputs, train_targets.ravel())
    
    # Generate forecasts
    svr_forecast = svr_regressor.predict(test_inputs).flatten()
    all_forecasts['SVR'] = svr_forecast
    
    # Calculate error metric
    error_metrics['Support Vector Regression'] = root_mean_squared_error(test_targets, svr_forecast)
    
    # Export to file
    export_results(test_dates, svr_forecast, 'ForecastTemplate3-SVR.csv', RESULTS_PATH)
    
    # Initialize and train MLP model
    mlp_regressor = MLPRegressor(
        hidden_layer_sizes=(30, 30),
        max_iter=1000,
        activation='relu',
        random_state=SEED
    )
    mlp_regressor.fit(train_inputs, train_targets.ravel())
    
    # Generate forecasts
    mlp_forecast = mlp_regressor.predict(test_inputs).flatten()
    all_forecasts['MLP'] = mlp_forecast
    
    # Calculate error metric
    error_metrics['Multilayer Perceptron'] = root_mean_squared_error(test_targets, mlp_forecast)
    
    # Export to file
    export_results(test_dates, mlp_forecast, 'ForecastTemplate3-ANN.csv', RESULTS_PATH)
    
    # Build RNN architecture
    rnn_regressor = Sequential([
        SimpleRNN(20, activation='relu', input_shape=(LOOKBACK, 1)),
        Dense(1)
    ])
    
    # Compile model with optimizer and loss
    rnn_regressor.compile(optimizer='adam', loss='mean_squared_error')
    
    # Reshape inputs for RNN (samples, timesteps, features)
    train_inputs_3d = train_inputs.reshape(-1, LOOKBACK, 1)
    test_inputs_3d = test_inputs.reshape(-1, LOOKBACK, 1)
    
    # Train the model
    rnn_regressor.fit(train_inputs_3d, train_targets, epochs=50, batch_size=30, verbose=0)
    
    # Generate forecasts
    rnn_forecast = rnn_regressor.predict(test_inputs_3d, batch_size=1, verbose=0).flatten()
    all_forecasts['RNN'] = rnn_forecast
    
    # Calculate error metric
    error_metrics['Recurrent Neural Network'] = root_mean_squared_error(test_targets, rnn_forecast)
    
    # Export to file
    export_results(test_dates, rnn_forecast, 'ForecastTemplate3-RNN.csv', RESULTS_PATH)
    
    # Print performance table
    print("\n" + "="*60)
    print("MODEL PERFORMANCE SUMMARY (RMSE)")
    print("="*60)
    performance_table = pd.DataFrame(list(error_metrics.items()), columns=['Model', 'RMSE'])
    print(performance_table.to_string(index=False))
    print("="*60 + "\n")
    
    # Create first comparison plot
    visualize_forecasts(
        test_dates, test_targets,
        all_forecasts['LinearReg'], all_forecasts['SVR'],
        'LinearReg', 'SVR', RESULTS_PATH
    )
    
    # Create second comparison plot
    visualize_forecasts(
        test_dates, test_targets,
        all_forecasts['MLP'], all_forecasts['RNN'],
        'MLP', 'RNN', RESULTS_PATH
    )
    


if __name__ == "__main__":
    run_forecasting_experiment()