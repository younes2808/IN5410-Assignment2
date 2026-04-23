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



def visualize_forecasts(date_index, true_values, forecast_a, forecast_b,
                        label_a, label_b, save_path):
    """Generate publication-quality comparison plot."""

    # scientific theme
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(14, 7), dpi=150)

    colors = {
        "truth": "#fa8b04",
        "a": "#d62728",         
        "b": "#1f77b4"          
    }

    ax.plot(date_index, true_values,
            label="Ground Truth",
            color=colors["truth"],
            linewidth=2.5,
            alpha=0.95)

    ax.plot(date_index, forecast_a,
            label=f"{label_a} Forecast",
            color=colors["a"],
            linestyle="--",
            linewidth=2.0,
            alpha=0.9)

    ax.plot(date_index, forecast_b,
            label=f"{label_b} Forecast",
            color=colors["b"],
            linestyle="-.",
            linewidth=2.0,
            alpha=0.9)

    # Labels
    ax.set_xlabel("Date (November 2013)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Power Output (MW)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Wind Power Forecasting Comparison: {label_a} vs {label_b}",
        fontsize=14,
        fontweight="bold",
        pad=12
    )

    # Legend (clean, no frame clutter)
    ax.legend(frameon=True,
              fancybox=False,
              edgecolor="black",
              fontsize=10,
              loc="upper right")

    # Grid refinement (lighter, scientific style)
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.4)

    # Remove unnecessary spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Tight y-limits with margin
    y_min = min(true_values.min(), forecast_a.min(), forecast_b.min())
    y_max = max(true_values.max(), forecast_a.max(), forecast_b.max())
    margin = (y_max - y_min) * 0.08
    ax.set_ylim(y_min - margin, y_max + margin)

    # Date formatting (clean scientific time axis)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()

    plt.savefig(
        f"{save_path}/comparison_{label_a}_{label_b}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)


def run_forecasting_experiment():
    # Set experiment parameters
    LOOKBACK = 24  # Use past 24 hours to predict next hour
    SEED = 10
    RESULTS_PATH = "."
    
    # Read training data (should NOT include November 2013)
    raw_data = pd.read_csv("TrainData.csv", parse_dates=['TIMESTAMP'])
    
    # Filter to get training data (before November 2013)
    train_data = raw_data[raw_data['TIMESTAMP'] < '2013-11-01'].copy()
    train_power = train_data['POWER'].values
    
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
    print("="*60 + "\n")
    performance_table = pd.DataFrame(list(error_metrics.items()), columns=['Model', 'RMSE'])
    print(performance_table.to_string(index=False))
    print("="*60 + "\n")
    
    # first plot (LR vs SVR)
    visualize_forecasts(
        test_dates, test_targets,
        all_forecasts['LinearReg'], all_forecasts['SVR'],
        'LinearReg', 'SVR', RESULTS_PATH
    )
    
    # second plot (ANN vs RNN)
    visualize_forecasts(
        test_dates, test_targets,
        all_forecasts['MLP'], all_forecasts['RNN'],
        'ANN', 'RNN', RESULTS_PATH
    )
    
    print("Forecasting complete! Check output files and plots.")


if __name__ == "__main__":
    run_forecasting_experiment()