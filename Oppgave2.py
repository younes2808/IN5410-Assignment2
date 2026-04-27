from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#Constants for output directory and column names 
OUTPUT_DIR = Path("out/part_2")
TARGET_COLUMN = "POWER"
WIND_SPEED_COLUMN = "WS10"
WIND_DIRECTION_COLUMN = "wind_direction"

#RMSE function 
def calculate_rmse(actual_values, predicted_values):
    return np.sqrt(mean_squared_error(actual_values, predicted_values))

#Load the three datasets
def load_datasets():
    training_data = pd.read_csv("TrainData.csv", index_col="TIMESTAMP")
    forecast_data = pd.read_csv("WeatherForecastInput.csv", index_col="TIMESTAMP")
    solution_data = pd.read_csv("Solution.csv", index_col="TIMESTAMP")
    return training_data, forecast_data, solution_data

#Compute wind direction from U10 and V10
def add_wind_direction(dataframe):
    dataframe = dataframe.copy()
    dataframe[WIND_DIRECTION_COLUMN] = np.degrees(
        np.arctan2(dataframe["V10"], dataframe["U10"])
    )
    return dataframe

#Create feature sets for single and multiple regression
def build_feature_sets(training_data, forecast_data):
    single_feature = [WIND_SPEED_COLUMN]
    multiple_features = [WIND_SPEED_COLUMN, WIND_DIRECTION_COLUMN]

    return {
        "Linear Regression": {
            "training": training_data[single_feature],
            "forecast": forecast_data[single_feature],
        },
        "Multiple Linear Regression": {
            "training": training_data[multiple_features],
            "forecast": forecast_data[multiple_features],
        },
    }

#Train linear regression model and return predictions 
def train_and_predict(training_features, training_targets, forecast_features):
    model = LinearRegression()
    model.fit(training_features, training_targets)
    return model.predict(forecast_features)

#Run both models and collect predictions and RMSE 
def evaluate_models(feature_sets, training_targets, actual_values):
    model_results = []

    for model_name, features in feature_sets.items():
        predictions = train_and_predict(
            features["training"],
            training_targets,
            features["forecast"],
        )
        model_results.append(
            {
                "name": model_name,
                "predictions": predictions,
                "rmse": calculate_rmse(actual_values, predictions),
            }
        )

    return model_results

#Save forecast file for task 2
def save_forecast(predictions, timestamps):
    forecast_frame = pd.DataFrame(predictions, index=timestamps, columns=[TARGET_COLUMN])
    forecast_frame.to_csv(OUTPUT_DIR / "ForecastTemplate2.csv")

#Print RMSE comparison table
def print_rmse_table(model_results):
    rmse_table = pd.DataFrame(
        [{"Model": result["name"], "RMSE": result["rmse"]} for result in model_results]
    )
    print(rmse_table)

#Format axis for plotting
def format_axis(axis):
    axis.set_xlabel("Time (November 2013)")
    axis.set_ylabel("Wind Power")
    axis.legend()
    axis.xaxis.set_major_locator(mdates.DayLocator())
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m"))
    axis.tick_params(axis="x", rotation=45)

#Plot actual values and model predictions
def plot_results(solution_dates, forecast_dates, actual_values, model_results):
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.plot(solution_dates, actual_values, label="Real Data")

    for result in model_results:
        axis.plot(forecast_dates, result["predictions"], label=result["name"])

    format_axis(axis)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "part_2_plot.eps", transparent=False, bbox_inches="tight")
    figure.savefig(OUTPUT_DIR / "part_2_plot.png", dpi=200, bbox_inches="tight")
    plt.show()

#Run the full pipeline for task 2
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_data, forecast_data, solution_data = load_datasets()
    training_data = add_wind_direction(training_data)
    forecast_data = add_wind_direction(forecast_data)

    training_targets = training_data[TARGET_COLUMN]
    actual_values = solution_data[TARGET_COLUMN]
    feature_sets = build_feature_sets(training_data, forecast_data)
    model_results = evaluate_models(feature_sets, training_targets, actual_values)

    multiple_regression_result = next(
        result for result in model_results if result["name"] == "Multiple Linear Regression"
    )
    save_forecast(multiple_regression_result["predictions"], forecast_data.index)
    print_rmse_table(model_results)

    solution_dates = pd.to_datetime(solution_data.index, format="%Y%m%d %H:%M")
    forecast_dates = pd.to_datetime(forecast_data.index, format="%Y%m%d %H:%M")
    plot_results(
        solution_dates,
        forecast_dates,
        actual_values.to_numpy(),
        model_results,
    )


if __name__ == "__main__":
    main()
