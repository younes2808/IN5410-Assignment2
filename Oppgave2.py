import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def load_data():
    train_data = pd.read_csv("TrainData.csv", index_col="TIMESTAMP")
    weather_forecast_input = pd.read_csv("WeatherForecastInput.csv", index_col="TIMESTAMP")
    solution = pd.read_csv("Solution.csv", index_col="TIMESTAMP")
    return train_data, weather_forecast_input, solution


def add_wind_direction(dataframe):
    dataframe["wind_direction"] = (
        np.arctan2(dataframe["V10"], dataframe["U10"]) * (180 / np.pi)
    )
    return dataframe


def save_predictions(predictions, index, filename):
    pd.DataFrame(predictions, index=index, columns=["POWER"]).to_csv(filename)


def part_2_style_assignment():
    output_dir = "out/part_2"
    os.makedirs(output_dir, exist_ok=True)

    train_data, weather_forecast_input, solution = load_data()

    train_data = add_wind_direction(train_data)
    weather_forecast_input = add_wind_direction(weather_forecast_input)

    X_train = train_data[["WS10", "wind_direction"]]
    y_train = train_data[["POWER"]]

    mlr_model = LinearRegression()
    mlr_model.fit(X_train, y_train)

    X_test = weather_forecast_input[["WS10", "wind_direction"]]
    wind_power_predictions = mlr_model.predict(X_test)
    save_predictions(
        wind_power_predictions,
        X_test.index,
        os.path.join(output_dir, "ForecastTemplate2.csv"),
    )

    true_wind_power = solution[["POWER"]]
    rmse_mlr = root_mean_squared_error(true_wind_power, wind_power_predictions)
    print("RMSE:", rmse_mlr)

    X_train_ws = train_data[["WS10"]]
    X_test_ws = weather_forecast_input[["WS10"]]

    lr_model_ws = LinearRegression()
    lr_model_ws.fit(X_train_ws, y_train)
    wind_power_predictions_ws = lr_model_ws.predict(X_test_ws)

    rmse_lr = root_mean_squared_error(true_wind_power, wind_power_predictions_ws)
    print("RMSE (Linear Regression with Wind Speed only):", rmse_lr)

    solution_dates = pd.to_datetime(solution.index, format="%Y%m%d %H:%M")
    forecast_dates = pd.to_datetime(weather_forecast_input.index, format="%Y%m%d %H:%M")

    plt.figure(figsize=(10, 4))
    plt.plot(solution_dates, true_wind_power, label="Real Data")
    plt.plot(forecast_dates, wind_power_predictions_ws, label="Linear Regression")
    plt.plot(forecast_dates, wind_power_predictions, label="Multiple Linear Regression")
    plt.xlabel("Date (dd-mm)")
    plt.ylabel("Wind Power")
    plt.legend()
    plt.tight_layout()
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%d-%m"))
    plt.xticks(rotation=45)
    plt.savefig(
        os.path.join(output_dir, "part_2_plot.eps"),
        transparent=False,
        bbox_inches="tight",
    )
    plt.show()

    rmse_table = pd.DataFrame(
        {
            "Model": ["Linear Regression", "Multiple Linear Regression"],
            "RMSE": [rmse_lr, rmse_mlr],
        }
    )
    print(rmse_table)


if __name__ == "__main__":
    part_2_style_assignment()
