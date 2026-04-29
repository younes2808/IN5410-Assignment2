
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

#Constants
RANDOM_SEED = 10
FEATURE_COLUMN = "WS10"
TARGET_COLUMN = "POWER"

#RMSE calculation: measures how well predictions match true values
def calculate_rmse(actual_values, predicted_values):
    return np.sqrt(mean_squared_error(actual_values, predicted_values))

#Load training data, test input, and solution data from csv files
def load_data():
    train_data = pd.read_csv("TrainData.csv", index_col="TIMESTAMP")
    test_features = pd.read_csv("WeatherForecastInput.csv", index_col="TIMESTAMP")
    test_targets = pd.read_csv("Solution.csv", index_col="TIMESTAMP")
    return train_data, test_features, test_targets

#Select input feature(s) and target variable
def prepare_features_and_targets(train_data, test_features):
    training_features = train_data[[FEATURE_COLUMN]]
    training_targets = train_data[TARGET_COLUMN]
    forecast_features = test_features[[FEATURE_COLUMN]]
    return training_features, training_targets, forecast_features

#Save model predictions to CSV with correct timestamps
def save_predictions(predictions, timestamps, output_path):
    prediction_frame = pd.DataFrame(
        predictions,
        index=timestamps,
        columns=[TARGET_COLUMN],
    )
    prediction_frame.to_csv(output_path)

#Create all models for comparison
def create_models(number_of_training_samples):
    neighbor_count = max(1, int(np.sqrt(number_of_training_samples)))
    return {
        "Linear Regression": LinearRegression(),
        "KNN": KNeighborsRegressor(n_neighbors=neighbor_count),
        "SVR": SVR(),
        "Neural Network": MLPRegressor(
            hidden_layer_sizes=(100, 100),
            max_iter=10000,
            activation="relu",
            random_state=RANDOM_SEED,
        ),
    }

#Train model, generate predictions, and compute RMSE
def train_and_evaluate_model(
    model_name,
    model,
    training_features,
    training_targets,
    test_features,
    test_targets,
):
    model.fit(training_features, training_targets)
    predictions = model.predict(test_features)
    rmse_value = calculate_rmse(test_targets, predictions)
    return {
        "name": model_name,
        "predictions": predictions,
        "rmse": rmse_value,
    }

#Save all forecast files for the models
def save_all_predictions(model_results, timestamps):
    output_filenames = {
        "Linear Regression": "ForecastTemplate1-LR.csv",
        "KNN": "ForecastTemplate1-kNN.csv",
        "SVR": "ForecastTemplate1-SVR.csv",
        "Neural Network": "ForecastTemplate1-NN.csv",
    }

    for result in model_results:
        output_path = output_filenames[result["name"]]
        save_predictions(result["predictions"], timestamps, output_path)

#Print a table with RMSE results for all models
def print_rmse_table(model_results):
    rmse_table = pd.DataFrame(
        [{"Model": result["name"], "RMSE": result["rmse"]} for result in model_results]
    )
    print(rmse_table)

#Plot actual vs predicted values
def format_prediction_axis(axis, dates, actual_values, predicted_values, title):
    axis.plot(dates, actual_values, label="Real Data")
    axis.plot(dates, predicted_values, label=title)
    axis.set_title(title)
    axis.set_xlabel("Time (November 2013)")
    axis.set_ylabel("Wind Power")
    axis.legend()
    axis.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m"))
    axis.tick_params(axis="x", rotation=45)

#Combine all plots into one figure
def plot_model_predictions(test_dates, actual_values, model_results):
    figure, axes = plt.subplots(2, 2, figsize=(10, 6))

    for axis, result in zip(axes.flatten(), model_results):
        format_prediction_axis(
            axis,
            test_dates,
            actual_values,
            result["predictions"],
            result["name"],
        )

    figure.tight_layout()
    figure.savefig("part_1_plot.eps", transparent=False, bbox_inches="tight")
    figure.savefig("part_1_plot.png", dpi=200, bbox_inches="tight")
    plt.show()

#Run the full pipeline: loading data, training models, and plotting
def main():
    train_data, test_data, solution_data = load_data()
    training_features, training_targets, forecast_features = prepare_features_and_targets(
        train_data,
        test_data,
    )
    test_targets = solution_data[TARGET_COLUMN]

    models = create_models(len(training_features))
    model_results = []

    for model_name, model in models.items():
        result = train_and_evaluate_model(
            model_name,
            model,
            training_features,
            training_targets,
            forecast_features,
            test_targets,
        )
        model_results.append(result)

    save_all_predictions(model_results, forecast_features.index)
    print_rmse_table(model_results)

    test_dates = pd.to_datetime(forecast_features.index, format="%Y%m%d %H:%M")
    actual_values = test_targets.to_numpy()
    plot_model_predictions(test_dates, actual_values, model_results)


if __name__ == "__main__":
    main()
