import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def load_data():
    train_data = pd.read_csv("TrainData.csv", index_col="TIMESTAMP")
    test_X = pd.read_csv("WeatherForecastInput.csv", index_col="TIMESTAMP")
    test_Y = pd.read_csv("Solution.csv", index_col="TIMESTAMP")
    return train_data, test_X, test_Y


def save_predictions(predictions, index, filename):
    pd.DataFrame(predictions, index=index, columns=["POWER"]).to_csv(filename)


def part_1_style_assignment():
    rnd_seed = 42
    output_dir = "out/part_1"
    os.makedirs(output_dir, exist_ok=True)

    train_data, test_X, test_Y = load_data()

    train_X = train_data[["WS10"]]
    train_Y = train_data[["POWER"]]
    test_X = test_X[["WS10"]]

    lin_reg_model = LinearRegression()
    lin_reg_model.fit(train_X, train_Y)
    lin_reg_predictions = lin_reg_model.predict(test_X)
    rmse_lin_reg = root_mean_squared_error(test_Y, lin_reg_predictions)
    save_predictions(
        lin_reg_predictions,
        test_X.index,
        os.path.join(output_dir, "ForecastTemplate1-LR.csv"),
    )

    knn_model = KNeighborsRegressor(n_neighbors=int(np.sqrt(len(train_X))))
    knn_model.fit(train_X, train_Y)
    knn_predictions = knn_model.predict(test_X)
    rmse_knn = root_mean_squared_error(test_Y, knn_predictions)
    save_predictions(
        knn_predictions,
        test_X.index,
        os.path.join(output_dir, "ForecastTemplate1-kNN.csv"),
    )

    svr_model = SVR()
    svr_model.fit(train_X, train_Y.values.ravel())
    svr_predictions = svr_model.predict(test_X)
    rmse_svr = root_mean_squared_error(test_Y, svr_predictions)
    save_predictions(
        svr_predictions,
        test_X.index,
        os.path.join(output_dir, "ForecastTemplate1-SVR.csv"),
    )

    neural_network = MLPRegressor(
        hidden_layer_sizes=(100, 100),
        max_iter=10000,
        activation="relu",
        random_state=rnd_seed,
    )
    neural_network.fit(train_X, train_Y.values.ravel())
    nn_predictions = neural_network.predict(test_X)
    rmse_nn = root_mean_squared_error(test_Y, nn_predictions)
    save_predictions(
        nn_predictions,
        test_X.index,
        os.path.join(output_dir, "ForecastTemplate1-NN.csv"),
    )

    table_data = {
        "Model": ["Linear Regression", "KNN", "SVR", "Neural Network"],
        "RMSE": [rmse_lin_reg, rmse_knn, rmse_svr, rmse_nn],
    }
    print(pd.DataFrame(table_data))

    test_dates = pd.to_datetime(test_X.index, format="%Y%m%d %H:%M")
    true_values = test_Y["POWER"].to_numpy()

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 2, 1)
    plt.plot(test_dates, true_values, label="Real Data")
    plt.plot(test_dates, lin_reg_predictions, label="Linear Regression")
    plt.xlabel("Time (November 2013)")
    plt.ylabel("Wind Power")
    plt.title("Linear Regression")
    plt.legend()
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%d-%m"))
    plt.xticks(rotation=45)

    plt.subplot(2, 2, 2)
    plt.plot(test_dates, true_values, label="Real Data")
    plt.plot(test_dates, knn_predictions, label="KNN")
    plt.xlabel("Time (November 2013)")
    plt.ylabel("Wind Power")
    plt.title("KNN")
    plt.legend()
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%d-%m"))
    plt.xticks(rotation=45)

    plt.subplot(2, 2, 3)
    plt.plot(test_dates, true_values, label="Real Data")
    plt.plot(test_dates, svr_predictions, label="SVR")
    plt.xlabel("Time (November 2013)")
    plt.ylabel("Wind Power")
    plt.title("SVR")
    plt.legend()
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%d-%m"))
    plt.xticks(rotation=45)

    plt.subplot(2, 2, 4)
    plt.plot(test_dates, true_values, label="Real Data")
    plt.plot(test_dates, nn_predictions, label="Neural Network")
    plt.xlabel("Time (November 2013)")
    plt.ylabel("Wind Power")
    plt.title("Neural Network")
    plt.legend()
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%d-%m"))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "part_1_plot.eps"),
        transparent=False,
        bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    part_1_style_assignment()
