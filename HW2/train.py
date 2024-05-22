import os
import pickle
import click
import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

mlflow.set_tracking_uri("http://127.0.0.1:5000")
print(f"tracking URI: '{mlflow.get_tracking_uri()}'")
mlflow.set_experiment("my-experiment-2")



def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    params = {"max_depth": 10, "n_estimators": 100, "min_samples_split": 2, "min_samples_leaf": 1, "random_state": 0}
    mlflow.log_params(params)

    rf = RandomForestRegressor(**params)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)

    mlflow.sklearn.log_model(rf, artifact_path="models")
    print(f"default artifacts URI: '{mlflow.get_artifact_uri()}'")

    rmse = mean_squared_error(y_val, y_pred, squared=False)
    print(f"RMSE: {rmse}")
    mlflow.log_metric("rmse", rmse)


with mlflow.start_run():
    run_train()