
from tqdm import tqdm
import os
import mlflow
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from digitalhub import from_mlflow_run, get_mlflow_model_metrics

import requests
import datetime
import pandas as pd

def train(project, train_data):
    # Enable MLflow autologging for sklearn
    mlflow.sklearn.autolog(log_datasets=True)

    df = train_data.as_df()
    train_data = df[:30000]
    val_data = df[30000:]
    model = LinearRegression()

    # data labeling
    target = "duration_min"
    num_features = ["passenger_count", "trip_distance", "fare_amount", "total_amount"]
    cat_features = ["PULocationID", "DOLocationID"]

    model.fit(train_data[num_features + cat_features], train_data[target])

    val_preds = model.predict(val_data[num_features + cat_features])
    val_data['prediction'] = val_preds

    mae = mean_absolute_error(val_data.duration_min, val_data.prediction)
    # Get MLflow run information
    run_id = mlflow.last_active_run().info.run_id

    # Extract MLflow run artifacts and metadata for DigitalHub integration
    model_params = from_mlflow_run(run_id)
    metrics = get_mlflow_model_metrics(run_id)

    # Register model in DigitalHub with MLflow metadata
    model = project.log_model(name="taxi-predictor", kind="mlflow", **model_params)
    model.log_metrics(metrics)

