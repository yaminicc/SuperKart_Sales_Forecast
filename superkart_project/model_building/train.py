"""
Model training, evaluation and promotion step.

1. Loads the train/test splits produced by prep.py.
2. Computes a naive baseline (mean sales per Store x Product_Type group).
3. Tunes an XGBoost regressor with GridSearchCV, logging every trial to MLflow.
4. Evaluates the best model on the held-out test set.
5. Promotion gate: the new model is saved for deployment ONLY if it beats the
   baseline AND is at least as good as the model currently in production.
   This is what stops the pipeline from "automatically deploying worse models".
"""
import json
import os

import joblib
import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import make_column_transformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# MLflow: GitHub Actions starts a local server on port 5000; override via env
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("SuperKart-Sales-Forecast-Experiment")

DEPLOY_DIR = "superkart_project/deployment"
MODEL_PATH = os.path.join(DEPLOY_DIR, "best_superkart_sales_model_v1.joblib")
METRICS_PATH = os.path.join(DEPLOY_DIR, "model_metrics.json")

# --- Load splits (downloaded from the previous job's artifact) -----------
Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest = pd.read_csv("ytest.csv").squeeze()

numeric_features = ["Product_Weight", "Product_Allocated_Area", "Product_MRP"]
categorical_features = [
    "Product_Sugar_Content", "Product_Type", "Store_Id",
    "Store_Size", "Store_Location_City_Type", "Store_Type",
]


def evaluate(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


# --- Baseline: mean sales per (Store_Id, Product_Type) -------------------
# If XGBoost cannot beat this, the pipeline is automating nothing useful.
group_cols = ["Store_Id", "Product_Type"]
group_means = ytrain.groupby([Xtrain[c] for c in group_cols]).mean()
global_mean = float(ytrain.mean())


def baseline_predict(X):
    keys = list(zip(*[X[c] for c in group_cols]))
    return np.array([group_means.get(k, global_mean) for k in keys])


baseline_metrics = evaluate(ytest, baseline_predict(Xtest))
print("Baseline (group mean) test metrics:", baseline_metrics)

# --- Model pipeline -------------------------------------------------------
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features),
)
xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
model_pipeline = make_pipeline(preprocessor, xgb_model)

param_grid = {
    "xgbregressor__n_estimators": [100, 200, 300],
    "xgbregressor__max_depth": [3, 4, 5],
    "xgbregressor__learning_rate": [0.05, 0.1],
    "xgbregressor__subsample": [0.8, 1.0],
    "xgbregressor__colsample_bytree": [0.8, 1.0],
}

with mlflow.start_run():
    mlflow.log_metrics({f"baseline_test_{k}": v for k, v in baseline_metrics.items()})

    grid_search = GridSearchCV(
        model_pipeline, param_grid, cv=5,
        scoring="neg_root_mean_squared_error", n_jobs=-1,
    )
    grid_search.fit(Xtrain, ytrain)

    # Log every trial as a nested run so they can be compared in the MLflow UI
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(results["params"][i])
            mlflow.log_metric("cv_rmse", -results["mean_test_score"][i])
            mlflow.log_metric("cv_rmse_std", results["std_test_score"][i])

    mlflow.log_params(grid_search.best_params_)
    best_model = grid_search.best_estimator_

    train_metrics = evaluate(ytrain, best_model.predict(Xtrain))
    test_metrics = evaluate(ytest, best_model.predict(Xtest))
    mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
    mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

    print("Best params:", grid_search.best_params_)
    print("Train metrics:", train_metrics)
    print("Test metrics: ", test_metrics)

    # --- Promotion gate ---------------------------------------------------
    beats_baseline = test_metrics["rmse"] < baseline_metrics["rmse"]

    current_rmse = None
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            current_rmse = json.load(f).get("test", {}).get("rmse")
    # Allow a tiny tolerance so re-running on identical data does not flip-flop
    beats_current = current_rmse is None or test_metrics["rmse"] <= current_rmse * 1.01

    mlflow.log_metric("beats_baseline", int(beats_baseline))
    mlflow.log_metric("promoted", int(beats_baseline and beats_current))

    if not beats_baseline:
        raise SystemExit(
            f"Model NOT promoted: test RMSE {test_metrics['rmse']:.1f} does not beat "
            f"baseline RMSE {baseline_metrics['rmse']:.1f}."
        )
    if not beats_current:
        print(
            f"Model NOT promoted: test RMSE {test_metrics['rmse']:.1f} is worse than the "
            f"current production model ({current_rmse:.1f}). Keeping the existing model."
        )
    else:
        os.makedirs(DEPLOY_DIR, exist_ok=True)
        joblib.dump(best_model, MODEL_PATH)
        with open(METRICS_PATH, "w") as f:
            json.dump(
                {
                    "best_params": grid_search.best_params_,
                    "baseline": baseline_metrics,
                    "train": train_metrics,
                    "test": test_metrics,
                },
                f, indent=2,
            )
        mlflow.log_artifact(MODEL_PATH, artifact_path="model")
        mlflow.log_artifact(METRICS_PATH, artifact_path="model")
        print(f"Model PROMOTED and saved to {MODEL_PATH}")
