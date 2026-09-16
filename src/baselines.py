import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import lightgbm as lgb
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

from src.data_loader import load_dataset, split_dataset, get_target_col
from src.features import prepare_features, get_feature_columns

RESULTS_DIR = PROJECT_ROOT / "results"


def compute_point_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0
    return {
        "MAE": round(float(mae), 2),
        "RMSE": round(float(rmse), 2),
        "MAPE": round(float(mape), 2),
    }


def run_baselines_for_dataset(dataset_name: str) -> dict:
    print(f"\nRunning baseline models for {dataset_name.upper()}...")

    df = load_dataset(dataset_name)
    target_col = get_target_col(dataset_name)
    df_feat = prepare_features(df, target_col, dataset_name)

    train, val, test = split_dataset(df_feat, dataset_name)
    feature_cols = get_feature_columns(dataset_name)

    X_train, y_train = train[feature_cols], train[target_col]
    X_val, y_val = val[feature_cols], val[target_col]
    X_test, y_test = test[feature_cols], test[target_col]

    print(f"Train samples: {len(X_train)} ({train.index.min()} to {train.index.max()})")
    print(f"Val samples:   {len(X_val)} ({val.index.min()} to {val.index.max()})")
    print(f"Test samples:  {len(X_test)} ({test.index.min()} to {test.index.max()})")
    print(f"Features ({len(feature_cols)}): {', '.join(feature_cols)}\n")

    results = {}

    # LightGBM
    print("Fitting LightGBM regressor...")
    model_lgb = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
    )
    model_lgb.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    pred_lgb = model_lgb.predict(X_test)
    results["lightgbm"] = compute_point_metrics(y_test, pred_lgb)
    print(f"  LightGBM -> MAE: {results['lightgbm']['MAE']} MW, MAPE: {results['lightgbm']['MAPE']}%")

    # XGBoost
    print("Fitting XGBoost regressor...")
    model_xgb = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50,
    )
    model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    pred_xgb = model_xgb.predict(X_test)
    results["xgboost"] = compute_point_metrics(y_test, pred_xgb)
    print(f"  XGBoost  -> MAE: {results['xgboost']['MAE']} MW, MAPE: {results['xgboost']['MAPE']}%")

    # Prophet with additive components
    print("Fitting Prophet...")
    df_prophet_train = pd.DataFrame({"ds": train.index, "y": train[target_col].values})
    model_prophet = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        seasonality_mode="additive",
        changepoint_prior_scale=0.05,
    )
    model_prophet.fit(df_prophet_train)

    df_test_ds = pd.DataFrame({"ds": test.index})
    forecast = model_prophet.predict(df_test_ds)
    pred_prophet = forecast["yhat"].values
    results["prophet"] = compute_point_metrics(y_test.values, pred_prophet)
    print(f"  Prophet  -> MAE: {results['prophet']['MAE']} MW, MAPE: {results['prophet']['MAPE']}%")

    # Naive seasonal persistence (1-week lag)
    print("Computing naive seasonal baseline (1-week lag)...")
    combined_vals = np.concatenate([val[target_col].values[-168:], test[target_col].values])
    naive_pred = combined_vals[:-168]
    results["naive_seasonal_1wk"] = compute_point_metrics(test[target_col].values, naive_pred)
    print(f"  Naive    -> MAE: {results['naive_seasonal_1wk']['MAE']} MW, MAPE: {results['naive_seasonal_1wk']['MAPE']}%")

    out_dir = RESULTS_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "baselines.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved baseline metrics to {out_path}")

    # Display clean summary table in terminal
    df_res = pd.DataFrame([
        {"Model": m, "MAE (MW)": vals["MAE"], "RMSE (MW)": vals["RMSE"], "MAPE (%)": vals["MAPE"]}
        for m, vals in results.items()
    ]).set_index("Model")
    print(f"\nBaseline Performance Table ({dataset_name.upper()}):")
    print(df_res.to_string())

    return results


if __name__ == "__main__":
    run_baselines_for_dataset("ercot")
    run_baselines_for_dataset("gefcom")
