import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# import lightgbm as lgb
import numpy as np
import pandas as pd
# import xgboost as xgb

from src.data_loader import load_dataset, split_dataset, get_target_col
from src.features import prepare_features, get_feature_columns
from src.evaluation import compute_point_metrics, get_rolling_schedule

import os
ENV_RESULTS = os.environ.get("GRIDEVAL_RESULTS_DIR", None)
RESULTS_DIR = Path(ENV_RESULTS) if ENV_RESULTS else (PROJECT_ROOT / "results")


def rolling_block_evaluation(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    full_df,
    target_col,
    feature_cols,
    test_index,
    horizon: int = 24,
    step: int = 24
):
    """
    Evaluate GBDTs in 24-observation block rolling forecast mode.

    Forecast Schedule & Policy:
    - Horizon: 24 observation steps per forecast block
    - Step: 24 observation steps between forecast origins
    - Refitting Policy: Single fit on training split with early stopping on validation split.
    - Recursive Lag & Rolling Update: Uses prior model predictions for future target values within the forecast horizon, preventing future-target leakage.
    - Strict Chronological Identity Validation: Uses shared get_rolling_schedule to ensure 1:1 identity alignment.
    """
    if test_index.duplicated().any():
        raise ValueError("test_index contains duplicate entries.")

    if isinstance(test_index, pd.MultiIndex):
        flat_test = list(test_index)
        if flat_test != sorted(flat_test):
            raise ValueError("test_index must be strictly sorted in chronological order.")
    else:
        if not test_index.is_monotonic_increasing:
            raise ValueError("test_index must be strictly sorted in chronological order.")

    eval_set = [(X_val, y_val)]
    estimator_module = model.__class__.__module__.split(".", 1)[0]
    if estimator_module == "lightgbm":
        import lightgbm as lgb

        model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )
    else:
        model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

    # Forecasts are generated one identity at a time because recursive lag
    # features depend on the preceding prediction. Avoid spawning a thread pool
    # for every single-row prediction after the model has finished fitting.
    get_params = getattr(model, "get_params", None)
    set_params = getattr(model, "set_params", None)
    if get_params is not None and set_params is not None and "n_jobs" in get_params(deep=False):
        set_params(n_jobs=1)

    schedule = get_rolling_schedule(test_index, horizon=horizon, step=step)

    preds = []
    actuals = []
    evaluated_identities = []
    current_df = full_df.copy()

    for block in schedule:
        # Reset current_df target values up to origin to actual full_df values,
        # ensuring newly available actuals between blocks are fully incorporated without carrying forward past prediction errors.
        origin_ident = block["origin_identity"]
        origin_idx = full_df.index.get_loc(origin_ident)
        # Reset all history prior to origin_idx to actuals from full_df
        current_df.iloc[:origin_idx, current_df.columns.get_loc(target_col)] = full_df[target_col].iloc[:origin_idx]

        for idx_in_test, target_ident in enumerate(block["target_identities"]):
            if target_ident not in full_df.index:
                raise KeyError(f"Test identity {target_ident} not found in full dataset index.")
            target_idx = full_df.index.get_loc(target_ident)

            row = current_df.iloc[[target_idx]].copy()

            for lag in [1, 2, 3, 24, 168]:
                if target_idx - lag >= 0:
                    row[f"lag_{lag}"] = current_df[target_col].iloc[target_idx - lag]

            for w in [24, 168]:
                if target_idx - 1 >= 0:
                    hist = current_df[target_col].iloc[max(0, target_idx - w):target_idx]
                    row[f"rolling_mean_{w}"] = hist.mean()
                    row[f"rolling_std_{w}"] = hist.std()

            X_row = row[feature_cols]
            pred = model.predict(X_row)[0]

            current_df.loc[current_df.index[target_idx], target_col] = pred
            preds.append(pred)
            actuals.append(full_df[target_col].iloc[target_idx])
            evaluated_identities.append(target_ident)

    return {
        "y_pred": np.array(preds),
        "y_true": np.array(actuals),
        "test_identities": evaluated_identities,
        "schedule": schedule
    }


def run_baselines_for_dataset(dataset_name: str, include_weather: bool = False) -> dict:
    import lightgbm as lgb
    import xgboost as xgb
    print(f"\nRunning baseline models for {dataset_name.upper()}...")

    df = load_dataset(dataset_name)
    target_col = get_target_col(dataset_name)
    df_feat = prepare_features(df, target_col, dataset_name)

    train, val, test = split_dataset(df_feat, dataset_name)
    feature_cols = get_feature_columns(dataset_name, include_weather=include_weather)

    X_train, y_train = train[feature_cols], train[target_col]
    X_val, y_val = val[feature_cols], val[target_col]
    X_test, y_test = test[feature_cols], test[target_col]

    print(f"Train samples: {len(X_train)} ({train.index.min()} to {train.index.max()})")
    print(f"Val samples:   {len(X_val)} ({val.index.min()} to {val.index.max()})")
    print(f"Test samples:  {len(X_test)} ({test.index.min()} to {test.index.max()})")
    print(f"Features ({len(feature_cols)}): {', '.join(feature_cols)}\n")

    results = {}

    # LightGBM with 24h rolling block evaluation
    print("Fitting LightGBM regressor and running 24h rolling block evaluation...")
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
    res_lgb = rolling_block_evaluation(
        model_lgb, X_train, y_train, X_val, y_val, df_feat, target_col, feature_cols, y_test.index, horizon=24, step=24
    )
    pred_lgb = res_lgb["y_pred"]

    test_len = len(y_test)
    if len(pred_lgb) != test_len:
        raise ValueError(f"LightGBM prediction length {len(pred_lgb)} does not match test set length {test_len}.")

    results["lightgbm"] = compute_point_metrics(y_test.values, pred_lgb)
    print(f"  LightGBM -> MAE: {results['lightgbm']['MAE']} MW, MAPE: {results['lightgbm']['MAPE']}%")

    # XGBoost with 24h rolling block evaluation
    print("Fitting XGBoost regressor and running 24h rolling block evaluation...")
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
    res_xgb = rolling_block_evaluation(
        model_xgb, X_train, y_train, X_val, y_val, df_feat, target_col, feature_cols, y_test.index, horizon=24, step=24
    )
    pred_xgb = res_xgb["y_pred"]
    if len(pred_xgb) != test_len:
        raise ValueError(f"XGBoost prediction length {len(pred_xgb)} does not match test set length {test_len}.")

    results["xgboost"] = compute_point_metrics(y_test.values, pred_xgb)
    print(f"  XGBoost  -> MAE: {results['xgboost']['MAE']} MW, MAPE: {results['xgboost']['MAPE']}%")

    out_dir = RESULTS_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "baselines.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save timestamp-level predictions for dashboard & audit aligned to y_test.index
    df_preds = pd.DataFrame({
        "actual": y_test.values,
        "lightgbm": pred_lgb,
        "xgboost": pred_xgb
    }, index=y_test.index)
    df_preds.to_csv(out_dir / "predictions_baselines.csv")
    print(f"\nSaved baseline predictions to {out_dir / 'predictions_baselines.csv'}")
    print(f"\nSaved baseline metrics to {out_path}")

    df_res = pd.DataFrame([
        {"Model": m, "MAE (MW)": vals["MAE"], "RMSE (MW)": vals["RMSE"], "MAPE (%)": vals["MAPE"]}
        for m, vals in results.items()
    ]).set_index("Model")
    print(f"\nBaseline Performance Table ({dataset_name.upper()}):")
    print(df_res.to_string())

    return results


if __name__ == "__main__":
    pass
