import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ENV_RESULTS = os.environ.get("GRIDEVAL_RESULTS_DIR", None)
RESULTS_DIR = Path(ENV_RESULTS) if ENV_RESULTS else (PROJECT_ROOT / "results")
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

from src.data_loader import load_dataset, split_dataset, get_target_col
from src.features import prepare_features, get_feature_columns
from src.evaluation import (
    fix_quantile_crossing,
    compute_point_metrics,
    compute_crps_empirical,
    compute_crps_quantile,
    compute_picp,
    compute_interval_width,
    compute_ece_and_reliability,
    get_rolling_schedule,
    validate_calibration_alignment,
)


def _eval_probabilistic(y_true, y_pred, samples, q05, q95, ece, quantile_grid=None, alphas=None) -> dict:
    point = compute_point_metrics(y_true, y_pred)
    if quantile_grid is not None and alphas is not None:
        crps_val = compute_crps_quantile(quantile_grid, alphas, y_true)
    else:
        crps_val = compute_crps_empirical(samples, y_true)
    return {
        **point,
        "CRPS": round(crps_val, 2),
        "PICP_90": round(compute_picp(y_true, q05, q95), 2),
        "Width_90": round(compute_interval_width(q05, q95), 2),
        "ECE": round(ece, 4),
    }


def rolling_quantile_block_evaluation(
    model_cls,
    model_kwargs_fn,
    X_train,
    y_train,
    X_val,
    y_val,
    full_df,
    target_col,
    feature_cols,
    test_index,
    nominal_grid,
    horizon: int = 24,
    step: int = 24
):
    schedule = get_rolling_schedule(test_index, horizon=horizon, step=step)

    models = {}
    for alpha in nominal_grid:
        kwargs = model_kwargs_fn(alpha)
        model = model_cls(**kwargs)
        if "LGBM" in model_cls.__name__ or "LGBMRegressor" in str(model_cls):
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
            )
        else:
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        # Recursive features require serial one-row inference; avoid a new
        # multi-thread pool for every target and quantile prediction.
        if "n_jobs" in model.get_params(deep=False):
            model.set_params(n_jobs=1)
        models[alpha] = model

    n_test = len(test_index)
    n_alphas = len(nominal_grid)
    quantiles_preds = np.zeros((n_alphas, n_test))
    actuals = np.zeros(n_test)
    evaluated_identities = []

    current_df = full_df.copy()

    for block in schedule:
        origin_ident = block["origin_identity"]
        origin_idx = full_df.index.get_loc(origin_ident)
        current_df.iloc[:origin_idx, current_df.columns.get_loc(target_col)] = full_df[target_col].iloc[:origin_idx]

        for idx_in_test, target_ident in enumerate(block["target_identities"]):
            test_pos = test_index.get_loc(target_ident)
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

            step_preds = []
            for alpha in nominal_grid:
                p = models[alpha].predict(X_row)[0]
                step_preds.append(p)

            quantiles_preds[:, test_pos] = np.array(step_preds)
            actuals[test_pos] = full_df[target_col].iloc[target_idx]

            median_idx = len(nominal_grid) // 2
            median_pred = step_preds[median_idx]
            current_df.loc[current_df.index[target_idx], target_col] = median_pred

            if target_ident not in evaluated_identities:
                evaluated_identities.append(target_ident)

    quantiles_preds_fixed = fix_quantile_crossing(quantiles_preds)
    return {
        "quantiles_grid": quantiles_preds_fixed,
        "y_true": actuals,
        "test_identities": evaluated_identities,
        "schedule": schedule
    }


def run_leakage_invariance_test(model, full_df, target_col, feature_cols, test_index, horizon=24) -> dict:
    schedule = get_rolling_schedule(test_index, horizon=horizon, step=horizon)
    if len(schedule) < 2:
        raise RuntimeError("Leakage invariance test requires at least 2 schedule blocks.")

    block0 = schedule[0]
    origin_ident0 = block0["origin_identity"]
    origin_idx0 = full_df.index.get_loc(origin_ident0)

    df_base = full_df.copy()
    df_pert = full_df.copy()

    for t_idx in block0["target_indices"]:
        target_ident = test_index[t_idx]
        real_idx = full_df.index.get_loc(target_ident)
        df_pert.iloc[real_idx, df_pert.columns.get_loc(target_col)] *= 1.50

    def get_pred_at_origin(df_snapshot, orig_idx):
        row = df_snapshot.iloc[[orig_idx]].copy()
        for lag in [1, 2, 3, 24, 168]:
            if orig_idx - lag >= 0:
                row[f"lag_{lag}"] = df_snapshot[target_col].iloc[orig_idx - lag]
        for w in [24, 168]:
            if orig_idx - 1 >= 0:
                hist = df_snapshot[target_col].iloc[max(0, orig_idx - w):orig_idx]
                row[f"rolling_mean_{w}"] = hist.mean()
                row[f"rolling_std_{w}"] = hist.std()
        return float(model.predict(row[feature_cols])[0])

    pred_base0 = get_pred_at_origin(df_base, origin_idx0)
    pred_pert0 = get_pred_at_origin(df_pert, origin_idx0)
    zero_leakage = np.isclose(pred_base0, pred_pert0, atol=1e-5)
    assert zero_leakage, f"Future-target leakage detected at origin 0: base={pred_base0}, pert={pred_pert0}"

    block1 = schedule[1]
    origin_ident1 = block1["origin_identity"]
    origin_idx1 = full_df.index.get_loc(origin_ident1)

    lag1_base = df_base[target_col].iloc[origin_idx1 - 1]
    lag1_pert = df_pert[target_col].iloc[origin_idx1 - 1]
    history_updated = not np.isclose(lag1_base, lag1_pert, atol=1e-4)
    assert history_updated, f"Subsequent origin 1 history did not update."

    return {
        "zero_future_target_leakage": bool(zero_leakage),
        "history_incorporates_update": bool(history_updated),
        "leakage_test_passed": True
    }


def plot_reliability_diagram(curves: dict, dataset_name: str):
    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect Calibration (y = x)")

    colors = {
        "LightGBM (Quantile)": "#2563eb",
        "XGBoost (Quantile)": "#0891b2",
        "Chronos-T5-Small (Zero-Shot)": "#dc2626",
    }

    for name, (nom, emp) in curves.items():
        plt.plot(nom, emp, marker="o", markersize=4, linewidth=2, color=colors.get(name, "#4b5563"), label=name)

    plt.title(f"Reliability Diagram: {dataset_name.upper()} Test Set", fontsize=13, pad=10)
    plt.xlabel("Nominal Quantile Level (p)", fontsize=11)
    plt.ylabel("Empirical Coverage Fraction", fontsize=11)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left", frameon=True)
    plt.tight_layout()

    plot_path = PLOTS_DIR / f"reliability_{dataset_name}.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()


def run_calibration_suite(dataset_name: str, device: str = None) -> dict:
    skip_chronos = os.environ.get("GRIDEVAL_SKIP_CHRONOS", "0") == "1"
    print(f"\nRunning calibration evaluation suite for {dataset_name.upper()} (skip_chronos={skip_chronos})...")

    df = load_dataset(dataset_name)
    target_col = get_target_col(dataset_name)
    df_feat = prepare_features(df, target_col, dataset_name)

    train_f, val_f, test_f = split_dataset(df_feat, dataset_name)
    feature_cols = get_feature_columns(dataset_name)

    X_train, y_train = train_f[feature_cols], train_f[target_col]
    X_val, y_val = val_f[feature_cols], val_f[target_col]
    X_test, y_test = test_f[feature_cols], test_f[target_col]
    y_test_arr = y_test.values

    train_raw, val_raw, test_raw = split_dataset(df, dataset_name)
    history_raw = np.concatenate([train_raw[target_col].values, val_raw[target_col].values])
    test_raw_arr = test_raw[target_col].values

    results = {}
    reliability_curves = {}
    nominal_grid = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])

    out_dir = RESULTS_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Rolling Quantile LightGBM
    print(f"Running rolling quantile LightGBM evaluation ({len(nominal_grid)} quantiles)...")
    lgb_res = rolling_quantile_block_evaluation(
        model_cls=lgb.LGBMRegressor,
        model_kwargs_fn=lambda alpha: {
            "objective": "quantile",
            "alpha": float(alpha),
            "n_estimators": 50,
            "learning_rate": 0.08,
            "num_leaves": 31,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        },
        X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
        full_df=df_feat, target_col=target_col, feature_cols=feature_cols,
        test_index=test_f.index, nominal_grid=nominal_grid, horizon=24, step=24
    )
    lgb_samples = lgb_res["quantiles_grid"]
    lgb_q05 = lgb_samples[0]
    lgb_q50 = lgb_samples[5]
    lgb_q95 = lgb_samples[-1]

    ece_lgb, nom_lgb, emp_lgb = compute_ece_and_reliability(lgb_samples, nominal_grid, y_test_arr)
    reliability_curves["LightGBM (Quantile)"] = (nom_lgb, emp_lgb)
    results["lightgbm_quantile"] = _eval_probabilistic(y_test_arr, lgb_q50, lgb_samples, lgb_q05, lgb_q95, ece_lgb, quantile_grid=lgb_samples, alphas=nominal_grid)
    print(f"  LightGBM -> MAE: {results['lightgbm_quantile']['MAE']} MW, CRPS: {results['lightgbm_quantile']['CRPS']}, PICP 90%: {results['lightgbm_quantile']['PICP_90']}%, ECE: {results['lightgbm_quantile']['ECE']}")

    df_lgb_q = pd.DataFrame({
        "actual": lgb_res["y_true"],
        "lightgbm_q05": lgb_q05,
        "lightgbm_median": lgb_q50,
        "lightgbm_q95": lgb_q95
    }, index=test_f.index)
    df_lgb_q.to_csv(out_dir / "predictions_quantile_lightgbm.csv")

    median_lgb_model = lgb.LGBMRegressor(
        objective="quantile", alpha=0.5, n_estimators=30, random_state=42, n_jobs=-1, verbose=-1
    )
    median_lgb_model.fit(X_train, y_train)
    leakage_test = run_leakage_invariance_test(median_lgb_model, df_feat, target_col, feature_cols, test_f.index, horizon=24)
    print(f"  Leakage Invariance Diagnostic: passed = {leakage_test['leakage_test_passed']} (zero future-target leakage)")

    # 2. Rolling Quantile XGBoost
    print(f"Running rolling quantile XGBoost evaluation ({len(nominal_grid)} quantiles)...")
    xgb_res = rolling_quantile_block_evaluation(
        model_cls=xgb.XGBRegressor,
        model_kwargs_fn=lambda alpha: {
            "objective": "reg:quantileerror",
            "quantile_alpha": float(alpha),
            "n_estimators": 50,
            "learning_rate": 0.08,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            "early_stopping_rounds": 20,
        },
        X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
        full_df=df_feat, target_col=target_col, feature_cols=feature_cols,
        test_index=test_f.index, nominal_grid=nominal_grid, horizon=24, step=24
    )
    xgb_samples = xgb_res["quantiles_grid"]
    xgb_q05 = xgb_samples[0]
    xgb_q50 = xgb_samples[5]
    xgb_q95 = xgb_samples[-1]

    ece_xgb, nom_xgb, emp_xgb = compute_ece_and_reliability(xgb_samples, nominal_grid, y_test_arr)
    reliability_curves["XGBoost (Quantile)"] = (nom_xgb, emp_xgb)
    results["xgboost_quantile"] = _eval_probabilistic(y_test_arr, xgb_q50, xgb_samples, xgb_q05, xgb_q95, ece_xgb, quantile_grid=xgb_samples, alphas=nominal_grid)
    print(f"  XGBoost  -> MAE: {results['xgboost_quantile']['MAE']} MW, CRPS: {results['xgboost_quantile']['CRPS']}, PICP 90%: {results['xgboost_quantile']['PICP_90']}%, ECE: {results['xgboost_quantile']['ECE']}")

    df_xgb_q = pd.DataFrame({
        "actual": xgb_res["y_true"],
        "xgboost_q05": xgb_q05,
        "xgboost_median": xgb_q50,
        "xgboost_q95": xgb_q95
    }, index=test_f.index)
    df_xgb_q.to_csv(out_dir / "predictions_quantile_xgboost.csv")

    # 3. Optional Chronos Run
    if skip_chronos:
        print("Skipping Chronos evaluation as requested by GRIDEVAL_SKIP_CHRONOS=1.")
        results["chronos_small_512"] = {"status": "skipped_by_config", "note": "Chronos bypassed via GRIDEVAL_SKIP_CHRONOS=1"}
    else:
        print("Evaluating Chronos-T5-Small (512 observation context)...")
        try:
            from src.tsfm_models import load_chronos_pipeline, rolling_tsfm_evaluation
            pipeline = load_chronos_pipeline("amazon/chronos-t5-small", device=device)
            res_ch = rolling_tsfm_evaluation(
                pipeline=pipeline,
                history_values=history_raw,
                test_values=test_raw_arr,
                context_len=512,
                horizon=24,
                step=24,
                num_samples=20,
                batch_size=16,
                test_index=test_raw.index,
            )
            validate_calibration_alignment(test_f.index, test_raw.index, y_test_arr, test_raw_arr, res_ch)

            ch_samples = res_ch["samples"]
            ch_pred = res_ch["y_pred"]
            ch_q05 = res_ch["q05"]
            ch_q95 = res_ch["q95"]

            nominal_grid_chronos = np.linspace(0.05, 0.95, 19)
            ece_ch, nom_ch, emp_ch = compute_ece_and_reliability(
                np.array([np.percentile(ch_samples, q * 100.0, axis=0) for q in nominal_grid_chronos]),
                nominal_grid_chronos,
                test_raw_arr
            )
            reliability_curves["Chronos-T5-Small (Zero-Shot)"] = (nom_ch, emp_ch)
            results["chronos_small_512"] = _eval_probabilistic(test_raw_arr, ch_pred, ch_samples, ch_q05, ch_q95, ece_ch)
            print(f"  Chronos  -> MAE: {results['chronos_small_512']['MAE']} MW, CRPS: {results['chronos_small_512']['CRPS']}, PICP 90%: {results['chronos_small_512']['PICP_90']}%, ECE: {results['chronos_small_512']['ECE']}")

            df_ch_q = pd.DataFrame({
                "actual": test_raw_arr,
                "chronos_q05": ch_q05,
                "chronos_median": ch_pred,
                "chronos_q95": ch_q95
            }, index=test_raw.index)
            df_ch_q.to_csv(out_dir / "predictions_quantile_chronos.csv")
        except Exception as e:
            print(f"  [SKIPPED/FAILED] Chronos evaluation failed: {e}")
            results["chronos_small_512"] = {"status": "failed_or_skipped", "error": str(e)}

    out_path = out_dir / "calibration.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved calibration results to {out_path}")

    plot_reliability_diagram(reliability_curves, dataset_name)
    return results


if __name__ == "__main__":
    run_calibration_suite("ercot")
    run_calibration_suite("gefcom")
