import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet
import xgboost as xgb

from src.data_loader import load_dataset, split_dataset, get_target_col
from src.features import prepare_features, get_feature_columns
from src.tsfm_models import load_chronos_pipeline, rolling_tsfm_evaluation
from src.evaluation import (
    compute_point_metrics,
    compute_crps_empirical,
    compute_picp,
    compute_interval_width,
    compute_ece_and_reliability,
)

RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def train_quantile_lgbm(X_train, y_train, X_val, y_val, alpha: float):
    """Train LightGBM for target quantile alpha with pinball loss."""
    model = lgb.LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        n_estimators=100,
        learning_rate=0.08,
        num_leaves=31,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
    )
    return model


def train_quantile_xgb(X_train, y_train, X_val, y_val, alpha: float):
    """Train XGBoost for target quantile alpha."""
    model = xgb.XGBRegressor(
        objective="reg:quantileerror",
        quantile_alpha=alpha,
        n_estimators=100,
        learning_rate=0.08,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=20,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def plot_reliability_diagram(curves: dict, dataset_name: str):
    """Plot calibration reliability diagram comparing nominal vs empirical quantiles."""
    plt.figure(figsize=(7, 6))

    # Perfect calibration diagonal
    plt.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect Calibration (y = x)")

    colors = {
        "LightGBM (Quantile)": "#2563eb",
        "XGBoost (Quantile)": "#0891b2",
        "Prophet (Gaussian)": "#f59e0b",
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
    print(f"Saved reliability diagram to {plot_path}")


def run_calibration_suite(dataset_name: str, device: str = None) -> dict:
    print(f"\nRunning calibration evaluation suite for {dataset_name.upper()}...")

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

    # 1. Quantile LightGBM
    print(f"Training quantile LightGBM models ({len(nominal_grid)} quantiles)...")
    lgb_quantiles_grid = []
    for q in nominal_grid:
        m = train_quantile_lgbm(X_train, y_train, X_val, y_val, float(q))
        lgb_quantiles_grid.append(m.predict(X_test))
    lgb_samples = np.array(lgb_quantiles_grid)

    lgb_q05 = lgb_samples[0]
    lgb_q50 = lgb_samples[5]
    lgb_q95 = lgb_samples[-1]

    ece_lgb, nom_lgb, emp_lgb = compute_ece_and_reliability(lgb_samples, y_test_arr)
    reliability_curves["LightGBM (Quantile)"] = (nom_lgb, emp_lgb)

    point_lgb = compute_point_metrics(y_test_arr, lgb_q50)
    results["lightgbm_quantile"] = {
        **point_lgb,
        "CRPS": round(compute_crps_empirical(lgb_samples, y_test_arr), 2),
        "PICP_90": round(compute_picp(y_test_arr, lgb_q05, lgb_q95), 2),
        "Width_90": round(compute_interval_width(lgb_q05, lgb_q95), 2),
        "ECE": round(ece_lgb, 4),
    }
    print(f"  LightGBM -> MAE: {point_lgb['MAE']} MW, CRPS: {results['lightgbm_quantile']['CRPS']}, PICP 90%: {results['lightgbm_quantile']['PICP_90']}%, ECE: {results['lightgbm_quantile']['ECE']}")

    # 2. Quantile XGBoost
    print(f"Training quantile XGBoost models ({len(nominal_grid)} quantiles)...")
    xgb_quantiles_grid = []
    for q in nominal_grid:
        m = train_quantile_xgb(X_train, y_train, X_val, y_val, float(q))
        xgb_quantiles_grid.append(m.predict(X_test))
    xgb_samples = np.array(xgb_quantiles_grid)

    xgb_q05 = xgb_samples[0]
    xgb_q50 = xgb_samples[5]
    xgb_q95 = xgb_samples[-1]

    ece_xgb, nom_xgb, emp_xgb = compute_ece_and_reliability(xgb_samples, y_test_arr)
    reliability_curves["XGBoost (Quantile)"] = (nom_xgb, emp_xgb)

    point_xgb = compute_point_metrics(y_test_arr, xgb_q50)
    results["xgboost_quantile"] = {
        **point_xgb,
        "CRPS": round(compute_crps_empirical(xgb_samples, y_test_arr), 2),
        "PICP_90": round(compute_picp(y_test_arr, xgb_q05, xgb_q95), 2),
        "Width_90": round(compute_interval_width(xgb_q05, xgb_q95), 2),
        "ECE": round(ece_xgb, 4),
    }
    print(f"  XGBoost  -> MAE: {point_xgb['MAE']} MW, CRPS: {results['xgboost_quantile']['CRPS']}, PICP 90%: {results['xgboost_quantile']['PICP_90']}%, ECE: {results['xgboost_quantile']['ECE']}")

    # 3. Prophet (Gaussian prediction intervals)
    print("Evaluating Prophet intervals...")
    df_p_train = pd.DataFrame({"ds": train_f.index, "y": train_f[target_col].values})
    m_p = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        seasonality_mode="additive",
        interval_width=0.90,
    )
    m_p.fit(df_p_train)
    df_test_ds = pd.DataFrame({"ds": test_f.index})
    fc_p = m_p.predict(df_test_ds)

    p_pred = fc_p["yhat"].values
    p_q05 = fc_p["yhat_lower"].values
    p_q95 = fc_p["yhat_upper"].values

    p_std = np.maximum((p_q95 - p_q05) / (2 * 1.645), 1e-4)
    rng = np.random.default_rng(42)
    p_samples = rng.normal(p_pred[None, :], p_std[None, :], size=(100, len(p_pred)))

    ece_p, nom_p, emp_p = compute_ece_and_reliability(p_samples, y_test_arr)
    reliability_curves["Prophet (Gaussian)"] = (nom_p, emp_p)

    point_p = compute_point_metrics(y_test_arr, p_pred)
    results["prophet"] = {
        **point_p,
        "CRPS": round(compute_crps_empirical(p_samples, y_test_arr), 2),
        "PICP_90": round(compute_picp(y_test_arr, p_q05, p_q95), 2),
        "Width_90": round(compute_interval_width(p_q05, p_q95), 2),
        "ECE": round(ece_p, 4),
    }
    print(f"  Prophet  -> MAE: {point_p['MAE']} MW, CRPS: {results['prophet']['CRPS']}, PICP 90%: {results['prophet']['PICP_90']}%, ECE: {results['prophet']['ECE']}")

    # 4. Chronos-T5-Small zero-shot samples
    print("Evaluating Chronos-T5-Small (512h context)...")
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
    )

    ch_samples = res_ch["samples"]
    ch_pred = res_ch["y_pred"]
    ch_q05 = res_ch["q05"]
    ch_q95 = res_ch["q95"]

    ece_ch, nom_ch, emp_ch = compute_ece_and_reliability(ch_samples, test_raw_arr)
    reliability_curves["Chronos-T5-Small (Zero-Shot)"] = (nom_ch, emp_ch)

    point_ch = compute_point_metrics(test_raw_arr, ch_pred)
    results["chronos_small_512"] = {
        **point_ch,
        "CRPS": round(compute_crps_empirical(ch_samples, test_raw_arr), 2),
        "PICP_90": round(compute_picp(test_raw_arr, ch_q05, ch_q95), 2),
        "Width_90": round(compute_interval_width(ch_q05, ch_q95), 2),
        "ECE": round(ece_ch, 4),
    }
    print(f"  Chronos  -> MAE: {point_ch['MAE']} MW, CRPS: {results['chronos_small_512']['CRPS']}, PICP 90%: {results['chronos_small_512']['PICP_90']}%, ECE: {results['chronos_small_512']['ECE']}")

    out_dir = RESULTS_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "calibration.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved calibration results to {out_path}")

    plot_reliability_diagram(reliability_curves, dataset_name)

    # Display clean summary table in terminal
    df_cal = pd.DataFrame([
        {
            "Model": m,
            "MAE (MW)": vals["MAE"],
            "RMSE (MW)": vals["RMSE"],
            "MAPE (%)": vals["MAPE"],
            "CRPS": vals["CRPS"],
            "PICP 90% (%)": vals["PICP_90"],
            "Width 90% (MW)": vals["Width_90"],
            "ECE": vals["ECE"],
        }
        for m, vals in results.items()
    ]).set_index("Model")
    print(f"\nCalibration Evaluation Table ({dataset_name.upper()}):")
    print(df_cal.to_string())

    return results


if __name__ == "__main__":
    run_calibration_suite("ercot")
    run_calibration_suite("gefcom")
