import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data_loader import load_dataset, split_dataset, get_target_col
from src.tsfm_models import load_chronos_pipeline, rolling_tsfm_evaluation

RESULTS_DIR = PROJECT_ROOT / "results"
CONTEXT_LENGTHS = [72, 168, 512, 1000]


def run_context_sweep_for_dataset(dataset_name: str, model_id: str = "amazon/chronos-t5-small", device: str = None) -> dict:
    """Run context length sensitivity sweep on the specified dataset."""
    print(f"\nRunning TSFM context sweep for {dataset_name.upper()} using {model_id}...")

    df = load_dataset(dataset_name)
    target_col = get_target_col(dataset_name)
    train, val, test = split_dataset(df, dataset_name)

    train_vals = train[target_col].values
    val_vals = val[target_col].values
    test_vals = test[target_col].values

    history_vals = np.concatenate([train_vals, val_vals])
    pipeline = load_chronos_pipeline(model_id, device=device)

    sweep_results = {}

    for ctx in CONTEXT_LENGTHS:
        print(f"\nEvaluating context length: {ctx} hours...")

        res = rolling_tsfm_evaluation(
            pipeline=pipeline,
            history_values=history_vals,
            test_values=test_vals,
            context_len=ctx,
            horizon=24,
            step=24,
            num_samples=20,
            batch_size=16,
        )

        y_true = res["y_true"]
        y_pred = res["y_pred"]
        q05 = res["q05"]
        q95 = res["q95"]

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0

        coverage_90 = np.mean((y_true >= q05) & (y_true <= q95)) * 100.0
        width_90 = np.mean(q95 - q05)

        sweep_results[ctx] = {
            "MAE": round(float(mae), 2),
            "RMSE": round(float(rmse), 2),
            "MAPE": round(float(mape), 2),
            "Coverage_90": round(float(coverage_90), 2),
            "Width_90": round(float(width_90), 2),
        }

        print(f"  Context {ctx}h -> MAE: {mae:.2f} MW, MAPE: {mape:.2f}%, PICP 90%: {coverage_90:.1f}%, Width: {width_90:.2f} MW")

    out_dir = RESULTS_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tsfm_context_sweep.json"
    with open(out_path, "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\nSaved context sweep results to {out_path}")

    # Display clean summary table in terminal
    df_sweep = pd.DataFrame([
        {
            "Context": f"{ctx}h",
            "MAE (MW)": vals["MAE"],
            "RMSE (MW)": vals["RMSE"],
            "MAPE (%)": vals["MAPE"],
            "Coverage 90% (%)": vals["Coverage_90"],
            "Width 90% (MW)": vals["Width_90"],
        }
        for ctx, vals in sweep_results.items()
    ]).set_index("Context")
    print(f"\nTSFM Context Scaling Table ({dataset_name.upper()}):")
    print(df_sweep.to_string())

    return sweep_results


if __name__ == "__main__":
    run_context_sweep_for_dataset("ercot")
    run_context_sweep_for_dataset("gefcom")
