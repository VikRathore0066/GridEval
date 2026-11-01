import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.data_loader import load_dataset, split_dataset, get_target_col
from src.tsfm_models import load_chronos_pipeline, rolling_tsfm_evaluation
from src.evaluation import compute_point_metrics, compute_picp, compute_interval_width, get_rolling_schedule

import os
ENV_RESULTS = os.environ.get("GRIDEVAL_RESULTS_DIR", None)
RESULTS_DIR = Path(ENV_RESULTS) if ENV_RESULTS else (PROJECT_ROOT / "results")
CONTEXT_LENGTHS = [72, 168, 512, 1000]


def run_context_sweep_for_dataset(cfg: dict, device: str = None) -> dict:
    dataset_name = cfg["dataset"]
    model_id = cfg.get("model_id", "amazon/chronos-t5-small")
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
    contexts = cfg.get("tsfm", {}).get("context_lengths", [72, 168, 512, 1000])

    for ctx in contexts:
        print(f"\nEvaluating context length: {ctx} observations...")

        res = rolling_tsfm_evaluation(
            pipeline=pipeline,
            history_values=history_vals,
            test_values=test_vals,
            context_len=ctx,
            horizon=cfg.get("tsfm", {}).get("horizon", 24),
            step=cfg.get("tsfm", {}).get("step", 24),
            num_samples=cfg.get("tsfm", {}).get("num_samples", 20),
            batch_size=16,
            test_index=test.index,
        )
        if "test_identities" not in res or "schedule" not in res:
            raise ValueError("Chronos evaluation result missing required test_identities or schedule.")
        if list(res["test_identities"]) != list(test.index):
            raise ValueError("Chronos evaluated test_identities do not match test.index.")
        expected_sched = get_rolling_schedule(test.index, horizon=cfg.get("tsfm", {}).get("horizon", 24), step=cfg.get("tsfm", {}).get("step", 24))
        if res["schedule"] != expected_sched:
            raise ValueError("Chronos returned schedule does not match expected rolling schedule.")

        y_true = res["y_true"]
        y_pred = res["y_pred"]
        q05 = res["q05"]
        q95 = res["q95"]

        point = compute_point_metrics(y_true, y_pred)
        coverage_90 = compute_picp(y_true, q05, q95)
        width_90 = compute_interval_width(q05, q95)

        sweep_results[ctx] = {
            **point,
            "Coverage_90": coverage_90,
            "Width_90": width_90,
        }

        print(f"  Context {ctx} observations -> MAE: {point['MAE']} MW, MAPE: {point['MAPE']}%, PICP 90%: {coverage_90}%, Width: {width_90} MW")

    out_dir = RESULTS_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tsfm_context_sweep.json"
    with open(out_path, "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"\nSaved context sweep results to {out_path}")

    # Display clean summary table in terminal
    df_sweep = pd.DataFrame([
        {
            "Context": f"{ctx} observations",
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
    run_context_sweep_for_dataset({"dataset": "ercot", "model_id": "amazon/chronos-t5-small"})
