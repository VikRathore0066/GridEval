import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import yaml

from src.utils import set_all_seeds
from src.experiments import execute_experiment, RESULTS_DIR, RUN_PROTOCOL_VERSION

import importlib

CONFIGS_DIR = PROJECT_ROOT / "configs"


def get_pipeline_fn(pipeline_module_name: str):
    mod = importlib.import_module(pipeline_module_name)
    return mod.run


def get_experiment_registry():
    return [
        ("ercot_baselines", CONFIGS_DIR / "ercot_baselines.yaml", "src.pipelines.baselines"),
        ("gefcom_baselines", CONFIGS_DIR / "gefcom_baselines.yaml", "src.pipelines.baselines"),
        ("ercot_tsfm", CONFIGS_DIR / "ercot_tsfm.yaml", "src.pipelines.tsfm"),
        ("gefcom_tsfm", CONFIGS_DIR / "gefcom_tsfm.yaml", "src.pipelines.tsfm"),
        ("ercot_calibration", CONFIGS_DIR / "ercot_calibration.yaml", "src.pipelines.calibration"),
        ("gefcom_calibration", CONFIGS_DIR / "gefcom_calibration.yaml", "src.pipelines.calibration"),
    ]


def build_master_tables():
    """Aggregate individual JSON results into summary CSV master tables."""
    print("\nCompiling master results tables...")

    # Table 1: Point accuracy metrics
    acc_rows = []
    for ds in ["ercot", "gefcom"]:
        b_path = RESULTS_DIR / ds / "baselines.json"
        if b_path.exists():
            with open(b_path) as f:
                b_data = json.load(f)
            for m, vals in b_data.items():
                acc_rows.append({"Dataset": ds.upper(), "Model": m, "Type": "Classical", **vals})

        # Use the current shared-protocol Chronos calibration run for point
        # metrics. The separate context sweep may be historical or incomplete.
        c_path = RESULTS_DIR / ds / "calibration.json"
        if c_path.exists():
            with open(c_path) as f:
                c_data = json.load(f)
            chronos = c_data.get("chronos_small_512", {})
            if all(key in chronos for key in ("MAE", "RMSE", "MAPE")):
                acc_rows.append({
                    "Dataset": ds.upper(),
                    "Model": "Chronos-T5-Small (512-observations)",
                    "Type": "Zero-Shot TSFM",
                    "MAE": chronos["MAE"],
                    "RMSE": chronos["RMSE"],
                    "MAPE": chronos["MAPE"],
                })

    if acc_rows:
        df_acc = pd.DataFrame(acc_rows).set_index(["Dataset", "Model"])
        acc_out = RESULTS_DIR / "master_table.csv"
        df_acc.to_csv(acc_out)
        print(f"Saved: {acc_out}")
        print(df_acc.to_string())

    # Table 2: Probabilistic calibration metrics
    cal_rows = []
    for ds in ["ercot", "gefcom"]:
        c_path = RESULTS_DIR / ds / "calibration.json"
        if c_path.exists():
            with open(c_path) as f:
                c_data = json.load(f)
            for m, vals in c_data.items():
                cal_rows.append({
                    "Dataset": ds.upper(),
                    "Model": m,
                    "MAE": vals["MAE"],
                    "RMSE": vals["RMSE"],
                    "MAPE (%)": vals["MAPE"],
                    "CRPS": vals["CRPS"],
                    "PICP 90% (%)": vals["PICP_90"],
                    "Width 90% (MW)": vals["Width_90"],
                    "ECE": vals["ECE"],
                })

    if cal_rows:
        df_cal = pd.DataFrame(cal_rows).set_index(["Dataset", "Model"])
        cal_out = RESULTS_DIR / "calibration_master.csv"
        df_cal.to_csv(cal_out)
        print(f"\nSaved: {cal_out}")
        print(df_cal.to_string())


def main():
    parser = argparse.ArgumentParser(description="GridEval benchmark experiment runner")
    parser.add_argument("--experiment", "-e", type=str, default=None, help="Run a specific experiment by name")
    parser.add_argument("--skip-existing", action="store_true", help="Skip experiments that have already completed")
    parser.add_argument("--compile-only", action="store_true", help="Only compile CSV tables from existing JSON outputs")
    args = parser.parse_args()

    print(f"Starting experiments at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if args.compile_only:
        build_master_tables()
        return

    set_all_seeds(42)

    for name, config_file, pipeline_module_path in get_experiment_registry():
        if args.experiment and name != args.experiment:
            continue

        cfg_path = Path(config_file)
        if not cfg_path.exists():
            print(f"Warning: config not found: {cfg_path}. Skipping.")
            continue
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        run_file = RESULTS_DIR / f"{name}_run.json"
        if args.skip_existing and run_file.exists():
            try:
                with open(run_file) as f:
                    r_data = json.load(f)
                is_current = (
                    r_data.get("status") == "SUCCESS"
                    and r_data.get("protocol_version") == RUN_PROTOCOL_VERSION
                    and r_data.get("config") == cfg
                )
                if is_current:
                    print(f"Skipping completed current-protocol experiment: {name}")
                    continue
                print(f"Re-running stale or incompatible result: {name}")
            except (OSError, json.JSONDecodeError):
                print(f"Re-running experiment with unreadable result metadata: {name}")

        pipeline_fn = get_pipeline_fn(pipeline_module_path)
        execute_experiment(name, cfg, pipeline_fn)

    build_master_tables()
    print("\nBenchmark runs finished.")


if __name__ == "__main__":
    main()
