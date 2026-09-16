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
from src.experiments import execute_experiment, RESULTS_DIR
from src.pipelines import baselines, tsfm, calibration

CONFIGS_DIR = PROJECT_ROOT / "configs"

EXPERIMENT_REGISTRY = [
    ("ercot_baselines", CONFIGS_DIR / "ercot_baselines.yaml", baselines.run),
    ("gefcom_baselines", CONFIGS_DIR / "gefcom_baselines.yaml", baselines.run),
    ("ercot_tsfm", CONFIGS_DIR / "ercot_tsfm.yaml", tsfm.run),
    ("gefcom_tsfm", CONFIGS_DIR / "gefcom_tsfm.yaml", tsfm.run),
    ("ercot_calibration", CONFIGS_DIR / "ercot_calibration.yaml", calibration.run),
    ("gefcom_calibration", CONFIGS_DIR / "gefcom_calibration.yaml", calibration.run),
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

        t_path = RESULTS_DIR / ds / "tsfm_context_sweep.json"
        if t_path.exists():
            with open(t_path) as f:
                t_data = json.load(f)
            if "512" in t_data or 512 in t_data:
                m512 = t_data.get("512") or t_data.get(512)
                acc_rows.append({
                    "Dataset": ds.upper(),
                    "Model": "Chronos-T5-Small (512h)",
                    "Type": "Zero-Shot TSFM",
                    "MAE": m512["MAE"],
                    "RMSE": m512["RMSE"],
                    "MAPE": m512["MAPE"],
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

    for name, config_file, pipeline_fn in EXPERIMENT_REGISTRY:
        if args.experiment and name != args.experiment:
            continue

        run_file = RESULTS_DIR / f"{name}_run.json"
        if args.skip_existing and run_file.exists():
            try:
                with open(run_file) as f:
                    r_data = json.load(f)
                if r_data.get("status") == "SUCCESS":
                    print(f"Skipping completed experiment: {name}")
                    continue
            except Exception:
                pass

        cfg_path = Path(config_file)
        if not cfg_path.exists():
            print(f"Warning: config not found: {cfg_path}. Skipping.")
            continue
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        execute_experiment(name, cfg, pipeline_fn)

    build_master_tables()
    print("\nBenchmark runs finished.")


if __name__ == "__main__":
    main()
