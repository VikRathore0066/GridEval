
import json
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "datasets" / "v1" / "ercot_yearly"
RESULTS_DIR = PROJECT_ROOT / "results"

def ingest_ercot(output_parquet=None, output_audit=None, output_splits_dir=None):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Raw yearly ERCOT directory not found at {RAW_DIR}")

    dfs = []
    global_order = 0
    csv_files = sorted(RAW_DIR.glob("*_ercot_hourly_load_data.csv"))
    if not csv_files:
        raise ValueError(f"No yearly ERCOT CSV files found under {RAW_DIR}")
    for p in csv_files:
        d = pd.read_csv(p)
        d["source_file"] = p.name
        d["source_row"] = range(len(d))

        for col in d.columns:
            if col not in ["TIME", "source_file", "source_row"]:
                if d[col].dtype == object or pd.api.types.is_string_dtype(d[col]):
                    d[col] = pd.to_numeric(d[col].astype(str).str.replace(",", "").str.strip(), errors="coerce")
                else:
                    d[col] = pd.to_numeric(d[col], errors="coerce")

        time_str = d["TIME"].astype(str).str.replace(" DST", "", regex=False).str.strip()
        is_24 = time_str.str.contains("24:00")
        cleaned = time_str.str.replace(" 24:00", " 00:00")
        dt = pd.to_datetime(cleaned, format="mixed")
        dt[is_24] = dt[is_24] + pd.Timedelta(days=1)
        d["timestamp"] = dt
        d["source_order"] = range(global_order, global_order + len(d))
        global_order += len(d)

        d = d.drop(columns=["TIME"])
        dfs.append(d)

    combined = pd.concat(dfs, ignore_index=True)
    if "ERCOT" in combined.columns and "load" not in combined.columns:
        combined = combined.rename(columns={"ERCOT": "load"})

    # Occurrence identifier (fold/sequence ID per timestamp) preserving source order & ambiguity
    combined["occurrence"] = combined.groupby("timestamp").cumcount()
    combined = combined.set_index(["timestamp", "occurrence"]).sort_index()

    out_parquet = Path(output_parquet) if output_parquet else (PROCESSED_DIR / "ercot_clean.parquet")
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_parquet)
    print(f"Canonical ERCOT dataset written to {out_parquet}, shape: {combined.shape}")

    # Audit missing target rows with complete source provenance
    missing_mask = combined["load"].isna()
    audit_records = []
    if missing_mask.any():
        missing_rows = combined[missing_mask]
        for idx_val, row in missing_rows.iterrows():
            # Confirm raw record missing status directly from row data
            raw_vals = {c: row[c] for c in row.index if c not in ["source_file", "source_row", "source_order"]}
            all_missing = all(pd.isna(v) for v in raw_vals.values())
            reason_str = "Excluded due to unrecorded raw measurement (all load and zone columns missing in source data)" if all_missing else "Excluded due to unrecorded raw load measurement"
            audit_records.append({
                "dataset": "ercot",
                "source_file": row.get("source_file", "unknown"),
                "source_row": int(row.get("source_row", -1)),
                "source_order": int(row.get("source_order", -1)),
                "identity": list(idx_val) if isinstance(idx_val, tuple) else str(idx_val),
                "reason": reason_str
            })
    audit_path = Path(output_audit) if output_audit else (RESULTS_DIR / "ercot_missing_audit.json")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with open(audit_path, "w") as af:
        json.dump(audit_records, af, indent=2, default=str)
    print(f"Audit log written to {audit_path}")

    # Regenerate split indices
    splits_dir = Path(output_splits_dir) if output_splits_dir else PROCESSED_DIR
    splits_dir.mkdir(parents=True, exist_ok=True)

    train_df = combined.loc["2020-01-01 00:00:00":"2022-12-31 23:00:00"].reset_index()[["timestamp", "occurrence"]]
    val_df = combined.loc["2023-01-01 00:00:00":"2023-12-31 23:00:00"].reset_index()[["timestamp", "occurrence"]]
    test_df = combined.loc["2024-01-01 00:00:00":"2024-12-31 23:00:00"].reset_index()[["timestamp", "occurrence"]]

    train_df.to_parquet(splits_dir / "ercot_train_idx.parquet")
    val_df.to_parquet(splits_dir / "ercot_val_idx.parquet")
    test_df.to_parquet(splits_dir / "ercot_test_idx.parquet")
    print("Canonical split indices written successfully.")

if __name__ == "__main__":
    ingest_ercot()
