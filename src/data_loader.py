from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DATASET_CONFIGS = {
    "ercot": {
        "file": "ercot_clean.parquet",
        "target_col": "load",
        "train_range": ("2020-01-01 00:00:00", "2022-12-31 23:00:00"),
        "val_range": ("2023-01-01 00:00:00", "2023-12-31 23:00:00"),
        "test_range": ("2024-01-01 00:00:00", "2024-12-31 23:00:00"),
    },
    "gefcom": {
        "file": "gefcom_ct_clean.parquet",
        "target_col": "load",
        "train_range": ("2012-01-01 00:00:00", "2014-12-31 23:00:00"),
        "val_range": ("2015-01-01 00:00:00", "2015-12-31 23:00:00"),
        "test_range": ("2016-01-01 00:00:00", "2016-12-31 23:00:00"),
    },
}


def load_dataset(name: str) -> pd.DataFrame:
    """Load cleaned parquet data for the requested dataset."""
    if name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset '{name}'. Expected 'ercot' or 'gefcom'.")
    path = PROCESSED_DIR / DATASET_CONFIGS[name]["file"]
    if not path.exists():
        raise FileNotFoundError(f"Processed data file not found: {path}")
    return pd.read_parquet(path)


def split_dataset(df: pd.DataFrame, dataset_name: str):
    """Split dataset chronologically into train, validation, and test subsets."""
    cfg = DATASET_CONFIGS[dataset_name]
    train = df.loc[cfg["train_range"][0]:cfg["train_range"][1]]
    val = df.loc[cfg["val_range"][0]:cfg["val_range"][1]]
    test = df.loc[cfg["test_range"][0]:cfg["test_range"][1]]

    assert train.index.max() < val.index.min(), f"Train and validation split overlap in {dataset_name}"
    assert val.index.max() < test.index.min(), f"Validation and test split overlap in {dataset_name}"

    return train, val, test


def get_target_col(dataset_name: str) -> str:
    """Return target column name for dataset."""
    return DATASET_CONFIGS[dataset_name]["target_col"]
