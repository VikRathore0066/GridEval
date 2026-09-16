import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest
from src.evaluation import (
    compute_point_metrics,
    compute_crps_empirical,
    compute_picp,
    compute_interval_width,
    compute_ece_and_reliability,
)
from src.data_loader import load_dataset, split_dataset, get_target_col


def test_evaluation_metrics():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([105.0, 195.0, 310.0])

    pt = compute_point_metrics(y_true, y_pred)
    assert pt["MAE"] == pytest.approx(6.67, abs=1e-2)
    assert "RMSE" in pt
    assert "MAPE" in pt

    # CRPS
    samples = np.array([
        [95.0, 190.0, 290.0],
        [100.0, 200.0, 300.0],
        [105.0, 210.0, 310.0],
    ])
    crps = compute_crps_empirical(samples, y_true)
    assert crps >= 0.0

    # PICP and width
    lower = np.array([90.0, 190.0, 290.0])
    upper = np.array([110.0, 210.0, 310.0])
    picp = compute_picp(y_true, lower, upper)
    width = compute_interval_width(lower, upper)
    assert picp == 100.0
    assert width == 20.0

    # ECE
    ece, nominal, empirical = compute_ece_and_reliability(samples, y_true, num_bins=5)
    assert 0.0 <= ece <= 1.0
    assert len(nominal) == 5
    assert len(empirical) == 5


def test_data_loader_and_splits():
    for name in ["ercot", "gefcom"]:
        df = load_dataset(name)
        assert len(df) > 0
        train, val, test = split_dataset(df, name)
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0
        assert train.index.max() < val.index.min()
        assert val.index.max() < test.index.min()
        assert get_target_col(name) == "load"


if __name__ == "__main__":
    test_evaluation_metrics()
    test_data_loader_and_splits()
    print("All evaluation and data loader tests passed.")
