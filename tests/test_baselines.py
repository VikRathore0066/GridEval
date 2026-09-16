import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest
from src.baselines import compute_point_metrics


def test_compute_point_metrics_exact():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([100.0, 200.0, 300.0])
    metrics = compute_point_metrics(y_true, y_pred)
    assert metrics["MAE"] == 0.0
    assert metrics["RMSE"] == 0.0
    assert metrics["MAPE"] == 0.0


def test_compute_point_metrics_deviation():
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([110.0, 180.0])
    metrics = compute_point_metrics(y_true, y_pred)
    assert metrics["MAE"] == 15.0
    assert metrics["RMSE"] == pytest.approx(15.81, rel=1e-2)
    assert metrics["MAPE"] == 10.0


if __name__ == "__main__":
    test_compute_point_metrics_exact()
    test_compute_point_metrics_deviation()
    print("All baseline calculation tests passed.")
