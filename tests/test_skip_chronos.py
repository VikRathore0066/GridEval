import os
import json
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch, Mock
import numpy as np
import pandas as pd

_RESULTS_TEMP = tempfile.TemporaryDirectory(prefix="grideval_skip_chronos_")
TEST_RESULTS_DIR = Path(_RESULTS_TEMP.name)
_OLD_SKIP = os.environ.get("GRIDEVAL_SKIP_CHRONOS")
_OLD_RESULTS_DIR = os.environ.get("GRIDEVAL_RESULTS_DIR")
os.environ["GRIDEVAL_SKIP_CHRONOS"] = "1"
os.environ["GRIDEVAL_RESULTS_DIR"] = str(TEST_RESULTS_DIR)
try:
    from src.run_calibration import run_calibration_suite
finally:
    if _OLD_SKIP is None:
        os.environ.pop("GRIDEVAL_SKIP_CHRONOS", None)
    else:
        os.environ["GRIDEVAL_SKIP_CHRONOS"] = _OLD_SKIP
    if _OLD_RESULTS_DIR is None:
        os.environ.pop("GRIDEVAL_RESULTS_DIR", None)
    else:
        os.environ["GRIDEVAL_RESULTS_DIR"] = _OLD_RESULTS_DIR

class TestSkipChronosFast(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        _RESULTS_TEMP.cleanup()

    @patch("src.run_calibration.load_dataset")
    @patch("src.run_calibration.split_dataset")
    @patch("src.run_calibration.prepare_features")
    @patch("src.run_calibration.get_feature_columns")
    @patch("src.run_calibration.get_target_col")
    @patch("src.run_calibration.rolling_quantile_block_evaluation")
    @patch("src.run_calibration.run_leakage_invariance_test")
    @patch("json.dump")
    @patch("pandas.DataFrame.to_csv")
    def test_skip_branch(self, mock_to_csv, mock_json_dump, mock_leakage, mock_rolling, mock_target, mock_feat_cols, mock_prep, mock_split, mock_load):
        dates = pd.date_range("2024-01-01", periods=10, freq="h")
        df = pd.DataFrame({"load": [100.0]*10, "f1": [1]*10}, index=dates)
        mock_load.return_value = df
        mock_target.return_value = "load"
        mock_feat_cols.return_value = ["f1"]
        mock_prep.return_value = df
        mock_split.return_value = (df, df, df)

        dummy_grid = np.zeros((11, 10))
        mock_rolling.return_value = {
            "quantiles_grid": dummy_grid,
            "y_true": np.zeros(10),
            "test_identities": list(dates),
            "schedule": []
        }
        mock_leakage.return_value = {"leakage_test_passed": True}

        mock_to_csv.reset_mock()
        mock_json_dump.reset_mock()

        import src
        fake_tsfm = ModuleType("src.tsfm_models")
        fake_tsfm.load_chronos_pipeline = Mock()
        fake_tsfm.rolling_tsfm_evaluation = Mock()
        with patch.dict("sys.modules", {"src.tsfm_models": fake_tsfm}), patch.object(
            src, "tsfm_models", fake_tsfm, create=True
        ), patch.dict(os.environ, {"GRIDEVAL_SKIP_CHRONOS": "1"}):
            res = run_calibration_suite("gefcom")

        # 1. Assert Chronos loader and inference function are both never called
        fake_tsfm.load_chronos_pipeline.assert_not_called()
        fake_tsfm.rolling_tsfm_evaluation.assert_not_called()

        # 2. Assert calibration.json write call count is exactly 1 via mock_json_dump
        self.assertEqual(mock_json_dump.call_count, 1, "calibration.json must be written exactly once")

        # Verify status in returned dict
        self.assertEqual(res["chronos_small_512"]["status"], "skipped_by_config")

        # 3. Assert mock CSV write counts (mocked DataFrame.to_csv calls)
        csv_names_called = [
            Path(str(call.args[0] if call.args else call.kwargs.get("path", ""))).name.casefold()
            for call in mock_to_csv.mock_calls
        ]
        lgb_writes = sum(1 for name in csv_names_called if "lightgbm" in name)
        xgb_writes = sum(1 for name in csv_names_called if "xgboost" in name)
        chronos_writes = sum(1 for name in csv_names_called if "chronos" in name)

        self.assertEqual(lgb_writes, 1, "LightGBM quantile predictions CSV write call expected once")
        self.assertEqual(xgb_writes, 1, "XGBoost quantile predictions CSV write call expected once")
        self.assertEqual(chronos_writes, 0, "Chronos quantile predictions CSV write call must not occur")

        print("All assertions passed successfully!")

if __name__ == "__main__":
    unittest.main()
