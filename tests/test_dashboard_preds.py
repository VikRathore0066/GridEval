import os
import json
import unittest
from pathlib import Path
import pandas as pd
import numpy as np
import tempfile

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import (
    filter_comparison_models,
    load_grid_load_data,
    load_forecast_window,
    verify_prediction_alignment,
    load_prediction_csv,
)
from src.data_loader import load_dataset, split_dataset

class TestDashboardFixes(unittest.TestCase):
    def test_removed_model_rows_are_not_shown(self):
        history = pd.DataFrame(
            {
                "Model": ["lightgbm", "naive_seasonal_1wk", "prophet", "chronos_small_512"],
                "MAE": [1.0, 8.0, 99.0, 3.0],
            }
        )
        visible = filter_comparison_models(history)
        self.assertEqual(visible["Model"].tolist(), ["lightgbm", "chronos_small_512"])
        self.assertEqual(
            history["Model"].tolist(),
            ["lightgbm", "naive_seasonal_1wk", "prophet", "chronos_small_512"],
        )

    def test_saved_ercot_quantile_forecast_matches_current_test_split(self):
        forecast = load_forecast_window(
            "ercot",
            "lightgbm_quantile",
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-12-31 23:00"),
        )
        self.assertTrue(forecast["has_quantiles"])
        self.assertEqual(len(forecast["index"]), 8784)
        self.assertEqual(len(forecast["actual"]), 8784)

    def test_ercot_multiindex_loading(self):
        df = load_grid_load_data("ercot")
        if df is not None and not df.empty:
            self.assertIsInstance(df.index, pd.MultiIndex)
            self.assertIn("timestamp", df.index.names)
            self.assertIn("occurrence", df.index.names)

    def test_prediction_alignment_valid_gefcom_roundtrip(self):
        df_full = load_dataset("gefcom")
        _, _, test_split = split_dataset(df_full, "gefcom")
        sub = test_split.head(10).copy()
        sub["actual"] = sub["load"]
        sub["lightgbm_median"] = sub["load"]

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "gefcom_preds.csv"
            sub.to_csv(csv_path)
            loaded_df = load_prediction_csv(csv_path, "gefcom")
            self.assertIsInstance(loaded_df.index, pd.DatetimeIndex)
            res = verify_prediction_alignment(loaded_df, "gefcom", loaded_df.index[0], loaded_df.index[-1])
            self.assertTrue(res)

    def test_prediction_alignment_reordered_gefcom(self):
        df_full = load_dataset("gefcom")
        _, _, test_split = split_dataset(df_full, "gefcom")
        sub = test_split.head(5).copy()
        sub["actual"] = sub["load"]
        sub = sub.iloc[::-1]

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "gefcom_reordered.csv"
            sub.to_csv(csv_path)
            loaded_df = load_prediction_csv(csv_path, "gefcom")
            with self.assertRaises(ValueError):
                verify_prediction_alignment(loaded_df, "gefcom", loaded_df.index[-1], loaded_df.index[0])

    def test_prediction_alignment_invalid_actuals(self):
        df_full = load_dataset("gefcom")
        _, _, test_split = split_dataset(df_full, "gefcom")
        sub = test_split.head(5).copy()
        sub["actual"] = sub["load"] + 9999.0

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "gefcom_bad_actuals.csv"
            sub.to_csv(csv_path)
            loaded_df = load_prediction_csv(csv_path, "gefcom")
            with self.assertRaises(ValueError):
                verify_prediction_alignment(loaded_df, "gefcom", loaded_df.index[0], loaded_df.index[-1])

    def test_prediction_alignment_multiindex_ercot_roundtrip_with_dst(self):
        df_full = load_dataset("ercot")
        _, _, test_split = split_dataset(df_full, "ercot")

        ts_counts = test_split.index.get_level_values("timestamp").value_counts()
        repeated_ts = ts_counts[ts_counts > 1].index
        self.assertGreater(len(repeated_ts), 0, "No repeated ERCOT timestamp found in test split for DST round-trip test.")
        target_ts = repeated_ts[0]
        sub = test_split[test_split.index.get_level_values("timestamp") == target_ts].copy()

        sub["actual"] = sub["load"]
        sub["lightgbm_median"] = sub["load"]

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "ercot_preds.csv"
            sub.to_csv(csv_path)
            loaded_df = load_prediction_csv(csv_path, "ercot")
            self.assertIsInstance(loaded_df.index, pd.MultiIndex)
            self.assertIn("timestamp", loaded_df.index.names)
            self.assertIn("occurrence", loaded_df.index.names)
            res = verify_prediction_alignment(loaded_df, "ercot", loaded_df.index[0][0], loaded_df.index[-1][0])
            self.assertTrue(res)

    def test_full_year_csv_to_selected_window_integration(self):
        df_full = load_dataset("ercot")
        _, _, test_split = split_dataset(df_full, "ercot")

        # Simulate a full test year CSV being written out
        full_test_preds = test_split.copy()
        full_test_preds["actual"] = full_test_preds["load"]
        full_test_preds["lightgbm_median"] = full_test_preds["load"]

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "ercot_full_year_preds.csv"
            full_test_preds.to_csv(csv_path)

            loaded_full = load_prediction_csv(csv_path, "ercot")

            # Select a shorter date window within the test split
            all_timestamps = loaded_full.index.get_level_values("timestamp")
            start_w = all_timestamps[10]
            end_w = all_timestamps[20]

            res = verify_prediction_alignment(loaded_full, "ercot", start_w, end_w)
            self.assertTrue(res)

    def test_ercot_defect_rejections(self):
        df_full = load_dataset("ercot")
        _, _, test_split = split_dataset(df_full, "ercot")
        ts_counts = test_split.index.get_level_values("timestamp").value_counts()
        repeated_ts = ts_counts[ts_counts > 1].index
        if len(repeated_ts) > 0:
            target_ts = repeated_ts[0]
            sub = test_split[test_split.index.get_level_values("timestamp") == target_ts].copy()
        else:
            sub = test_split.head(2).copy()

        sub["actual"] = sub["load"]
        sub["lightgbm_median"] = sub["load"]
        start_t = sub.index[0][0]
        end_t = sub.index[-1][0]

        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Wrong occurrence (tamper with occurrence level)
            bad_occ = sub.copy()
            idx_unwrapped = bad_occ.index.to_frame(index=False)
            idx_unwrapped["occurrence"] = idx_unwrapped["occurrence"] + 99
            bad_occ.index = pd.MultiIndex.from_frame(idx_unwrapped)
            p1 = Path(tmpdir) / "bad_occ.csv"
            bad_occ.to_csv(p1)
            df1 = load_prediction_csv(p1, "ercot")
            with self.assertRaises(ValueError):
                verify_prediction_alignment(df1, "ercot", start_t, end_t)

            # 2. Missing expected rows (drop one row)
            bad_miss = sub.iloc[:-1].copy()
            p2 = Path(tmpdir) / "bad_miss.csv"
            bad_miss.to_csv(p2)
            df2 = load_prediction_csv(p2, "ercot")
            with self.assertRaises(ValueError):
                verify_prediction_alignment(df2, "ercot", start_t, end_t)

            # 3. Duplicate keys (duplicate first row)
            bad_dup = pd.concat([sub, sub.head(1)])
            p3 = Path(tmpdir) / "bad_dup.csv"
            bad_dup.to_csv(p3)
            df3 = load_prediction_csv(p3, "ercot")
            with self.assertRaises(ValueError):
                verify_prediction_alignment(df3, "ercot", start_t, end_t)

            # 4. Extra rows (append train row)
            train_df, _, _ = split_dataset(df_full, "ercot")
            bad_extra = pd.concat([sub, train_df.head(1)])
            p4 = Path(tmpdir) / "bad_extra.csv"
            bad_extra.to_csv(p4)
            df4 = load_prediction_csv(p4, "ercot")
            with self.assertRaises(ValueError):
                verify_prediction_alignment(df4, "ercot", start_t, end_t)

            # 5. Reordered rows
            if len(sub) > 1:
                bad_reord = sub.iloc[::-1].copy()
                p5 = Path(tmpdir) / "bad_reord.csv"
                bad_reord.to_csv(p5)
                df5 = load_prediction_csv(p5, "ercot")
                with self.assertRaises(ValueError):
                    verify_prediction_alignment(df5, "ercot", start_t, end_t)

            # 6. Incorrect actual values
            bad_act = sub.copy()
            bad_act["actual"] = bad_act["actual"] + 5000.0
            p6 = Path(tmpdir) / "bad_act.csv"
            bad_act.to_csv(p6)
            df6 = load_prediction_csv(p6, "ercot")
            with self.assertRaises(ValueError):
                verify_prediction_alignment(df6, "ercot", start_t, end_t)

    def test_chronos_forecast_is_present_and_aligned(self):
        chronos_csv = PROJECT_ROOT / "results" / "ercot" / "predictions_quantile_chronos.csv"
        self.assertTrue(chronos_csv.exists())
        predictions = load_prediction_csv(chronos_csv, "ercot")
        self.assertIn("chronos_q05", predictions.columns)
        self.assertIn("chronos_q95", predictions.columns)
        verify_prediction_alignment(
            predictions,
            "ercot",
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-12-31 23:00"),
        )

if __name__ == "__main__":
    unittest.main()
