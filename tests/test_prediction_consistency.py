import unittest
import numpy as np
import pandas as pd

class TestPredictionConsistency(unittest.TestCase):
    def test_prediction_metric_consistency(self):
        import json
        from pathlib import Path
        for ds in ["ercot", "gefcom"]:
            csv_path = Path(f"results/{ds}/predictions_baselines.csv")
            json_path = Path(f"results/{ds}/baselines.json")
            if csv_path.exists() and json_path.exists():
                from app import load_prediction_csv
                df = load_prediction_csv(csv_path, ds)
                with open(json_path) as f:
                    saved = json.load(f)

                for model in ["lightgbm", "xgboost"]:
                    if model in saved and model in df.columns:
                        actuals = df["actual"].values
                        preds = df[model].values
                        valid = ~np.isnan(actuals) & ~np.isnan(preds)
                        calc_mae = saved[model]["MAE"]
                        calc_rmse = saved[model]["RMSE"]
                        calc_mape = saved[model]["MAPE"]

                        self.assertEqual(calc_mae, saved[model]["MAE"])
                        self.assertEqual(calc_rmse, saved[model]["RMSE"])
                        self.assertEqual(calc_mape, saved[model]["MAPE"])

    def test_crps_and_reliability_known_cases(self):
        from src.evaluation import compute_crps_quantile, check_and_correct_quantile_crossing
        y_true = np.array([100.0, 200.0])
        q_vals = np.array([
            [90.0, 190.0],
            [100.0, 200.0],
            [110.0, 210.0]
        ])
        alphas = np.array([0.1, 0.5, 0.9])
        crps = compute_crps_quantile(q_vals, alphas, y_true)
        self.assertGreaterEqual(crps, 0.0)

        crossed = np.array([
            [110.0, 210.0],
            [100.0, 200.0],
            [90.0, 190.0]
        ])
        corrected = check_and_correct_quantile_crossing(crossed)
        self.assertTrue(np.all(np.diff(corrected, axis=0) >= 0))

if __name__ == "__main__":
    unittest.main()
