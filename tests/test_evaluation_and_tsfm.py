import unittest
import numpy as np
import pandas as pd

class TestEvaluationAndTsfm(unittest.TestCase):
    def test_evaluation_metrics(self):
        from src.evaluation import (
            compute_point_metrics,
            compute_crps_empirical,
            compute_crps_quantile,
            compute_picp,
            compute_interval_width,
            compute_ece_and_reliability,
        )
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([105.0, 195.0, 310.0])

        pt = compute_point_metrics(y_true, y_pred)
        self.assertAlmostEqual(pt["MAE"], 6.67, delta=0.05)
        self.assertIn("RMSE", pt)
        self.assertIn("MAPE", pt)

        samples = np.array([
            [95.0, 190.0, 290.0],
            [100.0, 200.0, 300.0],
            [105.0, 210.0, 310.0],
        ])
        crps = compute_crps_empirical(samples, y_true)
        self.assertGreaterEqual(crps, 0.0)

        q_values = np.array([
            [95.0, 190.0, 290.0],
            [100.0, 200.0, 300.0],
            [105.0, 210.0, 310.0],
        ])
        alpha_values = np.array([0.1, 0.5, 0.9])
        crps_q = compute_crps_quantile(q_values, alpha_values, y_true)
        self.assertGreaterEqual(crps_q, 0.0)

        lower = np.array([90.0, 190.0, 290.0])
        upper = np.array([110.0, 210.0, 310.0])
        picp = compute_picp(y_true, lower, upper)
        width = compute_interval_width(lower, upper)
        self.assertEqual(picp, 100.0)
        self.assertEqual(width, 20.0)

        alphas_grid = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        q_grid = np.array([np.percentile(samples, a * 100.0, axis=0) for a in alphas_grid])
        ece, nominal, empirical = compute_ece_and_reliability(q_grid, alphas_grid, y_true)
        self.assertTrue(0.0 <= ece <= 1.0)
        self.assertEqual(len(nominal), 5)
        self.assertEqual(len(empirical), 5)

    def test_data_loader_and_splits(self):
        from src.data_loader import load_dataset, split_dataset, get_target_col
        for name in ["ercot", "gefcom"]:
            df = load_dataset(name)
            self.assertGreater(len(df), 0)
            train, val, test = split_dataset(df, name)
            self.assertGreater(len(train), 0)
            self.assertGreater(len(val), 0)
            self.assertGreater(len(test), 0)
            self.assertLess(train.index.max(), val.index.min())
            self.assertLess(val.index.max(), test.index.min())
            self.assertEqual(get_target_col(name), "load")

if __name__ == "__main__":
    unittest.main()
