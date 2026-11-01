import unittest
import numpy as np
import pandas as pd

from src.evaluation import compute_point_metrics, get_rolling_schedule

class TestBaselines(unittest.TestCase):
    def test_compute_point_metrics_exact(self):
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 190.0, 310.0])
        m = compute_point_metrics(y_true, y_pred)
        self.assertEqual(m["MAE"], 10.0)
        self.assertEqual(m["RMSE"], 10.0)

    def test_get_rolling_schedule_ercot_multiindex_with_gap(self):
        ts = pd.date_range("2024-01-01", periods=10, freq="h")
        multi_idx = pd.MultiIndex.from_tuples([(t, 0) for t in ts], names=["timestamp", "occurrence"])
        sched = get_rolling_schedule(multi_idx, horizon=4, step=4)
        self.assertGreater(len(sched), 0)

    def test_dual_evaluator_identity_equivalence(self):
        try:
            import torch
            from src.tsfm_models import rolling_tsfm_evaluation
        except Exception as exc:
            self.skipTest(f"Chronos integration runtime is unavailable: {exc}")

        from src.baselines import rolling_block_evaluation
        from src.evaluation import get_rolling_schedule

        class FakeChronosPipeline:
            def __init__(self):
                self.contexts_seen = []
            def predict(self, inputs, prediction_length, num_samples=20, **kwargs):
                self.contexts_seen.append(inputs.detach().cpu().numpy() if hasattr(inputs, "detach") else np.array(inputs))
                batch_size = inputs.shape[0] if hasattr(inputs, "shape") else 1
                return torch.zeros((batch_size, num_samples, prediction_length))

        class RecordingGBDTModel:
            def __init__(self):
                self.inputs_seen = []
            def fit(self, X, y, eval_set=None, verbose=False):
                return self
            def predict(self, X):
                self.inputs_seen.append(X.copy())
                return np.full(len(X), 42.0)

        dates = pd.date_range("2024-01-01", periods=50, freq="h")
        load_vals = np.linspace(1000, 3000, 50)
        df = pd.DataFrame({
            "load": load_vals,
            "hour": [d.hour for d in dates],
            "dayofweek": 0, "month": 1, "day": 1,
            "lag_1": 0.0, "lag_2": 0.0, "lag_3": 0.0, "lag_24": 0.0, "lag_168": 0.0,
            "rolling_mean_24": 2000.0, "rolling_std_24": 100.0,
            "rolling_mean_168": 2000.0, "rolling_std_168": 100.0
        }, index=dates)

        X_dummy = df[["hour", "dayofweek", "month", "day", "lag_1", "lag_2", "lag_3", "lag_24", "lag_168", "rolling_mean_24", "rolling_std_24", "rolling_mean_168", "rolling_std_168"]]
        y_dummy = df["load"]

        train_df = df.iloc[:30]
        test_df = df.iloc[30:]
        test_idx = test_df.index

        gbdt_model = RecordingGBDTModel()
        gbdt_res = rolling_block_evaluation(
            gbdt_model, X_dummy.iloc[:30], y_dummy.iloc[:30], X_dummy.iloc[:30], y_dummy.iloc[:30],
            df, "load", list(X_dummy.columns), test_idx, horizon=10, step=10
        )

        history = train_df["load"].values
        test_vals = test_df["load"].values
        pipeline = FakeChronosPipeline()
        tsfm_res = rolling_tsfm_evaluation(
            pipeline=pipeline,
            history_values=history,
            test_values=test_vals,
            context_len=20,
            horizon=10,
            step=10,
            num_samples=5,
            batch_size=1,
            test_index=test_idx,
        )

        self.assertEqual(gbdt_res["schedule"], tsfm_res["schedule"])
        self.assertEqual(list(gbdt_res["test_identities"]), list(tsfm_res["test_identities"]))

        self.assertEqual(len(pipeline.contexts_seen), len(gbdt_res["schedule"]))
        for block_idx, block in enumerate(gbdt_res["schedule"]):
            origin_ident = block["origin_identity"]
            origin_loc = df.index.get_loc(origin_ident)
            ctx_seen = pipeline.contexts_seen[block_idx]
            expected_slice_end = origin_loc
            expected_slice_start = max(0, expected_slice_end - 20)
            expected_vals = df["load"].iloc[expected_slice_start:expected_slice_end].values
            actual_tail = ctx_seen.flatten()[-len(expected_vals):]
            self.assertTrue(np.allclose(actual_tail, expected_vals))

            if block_idx < len(gbdt_model.inputs_seen):
                gbdt_input_row = gbdt_model.inputs_seen[block_idx * 10]
                expected_lag1 = df["load"].iloc[origin_loc - 1]
                self.assertTrue(np.isclose(gbdt_input_row["lag_1"].iloc[0], expected_lag1))

    def test_multiblock_rolling_block_evaluation(self):
        from src.baselines import rolling_block_evaluation
        class RecordingDummyModel:
            def __init__(self):
                self.inputs_seen = []
            def fit(self, X, y, eval_set=None, verbose=False):
                return self
            def predict(self, X):
                self.inputs_seen.append(X.copy())
                return np.array([999.0])
        dates = pd.date_range("2024-01-01", periods=30, freq="h")
        load_vals = np.linspace(1000, 4000, 30)
        df = pd.DataFrame({
            "load": load_vals,
            "hour": [d.hour for d in dates],
            "dayofweek": 0, "month": 1, "day": 1,
            "lag_1": 0.0, "lag_2": 0.0, "lag_3": 0.0, "lag_24": 0.0, "lag_168": 0.0,
            "rolling_mean_24": 2000.0, "rolling_std_24": 100.0,
            "rolling_mean_168": 2000.0, "rolling_std_168": 100.0
        }, index=dates)
        X_dummy = df[["hour", "dayofweek", "month", "day", "lag_1", "lag_2", "lag_3", "lag_24", "lag_168", "rolling_mean_24", "rolling_std_24", "rolling_mean_168", "rolling_std_168"]]
        y_dummy = df["load"]
        test_idx = dates[10:16]
        dummy_model = RecordingDummyModel()
        res = rolling_block_evaluation(
            dummy_model, X_dummy, y_dummy, X_dummy, y_dummy, df, "load", list(X_dummy.columns), test_idx, horizon=2, step=2
        )
        self.assertEqual(len(res["y_pred"]), 6)

        inputs = dummy_model.inputs_seen
        self.assertTrue(np.isclose(inputs[0]["lag_1"].iloc[0], load_vals[9]))
        self.assertTrue(np.isclose(inputs[1]["lag_1"].iloc[0], 999.0))
        self.assertTrue(np.isclose(inputs[2]["lag_1"].iloc[0], load_vals[11]))
        self.assertTrue(np.isclose(inputs[3]["lag_1"].iloc[0], 999.0))
        self.assertTrue(np.isclose(inputs[4]["lag_1"].iloc[0], load_vals[13]))
        self.assertTrue(np.isclose(inputs[5]["lag_1"].iloc[0], 999.0))

if __name__ == "__main__":
    unittest.main()
