import os
import json
import unittest
from pathlib import Path
import yaml
import pandas as pd
import numpy as np

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

class TestCalibrationProtocolConsistency(unittest.TestCase):
    def test_yaml_configs_quantile_grid(self):
        for ds in ["ercot", "gefcom"]:
            cfg_path = PROJECT_ROOT / "configs" / f"{ds}_calibration.yaml"
            self.assertTrue(cfg_path.exists())
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            self.assertEqual(cfg.get("nominal_quantiles"), 11, f"Config {cfg_path} must specify nominal_quantiles: 11")

    def test_run_calibration_grid_match(self):
        from src.run_calibration import run_calibration_suite
        import inspect
        src_code = inspect.getsource(run_calibration_suite)
        self.assertIn("0.05, 0.1, 0.2", src_code)
        self.assertIn("len(nominal_grid)", src_code)

    def test_readme_reports_regenerated_results_and_open_limits(self):
        content = (PROJECT_ROOT / "README.md").read_text()
        self.assertIn("Regenerated Benchmark Results", content)
        self.assertIn("2,337.55", content)
        self.assertIn("None of the models meets the 90% coverage target.", content)
        self.assertIn("context-length sweep artifacts have not been regenerated", content)

if __name__ == "__main__":
    unittest.main()
