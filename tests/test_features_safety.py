import unittest
import numpy as np
import pandas as pd

class TestFeaturesSafety(unittest.TestCase):
    def test_feature_exclusions(self):
        from src.features import get_feature_columns
        base_feats = get_feature_columns("gefcom", include_weather=False)
        self.assertNotIn("drybulb", base_feats)
        self.assertNotIn("dewpoint", base_feats)

        oracle_feats = get_feature_columns("gefcom", include_weather=True)
        self.assertIn("drybulb", oracle_feats)

if __name__ == "__main__":
    unittest.main()
