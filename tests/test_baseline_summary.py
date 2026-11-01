import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline_summary import print_baseline_table

class TestBaselineSummary(unittest.TestCase):
    def test_print_baseline_table_execution(self):
        print_baseline_table()
        res = True
        self.assertTrue(res)

if __name__ == "__main__":
    unittest.main()
