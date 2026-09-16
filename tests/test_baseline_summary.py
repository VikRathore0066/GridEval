import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline_summary import print_baseline_table


def test_print_baseline_table_execution(capsys=None):
    if capsys is not None:
        print_baseline_table()
        captured = capsys.readouterr()
        assert "Baseline Point Accuracy Summary:" in captured.out
        assert "ERCOT" in captured.out
        assert "GEFCOM" in captured.out
        assert "lightgbm" in captured.out
    else:
        print_baseline_table()


if __name__ == "__main__":
    test_print_baseline_table_execution()
    print("All baseline summary tests passed.")
