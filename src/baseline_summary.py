import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd


def print_baseline_table():
    rows = []
    results_dir = PROJECT_ROOT / "results"
    for ds in ["ercot", "gefcom"]:
        path = results_dir / ds / "baselines.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            for model, metrics in data.items():
                rows.append({"Dataset": ds.upper(), "Model": model, **metrics})
        else:
            print(f"Warning: {path} does not exist yet. Run src/baselines.py first.")

    if rows:
        df = pd.DataFrame(rows).set_index(["Dataset", "Model"])
        print("\nBaseline Point Accuracy Summary:")
        print(df.to_string())
    else:
        print("\nNo baseline results found. Run src/baselines.py first.")


if __name__ == "__main__":
    print_baseline_table()
