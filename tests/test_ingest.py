import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import ingest_ercot as ingest_module
from ingest_ercot import ingest_ercot as run_ingest_ercot
from src.data_loader import load_dataset


class TestIngest(unittest.TestCase):
    def _write_raw_fixture(self, raw_dir: Path) -> None:
        """Create a small source fixture that exercises parsing, DST folds, and audit handling."""
        raw_dir.mkdir(parents=True, exist_ok=True)
        yearly_rows = {
            "2016": [
                {"TIME": "2016-11-07 00:00", "ERCOT": None, "ZONE_NORTH": None, "ZONE_SOUTH": None},
            ],
            "2020": [
                {"TIME": "2020-01-01 00:00", "ERCOT": 100.0, "ZONE_NORTH": 60.0, "ZONE_SOUTH": 40.0},
                {"TIME": "2020-06-01 24:00", "ERCOT": 110.0, "ZONE_NORTH": 65.0, "ZONE_SOUTH": 45.0},
                {"TIME": "2022-12-31 23:00", "ERCOT": 120.0, "ZONE_NORTH": 70.0, "ZONE_SOUTH": 50.0},
            ],
            "2023": [
                {"TIME": "2023-01-01 00:00", "ERCOT": 130.0, "ZONE_NORTH": 75.0, "ZONE_SOUTH": 55.0},
                {"TIME": "2023-12-31 23:00", "ERCOT": 140.0, "ZONE_NORTH": 80.0, "ZONE_SOUTH": 60.0},
            ],
            "2024": [
                {"TIME": "2024-11-03 02:00", "ERCOT": 150.0, "ZONE_NORTH": 85.0, "ZONE_SOUTH": 65.0},
                {"TIME": "2024-11-03 02:00", "ERCOT": 155.0, "ZONE_NORTH": 87.0, "ZONE_SOUTH": 68.0},
                {"TIME": "2024-11-04 24:00", "ERCOT": 160.0, "ZONE_NORTH": 90.0, "ZONE_SOUTH": 70.0},
            ],
        }
        for year, rows in yearly_rows.items():
            path = raw_dir / f"{year}_ercot_hourly_load_data.csv"
            pd.DataFrame(rows, columns=["TIME", "ERCOT", "ZONE_NORTH", "ZONE_SOUTH"]).to_csv(path, index=False)

    def test_source_to_output_strict_fidelity_ordered_splits_and_audit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            raw_dir = tmp_path / "raw"
            self._write_raw_fixture(raw_dir)
            out_parquet = tmp_path / "ercot_clean_test.parquet"
            out_audit = tmp_path / "ercot_missing_audit_test.json"
            splits_dir = tmp_path / "splits"

            with patch.object(ingest_module, "RAW_DIR", raw_dir):
                run_ingest_ercot(
                    output_parquet=out_parquet,
                    output_audit=out_audit,
                    output_splits_dir=splits_dir,
                )

            self.assertTrue(out_parquet.exists())
            self.assertTrue(out_audit.exists())
            with out_audit.open() as audit_file:
                audit = json.load(audit_file)
            self.assertEqual(len(audit), 1)
            record = audit[0]
            self.assertEqual(record["dataset"], "ercot")
            self.assertEqual(record["source_file"], "2016_ercot_hourly_load_data.csv")
            self.assertEqual(record["source_row"], 0)
            self.assertEqual(record["source_order"], 0)
            self.assertEqual(record["identity"], ["2016-11-07 00:00:00", 0])
            self.assertIn("all load and zone columns missing", record["reason"])

            raw_files = sorted(raw_dir.glob("*_ercot_hourly_load_data.csv"))
            output = pd.read_parquet(out_parquet)
            global_order = 0
            for raw_path in raw_files:
                raw = pd.read_csv(raw_path)
                time_text = raw["TIME"].astype(str).str.replace(" DST", "", regex=False).str.strip()
                is_24 = time_text.str.contains("24:00")
                parsed_time = pd.to_datetime(time_text.str.replace(" 24:00", " 00:00"), format="mixed")
                parsed_time[is_24] += pd.Timedelta(days=1)
                expected_occurrence = pd.Series(parsed_time).groupby(pd.Series(parsed_time)).cumcount().to_numpy()

                observed = output[output["source_file"] == raw_path.name].sort_values("source_row")
                self.assertEqual(len(observed), len(raw))
                np.testing.assert_array_equal(observed["source_row"].to_numpy(), np.arange(len(raw)))
                np.testing.assert_array_equal(observed["source_order"].to_numpy(), np.arange(global_order, global_order + len(raw)))

                for source_column in raw.columns:
                    if source_column == "TIME":
                        continue
                    output_column = "load" if source_column == "ERCOT" else source_column
                    self.assertIn(output_column, observed.columns)
                    source_values = pd.to_numeric(
                        raw[source_column].astype(str).str.replace(",", "", regex=False).str.strip(),
                        errors="coerce",
                    )
                    output_values = observed[output_column]
                    np.testing.assert_array_equal(output_values.isna().to_numpy(), source_values.isna().to_numpy())
                    valid = ~source_values.isna().to_numpy()
                    if valid.any():
                        np.testing.assert_allclose(output_values.to_numpy()[valid], source_values.to_numpy()[valid])

                np.testing.assert_array_equal(
                    pd.to_datetime(observed.index.get_level_values("timestamp")),
                    pd.to_datetime(parsed_time),
                )
                np.testing.assert_array_equal(observed.index.get_level_values("occurrence"), expected_occurrence)
                global_order += len(raw)

            duplicate_time = pd.Timestamp("2024-11-03 02:00")
            duplicate_rows = output.loc[duplicate_time]
            self.assertEqual(len(duplicate_rows), 2)
            self.assertEqual(list(duplicate_rows.index), [0, 1])
            self.assertNotEqual(duplicate_rows["load"].iloc[0], duplicate_rows["load"].iloc[1])

            expected_ranges = {
                "train": (pd.Timestamp("2020-01-01"), pd.Timestamp("2022-12-31 23:00")),
                "val": (pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31 23:00")),
                "test": (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31 23:00")),
            }
            timestamp_level = output.index.get_level_values("timestamp")
            split_lists = {}
            for split_name in ("train", "val", "test"):
                split_path = splits_dir / f"ercot_{split_name}_idx.parquet"
                split_df = pd.read_parquet(split_path)
                split_lists[split_name] = list(zip(split_df["timestamp"], split_df["occurrence"]))
                start, end = expected_ranges[split_name]
                expected = list(output.index[(timestamp_level >= start) & (timestamp_level <= end)])
                self.assertEqual(split_lists[split_name], expected)

            self.assertTrue(set(split_lists["train"]).isdisjoint(split_lists["val"]))
            self.assertTrue(set(split_lists["train"]).isdisjoint(split_lists["test"]))
            self.assertTrue(set(split_lists["val"]).isdisjoint(split_lists["test"]))
            assigned = split_lists["train"] + split_lists["val"] + split_lists["test"]
            modeled = [identity for identity in output.index if identity[0].year in (2020, 2021, 2022, 2023, 2024)]
            self.assertEqual(set(assigned), set(modeled))

    def test_empty_raw_input_handling(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            empty_raw = tmp_path / "empty_raw"
            empty_raw.mkdir()
            with patch.object(ingest_module, "RAW_DIR", empty_raw):
                with self.assertRaisesRegex(ValueError, "No yearly ERCOT CSV files"):
                    run_ingest_ercot(
                        output_parquet=tmp_path / "out.parquet",
                        output_audit=tmp_path / "audit.json",
                        output_splits_dir=tmp_path / "splits",
                    )

    def test_readonly_loader(self):
        df = load_dataset("ercot")
        self.assertGreater(len(df), 0)


if __name__ == "__main__":
    unittest.main()
