from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_engine.acquisition import (
    backfill_missing_quality_snapshots,
    build_download_csv_path,
    build_download_dataset_id,
    gap_fill_dataset_from_secondary,
    ingest_csv_to_store,
)
from backtest_engine.catalog import ResultCatalog
from backtest_engine.duckdb_store import DuckDBStore


class AcquisitionHelperTests(unittest.TestCase):
    def test_build_download_paths_are_stable(self) -> None:
        dataset_id = build_download_dataset_id("aapl", source="massive", history_window="2y", resolution="1m")
        self.assertEqual(dataset_id, "AAPL_massive_2y_1m")
        csv_path = build_download_csv_path("aapl", source="massive", history_window="2y", resolution="1m")
        self.assertEqual(csv_path, Path("data") / "AAPL_massive_2y_1m.csv")

    def test_ingest_csv_to_store_writes_parquet_and_reports_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            csv_path = root / "AAPL_massive_2y_1m.csv"
            frame = pd.DataFrame(
                [
                    {"timestamp": "2026-01-01T14:30:00Z", "open": 100, "high": 101, "low": 99, "close": 100.5, "volume": 10},
                    {"timestamp": "2026-01-01T14:31:00Z", "open": 100.5, "high": 102, "low": 100, "close": 101.25, "volume": 20},
                ]
            )
            frame.to_csv(csv_path, index=False)
            store = DuckDBStore(db_path=root / "history.duckdb", data_dir=root / "parquet")
            artifact = ingest_csv_to_store(csv_path, dataset_id="AAPL_massive_2y_1m", store=store)
            self.assertEqual(artifact.dataset_id, "AAPL_massive_2y_1m")
            self.assertEqual(artifact.bar_count, 2)
            self.assertTrue(Path(artifact.parquet_path).exists())
            described = store.describe_dataset("AAPL_massive_2y_1m")
            self.assertEqual(described["bar_count"], 2)
            self.assertEqual(described["coverage_start"], "2026-01-01T14:30:00+00:00")
            self.assertEqual(described["coverage_end"], "2026-01-01T14:31:00+00:00")

    def test_inspect_dataset_quality_flags_intraday_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = DuckDBStore(db_path=root / "history.duckdb", data_dir=root / "parquet")
            frame = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(
                        [
                            "2026-01-01T14:30:00Z",
                            "2026-01-01T14:31:00Z",
                            "2026-01-01T14:35:00Z",
                        ],
                        utc=True,
                    ),
                    "open": [100.0, 101.0, 102.0],
                    "high": [101.0, 102.0, 103.0],
                    "low": [99.0, 100.0, 101.0],
                    "close": [100.5, 101.5, 102.5],
                    "volume": [10, 20, 30],
                }
            )
            store.write_parquet("AAPL_massive_2y_1m", frame)
            quality = store.inspect_dataset_quality("AAPL_massive_2y_1m", "1m")
            self.assertEqual(quality["quality_state"], "gappy")
            self.assertEqual(quality["suspicious_gap_count"], 1)
            self.assertEqual(quality["max_suspicious_gap"], "0 days 00:04:00")
            self.assertEqual(quality["repair_request_start"], "2026-01-01")
            self.assertEqual(quality["repair_request_end"], "2026-01-01")
            self.assertEqual(len(quality["suspicious_gap_ranges"]), 1)

    def test_inspect_dataset_quality_ignores_cross_session_intraday_gap(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = DuckDBStore(db_path=root / "history.duckdb", data_dir=root / "parquet")
            frame = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(
                        [
                            "2026-01-02T00:00:00Z",  # 7:00 PM ET on the prior local trading day
                            "2026-01-02T14:31:00Z",  # 9:31 AM ET on the next local day
                            "2026-01-02T14:32:00Z",
                        ],
                        utc=True,
                    ),
                    "open": [100.0, 101.0, 101.5],
                    "high": [101.0, 102.0, 102.5],
                    "low": [99.0, 100.0, 101.0],
                    "close": [100.5, 101.5, 102.0],
                    "volume": [10, 20, 30],
                }
            )
            store.write_parquet("AAPL_interactive_brokers_10y_1m", frame)
            quality = store.inspect_dataset_quality("AAPL_interactive_brokers_10y_1m", "1m")
            self.assertEqual(quality["quality_state"], "gap_free")
            self.assertEqual(quality["suspicious_gap_count"], 0)

    def test_inspect_dataset_quality_can_classify_sparse_intraday(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = DuckDBStore(db_path=root / "history.duckdb", data_dir=root / "parquet")
            timestamps: list[pd.Timestamp] = []
            day_starts = [
                pd.Timestamp("2026-01-05T14:30:00Z"),
                pd.Timestamp("2026-01-06T14:30:00Z"),
                pd.Timestamp("2026-01-07T14:30:00Z"),
                pd.Timestamp("2026-01-08T14:30:00Z"),
            ]
            for day_idx, start in enumerate(day_starts):
                for minute in range(390):
                    if day_idx < 2 and minute in {120, 121, 122, 123, 240, 241, 242, 243}:
                        continue
                    timestamps.append(start + pd.Timedelta(minutes=minute))
            frame = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(timestamps, utc=True),
                    "open": np.linspace(100.0, 120.0, num=len(timestamps)),
                    "high": np.linspace(100.5, 120.5, num=len(timestamps)),
                    "low": np.linspace(99.5, 119.5, num=len(timestamps)),
                    "close": np.linspace(100.25, 120.25, num=len(timestamps)),
                    "volume": np.full(len(timestamps), 10),
                }
            )
            store.write_parquet("O_interactive_brokers_10y_1m", frame)
            quality = store.inspect_dataset_quality("O_interactive_brokers_10y_1m", "1m")
            self.assertEqual(quality["quality_state"], "sparse_intraday")
            self.assertGreaterEqual(int(quality["suspicious_gap_count"]), 2)
            self.assertIsNone(quality["repair_request_start"])
            self.assertIsNone(quality["repair_request_end"])

    def test_backfill_missing_quality_snapshots_persists_quality_for_existing_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            catalog = ResultCatalog(root / "catalog.sqlite")
            store = DuckDBStore(db_path=root / "history.duckdb", data_dir=root / "parquet")
            frame = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(
                        ["2026-01-01T14:30:00Z", "2026-01-01T14:31:00Z", "2026-01-01T14:35:00Z"],
                        utc=True,
                    ),
                    "open": [100.0, 101.0, 102.0],
                    "high": [101.0, 102.0, 103.0],
                    "low": [99.0, 100.0, 101.0],
                    "close": [100.5, 101.5, 102.5],
                    "volume": [10, 20, 30],
                }
            )
            parquet_path = store.write_parquet("AAPL_massive_2y_1m", frame)
            catalog.upsert_acquisition_dataset(
                dataset_id="AAPL_massive_2y_1m",
                source="massive",
                symbol="AAPL",
                resolution="1m",
                history_window="2y",
                parquet_path=str(parquet_path),
                coverage_start="2026-01-01T14:30:00+00:00",
                coverage_end="2026-01-01T14:35:00+00:00",
                bar_count=3,
                ingested=True,
                last_status="ingested",
            )
            updated = backfill_missing_quality_snapshots(catalog=catalog, store=store)
            self.assertEqual(updated, 1)
            record = catalog.load_acquisition_dataset("AAPL_massive_2y_1m")
            self.assertIsNotNone(record)
            self.assertEqual(record.quality_state, "gappy")
            self.assertIsNotNone(record.quality_analyzed_at)

    def test_ingest_csv_to_store_can_merge_partial_refresh_into_existing_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = DuckDBStore(db_path=root / "history.duckdb", data_dir=root / "parquet")
            base_csv = root / "AAPL_base.csv"
            pd.DataFrame(
                [
                    {"timestamp": "2026-01-01T14:30:00Z", "open": 100, "high": 101, "low": 99, "close": 100.5, "volume": 10},
                    {"timestamp": "2026-01-01T14:31:00Z", "open": 100.5, "high": 102, "low": 100, "close": 101.25, "volume": 20},
                ]
            ).to_csv(base_csv, index=False)
            ingest_csv_to_store(base_csv, dataset_id="AAPL_massive_2y_1m", store=store)

            refresh_csv = root / "AAPL_refresh.csv"
            pd.DataFrame(
                [
                    {"timestamp": "2026-01-01T14:31:00Z", "open": 100.75, "high": 102.25, "low": 100.1, "close": 101.5, "volume": 22},
                    {"timestamp": "2026-01-01T14:32:00Z", "open": 101.5, "high": 103, "low": 101, "close": 102.0, "volume": 25},
                ]
            ).to_csv(refresh_csv, index=False)
            artifact = ingest_csv_to_store(
                refresh_csv,
                dataset_id="AAPL_massive_2y_1m",
                store=store,
                merge_existing=True,
            )
            self.assertEqual(artifact.bar_count, 3)
            loaded = store.load("AAPL_massive_2y_1m")
            self.assertEqual(len(loaded), 3)
            self.assertAlmostEqual(float(loaded.loc[pd.Timestamp("2026-01-01T14:31:00Z"), "close"]), 101.5)

    def test_gap_fill_dataset_from_secondary_merges_missing_bars_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = DuckDBStore(db_path=root / "history.duckdb", data_dir=root / "parquet")
            primary = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(
                        ["2026-01-01T14:30:00Z", "2026-01-01T14:32:00Z"],
                        utc=True,
                    ),
                    "open": [100.0, 102.0],
                    "high": [101.0, 103.0],
                    "low": [99.0, 101.0],
                    "close": [100.5, 102.5],
                    "volume": [10, 30],
                }
            )
            secondary = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(
                        [
                            "2026-01-01T14:30:00Z",
                            "2026-01-01T14:31:00Z",
                            "2026-01-01T14:32:00Z",
                        ],
                        utc=True,
                    ),
                    "open": [100.0, 101.0, 102.0],
                    "high": [101.0, 102.0, 103.0],
                    "low": [99.0, 100.0, 101.0],
                    "close": [100.5, 101.5, 102.5],
                    "volume": [10, 20, 30],
                }
            )
            store.write_parquet("AAPL_massive_2y_1m", primary)
            store.write_parquet("AAPL_stooq_max_1m", secondary)
            artifact = gap_fill_dataset_from_secondary(
                "AAPL_massive_2y_1m",
                "AAPL_stooq_max_1m",
                store=store,
                start="2026-01-01",
                end="2026-01-01",
            )
            self.assertEqual(artifact.bar_count, 3)
            loaded = store.load("AAPL_massive_2y_1m")
            self.assertEqual(len(loaded), 3)
            self.assertAlmostEqual(float(loaded.loc[pd.Timestamp("2026-01-01T14:31:00Z"), "close"]), 101.5)
            self.assertAlmostEqual(float(loaded.loc[pd.Timestamp("2026-01-01T14:30:00Z"), "close"]), 100.5)

    def test_compare_dataset_parity_flags_matching_secondary_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = DuckDBStore(db_path=root / "history.duckdb", data_dir=root / "parquet")
            primary = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(
                        ["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", "2026-01-03T00:00:00Z"],
                        utc=True,
                    ),
                    "open": [100.0, 101.0, 102.0],
                    "high": [101.0, 102.0, 103.0],
                    "low": [99.0, 100.0, 101.0],
                    "close": [100.0, 101.0, 102.0],
                    "volume": [10, 20, 30],
                }
            )
            secondary = primary.copy()
            secondary["close"] = [100.05, 100.98, 102.03]
            store.write_parquet("AAPL_massive_2y_1d", primary)
            store.write_parquet("AAPL_stooq_max_1d", secondary)
            parity = store.compare_dataset_parity(
                "AAPL_massive_2y_1d",
                "AAPL_stooq_max_1d",
            )
            self.assertEqual(parity["parity_state"], "matching")
            self.assertEqual(parity["overlap_bar_count"], 3)
            self.assertLess(float(parity["close_mean_abs_bps"] or 0.0), 25.0)


if __name__ == "__main__":
    unittest.main()
