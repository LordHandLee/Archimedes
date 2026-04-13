from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from backtest_engine.acquisition_policy import (
    ACQUISITION_ACTION_GAP_FILL_SECONDARY,
    ACQUISITION_ACTION_DOWNLOAD,
    ACQUISITION_ACTION_INGEST_EXISTING,
    ACQUISITION_ACTION_SKIP_FRESH,
    ACQUISITION_PLAN_BACKFILL,
    ACQUISITION_PLAN_BACKFILL_GAP_REPAIR_REFRESH,
    ACQUISITION_PLAN_COMPOUND_MULTI_WINDOW_REFRESH,
    ACQUISITION_PLAN_CROSS_SOURCE_GAP_FILL,
    ACQUISITION_PLAN_CROSS_SOURCE_GAP_FILL_PLUS_INCREMENTAL_REFRESH,
    ACQUISITION_PLAN_GAP_REPAIR_REFRESH,
    ACQUISITION_PLAN_INCREMENTAL_GAP_REPAIR_REFRESH,
    ACQUISITION_PLAN_INCREMENTAL_REFRESH,
    ACQUISITION_PLAN_MULTI_WINDOW_GAP_REPAIR_REFRESH,
    decide_acquisition_policy,
)
from backtest_engine.catalog import ResultCatalog
from backtest_engine.duckdb_store import DuckDBStore


class AcquisitionPolicyTests(unittest.TestCase):
    def test_missing_dataset_downloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            decision = decide_acquisition_policy(
                "AAPL",
                source="massive",
                resolution="1m",
                history_window="2y",
                catalog=ResultCatalog(root / "catalog.sqlite"),
                store=DuckDBStore(db_path=root / "history.duckdb", data_dir=root / "parquet"),
                data_dir=root,
                now=pd.Timestamp("2026-04-09T12:00:00Z"),
            )
            self.assertEqual(decision.action, ACQUISITION_ACTION_DOWNLOAD)

    def test_existing_csv_without_ingest_prefers_ingest_existing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            csv_path = root / "AAPL_massive_2y_1m.csv"
            csv_path.write_text("timestamp,open,high,low,close,volume\n", encoding="utf-8")
            decision = decide_acquisition_policy(
                "AAPL",
                source="massive",
                resolution="1m",
                history_window="2y",
                catalog=ResultCatalog(root / "catalog.sqlite"),
                store=DuckDBStore(db_path=root / "history.duckdb", data_dir=root / "parquet"),
                data_dir=root,
                now=pd.Timestamp("2026-04-09T12:00:00Z"),
            )
            self.assertEqual(decision.action, ACQUISITION_ACTION_INGEST_EXISTING)

    def test_fresh_ingested_dataset_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            catalog = ResultCatalog(root / "catalog.sqlite")
            store = DuckDBStore(db_path=root / "history.duckdb", data_dir=root / "parquet")
            dataset_id = "AAPL_massive_1d_1m"
            frame = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(
                        ["2026-04-08T14:30:00Z", "2026-04-08T14:31:00Z"], utc=True
                    ),
                    "open": [1.0, 1.1],
                    "high": [1.2, 1.3],
                    "low": [0.9, 1.0],
                    "close": [1.1, 1.2],
                    "volume": [10, 12],
                }
            )
            parquet_path = store.write_parquet(dataset_id, frame)
            catalog.upsert_acquisition_dataset(
                dataset_id=dataset_id,
                source="massive",
                symbol="AAPL",
                resolution="1m",
                history_window="1d",
                parquet_path=str(parquet_path),
                coverage_start="2026-04-08T14:30:00+00:00",
                coverage_end="2026-04-08T14:31:00+00:00",
                bar_count=2,
                ingested=True,
                last_status="ingested",
            )
            decision = decide_acquisition_policy(
                "AAPL",
                source="massive",
                resolution="1m",
                history_window="1d",
                catalog=catalog,
                store=store,
                data_dir=root,
                now=pd.Timestamp("2026-04-09T12:00:00Z"),
            )
            self.assertEqual(decision.action, ACQUISITION_ACTION_SKIP_FRESH)

    def test_force_refresh_redownloads_even_if_dataset_is_fresh(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            catalog = ResultCatalog(root / "catalog.sqlite")
            store = DuckDBStore(db_path=root / "history.duckdb", data_dir=root / "parquet")
            dataset_id = "AAPL_massive_2y_1m"
            frame = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(
                        ["2026-04-08T14:30:00Z", "2026-04-08T14:31:00Z"], utc=True
                    ),
                    "open": [1.0, 1.1],
                    "high": [1.2, 1.3],
                    "low": [0.9, 1.0],
                    "close": [1.1, 1.2],
                    "volume": [10, 12],
                }
            )
            parquet_path = store.write_parquet(dataset_id, frame)
            catalog.upsert_acquisition_dataset(
                dataset_id=dataset_id,
                source="massive",
                symbol="AAPL",
                resolution="1m",
                history_window="2y",
                parquet_path=str(parquet_path),
                coverage_start="2026-04-08T14:30:00+00:00",
                coverage_end="2026-04-08T14:31:00+00:00",
                bar_count=2,
                ingested=True,
                last_status="ingested",
            )
            decision = decide_acquisition_policy(
                "AAPL",
                source="massive",
                resolution="1m",
                history_window="2y",
                catalog=catalog,
                store=store,
                data_dir=root,
                now=pd.Timestamp("2026-04-09T12:00:00Z"),
                force_refresh=True,
            )
            self.assertEqual(decision.action, ACQUISITION_ACTION_DOWNLOAD)

    def test_stale_dataset_redownloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            catalog = ResultCatalog(root / "catalog.sqlite")
            store = DuckDBStore(db_path=root / "history.duckdb", data_dir=root / "parquet")
            dataset_id = "AAPL_stooq_2y_1d"
            frame = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(["2026-03-01T00:00:00Z"], utc=True),
                    "open": [1.0],
                    "high": [1.1],
                    "low": [0.9],
                    "close": [1.05],
                    "volume": [100],
                }
            )
            parquet_path = store.write_parquet(dataset_id, frame)
            catalog.upsert_acquisition_dataset(
                dataset_id=dataset_id,
                source="stooq",
                symbol="AAPL",
                resolution="1d",
                history_window="max",
                parquet_path=str(parquet_path),
                coverage_start="2026-03-01T00:00:00+00:00",
                coverage_end="2026-03-01T00:00:00+00:00",
                bar_count=1,
                ingested=True,
                last_status="ingested",
            )
            decision = decide_acquisition_policy(
                "AAPL",
                source="stooq",
                resolution="1d",
                history_window="max",
                catalog=catalog,
                store=store,
                data_dir=root,
                now=pd.Timestamp("2026-04-09T12:00:00Z"),
            )
            self.assertEqual(decision.action, ACQUISITION_ACTION_DOWNLOAD)

    def test_fresh_but_gappy_dataset_redownloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            catalog = ResultCatalog(root / "catalog.sqlite")
            store = DuckDBStore(db_path=root / "history.duckdb", data_dir=root / "parquet")
            dataset_id = "AAPL_massive_1d_1m"
            frame = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(
                        [
                            "2026-04-08T14:30:00Z",
                            "2026-04-08T14:31:00Z",
                            "2026-04-08T14:35:00Z",
                        ],
                        utc=True,
                    ),
                    "open": [1.0, 1.1, 1.2],
                    "high": [1.2, 1.3, 1.4],
                    "low": [0.9, 1.0, 1.1],
                    "close": [1.1, 1.2, 1.3],
                    "volume": [10, 12, 14],
                }
            )
            parquet_path = store.write_parquet(dataset_id, frame)
            catalog.upsert_acquisition_dataset(
                dataset_id=dataset_id,
                source="massive",
                symbol="AAPL",
                resolution="1m",
                history_window="1d",
                parquet_path=str(parquet_path),
                coverage_start="2026-04-08T14:30:00+00:00",
                coverage_end="2026-04-08T14:35:00+00:00",
                bar_count=3,
                ingested=True,
                last_status="ingested",
            )
            decision = decide_acquisition_policy(
                "AAPL",
                source="massive",
                resolution="1m",
                history_window="1d",
                catalog=catalog,
                store=store,
                data_dir=root,
                now=pd.Timestamp("2026-04-09T12:00:00Z"),
            )
            self.assertEqual(decision.action, ACQUISITION_ACTION_DOWNLOAD)
            self.assertEqual(decision.quality_state, "gappy")
            self.assertEqual(decision.suspicious_gap_count, 1)
            self.assertEqual(decision.plan_type, ACQUISITION_PLAN_GAP_REPAIR_REFRESH)
            self.assertEqual(decision.request_start, "2026-04-08")
            self.assertEqual(decision.request_end, "2026-04-08")
            self.assertTrue(decision.merge_with_existing)

    def test_fresh_sparse_intraday_dataset_is_not_forced_into_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            catalog = ResultCatalog(root / "catalog.sqlite")
            store = DuckDBStore(db_path=root / "history.duckdb", data_dir=root / "parquet")
            timestamps: list[pd.Timestamp] = []
            day_starts = [
                pd.Timestamp("2026-04-07T13:30:00Z"),
                pd.Timestamp("2026-04-08T13:30:00Z"),
                pd.Timestamp("2026-04-09T13:30:00Z"),
                pd.Timestamp("2026-04-10T13:30:00Z"),
            ]
            for day_idx, start in enumerate(day_starts):
                for minute in range(390):
                    if day_idx < 2 and minute in {120, 121, 122, 123, 240, 241, 242, 243}:
                        continue
                    timestamps.append(start + pd.Timedelta(minutes=minute))
            frame = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(timestamps, utc=True),
                    "open": [1.0 + (idx * 0.001) for idx in range(len(timestamps))],
                    "high": [1.1 + (idx * 0.001) for idx in range(len(timestamps))],
                    "low": [0.9 + (idx * 0.001) for idx in range(len(timestamps))],
                    "close": [1.05 + (idx * 0.001) for idx in range(len(timestamps))],
                    "volume": [10] * len(timestamps),
                }
            )
            parquet_path = store.write_parquet("O_interactive_brokers_1d_1m", frame)
            quality_snapshot = store.inspect_dataset_quality("O_interactive_brokers_1d_1m", "1m")
            quality_snapshot["quality_analyzed_at"] = pd.Timestamp.now("UTC").isoformat()
            catalog.upsert_acquisition_dataset(
                dataset_id="O_interactive_brokers_1d_1m",
                source="interactive_brokers",
                symbol="O",
                resolution="1m",
                history_window="1d",
                parquet_path=str(parquet_path),
                coverage_start=str(frame["timestamp"].min().isoformat()),
                coverage_end=str(frame["timestamp"].max().isoformat()),
                bar_count=len(frame),
                ingested=True,
                last_status="ingested",
                quality_snapshot=quality_snapshot,
            )
            decision = decide_acquisition_policy(
                "O",
                source="interactive_brokers",
                resolution="1m",
                history_window="1d",
                catalog=catalog,
                store=store,
                data_dir=root,
                now=pd.Timestamp("2026-04-11T12:00:00Z"),
            )
            self.assertEqual(decision.quality_state, "sparse_intraday")
            self.assertEqual(decision.action, ACQUISITION_ACTION_SKIP_FRESH)

    def test_stale_dataset_gets_incremental_refresh_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            catalog = ResultCatalog(root / "catalog.sqlite")
            store = DuckDBStore(db_path=root / "history.duckdb", data_dir=root / "parquet")
            dataset_id = "AAPL_massive_2y_1m"
            frame = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(
                        ["2026-04-01T14:30:00Z", "2026-04-01T14:31:00Z"],
                        utc=True,
                    ),
                    "open": [1.0, 1.1],
                    "high": [1.2, 1.3],
                    "low": [0.9, 1.0],
                    "close": [1.1, 1.2],
                    "volume": [10, 12],
                }
            )
            parquet_path = store.write_parquet(dataset_id, frame)
            catalog.upsert_acquisition_dataset(
                dataset_id=dataset_id,
                source="massive",
                symbol="AAPL",
                resolution="1m",
                history_window="2y",
                parquet_path=str(parquet_path),
                coverage_start="2024-04-08T14:30:00+00:00",
                coverage_end="2026-04-01T14:31:00+00:00",
                bar_count=2,
                ingested=True,
                last_status="ingested",
            )
            decision = decide_acquisition_policy(
                "AAPL",
                source="massive",
                resolution="1m",
                history_window="2y",
                catalog=catalog,
                store=store,
                data_dir=root,
                now=pd.Timestamp("2026-04-09T12:00:00Z"),
            )
            self.assertEqual(decision.action, ACQUISITION_ACTION_DOWNLOAD)
            self.assertEqual(decision.plan_type, ACQUISITION_PLAN_INCREMENTAL_REFRESH)
            self.assertEqual(decision.request_start, "2026-04-01")
            self.assertEqual(decision.request_end, "2026-04-09")
            self.assertTrue(decision.merge_with_existing)

    def test_stale_dataset_with_gap_near_right_edge_gets_incremental_gap_repair_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            catalog = ResultCatalog(root / "catalog.sqlite")
            store = DuckDBStore(db_path=root / "history.duckdb", data_dir=root / "parquet")
            dataset_id = "AAPL_stooq_max_1d"
            frame = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(
                        ["2026-04-01T00:00:00Z", "2026-04-02T00:00:00Z", "2026-04-09T00:00:00Z"],
                        utc=True,
                    ),
                    "open": [1.0, 1.1, 1.2],
                    "high": [1.2, 1.3, 1.4],
                    "low": [0.9, 1.0, 1.1],
                    "close": [1.1, 1.2, 1.3],
                    "volume": [10, 12, 14],
                }
            )
            parquet_path = store.write_parquet(dataset_id, frame)
            catalog.upsert_acquisition_dataset(
                dataset_id=dataset_id,
                source="stooq",
                symbol="AAPL",
                resolution="1d",
                history_window="max",
                parquet_path=str(parquet_path),
                coverage_start="2026-04-01T00:00:00+00:00",
                coverage_end="2026-04-09T00:00:00+00:00",
                bar_count=3,
                ingested=True,
                last_status="ingested",
            )
            decision = decide_acquisition_policy(
                "AAPL",
                source="stooq",
                resolution="1d",
                history_window="max",
                catalog=catalog,
                store=store,
                data_dir=root,
                now=pd.Timestamp("2026-04-20T12:00:00Z"),
            )
            self.assertEqual(decision.action, ACQUISITION_ACTION_DOWNLOAD)
            self.assertEqual(decision.plan_type, ACQUISITION_PLAN_INCREMENTAL_GAP_REPAIR_REFRESH)
            self.assertEqual(decision.request_start, "2026-04-02")
            self.assertEqual(decision.request_end, "2026-04-20")
            self.assertTrue(decision.merge_with_existing)

    def test_fresh_dataset_with_shallow_history_gets_backfill_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            catalog = ResultCatalog(root / "catalog.sqlite")
            store = DuckDBStore(db_path=root / "history.duckdb", data_dir=root / "parquet")
            dataset_id = "AAPL_stooq_2y_1d"
            frame = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(
                        ["2026-04-06T00:00:00Z", "2026-04-08T00:00:00Z"],
                        utc=True,
                    ),
                    "open": [1.0, 1.1],
                    "high": [1.2, 1.3],
                    "low": [0.9, 1.0],
                    "close": [1.1, 1.2],
                    "volume": [10, 12],
                }
            )
            parquet_path = store.write_parquet(dataset_id, frame)
            catalog.upsert_acquisition_dataset(
                dataset_id=dataset_id,
                source="stooq",
                symbol="AAPL",
                resolution="1d",
                history_window="2y",
                parquet_path=str(parquet_path),
                coverage_start="2026-04-06T00:00:00+00:00",
                coverage_end="2026-04-08T00:00:00+00:00",
                bar_count=2,
                ingested=True,
                last_status="ingested",
            )
            decision = decide_acquisition_policy(
                "AAPL",
                source="stooq",
                resolution="1d",
                history_window="2y",
                catalog=catalog,
                store=store,
                data_dir=root,
                now=pd.Timestamp("2026-04-09T12:00:00Z"),
            )
            self.assertEqual(decision.action, ACQUISITION_ACTION_DOWNLOAD)
            self.assertEqual(decision.plan_type, ACQUISITION_PLAN_BACKFILL)
            self.assertTrue(decision.merge_with_existing)

    def test_fresh_dataset_with_shallow_history_and_gap_near_left_edge_gets_backfill_gap_repair_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            catalog = ResultCatalog(root / "catalog.sqlite")
            store = DuckDBStore(db_path=root / "history.duckdb", data_dir=root / "parquet")
            dataset_id = "AAPL_stooq_2y_1d"
            frame = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(
                        ["2026-04-06T00:00:00Z", "2026-04-11T00:00:00Z", "2026-04-12T00:00:00Z"],
                        utc=True,
                    ),
                    "open": [1.0, 1.1, 1.2],
                    "high": [1.2, 1.3, 1.4],
                    "low": [0.9, 1.0, 1.1],
                    "close": [1.1, 1.2, 1.3],
                    "volume": [10, 12, 14],
                }
            )
            parquet_path = store.write_parquet(dataset_id, frame)
            catalog.upsert_acquisition_dataset(
                dataset_id=dataset_id,
                source="stooq",
                symbol="AAPL",
                resolution="1d",
                history_window="2y",
                parquet_path=str(parquet_path),
                coverage_start="2026-04-06T00:00:00+00:00",
                coverage_end="2026-04-12T00:00:00+00:00",
                bar_count=3,
                ingested=True,
                last_status="ingested",
            )
            decision = decide_acquisition_policy(
                "AAPL",
                source="stooq",
                resolution="1d",
                history_window="2y",
                catalog=catalog,
                store=store,
                data_dir=root,
                now=pd.Timestamp("2026-04-13T12:00:00Z"),
            )
            self.assertEqual(decision.action, ACQUISITION_ACTION_DOWNLOAD)
            self.assertEqual(decision.plan_type, ACQUISITION_PLAN_BACKFILL_GAP_REPAIR_REFRESH)
            self.assertEqual(decision.request_start, "2024-04-13")
            self.assertEqual(decision.request_end, "2026-04-11")
            self.assertTrue(decision.merge_with_existing)

    def test_fresh_dataset_with_multiple_disjoint_gaps_gets_multi_window_repair_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            catalog = ResultCatalog(root / "catalog.sqlite")
            store = DuckDBStore(db_path=root / "history.duckdb", data_dir=root / "parquet")
            dataset_id = "AAPL_stooq_max_1d"
            frame = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(
                        [
                            "2026-01-01T00:00:00Z",
                            "2026-01-02T00:00:00Z",
                            "2026-01-10T00:00:00Z",
                            "2026-01-11T00:00:00Z",
                            "2026-01-20T00:00:00Z",
                        ],
                        utc=True,
                    ),
                    "open": [1.0, 1.1, 1.2, 1.3, 1.4],
                    "high": [1.2, 1.3, 1.4, 1.5, 1.6],
                    "low": [0.9, 1.0, 1.1, 1.2, 1.3],
                    "close": [1.1, 1.2, 1.3, 1.4, 1.5],
                    "volume": [10, 12, 14, 16, 18],
                }
            )
            parquet_path = store.write_parquet(dataset_id, frame)
            catalog.upsert_acquisition_dataset(
                dataset_id=dataset_id,
                source="stooq",
                symbol="AAPL",
                resolution="1d",
                history_window="max",
                parquet_path=str(parquet_path),
                coverage_start="2026-01-01T00:00:00+00:00",
                coverage_end="2026-01-20T00:00:00+00:00",
                bar_count=5,
                ingested=True,
                last_status="ingested",
            )
            decision = decide_acquisition_policy(
                "AAPL",
                source="stooq",
                resolution="1d",
                history_window="max",
                catalog=catalog,
                store=store,
                data_dir=root,
                now=pd.Timestamp("2026-01-25T12:00:00Z"),
            )
            self.assertEqual(decision.action, ACQUISITION_ACTION_DOWNLOAD)
            self.assertEqual(decision.plan_type, ACQUISITION_PLAN_MULTI_WINDOW_GAP_REPAIR_REFRESH)
            self.assertEqual(
                decision.request_windows,
                (("2026-01-02", "2026-01-10"), ("2026-01-11", "2026-01-20")),
            )

    def test_stale_dataset_with_internal_gap_gets_compound_multi_window_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            catalog = ResultCatalog(root / "catalog.sqlite")
            store = DuckDBStore(db_path=root / "history.duckdb", data_dir=root / "parquet")
            dataset_id = "AAPL_stooq_max_1d"
            frame = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(
                        [
                            "2026-04-01T00:00:00Z",
                            "2026-04-02T00:00:00Z",
                            "2026-04-10T00:00:00Z",
                            "2026-04-11T00:00:00Z",
                            "2026-04-12T00:00:00Z",
                        ],
                        utc=True,
                    ),
                    "open": [1.0, 1.1, 1.2, 1.3, 1.4],
                    "high": [1.2, 1.3, 1.4, 1.5, 1.6],
                    "low": [0.9, 1.0, 1.1, 1.2, 1.3],
                    "close": [1.1, 1.2, 1.3, 1.4, 1.5],
                    "volume": [10, 12, 14, 16, 18],
                }
            )
            parquet_path = store.write_parquet(dataset_id, frame)
            catalog.upsert_acquisition_dataset(
                dataset_id=dataset_id,
                source="stooq",
                symbol="AAPL",
                resolution="1d",
                history_window="max",
                parquet_path=str(parquet_path),
                coverage_start="2026-04-01T00:00:00+00:00",
                coverage_end="2026-04-12T00:00:00+00:00",
                bar_count=5,
                ingested=True,
                last_status="ingested",
            )
            decision = decide_acquisition_policy(
                "AAPL",
                source="stooq",
                resolution="1d",
                history_window="max",
                catalog=catalog,
                store=store,
                data_dir=root,
                now=pd.Timestamp("2026-04-25T12:00:00Z"),
            )
            self.assertEqual(decision.action, ACQUISITION_ACTION_DOWNLOAD)
            self.assertEqual(decision.plan_type, ACQUISITION_PLAN_COMPOUND_MULTI_WINDOW_REFRESH)
            self.assertEqual(
                decision.request_windows,
                (("2026-04-02", "2026-04-10"), ("2026-04-12", "2026-04-25")),
            )

    def test_fresh_gappy_dataset_prefers_matching_secondary_source_gap_fill(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            catalog = ResultCatalog(root / "catalog.sqlite")
            store = DuckDBStore(db_path=root / "history.duckdb", data_dir=root / "parquet")
            primary_id = "AAPL_massive_max_1d"
            primary = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(
                        ["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", "2026-01-10T00:00:00Z"],
                        utc=True,
                    ),
                    "open": [100.0, 101.0, 102.0],
                    "high": [101.0, 102.0, 103.0],
                    "low": [99.0, 100.0, 101.0],
                    "close": [100.0, 101.0, 102.0],
                    "volume": [10, 20, 30],
                }
            )
            secondary_id = "AAPL_stooq_max_1d"
            secondary = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(
                        [
                            "2026-01-01T00:00:00Z",
                            "2026-01-02T00:00:00Z",
                            "2026-01-03T00:00:00Z",
                            "2026-01-04T00:00:00Z",
                            "2026-01-05T00:00:00Z",
                            "2026-01-06T00:00:00Z",
                            "2026-01-07T00:00:00Z",
                            "2026-01-08T00:00:00Z",
                            "2026-01-09T00:00:00Z",
                            "2026-01-10T00:00:00Z",
                        ],
                        utc=True,
                    ),
                    "open": [100.0, 101.0, 101.1, 101.2, 101.3, 101.4, 101.5, 101.6, 101.7, 102.0],
                    "high": [101.0, 102.0, 102.1, 102.2, 102.3, 102.4, 102.5, 102.6, 102.7, 103.0],
                    "low": [99.0, 100.0, 100.1, 100.2, 100.3, 100.4, 100.5, 100.6, 100.7, 101.0],
                    "close": [100.02, 100.98, 101.1, 101.2, 101.3, 101.4, 101.5, 101.6, 101.7, 102.03],
                    "volume": [10, 20, 21, 22, 23, 24, 25, 26, 27, 30],
                }
            )
            primary_path = store.write_parquet(primary_id, primary)
            secondary_path = store.write_parquet(secondary_id, secondary)
            catalog.upsert_acquisition_dataset(
                dataset_id=primary_id,
                source="massive",
                symbol="AAPL",
                resolution="1d",
                history_window="max",
                parquet_path=str(primary_path),
                coverage_start="2026-01-01T00:00:00+00:00",
                coverage_end="2026-01-10T00:00:00+00:00",
                bar_count=3,
                ingested=True,
                last_status="ingested",
            )
            catalog.upsert_acquisition_dataset(
                dataset_id=secondary_id,
                source="stooq",
                symbol="AAPL",
                resolution="1d",
                history_window="max",
                parquet_path=str(secondary_path),
                coverage_start="2026-01-01T00:00:00+00:00",
                coverage_end="2026-01-10T00:00:00+00:00",
                bar_count=10,
                ingested=True,
                last_status="ingested",
            )
            decision = decide_acquisition_policy(
                "AAPL",
                source="massive",
                resolution="1d",
                history_window="max",
                catalog=catalog,
                store=store,
                data_dir=root,
                now=pd.Timestamp("2026-01-12T12:00:00Z"),
            )
            self.assertEqual(decision.action, ACQUISITION_ACTION_GAP_FILL_SECONDARY)
            self.assertEqual(decision.plan_type, ACQUISITION_PLAN_CROSS_SOURCE_GAP_FILL)
            self.assertEqual(decision.secondary_source, "stooq")
            self.assertEqual(decision.secondary_dataset_id, secondary_id)
            self.assertEqual(decision.parity_state, "matching")
            self.assertGreaterEqual(decision.parity_overlap_bars, 3)

    def test_stale_gappy_dataset_can_hybrid_secondary_gap_fill_then_primary_incremental_refresh(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            catalog = ResultCatalog(root / "catalog.sqlite")
            store = DuckDBStore(db_path=root / "history.duckdb", data_dir=root / "parquet")
            primary_id = "AAPL_massive_max_1d"
            primary = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(
                        ["2026-04-01T00:00:00Z", "2026-04-02T00:00:00Z", "2026-04-10T00:00:00Z"],
                        utc=True,
                    ),
                    "open": [100.0, 101.0, 102.0],
                    "high": [101.0, 102.0, 103.0],
                    "low": [99.0, 100.0, 101.0],
                    "close": [100.0, 101.0, 102.0],
                    "volume": [10, 20, 30],
                }
            )
            secondary_id = "AAPL_stooq_max_1d"
            secondary = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(
                        [
                            "2026-04-01T00:00:00Z",
                            "2026-04-02T00:00:00Z",
                            "2026-04-03T00:00:00Z",
                            "2026-04-04T00:00:00Z",
                            "2026-04-05T00:00:00Z",
                            "2026-04-06T00:00:00Z",
                            "2026-04-07T00:00:00Z",
                            "2026-04-08T00:00:00Z",
                            "2026-04-09T00:00:00Z",
                            "2026-04-10T00:00:00Z",
                            "2026-04-11T00:00:00Z",
                            "2026-04-12T00:00:00Z",
                        ],
                        utc=True,
                    ),
                    "open": [100.0, 101.0, 101.1, 101.2, 101.3, 101.4, 101.5, 101.6, 101.7, 102.0, 102.1, 102.2],
                    "high": [101.0, 102.0, 102.1, 102.2, 102.3, 102.4, 102.5, 102.6, 102.7, 103.0, 103.1, 103.2],
                    "low": [99.0, 100.0, 100.1, 100.2, 100.3, 100.4, 100.5, 100.6, 100.7, 101.0, 101.1, 101.2],
                    "close": [100.02, 100.98, 101.1, 101.2, 101.3, 101.4, 101.5, 101.6, 101.7, 102.03, 102.1, 102.2],
                    "volume": [10, 20, 21, 22, 23, 24, 25, 26, 27, 30, 31, 32],
                }
            )
            primary_path = store.write_parquet(primary_id, primary)
            secondary_path = store.write_parquet(secondary_id, secondary)
            catalog.upsert_acquisition_dataset(
                dataset_id=primary_id,
                source="massive",
                symbol="AAPL",
                resolution="1d",
                history_window="max",
                parquet_path=str(primary_path),
                coverage_start="2026-04-01T00:00:00+00:00",
                coverage_end="2026-04-10T00:00:00+00:00",
                bar_count=3,
                ingested=True,
                last_status="ingested",
            )
            catalog.upsert_acquisition_dataset(
                dataset_id=secondary_id,
                source="stooq",
                symbol="AAPL",
                resolution="1d",
                history_window="max",
                parquet_path=str(secondary_path),
                coverage_start="2026-04-01T00:00:00+00:00",
                coverage_end="2026-04-12T00:00:00+00:00",
                bar_count=12,
                ingested=True,
                last_status="ingested",
            )
            decision = decide_acquisition_policy(
                "AAPL",
                source="massive",
                resolution="1d",
                history_window="max",
                catalog=catalog,
                store=store,
                data_dir=root,
                now=pd.Timestamp("2026-04-20T12:00:00Z"),
            )
            self.assertEqual(decision.action, ACQUISITION_ACTION_DOWNLOAD)
            self.assertEqual(decision.plan_type, ACQUISITION_PLAN_CROSS_SOURCE_GAP_FILL_PLUS_INCREMENTAL_REFRESH)
            self.assertEqual(decision.secondary_source, "stooq")
            self.assertEqual(decision.secondary_dataset_id, secondary_id)
            self.assertEqual(decision.secondary_request_windows, (("2026-04-02", "2026-04-10"),))
            self.assertEqual(decision.request_windows, (("2026-04-10", "2026-04-20"),))


if __name__ == "__main__":
    unittest.main()
