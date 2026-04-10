from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from backtest_engine.catalog import ResultCatalog


class AcquisitionCatalogTests(unittest.TestCase):
    def test_acquisition_run_attempt_and_dataset_metadata_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "catalog.sqlite"
            catalog = ResultCatalog(db_path)
            catalog.start_acquisition_run(
                acquisition_run_id="acq_001",
                trigger_type="interactive_download",
                source="massive",
                universe_id="u_core",
                universe_name="Core ETFs",
                started_at="2026-04-09T13:00:00+00:00",
                status="running",
                symbol_count=2,
            )
            catalog.record_acquisition_attempt(
                attempt_id="acq_001_0001",
                acquisition_run_id="acq_001",
                seq=1,
                source="massive",
                symbol="SPY",
                dataset_id="SPY_massive_2y_1m",
                status="ingested",
                started_at="2026-04-09T13:00:00+00:00",
                finished_at="2026-04-09T13:02:00+00:00",
                csv_path="data/SPY_massive_2y_1m.csv",
                parquet_path="data/parquet/SPY_massive_2y_1m.parquet",
                coverage_start="2024-04-09T13:00:00+00:00",
                coverage_end="2026-04-09T13:00:00+00:00",
                bar_count=123,
                ingested=True,
                resolution="1m",
                history_window="2y",
            )
            catalog.record_acquisition_attempt(
                attempt_id="acq_001_0002",
                acquisition_run_id="acq_001",
                seq=2,
                source="massive",
                symbol="QQQ",
                dataset_id="QQQ_massive_2y_1m",
                status="download_error",
                started_at="2026-04-09T13:03:00+00:00",
                finished_at="2026-04-09T13:04:00+00:00",
                csv_path="data/QQQ_massive_2y_1m.csv",
                ingested=False,
                error_message="QQQ failed",
                resolution="1m",
                history_window="2y",
            )
            catalog.finish_acquisition_run(
                "acq_001",
                finished_at="2026-04-09T13:05:00+00:00",
                status="partial",
                success_count=1,
                failed_count=1,
                ingested_count=1,
                notes="1 succeeded, 1 failed.",
            )

            runs = catalog.load_acquisition_runs()
            attempts = catalog.load_acquisition_attempts()
            datasets = catalog.load_acquisition_datasets()

            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0].status, "partial")
            self.assertEqual(runs[0].success_count, 1)
            self.assertEqual(runs[0].failed_count, 1)
            self.assertEqual(len(attempts), 2)
            by_dataset = {row.dataset_id: row for row in datasets}
            self.assertTrue(by_dataset["SPY_massive_2y_1m"].ingested)
            self.assertEqual(by_dataset["SPY_massive_2y_1m"].bar_count, 123)
            self.assertEqual(by_dataset["SPY_massive_2y_1m"].last_status, "ingested")
            self.assertFalse(by_dataset["QQQ_massive_2y_1m"].ingested)
            self.assertEqual(by_dataset["QQQ_massive_2y_1m"].last_status, "download_error")
            self.assertEqual(by_dataset["QQQ_massive_2y_1m"].last_error, "QQQ failed")

    def test_task_run_history_and_acquisition_filters(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "catalog.sqlite"
            catalog = ResultCatalog(db_path)
            catalog.add_task_run(
                run_id="task_run_001",
                task_id="task_core",
                started_at="2026-04-09T13:00:00+00:00",
                status="running",
                ticker_count=2,
                log_path="data/scheduler_logs/task_core/run.log",
            )
            catalog.finish_task_run(
                "task_run_001",
                finished_at="2026-04-09T13:05:00+00:00",
                status="partial",
                error_message="1 symbol failed",
            )
            catalog.start_acquisition_run(
                acquisition_run_id="acq_task_001",
                trigger_type="scheduled_task",
                source="massive",
                universe_id="u_core",
                universe_name="Core ETFs",
                task_id="task_core",
                started_at="2026-04-09T13:00:00+00:00",
                status="running",
                symbol_count=2,
            )
            catalog.record_acquisition_attempt(
                attempt_id="acq_task_001_0001",
                acquisition_run_id="acq_task_001",
                seq=1,
                source="massive",
                symbol="SPY",
                dataset_id="SPY_massive_2y_1m",
                status="ingested",
                started_at="2026-04-09T13:00:00+00:00",
                finished_at="2026-04-09T13:02:00+00:00",
                bar_count=123,
                ingested=True,
                task_id="task_core",
                universe_id="u_core",
            )
            catalog.record_acquisition_attempt(
                attempt_id="acq_task_001_0002",
                acquisition_run_id="acq_task_001",
                seq=2,
                source="massive",
                symbol="QQQ",
                dataset_id="QQQ_massive_2y_1m",
                status="download_error",
                started_at="2026-04-09T13:03:00+00:00",
                finished_at="2026-04-09T13:04:00+00:00",
                ingested=False,
                error_message="QQQ failed",
                task_id="task_core",
                universe_id="u_core",
            )
            catalog.finish_acquisition_run(
                "acq_task_001",
                finished_at="2026-04-09T13:05:00+00:00",
                status="partial",
                success_count=1,
                failed_count=1,
                ingested_count=1,
                notes="partial scheduled run",
            )

            task_runs = catalog.load_task_runs(task_id="task_core")
            filtered_runs = catalog.load_acquisition_runs(task_id="task_core", universe_id="u_core")
            filtered_attempts = catalog.load_acquisition_attempts(task_id="task_core", universe_id="u_core")

            self.assertEqual(len(task_runs), 1)
            self.assertEqual(task_runs[0].status, "partial")
            self.assertEqual(task_runs[0].error_message, "1 symbol failed")
            self.assertEqual(len(filtered_runs), 1)
            self.assertEqual(filtered_runs[0].task_id, "task_core")
            self.assertEqual(len(filtered_attempts), 2)
            self.assertEqual({row.symbol for row in filtered_attempts}, {"SPY", "QQQ"})

    def test_skipped_attempt_does_not_pollute_dataset_last_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "catalog.sqlite"
            catalog = ResultCatalog(db_path)
            catalog.start_acquisition_run(
                acquisition_run_id="acq_skip_001",
                trigger_type="interactive_download",
                source="massive",
                started_at="2026-04-09T13:00:00+00:00",
                status="running",
                symbol_count=1,
            )
            catalog.record_acquisition_attempt(
                attempt_id="acq_skip_001_0001",
                acquisition_run_id="acq_skip_001",
                seq=1,
                source="massive",
                symbol="SPY",
                dataset_id="SPY_massive_2y_1m",
                status="skipped",
                started_at="2026-04-09T13:00:00+00:00",
                finished_at="2026-04-09T13:01:00+00:00",
                csv_path="data/SPY_massive_2y_1m.csv",
                parquet_path="data/parquet/SPY_massive_2y_1m.parquet",
                coverage_start="2024-04-09T13:00:00+00:00",
                coverage_end="2026-04-09T13:00:00+00:00",
                bar_count=123,
                ingested=True,
                error_message="Dataset is already fresh.",
                resolution="1m",
                history_window="2y",
            )
            datasets = {row.dataset_id: row for row in catalog.load_acquisition_datasets()}
            self.assertEqual(datasets["SPY_massive_2y_1m"].last_status, "skipped")
            self.assertEqual(datasets["SPY_massive_2y_1m"].last_error, "")

    def test_successful_attempt_does_not_expose_policy_reason_as_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "catalog.sqlite"
            catalog = ResultCatalog(db_path)
            catalog.start_acquisition_run(
                acquisition_run_id="acq_success_001",
                trigger_type="interactive_download",
                source="interactive_brokers",
                started_at="2026-04-09T13:00:00+00:00",
                status="running",
                symbol_count=1,
            )
            catalog.record_acquisition_attempt(
                attempt_id="acq_success_001_0001",
                acquisition_run_id="acq_success_001",
                seq=1,
                source="interactive_brokers",
                symbol="KO",
                dataset_id="KO_interactive_brokers_10y_1m",
                status="ingested",
                started_at="2026-04-09T13:00:00+00:00",
                finished_at="2026-04-09T13:20:00+00:00",
                csv_path="data/KO_interactive_brokers_10y_1m.csv",
                parquet_path="data/parquet/KO_interactive_brokers_10y_1m.parquet",
                coverage_start="2016-04-11T00:00:00+00:00",
                coverage_end="2026-04-09T00:00:00+00:00",
                bar_count=2217581,
                ingested=True,
                error_message="No canonical dataset exists yet.",
                resolution="1m",
                history_window="10y",
            )

            attempts = catalog.load_acquisition_attempts()
            datasets = {row.dataset_id: row for row in catalog.load_acquisition_datasets()}

            self.assertEqual(len(attempts), 1)
            self.assertIsNone(attempts[0].error_message)
            self.assertEqual(datasets["KO_interactive_brokers_10y_1m"].last_error, "")


if __name__ == "__main__":
    unittest.main()
