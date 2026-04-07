from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from backtest_engine.catalog import ResultCatalog
from backtest_engine.execution import BatchExecutionBenchmark, ExecutionMode


class BatchBenchmarkPersistenceTest(unittest.TestCase):
    def test_batch_benchmarks_round_trip_through_catalog(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "benchmarks.sqlite"
            catalog = ResultCatalog(db_path)
            benchmarks = [
                BatchExecutionBenchmark(
                    dataset_id="asset_a",
                    strategy="SMACrossStrategy",
                    timeframe="1 minutes",
                    requested_execution_mode=ExecutionMode.AUTO,
                    resolved_execution_mode=ExecutionMode.VECTORIZED,
                    engine_impl="vectorized",
                    engine_version="1",
                    bars=240,
                    total_params=6,
                    cached_runs=1,
                    uncached_runs=5,
                    duration_seconds=0.8125,
                    chunk_count=3,
                    chunk_sizes=(2, 2, 1),
                    effective_param_batch_size=2,
                    prepared_context_reused=True,
                ),
                BatchExecutionBenchmark(
                    dataset_id="asset_a",
                    strategy="SMACrossStrategy",
                    timeframe="5 minutes",
                    requested_execution_mode=ExecutionMode.AUTO,
                    resolved_execution_mode=ExecutionMode.REFERENCE,
                    engine_impl="reference",
                    engine_version="1",
                    bars=48,
                    total_params=6,
                    cached_runs=0,
                    uncached_runs=6,
                    duration_seconds=1.337,
                ),
            ]

            catalog.save_batch_benchmarks("batch_001", benchmarks)

            reloaded = ResultCatalog(db_path).load_batch_benchmarks("batch_001")

        self.assertEqual(len(reloaded), 2)
        self.assertEqual(reloaded[0].batch_id, "batch_001")
        self.assertEqual(reloaded[0].dataset_id, "asset_a")
        self.assertEqual(reloaded[0].resolved_execution_mode, "vectorized")
        self.assertEqual(reloaded[0].chunk_sizes, (2, 2, 1))
        self.assertEqual(reloaded[0].effective_param_batch_size, 2)
        self.assertTrue(reloaded[0].prepared_context_reused)
        self.assertEqual(reloaded[1].resolved_execution_mode, "reference")
        self.assertEqual(reloaded[1].chunk_count, 0)
        self.assertEqual(reloaded[1].chunk_sizes, ())

    def test_saving_batch_benchmarks_replaces_previous_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "benchmarks.sqlite"
            catalog = ResultCatalog(db_path)
            catalog.save_batch_benchmarks(
                "batch_001",
                [
                    BatchExecutionBenchmark(
                        dataset_id="asset_a",
                        strategy="SMACrossStrategy",
                        timeframe="1 minutes",
                        requested_execution_mode=ExecutionMode.AUTO,
                        resolved_execution_mode=ExecutionMode.VECTORIZED,
                        engine_impl="vectorized",
                        engine_version="1",
                        bars=240,
                        total_params=4,
                        cached_runs=0,
                        uncached_runs=4,
                        duration_seconds=0.5,
                    ),
                    BatchExecutionBenchmark(
                        dataset_id="asset_a",
                        strategy="SMACrossStrategy",
                        timeframe="5 minutes",
                        requested_execution_mode=ExecutionMode.AUTO,
                        resolved_execution_mode=ExecutionMode.REFERENCE,
                        engine_impl="reference",
                        engine_version="1",
                        bars=48,
                        total_params=4,
                        cached_runs=0,
                        uncached_runs=4,
                        duration_seconds=0.75,
                    ),
                ],
            )
            catalog.save_batch_benchmarks(
                "batch_001",
                [
                    BatchExecutionBenchmark(
                        dataset_id="asset_b",
                        strategy="ZScoreMeanReversionStrategy",
                        timeframe="1 minutes",
                        requested_execution_mode=ExecutionMode.VECTORIZED,
                        resolved_execution_mode=ExecutionMode.VECTORIZED,
                        engine_impl="vectorized",
                        engine_version="1",
                        bars=300,
                        total_params=3,
                        cached_runs=1,
                        uncached_runs=2,
                        duration_seconds=0.4,
                        chunk_count=2,
                        chunk_sizes=(2, 1),
                        effective_param_batch_size=2,
                        prepared_context_reused=True,
                    )
                ],
            )

            reloaded = catalog.load_batch_benchmarks("batch_001")

        self.assertEqual(len(reloaded), 1)
        self.assertEqual(reloaded[0].dataset_id, "asset_b")
        self.assertEqual(reloaded[0].strategy, "ZScoreMeanReversionStrategy")
        self.assertEqual(reloaded[0].seq, 1)


if __name__ == "__main__":
    unittest.main()
