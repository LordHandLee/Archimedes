from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from backtest_engine.catalog import ResultCatalog
from backtest_engine.engine_comparison import duration_seconds, summarize_engine_batch, summarize_engine_runs
from backtest_engine.metrics import PerformanceMetrics


class EngineComparisonTest(unittest.TestCase):
    def test_duration_seconds_handles_complete_and_incomplete_runs(self) -> None:
        self.assertEqual(
            duration_seconds("2024-01-02T14:30:00+00:00", "2024-01-02T14:30:02.500000+00:00"),
            2.5,
        )
        self.assertIsNone(duration_seconds("2024-01-02T14:30:00+00:00", None))

    def test_catalog_peers_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            catalog = ResultCatalog(Path(tmpdir) / "compare.sqlite")

            logical_run_id = "logical_123"
            base_kwargs = dict(
                batch_id="batch_1",
                strategy="SMACrossStrategy",
                params={"fast": 5, "slow": 20, "target": 1.0},
                timeframe="1 minutes",
                start="2024-01-02T14:30:00+00:00",
                end="2024-01-02T18:30:00+00:00",
                dataset_id="compare_ds",
                starting_cash=100_000.0,
                status="finished",
                logical_run_id=logical_run_id,
                engine_version="1",
            )

            catalog.save(
                run_id="ref_run",
                metrics=PerformanceMetrics(total_return=0.12, cagr=0.12, max_drawdown=0.04, sharpe=1.2, rolling_sharpe=1.1),
                run_started_at="2024-01-02T14:30:00+00:00",
                run_finished_at="2024-01-02T14:30:10+00:00",
                requested_execution_mode="reference",
                resolved_execution_mode="reference",
                engine_impl="reference",
                fallback_reason=None,
                **base_kwargs,
            )
            catalog.save(
                run_id="vec_run_old",
                metrics=PerformanceMetrics(total_return=0.119, cagr=0.119, max_drawdown=0.041, sharpe=1.18, rolling_sharpe=1.08),
                run_started_at="2024-01-02T14:31:00+00:00",
                run_finished_at="2024-01-02T14:31:03+00:00",
                requested_execution_mode="vectorized",
                resolved_execution_mode="vectorized",
                engine_impl="vectorized",
                fallback_reason=None,
                **base_kwargs,
            )
            catalog.save(
                run_id="vec_run_new",
                metrics=PerformanceMetrics(total_return=0.1205, cagr=0.1205, max_drawdown=0.0405, sharpe=1.199, rolling_sharpe=1.099),
                run_started_at="2024-01-02T14:32:00+00:00",
                run_finished_at="2024-01-02T14:32:02+00:00",
                requested_execution_mode="vectorized",
                resolved_execution_mode="vectorized",
                engine_impl="vectorized",
                fallback_reason=None,
                **base_kwargs,
            )

            peers = catalog.load_runs_for_logical_run_id(logical_run_id)
            self.assertEqual(len(peers), 3)

            summary = summarize_engine_runs(peers)
            self.assertEqual(summary.logical_run_id, logical_run_id)
            self.assertIsNotNone(summary.latest_reference)
            self.assertIsNotNone(summary.latest_vectorized)
            self.assertEqual(summary.latest_reference.run_id, "ref_run")
            self.assertEqual(summary.latest_vectorized.run_id, "vec_run_new")
            self.assertAlmostEqual(summary.speedup_vs_reference or 0.0, 5.0, places=6)
            self.assertAlmostEqual(summary.total_return_delta or 0.0, 0.0005, places=9)
            self.assertAlmostEqual(summary.sharpe_delta or 0.0, -0.001, places=9)
            self.assertAlmostEqual(summary.max_drawdown_delta or 0.0, 0.0005, places=9)
            self.assertEqual(summary.available_engines, ("reference", "vectorized"))

            batch_summary = summarize_engine_batch(peers)
            self.assertEqual(batch_summary.total_groups, 1)
            self.assertEqual(batch_summary.paired_groups, 1)
            self.assertEqual(batch_summary.reference_only_groups, 0)
            self.assertEqual(batch_summary.vectorized_only_groups, 0)
            self.assertAlmostEqual(batch_summary.median_speedup_vs_reference or 0.0, 5.0, places=6)
            self.assertAlmostEqual(batch_summary.mean_speedup_vs_reference or 0.0, 5.0, places=6)
            self.assertAlmostEqual(batch_summary.max_abs_total_return_delta or 0.0, 0.0005, places=9)
            self.assertAlmostEqual(batch_summary.max_abs_sharpe_delta or 0.0, 0.001, places=9)
            self.assertAlmostEqual(batch_summary.max_abs_max_drawdown_delta or 0.0, 0.0005, places=9)

    def test_batch_summary_counts_reference_only_and_vectorized_only_groups(self) -> None:
        class _Run:
            def __init__(self, run_id: str, logical_run_id: str, engine_impl: str) -> None:
                self.run_id = run_id
                self.logical_run_id = logical_run_id
                self.engine_impl = engine_impl
                self.requested_execution_mode = engine_impl
                self.resolved_execution_mode = engine_impl
                self.engine_version = "1"
                self.run_started_at = "2024-01-02T14:30:00+00:00"
                self.run_finished_at = "2024-01-02T14:30:01+00:00"
                self.status = "finished"
                self.metrics = {
                    "total_return": 0.1,
                    "sharpe": 1.0,
                    "rolling_sharpe": 1.0,
                    "max_drawdown": 0.05,
                }
                self.fallback_reason = None

        runs = [
            _Run("ref_a", "group_a", "reference"),
            _Run("vec_a", "group_a", "vectorized"),
            _Run("ref_b", "group_b", "reference"),
            _Run("vec_c", "group_c", "vectorized"),
        ]
        summary = summarize_engine_batch(runs)
        self.assertEqual(summary.total_groups, 3)
        self.assertEqual(summary.paired_groups, 1)
        self.assertEqual(summary.reference_only_groups, 1)
        self.assertEqual(summary.vectorized_only_groups, 1)


if __name__ == "__main__":
    unittest.main()
