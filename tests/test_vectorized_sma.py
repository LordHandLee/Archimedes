from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_engine.catalog import ResultCatalog
from backtest_engine.engine import BacktestConfig
from backtest_engine.execution import ExecutionMode, ExecutionOrchestrator, ExecutionRequest
from backtest_engine.grid_search import (
    GridSearch,
    GridSpec,
    IndependentAssetTarget,
    run_independent_asset_grid_search,
)
from backtest_engine.sample_strategies import SMACrossStrategy
from backtest_engine.vectorized_strategies import SMACrossVectorizedAdapter


def _make_bars(periods: int = 240) -> pd.DataFrame:
    index = pd.date_range("2024-01-02 14:30", periods=periods, freq="1min", tz="UTC")
    base = 100.0 + np.linspace(0, 12, periods)
    wave = np.sin(np.arange(periods) / 6.0) * 1.5
    close = base + wave
    open_ = np.roll(close, 1)
    open_[0] = close[0] - 0.25
    open_ += np.cos(np.arange(periods) / 9.0) * 0.15
    bars = pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + 0.4,
            "low": np.minimum(open_, close) - 0.4,
            "close": close,
            "volume": 1000,
        },
        index=index,
    )
    return bars


class VectorizedSMATest(unittest.TestCase):
    def setUp(self) -> None:
        self.bars = _make_bars()
        self.config = BacktestConfig(
            timeframe="1 minutes",
            use_cache=False,
            base_execution=False,
            allow_short=False,
            fee_rate=0.0002,
            slippage=0.0001,
        )
        self.params = {"fast": 5, "slow": 20, "target": 1.0}

    def _resample_5m(self) -> pd.DataFrame:
        return self.bars.resample("5min", label="right", closed="right").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()

    def test_vectorized_matches_reference_for_sma(self) -> None:
        orchestrator = ExecutionOrchestrator()
        reference = orchestrator.execute(
            ExecutionRequest(
                data=self.bars,
                base_data=self.bars,
                dataset_id="sma_parity",
                strategy_cls=SMACrossStrategy,
                strategy_params=self.params,
                config=self.config,
                requested_execution_mode=ExecutionMode.REFERENCE,
            )
        )
        vectorized = orchestrator.execute(
            ExecutionRequest(
                data=self.bars,
                base_data=self.bars,
                dataset_id="sma_parity",
                strategy_cls=SMACrossStrategy,
                strategy_params=self.params,
                config=self.config,
                requested_execution_mode=ExecutionMode.VECTORIZED,
            )
        )

        self.assertEqual(reference.resolved_execution_mode.value, "reference")
        self.assertEqual(vectorized.resolved_execution_mode.value, "vectorized")
        self.assertEqual(reference.logical_run_id, vectorized.logical_run_id)
        self.assertNotEqual(reference.run_id, vectorized.run_id)
        self.assertEqual(len(reference.trades), len(vectorized.trades))
        self.assertTrue(np.allclose(reference.equity_curve.to_numpy(), vectorized.equity_curve.to_numpy(), atol=1e-8))
        self.assertAlmostEqual(reference.metrics.total_return, vectorized.metrics.total_return, places=10)
        self.assertAlmostEqual(reference.metrics.max_drawdown, vectorized.metrics.max_drawdown, places=10)
        if np.isnan(reference.metrics.sharpe) and np.isnan(vectorized.metrics.sharpe):
            pass
        else:
            self.assertAlmostEqual(reference.metrics.sharpe, vectorized.metrics.sharpe, places=10)
        for ref_trade, vec_trade in zip(reference.trades, vectorized.trades):
            self.assertEqual(ref_trade.side, vec_trade.side)
            self.assertAlmostEqual(ref_trade.qty, vec_trade.qty, places=10)
            self.assertAlmostEqual(ref_trade.price, vec_trade.price, places=10)
            self.assertAlmostEqual(ref_trade.fee, vec_trade.fee, places=10)

    def test_vectorized_matches_reference_with_fill_ratio(self) -> None:
        config = replace(self.config, fill_ratio=0.5)
        orchestrator = ExecutionOrchestrator()
        reference = orchestrator.execute(
            ExecutionRequest(
                data=self.bars,
                base_data=self.bars,
                dataset_id="sma_fill_ratio",
                strategy_cls=SMACrossStrategy,
                strategy_params=self.params,
                config=config,
                requested_execution_mode=ExecutionMode.REFERENCE,
            )
        )
        vectorized = orchestrator.execute(
            ExecutionRequest(
                data=self.bars,
                base_data=self.bars,
                dataset_id="sma_fill_ratio",
                strategy_cls=SMACrossStrategy,
                strategy_params=self.params,
                config=config,
                requested_execution_mode=ExecutionMode.VECTORIZED,
            )
        )
        self.assertEqual(len(reference.trades), len(vectorized.trades))
        self.assertTrue(np.allclose(reference.equity_curve.to_numpy(), vectorized.equity_curve.to_numpy(), atol=1e-8))
        self.assertAlmostEqual(reference.metrics.total_return, vectorized.metrics.total_return, places=10)

    def test_vectorized_matches_reference_with_fill_on_close(self) -> None:
        config = replace(self.config, fill_on_close=True)
        orchestrator = ExecutionOrchestrator()
        reference = orchestrator.execute(
            ExecutionRequest(
                data=self.bars,
                base_data=self.bars,
                dataset_id="sma_fill_on_close",
                strategy_cls=SMACrossStrategy,
                strategy_params=self.params,
                config=config,
                requested_execution_mode=ExecutionMode.REFERENCE,
            )
        )
        vectorized = orchestrator.execute(
            ExecutionRequest(
                data=self.bars,
                base_data=self.bars,
                dataset_id="sma_fill_on_close",
                strategy_cls=SMACrossStrategy,
                strategy_params=self.params,
                config=config,
                requested_execution_mode=ExecutionMode.VECTORIZED,
            )
        )
        self.assertEqual(len(reference.trades), len(vectorized.trades))
        self.assertTrue(np.allclose(reference.equity_curve.to_numpy(), vectorized.equity_curve.to_numpy(), atol=1e-8))
        self.assertAlmostEqual(reference.metrics.total_return, vectorized.metrics.total_return, places=10)

    def test_grid_search_auto_uses_vectorized_for_supported_sma(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            catalog = ResultCatalog(Path(tmpdir) / "grid.sqlite")
            search = GridSearch(
                dataset_id="grid_sma",
                data_loader=lambda _: self.bars,
                strategy_cls=SMACrossStrategy,
                base_config=self.config,
                catalog=catalog,
            )
            frame = search.run(
                GridSpec(
                    params={"fast": [4, 5], "slow": [18, 20]},
                    timeframes=["1 minutes"],
                    horizons=[(None, None)],
                    execution_mode=ExecutionMode.AUTO,
                ),
                make_heatmap=False,
            )
        self.assertEqual(sorted(frame["resolved_execution_mode"].unique().tolist()), ["vectorized"])
        self.assertEqual(sorted(frame["engine_impl"].unique().tolist()), ["vectorized"])

    def test_auto_single_run_uses_vectorized_for_supported_resampled_5m(self) -> None:
        bars_5m = self._resample_5m()
        orchestrator = ExecutionOrchestrator()
        auto_result = orchestrator.execute(
            ExecutionRequest(
                data=bars_5m,
                base_data=self.bars,
                dataset_id="sma_auto_5m",
                strategy_cls=SMACrossStrategy,
                strategy_params=self.params,
                config=replace(self.config, timeframe="5 minutes", base_execution=True),
                requested_execution_mode=ExecutionMode.AUTO,
            )
        )
        explicit_vectorized = orchestrator.execute(
            ExecutionRequest(
                data=bars_5m,
                base_data=None,
                dataset_id="sma_auto_5m",
                strategy_cls=SMACrossStrategy,
                strategy_params=self.params,
                config=replace(self.config, timeframe="5 minutes", base_execution=False),
                requested_execution_mode=ExecutionMode.VECTORIZED,
            )
        )
        self.assertEqual(auto_result.resolved_execution_mode.value, "vectorized")
        self.assertEqual(auto_result.engine_impl, "vectorized")
        self.assertTrue(np.allclose(auto_result.equity_curve.to_numpy(), explicit_vectorized.equity_curve.to_numpy(), atol=1e-8))
        self.assertAlmostEqual(auto_result.metrics.total_return, explicit_vectorized.metrics.total_return, places=10)

    def test_grid_search_auto_uses_vectorized_for_supported_resampled_5m(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            catalog = ResultCatalog(Path(tmpdir) / "grid.sqlite")

            def loader(tf: str) -> pd.DataFrame:
                if tf == "1 minutes":
                    return self.bars
                return self._resample_5m()

            search = GridSearch(
                dataset_id="grid_sma",
                data_loader=loader,
                strategy_cls=SMACrossStrategy,
                base_config=self.config,
                catalog=catalog,
            )
            frame = search.run(
                GridSpec(
                    params={"fast": [4], "slow": [18]},
                    timeframes=["5 minutes"],
                    horizons=[(None, None)],
                    execution_mode=ExecutionMode.AUTO,
                ),
                make_heatmap=False,
            )
        self.assertEqual(sorted(frame["resolved_execution_mode"].unique().tolist()), ["vectorized"])
        self.assertEqual(sorted(frame["engine_impl"].unique().tolist()), ["vectorized"])

    def test_grid_search_auto_falls_back_when_vectorized_settings_are_unsupported(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            catalog = ResultCatalog(Path(tmpdir) / "grid.sqlite")

            def loader(tf: str) -> pd.DataFrame:
                if tf == "1 minutes":
                    return self.bars
                return self._resample_5m()

            search = GridSearch(
                dataset_id="grid_sma",
                data_loader=loader,
                strategy_cls=SMACrossStrategy,
                base_config=replace(self.config, intrabar_sim=True),
                catalog=catalog,
            )
            frame = search.run(
                GridSpec(
                    params={"fast": [4], "slow": [18]},
                    timeframes=["5 minutes"],
                    horizons=[(None, None)],
                    execution_mode=ExecutionMode.AUTO,
                ),
                make_heatmap=False,
            )
        self.assertEqual(sorted(frame["resolved_execution_mode"].unique().tolist()), ["reference"])
        self.assertEqual(sorted(frame["engine_impl"].unique().tolist()), ["reference"])

    def test_grid_search_explicit_vectorized_uses_same_timeframe_resampled_bars(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            catalog = ResultCatalog(Path(tmpdir) / "grid.sqlite")

            def loader(tf: str) -> pd.DataFrame:
                if tf == "1 minutes":
                    return self.bars
                return self.bars.resample("5min", label="right", closed="right").agg(
                    {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
                ).dropna()

            search = GridSearch(
                dataset_id="grid_sma",
                data_loader=loader,
                strategy_cls=SMACrossStrategy,
                base_config=self.config,
                catalog=catalog,
            )
            frame = search.run(
                GridSpec(
                    params={"fast": [4], "slow": [18]},
                    timeframes=["5 minutes"],
                    horizons=[(None, None)],
                    execution_mode=ExecutionMode.VECTORIZED,
                ),
                make_heatmap=False,
            )
        self.assertEqual(sorted(frame["resolved_execution_mode"].unique().tolist()), ["vectorized"])
        self.assertEqual(sorted(frame["engine_impl"].unique().tolist()), ["vectorized"])

    def test_grid_search_chunked_vectorized_matches_unchunked(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            catalog = ResultCatalog(Path(tmpdir) / "grid.sqlite")
            params = {"fast": [4, 5, 6], "slow": [18, 20]}

            unchunked_search = GridSearch(
                dataset_id="grid_sma",
                data_loader=lambda _: self.bars,
                strategy_cls=SMACrossStrategy,
                base_config=self.config,
                catalog=catalog,
            )
            chunked_search = GridSearch(
                dataset_id="grid_sma",
                data_loader=lambda _: self.bars,
                strategy_cls=SMACrossStrategy,
                base_config=replace(self.config, vectorized_param_batch_size=2),
                catalog=catalog,
            )
            unchunked = unchunked_search.run(
                GridSpec(
                    params=params,
                    timeframes=["1 minutes"],
                    horizons=[(None, None)],
                    execution_mode=ExecutionMode.VECTORIZED,
                ),
                make_heatmap=False,
            ).sort_values(["fast", "slow"]).reset_index(drop=True)
            chunked = chunked_search.run(
                GridSpec(
                    params=params,
                    timeframes=["1 minutes"],
                    horizons=[(None, None)],
                    execution_mode=ExecutionMode.VECTORIZED,
                ),
                make_heatmap=False,
            ).sort_values(["fast", "slow"]).reset_index(drop=True)

        self.assertEqual(unchunked[["fast", "slow"]].to_dict("records"), chunked[["fast", "slow"]].to_dict("records"))
        self.assertTrue(np.allclose(unchunked["total_return"].to_numpy(), chunked["total_return"].to_numpy(), atol=1e-10))
        self.assertEqual(sorted(chunked["resolved_execution_mode"].unique().tolist()), ["vectorized"])
        self.assertEqual(sorted(chunked["engine_impl"].unique().tolist()), ["vectorized"])

    def test_chunked_vectorized_reuses_prepared_sma_context(self) -> None:
        counts = {"prepare": 0, "build": 0}
        original_prepare = SMACrossVectorizedAdapter.prepare_order_plan_context
        original_build = SMACrossVectorizedAdapter.build_order_plan_from_context

        def wrapped_prepare(self, data, param_grid, config):
            counts["prepare"] += 1
            return original_prepare(self, data, param_grid, config)

        def wrapped_build(self, data, param_grid, config, prepared_context=None):
            counts["build"] += 1
            return original_build(self, data, param_grid, config, prepared_context=prepared_context)

        SMACrossVectorizedAdapter.prepare_order_plan_context = wrapped_prepare
        SMACrossVectorizedAdapter.build_order_plan_from_context = wrapped_build
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                catalog = ResultCatalog(Path(tmpdir) / "grid.sqlite")
                search = GridSearch(
                    dataset_id="grid_sma",
                    data_loader=lambda _: self.bars,
                    strategy_cls=SMACrossStrategy,
                    base_config=replace(self.config, vectorized_param_batch_size=2),
                    catalog=catalog,
                )
                frame = search.run(
                    GridSpec(
                        params={"fast": [4, 5, 6], "slow": [18, 20]},
                        timeframes=["1 minutes"],
                        horizons=[(None, None)],
                        execution_mode=ExecutionMode.VECTORIZED,
                    ),
                    make_heatmap=False,
                )
        finally:
            SMACrossVectorizedAdapter.prepare_order_plan_context = original_prepare
            SMACrossVectorizedAdapter.build_order_plan_from_context = original_build

        self.assertEqual(len(frame), 6)
        self.assertEqual(counts["prepare"], 1)
        self.assertEqual(counts["build"], 3)

    def test_grid_search_exposes_batch_benchmark_for_chunked_vectorized_study(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            catalog = ResultCatalog(Path(tmpdir) / "grid.sqlite")
            search = GridSearch(
                dataset_id="grid_sma",
                data_loader=lambda _: self.bars,
                strategy_cls=SMACrossStrategy,
                base_config=replace(self.config, vectorized_param_batch_size=2),
                catalog=catalog,
            )
            frame = search.run(
                GridSpec(
                    params={"fast": [4, 5, 6], "slow": [18, 20]},
                    timeframes=["1 minutes"],
                    horizons=[(None, None)],
                    execution_mode=ExecutionMode.VECTORIZED,
                ),
                make_heatmap=False,
            )

        self.assertEqual(len(frame), 6)
        self.assertEqual(len(search.last_batch_benchmarks), 1)
        benchmark = search.last_batch_benchmarks[0]
        self.assertEqual(benchmark.dataset_id, "grid_sma")
        self.assertEqual(benchmark.strategy, "SMACrossStrategy")
        self.assertEqual(benchmark.resolved_execution_mode, ExecutionMode.VECTORIZED)
        self.assertEqual(benchmark.total_params, 6)
        self.assertEqual(benchmark.uncached_runs, 6)
        self.assertEqual(benchmark.chunk_count, 3)
        self.assertEqual(benchmark.chunk_sizes, (2, 2, 2))
        self.assertEqual(benchmark.effective_param_batch_size, 2)
        self.assertTrue(benchmark.prepared_context_reused)
        self.assertEqual(frame.attrs["batch_benchmarks"], (benchmark,))

    def test_independent_asset_grid_search_returns_combined_results_and_study_benchmark(self) -> None:
        bars_b = self.bars.copy()
        bars_b["open"] += 1.25
        bars_b["high"] += 1.25
        bars_b["low"] += 1.25
        bars_b["close"] += 1.25

        with tempfile.TemporaryDirectory() as tmpdir:
            catalog = ResultCatalog(Path(tmpdir) / "grid.sqlite")
            frame = run_independent_asset_grid_search(
                targets=[
                    IndependentAssetTarget(dataset_id="asset_a", data_loader=lambda _: self.bars),
                    IndependentAssetTarget(dataset_id="asset_b", data_loader=lambda _: bars_b),
                ],
                strategy_cls=SMACrossStrategy,
                base_config=replace(self.config, vectorized_param_batch_size=1),
                grid=GridSpec(
                    params={"fast": [4, 5], "slow": [18]},
                    timeframes=["1 minutes"],
                    horizons=[(None, None)],
                    execution_mode=ExecutionMode.VECTORIZED,
                ),
                catalog=catalog,
                make_heatmap=False,
            )

        self.assertEqual(sorted(frame["dataset_id"].unique().tolist()), ["asset_a", "asset_b"])
        self.assertEqual(len(frame), 4)
        study_benchmark = frame.attrs["study_benchmark"]
        self.assertEqual(study_benchmark.asset_count, 2)
        self.assertEqual(study_benchmark.total_runs, 4)
        self.assertEqual(study_benchmark.vectorized_batches, 2)
        self.assertEqual(study_benchmark.reference_batches, 0)
        self.assertEqual(study_benchmark.total_chunk_count, 4)
        self.assertEqual(study_benchmark.prepared_context_reused_batches, 2)
        self.assertEqual(len(study_benchmark.per_batch), 2)
        self.assertEqual(len(frame.attrs["batch_benchmarks"]), 2)


if __name__ == "__main__":
    unittest.main()
