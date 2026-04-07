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
from backtest_engine.grid_search import GridSearch, GridSpec
from backtest_engine.sample_strategies import ZScoreMeanReversionStrategy


def _make_mean_reverting_bars(periods: int = 320) -> pd.DataFrame:
    index = pd.date_range("2024-01-03 14:30", periods=periods, freq="1min", tz="UTC")
    core = 100.0 + np.sin(np.arange(periods) / 5.0) * 2.2 + np.cos(np.arange(periods) / 17.0) * 0.8
    drift = np.linspace(-0.5, 0.5, periods)
    close = core + drift
    open_ = np.roll(close, 1)
    open_[0] = close[0] - 0.2
    open_ += np.cos(np.arange(periods) / 8.0) * 0.1
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + 0.35,
            "low": np.minimum(open_, close) - 0.35,
            "close": close,
            "volume": 1000,
        },
        index=index,
    )


class VectorizedZScoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.bars = _make_mean_reverting_bars()
        self.config = BacktestConfig(
            timeframe="1 minutes",
            use_cache=False,
            base_execution=False,
            allow_short=False,
            fee_rate=0.0002,
            slippage=0.0001,
        )
        self.params = {
            "half_life_lookback": 80,
            "half_life_factor": 1.5,
            "std_len": 20,
            "vol_len": 14,
            "atr_mult": 1.0,
            "long_entry_z": -1.0,
            "long_exit_z": 0.0,
            "target": 1.0,
        }

    def _resample_5m(self) -> pd.DataFrame:
        return self.bars.resample("5min", label="right", closed="right").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()

    def test_vectorized_matches_reference_for_zscore(self) -> None:
        orchestrator = ExecutionOrchestrator()
        reference = orchestrator.execute(
            ExecutionRequest(
                data=self.bars,
                base_data=self.bars,
                dataset_id="zscore_parity",
                strategy_cls=ZScoreMeanReversionStrategy,
                strategy_params=self.params,
                config=self.config,
                requested_execution_mode=ExecutionMode.REFERENCE,
            )
        )
        vectorized = orchestrator.execute(
            ExecutionRequest(
                data=self.bars,
                base_data=self.bars,
                dataset_id="zscore_parity",
                strategy_cls=ZScoreMeanReversionStrategy,
                strategy_params=self.params,
                config=self.config,
                requested_execution_mode=ExecutionMode.VECTORIZED,
            )
        )

        self.assertEqual(reference.logical_run_id, vectorized.logical_run_id)
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

    def test_vectorized_matches_reference_with_fill_ratio_for_zscore(self) -> None:
        config = replace(self.config, fill_ratio=0.5)
        orchestrator = ExecutionOrchestrator()
        reference = orchestrator.execute(
            ExecutionRequest(
                data=self.bars,
                base_data=self.bars,
                dataset_id="zscore_fill_ratio",
                strategy_cls=ZScoreMeanReversionStrategy,
                strategy_params=self.params,
                config=config,
                requested_execution_mode=ExecutionMode.REFERENCE,
            )
        )
        vectorized = orchestrator.execute(
            ExecutionRequest(
                data=self.bars,
                base_data=self.bars,
                dataset_id="zscore_fill_ratio",
                strategy_cls=ZScoreMeanReversionStrategy,
                strategy_params=self.params,
                config=config,
                requested_execution_mode=ExecutionMode.VECTORIZED,
            )
        )
        self.assertEqual(len(reference.trades), len(vectorized.trades))
        self.assertTrue(np.allclose(reference.equity_curve.to_numpy(), vectorized.equity_curve.to_numpy(), atol=1e-8))
        self.assertAlmostEqual(reference.metrics.total_return, vectorized.metrics.total_return, places=10)

    def test_vectorized_matches_reference_with_fill_on_close_for_zscore(self) -> None:
        config = replace(self.config, fill_on_close=True)
        orchestrator = ExecutionOrchestrator()
        reference = orchestrator.execute(
            ExecutionRequest(
                data=self.bars,
                base_data=self.bars,
                dataset_id="zscore_fill_on_close",
                strategy_cls=ZScoreMeanReversionStrategy,
                strategy_params=self.params,
                config=config,
                requested_execution_mode=ExecutionMode.REFERENCE,
            )
        )
        vectorized = orchestrator.execute(
            ExecutionRequest(
                data=self.bars,
                base_data=self.bars,
                dataset_id="zscore_fill_on_close",
                strategy_cls=ZScoreMeanReversionStrategy,
                strategy_params=self.params,
                config=config,
                requested_execution_mode=ExecutionMode.VECTORIZED,
            )
        )
        self.assertEqual(len(reference.trades), len(vectorized.trades))
        self.assertTrue(np.allclose(reference.equity_curve.to_numpy(), vectorized.equity_curve.to_numpy(), atol=1e-8))
        self.assertAlmostEqual(reference.metrics.total_return, vectorized.metrics.total_return, places=10)

    def test_grid_search_auto_uses_vectorized_for_supported_zscore(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            catalog = ResultCatalog(Path(tmpdir) / "grid.sqlite")
            search = GridSearch(
                dataset_id="grid_zscore",
                data_loader=lambda _: self.bars,
                strategy_cls=ZScoreMeanReversionStrategy,
                base_config=self.config,
                catalog=catalog,
            )
            frame = search.run(
                GridSpec(
                    params={
                        "half_life_lookback": [60],
                        "half_life_factor": [1.5],
                        "std_len": [20],
                        "vol_len": [14],
                        "atr_mult": [1.0],
                        "long_entry_z": [-1.2, -1.0],
                        "long_exit_z": [-0.25, 0.0],
                        "target": [1.0],
                    },
                    timeframes=["1 minutes"],
                    horizons=[(None, None)],
                    execution_mode=ExecutionMode.AUTO,
                ),
                make_heatmap=False,
            )
        self.assertEqual(sorted(frame["resolved_execution_mode"].unique().tolist()), ["vectorized"])
        self.assertEqual(sorted(frame["engine_impl"].unique().tolist()), ["vectorized"])

    def test_grid_search_explicit_vectorized_supports_resampled_5m_zscore(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            catalog = ResultCatalog(Path(tmpdir) / "grid.sqlite")

            def loader(tf: str) -> pd.DataFrame:
                if tf == "1 minutes":
                    return self.bars
                return self._resample_5m()

            search = GridSearch(
                dataset_id="grid_zscore",
                data_loader=loader,
                strategy_cls=ZScoreMeanReversionStrategy,
                base_config=self.config,
                catalog=catalog,
            )
            frame = search.run(
                GridSpec(
                    params={
                        "half_life_lookback": [60],
                        "half_life_factor": [1.5],
                        "std_len": [20],
                        "vol_len": [14],
                        "atr_mult": [1.0],
                        "long_entry_z": [-1.2],
                        "long_exit_z": [0.0],
                        "target": [1.0],
                    },
                    timeframes=["5 minutes"],
                    horizons=[(None, None)],
                    execution_mode=ExecutionMode.VECTORIZED,
                ),
                make_heatmap=False,
            )
        self.assertEqual(sorted(frame["resolved_execution_mode"].unique().tolist()), ["vectorized"])
        self.assertEqual(sorted(frame["engine_impl"].unique().tolist()), ["vectorized"])

    def test_grid_search_auto_uses_vectorized_for_supported_resampled_5m_zscore(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            catalog = ResultCatalog(Path(tmpdir) / "grid.sqlite")

            def loader(tf: str) -> pd.DataFrame:
                if tf == "1 minutes":
                    return self.bars
                return self._resample_5m()

            search = GridSearch(
                dataset_id="grid_zscore",
                data_loader=loader,
                strategy_cls=ZScoreMeanReversionStrategy,
                base_config=self.config,
                catalog=catalog,
            )
            frame = search.run(
                GridSpec(
                    params={
                        "half_life_lookback": [60],
                        "half_life_factor": [1.5],
                        "std_len": [20],
                        "vol_len": [14],
                        "atr_mult": [1.0],
                        "long_entry_z": [-1.2],
                        "long_exit_z": [0.0],
                        "target": [1.0],
                    },
                    timeframes=["5 minutes"],
                    horizons=[(None, None)],
                    execution_mode=ExecutionMode.AUTO,
                ),
                make_heatmap=False,
            )
        self.assertEqual(sorted(frame["resolved_execution_mode"].unique().tolist()), ["vectorized"])
        self.assertEqual(sorted(frame["engine_impl"].unique().tolist()), ["vectorized"])

    def test_auto_single_run_uses_vectorized_for_supported_resampled_5m_zscore(self) -> None:
        bars_5m = self._resample_5m()
        orchestrator = ExecutionOrchestrator()
        auto_result = orchestrator.execute(
            ExecutionRequest(
                data=bars_5m,
                base_data=self.bars,
                dataset_id="zscore_auto_5m",
                strategy_cls=ZScoreMeanReversionStrategy,
                strategy_params=self.params,
                config=replace(self.config, timeframe="5 minutes", base_execution=True),
                requested_execution_mode=ExecutionMode.AUTO,
            )
        )
        explicit_vectorized = orchestrator.execute(
            ExecutionRequest(
                data=bars_5m,
                base_data=None,
                dataset_id="zscore_auto_5m",
                strategy_cls=ZScoreMeanReversionStrategy,
                strategy_params=self.params,
                config=replace(self.config, timeframe="5 minutes", base_execution=False),
                requested_execution_mode=ExecutionMode.VECTORIZED,
            )
        )
        self.assertEqual(auto_result.resolved_execution_mode.value, "vectorized")
        self.assertEqual(auto_result.engine_impl, "vectorized")
        self.assertTrue(np.allclose(auto_result.equity_curve.to_numpy(), explicit_vectorized.equity_curve.to_numpy(), atol=1e-8))
        self.assertAlmostEqual(auto_result.metrics.total_return, explicit_vectorized.metrics.total_return, places=10)


if __name__ == "__main__":
    unittest.main()
