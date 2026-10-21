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
from backtest_engine.sample_strategies import (
    AdaptiveOUKalmanZScoreStrategy,
    RegimeFilteredAdaptiveOUKalmanZScoreStrategy,
    compute_adaptive_ou_kalman_zscore_features,
    compute_prism_regime_filter_features,
    compute_regime_filtered_adaptive_ou_kalman_zscore_features,
    ZScoreMeanReversionStrategy,
)


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
        self.adaptive_params = {
            "ou_lookback": 80,
            "half_life_factor": 1.0,
            "min_half_life": 2.0,
            "max_half_life": 160.0,
            "std_half_life_factor": 1.0,
            "kalman_process_var_ratio": 0.02,
            "ou_mean_pull": 1.0,
            "z_smooth_type": "None",
            "z_smooth_len": 3,
            "exit_signal_mode": "Level",
            "long_entry_z": -0.6,
            "long_exit_z": -0.05,
            "target": 1.0,
        }
        self.regime_params = {
            **self.adaptive_params,
            "use_regime_filter": True,
            "regime_timeframe": "",
            "use_bbw_regime": True,
            "use_adx_regime": True,
            "use_er_regime": True,
            "bbw_len": 80,
            "bbw_percentile_threshold": 50.0,
            "adx_len": 14,
            "adx_smoothing_len": 14,
            "adx_threshold": 20.0,
            "er_len": 14,
            "er_threshold": 0.3,
            "exit_on_regime_loss": True,
            "use_atr_stop": True,
            "atr_stop_len": 14,
            "atr_stop_mult": 2.0,
            "atr_stop_update_mode": "Trailing",
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
        self.assertGreater(len(reference.trades), 0)
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

    def test_adaptive_ou_kalman_features_are_dynamic_and_finite(self) -> None:
        features = compute_adaptive_ou_kalman_zscore_features(self.bars, self.adaptive_params)
        ready = features.iloc[int(self.adaptive_params["ou_lookback"]) :]
        self.assertGreater(ready["z_score"].notna().sum(), 0)
        self.assertGreater(ready["kalman_mean"].notna().sum(), 0)
        self.assertTrue((ready["effective_half_life_bars"].dropna() > 0.0).all())
        self.assertGreater(ready["effective_half_life_bars"].dropna().nunique(), 1)
        expected_exit = (features["z_score"] >= float(self.adaptive_params["long_exit_z"])).fillna(False)
        self.assertTrue(features["long_exit_signal"].equals(expected_exit))

    def test_vectorized_matches_reference_for_adaptive_ou_kalman_zscore(self) -> None:
        orchestrator = ExecutionOrchestrator()
        reference = orchestrator.execute(
            ExecutionRequest(
                data=self.bars,
                base_data=self.bars,
                dataset_id="adaptive_ou_kalman_zscore_parity",
                strategy_cls=AdaptiveOUKalmanZScoreStrategy,
                strategy_params=self.adaptive_params,
                config=self.config,
                requested_execution_mode=ExecutionMode.REFERENCE,
            )
        )
        vectorized = orchestrator.execute(
            ExecutionRequest(
                data=self.bars,
                base_data=self.bars,
                dataset_id="adaptive_ou_kalman_zscore_parity",
                strategy_cls=AdaptiveOUKalmanZScoreStrategy,
                strategy_params=self.adaptive_params,
                config=self.config,
                requested_execution_mode=ExecutionMode.VECTORIZED,
            )
        )

        self.assertEqual(reference.logical_run_id, vectorized.logical_run_id)
        self.assertGreater(len(reference.trades), 0)
        self.assertEqual(len(reference.trades), len(vectorized.trades))
        self.assertTrue(np.allclose(reference.equity_curve.to_numpy(), vectorized.equity_curve.to_numpy(), atol=1e-8))
        self.assertAlmostEqual(reference.metrics.total_return, vectorized.metrics.total_return, places=10)

    def test_grid_search_auto_uses_vectorized_for_supported_adaptive_ou_kalman_zscore(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            catalog = ResultCatalog(Path(tmpdir) / "grid.sqlite")
            search = GridSearch(
                dataset_id="grid_adaptive_ou_kalman_zscore",
                data_loader=lambda _: self.bars,
                strategy_cls=AdaptiveOUKalmanZScoreStrategy,
                base_config=self.config,
                catalog=catalog,
            )
            frame = search.run(
                GridSpec(
                    params={
                        "ou_lookback": [70],
                        "half_life_factor": [1.0],
                        "min_half_life": [2.0],
                        "max_half_life": [160.0],
                        "std_half_life_factor": [1.0],
                        "kalman_process_var_ratio": [0.02],
                        "ou_mean_pull": [1.0],
                        "exit_signal_mode": ["Level"],
                        "long_entry_z": [-0.8, -0.6],
                        "long_exit_z": [-0.1, -0.05],
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

    def test_regime_filtered_adaptive_ou_kalman_features_gate_entries(self) -> None:
        features = compute_regime_filtered_adaptive_ou_kalman_zscore_features(self.bars, self.regime_params)
        self.assertIn("regime_is_choppy", features)
        self.assertIn("atr_stop_distance", features)
        self.assertIn("atr_stop_price", features)
        blocked = ~features["regime_is_choppy"].astype(bool)
        self.assertTrue((~features.loc[blocked, "long_entry_signal"].astype(bool)).all())
        self.assertGreater(features["atr_stop_distance"].notna().sum(), 0)
        self.assertTrue(features.loc[~features["long_state"].astype(bool), "atr_stop_price"].isna().all())

        trailing_params = {**self.regime_params, "use_regime_filter": False, "atr_stop_update_mode": "Trailing"}
        trailing_features = compute_regime_filtered_adaptive_ou_kalman_zscore_features(self.bars, trailing_params)
        stop = trailing_features["atr_stop_price"]
        state = trailing_features["long_state"].astype(bool)
        group_id = state.ne(state.shift(fill_value=False)).cumsum()
        for _, segment in stop[state].groupby(group_id[state]):
            diffs = segment.dropna().diff().dropna()
            self.assertTrue((diffs >= -1e-12).all())

    def test_regime_filtered_reference_buy_fills_remain_inside_regime(self) -> None:
        features = compute_regime_filtered_adaptive_ou_kalman_zscore_features(self.bars, self.regime_params)
        result = ExecutionOrchestrator().execute(
            ExecutionRequest(
                data=self.bars,
                base_data=self.bars,
                dataset_id="regime_filtered_ou_kalman_fill_filter",
                strategy_cls=RegimeFilteredAdaptiveOUKalmanZScoreStrategy,
                strategy_params=self.regime_params,
                config=self.config,
                requested_execution_mode=ExecutionMode.REFERENCE,
            )
        )
        buy_trades = [trade for trade in result.trades if trade.side == "buy"]
        self.assertGreater(len(buy_trades), 0)
        for trade in buy_trades:
            pos = features.index.searchsorted(pd.Timestamp(trade.timestamp), side="right") - 1
            self.assertGreaterEqual(pos, 0)
            self.assertTrue(bool(features["regime_is_choppy"].iloc[int(pos)]))

    def test_regime_filter_honors_day_timeframe(self) -> None:
        periods = 180
        index = pd.date_range("2024-01-01", periods=periods, freq="4h", tz="UTC")
        trend = np.linspace(0.0, 9.0, periods)
        wave = np.sin(np.arange(periods) / 3.0) * 1.8
        close = 100.0 + trend + wave
        open_ = np.roll(close, 1)
        open_[0] = close[0]
        bars = pd.DataFrame(
            {
                "open": open_,
                "high": np.maximum(open_, close) + 0.5,
                "low": np.minimum(open_, close) - 0.5,
                "close": close,
                "volume": 1000,
            },
            index=index,
        )
        params = {
            **self.regime_params,
            "use_regime_filter": False,
            "regime_timeframe": "1 day",
            "bbw_basis_len": 3,
            "bbw_len": 3,
            "er_len": 2,
        }

        daily = compute_prism_regime_filter_features(bars, params)
        same_chart = compute_prism_regime_filter_features(bars, {**params, "regime_timeframe": ""})

        self.assertEqual(len(daily), len(bars))
        self.assertFalse(daily["regime_er"].equals(same_chart["regime_er"]))
        per_day_unique = daily["regime_er"].groupby(daily.index.normalize()).nunique(dropna=True)
        self.assertTrue((per_day_unique <= 1).all())

    def test_regime_filter_rejects_invalid_timeframe(self) -> None:
        with self.assertRaises(ValueError):
            compute_prism_regime_filter_features(self.bars, {**self.regime_params, "regime_timeframe": "not a timeframe"})

    def test_vectorized_matches_reference_for_regime_filtered_ou_kalman_atr_stop(self) -> None:
        params = {
            **self.regime_params,
            "use_regime_filter": False,
            "long_entry_z": -0.6,
            "long_exit_z": -0.05,
            "atr_stop_mult": 0.75,
        }
        orchestrator = ExecutionOrchestrator()
        reference = orchestrator.execute(
            ExecutionRequest(
                data=self.bars,
                base_data=self.bars,
                dataset_id="regime_filtered_ou_kalman_atr_stop",
                strategy_cls=RegimeFilteredAdaptiveOUKalmanZScoreStrategy,
                strategy_params=params,
                config=self.config,
                requested_execution_mode=ExecutionMode.REFERENCE,
            )
        )
        vectorized = orchestrator.execute(
            ExecutionRequest(
                data=self.bars,
                base_data=self.bars,
                dataset_id="regime_filtered_ou_kalman_atr_stop",
                strategy_cls=RegimeFilteredAdaptiveOUKalmanZScoreStrategy,
                strategy_params=params,
                config=self.config,
                requested_execution_mode=ExecutionMode.VECTORIZED,
            )
        )
        self.assertGreater(len(reference.trades), 0)
        self.assertEqual(len(reference.trades), len(vectorized.trades))
        self.assertTrue(np.allclose(reference.equity_curve.to_numpy(), vectorized.equity_curve.to_numpy(), atol=1e-8))
        self.assertAlmostEqual(reference.metrics.total_return, vectorized.metrics.total_return, places=10)

    def test_auto_single_run_uses_vectorized_for_regime_filtered_ou_kalman_atr_stop(self) -> None:
        params = {
            **self.regime_params,
            "use_regime_filter": False,
            "long_entry_z": -0.6,
            "long_exit_z": -0.05,
            "atr_stop_mult": 0.75,
        }
        result = ExecutionOrchestrator().execute(
            ExecutionRequest(
                data=self.bars,
                base_data=self.bars,
                dataset_id="regime_filtered_ou_kalman_auto",
                strategy_cls=RegimeFilteredAdaptiveOUKalmanZScoreStrategy,
                strategy_params=params,
                config=self.config,
                requested_execution_mode=ExecutionMode.AUTO,
            )
        )
        self.assertEqual(result.resolved_execution_mode.value, "vectorized")
        self.assertEqual(result.engine_impl, "vectorized")
        self.assertIsNone(result.fallback_reason)
        self.assertGreater(len(result.trades), 0)

    def test_regime_filtered_ou_kalman_places_atr_stop_after_fill(self) -> None:
        params = {**self.regime_params, "use_regime_filter": False}
        strategy = RegimeFilteredAdaptiveOUKalmanZScoreStrategy(**params)
        strategy.initialize(self.bars)
        ready = strategy.features["atr_stop_distance"].dropna()
        self.assertFalse(ready.empty)
        timestamp = ready.index[0]

        class FakeBroker:
            position_qty = 10.0
            avg_price = 100.0

            def __init__(self) -> None:
                self.cancelled: list[str | None] = []
                self.stop_orders: list[tuple[float, float, str | None]] = []

            def cancel_orders(self, tag: str | None = None) -> None:
                self.cancelled.append(tag)

            def sell_stop(self, qty: float, stop_price: float, tag: str | None = None) -> None:
                self.stop_orders.append((qty, stop_price, tag))

        broker = FakeBroker()
        strategy.on_after_fill(timestamp, self.bars.loc[timestamp], broker)
        self.assertEqual(len(broker.stop_orders), 1)
        qty, stop_price, tag = broker.stop_orders[0]
        self.assertEqual(qty, abs(broker.position_qty))
        self.assertLess(stop_price, broker.avg_price)
        self.assertEqual(tag, "regime_ou_kalman_atr_stop")

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
