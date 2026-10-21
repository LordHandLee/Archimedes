from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from backtest_engine.engine import BacktestConfig
from backtest_engine.execution import ExecutionMode, ExecutionOrchestrator, ExecutionRequest
from backtest_engine.sample_strategies import (
    SMATrendRegimeStrategy,
    compute_sma_trend_regime_features,
)


def _make_trend_regime_bars(periods: int = 260) -> pd.DataFrame:
    index = pd.date_range("2024-01-02 14:30", periods=periods, freq="1min", tz="UTC")
    cycle = np.sin(np.arange(periods) / 11.0) * 4.0
    drift = np.sin(np.arange(periods) / 47.0) * 2.0
    close = 100.0 + cycle + drift
    open_ = np.roll(close, 1)
    open_[0] = close[0] - 0.15
    high = np.maximum(open_, close) + 0.35
    low = np.minimum(open_, close) - 0.35
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1000.0,
        },
        index=index,
    )


def _make_stop_reentry_bars() -> pd.DataFrame:
    index = pd.date_range("2024-01-02 14:30", periods=11, freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 120.0, 118.0, 119.0, 120.0, 121.0],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 121.0, 119.0, 120.0, 121.0, 122.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 119.0, 117.0, 118.0, 119.0, 120.0],
            "close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 120.0, 118.0, 119.0, 120.0, 121.0],
            "volume": [1000.0] * 11,
        },
        index=index,
    )


class SMATrendRegimeTest(unittest.TestCase):
    def test_features_emit_flip_entries(self) -> None:
        bars = _make_trend_regime_bars()
        params = {
            "fastest_len": 3,
            "fast_len": 5,
            "med_len": 7,
            "slow_len": 9,
            "slowest_len": 14,
            "atr_stop_len": 5,
            "atr_stop_mult": 2.0,
            "target": 0.5,
        }

        features = compute_sma_trend_regime_features(bars, params)

        self.assertTrue({"fastest_sma", "slowest_sma", "trend_state", "atr_stop_price"}.issubset(features.columns))
        self.assertGreater(features["long_entry_signal"].sum(), 0)
        self.assertGreater(features["short_entry_signal"].sum(), 0)
        self.assertTrue((features.loc[features["long_entry_signal"], "trend_state"].astype(int) == 1).all())
        self.assertTrue((features.loc[features["short_entry_signal"], "trend_state"].astype(int) == -1).all())

    def test_places_long_and_short_atr_stops(self) -> None:
        bars = _make_trend_regime_bars()
        params = {
            "fastest_len": 3,
            "slowest_len": 14,
            "atr_stop_len": 5,
            "atr_stop_mult": 2.0,
            "target": 0.5,
        }
        strategy = SMATrendRegimeStrategy(**params)
        strategy.initialize(bars)
        timestamp = strategy.features["atr_stop_distance"].dropna().index[0]

        class FakeBroker:
            def __init__(self, position_qty: float) -> None:
                self.position_qty = position_qty
                self.avg_price = 100.0
                self.cancelled: list[str | None] = []
                self.sell_stops: list[tuple[float, float, str | None]] = []
                self.buy_stops: list[tuple[float, float, str | None]] = []

            def cancel_orders(self, tag: str | None = None) -> None:
                self.cancelled.append(tag)

            def sell_stop(self, qty: float, stop_price: float, tag: str | None = None) -> None:
                self.sell_stops.append((qty, stop_price, tag))

            def buy_stop(self, qty: float, stop_price: float, tag: str | None = None) -> None:
                self.buy_stops.append((qty, stop_price, tag))

        long_broker = FakeBroker(10.0)
        strategy.on_after_fill(timestamp, bars.loc[timestamp], long_broker)
        self.assertEqual(len(long_broker.sell_stops), 1)
        self.assertEqual(long_broker.sell_stops[0][0], abs(long_broker.position_qty))
        self.assertLess(long_broker.sell_stops[0][1], long_broker.avg_price)
        self.assertEqual(long_broker.sell_stops[0][2], "sma_trend_regime_atr_stop")

        short_broker = FakeBroker(-10.0)
        strategy.on_after_fill(timestamp, bars.loc[timestamp], short_broker)
        self.assertEqual(len(short_broker.buy_stops), 1)
        self.assertEqual(short_broker.buy_stops[0][0], abs(short_broker.position_qty))
        self.assertGreater(short_broker.buy_stops[0][1], short_broker.avg_price)
        self.assertEqual(short_broker.buy_stops[0][2], "sma_trend_regime_atr_stop")

    def test_vectorized_matches_reference_with_atr_stops(self) -> None:
        bars = _make_trend_regime_bars()
        config = BacktestConfig(
            timeframe="1 minutes",
            use_cache=False,
            base_execution=False,
            allow_short=True,
            fee_rate=0.0002,
            slippage=0.0001,
            max_gross_leverage=1.0,
        )
        params = {
            "fastest_len": 3,
            "fast_len": 5,
            "med_len": 7,
            "slow_len": 9,
            "slowest_len": 14,
            "atr_stop_len": 5,
            "atr_stop_mult": 1.4,
            "target": 0.4,
        }

        orchestrator = ExecutionOrchestrator()
        reference = orchestrator.execute(
            ExecutionRequest(
                data=bars,
                base_data=bars,
                dataset_id="sma_trend_regime",
                strategy_cls=SMATrendRegimeStrategy,
                strategy_params=params,
                config=config,
                requested_execution_mode=ExecutionMode.REFERENCE,
            )
        )
        vectorized = orchestrator.execute(
            ExecutionRequest(
                data=bars,
                base_data=bars,
                dataset_id="sma_trend_regime",
                strategy_cls=SMATrendRegimeStrategy,
                strategy_params=params,
                config=replace(config),
                requested_execution_mode=ExecutionMode.VECTORIZED,
            )
        )

        self.assertEqual(reference.resolved_execution_mode.value, "reference")
        self.assertEqual(vectorized.resolved_execution_mode.value, "vectorized")
        self.assertGreater(len(reference.trades), 0)
        self.assertEqual(len(reference.trades), len(vectorized.trades))
        self.assertTrue(np.allclose(reference.equity_curve.to_numpy(), vectorized.equity_curve.to_numpy(), atol=1e-8))
        self.assertAlmostEqual(reference.metrics.total_return, vectorized.metrics.total_return, places=10)

    def test_auto_uses_vectorized(self) -> None:
        bars = _make_trend_regime_bars()
        result = ExecutionOrchestrator().execute(
            ExecutionRequest(
                data=bars,
                base_data=bars,
                dataset_id="sma_trend_regime_auto",
                strategy_cls=SMATrendRegimeStrategy,
                strategy_params={
                    "fastest_len": 3,
                    "slowest_len": 14,
                    "atr_stop_len": 5,
                    "atr_stop_mult": 1.4,
                    "target": 0.4,
                },
                config=BacktestConfig(
                    timeframe="1 minutes",
                    use_cache=False,
                    base_execution=False,
                    allow_short=True,
                ),
                requested_execution_mode=ExecutionMode.AUTO,
            )
        )

        self.assertEqual(result.resolved_execution_mode.value, "vectorized")
        self.assertEqual(result.engine_impl, "vectorized")
        self.assertIsNone(result.fallback_reason)

    def test_reenters_immediately_after_stop_when_regime_remains_valid(self) -> None:
        bars = _make_stop_reentry_bars()
        config = BacktestConfig(
            timeframe="1 minutes",
            use_cache=False,
            base_execution=False,
            allow_short=False,
            fee_rate=0.0,
            slippage=0.0,
        )
        params = {
            "fastest_len": 1,
            "slowest_len": 5,
            "atr_stop_len": 1,
            "atr_stop_mult": 1.0,
            "enter_on_initial_state": True,
            "target": 0.5,
        }
        orchestrator = ExecutionOrchestrator()
        with_reentry = orchestrator.execute(
            ExecutionRequest(
                data=bars,
                base_data=bars,
                dataset_id="sma_trend_reentry",
                strategy_cls=SMATrendRegimeStrategy,
                strategy_params={**params, "reenter_after_stop": True},
                config=config,
                requested_execution_mode=ExecutionMode.REFERENCE,
            )
        )
        without_reentry = orchestrator.execute(
            ExecutionRequest(
                data=bars,
                base_data=bars,
                dataset_id="sma_trend_no_reentry",
                strategy_cls=SMATrendRegimeStrategy,
                strategy_params={**params, "reenter_after_stop": False},
                config=config,
                requested_execution_mode=ExecutionMode.REFERENCE,
            )
        )

        self.assertEqual([trade.side for trade in without_reentry.trades], ["buy", "sell"])
        self.assertEqual([trade.side for trade in with_reentry.trades], ["buy", "sell", "buy"])
        self.assertEqual(with_reentry.trades[1].timestamp, bars.index[7])
        self.assertEqual(with_reentry.trades[2].timestamp, bars.index[8])

        vectorized = orchestrator.execute(
            ExecutionRequest(
                data=bars,
                base_data=bars,
                dataset_id="sma_trend_reentry_vectorized",
                strategy_cls=SMATrendRegimeStrategy,
                strategy_params={**params, "reenter_after_stop": True},
                config=replace(config),
                requested_execution_mode=ExecutionMode.VECTORIZED,
            )
        )
        self.assertEqual([trade.side for trade in vectorized.trades], ["buy", "sell", "buy"])
        self.assertEqual(vectorized.trades[1].timestamp, bars.index[7])
        self.assertEqual(vectorized.trades[2].timestamp, bars.index[8])
        self.assertTrue(np.allclose(with_reentry.equity_curve.to_numpy(), vectorized.equity_curve.to_numpy()))

    def test_reentry_forces_after_fill_recalc_when_user_setting_is_off(self) -> None:
        bars = _make_stop_reentry_bars()
        config = BacktestConfig(
            timeframe="1 minutes",
            use_cache=False,
            base_execution=False,
            allow_short=False,
            fee_rate=0.0,
            slippage=0.0,
            recalc_on_fill=False,
        )
        params = {
            "fastest_len": 1,
            "slowest_len": 5,
            "atr_stop_len": 1,
            "atr_stop_mult": 1.0,
            "enter_on_initial_state": True,
            "reenter_after_stop": True,
            "target": 0.5,
        }
        orchestrator = ExecutionOrchestrator()

        reference = orchestrator.execute(
            ExecutionRequest(
                data=bars,
                base_data=bars,
                dataset_id="sma_trend_reentry_forces_recalc_ref",
                strategy_cls=SMATrendRegimeStrategy,
                strategy_params=params,
                config=config,
                requested_execution_mode=ExecutionMode.REFERENCE,
            )
        )
        vectorized = orchestrator.execute(
            ExecutionRequest(
                data=bars,
                base_data=bars,
                dataset_id="sma_trend_reentry_forces_recalc_vec",
                strategy_cls=SMATrendRegimeStrategy,
                strategy_params=params,
                config=replace(config),
                requested_execution_mode=ExecutionMode.VECTORIZED,
            )
        )
        batch = orchestrator.execute_param_grid(
            data=bars,
            base_data=bars,
            dataset_id="sma_trend_reentry_forces_recalc_grid",
            strategy_cls=SMATrendRegimeStrategy,
            param_grid=[params],
            config=replace(config),
            requested_execution_mode=ExecutionMode.VECTORIZED,
        )[0]

        self.assertEqual([trade.side for trade in reference.trades], ["buy", "sell", "buy"])
        self.assertEqual([trade.side for trade in vectorized.trades], ["buy", "sell", "buy"])
        self.assertEqual([trade.side for trade in batch.trades], ["buy", "sell", "buy"])
        self.assertTrue(np.allclose(reference.equity_curve.to_numpy(), vectorized.equity_curve.to_numpy()))

    def test_stop_recapture_mode_waits_for_close_to_reclaim_stop(self) -> None:
        bars = _make_stop_reentry_bars()
        config = BacktestConfig(
            timeframe="1 minutes",
            use_cache=False,
            base_execution=False,
            allow_short=False,
            fee_rate=0.0,
            slippage=0.0,
        )
        params = {
            "fastest_len": 1,
            "slowest_len": 5,
            "atr_stop_len": 1,
            "atr_stop_mult": 1.0,
            "enter_on_initial_state": True,
            "reenter_after_stop": True,
            "stop_reentry_mode": "StopRecapture",
            "target": 0.5,
        }

        result = ExecutionOrchestrator().execute(
            ExecutionRequest(
                data=bars,
                base_data=bars,
                dataset_id="sma_trend_recapture_mode",
                strategy_cls=SMATrendRegimeStrategy,
                strategy_params=params,
                config=config,
                requested_execution_mode=ExecutionMode.REFERENCE,
            )
        )

        self.assertEqual([trade.side for trade in result.trades], ["buy", "sell", "buy"])
        self.assertEqual(result.trades[1].timestamp, bars.index[7])
        self.assertEqual(result.trades[2].timestamp, bars.index[10])

    def test_auto_uses_vectorized_for_stop_reentry(self) -> None:
        bars = _make_stop_reentry_bars()
        result = ExecutionOrchestrator().execute(
            ExecutionRequest(
                data=bars,
                base_data=bars,
                dataset_id="sma_trend_reentry_auto",
                strategy_cls=SMATrendRegimeStrategy,
                strategy_params={
                    "fastest_len": 1,
                    "slowest_len": 5,
                    "atr_stop_len": 1,
                    "atr_stop_mult": 1.0,
                    "enter_on_initial_state": True,
                    "reenter_after_stop": True,
                    "target": 0.5,
                },
                config=BacktestConfig(
                    timeframe="1 minutes",
                    use_cache=False,
                    base_execution=False,
                    allow_short=False,
                ),
                requested_execution_mode=ExecutionMode.AUTO,
            )
        )

        self.assertEqual(result.resolved_execution_mode.value, "vectorized")
        self.assertEqual(result.engine_impl, "vectorized")
        self.assertIsNone(result.fallback_reason)
        self.assertEqual([trade.side for trade in result.trades], ["buy", "sell", "buy"])


if __name__ == "__main__":
    unittest.main()
