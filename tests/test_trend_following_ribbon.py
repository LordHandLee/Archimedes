from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from backtest_engine.engine import BacktestConfig
from backtest_engine.execution import ExecutionMode, ExecutionOrchestrator, ExecutionRequest
from backtest_engine.chart_snapshot import ChartSnapshotExporter
from backtest_engine.sample_strategies import (
    TrendFollowingRibbonStrategy,
    compute_trend_following_ribbon_features,
)


def _make_ribbon_bars(periods: int = 420) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=periods, freq="1D", tz="UTC")
    phase = np.arange(periods, dtype=float)
    close = 100.0 + np.sin(phase / 10.0) * 12.0 + np.sin(phase / 35.0) * 8.0 + np.linspace(-10.0, 15.0, periods)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) + 1.5
    low = np.minimum(open_, close) - 1.5
    volume = 1000.0 + (phase % 20.0)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )


def _ribbon_params() -> dict:
    return {
        "ma_type": "EMA",
        "regime_timeframe": "",
        "regime_fast_len": 10,
        "regime_slow_len": 30,
        "entry_ribbon_threshold": 5,
        "exit_ribbon_threshold": 2,
        "trend_check_period": 5,
        "trend_channel_rate_pct": 0.1,
        "use_linear_regression": False,
        "distance_ma_len": 10,
        "distance_atr_len": 5,
        "min_price_distance": -99.0,
        "use_short_atr_trail": True,
        "short_atr_len": 5,
        "short_atr_mult": 2.0,
        "target": 0.5,
    }


class TrendFollowingRibbonTest(unittest.TestCase):
    def test_features_emit_ribbon_signals(self) -> None:
        features = compute_trend_following_ribbon_features(_make_ribbon_bars(), _ribbon_params())

        self.assertTrue(
            {
                "ma_5",
                "ma_100",
                "trend_5",
                "bull_count",
                "bear_count",
                "regime_fast_sma",
                "regime_slow_sma",
                "ribbon_bull_regime",
                "ribbon_bear_regime",
                "new_ribbon_bull_regime",
                "distance_ma",
                "short_atr_trail_stop",
                "hard_stop_loss_price",
                "long_locked_out_bull_regime",
            }.issubset(features.columns)
        )
        self.assertGreater(features["long_entry_signal"].sum(), 0)
        self.assertGreater(features["short_entry_signal"].sum(), 0)
        self.assertGreaterEqual(features["short_atr_trail_hit"].sum(), 1)
        self.assertTrue(((features["bull_count"] >= 0) & (features["bull_count"] <= 20)).all())
        self.assertTrue(((features["bear_count"] >= 0) & (features["bear_count"] <= 20)).all())

    def test_long_entries_lock_after_close_below_regime_slow_sma_until_new_bull_regime(self) -> None:
        bars = _make_ribbon_bars()
        features = compute_trend_following_ribbon_features(bars, _ribbon_params())
        locked = features["long_locked_out_bull_regime"].astype(bool)
        below_slow = bars["close"] < features["regime_slow_sma"]

        self.assertGreater(locked.sum(), 0)
        self.assertEqual(int((features["long_entry_signal"] & locked).sum()), 0)
        lock_start = locked & ~locked.shift(1).fillna(False).astype(bool)
        self.assertTrue(below_slow.loc[lock_start].all())

        reset_candidates = (
            features["new_ribbon_bull_regime"].astype(bool)
            & (bars["close"] >= features["regime_slow_sma"])
            & locked.shift(1).fillna(False).astype(bool)
        )
        self.assertGreater(reset_candidates.sum(), 0)
        self.assertFalse(locked.loc[reset_candidates].any())

    def test_regime_lockout_can_be_disabled_while_slow_sma_exit_remains(self) -> None:
        bars = _make_ribbon_bars()
        features = compute_trend_following_ribbon_features(
            bars,
            {**_ribbon_params(), "use_regime_lockout": False},
        )

        self.assertEqual(int(features["long_locked_out_bull_regime"].sum()), 0)
        self.assertGreater(features["long_entry_signal"].sum(), 0)
        slow_exit = features["long_exit_signal"].astype(bool) & features["close_below_regime_slow_sma"].astype(bool)
        self.assertGreater(slow_exit.sum(), 0)

    def test_long_exit_when_close_below_regime_slow_sma(self) -> None:
        bars = _make_ribbon_bars()
        features = compute_trend_following_ribbon_features(bars, _ribbon_params())
        slow_exit = features["long_exit_signal"].astype(bool) & features["close_below_regime_slow_sma"].astype(bool)

        self.assertGreater(slow_exit.sum(), 0)
        self.assertTrue(features.loc[slow_exit, "long_locked_out_bull_regime"].astype(bool).all())

    def test_optional_hard_stop_loss_emits_exit_signal(self) -> None:
        params = {
            **_ribbon_params(),
            "use_hard_stop_loss": True,
            "hard_stop_loss_pct": 0.001,
        }
        features = compute_trend_following_ribbon_features(_make_ribbon_bars(), params)
        hard_stop_exit = (
            (features["long_exit_signal"].astype(bool) & features["long_hard_stop_loss_hit"].astype(bool))
            | (features["short_exit_signal"].astype(bool) & features["short_hard_stop_loss_hit"].astype(bool))
        )

        self.assertGreater(hard_stop_exit.sum(), 0)

    def test_chart_snapshot_uses_htf_sma_names_without_background_role(self) -> None:
        bars = _make_ribbon_bars()
        overlays, panes, styles = ChartSnapshotExporter._build_portfolio_asset_strategy_series(
            bars,
            [(None, "TrendFollowingRibbonStrategy", _ribbon_params())],
        )

        self.assertIn("HTF Fast SMA", overlays)
        self.assertIn("HTF Slow SMA", overlays)
        self.assertNotIn("Regime Fast SMA", overlays)
        self.assertNotIn("Regime Slow SMA", overlays)
        self.assertIn("Bull Count", panes)
        self.assertIn("Bear Count", panes)
        self.assertFalse(any(style.get("render_role") == "background_band" for style in styles.values()))

    def test_vectorized_matches_reference(self) -> None:
        bars = _make_ribbon_bars()
        params = _ribbon_params()
        config = BacktestConfig(
            timeframe="1 days",
            use_cache=False,
            base_execution=False,
            allow_short=True,
            fee_rate=0.0,
            slippage=0.0,
            max_gross_leverage=1.0,
        )
        orchestrator = ExecutionOrchestrator()

        reference = orchestrator.execute(
            ExecutionRequest(
                data=bars,
                base_data=bars,
                dataset_id="trend_following_ribbon_reference",
                strategy_cls=TrendFollowingRibbonStrategy,
                strategy_params=params,
                config=config,
                requested_execution_mode=ExecutionMode.REFERENCE,
            )
        )
        vectorized = orchestrator.execute(
            ExecutionRequest(
                data=bars,
                base_data=bars,
                dataset_id="trend_following_ribbon_vectorized",
                strategy_cls=TrendFollowingRibbonStrategy,
                strategy_params=params,
                config=replace(config),
                requested_execution_mode=ExecutionMode.VECTORIZED,
            )
        )

        self.assertEqual(reference.resolved_execution_mode.value, "reference")
        self.assertEqual(vectorized.resolved_execution_mode.value, "vectorized")
        self.assertGreater(len(reference.trades), 0)
        self.assertEqual(len(reference.trades), len(vectorized.trades))
        self.assertTrue(np.allclose(reference.equity_curve.to_numpy(), vectorized.equity_curve.to_numpy()))
        self.assertAlmostEqual(reference.metrics.total_return, vectorized.metrics.total_return, places=12)


if __name__ == "__main__":
    unittest.main()
