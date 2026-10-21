from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np
import pandas as pd

from backtest_engine.engine import BacktestConfig
from backtest_engine.execution import ExecutionMode, ExecutionOrchestrator, ExecutionRequest
from backtest_engine.sample_strategies import (
    SMARegimeFilteredZScoreMeanReversionStrategy,
    compute_sma_regime_filtered_zscore_mean_reversion_features,
)


def _make_regime_zscore_bars(periods: int = 420) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=periods, freq="1D", tz="UTC")
    phase = np.arange(periods, dtype=float)
    trend = np.piecewise(
        phase,
        [phase < 140, (phase >= 140) & (phase < 280), phase >= 280],
        [
            lambda x: x * 0.12,
            lambda x: 16.0 - (x - 140.0) * 0.14,
            lambda x: -3.0 + (x - 280.0) * 0.10,
        ],
    )
    close = 100.0 + trend + np.sin(phase / 4.0) * 5.0 + np.sin(phase / 17.0) * 2.0
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) + 1.0
    low = np.minimum(open_, close) - 1.0
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


def _params() -> dict:
    return {
        "half_life_lookback": 30,
        "half_life_factor": 1.0,
        "std_len": 12,
        "use_vol_norm": False,
        "long_entry_z": -0.8,
        "long_exit_z": 0.0,
        "short_entry_z": 0.8,
        "short_exit_z": 0.0,
        "fastest_len": 5,
        "fast_len": 8,
        "med_len": 12,
        "slow_len": 20,
        "slowest_len": 35,
        "regime_mode": "fastest_slowest",
        "regime_timeframe": "",
        "enable_longs": True,
        "enable_shorts": True,
        "use_sma_regime_filter": True,
        "exit_on_regime_loss": True,
        "target": 0.5,
    }


class SMARegimeFilteredZScoreTest(unittest.TestCase):
    def test_features_gate_zscore_entries_by_sma_regime_and_side_toggles(self) -> None:
        bars = _make_regime_zscore_bars()
        params = _params()

        features = compute_sma_regime_filtered_zscore_mean_reversion_features(bars, params)

        self.assertTrue(
            {
                "z_score",
                "half_life_mean",
                "sma_fastest",
                "sma_slowest",
                "sma_regime_state",
                "sma_bull_regime",
                "sma_bear_regime",
                "long_entry_signal",
                "short_entry_signal",
                "hard_stop_loss_price",
            }.issubset(features.columns)
        )
        self.assertGreater(features["long_entry_signal"].sum(), 0)
        self.assertGreater(features["short_entry_signal"].sum(), 0)
        self.assertTrue(features.loc[features["long_entry_signal"], "sma_bull_regime"].all())
        self.assertTrue(features.loc[features["short_entry_signal"], "sma_bear_regime"].all())

        no_longs = compute_sma_regime_filtered_zscore_mean_reversion_features(
            bars,
            {**params, "enable_longs": False},
        )
        no_shorts = compute_sma_regime_filtered_zscore_mean_reversion_features(
            bars,
            {**params, "enable_shorts": False},
        )
        self.assertEqual(int(no_longs["long_entry_signal"].sum()), 0)
        self.assertGreater(no_longs["short_entry_signal"].sum(), 0)
        self.assertGreater(no_shorts["long_entry_signal"].sum(), 0)
        self.assertEqual(int(no_shorts["short_entry_signal"].sum()), 0)

    def test_optional_hard_stop_loss_emits_exit_signal(self) -> None:
        params = {
            **_params(),
            "use_hard_stop_loss": True,
            "hard_stop_loss_pct": 0.001,
        }
        features = compute_sma_regime_filtered_zscore_mean_reversion_features(
            _make_regime_zscore_bars(),
            params,
        )
        hard_stop_exit = (
            (features["state_long_exit_signal"].astype(bool) & features["long_hard_stop_loss_hit"].astype(bool))
            | (features["state_short_exit_signal"].astype(bool) & features["short_hard_stop_loss_hit"].astype(bool))
        )

        self.assertGreater(hard_stop_exit.sum(), 0)

    def test_vectorized_matches_reference(self) -> None:
        bars = _make_regime_zscore_bars()
        params = _params()
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
                dataset_id="sma_regime_zscore_reference",
                strategy_cls=SMARegimeFilteredZScoreMeanReversionStrategy,
                strategy_params=params,
                config=config,
                requested_execution_mode=ExecutionMode.REFERENCE,
            )
        )
        vectorized = orchestrator.execute(
            ExecutionRequest(
                data=bars,
                base_data=bars,
                dataset_id="sma_regime_zscore_vectorized",
                strategy_cls=SMARegimeFilteredZScoreMeanReversionStrategy,
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
