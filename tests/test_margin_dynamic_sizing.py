from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from backtest_engine.broker import Broker
from backtest_engine.engine import BacktestConfig
from backtest_engine.sizing import _annual_vol_rolling_lengths, position_sizing_multiplier
from backtest_engine.vectorized_portfolio import VectorizedPortfolioEngine


class MarginAndDynamicSizingTests(unittest.TestCase):
    def _bar(self, price: float = 100.0) -> pd.Series:
        return pd.Series({"open": price, "high": price, "low": price, "close": price, "volume": 1_000})

    def test_margin_allows_target_above_cash_balance(self) -> None:
        ts = pd.Timestamp("2024-01-02 14:30", tz="UTC")
        cash_only = Broker(starting_cash=1_000.0, margin_enabled=False, max_gross_leverage=2.0)
        cash_only.target_percent(2.0, 100.0)
        cash_only.flush_orders(self._bar(100.0), ts)

        margin = Broker(starting_cash=1_000.0, margin_enabled=True, max_gross_leverage=2.0)
        margin.target_percent(2.0, 100.0)
        margin.flush_orders(self._bar(100.0), ts)

        self.assertAlmostEqual(cash_only.position_qty, 10.0)
        self.assertAlmostEqual(margin.position_qty, 20.0)
        self.assertAlmostEqual(margin.cash, -1_000.0)

    def test_dynamic_sizing_can_enforce_minimum_one_share_with_margin(self) -> None:
        ts = pd.Timestamp("2024-01-02 14:30", tz="UTC")
        multipliers = pd.Series([0.5], index=pd.DatetimeIndex([ts]))
        broker = Broker(
            starting_cash=50.0,
            margin_enabled=True,
            max_gross_leverage=2.0,
            position_sizing_model="annual_volatility_target",
            min_position_shares=1.0,
            sizing_multiplier_by_ts=multipliers,
        )
        broker.current_timestamp = ts
        broker.target_percent(0.1, 100.0)
        broker.flush_orders(self._bar(100.0), ts)

        self.assertAlmostEqual(broker.position_qty, 1.0)
        self.assertAlmostEqual(broker.cash, -50.0)

    def test_annual_volatility_multiplier_caps_at_configured_max(self) -> None:
        index = pd.date_range("2024-01-02", periods=8, freq="1D", tz="UTC")
        data = pd.DataFrame(
            {
                "open": [100.0, 100.1, 100.2, 100.3, 100.4, 100.5, 100.6, 100.7],
                "high": [101.0] * 8,
                "low": [99.0] * 8,
                "close": [100.0, 100.1, 100.2, 100.3, 100.4, 100.5, 100.6, 100.7],
                "volume": [1_000] * 8,
            },
            index=index,
        )
        config = BacktestConfig(
            timeframe="1 day",
            position_sizing_model="annual_volatility_target",
            annual_vol_window=3,
            annual_vol_min_periods=2,
            annual_vol_floor=0.05,
            max_volatility_multiplier=2.0,
        )

        multipliers = position_sizing_multiplier(data, config)

        self.assertEqual(len(multipliers), len(data))
        self.assertTrue(np.all(multipliers <= 2.0))
        self.assertAlmostEqual(float(multipliers.iloc[-1]), 2.0)

    def test_annual_volatility_window_converts_trading_days_to_intraday_bars(self) -> None:
        sessions = pd.date_range("2024-01-02", periods=25, freq="B", tz="America/New_York")
        timestamps = []
        for session in sessions:
            start = session.replace(hour=9, minute=30)
            timestamps.extend(pd.date_range(start, periods=26, freq="15min"))
        index = pd.DatetimeIndex(timestamps).tz_convert("UTC")
        data = pd.DataFrame(
            {
                "open": np.linspace(100.0, 106.5, len(index)),
                "high": np.linspace(100.1, 106.6, len(index)),
                "low": np.linspace(99.9, 106.4, len(index)),
                "close": np.linspace(100.0, 106.5, len(index)),
                "volume": [1_000] * len(index),
            },
            index=index,
        )
        config = BacktestConfig(
            timeframe="15 minutes",
            position_sizing_model="annual_volatility_target",
            annual_vol_window=20,
            annual_vol_min_periods=5,
            annual_vol_floor=0.05,
            max_volatility_multiplier=2.0,
        )

        window, min_periods, periods_per_year = _annual_vol_rolling_lengths(index, config)
        multipliers = position_sizing_multiplier(data, config)

        self.assertEqual(window, 520)
        self.assertEqual(min_periods, 130)
        self.assertAlmostEqual(periods_per_year, 6_552.0)
        self.assertAlmostEqual(float(multipliers.iloc[100]), 1.0)
        self.assertAlmostEqual(float(multipliers.iloc[-1]), 2.0)

    def test_portfolio_weight_to_qty_applies_minimum_share_only_for_dynamic_sizing(self) -> None:
        config = BacktestConfig(
            timeframe="1 day",
            position_sizing_model="annual_volatility_target",
            min_position_shares=1.0,
        )
        dynamic_qty = VectorizedPortfolioEngine._weights_to_qty(
            np.array([0.1]),
            50.0,
            np.array([100.0]),
            config=config,
        )
        fixed_qty = VectorizedPortfolioEngine._weights_to_qty(
            np.array([0.1]),
            50.0,
            np.array([100.0]),
            config=BacktestConfig(timeframe="1 day"),
        )

        self.assertAlmostEqual(float(dynamic_qty[0]), 1.0)
        self.assertAlmostEqual(float(fixed_qty[0]), 0.05)


if __name__ == "__main__":
    unittest.main()
