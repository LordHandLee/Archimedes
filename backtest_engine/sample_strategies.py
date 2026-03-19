from __future__ import annotations

import pandas as pd

from .strategy import Strategy


class SMACrossStrategy(Strategy):
    """
    Basic moving-average crossover.

    Parameters:
    - fast: int, fast SMA window
    - slow: int, slow SMA window
    - target: float, position size as fraction of equity (default 1.0)
    """

    def initialize(self, data: pd.DataFrame) -> None:
        fast = int(self.params.get("fast", 10))
        slow = int(self.params.get("slow", 30))
        if fast >= slow:
            raise ValueError("fast window must be < slow window")

        self.target = float(self.params.get("target", 1.0))
        self.data = pd.DataFrame(index=data.index)
        self.data["fast"] = data["close"].rolling(fast).mean()
        self.data["slow"] = data["close"].rolling(slow).mean()

    def on_bar(self, timestamp: pd.Timestamp, bar: pd.Series, broker) -> None:
        row = self.data.loc[timestamp]
        if pd.isna(row["fast"]) or pd.isna(row["slow"]):
            return

        if row["fast"] > row["slow"] and broker.position_qty <= 0:
            broker.target_percent(self.target, bar["close"])
        elif row["fast"] < row["slow"] and broker.position_qty > 0:
            broker.target_percent(0.0, bar["close"])


class InverseTurtleStrategy(Strategy):
    """
    Mean-reversion variant of the Turtle system using breakout channels with ATR stops.

    Parameters:
    - entry_len: int, lookback for breakout channel (default 20, uses prior bars)
    - exit_len: int, lookback for exit channel (default 10, uses prior bars)
    - atr_len: int, ATR length (default 14)
    - atr_mult: float, ATR multiplier for stop distance (default 2.0)
    - target: float, position size as fraction of equity (default 0.1 to mirror TV 10% equity)
    - use_atr_stop: bool, whether to apply ATR-based stops (default True)
    - use_prev_channels: bool, whether to shift channels/ATR by one bar to avoid lookahead (default True)
    """

    def initialize(self, data: pd.DataFrame) -> None:
        entry_len = int(self.params.get("entry_len", 20))
        exit_len = int(self.params.get("exit_len", 10))
        atr_len = int(self.params.get("atr_len", 14))
        self.atr_mult = float(self.params.get("atr_mult", 2.0))
        self.target = float(self.params.get("target", 1.0))
        self.use_atr_stop = bool(self.params.get("use_atr_stop", True))
        self.use_prev_channels = bool(self.params.get("use_prev_channels", True))

        df = pd.DataFrame(index=data.index)
        # Use prior-bar channels to better match Pine's non-repainting behaviour.
        upper = data["high"].rolling(entry_len).max()
        lower = data["low"].rolling(entry_len).min()
        exit_upper = data["high"].rolling(exit_len).max()
        exit_lower = data["low"].rolling(exit_len).min()
        if self.use_prev_channels:
            upper = upper.shift(1)
            lower = lower.shift(1)
            exit_upper = exit_upper.shift(1)
            exit_lower = exit_lower.shift(1)
        df["upper"] = upper
        df["lower"] = lower
        df["exit_upper"] = exit_upper
        df["exit_lower"] = exit_lower

        high = data["high"]
        low = data["low"]
        close = data["close"]
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(atr_len).mean()
        if self.use_prev_channels:
            atr = atr.shift(1)
        df["atr"] = atr

        self.indicators = df

    def on_bar(self, timestamp: pd.Timestamp, bar: pd.Series, broker) -> None:
        row = self.indicators.loc[timestamp]
        if row.isna().any():
            return

        price = bar["close"]
        # Entries
        if bar["low"] <= row["lower"] and broker.position_qty <= 0:
            broker.target_percent(self.target, price)
        elif bar["high"] >= row["upper"] and broker.position_qty >= 0:
            broker.target_percent(-self.target, price)

        # Channel exits
        if broker.position_qty > 0 and bar["high"] >= row["exit_upper"]:
            broker.target_percent(0.0, price)
        elif broker.position_qty < 0 and bar["low"] <= row["exit_lower"]:
            broker.target_percent(0.0, price)

        # ATR stops
        if self.use_atr_stop and broker.position_qty != 0:
            broker.cancel_orders(tag="atr_stop")
            if broker.position_qty > 0:
                stop = broker.avg_price - row["atr"] * self.atr_mult
                broker.sell_stop(abs(broker.position_qty), stop, tag="atr_stop")
            elif broker.position_qty < 0:
                stop = broker.avg_price + row["atr"] * self.atr_mult
                broker.buy_stop(abs(broker.position_qty), stop, tag="atr_stop")
