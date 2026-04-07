from __future__ import annotations

import numpy as np
import pandas as pd

from .strategy import Strategy


def _coerce_bool(value, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _rma(series: pd.Series, length: int) -> pd.Series:
    length = max(int(length), 1)
    return series.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()


def compute_zscore_mean_reversion_features(data: pd.DataFrame, params: dict) -> pd.DataFrame:
    src = data["close"].astype(float)
    high = data["high"].astype(float)
    low = data["low"].astype(float)
    prev_close = src.shift(1)

    half_life_lookback = max(int(params.get("half_life_lookback", 100)), 10)
    half_life_factor = max(float(params.get("half_life_factor", 1.5)), 1.0)
    std_len = max(int(params.get("std_len", 20)), 1)
    z_smooth_type = str(params.get("z_smooth_type", "None")).strip().lower()
    z_smooth_len = max(int(params.get("z_smooth_len", 5)), 1)
    use_vol_norm = _coerce_bool(params.get("use_vol_norm", True), True)
    vol_type = str(params.get("vol_type", "ATR")).strip().lower()
    vol_len = max(int(params.get("vol_len", 14)), 1)
    atr_mult = float(params.get("atr_mult", 1.0))
    long_entry_z = float(params.get("long_entry_z", -1.0))
    long_exit_z = float(params.get("long_exit_z", 0.0))

    lag = src.shift(1)
    delta = src - lag
    lag_std = lag.rolling(half_life_lookback).std(ddof=0)
    delta_std = delta.rolling(half_life_lookback).std(ddof=0)
    corr = lag.rolling(half_life_lookback).corr(delta)
    beta = (corr * (delta_std / lag_std.replace(0.0, np.nan))).replace([np.inf, -np.inf], np.nan)
    estimated_half_life = (-np.log(2.0) / beta).where(beta < 0.0)

    fallback_half_life = max(2.0, half_life_lookback / 2.0)
    base_half_life = estimated_half_life.where((estimated_half_life > 0.0) & estimated_half_life.notna(), fallback_half_life)
    effective_half_life = np.maximum(2.0, base_half_life.to_numpy(dtype=float) * half_life_factor)
    half_life_alpha = 1.0 - np.power(0.5, 1.0 / effective_half_life)

    src_values = src.to_numpy(dtype=float)
    half_life_mean = np.empty(len(src_values), dtype=float)
    half_life_mean[0] = src_values[0]
    for i in range(1, len(src_values)):
        alpha = float(half_life_alpha[i]) if np.isfinite(half_life_alpha[i]) else 1.0
        half_life_mean[i] = alpha * src_values[i] + (1.0 - alpha) * half_life_mean[i - 1]
    half_life_mean_series = pd.Series(half_life_mean, index=data.index, name="half_life_mean")

    spread_raw = src - half_life_mean_series
    true_range = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = _rma(true_range, vol_len) * atr_mult
    log_returns = np.log(src / prev_close)
    return_vol = log_returns.rolling(vol_len).std(ddof=0)
    price_vol = return_vol * src
    vol = atr if vol_type == "atr" else price_vol
    if use_vol_norm:
        spread = spread_raw / vol.replace(0.0, np.nan)
    else:
        spread = spread_raw

    spread_mean = spread.rolling(std_len).mean()
    spread_std = spread.rolling(std_len).std(ddof=0).replace(0.0, np.nan)
    spread_z_raw = (spread - spread_mean) / spread_std
    if z_smooth_type == "sma":
        z_score = spread_z_raw.rolling(z_smooth_len).mean()
    elif z_smooth_type == "ema":
        z_score = spread_z_raw.ewm(span=z_smooth_len, adjust=False, min_periods=z_smooth_len).mean()
    else:
        z_score = spread_z_raw

    prev_z = z_score.shift(1)
    long_entry_signal = ((z_score < long_entry_z) & (prev_z >= long_entry_z)).fillna(False)
    long_exit_signal = ((z_score > long_exit_z) & (prev_z <= long_exit_z)).fillna(False)

    long_state_values = np.zeros(len(data), dtype=bool)
    in_position = False
    for i, (exit_now, entry_now) in enumerate(zip(long_exit_signal.to_numpy(dtype=bool), long_entry_signal.to_numpy(dtype=bool))):
        if exit_now and in_position:
            in_position = False
        elif entry_now and not in_position:
            in_position = True
        long_state_values[i] = in_position

    return pd.DataFrame(
        {
            "half_life_mean": half_life_mean_series,
            "spread": spread,
            "z_score": z_score,
            "long_entry_signal": long_entry_signal.astype(bool),
            "long_exit_signal": long_exit_signal.astype(bool),
            "long_state": pd.Series(long_state_values, index=data.index),
            "estimated_half_life_bars": estimated_half_life,
            "effective_half_life_bars": pd.Series(effective_half_life, index=data.index),
            "atr": atr,
            "vol": vol,
        },
        index=data.index,
    )


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


class ZScoreMeanReversionStrategy(Strategy):
    """
    Long-only, fixed-target simplification of the TradingView z-score mean-reversion strategy.

    Intentionally omitted for version 1:
    - regime filter
    - dynamic equity / annual-vol sizing
    - short entries
    """

    def initialize(self, data: pd.DataFrame) -> None:
        self.target = float(self.params.get("target", 1.0))
        if self.target < 0:
            raise ValueError("target must be >= 0 for long-only z-score mean reversion")
        self.features = compute_zscore_mean_reversion_features(data, self.params)

    def on_bar(self, timestamp: pd.Timestamp, bar: pd.Series, broker) -> None:
        row = self.features.loc[timestamp]
        if bool(row.get("long_exit_signal", False)) and broker.position_qty > 0:
            broker.target_percent(0.0, bar["close"])
            return
        if bool(row.get("long_entry_signal", False)) and broker.position_qty <= 0:
            broker.target_percent(self.target, bar["close"])
