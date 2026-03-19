from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class PerformanceMetrics:
    total_return: float
    cagr: float
    max_drawdown: float
    sharpe: float
    rolling_sharpe: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "total_return": self.total_return,
            "cagr": self.cagr,
            "max_drawdown": self.max_drawdown,
            "sharpe": self.sharpe,
            "rolling_sharpe": self.rolling_sharpe,
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict())


def compute_metrics(
    equity_curve: pd.Series,
    risk_free_rate: float = 0.0,
    timeframe: str | None = None,
    annualization: str = "equities",
    session_seconds_per_day: float | None = None,
    sharpe_basis: str = "daily",
    rolling_window: int = 20,
) -> PerformanceMetrics:
    """
    Compute core performance metrics from an equity curve indexed by timestamp.
    """
    if sharpe_basis == "daily":
        equity_curve = _resample_equity_daily(equity_curve)
    elif timeframe:
        equity_curve = _resample_equity(equity_curve, timeframe)
    returns = equity_curve.pct_change().fillna(0.0)
    start_value = equity_curve.iloc[0]
    end_value = equity_curve.iloc[-1]
    total_return = (end_value / start_value) - 1.0

    elapsed = (equity_curve.index[-1] - equity_curve.index[0]).total_seconds()
    years = elapsed / (365.25 * 24 * 3600)
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else total_return

    max_drawdown = _max_drawdown(equity_curve)
    periods_per_year = _periods_per_year_from_index(
        equity_curve.index,
        timeframe,
        annualization,
        session_seconds_per_day,
        sharpe_basis,
    )
    sharpe = _sharpe_ratio(
        returns,
        risk_free_rate,
        equity_curve.index,
        timeframe,
        annualization,
        session_seconds_per_day,
        sharpe_basis,
    )
    rolling_sharpe = _rolling_sharpe_ratio(returns, risk_free_rate, periods_per_year, rolling_window)

    return PerformanceMetrics(
        total_return=float(total_return),
        cagr=float(cagr),
        max_drawdown=float(max_drawdown),
        sharpe=float(sharpe),
        rolling_sharpe=float(rolling_sharpe) if rolling_sharpe == rolling_sharpe else float("nan"),
    )


def sharpe_diagnostics(
    equity_curve: pd.Series,
    risk_free_rate: float = 0.0,
    timeframe: str | None = None,
    annualization: str = "equities",
    session_seconds_per_day: float | None = None,
    sharpe_basis: str = "daily",
    rolling_window: int = 20,
) -> Dict[str, float | int | str | None]:
    if sharpe_basis == "daily":
        equity_curve = _resample_equity_daily(equity_curve)
    elif timeframe:
        equity_curve = _resample_equity(equity_curve, timeframe)
    returns = equity_curve.pct_change().dropna()
    if returns.empty:
        return {
            "periods": 0,
            "mean": 0.0,
            "std": 0.0,
            "seconds_per_period": None,
            "periods_per_year": 0.0,
            "rf_per_period": 0.0,
            "annualization": annualization,
            "session_seconds_per_day": session_seconds_per_day,
            "basis": sharpe_basis,
        }
    periods_per_year = _periods_per_year_from_index(
        equity_curve.index,
        timeframe,
        annualization,
        session_seconds_per_day,
        sharpe_basis,
    )
    seconds_per_period = None
    if sharpe_basis != "daily":
        seconds_per_period = _seconds_per_period_from_index(equity_curve.index, timeframe)
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1 if periods_per_year > 0 else 0.0
    rolling_sharpe = _rolling_sharpe_ratio(returns, risk_free_rate, periods_per_year, rolling_window)
    return {
        "periods": int(returns.shape[0]),
        "mean": float(returns.mean()),
        "std": float(returns.std()),
        "seconds_per_period": float(seconds_per_period) if seconds_per_period else None,
        "periods_per_year": float(periods_per_year),
        "rf_per_period": float(rf_per_period),
        "annualization": annualization,
        "session_seconds_per_day": session_seconds_per_day,
        "basis": sharpe_basis,
        "rolling_sharpe": float(rolling_sharpe) if rolling_sharpe == rolling_sharpe else None,
    }


def _max_drawdown(equity: pd.Series) -> float:
    cumulative_max = equity.cummax()
    drawdown = (equity - cumulative_max) / cumulative_max
    return float(drawdown.min())


def _sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float,
    index: pd.DatetimeIndex,
    timeframe: str | None = None,
    annualization: str = "equities",
    session_seconds_per_day: float | None = None,
    sharpe_basis: str = "daily",
) -> float:
    if returns.std() == 0:
        return 0.0
    # Estimate periods per year based on timeframe if provided; otherwise use median sampling interval.
    seconds_per_period = None
    if sharpe_basis != "daily":
        seconds_per_period = _seconds_per_period_from_index(index, timeframe)
    periods_per_year = _periods_per_year(seconds_per_period, annualization, session_seconds_per_day, sharpe_basis)
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1 if periods_per_year > 0 else 0.0
    excess = returns - rf_per_period
    return float((excess.mean() / returns.std()) * np.sqrt(periods_per_year))


def _rolling_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float,
    periods_per_year: float,
    window: int,
) -> float:
    if window <= 1 or returns.empty:
        return float("nan")
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1 if periods_per_year > 0 else 0.0
    excess = returns - rf_per_period
    roll_mean = excess.rolling(window).mean()
    roll_std = excess.rolling(window).std()
    roll = (roll_mean / roll_std) * np.sqrt(periods_per_year)
    return float(roll.iloc[-1]) if not roll.empty else float("nan")


def _seconds_per_period_from_index(index: pd.DatetimeIndex, timeframe: str | None) -> float | None:
    seconds_per_period = None
    if timeframe:
        try:
            norm_tf = _normalize_freq(timeframe)
            offset = pd.tseries.frequencies.to_offset(norm_tf)
            seconds_per_period = offset.delta.total_seconds()
        except Exception:
            seconds_per_period = None
    if not seconds_per_period or seconds_per_period <= 0:
        seconds_per_period = np.median(np.diff(index.values).astype("timedelta64[s]").astype(float))
    return seconds_per_period


def _periods_per_year_from_index(
    index: pd.DatetimeIndex,
    timeframe: str | None,
    annualization: str,
    session_seconds_per_day: float | None,
    sharpe_basis: str,
) -> float:
    seconds_per_period = None
    if sharpe_basis != "daily":
        seconds_per_period = _seconds_per_period_from_index(index, timeframe)
    return _periods_per_year(seconds_per_period, annualization, session_seconds_per_day, sharpe_basis)


def _normalize_freq(tf: str) -> str:
    s = tf.strip().lower()
    s = s.replace("minute", "min").replace("mins", "min")
    s = s.replace("hour", "h")
    tokens = s.replace(" ", "")
    num = ""
    unit = ""
    for ch in tokens:
        if ch.isdigit():
            num += ch
        else:
            unit += ch
    if not num:
        num = "1"
    if unit in ("min", "m"):
        return f"{num}T"
    if unit in ("h", "hr"):
        return f"{num}H"
    return tf


def _resample_equity(equity: pd.Series, timeframe: str) -> pd.Series:
    norm_tf = _normalize_freq(timeframe)
    if not isinstance(equity.index, pd.DatetimeIndex):
        return equity
    resampled = equity.resample(norm_tf, label="right", closed="right").last().dropna()
    return resampled


def _resample_equity_daily(equity: pd.Series) -> pd.Series:
    if not isinstance(equity.index, pd.DatetimeIndex):
        return equity
    idx = equity.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    try:
        local = idx.tz_convert("America/New_York")
    except Exception:
        local = idx
    daily = equity.copy()
    daily.index = local.tz_localize(None)
    resampled = daily.resample("1D").last().dropna()
    return resampled


def _periods_per_year(
    seconds_per_period: float | None,
    annualization: str,
    session_seconds_per_day: float | None,
    sharpe_basis: str,
) -> float:
    mode = (annualization or "equities").strip().lower()
    basis = (sharpe_basis or "daily").strip().lower()
    if basis == "daily":
        if mode == "equities":
            return 252.0
        if mode == "crypto":
            return 365.25
        return 365.25
    if not seconds_per_period or seconds_per_period <= 0:
        return 1.0
    if mode == "equities":
        trading_seconds_per_day = session_seconds_per_day or (6.5 * 3600)
        return (252 * trading_seconds_per_day) / seconds_per_period
    if mode == "crypto":
        return (365.25 * 24 * 3600) / seconds_per_period
    return (365.25 * 24 * 3600) / seconds_per_period
