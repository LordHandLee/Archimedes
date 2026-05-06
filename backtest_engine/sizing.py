from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .metrics import _periods_per_year_from_index

if TYPE_CHECKING:
    from .engine import BacktestConfig


POSITION_SIZING_NONE = "none"
POSITION_SIZING_ANNUAL_VOLATILITY = "annual_volatility_target"


def normalize_position_sizing_model(value: object) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"", "none", "fixed", "off", "disabled"}:
        return POSITION_SIZING_NONE
    if text in {"annual_volatility", "annualized_volatility", "annual_volatility_target", "volatility_target"}:
        return POSITION_SIZING_ANNUAL_VOLATILITY
    return text


def effective_max_gross_leverage(config: "BacktestConfig") -> float:
    if not bool(getattr(config, "margin_enabled", False)):
        return 1.0
    try:
        leverage = float(getattr(config, "max_gross_leverage", 1.0) or 1.0)
    except Exception:
        leverage = 1.0
    return max(1.0, leverage)


def _configured_trading_days_per_year(config: "BacktestConfig") -> float:
    mode = str(getattr(config, "sharpe_annualization", "equities") or "equities").strip().lower()
    if mode == "equities":
        return 252.0
    if mode == "crypto":
        return 365.25
    return 365.25


def _timeframe_seconds(timeframe: object) -> float | None:
    text = str(timeframe or "").strip().lower()
    if not text:
        return None
    compact = (
        text.replace(" ", "")
        .replace("minutes", "min")
        .replace("minute", "min")
        .replace("mins", "min")
        .replace("hours", "h")
        .replace("hour", "h")
        .replace("hrs", "h")
        .replace("days", "d")
        .replace("day", "d")
    )
    number = ""
    unit = ""
    for char in compact:
        if char.isdigit() or char == ".":
            number += char
        else:
            unit += char
    try:
        amount = float(number or "1")
    except Exception:
        return None
    if amount <= 0:
        return None
    if unit in {"min", "m"}:
        return amount * 60.0
    if unit in {"h", "hr"}:
        return amount * 3600.0
    if unit in {"d"}:
        return amount * 24.0 * 3600.0
    return None


def _median_index_seconds(index: pd.DatetimeIndex) -> float | None:
    if len(index) < 2:
        return None
    try:
        seconds = np.median(np.diff(index.values).astype("timedelta64[s]").astype(float))
    except Exception:
        return None
    if not np.isfinite(seconds) or seconds <= 0:
        return None
    return float(seconds)


def _observed_bars_per_day(index: pd.DatetimeIndex, config: "BacktestConfig") -> float | None:
    if len(index) == 0:
        return None
    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    annualization = str(getattr(config, "sharpe_annualization", "equities") or "equities").strip().lower()
    if annualization == "equities":
        try:
            idx = idx.tz_convert("America/New_York")
        except Exception:
            pass
    days = pd.Series(idx.normalize())
    counts = days.value_counts(sort=False).sort_index()
    if len(counts) > 2:
        counts = counts.iloc[1:-1]
    counts = counts[counts > 0]
    if counts.empty:
        return None
    bars_per_day = float(counts.median())
    if not np.isfinite(bars_per_day) or bars_per_day <= 0:
        return None
    return bars_per_day


def _annual_vol_periods_per_year(index: pd.DatetimeIndex, config: "BacktestConfig") -> float:
    timeframe = getattr(config, "timeframe", None)
    annualization = getattr(config, "sharpe_annualization", "equities")
    session_seconds = getattr(config, "sharpe_session_seconds_per_day", None)
    if session_seconds is None:
        session_seconds = 6.5 * 3600 if str(annualization or "equities").strip().lower() == "equities" else 24.0 * 3600
    seconds_per_bar = _timeframe_seconds(timeframe) or _median_index_seconds(index)
    days_per_year = _configured_trading_days_per_year(config)
    if seconds_per_bar is not None and seconds_per_bar > 1.5 * 24.0 * 3600:
        periods_per_year = (365.25 * 24.0 * 3600) / seconds_per_bar
        if np.isfinite(periods_per_year) and periods_per_year > 0:
            return float(periods_per_year)
    observed_bars_per_day = _observed_bars_per_day(index, config)
    if observed_bars_per_day is not None:
        return float(observed_bars_per_day * days_per_year)
    basis = "daily"
    if seconds_per_bar is not None and seconds_per_bar < float(session_seconds):
        basis = "bars"
    periods_per_year = _periods_per_year_from_index(
        index,
        timeframe,
        annualization,
        session_seconds,
        basis,
    )
    if not np.isfinite(periods_per_year) or periods_per_year <= 0:
        periods_per_year = _configured_trading_days_per_year(config)
    return float(periods_per_year)


def _annual_vol_rolling_lengths(index: pd.DatetimeIndex, config: "BacktestConfig") -> tuple[int, int, float]:
    window_days = max(2, int(getattr(config, "annual_vol_window", 252) or 252))
    min_days = min(window_days, max(2, int(getattr(config, "annual_vol_min_periods", 20) or 20)))
    periods_per_year = _annual_vol_periods_per_year(index, config)
    days_per_year = _configured_trading_days_per_year(config)
    bars_per_day = periods_per_year / days_per_year
    if not np.isfinite(bars_per_day) or bars_per_day <= 0:
        bars_per_day = 1.0
    bars_per_day = max(1.0 / days_per_year, bars_per_day)
    window = max(2, int(round(window_days * bars_per_day)))
    min_periods = min(window, max(2, int(round(min_days * bars_per_day))))
    return window, min_periods, periods_per_year


def position_sizing_multiplier(data: pd.DataFrame, config: "BacktestConfig") -> pd.Series:
    if data is None or data.empty or "close" not in data.columns:
        return pd.Series(dtype=float)
    model = normalize_position_sizing_model(getattr(config, "position_sizing_model", POSITION_SIZING_NONE))
    index = pd.DatetimeIndex(data.index)
    if model == POSITION_SIZING_NONE:
        return pd.Series(1.0, index=index, name="position_sizing_multiplier")
    if model != POSITION_SIZING_ANNUAL_VOLATILITY:
        raise ValueError(f"Unsupported position_sizing_model: {model}")

    close = pd.to_numeric(data["close"], errors="coerce").astype(float)
    returns = close.pct_change()
    window, min_periods, periods_per_year = _annual_vol_rolling_lengths(index, config)
    annual_vol = returns.rolling(window=window, min_periods=min_periods).std() * np.sqrt(periods_per_year)
    floor = max(1e-9, float(getattr(config, "annual_vol_floor", 0.05) or 0.05))
    max_multiplier = max(0.0, float(getattr(config, "max_volatility_multiplier", 2.0) or 2.0))
    multiplier = 1.0 / annual_vol.clip(lower=floor)
    multiplier = multiplier.clip(upper=max_multiplier)
    multiplier = multiplier.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return multiplier.astype(float).rename("position_sizing_multiplier")


def apply_position_sizing_to_weights(
    weights: np.ndarray,
    data: pd.DataFrame,
    config: "BacktestConfig",
) -> np.ndarray:
    model = normalize_position_sizing_model(getattr(config, "position_sizing_model", POSITION_SIZING_NONE))
    if model == POSITION_SIZING_NONE:
        return weights
    multipliers = position_sizing_multiplier(data, config).reindex(data.index).fillna(1.0).to_numpy(dtype=float)
    if weights.ndim == 1:
        return weights.astype(float, copy=True) * multipliers
    return weights.astype(float, copy=True) * multipliers[:, None]
