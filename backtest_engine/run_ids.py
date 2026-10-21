from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Sequence

import pandas as pd

if TYPE_CHECKING:
    from .engine import BacktestConfig


_STRATEGY_LOGIC_VERSIONS = {
    "RegimeFilteredAdaptiveOUKalmanZScoreStrategy": 2,
    "SMARegimeFilteredZScoreMeanReversionStrategy": 2,
    "SMACrossStrategy": 2,
    "SMATrendRegimeStrategy": 5,
    "TrendFollowingRibbonStrategy": 4,
}

_SOURCE_FINGERPRINT_FILES = (
    "backtest_engine/broker.py",
    "backtest_engine/engine.py",
    "backtest_engine/execution.py",
    "backtest_engine/reference_engine.py",
    "backtest_engine/run_ids.py",
    "backtest_engine/sample_strategies.py",
    "backtest_engine/sizing.py",
    "backtest_engine/strategy.py",
    "backtest_engine/vectorized_engine.py",
    "backtest_engine/vectorized_portfolio.py",
    "backtest_engine/vectorized_strategies.py",
)


def _compute_backtest_source_fingerprint() -> str:
    root = Path(__file__).resolve().parent.parent
    digest = hashlib.sha256()
    for relative_path in _SOURCE_FINGERPRINT_FILES:
        path = root / relative_path
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        try:
            digest.update(path.read_bytes())
        except FileNotFoundError:
            digest.update(b"<missing>")
        digest.update(b"\0")
    return digest.hexdigest()[:16]


@lru_cache(maxsize=1)
def backtest_source_fingerprint() -> str:
    return _compute_backtest_source_fingerprint()


def current_backtest_source_fingerprint() -> str:
    return _compute_backtest_source_fingerprint()


IMPORTED_BACKTEST_SOURCE_FINGERPRINT = backtest_source_fingerprint()


def backtest_source_is_stale() -> bool:
    return current_backtest_source_fingerprint() != IMPORTED_BACKTEST_SOURCE_FINGERPRINT


def assert_backtest_source_current() -> None:
    if os.environ.get("BACKTEST_ALLOW_STALE_SOURCE") == "1":
        return
    current = current_backtest_source_fingerprint()
    if current == IMPORTED_BACKTEST_SOURCE_FINGERPRINT:
        return
    raise RuntimeError(
        "Backtest engine source changed after this Python process started. "
        "Restart the dashboard before running another backtest so strategy logic, "
        "vectorized adapters, and run IDs all come from the same code. "
        f"Loaded fingerprint {IMPORTED_BACKTEST_SOURCE_FINGERPRINT}; "
        f"current files fingerprint {current}."
    )


def compute_logical_run_id(
    *,
    dataset_id: str,
    strategy: str,
    params: Dict,
    config: BacktestConfig,
    data: pd.DataFrame,
) -> str:
    payload = {
        "dataset_id": dataset_id,
        "strategy": strategy,
        "strategy_logic_version": _STRATEGY_LOGIC_VERSIONS.get(strategy, 1),
        "source_fingerprint": backtest_source_fingerprint(),
        "params": params,
        "timeframe": config.timeframe,
        "start": str(data.index[0]),
        "end": str(data.index[-1]),
        "starting_cash": config.starting_cash,
        "fill_on_close": config.fill_on_close,
        "recalc_on_fill": config.recalc_on_fill,
        "max_recalc_passes": config.max_recalc_passes,
        "allow_short": config.allow_short,
        "borrow_rate": config.borrow_rate,
        "fill_ratio": config.fill_ratio,
        "fee_rate": config.fee_rate,
        "fee_schedule": config.fee_schedule,
        "slippage": config.slippage,
        "slippage_schedule": config.slippage_schedule,
        "risk_free_rate": config.risk_free_rate,
        "sharpe_basis": config.sharpe_basis,
        "sharpe_annualization": config.sharpe_annualization,
        "sharpe_session_seconds_per_day": config.sharpe_session_seconds_per_day,
        "margin_enabled": config.margin_enabled,
        "max_gross_leverage": config.max_gross_leverage,
        "position_sizing_model": config.position_sizing_model,
        "annual_vol_window": config.annual_vol_window,
        "annual_vol_min_periods": config.annual_vol_min_periods,
        "annual_vol_floor": config.annual_vol_floor,
        "max_volatility_multiplier": config.max_volatility_multiplier,
        "min_position_shares": config.min_position_shares,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def compute_engine_run_id(logical_run_id: str, engine_impl: str, engine_version: str) -> str:
    payload = {
        "logical_run_id": logical_run_id,
        "engine_impl": engine_impl,
        "engine_version": engine_version,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def compute_portfolio_logical_run_id(
    *,
    dataset_ids: Sequence[str],
    strategy: str,
    params: Dict,
    config: BacktestConfig,
    start: str,
    end: str,
    normalize_weights: bool,
    max_gross_exposure: float,
) -> str:
    payload = {
        "dataset_ids": list(dataset_ids),
        "strategy": strategy,
        "source_fingerprint": backtest_source_fingerprint(),
        "params": params,
        "timeframe": config.timeframe,
        "start": start,
        "end": end,
        "starting_cash": config.starting_cash,
        "fill_on_close": config.fill_on_close,
        "allow_short": config.allow_short,
        "fill_ratio": config.fill_ratio,
        "fee_rate": config.fee_rate,
        "fee_schedule": config.fee_schedule,
        "slippage": config.slippage,
        "slippage_schedule": config.slippage_schedule,
        "risk_free_rate": config.risk_free_rate,
        "sharpe_basis": config.sharpe_basis,
        "sharpe_annualization": config.sharpe_annualization,
        "sharpe_session_seconds_per_day": config.sharpe_session_seconds_per_day,
        "margin_enabled": config.margin_enabled,
        "max_gross_leverage": config.max_gross_leverage,
        "position_sizing_model": config.position_sizing_model,
        "annual_vol_window": config.annual_vol_window,
        "annual_vol_min_periods": config.annual_vol_min_periods,
        "annual_vol_floor": config.annual_vol_floor,
        "max_volatility_multiplier": config.max_volatility_multiplier,
        "min_position_shares": config.min_position_shares,
        "normalize_weights": normalize_weights,
        "max_gross_exposure": max_gross_exposure,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
