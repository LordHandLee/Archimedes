from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Dict, Sequence

import pandas as pd

if TYPE_CHECKING:
    from .engine import BacktestConfig


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
        "normalize_weights": normalize_weights,
        "max_gross_exposure": max_gross_exposure,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
