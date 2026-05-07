from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import sqlite3
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .catalog import ResultCatalog
from .duckdb_store import DuckDBStore
from .engine import BacktestConfig
from .live_market import (
    DEFAULT_LIVE_PROVIDER,
    LIVE_BAR_TIMEFRAME,
    InteractiveBrokersRealtimeBarApp,
    InteractiveBrokersRealtimeConfig,
    LiveMarketBar,
    LiveMarketDataStore,
    chart_timeframe_delta,
    chart_timeframe_to_pandas_rule,
    normalize_chart_timeframe,
    resample_ohlcv,
    sqlite_error_is_locked,
)
from .provider_config import load_provider_settings
from .sample_strategies import InverseTurtleStrategy, SMACrossStrategy, ZScoreMeanReversionStrategy
from .sizing import POSITION_SIZING_ANNUAL_VOLATILITY, _annual_vol_rolling_lengths, normalize_position_sizing_model


IB_CLIENT_OFFSET_LIVE_DEPLOYMENT = 6000
_DATASET_ID_SOURCE_MARKERS = (
    "massive_interactive_brokers",
    "interactive_brokers",
    "interactivebrokers",
    "massive",
    "polygon",
    "stooq",
    "alpaca",
    "ib",
)
_DATASET_PICKER_PROVIDER_PREFIXES = {"IB", "INTERACTIVE_BROKERS", "MASSIVE", "POLYGON", "STOOQ"}


@dataclass
class DeploymentRunnerConfig:
    catalog_path: Path
    live_store_path: Path
    poll_interval_seconds: float = 0.25
    command_batch_size: int = 25
    client_offset_base: int = IB_CLIENT_OFFSET_LIVE_DEPLOYMENT
    run_once: bool = False
    streams_enabled: bool = True


@dataclass
class DeploymentContext:
    context_id: str
    deployment_id: str
    parent_deployment_id: str
    portfolio_id: str
    symbol: str
    dataset_id: str
    strategy_block_id: str
    strategy_name: str
    strategy_version: str
    params: dict
    timeframe: str
    candidate_id: str
    source_type: str
    source_id: str
    target_id: str
    target_name: str
    webhook_url: str
    secret: str
    sizing: dict
    sizing_config_source: str
    account_snapshot: dict
    position_qty: float
    avg_price: float
    last_processed_bar_ts_ns: int = 0
    last_signal_bar_ts_ns: int = 0
    cached_bars: pd.DataFrame | None = None

    def as_payload(self) -> dict:
        return {
            "context_id": self.context_id,
            "deployment_id": self.deployment_id,
            "parent_deployment_id": self.parent_deployment_id,
            "portfolio_id": self.portfolio_id,
            "symbol": self.symbol,
            "dataset_id": self.dataset_id,
            "strategy_block_id": self.strategy_block_id,
            "strategy_name": self.strategy_name,
            "strategy_version": self.strategy_version,
            "params": dict(self.params or {}),
            "timeframe": self.timeframe,
            "candidate_id": self.candidate_id,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "target_name": self.target_name,
            "webhook_url": self.webhook_url,
            "secret": self.secret,
            "sizing": dict(self.sizing or {}),
            "sizing_config_source": self.sizing_config_source,
            "account_snapshot": dict(self.account_snapshot or {}),
            "position_qty": float(self.position_qty or 0.0),
            "avg_price": float(self.avg_price or 0.0),
            "last_processed_bar_ts_ns": int(self.last_processed_bar_ts_ns or 0),
            "last_signal_bar_ts_ns": int(self.last_signal_bar_ts_ns or 0),
            "cached_bars": self.cached_bars,
        }


def _decode_json_dict(raw) -> dict:
    if isinstance(raw, dict):
        return dict(raw)
    if not raw:
        return {}
    try:
        decoded = json.loads(str(raw))
    except Exception:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _normalize_picker_ticker(symbol: object) -> str:
    return str(symbol or "").strip().upper()


def market_symbol_from_dataset_id(dataset_id: object, symbol: object = None) -> str:
    normalized_symbol = _normalize_picker_ticker(symbol)
    if normalized_symbol:
        return normalized_symbol
    dataset_text = str(dataset_id or "").strip()
    if not dataset_text:
        return ""
    if dataset_text.lower().endswith((".csv", ".parquet", ".feather")):
        dataset_text = Path(dataset_text).stem
    colon_parts = [part.strip() for part in dataset_text.split(":") if part.strip()]
    if len(colon_parts) >= 2:
        return colon_parts[1].upper()
    normalized_text = dataset_text.replace("-", "_").replace(" ", "_")
    lowered = normalized_text.lower()
    for source_marker in _DATASET_ID_SOURCE_MARKERS:
        marker = f"_{source_marker}_"
        marker_idx = lowered.find(marker)
        if marker_idx > 0:
            candidate = dataset_text[:marker_idx].strip("_- ")
            if candidate:
                return candidate.upper()
    tokens = [token for token in re.split(r"[_\s]+", dataset_text) if token]
    if tokens and tokens[0].upper() in _DATASET_PICKER_PROVIDER_PREFIXES and len(tokens) > 1:
        return str(tokens[1]).strip().split(".")[0].upper()
    return dataset_text.upper()


def normalize_cached_bars(bars: object) -> pd.DataFrame:
    columns = ["open", "high", "low", "close", "volume"]
    if not isinstance(bars, pd.DataFrame) or bars.empty:
        return pd.DataFrame(columns=columns)
    frame = bars.copy()
    for column in columns:
        if column not in frame.columns:
            frame[column] = 0.0
    index = pd.to_datetime(frame.index, utc=True, errors="coerce")
    valid_mask = ~pd.isna(index)
    if not bool(np.asarray(valid_mask).any()):
        return pd.DataFrame(columns=columns)
    frame = frame.loc[valid_mask, columns].copy()
    frame.index = pd.DatetimeIndex(index[valid_mask]).tz_convert("UTC")
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=["open", "high", "low", "close"])
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    return frame[columns].astype(float)


def live_record_ohlcv_frame(record: dict, *, symbol: str = "") -> pd.DataFrame:
    columns = ["open", "high", "low", "close", "volume"]
    if not isinstance(record, dict):
        return pd.DataFrame(columns=columns)
    record_symbol = market_symbol_from_dataset_id(record.get("symbol"))
    if symbol and record_symbol and record_symbol != market_symbol_from_dataset_id(symbol):
        return pd.DataFrame(columns=columns)
    timestamp = pd.to_datetime(record.get("ts_utc"), utc=True, errors="coerce")
    if pd.isna(timestamp):
        return pd.DataFrame(columns=columns)
    values: dict[str, float] = {}
    for column in columns:
        try:
            values[column] = float(record.get(column) or 0.0)
        except Exception:
            values[column] = 0.0
    if not all(np.isfinite(values[column]) for column in ("open", "high", "low", "close")):
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame([values], index=pd.DatetimeIndex([pd.Timestamp(timestamp).tz_convert("UTC")]))
    return frame[columns]


def live_monitor_record_bucket_timestamp(record_ts: object, timeframe: str) -> pd.Timestamp | None:
    ts = pd.to_datetime(record_ts, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    ts = pd.Timestamp(ts).tz_convert("UTC").floor("min")
    normalized = normalize_chart_timeframe(timeframe)
    if normalized == LIVE_BAR_TIMEFRAME:
        return ts
    try:
        bucket_start = ts.floor(chart_timeframe_to_pandas_rule(normalized))
    except Exception:
        bucket_start = ts.floor("min")
    return pd.Timestamp(bucket_start).tz_convert("UTC")


def cached_bars_with_record(
    cached_bars: object,
    record: dict,
    *,
    timeframe: str,
    symbol: str,
) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    bars = normalize_cached_bars(cached_bars)
    frame = live_record_ohlcv_frame(record, symbol=symbol)
    if bars.empty:
        if frame.empty:
            return bars, None
        bars = resample_ohlcv(frame, timeframe)
        return normalize_cached_bars(bars), (pd.Timestamp(bars.index[-1]).tz_convert("UTC") if not bars.empty else None)
    if frame.empty:
        return bars, None
    source_ts = pd.Timestamp(frame.index[-1]).tz_convert("UTC")
    bucket_ts = live_monitor_record_bucket_timestamp(source_ts, timeframe)
    if bucket_ts is None:
        return bars, None
    row = frame.iloc[-1]
    incoming = {
        "open": float(row.get("open") or 0.0),
        "high": float(row.get("high") or 0.0),
        "low": float(row.get("low") or 0.0),
        "close": float(row.get("close") or 0.0),
        "volume": float(row.get("volume") or 0.0),
    }
    if not all(np.isfinite(incoming[column]) for column in ("open", "high", "low", "close")):
        return bars, None
    if bucket_ts in bars.index:
        existing = bars.loc[bucket_ts]
        if isinstance(existing, pd.DataFrame):
            existing = existing.iloc[-1]
        bars.loc[bucket_ts, ["open", "high", "low", "close", "volume"]] = [
            float(existing.get("open") or incoming["open"]),
            max(float(existing.get("high") or incoming["high"]), incoming["high"]),
            min(float(existing.get("low") or incoming["low"]), incoming["low"]),
            incoming["close"],
            max(float(existing.get("volume") or 0.0), incoming["volume"]),
        ]
    else:
        bars = pd.concat([bars, pd.DataFrame([incoming], index=pd.DatetimeIndex([bucket_ts]))])
    return normalize_cached_bars(bars), bucket_ts


def live_record_evaluation_timestamp(record: dict) -> pd.Timestamp | None:
    if not isinstance(record, dict):
        return None
    ts = pd.to_datetime(record.get("ts_utc"), utc=True, errors="coerce")
    received_at = pd.to_datetime(record.get("received_at"), utc=True, errors="coerce")
    if pd.isna(ts) and pd.isna(received_at):
        return None
    if pd.isna(ts):
        return pd.Timestamp(received_at).tz_convert("UTC")
    if pd.isna(received_at):
        return pd.Timestamp(ts).tz_convert("UTC")
    ts = pd.Timestamp(ts).tz_convert("UTC")
    received_at = pd.Timestamp(received_at).tz_convert("UTC")
    return received_at if received_at > ts else ts


def completed_bar_index(bars: pd.DataFrame, current_ts: object, timeframe: str) -> int | None:
    if bars is None or bars.empty:
        return None
    ts = pd.to_datetime(current_ts, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    index = pd.DatetimeIndex(bars.index)
    if index.tz is None:
        index = index.tz_localize("UTC")
    else:
        index = index.tz_convert("UTC")
    completed_at = index + chart_timeframe_delta(timeframe)
    eligible = completed_at <= pd.Timestamp(ts).tz_convert("UTC")
    if not bool(eligible.any()):
        return None
    return int(np.flatnonzero(np.asarray(eligible))[-1])


def max_runner_evaluation_gap(timeframe: str) -> pd.Timedelta:
    delta = chart_timeframe_delta(timeframe)
    normalized = normalize_chart_timeframe(timeframe)
    if normalized.endswith("d"):
        return delta * 2
    return max(delta * 2, pd.Timedelta(hours=2))


def signal_plan(previous_position: float, target_percent: float) -> list[tuple[str, str, float]]:
    def _sign(value: float) -> int:
        if value > 1e-9:
            return 1
        if value < -1e-9:
            return -1
        return 0

    previous_sign = _sign(float(previous_position or 0.0))
    desired_sign = _sign(float(target_percent or 0.0))
    if previous_sign == desired_sign:
        return []
    if desired_sign == 0:
        side = "LONG" if previous_sign > 0 else "SHORT"
        return [("EXIT", side, 0.0)] if previous_sign else []
    desired_side = "LONG" if desired_sign > 0 else "SHORT"
    if previous_sign == 0:
        return [("ENTRY", desired_side, float(desired_sign))]
    previous_side = "LONG" if previous_sign > 0 else "SHORT"
    return [("EXIT", previous_side, 0.0), ("ENTRY", desired_side, float(desired_sign))]


def strategy_class(strategy_name: str):
    mapping = {
        "SMACrossStrategy": SMACrossStrategy,
        "ZScoreMeanReversionStrategy": ZScoreMeanReversionStrategy,
        "InverseTurtleStrategy": InverseTurtleStrategy,
    }
    return mapping.get(str(strategy_name or "").strip())


def deployment_webhook_url_for_target(target_row: dict) -> str:
    base_url = str(target_row.get("base_url") or "").strip().rstrip("/")
    webhook_path = str(target_row.get("webhook_path") or "").strip()
    if webhook_path.startswith("http://") or webhook_path.startswith("https://"):
        return webhook_path
    if not base_url or not webhook_path:
        return ""
    return f"{base_url}{webhook_path if webhook_path.startswith('/') else '/' + webhook_path}"


def deployment_secret_value(target_row: dict) -> str:
    saved_secret = str(target_row.get("secret_value") or "").strip()
    if saved_secret:
        return saved_secret
    secret_ref = str(target_row.get("secret_ref") or "").strip()
    if not secret_ref:
        return ""
    return os.environ.get(secret_ref, "").strip()


def post_deployment_webhook_payload(url: str, payload: dict) -> dict:
    data = json.dumps(payload, separators=(",", ":"), default=str).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            body = response.read().decode("utf-8", errors="replace")
            status_code = int(response.getcode() or 0)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {int(exc.code)} from execution engine: {body[:500]}") from exc
    except Exception as exc:
        raise RuntimeError(f"Unable to reach execution engine webhook: {exc}") from exc
    try:
        decoded = json.loads(body) if body.strip() else {}
    except Exception:
        decoded = {"raw_response": body}
    if status_code >= 400:
        raise RuntimeError(f"HTTP {status_code} from execution engine: {body[:500]}")
    if isinstance(decoded, dict) and decoded.get("ok") is False:
        raise RuntimeError(str(decoded.get("error") or decoded.get("message") or decoded))
    return decoded if isinstance(decoded, dict) else {"response": decoded}


def deployment_position_qty_from_snapshot(symbol: str, snapshot: dict) -> tuple[float, float]:
    cleaned = market_symbol_from_dataset_id(symbol)
    for position in list((snapshot or {}).get("positions") or []):
        position_symbol = market_symbol_from_dataset_id(position.get("symbol") or position.get("product_id"))
        if position_symbol != cleaned:
            continue
        qty = 0.0
        for key in ("quantity", "qty", "position_qty", "shares", "contracts"):
            try:
                raw_qty = position.get(key)
                if raw_qty not in (None, ""):
                    qty = float(raw_qty)
                    break
            except Exception:
                continue
        side_text = str(
            position.get("side")
            or position.get("position_side")
            or position.get("asset_side")
            or ""
        ).strip().upper()
        if side_text == "SHORT" and qty > 0:
            qty = -qty
        avg_price = 0.0
        for key in ("avg_price", "avg_entry_price", "average_price", "cost_basis_unit"):
            try:
                raw_price = position.get(key)
                if raw_price not in (None, ""):
                    avg_price = float(raw_price)
                    break
            except Exception:
                continue
        return qty, avg_price
    return 0.0, 0.0


def deployment_account_equity(context: dict) -> float | None:
    account = dict(context.get("account_snapshot") or {})
    for key in ("equity", "net_liquidation", "net_liq", "portfolio_value"):
        try:
            value = account.get(key)
            if value not in (None, ""):
                parsed = float(value)
                if np.isfinite(parsed) and parsed > 0.0:
                    return parsed
        except Exception:
            continue
    return None


def deployment_account_buying_power(context: dict) -> float | None:
    account = dict(context.get("account_snapshot") or {})
    for key in ("buying_power", "available_funds", "cash", "cash_balance"):
        try:
            value = account.get(key)
            if value not in (None, ""):
                parsed = float(value)
                if np.isfinite(parsed) and parsed > 0.0:
                    return parsed
        except Exception:
            continue
    return None


def deployment_execution_config_payload(raw: object) -> dict:
    return _decode_json_dict(raw)


def effective_annual_volatility_details(
    bars: pd.DataFrame | None,
    timeframe: str,
    execution_config: dict,
) -> dict:
    normalized_config = deployment_execution_config_payload(execution_config)
    if not normalized_config and isinstance(execution_config, dict):
        normalized_config = dict(execution_config)
    model = normalize_position_sizing_model(execution_config.get("position_sizing_model"))
    details = {
        "position_sizing_model": model,
        "annual_volatility": None,
        "effective_annual_volatility": None,
        "volatility_multiplier": 1.0,
        "raw_volatility_multiplier": 1.0,
        "max_volatility_multiplier": float(normalized_config.get("max_volatility_multiplier", 2.0) or 2.0),
        "annual_vol_floor": float(normalized_config.get("annual_vol_floor", 0.05) or 0.05),
        "volatility_cap_applied": False,
        "annual_vol_floor_applied": False,
        "volatility_model_applied": False,
        "annual_vol_window": int(normalized_config.get("annual_vol_window", 252) or 252),
        "annual_vol_min_periods": int(normalized_config.get("annual_vol_min_periods", 20) or 20),
        "annual_vol_observations": 0,
    }
    if model != POSITION_SIZING_ANNUAL_VOLATILITY:
        return details
    normalized_bars = normalize_cached_bars(bars)
    if normalized_bars.empty:
        return details
    config = BacktestConfig(
        timeframe=normalize_chart_timeframe(timeframe),
        margin_enabled=bool(normalized_config.get("margin_enabled", False)),
        max_gross_leverage=float(normalized_config.get("max_gross_leverage", 1.0) or 1.0),
        position_sizing_model=str(normalized_config.get("position_sizing_model", POSITION_SIZING_ANNUAL_VOLATILITY)),
        annual_vol_window=int(normalized_config.get("annual_vol_window", 252) or 252),
        annual_vol_min_periods=int(normalized_config.get("annual_vol_min_periods", 20) or 20),
        annual_vol_floor=float(normalized_config.get("annual_vol_floor", 0.05) or 0.05),
        max_volatility_multiplier=float(normalized_config.get("max_volatility_multiplier", 2.0) or 2.0),
        min_position_shares=float(normalized_config.get("min_position_shares", 1.0) or 1.0),
    )
    close = pd.to_numeric(normalized_bars["close"], errors="coerce").astype(float)
    returns = close.pct_change()
    window, min_periods, periods_per_year = _annual_vol_rolling_lengths(pd.DatetimeIndex(normalized_bars.index), config)
    annual_vol = returns.rolling(window=window, min_periods=min_periods).std() * np.sqrt(periods_per_year)
    clean_vol = annual_vol.replace([np.inf, -np.inf], np.nan).dropna()
    details.update(
        {
            "annual_vol_window_bars": int(window),
            "annual_vol_min_periods_bars": int(min_periods),
            "annual_vol_periods_per_year": float(periods_per_year),
            "annual_vol_observations": int(returns.dropna().shape[0]),
        }
    )
    if clean_vol.empty:
        return details
    try:
        annual_volatility = float(clean_vol.iloc[-1])
    except Exception:
        annual_volatility = np.nan
    if not np.isfinite(annual_volatility) or annual_volatility <= 0.0:
        return details
    floor = max(1e-9, float(config.annual_vol_floor or 0.05))
    adjusted_volatility = max(annual_volatility, floor)
    raw_multiplier = 1.0 / adjusted_volatility
    max_multiplier = max(0.0, float(config.max_volatility_multiplier or 2.0))
    multiplier = min(raw_multiplier, max_multiplier) if max_multiplier > 0.0 else 0.0
    if not np.isfinite(multiplier) or multiplier <= 0.0:
        multiplier = 1.0
    details.update(
        {
            "annual_volatility": float(annual_volatility),
            "effective_annual_volatility": float(1.0 / multiplier) if multiplier > 0.0 else None,
            "volatility_multiplier": float(multiplier),
            "raw_volatility_multiplier": float(raw_multiplier),
            "max_volatility_multiplier": float(max_multiplier),
            "annual_vol_floor": float(floor),
            "volatility_cap_applied": bool(raw_multiplier > multiplier + 1e-12),
            "annual_vol_floor_applied": bool(annual_volatility < floor),
            "volatility_model_applied": True,
        }
    )
    return details


def position_size_override_payload(
    sizing: dict,
    context: dict,
    *,
    action: str,
    target_percent: float,
    mark_price: float,
    bars: pd.DataFrame | None = None,
) -> dict:
    sizing_payload = dict(sizing or {})
    raw_override = sizing_payload.get("position_size_override")
    override = dict(raw_override) if isinstance(raw_override, dict) else {}
    qty_type = str(sizing_payload.get("qty_type") or "percent_equity").strip().lower()
    execution_config = deployment_execution_config_payload(sizing_payload.get("execution_config"))
    execution_model = normalize_position_sizing_model(execution_config.get("position_sizing_model"))
    try:
        qty_value = float(sizing_payload.get("qty_value"))
    except Exception:
        qty_value = 0.0
    if not override:
        override = {"mode": "none"}
        if str(action or "").upper() == "ENTRY" and qty_value > 0.0:
            if qty_type in {"cash", "target_notional", "notional", "dollars"}:
                override.update(
                    {
                        "mode": "target_notional",
                        "target_notional": qty_value,
                        "target_qty": None,
                        "base_target_notional": qty_value,
                    }
                )
            elif qty_type in {"fixed", "shares", "target_qty", "quantity"}:
                override.update(
                    {
                        "mode": "target_qty",
                        "target_qty": qty_value,
                        "target_notional": qty_value * float(mark_price or 0.0) if mark_price else None,
                    }
                )
            elif qty_type in {"percent_equity", "percent", "pct_equity"}:
                account_equity = deployment_account_equity(context)
                if account_equity is None or account_equity <= 0.0:
                    raise ValueError("account equity is required for percent_equity live sizing")
                buying_power = deployment_account_buying_power(context)
                volatility_details = effective_annual_volatility_details(
                    bars,
                    str(context.get("timeframe") or LIVE_BAR_TIMEFRAME),
                    execution_config,
                )
                annual_vol = volatility_details.get("annual_volatility")
                volatility_multiplier = float(volatility_details.get("volatility_multiplier") or 1.0)
                deployable_equity = account_equity * (qty_value / 100.0)
                base_target_notional = deployable_equity * abs(float(target_percent or 0.0))
                gross_cap_multiplier = float(execution_config.get("max_gross_leverage", 1.0) or 1.0)
                if not bool(execution_config.get("margin_enabled", False)):
                    gross_cap_multiplier = 1.0
                gross_cap_multiplier = max(1.0, gross_cap_multiplier)
                account_notional_cap = account_equity * gross_cap_multiplier
                max_deployment_notional = account_notional_cap
                if execution_model == POSITION_SIZING_ANNUAL_VOLATILITY:
                    max_volatility_multiplier = max(
                        0.0,
                        float(execution_config.get("max_volatility_multiplier", 2.0) or 2.0),
                    )
                    volatility_notional_cap = base_target_notional * max_volatility_multiplier
                    max_deployment_notional = min(max_deployment_notional, volatility_notional_cap)
                if buying_power is not None and buying_power > 0.0 and not bool(execution_config.get("margin_enabled", False)):
                    max_deployment_notional = min(max_deployment_notional, buying_power)
                target_notional = min(base_target_notional * volatility_multiplier, max_deployment_notional)
                if not np.isfinite(target_notional) or target_notional <= 0.0:
                    raise ValueError("percent_equity live sizing resolved to target_notional <= 0")
                override.update(
                    {
                        "mode": "target_notional",
                        "target_notional": float(target_notional),
                        "target_qty": None,
                        "base_target_notional": float(base_target_notional),
                        "deployment_slice_equity": float(deployable_equity),
                        "max_deployment_notional": float(max_deployment_notional),
                        "volatility_multiplier": float(volatility_multiplier),
                        "raw_volatility_multiplier": float(volatility_details.get("raw_volatility_multiplier") or volatility_multiplier),
                        "volatility_multiplier_cap": float(volatility_details.get("max_volatility_multiplier") or 0.0),
                        "volatility_cap_applied": bool(volatility_details.get("volatility_cap_applied", False)),
                        "effective_annual_volatility": volatility_details.get("effective_annual_volatility"),
                        "annual_vol_floor": volatility_details.get("annual_vol_floor"),
                        "annual_vol_floor_applied": bool(volatility_details.get("annual_vol_floor_applied", False)),
                        "account_equity": float(account_equity),
                        "account_buying_power": float(buying_power) if buying_power is not None else None,
                        "account_notional_cap": float(account_notional_cap),
                    }
                )
                if annual_vol is not None and np.isfinite(float(annual_vol)) and float(annual_vol) > 0.0:
                    override["annual_volatility"] = float(annual_vol)
                override["volatility_sizing"] = volatility_details
    for key in (
        "deployment_id",
        "parent_deployment_id",
        "portfolio_id",
        "strategy_block_id",
        "dataset_id",
        "candidate_id",
        "source_type",
        "source_id",
        "strategy_name",
        "strategy_version",
        "timeframe",
    ):
        value = context.get(key)
        if value not in (None, "") and override.get(key) in (None, ""):
            override[key] = value
    min_shares_source = None
    if "min_shares" in sizing_payload:
        min_shares_source = sizing_payload.get("min_shares")
    elif "min_position_shares" in sizing_payload:
        min_shares_source = sizing_payload.get("min_position_shares")
    if min_shares_source not in (None, "") and override.get("min_shares") in (None, ""):
        try:
            min_shares_value = max(0.0, float(min_shares_source or 0.0))
        except Exception:
            min_shares_value = 0.0
        if min_shares_value > 0.0 and str(override.get("mode") or "") == "target_notional":
            try:
                target_notional_for_min = float(override.get("target_notional") or 0.0)
                min_share_notional = min_shares_value * float(mark_price or 0.0)
            except Exception:
                target_notional_for_min = 0.0
                min_share_notional = 0.0
            if target_notional_for_min > 0.0 and min_share_notional > target_notional_for_min and np.isfinite(min_share_notional):
                override["requested_min_shares"] = float(min_shares_value)
                override["min_shares_suppressed_by_target_notional"] = True
                min_shares_value = 0.0
        override["min_shares"] = float(min_shares_value)
    trace = dict(override.get("sizing_trace") or {})
    trace.update(
        {
            "qty_type": qty_type,
            "qty_value": qty_value,
            "strategy_target_percent": float(target_percent or 0.0),
            "mark_price": float(mark_price or 0.0),
            "execution_config": execution_config,
            "sizing_config_source": str(context.get("sizing_config_source") or ""),
        }
    )
    for key in (
        "account_equity",
        "account_buying_power",
        "deployment_slice_equity",
        "base_target_notional",
        "target_notional",
        "max_deployment_notional",
        "annual_volatility",
        "effective_annual_volatility",
        "volatility_multiplier",
        "raw_volatility_multiplier",
        "volatility_multiplier_cap",
        "volatility_cap_applied",
        "annual_vol_floor",
        "annual_vol_floor_applied",
        "min_shares",
        "requested_min_shares",
        "min_shares_suppressed_by_target_notional",
    ):
        if key in override:
            trace[key] = override.get(key)
    override["sizing_trace"] = trace
    return override


def deployment_signal_payload(
    context: dict,
    *,
    action: str,
    side: str,
    target_percent: float,
    bar_ts: pd.Timestamp,
    bar_index: int,
    price: float,
    bars: pd.DataFrame | None = None,
) -> dict:
    timestamp = pd.Timestamp(bar_ts)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    sizing = dict(context.get("sizing") or {})
    qty_type = str(sizing.get("qty_type") or "percent_equity")
    try:
        qty_value = float(sizing.get("qty_value"))
    except Exception:
        qty_value = abs(float(target_percent or 0.0)) * 100.0
    event_id = (
        f"{context.get('deployment_id')}:{context.get('symbol')}:{context.get('timeframe')}:"
        f"{str(action).upper()}:{str(side).upper()}:{int(timestamp.value)}"
    )
    override = position_size_override_payload(
        sizing,
        context,
        action=str(action).upper(),
        target_percent=target_percent,
        mark_price=price,
        bars=bars,
    )
    return {
        "secret": context.get("secret") or "",
        "symbol": context.get("symbol") or "",
        "action": str(action).upper(),
        "side": str(side).upper(),
        "price": float(price),
        "time": int(timestamp.timestamp()),
        "sent_ts": int(pd.Timestamp.now(tz="UTC").timestamp()),
        "event_id": event_id,
        "bar_index": int(bar_index),
        "bar_timestamp_utc": timestamp.isoformat(),
        "deployment_id": context.get("deployment_id") or "",
        "parent_deployment_id": context.get("parent_deployment_id") or "",
        "portfolio_id": context.get("portfolio_id") or "",
        "candidate_id": context.get("candidate_id") or "",
        "source_type": context.get("source_type") or "",
        "source_id": context.get("source_id") or "",
        "dataset_id": context.get("dataset_id") or "",
        "strategy_block_id": context.get("strategy_block_id") or "",
        "strategy_name": context.get("strategy_name") or "",
        "strategy_version": context.get("strategy_version") or "",
        "strategy_params": dict(context.get("params") or {}),
        "timeframe": context.get("timeframe") or "",
        "qty_type": qty_type,
        "qty_value": qty_value,
        "annual_vol": override.get("annual_volatility"),
        "sizing_authority": "quant_backtest_engine",
        "position_size_override": override,
        "sizing_trace": dict(override.get("sizing_trace") or {}),
        "strategy_target_percent": float(target_percent or 0.0),
        "dashboard_origin": "quant_backtest_engine.live_deployment_runner_process",
    }


def deployment_signal_sizing_summary(payload: dict) -> str:
    override = dict((payload or {}).get("position_size_override") or {})
    trace = dict(override.get("sizing_trace") or (payload or {}).get("sizing_trace") or {})

    def _float_value(key: str) -> float | None:
        for source in (override, trace, payload or {}):
            try:
                value = source.get(key)
                if value not in (None, ""):
                    parsed = float(value)
                    if np.isfinite(parsed):
                        return parsed
            except Exception:
                continue
        return None

    def _money(value: float | None) -> str:
        return "n/a" if value is None else f"${value:,.2f}"

    def _ratio(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.4f}"

    mode = str(override.get("mode") or "none")
    parts = [f"mode={mode}"]
    if mode == "target_qty":
        parts.append(f"qty={_ratio(_float_value('target_qty'))}")
        parts.append(f"notional={_money(_float_value('target_notional'))}")
    elif mode == "target_notional":
        parts.append(f"base={_money(_float_value('base_target_notional'))}")
        parts.append(f"final={_money(_float_value('target_notional'))}")
        vol = _float_value("annual_volatility")
        if vol is not None:
            parts.append(f"ann_vol={vol:.2%}")
        mult = _float_value("volatility_multiplier")
        if mult is not None:
            parts.append(f"vol_mult={mult:.2f}x")
    if bool(override.get("min_shares_suppressed_by_target_notional")):
        parts.append("min_shares_suppressed=yes")
    if bool(override.get("volatility_cap_applied")):
        parts.append("vol_cap=yes")
    if bool(override.get("annual_vol_floor_applied")):
        parts.append("vol_floor=yes")
    return "Sizing " + ", ".join(parts)


class LiveDeploymentSignalBroker:
    def __init__(self, *, position_qty: float = 0.0, avg_price: float = 0.0) -> None:
        self.position_qty = float(position_qty or 0.0)
        self.avg_price = float(avg_price or 0.0)
        self.target_percent_calls: list[dict] = []
        self.unsupported_orders: list[str] = []

    def target_percent(
        self,
        target: float,
        mark_price: float,
        earliest_ts: pd.Timestamp | None = None,
        order_type: str = "market",
        limit_price: float | None = None,
        stop_price: float | None = None,
        tag: str | None = None,
    ) -> None:
        self.target_percent_calls.append(
            {
                "target": float(target),
                "mark_price": float(mark_price),
                "earliest_ts": earliest_ts,
                "order_type": order_type,
                "limit_price": limit_price,
                "stop_price": stop_price,
                "tag": tag,
            }
        )

    def cancel_orders(self, tag: str | None = None) -> None:
        self.unsupported_orders.append(f"cancel_orders:{tag or ''}")

    def buy(self, qty: float, earliest_ts: pd.Timestamp | None = None, tag: str | None = None) -> None:
        self.unsupported_orders.append("buy")

    def sell(self, qty: float, earliest_ts: pd.Timestamp | None = None, tag: str | None = None) -> None:
        self.unsupported_orders.append("sell")

    def buy_limit(self, qty: float, limit_price: float, earliest_ts: pd.Timestamp | None = None, tag: str | None = None) -> None:
        self.unsupported_orders.append("buy_limit")

    def sell_limit(self, qty: float, limit_price: float, earliest_ts: pd.Timestamp | None = None, tag: str | None = None) -> None:
        self.unsupported_orders.append("sell_limit")

    def buy_stop(self, qty: float, stop_price: float, earliest_ts: pd.Timestamp | None = None, tag: str | None = None) -> None:
        self.unsupported_orders.append("buy_stop")

    def sell_stop(self, qty: float, stop_price: float, earliest_ts: pd.Timestamp | None = None, tag: str | None = None) -> None:
        self.unsupported_orders.append("sell_stop")


class RunnerSymbolStream:
    def __init__(
        self,
        *,
        symbol: str,
        store: LiveMarketDataStore,
        config: InteractiveBrokersRealtimeConfig,
        bar_callback,
        status_callback,
        error_callback,
    ) -> None:
        self.symbol = market_symbol_from_dataset_id(symbol)
        self.store = store
        self.config = config
        self.bar_callback = bar_callback
        self.status_callback = status_callback
        self.error_callback = error_callback
        self.app: InteractiveBrokersRealtimeBarApp | None = None
        self.last_bar_monotonic: float | None = None

    def start(self) -> None:
        if not self.symbol:
            raise ValueError("A ticker symbol is required for live deployment streaming.")

        def _on_bar(bar: LiveMarketBar) -> None:
            self.last_bar_monotonic = time.monotonic()
            try:
                self.store.upsert_bar(bar)
            except sqlite3.Error as exc:
                if sqlite_error_is_locked(exc):
                    self.status_callback(self.symbol, f"Live market store busy while writing {self.symbol}.")
                    return
                raise
            record = bar.as_record()
            record["is_partial"] = False
            self.bar_callback(self.symbol, record)

        self.app = InteractiveBrokersRealtimeBarApp(
            symbols=[self.symbol],
            config=self.config,
            bar_callback=_on_bar,
            partial_callback=None,
            status_callback=lambda text: self.status_callback(self.symbol, text),
            error_callback=lambda message: self.error_callback(self.symbol, message),
        )
        self.app.connect_and_start()

    def poll(self) -> None:
        if self.app is not None:
            self.app.flush_stale_partials()

    def stop(self) -> None:
        if self.app is not None:
            self.app.stop()
            self.app = None


class LiveDeploymentRunnerService:
    def __init__(self, config: DeploymentRunnerConfig) -> None:
        self.config = config
        self.catalog = ResultCatalog(config.catalog_path)
        self.store = LiveMarketDataStore(config.live_store_path)
        self.runner_id = f"live-runner:{os.getpid()}:{uuid.uuid4().hex[:8]}"
        self.contexts: dict[str, DeploymentContext] = {}
        self.symbol_index: dict[str, set[str]] = {}
        self.streams: dict[str, RunnerSymbolStream] = {}
        self.stream_counter = 0
        self.stop_requested = False

    def emit(
        self,
        event_type: str,
        *,
        deployment_id: str = "",
        context_id: str = "",
        symbol: str = "",
        severity: str = "info",
        message: str = "",
        payload: dict | None = None,
        event_id: str = "",
    ) -> None:
        self.catalog.save_deployment_runner_event(
            event_type=event_type,
            deployment_id=deployment_id,
            context_id=context_id,
            symbol=symbol,
            severity=severity,
            message=message,
            payload_json=payload or {},
            event_id=event_id,
        )

    def run_forever(self) -> None:
        self.emit("runner_started", message=f"Live deployment runner started ({self.runner_id}).", payload={"runner_id": self.runner_id})
        while not self.stop_requested:
            self.process_queued_commands()
            for symbol, stream in list(self.streams.items()):
                try:
                    stream.poll()
                except Exception as exc:
                    self.emit("stream_error", symbol=symbol, severity="error", message=str(exc))
            if self.config.run_once:
                break
            time.sleep(max(0.05, float(self.config.poll_interval_seconds)))
        self.stop_all_streams()
        self.emit("runner_stopped", message=f"Live deployment runner stopped ({self.runner_id}).", payload={"runner_id": self.runner_id})

    def process_queued_commands(self) -> None:
        commands = self.catalog.claim_deployment_runner_commands(
            runner_id=self.runner_id,
            limit=self.config.command_batch_size,
        )
        for command in commands:
            error = ""
            try:
                payload = _decode_json_dict(command.payload_json)
                command_type = str(command.command_type or "").strip().lower()
                deployment_id = str(command.deployment_id or payload.get("deployment_id") or "")
                if command_type in {"arm", "start"}:
                    self.arm_deployment(deployment_id)
                elif command_type == "pause":
                    self.deactivate_deployment(deployment_id, status="paused")
                elif command_type == "stop":
                    self.deactivate_deployment(deployment_id, status="stopped")
                elif command_type == "shutdown_runner":
                    self.stop_requested = True
                elif command_type == "sync":
                    self.emit("runner_sync", deployment_id=deployment_id, message="Runner sync command acknowledged.")
                else:
                    raise ValueError(f"Unsupported live runner command: {command.command_type}")
            except Exception as exc:
                error = str(exc)
                deployment_id = str(command.deployment_id or "")
                if deployment_id:
                    self._update_deployment_status_tree(
                        deployment_id,
                        status="error",
                        status_reason=error[:500],
                        last_error_at=pd.Timestamp.now(tz="UTC").isoformat(),
                    )
                self.emit(
                    "command_error",
                    deployment_id=deployment_id,
                    severity="error",
                    message=error,
                    payload={"command_id": command.command_id, "command_type": command.command_type},
                )
            finally:
                self.catalog.finish_deployment_runner_command(command.command_id, error=error)

    def arm_deployment(self, deployment_id: str) -> None:
        deployment_id = str(deployment_id or "")
        if not deployment_id:
            raise ValueError("Deployment ID is required to arm a live deployment.")
        self.deactivate_deployment(deployment_id, status="", update_status=False)
        contexts, failures = self.build_contexts(deployment_id)
        if failures or not contexts:
            message = "; ".join(failures) if failures else "No executable live deployment contexts were built."
            self._update_deployment_status_tree(
                deployment_id,
                status="error",
                status_reason=message,
                last_error_at=pd.Timestamp.now(tz="UTC").isoformat(),
            )
            raise ValueError(message)
        conflicts = self.context_conflicts(contexts, deployment_id)
        if conflicts:
            message = "Live context conflict: " + "; ".join(conflicts)
            self._update_deployment_status_tree(
                deployment_id,
                status="error",
                status_reason=message,
                last_error_at=pd.Timestamp.now(tz="UTC").isoformat(),
            )
            raise ValueError(message)
        for context in contexts:
            self.contexts[context.context_id] = context
            self.symbol_index.setdefault(context.symbol, set()).add(context.context_id)
        if self.config.streams_enabled:
            for symbol in sorted(self.symbol_index):
                self.start_stream(symbol)
        timestamp = pd.Timestamp.now(tz="UTC").isoformat()
        self._update_deployment_status_tree(
            deployment_id,
            status="live",
            status_reason="Live runner process active; evaluating completed strategy bars.",
            armed_at=timestamp,
            started_at=timestamp,
        )
        self.emit(
            "deployment_armed",
            deployment_id=deployment_id,
            message=f"Deployment armed across {len(contexts)} strategy leg{'s' if len(contexts) != 1 else ''}.",
            payload={"contexts": [context.context_id for context in contexts], "symbols": sorted(self.symbol_index)},
        )
        for context in list(contexts):
            self.evaluate_context_now(context)

    def deactivate_deployment(self, deployment_id: str, *, status: str = "stopped", update_status: bool = True) -> None:
        root_id = str(deployment_id or "")
        if not root_id:
            return
        removed_symbols: set[str] = set()
        for context_id, context in list(self.contexts.items()):
            if root_id in {context.deployment_id, context.parent_deployment_id, context.portfolio_id}:
                self.contexts.pop(context_id, None)
                indexed = self.symbol_index.get(context.symbol)
                if indexed is not None:
                    indexed.discard(context_id)
                    if not indexed:
                        self.symbol_index.pop(context.symbol, None)
                        removed_symbols.add(context.symbol)
        for symbol in removed_symbols:
            self.stop_stream(symbol)
        if update_status and status:
            timestamp = pd.Timestamp.now(tz="UTC").isoformat()
            kwargs = {"stopped_at": timestamp} if status == "stopped" else {}
            self._update_deployment_status_tree(root_id, status=status, status_reason="", **kwargs)
            self.emit(
                f"deployment_{status}",
                deployment_id=root_id,
                message=f"Deployment {status}.",
                payload={"deployment_id": root_id},
            )

    def context_conflicts(self, contexts: Sequence[DeploymentContext], root_deployment_id: str) -> list[str]:
        conflicts: list[str] = []
        seen_new: dict[tuple[str, str], str] = {}
        root_id = str(root_deployment_id or "")
        for context in list(contexts or []):
            key = (context.target_id, context.symbol)
            label = context.deployment_id[:10]
            if key in seen_new:
                conflicts.append(f"{key[1]} on {key[0]} appears in both {seen_new[key]} and {label}")
            else:
                seen_new[key] = label
        for context in list(self.contexts.values()):
            if root_id in {context.deployment_id, context.parent_deployment_id, context.portfolio_id}:
                continue
            key = (context.target_id, context.symbol)
            if key in seen_new:
                conflicts.append(f"{key[1]} on {key[0]} is already live in deployment {context.deployment_id[:10]}")
        return conflicts

    def build_contexts(self, deployment_id: str) -> tuple[list[DeploymentContext], list[str]]:
        deployments = {row.deployment_id: asdict(row) for row in self.catalog.load_deployments()}
        target_rows = {row.target_id: asdict(row) for row in self.catalog.load_deployment_targets()}
        child_links = self.catalog.load_deployment_child_links()
        deployment_row = deployments.get(str(deployment_id or ""))
        if not deployment_row:
            return [], [f"Deployment {deployment_id} does not exist."]
        target_row = target_rows.get(str(deployment_row.get("target_id") or ""), {})
        if not target_row:
            return [], ["The selected deployment target is not configured."]
        webhook_url = deployment_webhook_url_for_target(target_row)
        if not webhook_url:
            return [], ["The selected deployment target has no webhook URL."]
        secret = deployment_secret_value(target_row)
        if not secret:
            return [], [f"Missing {target_row.get('secret_ref') or 'webhook secret'}."]
        snapshot = fetch_target_external_snapshot(target_row)
        account_snapshot = dict(snapshot.get("account") or {})
        activation_rows = self.activation_rows(deployment_row, deployments, child_links)
        parent_id = str(deployment_row.get("deployment_id") or "")
        contexts: list[DeploymentContext] = []
        failures: list[str] = []
        for row in activation_rows:
            child_id = str(row.get("deployment_id") or "")
            symbol_source = str(row.get("symbol") or row.get("dataset_id") or "").strip()
            symbol = market_symbol_from_dataset_id(symbol_source)
            if not symbol:
                failures.append(f"{child_id[:10] or 'deployment'}: missing symbol")
                continue
            strategy_name = str(row.get("strategy") or "").strip()
            strategy_cls = strategy_class(strategy_name)
            if strategy_cls is None:
                failures.append(f"{symbol}: unsupported live strategy {strategy_name or 'unknown'}")
                continue
            params = _decode_json_dict(row.get("params_json"))
            if strategy_name == "InverseTurtleStrategy" and bool(params.get("use_atr_stop", True)):
                failures.append(f"{symbol}: InverseTurtleStrategy live ATR stop orders are not supported yet")
                continue
            timeframe = normalize_chart_timeframe(str(row.get("timeframe") or LIVE_BAR_TIMEFRAME))
            position_qty, avg_price = deployment_position_qty_from_snapshot(symbol, snapshot)
            parent_deployment_id = str(row.get("parent_deployment_id") or "")
            strategy_block_id = self.strategy_block_id_for_child(child_id, child_links)
            context_id = f"{parent_id or child_id}:{child_id}:{symbol}"
            sizing_payload = _decode_json_dict(row.get("sizing_json"))
            contexts.append(
                DeploymentContext(
                    context_id=context_id,
                    deployment_id=child_id,
                    parent_deployment_id=parent_deployment_id,
                    portfolio_id=parent_deployment_id or (parent_id if parent_id != child_id else ""),
                    symbol=symbol,
                    dataset_id=str(row.get("dataset_id") or ""),
                    strategy_block_id=strategy_block_id,
                    strategy_name=strategy_name,
                    strategy_version=str(row.get("strategy_version") or ""),
                    params=params,
                    timeframe=timeframe,
                    candidate_id=str(row.get("candidate_id") or ""),
                    source_type=str(row.get("source_type") or ""),
                    source_id=str(row.get("source_id") or ""),
                    target_id=str(row.get("target_id") or deployment_row.get("target_id") or ""),
                    target_name=str(target_row.get("name") or ""),
                    webhook_url=webhook_url,
                    secret=secret,
                    sizing=sizing_payload,
                    sizing_config_source="deployment",
                    account_snapshot=account_snapshot,
                    position_qty=position_qty,
                    avg_price=avg_price,
                )
            )
        return contexts, failures

    def activation_rows(self, deployment_row: dict, deployments: dict[str, dict], child_links: Sequence[object]) -> list[dict]:
        deployment_id = str(deployment_row.get("deployment_id") or "")
        child_ids = {
            str(link.child_deployment_id)
            for link in child_links
            if str(link.parent_deployment_id or "") == deployment_id and str(link.child_deployment_id or "")
        }
        rows = [deployments[child_id] for child_id in child_ids if child_id in deployments]
        return rows if rows else [dict(deployment_row)]

    @staticmethod
    def strategy_block_id_for_child(child_deployment_id: str, child_links: Sequence[object]) -> str:
        for link in child_links:
            if str(link.child_deployment_id or "") == str(child_deployment_id or ""):
                return str(link.strategy_block_id or "")
        return ""

    def _update_deployment_status_tree(self, deployment_id: str, **kwargs) -> None:
        root_id = str(deployment_id or "")
        if not root_id:
            return
        self.catalog.update_deployment_status(root_id, **kwargs)
        for child_id in self.child_ids(root_id):
            self.catalog.update_deployment_status(child_id, **kwargs)

    def child_ids(self, deployment_id: str) -> list[str]:
        return [
            str(link.child_deployment_id)
            for link in self.catalog.load_deployment_child_links()
            if str(link.parent_deployment_id or "") == str(deployment_id or "") and str(link.child_deployment_id or "")
        ]

    def build_ib_config(self, *, client_offset: int = 0) -> InteractiveBrokersRealtimeConfig:
        settings = load_provider_settings(DEFAULT_LIVE_PROVIDER, catalog=self.catalog)

        def _int_value(value: object, fallback: int) -> int:
            try:
                return int(value)
            except Exception:
                return fallback

        def _float_value(value: object, fallback: float) -> float:
            try:
                return float(value)
            except Exception:
                return fallback

        return InteractiveBrokersRealtimeConfig(
            host=str(settings.get("host") or os.getenv("IB_HOST", "127.0.0.1")),
            port=_int_value(settings.get("port") or os.getenv("IB_PORT", "7497"), 7497),
            client_id=_int_value(settings.get("client_id") or os.getenv("IB_CLIENT_ID", "9301"), 9301) + int(client_offset),
            primary_exchange=str(settings.get("primary_exchange") or os.getenv("IB_PRIMARY_EXCHANGE", "")),
            use_rth=False,
            timeout_seconds=_float_value(settings.get("timeout_seconds"), 15.0),
        )

    def start_stream(self, symbol: str) -> None:
        cleaned = market_symbol_from_dataset_id(symbol)
        if not cleaned or cleaned in self.streams:
            return
        self.stream_counter += 1
        stream = RunnerSymbolStream(
            symbol=cleaned,
            store=self.store,
            config=self.build_ib_config(client_offset=self.config.client_offset_base + self.stream_counter),
            bar_callback=self.on_stream_bar,
            status_callback=self.on_stream_status,
            error_callback=self.on_stream_error,
        )
        try:
            stream.start()
        except Exception as exc:
            self.emit("stream_error", symbol=cleaned, severity="error", message=str(exc))
            raise
        self.streams[cleaned] = stream
        self.emit("stream_started", symbol=cleaned, message=f"Live deployment stream active for {cleaned}.")

    def stop_stream(self, symbol: str) -> None:
        cleaned = market_symbol_from_dataset_id(symbol)
        stream = self.streams.pop(cleaned, None)
        if stream is None:
            return
        try:
            stream.stop()
        finally:
            self.emit("stream_stopped", symbol=cleaned, message=f"Live deployment stream stopped for {cleaned}.")

    def stop_all_streams(self) -> None:
        for symbol in list(self.streams):
            self.stop_stream(symbol)

    def on_stream_status(self, symbol: str, message: str) -> None:
        self.emit("stream_status", symbol=symbol, message=message)

    def on_stream_error(self, symbol: str, message: str) -> None:
        self.emit("stream_error", symbol=symbol, severity="error", message=message)

    def on_stream_bar(self, symbol: str, record: dict) -> None:
        self.process_record(record)

    def evaluate_context_now(self, context: DeploymentContext) -> None:
        now = pd.Timestamp.now(tz="UTC")
        self.process_context_bar(context, None, now, context.symbol)

    def process_record(self, record: dict) -> None:
        symbol = market_symbol_from_dataset_id(record.get("symbol"))
        if not symbol:
            return
        evaluation_ts = live_record_evaluation_timestamp(record)
        if evaluation_ts is None:
            return
        for context_id in list(self.symbol_index.get(symbol, set())):
            context = self.contexts.get(context_id)
            if context is not None:
                self.process_context_bar(context, record, evaluation_ts, symbol)

    def load_context_bars(self, context: DeploymentContext, *, record: dict | None = None) -> pd.DataFrame:
        symbol = market_symbol_from_dataset_id(context.symbol)
        timeframe = normalize_chart_timeframe(context.timeframe or LIVE_BAR_TIMEFRAME)
        lookback_days = live_deployment_history_lookback_days(context.as_payload())
        end_ts = pd.Timestamp.now(tz="UTC")
        start_ts = end_ts - pd.Timedelta(days=lookback_days)
        bars = normalize_cached_bars(context.cached_bars)
        stale_cache_last_ts: pd.Timestamp | None = None
        record_bucket_ts: pd.Timestamp | None = None
        if record is not None and not bars.empty:
            frame = live_record_ohlcv_frame(record, symbol=symbol)
            if not frame.empty:
                record_bucket_ts = live_monitor_record_bucket_timestamp(frame.index[-1], timeframe)
            if record_bucket_ts is not None:
                last_cached_ts = pd.Timestamp(bars.index[-1]).tz_convert("UTC")
                if record_bucket_ts - last_cached_ts > chart_timeframe_delta(timeframe):
                    stale_cache_last_ts = last_cached_ts
                    bars = pd.DataFrame()
        if bars.empty:
            frames: list[pd.DataFrame] = []
            dataset_id = str(context.dataset_id or "").strip()
            if dataset_id:
                try:
                    historical = DuckDBStore().resample(dataset_id, duckdb_interval_for_chart_timeframe(timeframe))
                    if historical is not None and not historical.empty:
                        historical = historical.loc[(historical.index >= start_ts) & (historical.index <= end_ts)]
                        frames.append(historical)
                except Exception:
                    pass
            live = self.store.load_recent_bars(symbol, provider=DEFAULT_LIVE_PROVIDER, limit=500000)
            if live is not None and not live.empty:
                live = live.loc[(live.index >= start_ts) & (live.index <= end_ts)]
                live = resample_ohlcv(live, timeframe)
                if not live.empty:
                    frames.append(live)
            if frames:
                bars = normalize_cached_bars(pd.concat(frames).sort_index())
            if stale_cache_last_ts is not None:
                self.emit(
                    "data_gap_reloaded",
                    deployment_id=context.deployment_id,
                    context_id=context.context_id,
                    symbol=symbol,
                    message=(
                        f"{symbol} runner cache gap detected from {stale_cache_last_ts.isoformat()} "
                        f"to {record_bucket_ts.isoformat() if record_bucket_ts is not None else 'unknown'}; reloaded bars."
                    ),
                    payload={
                        "cached_last_bar_timestamp_utc": stale_cache_last_ts.isoformat(),
                        "incoming_bar_timestamp_utc": record_bucket_ts.isoformat() if record_bucket_ts is not None else "",
                        "reloaded_bar_count": int(len(bars)),
                    },
                )
        if record is not None:
            bars, _updated_ts = cached_bars_with_record(bars, record, timeframe=timeframe, symbol=symbol)
        if not bars.empty:
            cutoff = start_ts - chart_timeframe_delta(timeframe)
            bars = bars.loc[(bars.index >= cutoff) & (bars.index <= end_ts + chart_timeframe_delta(timeframe))]
        context.cached_bars = bars
        return bars

    @staticmethod
    def strategy_decision_trace(strategy, bar_ts: pd.Timestamp, bars: pd.DataFrame) -> dict:
        timestamp = pd.Timestamp(bar_ts)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        trace: dict[str, object] = {"bar_timestamp_utc": timestamp.isoformat()}
        try:
            close = float(bars.loc[timestamp, "close"])
            if np.isfinite(close):
                trace["close"] = close
        except Exception:
            pass
        data = getattr(strategy, "data", None)
        if isinstance(data, pd.DataFrame) and timestamp in data.index:
            row = data.loc[timestamp]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[-1]
            loc = data.index.get_loc(timestamp)
            if isinstance(loc, slice):
                idx = int(loc.stop - 1)
            elif isinstance(loc, np.ndarray):
                idx = int(np.asarray(loc).nonzero()[0][-1])
            else:
                idx = int(loc)
            prev = data.iloc[idx - 1] if idx > 0 else pd.Series(dtype=float)
            for key in ("fast", "slow"):
                try:
                    value = float(row.get(key))
                    if np.isfinite(value):
                        trace[key] = value
                except Exception:
                    pass
                try:
                    value = float(prev.get(key))
                    if np.isfinite(value):
                        trace[f"prev_{key}"] = value
                except Exception:
                    pass
            if {"fast", "slow", "prev_fast", "prev_slow"}.issubset(trace):
                trace["crossed_above"] = bool(trace["fast"] > trace["slow"] and trace["prev_fast"] <= trace["prev_slow"])
                trace["crossed_below"] = bool(trace["fast"] < trace["slow"] and trace["prev_fast"] >= trace["prev_slow"])
        features = getattr(strategy, "features", None)
        if isinstance(features, pd.DataFrame) and timestamp in features.index:
            feature_row = features.loc[timestamp]
            if isinstance(feature_row, pd.DataFrame):
                feature_row = feature_row.iloc[-1]
            for key in ("long_entry_signal", "long_exit_signal", "z_score"):
                if key in feature_row:
                    value = feature_row.get(key)
                    if isinstance(value, (bool, np.bool_)):
                        trace[key] = bool(value)
                    else:
                        try:
                            parsed = float(value)
                            if np.isfinite(parsed):
                                trace[key] = parsed
                        except Exception:
                            pass
        return trace

    def process_context_bar(
        self,
        context: DeploymentContext,
        record: dict | None,
        record_ts: pd.Timestamp,
        symbol: str,
    ) -> None:
        timeframe = normalize_chart_timeframe(context.timeframe or LIVE_BAR_TIMEFRAME)
        try:
            bars = self.load_context_bars(context, record=record)
            source_bar_index = completed_bar_index(bars, record_ts, timeframe)
            if source_bar_index is None:
                return
            bar_ts = pd.Timestamp(bars.index[int(source_bar_index)])
            bar_ts = bar_ts.tz_convert("UTC") if bar_ts.tzinfo else bar_ts.tz_localize("UTC")
            bar_ts_ns = int(bar_ts.value)
            max_gap = max_runner_evaluation_gap(timeframe)
            if record is None and pd.Timestamp(record_ts).tz_convert("UTC") - bar_ts > max_gap:
                self.emit(
                    "data_stale_skipped",
                    deployment_id=context.deployment_id,
                    context_id=context.context_id,
                    symbol=symbol,
                    severity="warning",
                    message=f"{symbol} skipped stale runner evaluation at {bar_ts.isoformat()}; waiting for current live bars.",
                    payload={
                        "bar_timestamp_utc": bar_ts.isoformat(),
                        "record_timestamp_utc": pd.Timestamp(record_ts).tz_convert("UTC").isoformat(),
                        "max_gap_seconds": float(max_gap.total_seconds()),
                    },
                )
                return
            if source_bar_index > 0:
                prev_ts = pd.Timestamp(bars.index[int(source_bar_index) - 1])
                prev_ts = prev_ts.tz_convert("UTC") if prev_ts.tzinfo else prev_ts.tz_localize("UTC")
                observed_gap = bar_ts - prev_ts
                if observed_gap > max_gap:
                    self.emit(
                        "data_gap_skipped",
                        deployment_id=context.deployment_id,
                        context_id=context.context_id,
                        symbol=symbol,
                        severity="warning",
                        message=(
                            f"{symbol} skipped {bar_ts.isoformat()} because previous runner bar "
                            f"was {prev_ts.isoformat()}."
                        ),
                        payload={
                            "bar_timestamp_utc": bar_ts.isoformat(),
                            "previous_bar_timestamp_utc": prev_ts.isoformat(),
                            "gap_seconds": float(observed_gap.total_seconds()),
                            "max_gap_seconds": float(max_gap.total_seconds()),
                        },
                    )
                    return
            if bar_ts_ns <= int(context.last_processed_bar_ts_ns or 0):
                return
            cls = strategy_class(context.strategy_name)
            if cls is None:
                raise ValueError(f"Unsupported live strategy {context.strategy_name or 'unknown'}")
            strategy_bars = bars.iloc[: int(source_bar_index) + 1].copy()
            strategy = cls(**dict(context.params or {}))
            strategy.initialize(strategy_bars)
            broker = LiveDeploymentSignalBroker(position_qty=context.position_qty, avg_price=context.avg_price)
            bar_row = strategy_bars.iloc[-1]
            strategy.on_bar(bar_ts, bar_row, broker)
            decision_trace = self.strategy_decision_trace(strategy, bar_ts, strategy_bars)
            context.last_processed_bar_ts_ns = bar_ts_ns
            if broker.unsupported_orders:
                raise ValueError(
                    "Live runner only supports strategies that express desired state with target_percent; "
                    f"unsupported broker calls: {', '.join(sorted(set(broker.unsupported_orders)))}"
                )
            if not broker.target_percent_calls:
                self.emit(
                    "signal_evaluated",
                    deployment_id=context.deployment_id,
                    context_id=context.context_id,
                    symbol=symbol,
                    message=f"{symbol} evaluated {bar_ts.isoformat()}: no strategy signal.",
                    payload={
                        "bar_timestamp_utc": bar_ts.isoformat(),
                        "decision": "no_signal",
                        "strategy_decision_trace": decision_trace,
                    },
                )
                return
            target_call = broker.target_percent_calls[-1]
            target = float(target_call.get("target") or 0.0)
            plan = signal_plan(float(context.position_qty or 0.0), target)
            if not plan:
                self.emit(
                    "signal_evaluated",
                    deployment_id=context.deployment_id,
                    context_id=context.context_id,
                    symbol=symbol,
                    message=f"{symbol} evaluated {bar_ts.isoformat()}: target unchanged.",
                    payload={
                        "bar_timestamp_utc": bar_ts.isoformat(),
                        "decision": "target_unchanged",
                        "target_percent": target,
                        "position_qty": context.position_qty,
                        "strategy_decision_trace": decision_trace,
                    },
                )
                return
            price = float(bar_row.get("close") or target_call.get("mark_price") or 0.0)
            if not np.isfinite(price) or price <= 0.0:
                raise ValueError(f"Invalid live signal price for {symbol}: {price}")
            self.refresh_context_account_snapshot(context)
            signal_ts = bar_ts.isoformat()
            for action, side, position_after in plan:
                context_payload = context.as_payload()
                payload = deployment_signal_payload(
                    context_payload,
                    action=action,
                    side=side,
                    target_percent=target,
                    bar_ts=bar_ts,
                    bar_index=int(source_bar_index),
                    price=price,
                    bars=strategy_bars,
                )
                payload["strategy_decision_trace"] = decision_trace
                signal_event_id = str(payload.get("event_id") or "")
                if self.catalog.deployment_runner_event_exists(f"signal_sent:{signal_event_id}"):
                    self.emit(
                        "signal_duplicate_skipped",
                        deployment_id=context.deployment_id,
                        context_id=context.context_id,
                        symbol=symbol,
                        message=f"Skipped duplicate {action} {side} for {symbol} at {signal_ts}.",
                        payload=payload,
                    )
                    continue
                self.emit(
                    "signal_intent",
                    deployment_id=context.deployment_id,
                    context_id=context.context_id,
                    symbol=symbol,
                    message=f"{symbol} live deployment intends {action} {side}.",
                    payload=payload,
                    event_id=f"signal_intent:{signal_event_id}",
                )
                response = post_deployment_webhook_payload(str(context.webhook_url or ""), payload)
                sizing_summary = deployment_signal_sizing_summary(payload)
                context.position_qty = float(position_after)
                context.avg_price = price if float(position_after) != 0.0 else 0.0
                context.last_signal_bar_ts_ns = bar_ts_ns
                self.catalog.update_deployment_status(
                    context.deployment_id,
                    status="live",
                    status_reason=f"Last signal {action} {side} sent at {signal_ts}. {sizing_summary}",
                    last_signal_at=signal_ts,
                )
                if context.parent_deployment_id:
                    self.catalog.update_deployment_status(
                        context.parent_deployment_id,
                        status="live",
                        status_reason=f"Last signal {symbol} {action} {side} sent at {signal_ts}. {sizing_summary}",
                        last_signal_at=signal_ts,
                    )
                self.emit(
                    "signal_sent",
                    deployment_id=context.deployment_id,
                    context_id=context.context_id,
                    symbol=symbol,
                    message=f"{symbol} live deployment sent {action} {side}. {sizing_summary}",
                    payload={**payload, "webhook_response": response},
                    event_id=f"signal_sent:{signal_event_id}",
                )
            self.contexts[context.context_id] = context
        except Exception as exc:
            self.mark_context_error(context, str(exc))

    def refresh_context_account_snapshot(self, context: DeploymentContext) -> None:
        root_deployment_id = context.parent_deployment_id or context.deployment_id
        if root_deployment_id:
            try:
                with sqlite3.connect(self.catalog.db_path) as conn:
                    row = conn.execute(
                        """
                        SELECT health_json
                        FROM deployment_metric_snapshots
                        WHERE deployment_id=?
                        ORDER BY snapshot_ts DESC
                        LIMIT 1
                        """,
                        (root_deployment_id,),
                    ).fetchone()
            except Exception:
                row = None
            if row and row[0]:
                try:
                    health = json.loads(str(row[0]))
                except Exception:
                    health = {}
                account = dict(health.get("account") or {})
                if account:
                    context.account_snapshot = account

    def mark_context_error(self, context: DeploymentContext, message: str) -> None:
        timestamp = pd.Timestamp.now(tz="UTC").isoformat()
        if context.deployment_id:
            self.catalog.update_deployment_status(
                context.deployment_id,
                status="error",
                status_reason=str(message)[:500],
                last_error_at=timestamp,
            )
        if context.parent_deployment_id:
            self.catalog.update_deployment_status(
                context.parent_deployment_id,
                status="error",
                status_reason=str(message)[:500],
                last_error_at=timestamp,
            )
        self.contexts.pop(context.context_id, None)
        indexed = self.symbol_index.get(context.symbol)
        if indexed is not None:
            indexed.discard(context.context_id)
            if not indexed:
                self.symbol_index.pop(context.symbol, None)
                self.stop_stream(context.symbol)
        self.emit(
            "context_error",
            deployment_id=context.deployment_id,
            context_id=context.context_id,
            symbol=context.symbol,
            severity="error",
            message=f"{context.symbol} live deployment error: {message}",
        )


def duckdb_interval_for_chart_timeframe(timeframe: str) -> str:
    normalized = normalize_chart_timeframe(timeframe)
    try:
        if normalized.endswith("m"):
            value = max(1, int(normalized[:-1] or "1"))
            unit = "minute" if value == 1 else "minutes"
            return f"{value} {unit}"
        if normalized.endswith("h"):
            value = max(1, int(normalized[:-1] or "1"))
            unit = "hour" if value == 1 else "hours"
            return f"{value} {unit}"
        if normalized.endswith("d"):
            value = max(1, int(normalized[:-1] or "1"))
            unit = "day" if value == 1 else "days"
            return f"{value} {unit}"
    except Exception:
        pass
    return "1 minute"


def live_deployment_strategy_warmup_days(params: dict, timeframe: str) -> int:
    if not isinstance(params, dict):
        return 5
    bars_per_day = 1.0
    normalized = normalize_chart_timeframe(timeframe)
    try:
        if normalized.endswith("m"):
            minutes = max(1, int(normalized[:-1] or "1"))
            bars_per_day = max(1.0, (6.5 * 60.0) / float(minutes))
        elif normalized.endswith("h"):
            hours = max(1, int(normalized[:-1] or "1"))
            bars_per_day = max(1.0, 6.5 / float(hours))
    except Exception:
        bars_per_day = 1.0
    warmup_bars = 0
    hints = ("lookback", "window", "period", "length", "len", "slow", "fast")
    for key, value in params.items():
        name = str(key or "").strip().lower()
        if not name or not any(token in name for token in hints):
            continue
        try:
            parsed = int(float(value))
        except Exception:
            continue
        if 0 < parsed <= 10_000:
            warmup_bars = max(warmup_bars, parsed)
    if warmup_bars <= 0:
        return 5
    return max(5, int(np.ceil(warmup_bars / max(bars_per_day, 1.0))) + 5)


def live_deployment_history_lookback_days(context: dict) -> int:
    sizing = dict(context.get("sizing") or {})
    execution_config = deployment_execution_config_payload(sizing.get("execution_config"))
    params = dict(context.get("params") or {})
    timeframe = normalize_chart_timeframe(str(context.get("timeframe") or LIVE_BAR_TIMEFRAME))
    strategy_days = live_deployment_strategy_warmup_days(params, timeframe)
    required = strategy_days
    model = normalize_position_sizing_model(execution_config.get("position_sizing_model"))
    if model == POSITION_SIZING_ANNUAL_VOLATILITY:
        window_days = max(
            int(execution_config.get("annual_vol_window", 252) or 252),
            int(execution_config.get("annual_vol_min_periods", 20) or 20),
        )
        required = max(required, int(np.ceil(window_days * 1.6)) + 7)
    return max(5, required)


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (str(table_name or "").strip(),),
    ).fetchone()
    return row is not None


def fetch_target_external_snapshot(target_row: dict) -> dict:
    transport_mode = str(target_row.get("transport_mode") or "").strip().lower()
    if transport_mode == "co_located":
        snapshot = read_external_snapshot_from_sqlite(target_row)
        if snapshot:
            return snapshot
    return read_external_snapshot_from_http(target_row)


def read_external_snapshot_from_sqlite(target_row: dict) -> dict:
    db_path = Path(str(target_row.get("db_path") or "").strip())
    if not db_path.exists():
        return {}
    normalized_scope = str(target_row.get("broker_scope") or target_row.get("mode") or "").strip().lower()
    positions_table = "live_positions"
    account_query = "SELECT equity, cash, buying_power, updated_ts FROM account_snapshots ORDER BY id DESC LIMIT 1"
    if normalized_scope in {"alpaca", "alpaca_paper"}:
        positions_table = "alpaca_positions"
    elif normalized_scope in {"paper"}:
        positions_table = "positions"
        account_query = "SELECT equity, cash, updated_ts FROM account WHERE id=1"
    elif normalized_scope in {"coinbase"}:
        positions_table = "coinbase_positions"
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            positions = (
                [dict(row) for row in conn.execute(f"SELECT * FROM {positions_table} ORDER BY symbol ASC").fetchall()]
                if table_exists(conn, positions_table)
                else []
            )
            if normalized_scope == "paper":
                account_row = conn.execute(account_query).fetchone() if table_exists(conn, "account") else None
            else:
                account_row = conn.execute(account_query).fetchone() if table_exists(conn, "account_snapshots") else None
    except Exception:
        return {}
    latest_ts = 0
    for row in positions:
        latest_ts = max(latest_ts, int(row.get("updated_ts") or 0))
    if account_row is not None:
        try:
            latest_ts = max(latest_ts, int(account_row["updated_ts"] or 0))
        except Exception:
            pass
    snapshot_ts = (
        pd.Timestamp.utcfromtimestamp(int(latest_ts)).isoformat() + "+00:00"
        if latest_ts
        else pd.Timestamp.now(tz="UTC").isoformat()
    )
    return {
        "source": "sqlite",
        "snapshot_ts": snapshot_ts,
        "orders": [],
        "fills": [],
        "recent_trades": [],
        "positions": positions,
        "account": dict(account_row) if account_row is not None else {},
        "equity_curve": [],
    }


def external_endpoint_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}{path if str(path).startswith('/') else '/' + str(path)}"


def external_dashboard_data_path(target_row: dict) -> str:
    status_path = str(target_row.get("status_path") or "").strip()
    dashboard_path = str(target_row.get("dashboard_path") or "").strip()
    broker_scope = str(target_row.get("broker_scope") or target_row.get("mode") or "").strip().lower()
    if status_path == "/live_status" or broker_scope in {"public", "live"}:
        return "/live_dashboard_data"
    if status_path == "/coinbase_status" or broker_scope == "coinbase":
        return "/coinbase_dashboard_data"
    if status_path == "/status" or broker_scope == "paper":
        return "/dashboard_data"
    if status_path.endswith("_status"):
        return status_path[: -len("_status")] + "_dashboard_data"
    if dashboard_path in {"/live", "live"}:
        return "/live_dashboard_data"
    if dashboard_path in {"/coinbase", "coinbase"}:
        return "/coinbase_dashboard_data"
    return "/dashboard_data"


def read_external_json_url(url: str) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def read_external_snapshot_from_http(target_row: dict) -> dict:
    base_url = str(target_row.get("base_url") or "").rstrip("/")
    status_path = str(target_row.get("status_path") or "").strip()
    if not base_url:
        return {}
    payload = read_external_json_url(external_endpoint_url(base_url, status_path)) if status_path else {}
    dashboard_payload = read_external_json_url(external_endpoint_url(base_url, external_dashboard_data_path(target_row)))
    if not payload and not dashboard_payload:
        return {}
    merged = dict(payload)
    for key in ("account", "orders", "fills", "recent_trades", "positions", "equity_curve"):
        if key in dashboard_payload:
            merged[key] = dashboard_payload.get(key)
    return {
        "source": "http",
        "snapshot_ts": pd.Timestamp.now(tz="UTC").isoformat(),
        "orders": list(merged.get("orders") or []),
        "fills": list(merged.get("fills") or []),
        "recent_trades": list(merged.get("recent_trades") or []),
        "positions": list(merged.get("positions") or []),
        "account": dict(merged.get("account") or {}),
        "equity_curve": list(merged.get("equity_curve") or []),
    }


def acquire_runner_lock(catalog_path: Path):
    try:
        import fcntl
    except Exception:
        return None
    lock_root = Path(os.environ.get("TMPDIR") or "/tmp")
    digest = hashlib.sha1(str(catalog_path.resolve()).encode("utf-8")).hexdigest()[:16]
    lock_name = f"quant_live_runner_{digest}.lock"
    lock_file = open(lock_root / lock_name, "w", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_file.close()
        return None
    lock_file.write(str(os.getpid()))
    lock_file.flush()
    return lock_file


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the quant backtest live deployment runner process.")
    parser.add_argument("--catalog", required=True, help="Path to the dashboard/catalog SQLite database.")
    parser.add_argument("--live-store", required=True, help="Path to the live-market SQLite database.")
    parser.add_argument("--poll-interval", type=float, default=0.25)
    parser.add_argument("--once", action="store_true", help="Process queued commands once and exit.")
    parser.add_argument("--no-streams", action="store_true", help="Disable IB streams; useful for tests.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    catalog_path = Path(args.catalog)
    lock_file = acquire_runner_lock(catalog_path)
    if lock_file is None and not args.once:
        return 0
    config = DeploymentRunnerConfig(
        catalog_path=catalog_path,
        live_store_path=Path(args.live_store),
        poll_interval_seconds=float(args.poll_interval),
        run_once=bool(args.once),
        streams_enabled=not bool(args.no_streams),
    )
    service = LiveDeploymentRunnerService(config)

    def _request_stop(_signum, _frame) -> None:
        service.stop_requested = True

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    service.run_forever()
    if lock_file is not None:
        lock_file.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
