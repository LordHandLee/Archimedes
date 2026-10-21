from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sqlite3
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from PyQt6 import QtCore

from .catalog import ResultCatalog
from .chart_snapshot import ChartSnapshotExporter
from .duckdb_store import DuckDBStore
from .live_deployment_runner import (
    _decode_json_dict,
    fetch_target_external_snapshot,
    market_symbol_from_dataset_id,
)
from .live_market import (
    DEFAULT_STALE_QUOTE_SECONDS,
    DEFAULT_LIVE_PROVIDER,
    LIVE_BAR_TIMEFRAME,
    InteractiveBrokersRealtimeConfig,
    LiveMarketBar,
    LiveMarketDataStore,
    chart_timeframe_delta,
    chart_timeframe_label,
    compute_chart_indicators,
    normalize_chart_timeframe,
    resample_ohlcv,
    is_us_equity_total_session_open,
)
from .magellan import MagellanClient
from .provider_config import load_provider_settings


LIVE_MONITOR_MIN_SEED_BARS = 2
MAGELLAN_LIVE_UPDATE_TIMEOUT_MS = 100
SERVICE_PREVIEW_INTERVAL_SECONDS = 5.0
SERVICE_STALE_WARNING_INTERVAL_SECONDS = 60.0
SERVICE_HISTORICAL_SYNC_MAX_LOOKBACK = pd.Timedelta(days=5)
SERVICE_HISTORICAL_SYNC_CLIENT_OFFSET = 8500
DATASET_ERROR_STATUSES = {"error", "failed", "missing", "corrupt"}
LIVE_MONITOR_CHART_OPEN_STATUS = "live"


@dataclass
class LiveChartServiceConfig:
    catalog_path: Path
    live_store_path: Path
    poll_interval_seconds: float = 0.5
    command_batch_size: int = 25
    run_once: bool = False
    magellan_enabled: bool = True


@dataclass
class LiveChartSession:
    session_id: str
    deployment_id: str
    target_id: str
    symbol: str
    timeframe: str
    lookback: str
    indicator_ids: list[str]
    selected: dict
    bars: pd.DataFrame
    equity_curve: pd.Series
    strategy_contexts: list[tuple[str | None, str, dict]]
    local_trade_markers: list[dict]
    last_sent_bar_ts_ns: int
    last_sent_bar_index: int
    last_completed_replacement_bar_ts_ns: int
    bar_index_by_ts_ns: dict[int, int] = field(default_factory=dict)
    next_live_bar_index: int = 0
    last_preview_update_monotonic: float = 0.0
    last_stale_warning_monotonic: float = 0.0


def _charts_lookback_start(end_ts: pd.Timestamp, lookback: str) -> pd.Timestamp:
    value = str(lookback or "3mo").strip().lower()
    if value == "all":
        return pd.Timestamp("1970-01-01", tz="UTC")
    if value.endswith("d"):
        return end_ts - pd.Timedelta(days=max(1, int(value[:-1] or "1")))
    if value.endswith("mo"):
        return end_ts - pd.DateOffset(months=max(1, int(value[:-2] or "1")))
    if value.endswith("y"):
        return end_ts - pd.DateOffset(years=max(1, int(value[:-1] or "1")))
    return end_ts - pd.DateOffset(months=3)


def _normalize_bars(bars: object) -> pd.DataFrame:
    columns = ["open", "high", "low", "close", "volume"]
    if not isinstance(bars, pd.DataFrame) or bars.empty:
        return pd.DataFrame(columns=columns)
    frame = bars.copy()
    for column in columns:
        if column not in frame.columns:
            frame[column] = 0.0
    index = pd.to_datetime(frame.index, utc=True, errors="coerce")
    valid = ~pd.isna(index)
    if not bool(np.asarray(valid).any()):
        return pd.DataFrame(columns=columns)
    frame = frame.loc[valid, columns].copy()
    frame.index = pd.DatetimeIndex(index[valid]).tz_convert("UTC")
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=["open", "high", "low", "close"])
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    return frame[columns].astype(float)


def _equity_curve_series_from_points(points) -> pd.Series:
    rows: list[tuple[pd.Timestamp, float]] = []
    for point in list(points or []):
        if not isinstance(point, dict):
            continue
        raw_ts = (
            point.get("ts")
            or point.get("updated_ts")
            or point.get("timestamp")
            or point.get("timestamp_utc")
            or point.get("time")
        )
        raw_equity = point.get("equity")
        if raw_equity is None:
            raw_equity = point.get("value")
        try:
            equity = float(raw_equity)
        except Exception:
            continue
        if not np.isfinite(equity):
            continue
        try:
            numeric_ts = float(raw_ts)
        except Exception:
            numeric_ts = None
        if numeric_ts is not None and np.isfinite(numeric_ts):
            if numeric_ts > 1_000_000_000_000_000:
                timestamp = pd.to_datetime(int(numeric_ts), unit="ns", utc=True, errors="coerce")
            elif numeric_ts > 1_000_000_000_000:
                timestamp = pd.to_datetime(numeric_ts, unit="ms", utc=True, errors="coerce")
            else:
                timestamp = pd.to_datetime(numeric_ts, unit="s", utc=True, errors="coerce")
        else:
            timestamp = pd.to_datetime(raw_ts, utc=True, errors="coerce")
        if pd.isna(timestamp):
            continue
        rows.append((pd.Timestamp(timestamp).tz_convert("UTC"), equity))
    if not rows:
        return pd.Series(dtype=float, name="equity")
    series = pd.Series([value for _ts, value in rows], index=pd.DatetimeIndex([ts for ts, _value in rows]), name="equity")
    return series[~series.index.duplicated(keep="last")].sort_index().astype(float)


def _empty_trade_marker_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "seq",
            "ts_utc_ns",
            "bar_index",
            "side",
            "qty",
            "price",
            "fee",
            "realized_pnl",
            "equity_after",
            "event_type",
            "label",
        ]
    )


def _ensure_utc_timestamp(value: object) -> pd.Timestamp | None:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts).tz_convert("UTC")


def _extended_market_segments(
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    timezone: str = "America/New_York",
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    start_ts = _ensure_utc_timestamp(start)
    end_ts = _ensure_utc_timestamp(end)
    if start_ts is None or end_ts is None or end_ts <= start_ts:
        return []
    try:
        local_start = start_ts.tz_convert(timezone)
        local_end = end_ts.tz_convert(timezone)
    except Exception:
        return [(start_ts, end_ts)]
    segments: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    day = local_start.normalize()
    final_day = local_end.normalize()
    while day <= final_day:
        if day.dayofweek < 5:
            segment_start_local = day + pd.Timedelta(hours=4)
            segment_end_local = day + pd.Timedelta(hours=20)
            segment_start = max(start_ts, pd.Timestamp(segment_start_local).tz_convert("UTC"))
            segment_end = min(end_ts, pd.Timestamp(segment_end_local).tz_convert("UTC"))
            if segment_start < segment_end:
                segments.append((segment_start.floor("min"), segment_end.floor("min")))
        day += pd.Timedelta(days=1)
    return segments


def _minute_bars_cover_segment(
    bars: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    max_gap: pd.Timedelta = pd.Timedelta(minutes=2),
) -> bool:
    if bars is None or bars.empty:
        return False
    start_ts = _ensure_utc_timestamp(start)
    end_ts = _ensure_utc_timestamp(end)
    if start_ts is None or end_ts is None:
        return False
    start_ts = start_ts.floor("min")
    end_ts = end_ts.floor("min")
    if end_ts <= start_ts:
        return True
    normalized = _normalize_bars(bars)
    if normalized.empty:
        return False
    covered = normalized.loc[(normalized.index >= start_ts) & (normalized.index <= end_ts)]
    if covered.empty:
        return False
    stamps = pd.Index(covered.index).drop_duplicates().sort_values()
    if pd.Timestamp(stamps[0]).tz_convert("UTC") > start_ts + max_gap:
        return False
    if pd.Timestamp(stamps[-1]).tz_convert("UTC") < end_ts - max_gap:
        return False
    if len(stamps) <= 1:
        return (end_ts - start_ts) <= max_gap
    gaps = pd.Series(stamps).diff().dropna()
    if gaps.empty:
        return True
    return bool(gaps.max() <= max_gap)


def _minute_bars_cover_window(
    bars: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    max_gap: pd.Timedelta = pd.Timedelta(minutes=2),
) -> bool:
    segments = _extended_market_segments(start, end)
    if not segments:
        return True
    return all(
        _minute_bars_cover_segment(bars, segment_start, segment_end, max_gap=max_gap)
        for segment_start, segment_end in segments
    )


def _chart_sync_coverage_gap(timeframe: str) -> pd.Timedelta:
    delta = chart_timeframe_delta(timeframe)
    normalized = normalize_chart_timeframe(timeframe)
    if normalized.endswith("d"):
        return delta * 2
    return max(delta * 2, pd.Timedelta(minutes=2))


def _live_monitor_chart_block_reason(deployment_row: dict) -> str:
    status = str(deployment_row.get("status") or "draft").strip().lower() or "draft"
    if status == LIVE_MONITOR_CHART_OPEN_STATUS:
        return ""
    deployment_id = str(deployment_row.get("deployment_id") or "").strip()
    prefix = f"Deployment {deployment_id[:10]} is" if deployment_id else "Deployment is"
    if status == "armed":
        return f"{prefix} armed but not live yet. Wait for the runner to report live before opening Live Monitor charts."
    if status == "paused":
        return f"{prefix} paused. Resume or arm it until it is live before opening Live Monitor charts."
    if status == "stopped":
        return f"{prefix} stopped. Arm it and wait for live status before opening Live Monitor charts."
    if status in {"error", "failed"}:
        return f"{prefix} in error. Fix and arm it until it is live before opening Live Monitor charts."
    return f"{prefix} {status}. It must be live before opening Live Monitor charts."


def _completed_bars_for_timeframe(
    bars: pd.DataFrame,
    timeframe: str,
    *,
    now: object | None = None,
) -> pd.DataFrame:
    normalized = _normalize_bars(bars)
    if normalized.empty:
        return normalized
    now_ts = _ensure_utc_timestamp(now or pd.Timestamp.now(tz="UTC"))
    if now_ts is None:
        return normalized.iloc[:-1].copy()
    delta = chart_timeframe_delta(timeframe)
    completed_mask = normalized.index + delta <= now_ts
    return normalized.loc[completed_mask].copy()


def _bar_index_map_from_bars(bars: pd.DataFrame) -> dict[int, int]:
    normalized = _normalize_bars(bars)
    mapping: dict[int, int] = {}
    for index, timestamp in enumerate(normalized.index):
        mapping[int(pd.Timestamp(timestamp).tz_convert("UTC").value)] = int(index)
    return mapping


def _extend_strategy_series_to_live_chart_bars(
    series_map: dict[str, pd.Series],
    bars: pd.DataFrame,
) -> dict[str, pd.Series]:
    normalized = _normalize_bars(bars)
    if normalized.empty or not series_map:
        return series_map
    chart_index = normalized.index
    extended: dict[str, pd.Series] = {}
    for name, series in series_map.items():
        if series is None or series.empty:
            extended[name] = series
            continue
        values = pd.to_numeric(series, errors="coerce")
        if not isinstance(values.index, pd.DatetimeIndex):
            extended[name] = values
            continue
        utc_index = pd.to_datetime(values.index, utc=True, errors="coerce")
        valid_index = ~pd.isna(utc_index)
        if not bool(valid_index.any()):
            extended[name] = values
            continue
        mask = np.asarray(valid_index, dtype=bool)
        values = pd.Series(
            values.to_numpy(dtype=float)[mask],
            index=pd.DatetimeIndex(utc_index[mask]),
            name=getattr(series, "name", None),
        ).sort_index()
        values = values[~values.index.duplicated(keep="last")]
        aligned = values.reindex(chart_index)
        finite_values = values[np.isfinite(values.to_numpy(dtype=float))]
        if not finite_values.empty:
            last_completed_ts = pd.Timestamp(values.index[-1]).tz_convert("UTC")
            aligned.loc[aligned.index > last_completed_ts] = float(finite_values.iloc[-1])
        extended[name] = aligned
    return extended


def _series_update_payload_for_entries(
    series_map: dict[str, pd.Series],
    update_entries: Sequence[tuple[int, int, pd.Timestamp, pd.Series]],
    *,
    styles: dict[str, dict] | None = None,
) -> list[dict]:
    payload: list[dict] = []
    if not update_entries:
        return payload
    styles = styles or {}
    entry_ts_ns = [int(pd.Timestamp(timestamp).tz_convert("UTC").value) for _source_index, _bar_index, timestamp, _row in update_entries]
    for name, series in series_map.items():
        if series is None or series.empty:
            continue
        values = pd.to_numeric(series, errors="coerce")
        values_by_ts_ns: dict[int, float] = {}
        use_timestamp_index = isinstance(values.index, pd.DatetimeIndex)
        if use_timestamp_index:
            index = pd.DatetimeIndex(pd.to_datetime(values.index, utc=True, errors="coerce"))
            for raw_ts, raw_value in zip(index, values.to_numpy(), strict=False):
                if pd.isna(raw_ts):
                    continue
                try:
                    value_float = float(raw_value)
                except Exception:
                    continue
                if np.isfinite(value_float):
                    values_by_ts_ns[int(pd.Timestamp(raw_ts).tz_convert("UTC").value)] = value_float
        points: list[dict] = []
        for entry_idx, (source_index, bar_index, timestamp, _row) in enumerate(update_entries):
            value_float: float | None = None
            if use_timestamp_index:
                value_float = values_by_ts_ns.get(entry_ts_ns[entry_idx])
            elif 0 <= int(source_index) < len(values):
                try:
                    value_float = float(values.iloc[int(source_index)])
                except Exception:
                    value_float = None
            if value_float is None or not np.isfinite(value_float):
                continue
            points.append(
                {
                    "timestamp_utc_ns": str(int(pd.Timestamp(timestamp).tz_convert("UTC").value)),
                    "bar_index": int(bar_index),
                    "value": value_float,
                }
            )
        if not points:
            continue
        style = styles.get(name, {})
        item = {"name": name, "points": points}
        color = str(style.get("color") or "").strip()
        if color:
            item["color"] = color
        payload.append(item)
    return payload


def _latest_value_series_payload_for_entries(
    series_map: dict[str, pd.Series],
    update_entries: Sequence[tuple[int, int, pd.Timestamp, pd.Series]],
    *,
    styles: dict[str, dict] | None = None,
) -> list[dict]:
    payload: list[dict] = []
    if not update_entries:
        return payload
    styles = styles or {}
    for name, series in series_map.items():
        if series is None or series.empty:
            continue
        aligned = pd.to_numeric(series, errors="coerce").dropna()
        if aligned.empty:
            continue
        try:
            value_float = float(aligned.iloc[-1])
        except Exception:
            continue
        if not np.isfinite(value_float):
            continue
        points = [
            {
                "timestamp_utc_ns": str(int(pd.Timestamp(timestamp).tz_convert("UTC").value)),
                "bar_index": int(bar_index),
                "value": value_float,
            }
            for _source_index, bar_index, timestamp, _row in update_entries
        ]
        style = styles.get(name, {})
        item = {"name": name, "points": points}
        color = str(style.get("color") or "").strip()
        if color:
            item["color"] = color
        payload.append(item)
    return payload


class LiveMonitorChartService:
    def __init__(self, config: LiveChartServiceConfig) -> None:
        self.config = config
        self.catalog = ResultCatalog(config.catalog_path)
        self.store = LiveMarketDataStore(config.live_store_path)
        self.exporter = ChartSnapshotExporter(root_dir=Path("data/live_chart_snapshots"))
        self.service_id = f"live-chart:{os.getpid()}:{uuid.uuid4().hex[:8]}"
        self.sessions: dict[str, LiveChartSession] = {}
        self.stop_requested = False
        self.magellan = MagellanClient() if config.magellan_enabled else None
        self.last_runner_event_seq = self._latest_runner_event_seq()

    def emit(
        self,
        event_type: str,
        *,
        session_id: str = "",
        deployment_id: str = "",
        symbol: str = "",
        severity: str = "info",
        message: str = "",
        payload: dict | None = None,
        event_id: str = "",
    ) -> None:
        self.catalog.save_live_chart_event(
            event_type=event_type,
            session_id=session_id,
            deployment_id=deployment_id,
            symbol=symbol,
            severity=severity,
            message=message,
            payload_json=payload or {},
            event_id=event_id,
        )

    def _latest_runner_event_seq(self) -> int:
        try:
            with sqlite3.connect(self.catalog.db_path) as conn:
                row = conn.execute("SELECT COALESCE(MAX(event_seq), 0) FROM deployment_runner_events").fetchone()
        except Exception:
            return 0
        return int(row[0] or 0) if row else 0

    def run_forever(self) -> None:
        self.emit("service_started", message=f"Live monitor chart service started ({self.service_id}).")
        while not self.stop_requested:
            self.process_commands()
            self.poll_sessions()
            self.consume_runner_events()
            if self.config.run_once:
                break
            time.sleep(max(0.05, float(self.config.poll_interval_seconds)))
        if self.magellan is not None:
            self.magellan.shutdown()
        self.emit("service_stopped", message=f"Live monitor chart service stopped ({self.service_id}).")

    def process_commands(self) -> None:
        commands = self.catalog.claim_live_chart_commands(service_id=self.service_id, limit=self.config.command_batch_size)
        for command in commands:
            error = ""
            try:
                payload = _decode_json_dict(command.payload_json)
                command_type = str(command.command_type or "").strip().lower()
                if command_type == "open_chart":
                    self.open_chart(payload)
                elif command_type == "open_deployment_charts":
                    self.open_deployment_charts(payload)
                elif command_type == "close_chart":
                    self.close_chart(str(command.session_id or payload.get("session_id") or ""))
                elif command_type == "close_deployment_charts":
                    self.close_deployment_charts(str(command.deployment_id or payload.get("deployment_id") or ""))
                elif command_type == "close_all_charts":
                    self.close_all_charts()
                elif command_type == "warmup":
                    self.warmup()
                elif command_type == "shutdown_service":
                    self.close_all_charts()
                    self.stop_requested = True
                else:
                    raise ValueError(f"Unsupported live chart command: {command.command_type}")
            except Exception as exc:
                error = str(exc)
                self.emit(
                    "chart_command_error",
                    session_id=str(command.session_id or ""),
                    deployment_id=str(command.deployment_id or ""),
                    severity="error",
                    message=error,
                    payload={"command_id": command.command_id, "command_type": command.command_type},
                )
            finally:
                self.catalog.finish_live_chart_command(command.command_id, error=error)

    def warmup(self) -> None:
        if self.magellan is not None:
            self.magellan.warmup_async()
        self.emit(
            "service_warmed",
            payload={
                "service_id": self.service_id,
                "magellan_enabled": self.magellan is not None,
            },
        )

    def deployment_row(self, deployment_id: str) -> dict:
        rows = {row.deployment_id: row for row in self.catalog.load_deployments()}
        row = rows.get(str(deployment_id or ""))
        return row.__dict__ if row is not None else {}

    def target_row(self, target_id: str) -> dict:
        rows = {row.target_id: row for row in self.catalog.load_deployment_targets()}
        row = rows.get(str(target_id or ""))
        return row.__dict__ if row is not None else {}

    @staticmethod
    def ensure_session_index_state(session: LiveChartSession) -> None:
        if not isinstance(session.bar_index_by_ts_ns, dict) or not session.bar_index_by_ts_ns:
            session.bar_index_by_ts_ns = _bar_index_map_from_bars(session.bars)
        else:
            session.bar_index_by_ts_ns = {
                int(key): int(value)
                for key, value in session.bar_index_by_ts_ns.items()
                if str(key).strip() and str(value).strip()
            }
        next_index = max(session.bar_index_by_ts_ns.values(), default=-1) + 1
        try:
            configured_next = int(session.next_live_bar_index)
        except Exception:
            configured_next = 0
        session.next_live_bar_index = max(next_index, configured_next)

    @staticmethod
    def session_bar_index_for_timestamp(session: LiveChartSession, timestamp: object) -> int:
        LiveMonitorChartService.ensure_session_index_state(session)
        ts = _ensure_utc_timestamp(timestamp)
        if ts is None:
            return -1
        key = int(ts.value)
        existing = session.bar_index_by_ts_ns.get(key)
        if existing is not None:
            return int(existing)
        next_index = max(
            int(session.next_live_bar_index or 0),
            max(session.bar_index_by_ts_ns.values(), default=-1) + 1,
        )
        session.bar_index_by_ts_ns[key] = int(next_index)
        session.next_live_bar_index = int(next_index) + 1
        return int(next_index)

    def strategy_contexts_for_symbol(self, deployment_row: dict, symbol: str) -> list[tuple[str | None, str, dict]]:
        cleaned = market_symbol_from_dataset_id(symbol)
        if not cleaned:
            return []
        structure = _decode_json_dict(deployment_row.get("structure_json"))
        params = _decode_json_dict(deployment_row.get("params_json"))
        deployment_kind = str(deployment_row.get("deployment_kind") or "").strip()
        strategy = str(deployment_row.get("strategy") or "").strip()
        contexts: list[tuple[str | None, str, dict]] = []
        if deployment_kind == "portfolio_strategy_blocks":
            for block in list(structure.get("strategy_blocks") or []):
                datasets = [str(item).strip() for item in list(block.get("asset_dataset_ids") or []) if str(item).strip()]
                if datasets and cleaned not in {market_symbol_from_dataset_id(item) for item in datasets}:
                    continue
                block_strategy = str(block.get("strategy_name") or block.get("strategy") or strategy or "").strip()
                if not block_strategy:
                    continue
                label = str(block.get("display_name") or block.get("block_id") or block_strategy or "").strip()
                contexts.append((label or None, block_strategy, dict(block.get("strategy_params") or block.get("params") or {})))
            return contexts
        if strategy:
            contexts.append((None, strategy, params))
        return contexts

    def deployment_symbol_scope(self, deployment_row: dict) -> set[str]:
        symbols: set[str] = set()

        def add_symbol(value: object) -> None:
            resolved = market_symbol_from_dataset_id(value)
            if resolved:
                symbols.add(resolved)

        add_symbol(deployment_row.get("symbol"))
        add_symbol(deployment_row.get("dataset_id"))
        structure = _decode_json_dict(deployment_row.get("structure_json"))
        for dataset_id in list(structure.get("portfolio_dataset_ids") or []):
            add_symbol(dataset_id)
        for block in list(structure.get("strategy_blocks") or []):
            for dataset_id in list(block.get("asset_dataset_ids") or []):
                add_symbol(dataset_id)
            for asset in list(block.get("assets") or []):
                add_symbol(asset.get("symbol") or asset.get("dataset_id"))
        deployment_id = str(deployment_row.get("deployment_id") or "")
        if deployment_id:
            try:
                child_links = self.catalog.load_deployment_child_links(deployment_id)
            except Exception:
                child_links = []
            for link in child_links:
                add_symbol(getattr(link, "symbol", "") or getattr(link, "dataset_id", ""))
        return symbols

    def open_deployment_charts(self, payload: dict) -> None:
        deployment_id = str(payload.get("deployment_id") or "").strip()
        if not deployment_id:
            raise ValueError("deployment_id is required for deployment chart open.")
        selected = self.deployment_row(deployment_id)
        if not selected:
            raise ValueError(f"Deployment {deployment_id} was not found.")
        block_reason = _live_monitor_chart_block_reason(selected)
        if block_reason:
            self.emit(
                "deployment_chart_open_rejected",
                deployment_id=deployment_id,
                severity="warning",
                message=block_reason,
                payload={
                    "deployment_id": deployment_id,
                    "status": str(selected.get("status") or "draft"),
                },
            )
            raise ValueError(block_reason)
        timeframe = normalize_chart_timeframe(str(payload.get("timeframe") or selected.get("timeframe") or LIVE_BAR_TIMEFRAME))
        lookback = str(payload.get("lookback") or "3mo")
        indicator_ids = list(payload.get("indicator_ids") or [])
        if bool(payload.get("replace_existing", True)):
            self.close_deployment_charts(deployment_id)
        symbols = sorted(self.deployment_symbol_scope(selected))
        try:
            max_symbols = int(payload.get("max_symbols") or 0)
        except Exception:
            max_symbols = 0
        if max_symbols > 0:
            symbols = symbols[:max_symbols]
        if not symbols:
            raise ValueError(f"Deployment {deployment_id} does not expose any ticker symbols.")
        failures: list[str] = []
        opened = 0
        for symbol in symbols:
            try:
                self.open_chart(
                    {
                        "session_id": f"live-monitor:{deployment_id}:{symbol}",
                        "deployment_id": deployment_id,
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "lookback": lookback,
                        "indicator_ids": list(indicator_ids),
                    }
                )
                opened += 1
            except Exception as exc:
                failures.append(f"{symbol}: {exc}")
        self.emit(
            "deployment_charts_queued",
            deployment_id=deployment_id,
            message=f"Processed {opened} Live Monitor chart open(s) for deployment {deployment_id[:10]}.",
            payload={
                "deployment_id": deployment_id,
                "symbols": symbols,
                "opened": opened,
                "failures": failures,
                "timeframe": timeframe,
                "lookback": lookback,
            },
            severity="warning" if failures else "info",
        )
        if opened <= 0 and failures:
            raise ValueError("; ".join(failures[:3]))

    def historical_dataset_id_for_symbol(self, deployment_row: dict, symbol: str) -> str:
        cleaned = market_symbol_from_dataset_id(symbol)
        row_dataset_id = str(deployment_row.get("dataset_id") or "").strip()
        if row_dataset_id and market_symbol_from_dataset_id(row_dataset_id) == cleaned:
            return row_dataset_id
        candidates: list[tuple[int, pd.Timestamp, str]] = []
        try:
            records = self.catalog.load_acquisition_datasets()
        except Exception:
            return row_dataset_id if row_dataset_id and market_symbol_from_dataset_id(row_dataset_id) == cleaned else ""
        duck = DuckDBStore()
        for record in records:
            if market_symbol_from_dataset_id(getattr(record, "symbol", "")) != cleaned:
                continue
            resolution = str(getattr(record, "resolution", "") or "").strip().lower().replace(" ", "")
            if resolution not in {"1m", "1min", "1minute", "1minutes"}:
                continue
            if not bool(getattr(record, "ingested", False)):
                continue
            if str(getattr(record, "last_status", "") or "").strip().lower() in DATASET_ERROR_STATUSES:
                continue
            dataset_id = str(getattr(record, "dataset_id", "") or "").strip()
            if not dataset_id or not duck.dataset_path(dataset_id).exists():
                continue
            source_rank = {
                "interactive_brokers": 0,
                "massive": 1,
                "stooq": 2,
            }.get(str(getattr(record, "source", "") or "").strip().lower(), 9)
            end_ts = pd.to_datetime(getattr(record, "coverage_end", None), utc=True, errors="coerce")
            end_sort = pd.Timestamp.min.tz_localize("UTC") if pd.isna(end_ts) else pd.Timestamp(end_ts).tz_convert("UTC")
            candidates.append((source_rank, end_sort, dataset_id))
        if not candidates:
            return (
                row_dataset_id
                if row_dataset_id
                and market_symbol_from_dataset_id(row_dataset_id) == cleaned
                and DuckDBStore().dataset_path(row_dataset_id).exists()
                else ""
            )
        candidates.sort(key=lambda item: (item[0], -item[1].value))
        return candidates[0][2]

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

    def local_minute_bars_for_sync(
        self,
        deployment_row: dict,
        symbol: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        cleaned = market_symbol_from_dataset_id(symbol)
        frames: list[pd.DataFrame] = []
        dataset_id = self.historical_dataset_id_for_symbol(deployment_row, cleaned)
        if dataset_id:
            try:
                historical = DuckDBStore().load_range(dataset_id, start, end)
                if historical is not None and not historical.empty:
                    frames.append(historical)
            except Exception:
                pass
        try:
            live = self.store.load_recent_bars(cleaned, provider=DEFAULT_LIVE_PROVIDER, limit=500000)
            if live is not None and not live.empty:
                frames.append(live.loc[(live.index >= start) & (live.index <= end)])
        except sqlite3.Error:
            pass
        if not frames:
            return pd.DataFrame()
        return _normalize_bars(pd.concat(frames).sort_index())

    def live_chart_sync_window(
        self,
        deployment_row: dict,
        symbol: str,
        lookback: str,
        *,
        now: pd.Timestamp | None = None,
    ) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        cleaned = market_symbol_from_dataset_id(symbol)
        if not cleaned:
            return None
        now_ts = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
        now_ts = now_ts.tz_convert("UTC") if now_ts.tzinfo else now_ts.tz_localize("UTC")
        end = now_ts.floor("min") - pd.Timedelta(minutes=1)
        if end <= pd.Timestamp("1970-01-01", tz="UTC"):
            return None
        dataset_id = self.historical_dataset_id_for_symbol(deployment_row, cleaned)
        historical_latest: pd.Timestamp | None = None
        if dataset_id:
            try:
                dataset = DuckDBStore().load_range(dataset_id, end - SERVICE_HISTORICAL_SYNC_MAX_LOOKBACK, end)
                if dataset is not None and not dataset.empty:
                    historical_latest = pd.Timestamp(dataset.index[-1]).tz_convert("UTC")
            except Exception:
                pass
        if historical_latest is not None:
            start = max(
                historical_latest.floor("min") + pd.Timedelta(minutes=1),
                end - SERVICE_HISTORICAL_SYNC_MAX_LOOKBACK,
            )
        else:
            lookback_start = _charts_lookback_start(end, lookback)
            start = max(pd.Timestamp(lookback_start).tz_convert("UTC"), end - SERVICE_HISTORICAL_SYNC_MAX_LOOKBACK)
        start = pd.Timestamp(start).tz_convert("UTC").floor("min")
        if start >= end - pd.Timedelta(minutes=2):
            return None
        local = self.local_minute_bars_for_sync(deployment_row, cleaned, start, end)
        timeframe = normalize_chart_timeframe(str(deployment_row.get("timeframe") or LIVE_BAR_TIMEFRAME))
        if _minute_bars_cover_window(local, start, end, max_gap=_chart_sync_coverage_gap(timeframe)):
            return None
        return start, end

    def sync_live_store_gap_if_needed(
        self,
        deployment_row: dict,
        symbol: str,
        lookback: str,
        *,
        session_id: str,
        deployment_id: str,
    ) -> int:
        cleaned = market_symbol_from_dataset_id(symbol)
        window = self.live_chart_sync_window(deployment_row, cleaned, lookback)
        if window is None:
            return 0
        start, end = window
        self.emit(
            "chart_sync_started",
            session_id=session_id,
            deployment_id=deployment_id,
            symbol=cleaned,
            message=f"Syncing {cleaned} Live Monitor gap from IBKR before chart refresh.",
            payload={"symbol": cleaned, "start": start.isoformat(), "end": end.isoformat()},
        )
        config = self.build_ib_config(client_offset=SERVICE_HISTORICAL_SYNC_CLIENT_OFFSET + (abs(hash(cleaned)) % 400))
        from scripts.fetch_interactive_brokers import fetch_bars

        frames: list[pd.DataFrame] = []
        for segment_index, (segment_start, segment_end) in enumerate(_extended_market_segments(start, end)):
            segment_frame = fetch_bars(
                cleaned,
                start=segment_start,
                end=segment_end,
                history_window="1d",
                resolution=LIVE_BAR_TIMEFRAME,
                host=config.host,
                port=int(config.port),
                client_id=int(config.client_id) + int(segment_index),
                sec_type=config.sec_type,
                exchange=config.exchange,
                currency=config.currency,
                primary_exchange=config.primary_exchange or None,
                what_to_show=config.what_to_show,
                use_rth=bool(config.use_rth),
                chunk_duration="1d",
                pace_seconds=0.5,
                timeout_seconds=max(30.0, float(config.timeout_seconds)),
                discover_head_timestamp=False,
            )
            segment_frame = _normalize_bars(segment_frame)
            if not segment_frame.empty:
                segment_frame = segment_frame.loc[
                    (segment_frame.index >= segment_start) & (segment_frame.index <= segment_end)
                ]
                if not segment_frame.empty:
                    frames.append(segment_frame)
            time.sleep(0.25)
        frame = _normalize_bars(pd.concat(frames).sort_index()) if frames else pd.DataFrame()
        rows = 0
        batch: list[LiveMarketBar] = []
        received_at = pd.Timestamp.now(tz="UTC")
        for timestamp, row in _normalize_bars(frame).iterrows():
            batch.append(
                LiveMarketBar(
                    symbol=cleaned,
                    timestamp=pd.Timestamp(timestamp).tz_convert("UTC"),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row.get("volume", 0.0)),
                    provider=DEFAULT_LIVE_PROVIDER,
                    timeframe=LIVE_BAR_TIMEFRAME,
                    received_at=received_at,
                )
            )
            if len(batch) >= 500:
                rows += self.store.upsert_bars(batch)
                batch.clear()
        if batch:
            rows += self.store.upsert_bars(batch)
        self.emit(
            "chart_sync_finished",
            session_id=session_id,
            deployment_id=deployment_id,
            symbol=cleaned,
            message=f"Synced {rows} {cleaned} IBKR bar(s) into the live store before chart refresh.",
            payload={
                "symbol": cleaned,
                "rows": rows,
                "start": start.isoformat(),
                "end": end.isoformat(),
            },
        )
        return rows

    def build_bars(self, deployment_row: dict, symbol: str, timeframe: str, lookback: str) -> tuple[pd.DataFrame, str]:
        cleaned = market_symbol_from_dataset_id(symbol)
        frames: list[pd.DataFrame] = []
        dataset_id = self.historical_dataset_id_for_symbol(deployment_row, cleaned)
        end_ts = pd.Timestamp.now(tz="UTC")
        start_ts = _charts_lookback_start(end_ts, lookback)
        if dataset_id:
            try:
                duck = DuckDBStore()
                if str(lookback or "3mo") == "all":
                    historical = duck.load(dataset_id)
                else:
                    historical = duck.load_range(dataset_id, start_ts, end_ts)
                if historical is not None and not historical.empty:
                    frames.append(historical)
            except Exception:
                pass
        try:
            live = self.store.load_recent_bars(cleaned, provider=DEFAULT_LIVE_PROVIDER, limit=500000)
        except sqlite3.Error:
            live = pd.DataFrame()
        if live is not None and not live.empty:
            frames.append(live)
        if not frames:
            return pd.DataFrame(), f"No local historical or live bars are available for {cleaned} yet."
        raw = pd.concat(frames).sort_index()
        if str(lookback or "3mo") != "all":
            raw = raw.loc[(raw.index >= start_ts) & (raw.index <= end_ts)]
        bars = resample_ohlcv(raw, timeframe)
        bars = bars[~bars.index.duplicated(keep="last")]
        return _normalize_bars(bars), "ok"

    def indicator_series(
        self,
        bars: pd.DataFrame,
        indicator_ids: Sequence[str],
        strategy_contexts: Sequence[tuple[str | None, str, dict]],
        *,
        timeframe: str | None = None,
    ) -> tuple[dict[str, pd.Series], dict[str, pd.Series], dict[str, dict]]:
        overlays, panes, styles = compute_chart_indicators(bars, indicator_ids)
        strategy_bars = _completed_bars_for_timeframe(bars, timeframe or LIVE_BAR_TIMEFRAME)
        if strategy_bars.empty:
            return overlays, panes, styles
        try:
            strategy_overlays, strategy_panes, strategy_styles = ChartSnapshotExporter._build_portfolio_asset_strategy_series(
                strategy_bars,
                strategy_contexts,
            )
        except Exception:
            return overlays, panes, styles
        strategy_overlays = _extend_strategy_series_to_live_chart_bars(strategy_overlays, bars)
        strategy_panes = _extend_strategy_series_to_live_chart_bars(strategy_panes, bars)
        overlays.update(strategy_overlays)
        panes.update(strategy_panes)
        styles.update(strategy_styles)
        return overlays, panes, styles

    def open_chart(self, payload: dict) -> None:
        deployment_id = str(payload.get("deployment_id") or "").strip()
        symbol = market_symbol_from_dataset_id(payload.get("symbol"))
        if not deployment_id or not symbol:
            raise ValueError("deployment_id and symbol are required for live chart open.")
        timeframe = normalize_chart_timeframe(str(payload.get("timeframe") or LIVE_BAR_TIMEFRAME))
        lookback = str(payload.get("lookback") or "3mo")
        indicator_ids = list(payload.get("indicator_ids") or [])
        session_id = str(payload.get("session_id") or f"live-monitor:{deployment_id}:{symbol}")
        selected = self.deployment_row(deployment_id)
        if not selected:
            raise ValueError(f"Deployment {deployment_id} was not found.")
        block_reason = _live_monitor_chart_block_reason(selected)
        if block_reason:
            self.emit(
                "chart_open_rejected",
                session_id=session_id,
                deployment_id=deployment_id,
                symbol=symbol,
                severity="warning",
                message=block_reason,
                payload={
                    "session_id": session_id,
                    "deployment_id": deployment_id,
                    "symbol": symbol,
                    "status": str(selected.get("status") or "draft"),
                },
            )
            raise ValueError(block_reason)
        target = self.target_row(str(selected.get("target_id") or ""))
        snapshot = fetch_target_external_snapshot(target) if target else {}
        equity_curve = _equity_curve_series_from_points(snapshot.get("equity_curve"))
        self.emit(
            "chart_opening",
            session_id=session_id,
            deployment_id=deployment_id,
            symbol=symbol,
            message=f"Opening {symbol} Live Monitor chart.",
            payload={
                "session_id": session_id,
                "deployment_id": deployment_id,
                "symbol": symbol,
                "timeframe": timeframe,
                "lookback": lookback,
            },
        )
        snapshot_root = Path("data/live_chart_snapshots") / "deployments" / deployment_id / symbol / timeframe / lookback
        manifest = snapshot_root / "manifest.json"
        opened_cached = False
        if self.magellan is not None and manifest.exists():
            title = f"{symbol} {chart_timeframe_label(timeframe)}"
            self.magellan.open_live_session(
                session_id,
                title=title,
                subtitle=f"Deployment {deployment_id[:10]} live monitor",
                status_text="Opened cached chart snapshot while the chart service refreshes live data.",
                snapshot_path=snapshot_root,
                timeout_ms=1000,
            )
            self.emit(
                "chart_opened_cached",
                session_id=session_id,
                deployment_id=deployment_id,
                symbol=symbol,
                message=f"Opened cached {symbol} Live Monitor chart.",
                payload={
                    "session_id": session_id,
                    "deployment_id": deployment_id,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "lookback": lookback,
                    "snapshot_path": str(snapshot_root.resolve()),
                },
            )
            opened_cached = True
        if bool(payload.get("sync_before_open", True)):
            try:
                rows = self.sync_live_store_gap_if_needed(
                    selected,
                    symbol,
                    lookback,
                    session_id=session_id,
                    deployment_id=deployment_id,
                )
                if rows > 0 and self.magellan is not None and opened_cached:
                    self.magellan.send_live_update(
                        session_id,
                        title=f"{symbol} {chart_timeframe_label(timeframe)}",
                        status_text=f"Synced {rows} IBKR bar(s); refreshing chart.",
                        timeout_ms=MAGELLAN_LIVE_UPDATE_TIMEOUT_MS,
                    )
            except Exception as exc:
                self.emit(
                    "chart_sync_error",
                    session_id=session_id,
                    deployment_id=deployment_id,
                    symbol=symbol,
                    severity="warning",
                    message=f"{symbol} Live Monitor gap sync failed; opening best available chart: {exc}",
                    payload={"session_id": session_id, "symbol": symbol, "error": str(exc)},
                )
        bars, note = self.build_bars(selected, symbol, timeframe, lookback)
        if bars.empty:
            raise ValueError(note)
        completed_bars = bars.iloc[:-1].copy() if len(bars) > 1 else bars.copy()
        if len(completed_bars) < LIVE_MONITOR_MIN_SEED_BARS:
            raise ValueError(
                f"Waiting for at least {LIVE_MONITOR_MIN_SEED_BARS} completed {chart_timeframe_label(timeframe)} bars."
            )
        strategy_contexts = self.strategy_contexts_for_symbol(selected, symbol)
        overlays, panes, styles = self.indicator_series(
            bars,
            indicator_ids,
            strategy_contexts,
            timeframe=timeframe,
        )
        artifact = self.exporter.export_market_snapshot(
            symbol=symbol,
            timeframe=timeframe,
            bars=bars,
            overlays=overlays,
            panes=panes,
            series_styles=styles,
            equity_curve=equity_curve if not equity_curve.empty else None,
            trades_df=_empty_trade_marker_frame(),
            snapshot_root=snapshot_root,
            title=f"{symbol} {chart_timeframe_label(timeframe)}",
            subtitle=f"Deployment {deployment_id[:10]} live monitor",
            status_text="Deployment chart built by the live monitor chart service.",
            overwrite=True,
        )
        title = f"{symbol} {chart_timeframe_label(timeframe)}"
        if self.magellan is not None:
            if opened_cached:
                self.magellan.reload_live_seed(
                    session_id,
                    title=title,
                    subtitle=f"Deployment {deployment_id[:10]} live monitor",
                    status_text="Deployment chart refreshed by the live monitor chart service.",
                    snapshot_path=artifact.snapshot_root,
                    timeout_ms=5000,
                )
            else:
                self.magellan.open_live_session(
                    session_id,
                    title=title,
                    subtitle=f"Deployment {deployment_id[:10]} live monitor",
                    status_text="Deployment chart built by the live monitor chart service.",
                    snapshot_path=artifact.snapshot_root,
                    timeout_ms=5000,
                )
        self.sessions[session_id] = LiveChartSession(
            session_id=session_id,
            deployment_id=deployment_id,
            target_id=str(selected.get("target_id") or ""),
            symbol=symbol,
            timeframe=timeframe,
            lookback=lookback,
            indicator_ids=indicator_ids,
            selected=dict(selected),
            bars=bars.copy(),
            equity_curve=equity_curve,
            strategy_contexts=list(strategy_contexts),
            local_trade_markers=[],
            last_sent_bar_ts_ns=int(pd.Timestamp(bars.index[-1]).tz_convert("UTC").value),
            last_sent_bar_index=int(len(bars) - 1),
            last_completed_replacement_bar_ts_ns=int(pd.Timestamp(completed_bars.index[-1]).tz_convert("UTC").value),
            bar_index_by_ts_ns=_bar_index_map_from_bars(bars),
            next_live_bar_index=int(len(bars)),
        )
        self.emit(
            "chart_opened",
            session_id=session_id,
            deployment_id=deployment_id,
            symbol=symbol,
            message=f"Opened {symbol} Live Monitor chart.",
            payload={
                "session_id": session_id,
                "deployment_id": deployment_id,
                "symbol": symbol,
                "timeframe": timeframe,
                "lookback": lookback,
                "snapshot_path": str(artifact.snapshot_root),
                "bar_count": int(len(bars)),
            },
        )

    def close_chart(self, session_id: str) -> None:
        session = self.sessions.pop(str(session_id or ""), None)
        if self.magellan is not None and session_id:
            try:
                self.magellan.close_session(session_id, timeout_ms=300)
            except Exception:
                pass
        self.emit(
            "chart_closed",
            session_id=str(session_id or ""),
            deployment_id=session.deployment_id if session else "",
            symbol=session.symbol if session else "",
            message=f"Closed chart {session_id}.",
        )

    def close_deployment_charts(self, deployment_id: str) -> None:
        for session_id, session in list(self.sessions.items()):
            if session.deployment_id == str(deployment_id or ""):
                self.close_chart(session_id)

    def close_all_charts(self) -> None:
        for session_id in list(self.sessions):
            self.close_chart(session_id)

    def poll_sessions(self) -> None:
        for session in list(self.sessions.values()):
            try:
                self.poll_session(session)
            except Exception as exc:
                self.emit(
                    "chart_update_error",
                    session_id=session.session_id,
                    deployment_id=session.deployment_id,
                    symbol=session.symbol,
                    severity="error",
                    message=str(exc),
                )

    def poll_session(self, session: LiveChartSession) -> None:
        if self.magellan is None:
            return
        bars, _note = self.build_bars(session.selected, session.symbol, session.timeframe, session.lookback)
        if bars.empty or len(bars) < LIVE_MONITOR_MIN_SEED_BARS:
            return
        self.ensure_session_index_state(session)
        latest_ts = pd.Timestamp(bars.index[-1]).tz_convert("UTC")
        latest_ts_ns = int(latest_ts.value)
        now = time.monotonic()
        self._emit_stale_warning_if_needed(session, latest_ts, now)
        if latest_ts_ns <= int(session.last_sent_bar_ts_ns) and now - session.last_preview_update_monotonic < SERVICE_PREVIEW_INTERVAL_SECONDS:
            return
        update_entries: list[tuple[int, int, pd.Timestamp, pd.Series]] = []
        last_sent_ts_ns = int(session.last_sent_bar_ts_ns or 0)
        for source_index, (timestamp, row) in enumerate(bars.iterrows()):
            timestamp = pd.Timestamp(timestamp).tz_convert("UTC")
            ts_ns = int(timestamp.value)
            if ts_ns <= last_sent_ts_ns:
                continue
            bar_index = self.session_bar_index_for_timestamp(session, timestamp)
            if bar_index < 0:
                continue
            update_entries.append((int(source_index), int(bar_index), timestamp, row))
        if not update_entries:
            if latest_ts_ns != last_sent_ts_ns:
                session.bars = bars.copy()
                return
            latest_bar_index = self.session_bar_index_for_timestamp(session, latest_ts)
            if latest_bar_index < 0:
                return
            update_entries.append((int(len(bars) - 1), int(latest_bar_index), latest_ts, bars.iloc[-1]))
        overlays, panes, styles = self.indicator_series(
            bars,
            session.indicator_ids,
            session.strategy_contexts,
            timeframe=session.timeframe,
        )
        bar_payload = [
            {
                "timestamp_utc_ns": str(int(timestamp.value)),
                "bar_index": int(bar_index),
                "open": float(row.get("open") or 0.0),
                "high": float(row.get("high") or 0.0),
                "low": float(row.get("low") or 0.0),
                "close": float(row.get("close") or 0.0),
                "volume": float(row.get("volume") or 0.0),
            }
            for _source_index, bar_index, timestamp, row in update_entries
        ]
        self.magellan.send_live_update(
            session.session_id,
            title=f"{session.symbol} {chart_timeframe_label(session.timeframe)}",
            status_text=f"{session.symbol} live chart updated.",
            bars=bar_payload,
            overlay_series=_series_update_payload_for_entries(overlays, update_entries, styles=styles),
            pane_series=_series_update_payload_for_entries(panes, update_entries, styles=styles),
            equity_series=_latest_value_series_payload_for_entries({"equity": session.equity_curve}, update_entries),
            timeout_ms=MAGELLAN_LIVE_UPDATE_TIMEOUT_MS,
        )
        session.bars = bars.copy()
        _source_index, last_bar_index, last_timestamp, _row = update_entries[-1]
        session.last_sent_bar_ts_ns = int(pd.Timestamp(last_timestamp).tz_convert("UTC").value)
        session.last_sent_bar_index = int(last_bar_index)
        session.last_preview_update_monotonic = now

    def _emit_stale_warning_if_needed(
        self,
        session: LiveChartSession,
        latest_ts: pd.Timestamp,
        now_monotonic: float,
    ) -> None:
        if not is_us_equity_total_session_open():
            return
        if now_monotonic - session.last_stale_warning_monotonic < SERVICE_STALE_WARNING_INTERVAL_SECONDS:
            return
        now_utc = pd.Timestamp.now(tz="UTC")
        stale_after = max(
            chart_timeframe_delta(session.timeframe) + pd.Timedelta(seconds=float(DEFAULT_STALE_QUOTE_SECONDS)),
            pd.Timedelta(seconds=float(DEFAULT_STALE_QUOTE_SECONDS) * 2.0),
        )
        stale_for = now_utc - pd.Timestamp(latest_ts).tz_convert("UTC")
        if stale_for < stale_after:
            return
        session.last_stale_warning_monotonic = now_monotonic
        seconds = max(0, int(stale_for.total_seconds()))
        message = f"{session.symbol} live chart data is stale; latest bar is {seconds}s old."
        self.emit(
            "chart_data_stale",
            session_id=session.session_id,
            deployment_id=session.deployment_id,
            symbol=session.symbol,
            severity="warning",
            message=message,
            payload={
                "session_id": session.session_id,
                "symbol": session.symbol,
                "timeframe": session.timeframe,
                "latest_bar_utc": pd.Timestamp(latest_ts).tz_convert("UTC").isoformat(),
                "stale_for_seconds": seconds,
                "stale_after_seconds": int(stale_after.total_seconds()),
            },
        )
        if self.magellan is not None:
            self.magellan.send_live_update(
                session.session_id,
                title=f"{session.symbol} {chart_timeframe_label(session.timeframe)}",
                status_text=message,
                timeout_ms=MAGELLAN_LIVE_UPDATE_TIMEOUT_MS,
            )

    def consume_runner_events(self) -> None:
        events = self.catalog.load_deployment_runner_events(after_seq=int(self.last_runner_event_seq or 0), limit=500)
        if not events:
            return
        for event in events:
            self.last_runner_event_seq = max(int(self.last_runner_event_seq or 0), int(event.event_seq))
            if str(event.event_type or "") != "signal_sent":
                continue
            payload = _decode_json_dict(event.payload_json)
            symbol = market_symbol_from_dataset_id(payload.get("symbol") or event.symbol)
            deployment_ids = {
                str(payload.get("deployment_id") or ""),
                str(payload.get("parent_deployment_id") or ""),
                str(payload.get("portfolio_id") or ""),
            }
            deployment_ids.discard("")
            for session in list(self.sessions.values()):
                if session.symbol != symbol:
                    continue
                if deployment_ids and session.deployment_id not in deployment_ids:
                    continue
                marker = self.signal_trade_marker(payload, session.bars)
                if marker:
                    try:
                        marker_ts_ns = int(marker.get("timestamp_utc_ns") or 0)
                    except Exception:
                        marker_ts_ns = 0
                    if marker_ts_ns > 0:
                        stable_bar_index = self.session_bar_index_for_timestamp(
                            session,
                            pd.to_datetime(marker_ts_ns, unit="ns", utc=True, errors="coerce"),
                        )
                        if stable_bar_index >= 0:
                            marker["bar_index"] = int(stable_bar_index)
                    self.send_trade_marker(session, marker)

    @staticmethod
    def signal_trade_marker(payload: dict, bars: pd.DataFrame) -> dict | None:
        normalized = _normalize_bars(bars)
        if normalized.empty:
            return None
        ts = pd.to_datetime(payload.get("bar_timestamp_utc") or payload.get("time"), utc=True, errors="coerce")
        if pd.isna(ts):
            return None
        timestamp = pd.Timestamp(ts).tz_convert("UTC")
        try:
            bar_index = int(normalized.index.get_indexer([timestamp], method="nearest")[0])
        except Exception:
            return None
        action = str(payload.get("action") or "").strip().upper()
        position_side = str(payload.get("side") or "").strip().upper()
        if action == "ENTRY":
            marker_side = "buy"
            event_type = "entry"
            marker_color = "#22c55e"
        elif action == "EXIT":
            marker_side = "sell"
            event_type = "exit"
            marker_color = "#ef4444"
        else:
            marker_side = "sell" if position_side == "SHORT" else "buy"
            event_type = action.lower() or "signal"
            marker_color = "#22c55e" if marker_side == "buy" else "#ef4444"
        override = dict(payload.get("position_size_override") or {})
        qty = override.get("target_qty")
        if qty in (None, ""):
            try:
                qty = float(override.get("target_notional") or 0.0) / float(payload.get("price") or 0.0)
            except Exception:
                qty = 0.0
        try:
            qty_abs = abs(float(qty or 0.0))
        except Exception:
            qty_abs = 0.0
        signed_qty = qty_abs if marker_side == "buy" else -qty_abs
        return {
            "timestamp_utc_ns": str(int(timestamp.value)),
            "bar_index": bar_index,
            "side": marker_side,
            "qty": signed_qty,
            "quantity": signed_qty,
            "price": float(payload.get("price") or 0.0),
            "event": event_type,
            "event_type": event_type,
            "action": action,
            "position_side": position_side,
            "color": marker_color,
            "label": f"{action} {position_side}".strip(),
        }

    def send_trade_marker(self, session: LiveChartSession, marker: dict) -> None:
        if self.magellan is None:
            return
        key = (marker.get("timestamp_utc_ns"), marker.get("bar_index"), marker.get("side"), marker.get("event"))
        existing = {
            (item.get("timestamp_utc_ns"), item.get("bar_index"), item.get("side"), item.get("event"))
            for item in session.local_trade_markers
            if isinstance(item, dict)
        }
        if key in existing:
            return
        session.local_trade_markers.append(marker)
        self.magellan.send_live_update(
            session.session_id,
            title=f"{session.symbol} {chart_timeframe_label(session.timeframe)}",
            status_text=f"{marker.get('label', 'Signal')} sent.",
            trade_markers=[marker],
            timeout_ms=MAGELLAN_LIVE_UPDATE_TIMEOUT_MS,
        )
        self.emit(
            "chart_trade_marker",
            session_id=session.session_id,
            deployment_id=session.deployment_id,
            symbol=session.symbol,
            message=f"Added {marker.get('label', 'signal')} marker to {session.symbol}.",
            payload=marker,
        )


def acquire_service_lock(catalog_path: Path):
    try:
        import fcntl
    except Exception:
        return None
    lock_root = Path(os.environ.get("TMPDIR") or "/tmp")
    digest = hashlib.sha1(str(catalog_path.resolve()).encode("utf-8")).hexdigest()[:16]
    lock_file = open(lock_root / f"quant_live_chart_{digest}.lock", "w", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_file.close()
        return None
    lock_file.write(str(os.getpid()))
    lock_file.flush()
    return lock_file


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the live monitor chart service process.")
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--live-store", required=True)
    parser.add_argument("--poll-interval", type=float, default=0.5)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--no-magellan", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])
    catalog_path = Path(args.catalog).expanduser().resolve()
    live_store_path = Path(args.live_store).expanduser().resolve()
    lock_file = acquire_service_lock(catalog_path)
    if lock_file is None and not args.once:
        return 0
    service = LiveMonitorChartService(
        LiveChartServiceConfig(
            catalog_path=catalog_path,
            live_store_path=live_store_path,
            poll_interval_seconds=float(args.poll_interval),
            run_once=bool(args.once),
            magellan_enabled=not bool(args.no_magellan),
        )
    )

    def _request_stop(_signum, _frame) -> None:
        service.stop_requested = True

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    service.run_forever()
    app.processEvents()
    if lock_file is not None:
        lock_file.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
