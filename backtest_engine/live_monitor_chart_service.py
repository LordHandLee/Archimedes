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
from dataclasses import dataclass
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
    DEFAULT_LIVE_PROVIDER,
    LIVE_BAR_TIMEFRAME,
    LiveMarketDataStore,
    chart_timeframe_label,
    compute_chart_indicators,
    incremental_series_payload,
    latest_point_series_payload,
    normalize_chart_timeframe,
    resample_ohlcv,
)
from .magellan import MagellanClient


LIVE_MONITOR_MIN_SEED_BARS = 2
MAGELLAN_LIVE_UPDATE_TIMEOUT_MS = 100
SERVICE_PREVIEW_INTERVAL_SECONDS = 5.0
DATASET_ERROR_STATUSES = {"error", "failed", "missing", "corrupt"}


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
    last_preview_update_monotonic: float = 0.0


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
    ) -> tuple[dict[str, pd.Series], dict[str, pd.Series], dict[str, dict]]:
        overlays, panes, styles = compute_chart_indicators(bars, indicator_ids)
        try:
            strategy_overlays, strategy_panes, strategy_styles = ChartSnapshotExporter._build_portfolio_asset_strategy_series(
                bars,
                strategy_contexts,
            )
        except Exception:
            return overlays, panes, styles
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
        bars, note = self.build_bars(selected, symbol, timeframe, lookback)
        if bars.empty:
            raise ValueError(note)
        completed_bars = bars.iloc[:-1].copy() if len(bars) > 1 else bars.copy()
        if len(completed_bars) < LIVE_MONITOR_MIN_SEED_BARS:
            raise ValueError(
                f"Waiting for at least {LIVE_MONITOR_MIN_SEED_BARS} completed {chart_timeframe_label(timeframe)} bars."
            )
        strategy_contexts = self.strategy_contexts_for_symbol(selected, symbol)
        overlays, panes, styles = self.indicator_series(bars, indicator_ids, strategy_contexts)
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
        latest_ts = pd.Timestamp(bars.index[-1]).tz_convert("UTC")
        latest_ts_ns = int(latest_ts.value)
        now = time.monotonic()
        if latest_ts_ns <= int(session.last_sent_bar_ts_ns) and now - session.last_preview_update_monotonic < SERVICE_PREVIEW_INTERVAL_SECONDS:
            return
        session.bars = bars.copy()
        live_index = int(len(bars) - 1)
        latest_bar = bars.iloc[-1]
        overlays, panes, styles = self.indicator_series(bars, session.indicator_ids, session.strategy_contexts)
        self.magellan.send_live_update(
            session.session_id,
            title=f"{session.symbol} {chart_timeframe_label(session.timeframe)}",
            status_text=f"{session.symbol} live chart updated.",
            bars=[
                {
                    "timestamp_utc_ns": str(latest_ts_ns),
                    "bar_index": live_index,
                    "open": float(latest_bar.get("open") or 0.0),
                    "high": float(latest_bar.get("high") or 0.0),
                    "low": float(latest_bar.get("low") or 0.0),
                    "close": float(latest_bar.get("close") or 0.0),
                    "volume": float(latest_bar.get("volume") or 0.0),
                }
            ],
            overlay_series=incremental_series_payload(overlays, bar_index=live_index, timestamp=latest_ts, styles=styles),
            pane_series=incremental_series_payload(panes, bar_index=live_index, timestamp=latest_ts, styles=styles),
            equity_series=latest_point_series_payload({"equity": session.equity_curve}, bar_index=live_index, timestamp=latest_ts),
            timeout_ms=MAGELLAN_LIVE_UPDATE_TIMEOUT_MS,
        )
        session.last_sent_bar_ts_ns = latest_ts_ns
        session.last_sent_bar_index = live_index
        session.last_preview_update_monotonic = now

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
        side = str(payload.get("side") or "").strip().upper()
        marker_side = "buy" if side == "LONG" and action == "ENTRY" else "sell"
        override = dict(payload.get("position_size_override") or {})
        qty = override.get("target_qty")
        if qty in (None, ""):
            try:
                qty = float(override.get("target_notional") or 0.0) / float(payload.get("price") or 0.0)
            except Exception:
                qty = 0.0
        return {
            "timestamp_utc_ns": str(int(timestamp.value)),
            "bar_index": bar_index,
            "side": marker_side,
            "qty": float(qty or 0.0),
            "price": float(payload.get("price") or 0.0),
            "event": action.lower() or "signal",
            "event_type": action.lower() or "signal",
            "label": f"{action} {side}".strip(),
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
    catalog_path = Path(args.catalog)
    lock_file = acquire_service_lock(catalog_path)
    if lock_file is None and not args.once:
        return 0
    service = LiveMonitorChartService(
        LiveChartServiceConfig(
            catalog_path=catalog_path,
            live_store_path=Path(args.live_store),
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
