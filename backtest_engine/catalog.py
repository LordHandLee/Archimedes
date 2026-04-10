from __future__ import annotations

import json
import sqlite3
from hashlib import sha1
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import pandas as pd

from .metrics import PerformanceMetrics


_ACQUISITION_ERROR_STATUSES = {"download_error", "ingest_error", "failed", "gap_fill_error"}


@dataclass
class CachedRun:
    run_id: str
    logical_run_id: str | None
    batch_id: str | None
    strategy: str
    params: str
    timeframe: str
    start: str
    end: str
    dataset_id: str
    starting_cash: float | None
    metrics: PerformanceMetrics
    run_started_at: str
    run_finished_at: Optional[str]
    status: str
    requested_execution_mode: str | None
    resolved_execution_mode: str | None
    engine_impl: str | None
    engine_version: str | None
    fallback_reason: str | None


@dataclass
class BatchRecord:
    batch_id: str
    strategy: str
    dataset_id: str
    params: Dict
    timeframes: list[str]
    horizons: list[str]
    run_total: int | None
    status: str
    started_at: Optional[str]
    finished_at: Optional[str]


@dataclass
class BatchBenchmarkRecord:
    batch_id: str
    seq: int
    dataset_id: str
    strategy: str
    timeframe: str
    requested_execution_mode: str
    resolved_execution_mode: str
    engine_impl: str
    engine_version: str
    bars: int
    total_params: int
    cached_runs: int
    uncached_runs: int
    duration_seconds: float
    chunk_count: int
    chunk_sizes: tuple[int, ...]
    effective_param_batch_size: int | None
    prepared_context_reused: bool


@dataclass
class OptimizationStudyRecord:
    study_id: str
    batch_id: str
    strategy: str
    dataset_scope: tuple[str, ...]
    param_names: tuple[str, ...]
    timeframes: tuple[str, ...]
    horizons: tuple[str, ...]
    score_version: str
    aggregate_count: int
    created_at: str | None


@dataclass
class OptimizationCandidateRecord:
    candidate_id: str
    study_id: str
    timeframe: str
    start: str | None
    end: str | None
    param_key: str
    params_json: str
    source_type: str
    promotion_reason: str | None
    status: str
    metrics_json: str
    asset_results_json: str | None
    artifact_refs_json: str | None
    notes: str | None
    created_at: str | None
    updated_at: str | None


@dataclass
class WalkForwardStudyRecord:
    wf_study_id: str
    batch_id: str
    strategy: str
    dataset_id: str
    timeframe: str
    candidate_source_mode: str
    param_names: tuple[str, ...]
    schedule_json: str
    selection_rule: str
    params_json: str
    status: str
    description: str | None
    stitched_metrics_json: str | None
    stitched_equity_json: str | None
    fold_count: int
    created_at: str | None


@dataclass
class WalkForwardFoldRecord:
    wf_study_id: str
    fold_index: int
    train_study_id: str | None
    timeframe: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    selected_param_set_id: str | None
    selected_params_json: str
    train_rank: int | None
    train_robust_score: float | None
    test_run_id: str | None
    status: str


@dataclass
class WalkForwardFoldMetricsRecord:
    wf_study_id: str
    fold_index: int
    train_metrics_json: str
    test_metrics_json: str
    degradation_json: str
    param_drift_json: str


@dataclass
class MonteCarloStudyRecord:
    mc_study_id: str
    source_type: str
    source_id: str
    resampling_mode: str
    simulation_count: int
    seed: int | None
    cost_stress_json: str
    status: str
    description: str | None
    source_trade_count: int
    starting_equity: float
    summary_json: str
    fan_quantiles_json: str
    terminal_returns_json: str
    max_drawdowns_json: str
    terminal_equities_json: str
    original_path_json: str
    created_at: str | None


@dataclass
class MonteCarloPathRecord:
    mc_study_id: str
    path_id: str
    path_type: str
    path_json: str
    summary_json: str


@dataclass
class UniverseRecord:
    universe_id: str
    name: str
    description: str | None
    symbols_json: str
    dataset_ids_json: str
    source_preference: str | None
    created_at: str | None
    updated_at: str | None


@dataclass
class ProviderSettingsRecord:
    provider_id: str
    settings_json: str
    updated_at: str | None


@dataclass
class AcquisitionDatasetRecord:
    dataset_id: str
    source: str | None
    symbol: str | None
    resolution: str | None
    history_window: str | None
    csv_path: str | None
    parquet_path: str | None
    coverage_start: str | None
    coverage_end: str | None
    bar_count: int | None
    ingested: bool
    last_download_attempt_at: str | None
    last_download_success_at: str | None
    last_ingest_at: str | None
    last_status: str | None
    last_error: str | None
    last_run_id: str | None
    last_task_id: str | None
    created_at: str | None
    updated_at: str | None


@dataclass
class AcquisitionRunRecord:
    acquisition_run_id: str
    trigger_type: str
    source: str | None
    universe_id: str | None
    universe_name: str | None
    task_id: str | None
    started_at: str
    finished_at: str | None
    status: str
    symbol_count: int | None
    success_count: int
    failed_count: int
    ingested_count: int
    notes: str | None
    log_path: str | None
    created_at: str | None
    updated_at: str | None


@dataclass
class AcquisitionAttemptRecord:
    attempt_id: str
    acquisition_run_id: str
    seq: int
    source: str | None
    symbol: str | None
    dataset_id: str | None
    status: str
    started_at: str | None
    finished_at: str | None
    csv_path: str | None
    parquet_path: str | None
    coverage_start: str | None
    coverage_end: str | None
    bar_count: int | None
    ingested: bool
    error_message: str | None
    log_path: str | None
    task_id: str | None
    universe_id: str | None


@dataclass
class TaskRunRecord:
    run_id: str
    task_id: str
    started_at: str | None
    finished_at: str | None
    status: str
    ticker_count: int | None
    log_path: str | None
    error_message: str | None


class ResultCatalog:
    """Lightweight SQLite-backed cache for backtest runs."""

    def __init__(self, db_path: str | Path = "backtests.sqlite") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    logical_run_id TEXT,
                    batch_id TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    strategy TEXT NOT NULL,
                    params TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    start TEXT NOT NULL,
                    end TEXT NOT NULL,
                    dataset_id TEXT NOT NULL,
                    starting_cash REAL,
                    metrics TEXT NOT NULL,
                    run_started_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    run_finished_at TEXT,
                    status TEXT DEFAULT 'finished',
                    requested_execution_mode TEXT,
                    resolved_execution_mode TEXT,
                    engine_impl TEXT,
                    engine_version TEXT,
                    fallback_reason TEXT
                )
                """
            )
            # Backfill new columns if upgrading an existing DB.
            self._ensure_column(conn, "runs", "logical_run_id", "TEXT")
            self._ensure_column(conn, "runs", "batch_id", "TEXT")
            self._ensure_column(conn, "runs", "run_started_at", "TEXT")
            self._ensure_column(conn, "runs", "run_finished_at", "TEXT")
            self._ensure_column(conn, "runs", "status", "TEXT", default="'finished'")
            self._ensure_column(conn, "runs", "starting_cash", "REAL")
            self._ensure_column(conn, "runs", "requested_execution_mode", "TEXT")
            self._ensure_column(conn, "runs", "resolved_execution_mode", "TEXT")
            self._ensure_column(conn, "runs", "engine_impl", "TEXT")
            self._ensure_column(conn, "runs", "engine_version", "TEXT")
            self._ensure_column(conn, "runs", "fallback_reason", "TEXT")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS heatmaps (
                    heatmap_id TEXT PRIMARY KEY,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    description TEXT,
                    params TEXT NOT NULL,
                    file_path TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    run_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    side TEXT NOT NULL,
                    qty REAL NOT NULL,
                    price REAL NOT NULL,
                    fee REAL NOT NULL,
                    realized_pnl REAL NOT NULL,
                    equity_after REAL NOT NULL,
                    PRIMARY KEY (run_id, seq)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS batches (
                    batch_id TEXT PRIMARY KEY,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    strategy TEXT NOT NULL,
                    dataset_id TEXT NOT NULL,
                    params TEXT NOT NULL,
                    timeframes TEXT NOT NULL,
                    horizons TEXT NOT NULL,
                    run_total INTEGER,
                    status TEXT DEFAULT 'running',
                    started_at TEXT,
                    finished_at TEXT
                )
                """
            )
            self._ensure_column(conn, "batches", "started_at", "TEXT")
            self._ensure_column(conn, "batches", "finished_at", "TEXT")
            self._ensure_column(conn, "batches", "status", "TEXT", default="'running'")
            self._ensure_column(conn, "batches", "run_total", "INTEGER")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS batch_benchmarks (
                    batch_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    dataset_id TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    requested_execution_mode TEXT NOT NULL,
                    resolved_execution_mode TEXT NOT NULL,
                    engine_impl TEXT NOT NULL,
                    engine_version TEXT NOT NULL,
                    bars INTEGER NOT NULL,
                    total_params INTEGER NOT NULL,
                    cached_runs INTEGER NOT NULL,
                    uncached_runs INTEGER NOT NULL,
                    duration_seconds REAL NOT NULL,
                    chunk_count INTEGER DEFAULT 0,
                    chunk_sizes TEXT NOT NULL,
                    effective_param_batch_size INTEGER,
                    prepared_context_reused INTEGER DEFAULT 0,
                    PRIMARY KEY (batch_id, seq)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS scheduled_tasks (
                    task_id TEXT PRIMARY KEY,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    symbols TEXT NOT NULL,
                    schedule TEXT NOT NULL,
                    status TEXT DEFAULT 'active',
                    last_run_at TEXT,
                    last_run_status TEXT,
                    last_run_message TEXT,
                    next_run_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS universes (
                    universe_id TEXT PRIMARY KEY,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    name TEXT NOT NULL,
                    description TEXT,
                    symbols_json TEXT NOT NULL,
                    dataset_ids_json TEXT NOT NULL,
                    source_preference TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS provider_settings (
                    provider_id TEXT PRIMARY KEY,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    settings_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS acquisition_datasets (
                    dataset_id TEXT PRIMARY KEY,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    source TEXT,
                    symbol TEXT,
                    resolution TEXT,
                    history_window TEXT,
                    csv_path TEXT,
                    parquet_path TEXT,
                    coverage_start TEXT,
                    coverage_end TEXT,
                    bar_count INTEGER,
                    ingested INTEGER DEFAULT 0,
                    last_download_attempt_at TEXT,
                    last_download_success_at TEXT,
                    last_ingest_at TEXT,
                    last_status TEXT,
                    last_error TEXT,
                    last_run_id TEXT,
                    last_task_id TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS acquisition_runs (
                    acquisition_run_id TEXT PRIMARY KEY,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    trigger_type TEXT NOT NULL,
                    source TEXT,
                    universe_id TEXT,
                    universe_name TEXT,
                    task_id TEXT,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    status TEXT NOT NULL,
                    symbol_count INTEGER,
                    success_count INTEGER DEFAULT 0,
                    failed_count INTEGER DEFAULT 0,
                    ingested_count INTEGER DEFAULT 0,
                    notes TEXT,
                    log_path TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS acquisition_attempts (
                    attempt_id TEXT PRIMARY KEY,
                    acquisition_run_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    source TEXT,
                    symbol TEXT,
                    dataset_id TEXT,
                    status TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    csv_path TEXT,
                    parquet_path TEXT,
                    coverage_start TEXT,
                    coverage_end TEXT,
                    bar_count INTEGER,
                    ingested INTEGER DEFAULT 0,
                    error_message TEXT,
                    log_path TEXT,
                    task_id TEXT,
                    universe_id TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS optimization_studies (
                    study_id TEXT PRIMARY KEY,
                    batch_id TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    strategy TEXT NOT NULL,
                    dataset_scope TEXT NOT NULL,
                    param_names TEXT NOT NULL,
                    timeframes TEXT NOT NULL,
                    horizons TEXT NOT NULL,
                    score_version TEXT NOT NULL,
                    aggregate_count INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS optimization_aggregates (
                    study_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    timeframe TEXT NOT NULL,
                    start TEXT,
                    end TEXT,
                    param_key TEXT NOT NULL,
                    params_json TEXT NOT NULL,
                    dataset_count INTEGER NOT NULL,
                    run_count INTEGER NOT NULL,
                    median_total_return REAL NOT NULL,
                    median_cagr REAL,
                    median_sharpe REAL NOT NULL,
                    median_rolling_sharpe REAL,
                    worst_max_drawdown REAL NOT NULL,
                    sharpe_std REAL NOT NULL,
                    profitable_asset_ratio REAL NOT NULL,
                    robust_score REAL NOT NULL,
                    PRIMARY KEY (study_id, seq)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS optimization_asset_results (
                    study_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    timeframe TEXT NOT NULL,
                    start TEXT,
                    end TEXT,
                    dataset_id TEXT NOT NULL,
                    param_key TEXT NOT NULL,
                    params_json TEXT NOT NULL,
                    total_return REAL,
                    cagr REAL,
                    sharpe REAL,
                    rolling_sharpe REAL,
                    max_drawdown REAL,
                    run_id TEXT,
                    PRIMARY KEY (study_id, seq)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS optimization_candidates (
                    candidate_id TEXT PRIMARY KEY,
                    study_id TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    timeframe TEXT NOT NULL,
                    start TEXT,
                    end TEXT,
                    param_key TEXT NOT NULL,
                    params_json TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    promotion_reason TEXT,
                    status TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    asset_results_json TEXT,
                    artifact_refs_json TEXT,
                    notes TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS walk_forward_studies (
                    wf_study_id TEXT PRIMARY KEY,
                    batch_id TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    strategy TEXT NOT NULL,
                    dataset_id TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    candidate_source_mode TEXT NOT NULL,
                    param_names TEXT NOT NULL,
                    schedule_json TEXT NOT NULL,
                    selection_rule TEXT NOT NULL,
                    params_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    description TEXT,
                    stitched_metrics_json TEXT,
                    stitched_equity_json TEXT,
                    fold_count INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS walk_forward_folds (
                    wf_study_id TEXT NOT NULL,
                    fold_index INTEGER NOT NULL,
                    train_study_id TEXT,
                    timeframe TEXT NOT NULL,
                    train_start TEXT NOT NULL,
                    train_end TEXT NOT NULL,
                    test_start TEXT NOT NULL,
                    test_end TEXT NOT NULL,
                    selected_param_set_id TEXT,
                    selected_params_json TEXT NOT NULL,
                    train_rank INTEGER,
                    train_robust_score REAL,
                    test_run_id TEXT,
                    status TEXT NOT NULL,
                    PRIMARY KEY (wf_study_id, fold_index)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS walk_forward_fold_metrics (
                    wf_study_id TEXT NOT NULL,
                    fold_index INTEGER NOT NULL,
                    train_metrics_json TEXT NOT NULL,
                    test_metrics_json TEXT NOT NULL,
                    degradation_json TEXT NOT NULL,
                    param_drift_json TEXT NOT NULL,
                    PRIMARY KEY (wf_study_id, fold_index)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS monte_carlo_studies (
                    mc_study_id TEXT PRIMARY KEY,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    source_type TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    resampling_mode TEXT NOT NULL,
                    simulation_count INTEGER NOT NULL,
                    seed INTEGER,
                    cost_stress_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    description TEXT,
                    source_trade_count INTEGER NOT NULL,
                    starting_equity REAL NOT NULL,
                    summary_json TEXT NOT NULL,
                    fan_quantiles_json TEXT NOT NULL,
                    terminal_returns_json TEXT NOT NULL,
                    max_drawdowns_json TEXT NOT NULL,
                    terminal_equities_json TEXT NOT NULL,
                    original_path_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS monte_carlo_paths (
                    mc_study_id TEXT NOT NULL,
                    path_id TEXT NOT NULL,
                    path_type TEXT NOT NULL,
                    path_json TEXT NOT NULL,
                    summary_json TEXT NOT NULL,
                    PRIMARY KEY (mc_study_id, path_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS task_runs (
                    run_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    status TEXT NOT NULL,
                    ticker_count INTEGER,
                    log_path TEXT,
                    error_message TEXT
                )
                """
            )

    def _ensure_column(self, conn: sqlite3.Connection, table: str, column: str, col_type: str, default: str | None = None) -> None:
        cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
        if column not in cols:
            default_clause = f" DEFAULT {default}" if default else ""
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}{default_clause}")

    def fetch(self, run_id: str) -> Optional[CachedRun]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT
                    run_id,
                    logical_run_id,
                    batch_id,
                    strategy,
                    params,
                    timeframe,
                    start,
                    end,
                    dataset_id,
                    starting_cash,
                    metrics,
                    run_started_at,
                    run_finished_at,
                    status,
                    requested_execution_mode,
                    resolved_execution_mode,
                    engine_impl,
                    engine_version,
                    fallback_reason
                FROM runs WHERE run_id=?
                """,
                (run_id,),
            ).fetchone()
        if not row:
            return None
        return self._cached_run_from_row(row)

    def load_runs_for_logical_run_id(self, logical_run_id: str) -> list[CachedRun]:
        if not logical_run_id:
            return []
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT
                    run_id,
                    logical_run_id,
                    batch_id,
                    strategy,
                    params,
                    timeframe,
                    start,
                    end,
                    dataset_id,
                    starting_cash,
                    metrics,
                    run_started_at,
                    run_finished_at,
                    status,
                    requested_execution_mode,
                    resolved_execution_mode,
                    engine_impl,
                    engine_version,
                    fallback_reason
                FROM runs
                WHERE logical_run_id=?
                ORDER BY COALESCE(run_finished_at, run_started_at, created_at) DESC, run_id DESC
                """,
                (logical_run_id,),
            ).fetchall()
        return [self._cached_run_from_row(row) for row in rows]

    def _cached_run_from_row(self, row) -> CachedRun:
        metrics_json = json.loads(row[10]) if row[10] else {}
        if metrics_json:
            metrics = PerformanceMetrics(
                total_return=metrics_json.get("total_return", 0.0),
                cagr=metrics_json.get("cagr", 0.0),
                max_drawdown=metrics_json.get("max_drawdown", 0.0),
                sharpe=metrics_json.get("sharpe", 0.0),
                rolling_sharpe=metrics_json.get("rolling_sharpe", 0.0),
            )
        else:
            metrics = PerformanceMetrics(0, 0, 0, 0, 0)
        return CachedRun(
            run_id=row[0],
            logical_run_id=row[1],
            batch_id=row[2],
            strategy=row[3],
            params=row[4],
            timeframe=row[5],
            start=row[6],
            end=row[7],
            dataset_id=row[8],
            starting_cash=row[9],
            metrics=metrics,
            run_started_at=row[11],
            run_finished_at=row[12],
            status=row[13] or "finished",
            requested_execution_mode=row[14],
            resolved_execution_mode=row[15],
            engine_impl=row[16],
            engine_version=row[17],
            fallback_reason=row[18],
        )

    def fetch_batch(self, batch_id: str) -> Optional[BatchRecord]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT batch_id,strategy,dataset_id,params,timeframes,horizons,run_total,status,started_at,finished_at
                FROM batches WHERE batch_id=?
                """,
                (batch_id,),
            ).fetchone()
        if not row:
            return None
        return BatchRecord(
            batch_id=row[0],
            strategy=row[1],
            dataset_id=row[2],
            params=json.loads(row[3]) if row[3] else {},
            timeframes=json.loads(row[4]) if row[4] else [],
            horizons=json.loads(row[5]) if row[5] else [],
            run_total=row[6],
            status=row[7] or "running",
            started_at=row[8],
            finished_at=row[9],
        )

    def save(
        self,
        run_id: str,
        batch_id: str | None,
        strategy: str,
        params: Dict,
        timeframe: str,
        start: str,
        end: str,
        dataset_id: str,
        starting_cash: float | None,
        metrics: Optional[PerformanceMetrics],
        run_started_at: Optional[str] = None,
        run_finished_at: Optional[str] = None,
        status: str = "finished",
        logical_run_id: str | None = None,
        requested_execution_mode: str | None = None,
        resolved_execution_mode: str | None = None,
        engine_impl: str | None = None,
        engine_version: str | None = None,
        fallback_reason: str | None = None,
    ) -> None:
        encoded_params = json.dumps(params, sort_keys=True)
        metrics_json = metrics.to_json() if metrics else "{}"
        run_started_at = run_started_at or None
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs
                (
                    run_id,
                    logical_run_id,
                    batch_id,
                    strategy,
                    params,
                    timeframe,
                    start,
                    end,
                    dataset_id,
                    starting_cash,
                    metrics,
                    run_started_at,
                    run_finished_at,
                    status,
                    requested_execution_mode,
                    resolved_execution_mode,
                    engine_impl,
                    engine_version,
                    fallback_reason
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    logical_run_id,
                    batch_id,
                    strategy,
                    encoded_params,
                    timeframe,
                    start,
                    end,
                    dataset_id,
                    starting_cash,
                    metrics_json,
                    run_started_at,
                    run_finished_at,
                    status,
                    requested_execution_mode,
                    resolved_execution_mode,
                    engine_impl,
                    engine_version,
                    fallback_reason,
                ),
            )

    def save_trades(self, run_id: str, trades: list) -> None:
        if trades is None:
            return
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM trades WHERE run_id=?", (run_id,))
            rows = []
            for i, t in enumerate(trades, start=1):
                rows.append(
                    (
                        run_id,
                        i,
                        str(getattr(t, "timestamp", "")),
                        getattr(t, "side", ""),
                        float(getattr(t, "qty", 0.0)),
                        float(getattr(t, "price", 0.0)),
                        float(getattr(t, "fee", 0.0)),
                        float(getattr(t, "realized_pnl", 0.0)),
                        float(getattr(t, "equity_after", 0.0)),
                    )
                )
            if rows:
                conn.executemany(
                    """
                    INSERT INTO trades
                    (run_id, seq, timestamp, side, qty, price, fee, realized_pnl, equity_after)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )

    def load_trades(self, run_id: str) -> list[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT seq,timestamp,side,qty,price,fee,realized_pnl,equity_after
                FROM trades WHERE run_id=? ORDER BY seq ASC
                """,
                (run_id,),
            ).fetchall()
        return [
            {
                "seq": r[0],
                "timestamp": r[1],
                "side": r[2],
                "qty": r[3],
                "price": r[4],
                "fee": r[5],
                "realized_pnl": r[6],
                "equity_after": r[7],
            }
            for r in rows
        ]

    # -- scheduler tasks ----------------------------------------------------
    def upsert_task(self, task_id: str, symbols: Dict, schedule: Dict, status: str = "active") -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO scheduled_tasks
                (task_id, created_at, updated_at, symbols, schedule, status, last_run_at, last_run_status, last_run_message, next_run_at)
                VALUES (
                    COALESCE((SELECT task_id FROM scheduled_tasks WHERE task_id=?), ?),
                    COALESCE((SELECT created_at FROM scheduled_tasks WHERE task_id=?), CURRENT_TIMESTAMP),
                    CURRENT_TIMESTAMP,
                    ?, ?, ?, 
                    COALESCE((SELECT last_run_at FROM scheduled_tasks WHERE task_id=?), NULL),
                    COALESCE((SELECT last_run_status FROM scheduled_tasks WHERE task_id=?), NULL),
                    COALESCE((SELECT last_run_message FROM scheduled_tasks WHERE task_id=?), NULL),
                    COALESCE((SELECT next_run_at FROM scheduled_tasks WHERE task_id=?), NULL)
                )
                """,
                (
                    task_id,
                    task_id,
                    task_id,
                    json.dumps(symbols),
                    json.dumps(schedule),
                    status,
                    task_id,
                    task_id,
                    task_id,
                    task_id,
                ),
            )

    def delete_task(self, task_id: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM scheduled_tasks WHERE task_id=?", (task_id,))

    def update_task_status(self, task_id: str, status: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE scheduled_tasks SET status=?, updated_at=CURRENT_TIMESTAMP WHERE task_id=?",
                (status, task_id),
            )

    def update_task_run_info(
        self,
        task_id: str,
        last_run_at: str | None,
        last_run_status: str | None,
        last_run_message: str | None,
        next_run_at: str | None,
    ) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE scheduled_tasks
                SET last_run_at=?, last_run_status=?, last_run_message=?, next_run_at=?, updated_at=CURRENT_TIMESTAMP
                WHERE task_id=?
                """,
                (last_run_at, last_run_status, last_run_message, next_run_at, task_id),
            )

    def add_task_run(
        self,
        run_id: str,
        task_id: str,
        started_at: str,
        status: str,
        ticker_count: int | None = None,
        log_path: str | None = None,
        error_message: str | None = None,
    ) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO task_runs
                (run_id, task_id, started_at, finished_at, status, ticker_count, log_path, error_message)
                VALUES (?, ?, ?, NULL, ?, ?, ?, ?)
                """,
                (run_id, task_id, started_at, status, ticker_count, log_path, error_message),
            )

    def finish_task_run(self, run_id: str, finished_at: str, status: str, error_message: str | None = None) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE task_runs
                SET finished_at=?, status=?, error_message=?
                WHERE run_id=?
                """,
                (finished_at, status, error_message, run_id),
            )

    # -- universes ----------------------------------------------------------
    def save_universe(
        self,
        *,
        universe_id: str,
        name: str,
        description: str = "",
        symbols: Sequence[str] = (),
        dataset_ids: Sequence[str] = (),
        source_preference: str | None = None,
    ) -> str:
        normalized_symbols = sorted(
            {
                str(symbol).strip().upper()
                for symbol in list(symbols or ())
                if str(symbol).strip()
            }
        )
        normalized_dataset_ids = sorted(
            {
                str(dataset_id).strip()
                for dataset_id in list(dataset_ids or ())
                if str(dataset_id).strip()
            }
        )
        name = str(name).strip()
        if not name:
            raise ValueError("Universe name is required.")
        if not normalized_symbols and not normalized_dataset_ids:
            raise ValueError("Universe must contain at least one symbol or dataset.")
        encoded_symbols = json.dumps(normalized_symbols)
        encoded_dataset_ids = json.dumps(normalized_dataset_ids)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO universes
                (universe_id, created_at, updated_at, name, description, symbols_json, dataset_ids_json, source_preference)
                VALUES (
                    COALESCE((SELECT universe_id FROM universes WHERE universe_id=?), ?),
                    COALESCE((SELECT created_at FROM universes WHERE universe_id=?), CURRENT_TIMESTAMP),
                    CURRENT_TIMESTAMP,
                    ?, ?, ?, ?, ?
                )
                """,
                (
                    universe_id,
                    universe_id,
                    universe_id,
                    name,
                    str(description or "").strip(),
                    encoded_symbols,
                    encoded_dataset_ids,
                    str(source_preference or "").strip() or None,
                ),
            )
        return universe_id

    def delete_universe(self, universe_id: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM universes WHERE universe_id=?", (universe_id,))

    def load_universes(self) -> list[UniverseRecord]:
        if not self.db_path.exists():
            return []
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT universe_id, name, description, symbols_json, dataset_ids_json, source_preference, created_at, updated_at
                FROM universes
                ORDER BY LOWER(name) ASC, created_at DESC
                """
            ).fetchall()
        return [
            UniverseRecord(
                universe_id=str(row[0]),
                name=str(row[1]),
                description=row[2],
                symbols_json=str(row[3] or "[]"),
                dataset_ids_json=str(row[4] or "[]"),
                source_preference=row[5],
                created_at=row[6],
                updated_at=row[7],
            )
            for row in rows
        ]

    # -- provider settings --------------------------------------------------
    def save_provider_settings(self, provider_id: str, settings: Dict) -> None:
        normalized_provider = str(provider_id or "").strip().lower()
        if not normalized_provider:
            raise ValueError("Provider id is required.")
        encoded = json.dumps(dict(settings or {}), sort_keys=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO provider_settings
                (provider_id, created_at, updated_at, settings_json)
                VALUES (
                    ?,
                    COALESCE((SELECT created_at FROM provider_settings WHERE provider_id=?), CURRENT_TIMESTAMP),
                    CURRENT_TIMESTAMP,
                    ?
                )
                """,
                (normalized_provider, normalized_provider, encoded),
            )

    def load_provider_settings(self, provider_id: str) -> ProviderSettingsRecord | None:
        normalized_provider = str(provider_id or "").strip().lower()
        if not normalized_provider or not self.db_path.exists():
            return None
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT provider_id, settings_json, updated_at
                FROM provider_settings
                WHERE provider_id=?
                """,
                (normalized_provider,),
            ).fetchone()
        if not row:
            return None
        return ProviderSettingsRecord(
            provider_id=str(row[0]),
            settings_json=str(row[1] or "{}"),
            updated_at=row[2],
        )

    # -- acquisition catalog ------------------------------------------------
    def upsert_acquisition_dataset(
        self,
        *,
        dataset_id: str,
        source: str | None = None,
        symbol: str | None = None,
        resolution: str | None = None,
        history_window: str | None = None,
        csv_path: str | None = None,
        parquet_path: str | None = None,
        coverage_start: str | None = None,
        coverage_end: str | None = None,
        bar_count: int | None = None,
        ingested: bool | None = None,
        last_download_attempt_at: str | None = None,
        last_download_success_at: str | None = None,
        last_ingest_at: str | None = None,
        last_status: str | None = None,
        last_error: str | None = None,
        last_run_id: str | None = None,
        last_task_id: str | None = None,
    ) -> None:
        existing = None
        with sqlite3.connect(self.db_path) as conn:
            existing = conn.execute(
                """
                SELECT
                    source, symbol, resolution, history_window, csv_path, parquet_path,
                    coverage_start, coverage_end, bar_count, ingested,
                    last_download_attempt_at, last_download_success_at, last_ingest_at,
                    last_status, last_error, last_run_id, last_task_id
                FROM acquisition_datasets
                WHERE dataset_id=?
                """,
                (dataset_id,),
            ).fetchone()
            source_val = source if source is not None else (existing[0] if existing else None)
            symbol_val = symbol if symbol is not None else (existing[1] if existing else None)
            resolution_val = resolution if resolution is not None else (existing[2] if existing else None)
            history_val = history_window if history_window is not None else (existing[3] if existing else None)
            csv_path_val = csv_path if csv_path is not None else (existing[4] if existing else None)
            parquet_path_val = parquet_path if parquet_path is not None else (existing[5] if existing else None)
            coverage_start_val = coverage_start if coverage_start is not None else (existing[6] if existing else None)
            coverage_end_val = coverage_end if coverage_end is not None else (existing[7] if existing else None)
            bar_count_val = bar_count if bar_count is not None else (existing[8] if existing else None)
            if ingested is None:
                ingested_val = int(existing[9] or 0) if existing else 0
            else:
                ingested_val = 1 if ingested else 0
            last_attempt_val = last_download_attempt_at if last_download_attempt_at is not None else (existing[10] if existing else None)
            last_success_val = last_download_success_at if last_download_success_at is not None else (existing[11] if existing else None)
            last_ingest_val = last_ingest_at if last_ingest_at is not None else (existing[12] if existing else None)
            last_status_val = last_status if last_status is not None else (existing[13] if existing else None)
            last_error_val = last_error if last_error is not None else (existing[14] if existing else None)
            last_run_val = last_run_id if last_run_id is not None else (existing[15] if existing else None)
            last_task_val = last_task_id if last_task_id is not None else (existing[16] if existing else None)
            conn.execute(
                """
                INSERT OR REPLACE INTO acquisition_datasets
                (
                    dataset_id, created_at, updated_at, source, symbol, resolution, history_window,
                    csv_path, parquet_path, coverage_start, coverage_end, bar_count, ingested,
                    last_download_attempt_at, last_download_success_at, last_ingest_at,
                    last_status, last_error, last_run_id, last_task_id
                )
                VALUES (
                    ?, COALESCE((SELECT created_at FROM acquisition_datasets WHERE dataset_id=?), CURRENT_TIMESTAMP),
                    CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    dataset_id,
                    dataset_id,
                    source_val,
                    symbol_val,
                    resolution_val,
                    history_val,
                    csv_path_val,
                    parquet_path_val,
                    coverage_start_val,
                    coverage_end_val,
                    bar_count_val,
                    ingested_val,
                    last_attempt_val,
                    last_success_val,
                    last_ingest_val,
                    last_status_val,
                    last_error_val,
                    last_run_val,
                    last_task_val,
                ),
            )

    def start_acquisition_run(
        self,
        *,
        acquisition_run_id: str,
        trigger_type: str,
        source: str | None = None,
        universe_id: str | None = None,
        universe_name: str | None = None,
        task_id: str | None = None,
        started_at: str,
        status: str = "running",
        symbol_count: int | None = None,
        notes: str | None = None,
        log_path: str | None = None,
    ) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO acquisition_runs
                (
                    acquisition_run_id, created_at, updated_at, trigger_type, source, universe_id,
                    universe_name, task_id, started_at, finished_at, status, symbol_count,
                    success_count, failed_count, ingested_count, notes, log_path
                )
                VALUES (
                    ?, COALESCE((SELECT created_at FROM acquisition_runs WHERE acquisition_run_id=?), CURRENT_TIMESTAMP),
                    CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, COALESCE((SELECT finished_at FROM acquisition_runs WHERE acquisition_run_id=?), NULL),
                    ?, ?, COALESCE((SELECT success_count FROM acquisition_runs WHERE acquisition_run_id=?), 0),
                    COALESCE((SELECT failed_count FROM acquisition_runs WHERE acquisition_run_id=?), 0),
                    COALESCE((SELECT ingested_count FROM acquisition_runs WHERE acquisition_run_id=?), 0),
                    ?, ?
                )
                """,
                (
                    acquisition_run_id,
                    acquisition_run_id,
                    trigger_type,
                    source,
                    universe_id,
                    universe_name,
                    task_id,
                    started_at,
                    acquisition_run_id,
                    status,
                    symbol_count,
                    acquisition_run_id,
                    acquisition_run_id,
                    acquisition_run_id,
                    notes,
                    log_path,
                ),
            )

    def finish_acquisition_run(
        self,
        acquisition_run_id: str,
        *,
        finished_at: str,
        status: str,
        success_count: int,
        failed_count: int,
        ingested_count: int,
        notes: str | None = None,
    ) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE acquisition_runs
                SET finished_at=?, status=?, success_count=?, failed_count=?, ingested_count=?,
                    notes=COALESCE(?, notes), updated_at=CURRENT_TIMESTAMP
                WHERE acquisition_run_id=?
                """,
                (
                    finished_at,
                    status,
                    success_count,
                    failed_count,
                    ingested_count,
                    notes,
                    acquisition_run_id,
                ),
            )

    def record_acquisition_attempt(
        self,
        *,
        attempt_id: str,
        acquisition_run_id: str,
        seq: int,
        source: str | None = None,
        symbol: str | None = None,
        dataset_id: str | None = None,
        status: str,
        started_at: str | None = None,
        finished_at: str | None = None,
        csv_path: str | None = None,
        parquet_path: str | None = None,
        coverage_start: str | None = None,
        coverage_end: str | None = None,
        bar_count: int | None = None,
        ingested: bool = False,
        error_message: str | None = None,
        log_path: str | None = None,
        task_id: str | None = None,
        universe_id: str | None = None,
        resolution: str | None = None,
        history_window: str | None = None,
    ) -> None:
        stored_error_message = error_message if status in _ACQUISITION_ERROR_STATUSES else None
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO acquisition_attempts
                (
                    attempt_id, acquisition_run_id, seq, source, symbol, dataset_id, status,
                    started_at, finished_at, csv_path, parquet_path, coverage_start, coverage_end,
                    bar_count, ingested, error_message, log_path, task_id, universe_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    attempt_id,
                    acquisition_run_id,
                    seq,
                    source,
                    symbol,
                    dataset_id,
                    status,
                    started_at,
                    finished_at,
                    csv_path,
                    parquet_path,
                    coverage_start,
                    coverage_end,
                    bar_count,
                    1 if ingested else 0,
                    stored_error_message,
                    log_path,
                    task_id,
                    universe_id,
                ),
            )
        if dataset_id:
            success_statuses = {"downloaded", "ingested", "gap_filled", "success", "ingest_error"}
            last_download_success_at = finished_at if status in success_statuses or ingested else None
            last_ingest_at = finished_at if ingested else None
            dataset_last_error = stored_error_message if status in _ACQUISITION_ERROR_STATUSES else ""
            self.upsert_acquisition_dataset(
                dataset_id=dataset_id,
                source=source,
                symbol=symbol,
                resolution=resolution,
                history_window=history_window,
                csv_path=csv_path,
                parquet_path=parquet_path,
                coverage_start=coverage_start,
                coverage_end=coverage_end,
                bar_count=bar_count,
                ingested=ingested,
                last_download_attempt_at=finished_at or started_at,
                last_download_success_at=last_download_success_at,
                last_ingest_at=last_ingest_at,
                last_status=status,
                last_error=dataset_last_error,
                last_run_id=acquisition_run_id,
                last_task_id=task_id,
            )

    def load_acquisition_datasets(self) -> list[AcquisitionDatasetRecord]:
        if not self.db_path.exists():
            return []
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT
                    dataset_id, source, symbol, resolution, history_window, csv_path, parquet_path,
                    coverage_start, coverage_end, bar_count, ingested,
                    last_download_attempt_at, last_download_success_at, last_ingest_at,
                    last_status, last_error, last_run_id, last_task_id, created_at, updated_at
                FROM acquisition_datasets
                ORDER BY COALESCE(last_download_attempt_at, last_ingest_at, updated_at, created_at) DESC, dataset_id ASC
                """
            ).fetchall()
        return [
            AcquisitionDatasetRecord(
                dataset_id=str(row[0]),
                source=row[1],
                symbol=row[2],
                resolution=row[3],
                history_window=row[4],
                csv_path=row[5],
                parquet_path=row[6],
                coverage_start=row[7],
                coverage_end=row[8],
                bar_count=row[9],
                ingested=bool(row[10]),
                last_download_attempt_at=row[11],
                last_download_success_at=row[12],
                last_ingest_at=row[13],
                last_status=row[14],
                last_error=row[15],
                last_run_id=row[16],
                last_task_id=row[17],
                created_at=row[18],
                updated_at=row[19],
            )
            for row in rows
        ]

    def load_acquisition_runs(
        self,
        limit: int | None = None,
        *,
        task_id: str | None = None,
        universe_id: str | None = None,
    ) -> list[AcquisitionRunRecord]:
        if not self.db_path.exists():
            return []
        query = """
            SELECT
                acquisition_run_id, trigger_type, source, universe_id, universe_name, task_id,
                started_at, finished_at, status, symbol_count, success_count, failed_count,
                ingested_count, notes, log_path, created_at, updated_at
            FROM acquisition_runs
        """
        clauses: list[str] = []
        params: list[object] = []
        if task_id:
            clauses.append("task_id=?")
            params.append(task_id)
        if universe_id:
            clauses.append("universe_id=?")
            params.append(universe_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY COALESCE(finished_at, started_at, updated_at, created_at) DESC, acquisition_run_id DESC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(int(limit))
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        return [
            AcquisitionRunRecord(
                acquisition_run_id=str(row[0]),
                trigger_type=str(row[1]),
                source=row[2],
                universe_id=row[3],
                universe_name=row[4],
                task_id=row[5],
                started_at=str(row[6]),
                finished_at=row[7],
                status=str(row[8]),
                symbol_count=row[9],
                success_count=int(row[10] or 0),
                failed_count=int(row[11] or 0),
                ingested_count=int(row[12] or 0),
                notes=row[13],
                log_path=row[14],
                created_at=row[15],
                updated_at=row[16],
            )
            for row in rows
        ]

    def load_acquisition_attempts(
        self,
        *,
        acquisition_run_id: str | None = None,
        dataset_id: str | None = None,
        symbol: str | None = None,
        task_id: str | None = None,
        universe_id: str | None = None,
        limit: int | None = None,
    ) -> list[AcquisitionAttemptRecord]:
        if not self.db_path.exists():
            return []
        query = """
            SELECT
                attempt_id, acquisition_run_id, seq, source, symbol, dataset_id, status,
                started_at, finished_at, csv_path, parquet_path, coverage_start, coverage_end,
                bar_count, ingested, error_message, log_path, task_id, universe_id
            FROM acquisition_attempts
        """
        clauses: list[str] = []
        params: list[object] = []
        if acquisition_run_id:
            clauses.append("acquisition_run_id=?")
            params.append(acquisition_run_id)
        if dataset_id:
            clauses.append("dataset_id=?")
            params.append(dataset_id)
        if symbol:
            clauses.append("symbol=?")
            params.append(symbol)
        if task_id:
            clauses.append("task_id=?")
            params.append(task_id)
        if universe_id:
            clauses.append("universe_id=?")
            params.append(universe_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY COALESCE(finished_at, started_at) DESC, seq DESC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(int(limit))
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        return [
            AcquisitionAttemptRecord(
                attempt_id=str(row[0]),
                acquisition_run_id=str(row[1]),
                seq=int(row[2] or 0),
                source=row[3],
                symbol=row[4],
                dataset_id=row[5],
                status=str(row[6]),
                started_at=row[7],
                finished_at=row[8],
                csv_path=row[9],
                parquet_path=row[10],
                coverage_start=row[11],
                coverage_end=row[12],
                bar_count=row[13],
                ingested=bool(row[14]),
                error_message=(row[15] if str(row[6]) in _ACQUISITION_ERROR_STATUSES else None),
                log_path=row[16],
                task_id=row[17],
                universe_id=row[18],
            )
            for row in rows
        ]

    def load_task_runs(self, *, task_id: str | None = None, limit: int | None = None) -> list[TaskRunRecord]:
        if not self.db_path.exists():
            return []
        query = """
            SELECT
                run_id, task_id, started_at, finished_at, status, ticker_count, log_path, error_message
            FROM task_runs
        """
        clauses: list[str] = []
        params: list[object] = []
        if task_id:
            clauses.append("task_id=?")
            params.append(task_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY COALESCE(finished_at, started_at) DESC, run_id DESC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(int(limit))
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        return [
            TaskRunRecord(
                run_id=str(row[0]),
                task_id=str(row[1]),
                started_at=row[2],
                finished_at=row[3],
                status=str(row[4]),
                ticker_count=row[5],
                log_path=row[6],
                error_message=row[7],
            )
            for row in rows
        ]

    def save_batch(
        self,
        batch_id: str,
        strategy: str,
        dataset_id: str,
        params: Dict,
        timeframes: list[str],
        horizons: list[str],
        run_total: int | None,
        status: str,
        started_at: Optional[str] = None,
        finished_at: Optional[str] = None,
    ) -> None:
        encoded_params = json.dumps(params, sort_keys=True)
        encoded_timeframes = json.dumps(timeframes, sort_keys=False)
        encoded_horizons = json.dumps(horizons, sort_keys=False)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO batches
                (batch_id, strategy, dataset_id, params, timeframes, horizons, run_total, status, started_at, finished_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (batch_id, strategy, dataset_id, encoded_params, encoded_timeframes, encoded_horizons, run_total, status, started_at, finished_at),
            )

    def save_batch_benchmarks(self, batch_id: str, benchmarks: list | tuple) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM batch_benchmarks WHERE batch_id=?", (batch_id,))
            rows = []
            for seq, benchmark in enumerate(benchmarks, start=1):
                requested_mode = getattr(benchmark, "requested_execution_mode", "")
                resolved_mode = getattr(benchmark, "resolved_execution_mode", "")
                rows.append(
                    (
                        batch_id,
                        seq,
                        str(getattr(benchmark, "dataset_id", "")),
                        str(getattr(benchmark, "strategy", "")),
                        str(getattr(benchmark, "timeframe", "")),
                        getattr(requested_mode, "value", str(requested_mode)),
                        getattr(resolved_mode, "value", str(resolved_mode)),
                        str(getattr(benchmark, "engine_impl", "")),
                        str(getattr(benchmark, "engine_version", "")),
                        int(getattr(benchmark, "bars", 0)),
                        int(getattr(benchmark, "total_params", 0)),
                        int(getattr(benchmark, "cached_runs", 0)),
                        int(getattr(benchmark, "uncached_runs", 0)),
                        float(getattr(benchmark, "duration_seconds", 0.0)),
                        int(getattr(benchmark, "chunk_count", 0)),
                        json.dumps(list(getattr(benchmark, "chunk_sizes", ()))),
                        (
                            int(getattr(benchmark, "effective_param_batch_size"))
                            if getattr(benchmark, "effective_param_batch_size", None) is not None
                            else None
                        ),
                        1 if bool(getattr(benchmark, "prepared_context_reused", False)) else 0,
                    )
                )
            if rows:
                conn.executemany(
                    """
                    INSERT INTO batch_benchmarks
                    (
                        batch_id,
                        seq,
                        dataset_id,
                        strategy,
                        timeframe,
                        requested_execution_mode,
                        resolved_execution_mode,
                        engine_impl,
                        engine_version,
                        bars,
                        total_params,
                        cached_runs,
                        uncached_runs,
                        duration_seconds,
                        chunk_count,
                        chunk_sizes,
                        effective_param_batch_size,
                        prepared_context_reused
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )

    def load_batch_benchmarks(self, batch_id: str) -> list[BatchBenchmarkRecord]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT
                    batch_id,
                    seq,
                    dataset_id,
                    strategy,
                    timeframe,
                    requested_execution_mode,
                    resolved_execution_mode,
                    engine_impl,
                    engine_version,
                    bars,
                    total_params,
                    cached_runs,
                    uncached_runs,
                    duration_seconds,
                    chunk_count,
                    chunk_sizes,
                    effective_param_batch_size,
                    prepared_context_reused
                FROM batch_benchmarks
                WHERE batch_id=?
                ORDER BY seq ASC
                """,
                (batch_id,),
            ).fetchall()
        return [
            BatchBenchmarkRecord(
                batch_id=row[0],
                seq=row[1],
                dataset_id=row[2],
                strategy=row[3],
                timeframe=row[4],
                requested_execution_mode=row[5],
                resolved_execution_mode=row[6],
                engine_impl=row[7],
                engine_version=row[8],
                bars=row[9],
                total_params=row[10],
                cached_runs=row[11],
                uncached_runs=row[12],
                duration_seconds=row[13],
                chunk_count=row[14] or 0,
                chunk_sizes=tuple(json.loads(row[15]) if row[15] else []),
                effective_param_batch_size=row[16],
                prepared_context_reused=bool(row[17]),
            )
            for row in rows
        ]

    def save_optimization_study(
        self,
        *,
        study_id: str,
        batch_id: str,
        strategy: str,
        dataset_scope: list[str] | tuple[str, ...],
        param_names: list[str] | tuple[str, ...],
        timeframes: list[str] | tuple[str, ...],
        horizons: list[str] | tuple[str, ...],
        score_version: str,
        aggregates: pd.DataFrame,
        asset_results: pd.DataFrame,
    ) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO optimization_studies
                (study_id, batch_id, strategy, dataset_scope, param_names, timeframes, horizons, score_version, aggregate_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(study_id),
                    str(batch_id),
                    str(strategy),
                    json.dumps(list(dataset_scope)),
                    json.dumps(list(param_names)),
                    json.dumps(list(timeframes)),
                    json.dumps(list(horizons)),
                    str(score_version),
                    int(len(aggregates)),
                ),
            )
            conn.execute("DELETE FROM optimization_aggregates WHERE study_id=?", (str(study_id),))
            conn.execute("DELETE FROM optimization_asset_results WHERE study_id=?", (str(study_id),))
            if aggregates is not None and not aggregates.empty:
                rows = []
                for _, row in aggregates.iterrows():
                    rows.append(
                        (
                            str(study_id),
                            int(row["seq"]),
                            str(row["timeframe"]),
                            str(row.get("start", "") or ""),
                            str(row.get("end", "") or ""),
                            str(row["param_key"]),
                            str(row["params_json"]),
                            int(row["dataset_count"]),
                            int(row["run_count"]),
                            float(row["median_total_return"]),
                            float(row["median_cagr"]) if pd.notna(row.get("median_cagr")) else None,
                            float(row["median_sharpe"]),
                            float(row["median_rolling_sharpe"]) if pd.notna(row.get("median_rolling_sharpe")) else None,
                            float(row["worst_max_drawdown"]),
                            float(row["sharpe_std"]),
                            float(row["profitable_asset_ratio"]),
                            float(row["robust_score"]),
                        )
                    )
                conn.executemany(
                    """
                    INSERT INTO optimization_aggregates
                    (
                        study_id, seq, timeframe, start, end, param_key, params_json, dataset_count, run_count,
                        median_total_return, median_cagr, median_sharpe, median_rolling_sharpe,
                        worst_max_drawdown, sharpe_std, profitable_asset_ratio, robust_score
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
            if asset_results is not None and not asset_results.empty:
                rows = []
                for _, row in asset_results.iterrows():
                    rows.append(
                        (
                            str(study_id),
                            int(row["seq"]),
                            str(row["timeframe"]),
                            str(row.get("start", "") or ""),
                            str(row.get("end", "") or ""),
                            str(row["dataset_id"]),
                            str(row["param_key"]),
                            str(row["params_json"]),
                            float(row["total_return"]) if pd.notna(row.get("total_return")) else None,
                            float(row["cagr"]) if pd.notna(row.get("cagr")) else None,
                            float(row["sharpe"]) if pd.notna(row.get("sharpe")) else None,
                            float(row["rolling_sharpe"]) if pd.notna(row.get("rolling_sharpe")) else None,
                            float(row["max_drawdown"]) if pd.notna(row.get("max_drawdown")) else None,
                            str(row.get("run_id", "") or ""),
                        )
                    )
                conn.executemany(
                    """
                    INSERT INTO optimization_asset_results
                    (
                        study_id, seq, timeframe, start, end, dataset_id, param_key, params_json,
                        total_return, cagr, sharpe, rolling_sharpe, max_drawdown, run_id
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )

    def load_optimization_studies(self) -> list[OptimizationStudyRecord]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT study_id, batch_id, strategy, dataset_scope, param_names, timeframes, horizons, score_version, aggregate_count, created_at
                FROM optimization_studies
                ORDER BY created_at DESC, study_id DESC
                """
            ).fetchall()
        return [
            OptimizationStudyRecord(
                study_id=row[0],
                batch_id=row[1],
                strategy=row[2],
                dataset_scope=tuple(json.loads(row[3]) if row[3] else []),
                param_names=tuple(json.loads(row[4]) if row[4] else []),
                timeframes=tuple(json.loads(row[5]) if row[5] else []),
                horizons=tuple(json.loads(row[6]) if row[6] else []),
                score_version=row[7],
                aggregate_count=int(row[8] or 0),
                created_at=row[9],
            )
            for row in rows
        ]

    def load_optimization_aggregates(self, study_id: str) -> pd.DataFrame:
        if not study_id:
            return pd.DataFrame()
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(
                """
                SELECT *
                FROM optimization_aggregates
                WHERE study_id=?
                ORDER BY robust_score DESC, seq ASC
                """,
                conn,
                params=(study_id,),
            )

    def load_optimization_asset_results(self, study_id: str) -> pd.DataFrame:
        if not study_id:
            return pd.DataFrame()
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(
                """
                SELECT *
                FROM optimization_asset_results
                WHERE study_id=?
                ORDER BY timeframe ASC, start ASC, end ASC, param_key ASC, dataset_id ASC
                """,
                conn,
                params=(study_id,),
            )

    def save_optimization_candidate(
        self,
        *,
        study_id: str,
        timeframe: str,
        start: str,
        end: str,
        param_key: str,
        params_json: str,
        source_type: str,
        promotion_reason: str,
        status: str,
        metrics: Dict,
        asset_results: pd.DataFrame | None = None,
        artifact_refs: Dict | None = None,
        notes: str = "",
    ) -> str:
        candidate_id = sha1(
            "|".join(
                [
                    str(study_id),
                    str(timeframe),
                    str(start or ""),
                    str(end or ""),
                    str(param_key),
                ]
            ).encode("utf-8")
        ).hexdigest()
        metrics_json = json.dumps(metrics or {}, sort_keys=True)
        asset_results_json = None
        if asset_results is not None and not asset_results.empty:
            asset_results_json = asset_results.to_json(orient="records")
        artifact_refs_json = json.dumps(artifact_refs or {}, sort_keys=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO optimization_candidates
                (
                    candidate_id, study_id, timeframe, start, end, param_key, params_json,
                    source_type, promotion_reason, status, metrics_json,
                    asset_results_json, artifact_refs_json, notes
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(candidate_id) DO UPDATE SET
                    timeframe=excluded.timeframe,
                    start=excluded.start,
                    end=excluded.end,
                    param_key=excluded.param_key,
                    params_json=excluded.params_json,
                    source_type=excluded.source_type,
                    promotion_reason=excluded.promotion_reason,
                    status=excluded.status,
                    metrics_json=excluded.metrics_json,
                    asset_results_json=excluded.asset_results_json,
                    artifact_refs_json=excluded.artifact_refs_json,
                    notes=excluded.notes,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    candidate_id,
                    str(study_id),
                    str(timeframe),
                    str(start or ""),
                    str(end or ""),
                    str(param_key),
                    str(params_json),
                    str(source_type),
                    str(promotion_reason or ""),
                    str(status or "queued"),
                    metrics_json,
                    asset_results_json,
                    artifact_refs_json,
                    str(notes or ""),
                ),
            )
        return candidate_id

    def load_optimization_candidates(self, study_id: str = "") -> list[OptimizationCandidateRecord]:
        with sqlite3.connect(self.db_path) as conn:
            if study_id:
                rows = conn.execute(
                    """
                    SELECT
                        candidate_id, study_id, timeframe, start, end, param_key, params_json,
                        source_type, promotion_reason, status, metrics_json, asset_results_json,
                        artifact_refs_json, notes, created_at, updated_at
                    FROM optimization_candidates
                    WHERE study_id=?
                    ORDER BY created_at DESC, candidate_id DESC
                    """,
                    (study_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT
                        candidate_id, study_id, timeframe, start, end, param_key, params_json,
                        source_type, promotion_reason, status, metrics_json, asset_results_json,
                        artifact_refs_json, notes, created_at, updated_at
                    FROM optimization_candidates
                    ORDER BY created_at DESC, candidate_id DESC
                    """
                ).fetchall()
        return [
            OptimizationCandidateRecord(
                candidate_id=row[0],
                study_id=row[1],
                timeframe=row[2],
                start=row[3],
                end=row[4],
                param_key=row[5],
                params_json=row[6],
                source_type=row[7],
                promotion_reason=row[8],
                status=row[9],
                metrics_json=row[10],
                asset_results_json=row[11],
                artifact_refs_json=row[12],
                notes=row[13],
                created_at=row[14],
                updated_at=row[15],
            )
            for row in rows
        ]

    def delete_optimization_candidate(self, candidate_id: str) -> None:
        if not candidate_id:
            return
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM optimization_candidates WHERE candidate_id=?", (candidate_id,))

    def save_walk_forward_study(
        self,
        *,
        wf_study_id: str,
        batch_id: str,
        strategy: str,
        dataset_id: str,
        timeframe: str,
        candidate_source_mode: str,
        param_names: list[str] | tuple[str, ...],
        schedule_json: Dict,
        selection_rule: str,
        params_json: Dict,
        status: str,
        description: str = "",
        folds: pd.DataFrame | None = None,
        fold_metrics: pd.DataFrame | None = None,
        stitched_metrics: Dict | None = None,
        stitched_equity: pd.Series | None = None,
    ) -> None:
        encoded_schedule = json.dumps(schedule_json or {}, sort_keys=True)
        encoded_params = json.dumps(params_json or {}, sort_keys=True)
        encoded_param_names = json.dumps(list(param_names or ()))
        encoded_stitched_metrics = json.dumps(stitched_metrics or {}, sort_keys=True)
        encoded_stitched_equity = None
        if stitched_equity is not None and not stitched_equity.empty:
            encoded_stitched_equity = stitched_equity.to_json(date_format="iso", date_unit="ns")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO walk_forward_studies
                (
                    wf_study_id, batch_id, strategy, dataset_id, timeframe, candidate_source_mode,
                    param_names, schedule_json, selection_rule, params_json, status, description,
                    stitched_metrics_json, stitched_equity_json, fold_count
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(wf_study_id),
                    str(batch_id),
                    str(strategy),
                    str(dataset_id),
                    str(timeframe),
                    str(candidate_source_mode),
                    encoded_param_names,
                    encoded_schedule,
                    str(selection_rule),
                    encoded_params,
                    str(status),
                    str(description or ""),
                    encoded_stitched_metrics,
                    encoded_stitched_equity,
                    int(len(folds) if folds is not None else 0),
                ),
            )
            conn.execute("DELETE FROM walk_forward_folds WHERE wf_study_id=?", (str(wf_study_id),))
            conn.execute("DELETE FROM walk_forward_fold_metrics WHERE wf_study_id=?", (str(wf_study_id),))
            if folds is not None and not folds.empty:
                fold_rows = []
                for _, row in folds.iterrows():
                    fold_rows.append(
                        (
                            str(wf_study_id),
                            int(row["fold_index"]),
                            str(row.get("train_study_id", "") or ""),
                            str(row["timeframe"]),
                            str(row["train_start"]),
                            str(row["train_end"]),
                            str(row["test_start"]),
                            str(row["test_end"]),
                            str(row.get("selected_param_set_id", "") or ""),
                            str(row["selected_params_json"]),
                            int(row["train_rank"]) if pd.notna(row.get("train_rank")) else None,
                            float(row["train_robust_score"]) if pd.notna(row.get("train_robust_score")) else None,
                            str(row.get("test_run_id", "") or ""),
                            str(row["status"]),
                        )
                    )
                conn.executemany(
                    """
                    INSERT INTO walk_forward_folds
                    (
                        wf_study_id, fold_index, train_study_id, timeframe, train_start, train_end,
                        test_start, test_end, selected_param_set_id, selected_params_json, train_rank,
                        train_robust_score, test_run_id, status
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    fold_rows,
                )
            if fold_metrics is not None and not fold_metrics.empty:
                metric_rows = []
                for _, row in fold_metrics.iterrows():
                    metric_rows.append(
                        (
                            str(wf_study_id),
                            int(row["fold_index"]),
                            str(row["train_metrics_json"]),
                            str(row["test_metrics_json"]),
                            str(row["degradation_json"]),
                            str(row["param_drift_json"]),
                        )
                    )
                conn.executemany(
                    """
                    INSERT INTO walk_forward_fold_metrics
                    (
                        wf_study_id, fold_index, train_metrics_json, test_metrics_json, degradation_json, param_drift_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    metric_rows,
                )

    def load_walk_forward_studies(self) -> list[WalkForwardStudyRecord]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT
                    wf_study_id, batch_id, strategy, dataset_id, timeframe, candidate_source_mode,
                    param_names, schedule_json, selection_rule, params_json, status, description,
                    stitched_metrics_json, stitched_equity_json, fold_count, created_at
                FROM walk_forward_studies
                ORDER BY created_at DESC, wf_study_id DESC
                """
            ).fetchall()
        return [
            WalkForwardStudyRecord(
                wf_study_id=row[0],
                batch_id=row[1],
                strategy=row[2],
                dataset_id=row[3],
                timeframe=row[4],
                candidate_source_mode=row[5],
                param_names=tuple(json.loads(row[6]) if row[6] else []),
                schedule_json=row[7],
                selection_rule=row[8],
                params_json=row[9],
                status=row[10],
                description=row[11],
                stitched_metrics_json=row[12],
                stitched_equity_json=row[13],
                fold_count=int(row[14] or 0),
                created_at=row[15],
            )
            for row in rows
        ]

    def load_walk_forward_folds(self, wf_study_id: str) -> list[WalkForwardFoldRecord]:
        if not wf_study_id:
            return []
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT
                    wf_study_id, fold_index, train_study_id, timeframe, train_start, train_end,
                    test_start, test_end, selected_param_set_id, selected_params_json, train_rank,
                    train_robust_score, test_run_id, status
                FROM walk_forward_folds
                WHERE wf_study_id=?
                ORDER BY fold_index ASC
                """,
                (wf_study_id,),
            ).fetchall()
        return [
            WalkForwardFoldRecord(
                wf_study_id=row[0],
                fold_index=int(row[1]),
                train_study_id=row[2],
                timeframe=row[3],
                train_start=row[4],
                train_end=row[5],
                test_start=row[6],
                test_end=row[7],
                selected_param_set_id=row[8],
                selected_params_json=row[9],
                train_rank=int(row[10]) if row[10] is not None else None,
                train_robust_score=float(row[11]) if row[11] is not None else None,
                test_run_id=row[12],
                status=row[13],
            )
            for row in rows
        ]

    def load_walk_forward_fold_metrics(self, wf_study_id: str) -> list[WalkForwardFoldMetricsRecord]:
        if not wf_study_id:
            return []
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT
                    wf_study_id, fold_index, train_metrics_json, test_metrics_json, degradation_json, param_drift_json
                FROM walk_forward_fold_metrics
                WHERE wf_study_id=?
                ORDER BY fold_index ASC
                """,
                (wf_study_id,),
            ).fetchall()
        return [
            WalkForwardFoldMetricsRecord(
                wf_study_id=row[0],
                fold_index=int(row[1]),
                train_metrics_json=row[2],
                test_metrics_json=row[3],
                degradation_json=row[4],
                param_drift_json=row[5],
            )
            for row in rows
        ]

    def save_monte_carlo_study(
        self,
        *,
        mc_study_id: str,
        source_type: str,
        source_id: str,
        resampling_mode: str,
        simulation_count: int,
        seed: int | None,
        cost_stress_json: dict,
        status: str,
        description: str,
        source_trade_count: int,
        starting_equity: float,
        summary_json: dict,
        fan_quantiles_json: dict,
        terminal_returns_json: list[float],
        max_drawdowns_json: list[float],
        terminal_equities_json: list[float],
        original_path_json: list[float],
        representative_paths: list[dict],
    ) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO monte_carlo_studies
                (
                    mc_study_id,
                    source_type,
                    source_id,
                    resampling_mode,
                    simulation_count,
                    seed,
                    cost_stress_json,
                    status,
                    description,
                    source_trade_count,
                    starting_equity,
                    summary_json,
                    fan_quantiles_json,
                    terminal_returns_json,
                    max_drawdowns_json,
                    terminal_equities_json,
                    original_path_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(mc_study_id),
                    str(source_type),
                    str(source_id),
                    str(resampling_mode),
                    int(simulation_count),
                    int(seed) if seed is not None else None,
                    json.dumps(cost_stress_json or {}, sort_keys=True),
                    str(status),
                    str(description or ""),
                    int(source_trade_count),
                    float(starting_equity),
                    json.dumps(summary_json or {}, sort_keys=True),
                    json.dumps(fan_quantiles_json or {}, sort_keys=True),
                    json.dumps(list(terminal_returns_json or [])),
                    json.dumps(list(max_drawdowns_json or [])),
                    json.dumps(list(terminal_equities_json or [])),
                    json.dumps(list(original_path_json or [])),
                ),
            )
            conn.execute("DELETE FROM monte_carlo_paths WHERE mc_study_id=?", (str(mc_study_id),))
            rows = []
            for item in representative_paths or []:
                rows.append(
                    (
                        str(mc_study_id),
                        str(item.get("path_id") or ""),
                        str(item.get("path_type") or ""),
                        json.dumps(list(item.get("path") or [])),
                        json.dumps(dict(item.get("summary") or {}), sort_keys=True),
                    )
                )
            if rows:
                conn.executemany(
                    """
                    INSERT INTO monte_carlo_paths
                    (mc_study_id, path_id, path_type, path_json, summary_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    rows,
                )

    def load_monte_carlo_studies(self) -> list[MonteCarloStudyRecord]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT
                    mc_study_id, source_type, source_id, resampling_mode, simulation_count, seed,
                    cost_stress_json, status, description, source_trade_count, starting_equity,
                    summary_json, fan_quantiles_json, terminal_returns_json, max_drawdowns_json,
                    terminal_equities_json, original_path_json, created_at
                FROM monte_carlo_studies
                ORDER BY created_at DESC, mc_study_id DESC
                """
            ).fetchall()
        return [
            MonteCarloStudyRecord(
                mc_study_id=row[0],
                source_type=row[1],
                source_id=row[2],
                resampling_mode=row[3],
                simulation_count=int(row[4] or 0),
                seed=int(row[5]) if row[5] is not None else None,
                cost_stress_json=row[6],
                status=row[7],
                description=row[8],
                source_trade_count=int(row[9] or 0),
                starting_equity=float(row[10] or 0.0),
                summary_json=row[11],
                fan_quantiles_json=row[12],
                terminal_returns_json=row[13],
                max_drawdowns_json=row[14],
                terminal_equities_json=row[15],
                original_path_json=row[16],
                created_at=row[17],
            )
            for row in rows
        ]

    def load_monte_carlo_paths(self, mc_study_id: str) -> list[MonteCarloPathRecord]:
        if not mc_study_id:
            return []
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT mc_study_id, path_id, path_type, path_json, summary_json
                FROM monte_carlo_paths
                WHERE mc_study_id=?
                ORDER BY path_type ASC, path_id ASC
                """,
                (str(mc_study_id),),
            ).fetchall()
        return [
            MonteCarloPathRecord(
                mc_study_id=row[0],
                path_id=row[1],
                path_type=row[2],
                path_json=row[3],
                summary_json=row[4],
            )
            for row in rows
        ]

    def save_heatmap(self, heatmap_id: str, params: Dict, file_path: str, description: str = "") -> None:
        encoded_params = json.dumps(params, sort_keys=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO heatmaps
                (heatmap_id, description, params, file_path)
                VALUES (?, ?, ?, ?)
                """,
                (heatmap_id, description, encoded_params, file_path),
            )
