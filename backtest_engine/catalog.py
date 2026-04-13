from __future__ import annotations

import json
import sqlite3
from hashlib import sha1
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Dict, Optional, Sequence
import uuid

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
class DeploymentTargetRecord:
    target_id: str
    name: str
    mode: str
    broker_scope: str
    transport_mode: str
    base_url: str | None
    webhook_path: str | None
    status_path: str | None
    dashboard_path: str | None
    logs_path: str | None
    project_root: str | None
    db_path: str | None
    log_db_path: str | None
    secret_ref: str | None
    is_active: bool
    created_at: str | None
    updated_at: str | None


@dataclass
class ManualDeploymentDefinitionRecord:
    manual_definition_id: str
    deployment_kind: str
    strategy: str
    strategy_version: str | None
    dataset_id: str | None
    symbol: str | None
    dataset_scope_json: str
    timeframe: str | None
    params_json: str
    structure_json: str
    target_id: str | None
    mode: str | None
    sizing_json: str
    notes: str | None
    created_at: str | None
    updated_at: str | None


@dataclass
class DeploymentRecord:
    deployment_id: str
    parent_deployment_id: str | None
    deployment_kind: str
    source_type: str
    source_id: str
    candidate_id: str | None
    strategy: str
    strategy_version: str | None
    dataset_id: str | None
    symbol: str | None
    timeframe: str | None
    params_json: str
    structure_json: str
    validation_refs_json: str
    target_id: str | None
    mode: str | None
    sizing_json: str
    status: str
    status_reason: str | None
    last_signal_at: str | None
    last_sync_at: str | None
    last_error_at: str | None
    notes: str | None
    created_at: str | None
    updated_at: str | None
    armed_at: str | None
    started_at: str | None
    stopped_at: str | None


@dataclass
class DeploymentChildLinkRecord:
    parent_deployment_id: str
    child_deployment_id: str
    child_role: str | None
    dataset_id: str | None
    symbol: str | None
    strategy_block_id: str | None
    created_at: str | None


@dataclass
class DeploymentMetricSnapshotRecord:
    deployment_id: str
    snapshot_ts: str
    realized_pnl: float | None
    open_pnl: float | None
    trade_count: int | None
    win_count: int | None
    loss_count: int | None
    win_rate: float | None
    profit_factor: float | None
    sharpe: float | None
    current_position_json: str
    health_json: str


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
class AssetCatalogRecord:
    asset_id: str
    symbol: str
    display_symbol: str | None
    name: str | None
    asset_class: str | None
    security_type: str | None
    exchange: str | None
    country: str | None
    currency: str | None
    sector: str | None
    industry: str | None
    dataset_status: str | None
    dataset_count: int
    successful_dataset_count: int
    latest_dataset_id: str | None
    latest_source: str | None
    latest_download_at: str | None
    latest_success_at: str | None
    latest_failure_at: str | None
    latest_failure_reason: str | None
    coverage_start: str | None
    coverage_end: str | None
    freshness_status: str | None
    first_seen_at: str | None
    last_seen_at: str | None
    created_at: str | None
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
    quality_state: str | None
    quality_expected_interval: str | None
    suspicious_gap_count: int | None
    max_suspicious_gap: str | None
    suspicious_gap_ranges_json: str | None
    repair_request_start: str | None
    repair_request_end: str | None
    quality_analyzed_at: str | None
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

    _SCHEMA_INIT_LOCK = Lock()
    _SCHEMA_INIT_DONE: set[str] = set()

    def __init__(self, db_path: str | Path = "backtests.sqlite") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        db_key = str(self.db_path.resolve())
        if db_key not in self._SCHEMA_INIT_DONE:
            with self._SCHEMA_INIT_LOCK:
                if db_key not in self._SCHEMA_INIT_DONE:
                    self._init_schema()
                    self._SCHEMA_INIT_DONE.add(db_key)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except Exception:
            pass
        try:
            conn.execute("PRAGMA synchronous=NORMAL")
        except Exception:
            pass
        try:
            conn.execute("PRAGMA busy_timeout=30000")
        except Exception:
            pass
        return conn

    def _init_schema(self) -> None:
        with self.connect() as conn:
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
                CREATE TABLE IF NOT EXISTS asset_master (
                    asset_id TEXT PRIMARY KEY,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    symbol TEXT NOT NULL,
                    display_symbol TEXT,
                    name TEXT,
                    asset_class TEXT,
                    security_type TEXT,
                    exchange TEXT,
                    mic TEXT,
                    country TEXT,
                    currency TEXT,
                    is_active INTEGER DEFAULT 1,
                    is_delisted INTEGER DEFAULT 0,
                    is_tradable INTEGER DEFAULT 1,
                    first_seen_at TEXT,
                    last_seen_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS asset_identifiers (
                    asset_id TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    provider_symbol TEXT,
                    exchange_symbol TEXT,
                    conid TEXT,
                    isin TEXT,
                    cusip TEXT,
                    figi TEXT,
                    composite_figi TEXT,
                    shareclass_figi TEXT,
                    composite_key TEXT,
                    is_primary INTEGER DEFAULT 0,
                    PRIMARY KEY (asset_id, provider, provider_symbol)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS asset_classifications (
                    asset_id TEXT PRIMARY KEY,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    sector TEXT,
                    industry TEXT,
                    industry_group TEXT,
                    theme TEXT,
                    market TEXT,
                    region TEXT,
                    country TEXT,
                    exchange TEXT,
                    issuer TEXT,
                    brand TEXT,
                    tags_json TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS asset_profiles (
                    asset_id TEXT PRIMARY KEY,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    description TEXT,
                    website TEXT,
                    employee_count INTEGER,
                    ipo_date TEXT,
                    market_cap REAL,
                    shares_outstanding REAL,
                    avg_volume REAL,
                    beta REAL,
                    dividend_yield REAL,
                    expense_ratio REAL,
                    fund_family TEXT,
                    category_name TEXT,
                    raw_profile_json TEXT,
                    source TEXT,
                    as_of_date TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS asset_status (
                    asset_id TEXT PRIMARY KEY,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    reference_status TEXT,
                    dataset_status TEXT,
                    dataset_count INTEGER DEFAULT 0,
                    successful_dataset_count INTEGER DEFAULT 0,
                    latest_dataset_id TEXT,
                    latest_source TEXT,
                    latest_download_at TEXT,
                    latest_success_at TEXT,
                    latest_failure_at TEXT,
                    latest_failure_reason TEXT,
                    coverage_start TEXT,
                    coverage_end TEXT,
                    freshness_status TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS provider_credentials (
                    provider TEXT PRIMARY KEY,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    credential_label TEXT,
                    api_key_encrypted TEXT,
                    account_email TEXT,
                    base_url TEXT,
                    is_active INTEGER DEFAULT 1,
                    last_validated_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS provider_dataset_catalog (
                    provider TEXT NOT NULL,
                    dataset_code TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    dataset_group TEXT,
                    display_name TEXT,
                    description TEXT,
                    is_enabled INTEGER DEFAULT 1,
                    is_structured INTEGER DEFAULT 1,
                    supports_history INTEGER DEFAULT 0,
                    supports_incremental_refresh INTEGER DEFAULT 0,
                    supports_point_in_time INTEGER DEFAULT 0,
                    last_documentation_reviewed_at TEXT,
                    PRIMARY KEY (provider, dataset_code)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS provider_raw_payloads (
                    raw_payload_id TEXT PRIMARY KEY,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    provider TEXT NOT NULL,
                    dataset_code TEXT NOT NULL,
                    asset_id TEXT,
                    provider_symbol TEXT,
                    provider_record_key TEXT,
                    as_of_date TEXT,
                    fiscal_year INTEGER,
                    fiscal_period TEXT,
                    payload_json TEXT NOT NULL,
                    payload_hash TEXT,
                    fetched_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS provider_field_inventory (
                    provider TEXT NOT NULL,
                    dataset_code TEXT NOT NULL,
                    field_path TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    field_name TEXT,
                    observed_type TEXT,
                    first_seen_at TEXT,
                    last_seen_at TEXT,
                    mapping_status TEXT,
                    mapped_table TEXT,
                    mapped_column TEXT,
                    notes TEXT,
                    PRIMARY KEY (provider, dataset_code, field_path)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS provider_sync_runs (
                    provider_sync_run_id TEXT PRIMARY KEY,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    provider TEXT NOT NULL,
                    dataset_code TEXT,
                    started_at TEXT,
                    finished_at TEXT,
                    status TEXT NOT NULL,
                    asset_count INTEGER,
                    record_count INTEGER,
                    new_field_count INTEGER,
                    error_summary TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS simfin_company_snapshots (
                    record_key TEXT PRIMARY KEY,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    asset_id TEXT NOT NULL,
                    simfin_ticker TEXT NOT NULL,
                    simfin_id INTEGER,
                    market TEXT,
                    company_name TEXT,
                    industry_id INTEGER,
                    raw_payload_id TEXT,
                    payload_json TEXT NOT NULL,
                    as_of_date TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS simfin_shareprice_snapshots (
                    record_key TEXT PRIMARY KEY,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    asset_id TEXT NOT NULL,
                    simfin_ticker TEXT NOT NULL,
                    market TEXT,
                    variant TEXT,
                    price_date TEXT,
                    raw_payload_id TEXT,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS simfin_income_statements (
                    record_key TEXT PRIMARY KEY,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    asset_id TEXT NOT NULL,
                    simfin_ticker TEXT NOT NULL,
                    market TEXT,
                    variant TEXT,
                    report_date TEXT,
                    publish_date TEXT,
                    restated_date TEXT,
                    fiscal_year INTEGER,
                    fiscal_period TEXT,
                    raw_payload_id TEXT,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS simfin_balance_sheets (
                    record_key TEXT PRIMARY KEY,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    asset_id TEXT NOT NULL,
                    simfin_ticker TEXT NOT NULL,
                    market TEXT,
                    variant TEXT,
                    report_date TEXT,
                    publish_date TEXT,
                    restated_date TEXT,
                    fiscal_year INTEGER,
                    fiscal_period TEXT,
                    raw_payload_id TEXT,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS simfin_cash_flow_statements (
                    record_key TEXT PRIMARY KEY,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    asset_id TEXT NOT NULL,
                    simfin_ticker TEXT NOT NULL,
                    market TEXT,
                    variant TEXT,
                    report_date TEXT,
                    publish_date TEXT,
                    restated_date TEXT,
                    fiscal_year INTEGER,
                    fiscal_period TEXT,
                    raw_payload_id TEXT,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS simfin_metric_snapshots (
                    record_key TEXT PRIMARY KEY,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    asset_id TEXT NOT NULL,
                    simfin_ticker TEXT NOT NULL,
                    market TEXT,
                    variant TEXT,
                    report_date TEXT,
                    publish_date TEXT,
                    restated_date TEXT,
                    fiscal_year INTEGER,
                    fiscal_period TEXT,
                    raw_payload_id TEXT,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS simfin_indicator_facts (
                    record_key TEXT PRIMARY KEY,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    asset_id TEXT NOT NULL,
                    simfin_ticker TEXT NOT NULL,
                    market TEXT,
                    dataset_code TEXT NOT NULL,
                    report_date TEXT,
                    metric_name TEXT NOT NULL,
                    metric_value REAL,
                    value_text TEXT,
                    unit TEXT,
                    raw_payload_id TEXT,
                    payload_json TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS defeatbeta_company_profiles (
                    record_key TEXT PRIMARY KEY,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    asset_id TEXT NOT NULL,
                    provider_symbol TEXT NOT NULL,
                    report_date TEXT,
                    raw_payload_id TEXT,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS defeatbeta_officers (
                    record_key TEXT PRIMARY KEY,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    asset_id TEXT NOT NULL,
                    provider_symbol TEXT NOT NULL,
                    report_date TEXT,
                    officer_name TEXT,
                    officer_title TEXT,
                    raw_payload_id TEXT,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS defeatbeta_earnings_calendars (
                    record_key TEXT PRIMARY KEY,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    asset_id TEXT NOT NULL,
                    provider_symbol TEXT NOT NULL,
                    report_date TEXT,
                    raw_payload_id TEXT,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS defeatbeta_price_history (
                    record_key TEXT PRIMARY KEY,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    asset_id TEXT NOT NULL,
                    provider_symbol TEXT NOT NULL,
                    price_date TEXT,
                    raw_payload_id TEXT,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS defeatbeta_financial_statements (
                    record_key TEXT PRIMARY KEY,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    asset_id TEXT NOT NULL,
                    provider_symbol TEXT NOT NULL,
                    finance_type TEXT,
                    period_type TEXT,
                    report_date TEXT,
                    raw_payload_id TEXT,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS defeatbeta_dividend_events (
                    record_key TEXT PRIMARY KEY,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    asset_id TEXT NOT NULL,
                    provider_symbol TEXT NOT NULL,
                    report_date TEXT,
                    raw_payload_id TEXT,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS defeatbeta_split_events (
                    record_key TEXT PRIMARY KEY,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    asset_id TEXT NOT NULL,
                    provider_symbol TEXT NOT NULL,
                    report_date TEXT,
                    raw_payload_id TEXT,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS defeatbeta_news_articles (
                    record_key TEXT PRIMARY KEY,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    asset_id TEXT NOT NULL,
                    provider_symbol TEXT NOT NULL,
                    report_date TEXT,
                    headline TEXT,
                    raw_payload_id TEXT,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS defeatbeta_transcripts (
                    record_key TEXT PRIMARY KEY,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    asset_id TEXT NOT NULL,
                    provider_symbol TEXT NOT NULL,
                    report_date TEXT,
                    raw_payload_id TEXT,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS defeatbeta_revenue_breakdowns (
                    record_key TEXT PRIMARY KEY,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    asset_id TEXT NOT NULL,
                    provider_symbol TEXT NOT NULL,
                    breakdown_type TEXT,
                    report_date TEXT,
                    raw_payload_id TEXT,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS defeatbeta_share_counts (
                    record_key TEXT PRIMARY KEY,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    asset_id TEXT NOT NULL,
                    provider_symbol TEXT NOT NULL,
                    report_date TEXT,
                    raw_payload_id TEXT,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS defeatbeta_sec_filings (
                    record_key TEXT PRIMARY KEY,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    asset_id TEXT NOT NULL,
                    provider_symbol TEXT NOT NULL,
                    report_date TEXT,
                    filing_type TEXT,
                    raw_payload_id TEXT,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_asset_master_symbol ON asset_master(symbol)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_asset_identifiers_provider_symbol ON asset_identifiers(provider, provider_symbol)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_asset_master_class ON asset_master(asset_class)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_asset_status_status ON asset_status(dataset_status)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_provider_raw_payloads_lookup ON provider_raw_payloads(provider, dataset_code, asset_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_provider_field_inventory_review ON provider_field_inventory(provider, dataset_code, mapping_status)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_provider_sync_runs_provider_started ON provider_sync_runs(provider, started_at DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_simfin_company_snapshots_asset ON simfin_company_snapshots(asset_id, updated_at DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_simfin_shareprice_snapshots_asset ON simfin_shareprice_snapshots(asset_id, price_date DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_simfin_income_statements_asset ON simfin_income_statements(asset_id, report_date DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_simfin_balance_sheets_asset ON simfin_balance_sheets(asset_id, report_date DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_simfin_cash_flow_statements_asset ON simfin_cash_flow_statements(asset_id, report_date DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_simfin_metric_snapshots_asset ON simfin_metric_snapshots(asset_id, report_date DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_simfin_indicator_facts_asset ON simfin_indicator_facts(asset_id, dataset_code, metric_name)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_defeatbeta_company_profiles_asset ON defeatbeta_company_profiles(asset_id, updated_at DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_defeatbeta_officers_asset ON defeatbeta_officers(asset_id, report_date DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_defeatbeta_earnings_calendars_asset ON defeatbeta_earnings_calendars(asset_id, report_date DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_defeatbeta_price_history_asset ON defeatbeta_price_history(asset_id, price_date DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_defeatbeta_financial_statements_asset ON defeatbeta_financial_statements(asset_id, report_date DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_defeatbeta_news_articles_asset ON defeatbeta_news_articles(asset_id, report_date DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_defeatbeta_transcripts_asset ON defeatbeta_transcripts(asset_id, report_date DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_defeatbeta_revenue_breakdowns_asset ON defeatbeta_revenue_breakdowns(asset_id, report_date DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_defeatbeta_sec_filings_asset ON defeatbeta_sec_filings(asset_id, report_date DESC)"
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
                    last_task_id TEXT,
                    quality_state TEXT,
                    quality_expected_interval TEXT,
                    suspicious_gap_count INTEGER DEFAULT 0,
                    max_suspicious_gap TEXT,
                    suspicious_gap_ranges_json TEXT,
                    repair_request_start TEXT,
                    repair_request_end TEXT,
                    quality_analyzed_at TEXT
                )
                """
            )
            self._ensure_column(conn, "acquisition_datasets", "quality_state", "TEXT")
            self._ensure_column(conn, "acquisition_datasets", "quality_expected_interval", "TEXT")
            self._ensure_column(conn, "acquisition_datasets", "suspicious_gap_count", "INTEGER", default="0")
            self._ensure_column(conn, "acquisition_datasets", "max_suspicious_gap", "TEXT")
            self._ensure_column(conn, "acquisition_datasets", "suspicious_gap_ranges_json", "TEXT")
            self._ensure_column(conn, "acquisition_datasets", "repair_request_start", "TEXT")
            self._ensure_column(conn, "acquisition_datasets", "repair_request_end", "TEXT")
            self._ensure_column(conn, "acquisition_datasets", "quality_analyzed_at", "TEXT")
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
                "CREATE INDEX IF NOT EXISTS idx_acquisition_datasets_symbol_source_resolution ON acquisition_datasets(symbol, source, resolution, history_window)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_acquisition_datasets_last_attempt ON acquisition_datasets(last_download_attempt_at DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_acquisition_datasets_last_ingest ON acquisition_datasets(last_ingest_at DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_acquisition_runs_task_started ON acquisition_runs(task_id, started_at DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_acquisition_runs_universe_started ON acquisition_runs(universe_id, started_at DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_acquisition_runs_finished ON acquisition_runs(finished_at DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_acquisition_attempts_run_seq ON acquisition_attempts(acquisition_run_id, seq DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_acquisition_attempts_dataset_finished ON acquisition_attempts(dataset_id, finished_at DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_acquisition_attempts_symbol_finished ON acquisition_attempts(symbol, finished_at DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_acquisition_attempts_task_finished ON acquisition_attempts(task_id, finished_at DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_acquisition_attempts_universe_finished ON acquisition_attempts(universe_id, finished_at DESC)"
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
                CREATE TABLE IF NOT EXISTS deployment_targets (
                    target_id TEXT PRIMARY KEY,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    name TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    broker_scope TEXT NOT NULL,
                    transport_mode TEXT NOT NULL,
                    base_url TEXT,
                    webhook_path TEXT,
                    status_path TEXT,
                    dashboard_path TEXT,
                    logs_path TEXT,
                    project_root TEXT,
                    db_path TEXT,
                    log_db_path TEXT,
                    secret_ref TEXT,
                    is_active INTEGER DEFAULT 1
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS manual_deployment_definitions (
                    manual_definition_id TEXT PRIMARY KEY,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    deployment_kind TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    strategy_version TEXT,
                    dataset_id TEXT,
                    symbol TEXT,
                    dataset_scope_json TEXT NOT NULL,
                    timeframe TEXT,
                    params_json TEXT NOT NULL,
                    structure_json TEXT NOT NULL,
                    target_id TEXT,
                    mode TEXT,
                    sizing_json TEXT NOT NULL,
                    notes TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS deployments (
                    deployment_id TEXT PRIMARY KEY,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    parent_deployment_id TEXT,
                    deployment_kind TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    candidate_id TEXT,
                    strategy TEXT NOT NULL,
                    strategy_version TEXT,
                    dataset_id TEXT,
                    symbol TEXT,
                    timeframe TEXT,
                    params_json TEXT NOT NULL,
                    structure_json TEXT NOT NULL,
                    validation_refs_json TEXT NOT NULL,
                    target_id TEXT,
                    mode TEXT,
                    sizing_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    status_reason TEXT,
                    last_signal_at TEXT,
                    last_sync_at TEXT,
                    last_error_at TEXT,
                    notes TEXT,
                    armed_at TEXT,
                    started_at TEXT,
                    stopped_at TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS deployment_child_links (
                    parent_deployment_id TEXT NOT NULL,
                    child_deployment_id TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    child_role TEXT,
                    dataset_id TEXT,
                    symbol TEXT,
                    strategy_block_id TEXT,
                    PRIMARY KEY (parent_deployment_id, child_deployment_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS deployment_metric_snapshots (
                    deployment_id TEXT NOT NULL,
                    snapshot_ts TEXT NOT NULL,
                    realized_pnl REAL,
                    open_pnl REAL,
                    trade_count INTEGER,
                    win_count INTEGER,
                    loss_count INTEGER,
                    win_rate REAL,
                    profit_factor REAL,
                    sharpe REAL,
                    current_position_json TEXT NOT NULL,
                    health_json TEXT NOT NULL,
                    PRIMARY KEY (deployment_id, snapshot_ts)
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
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_task_runs_task_started ON task_runs(task_id, started_at DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_task_runs_finished ON task_runs(finished_at DESC)"
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
        with self.connect() as conn:
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

    def save_provider_credential(
        self,
        provider: str,
        *,
        credential_label: str | None = None,
        api_key: str | None = None,
        account_email: str | None = None,
        base_url: str | None = None,
        is_active: bool = True,
        last_validated_at: str | None = None,
    ) -> None:
        normalized_provider = str(provider or "").strip().lower()
        if not normalized_provider:
            raise ValueError("Provider is required.")
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO provider_credentials (
                    provider, created_at, updated_at, credential_label, api_key_encrypted,
                    account_email, base_url, is_active, last_validated_at
                )
                VALUES (?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(provider) DO UPDATE SET
                    updated_at=CURRENT_TIMESTAMP,
                    credential_label=excluded.credential_label,
                    api_key_encrypted=excluded.api_key_encrypted,
                    account_email=excluded.account_email,
                    base_url=excluded.base_url,
                    is_active=excluded.is_active,
                    last_validated_at=excluded.last_validated_at
                """,
                (
                    normalized_provider,
                    str(credential_label or "").strip() or None,
                    str(api_key or "").strip() or None,
                    str(account_email or "").strip() or None,
                    str(base_url or "").strip() or None,
                    1 if is_active else 0,
                    str(last_validated_at or "").strip() or None,
                ),
            )

    def load_provider_credential(self, provider: str) -> dict[str, object]:
        normalized_provider = str(provider or "").strip().lower()
        if not normalized_provider or not self.db_path.exists():
            return {}
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT
                    provider,
                    credential_label,
                    api_key_encrypted,
                    account_email,
                    base_url,
                    is_active,
                    last_validated_at,
                    updated_at
                FROM provider_credentials
                WHERE provider=?
                """,
                (normalized_provider,),
            ).fetchone()
        if not row:
            return {}
        return {
            "provider": str(row[0] or ""),
            "credential_label": str(row[1] or ""),
            "api_key": str(row[2] or ""),
            "account_email": str(row[3] or ""),
            "base_url": str(row[4] or ""),
            "is_active": bool(int(row[5] or 0)),
            "last_validated_at": row[6],
            "updated_at": row[7],
        }

    def start_provider_sync_run(self, provider: str, dataset_code: str | None = None) -> str:
        normalized_provider = str(provider or "").strip().lower()
        if not normalized_provider:
            raise ValueError("Provider is required.")
        provider_sync_run_id = f"sync_{uuid.uuid4().hex}"
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO provider_sync_runs (
                    provider_sync_run_id, provider, dataset_code, started_at, status
                )
                VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?)
                """,
                (
                    provider_sync_run_id,
                    normalized_provider,
                    str(dataset_code or "").strip() or None,
                    "running",
                ),
            )
        return provider_sync_run_id

    def finish_provider_sync_run(
        self,
        provider_sync_run_id: str,
        *,
        status: str,
        asset_count: int | None = None,
        record_count: int | None = None,
        new_field_count: int | None = None,
        error_summary: str | None = None,
    ) -> None:
        if not str(provider_sync_run_id or "").strip():
            raise ValueError("provider_sync_run_id is required.")
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE provider_sync_runs
                SET finished_at=CURRENT_TIMESTAMP,
                    status=?,
                    asset_count=?,
                    record_count=?,
                    new_field_count=?,
                    error_summary=?
                WHERE provider_sync_run_id=?
                """,
                (
                    str(status or "").strip() or "completed",
                    None if asset_count is None else int(asset_count),
                    None if record_count is None else int(record_count),
                    None if new_field_count is None else int(new_field_count),
                    str(error_summary or "").strip() or None,
                    str(provider_sync_run_id).strip(),
                ),
            )

    # -- asset information --------------------------------------------------
    @staticmethod
    def _normalize_asset_symbol(symbol: str | None) -> str:
        return str(symbol or "").strip().upper()

    @classmethod
    def _asset_id_from_symbol(cls, symbol: str | None) -> str:
        normalized = cls._normalize_asset_symbol(symbol)
        if not normalized:
            raise ValueError("Symbol is required to build an asset id.")
        return f"asset_{sha1(normalized.encode('utf-8')).hexdigest()[:16]}"

    def upsert_asset_master(
        self,
        *,
        symbol: str,
        display_symbol: str | None = None,
        name: str | None = None,
        asset_class: str | None = None,
        security_type: str | None = None,
        exchange: str | None = None,
        mic: str | None = None,
        country: str | None = None,
        currency: str | None = None,
        is_active: bool | None = None,
        is_delisted: bool | None = None,
        is_tradable: bool | None = None,
    ) -> str:
        normalized_symbol = self._normalize_asset_symbol(symbol)
        if not normalized_symbol:
            raise ValueError("Symbol is required.")
        asset_id = self._asset_id_from_symbol(normalized_symbol)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO asset_master (
                    asset_id, symbol, display_symbol, name, asset_class, security_type,
                    exchange, mic, country, currency, is_active, is_delisted, is_tradable,
                    first_seen_at, last_seen_at, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(asset_id) DO UPDATE SET
                    symbol=excluded.symbol,
                    display_symbol=COALESCE(NULLIF(asset_master.display_symbol, ''), excluded.display_symbol),
                    name=COALESCE(NULLIF(asset_master.name, ''), excluded.name),
                    asset_class=COALESCE(NULLIF(asset_master.asset_class, ''), excluded.asset_class),
                    security_type=COALESCE(NULLIF(asset_master.security_type, ''), excluded.security_type),
                    exchange=COALESCE(NULLIF(asset_master.exchange, ''), excluded.exchange),
                    mic=COALESCE(NULLIF(asset_master.mic, ''), excluded.mic),
                    country=COALESCE(NULLIF(asset_master.country, ''), excluded.country),
                    currency=COALESCE(NULLIF(asset_master.currency, ''), excluded.currency),
                    is_active=COALESCE(excluded.is_active, asset_master.is_active),
                    is_delisted=COALESCE(excluded.is_delisted, asset_master.is_delisted),
                    is_tradable=COALESCE(excluded.is_tradable, asset_master.is_tradable),
                    last_seen_at=CURRENT_TIMESTAMP,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    asset_id,
                    normalized_symbol,
                    str(display_symbol or "").strip() or normalized_symbol,
                    str(name or "").strip() or normalized_symbol,
                    str(asset_class or "").strip() or None,
                    str(security_type or "").strip() or None,
                    str(exchange or "").strip() or None,
                    str(mic or "").strip() or None,
                    str(country or "").strip() or None,
                    str(currency or "").strip() or None,
                    None if is_active is None else (1 if is_active else 0),
                    None if is_delisted is None else (1 if is_delisted else 0),
                    None if is_tradable is None else (1 if is_tradable else 0),
                ),
            )
        return asset_id

    def upsert_asset_status(
        self,
        *,
        asset_id: str,
        reference_status: str | None = None,
        dataset_status: str | None = None,
        dataset_count: int | None = None,
        successful_dataset_count: int | None = None,
        latest_dataset_id: str | None = None,
        latest_source: str | None = None,
        latest_download_at: str | None = None,
        latest_success_at: str | None = None,
        latest_failure_at: str | None = None,
        latest_failure_reason: str | None = None,
        coverage_start: str | None = None,
        coverage_end: str | None = None,
        freshness_status: str | None = None,
    ) -> None:
        if not str(asset_id or "").strip():
            raise ValueError("asset_id is required.")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO asset_status (
                    asset_id, reference_status, dataset_status, dataset_count, successful_dataset_count,
                    latest_dataset_id, latest_source, latest_download_at, latest_success_at,
                    latest_failure_at, latest_failure_reason, coverage_start, coverage_end,
                    freshness_status, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(asset_id) DO UPDATE SET
                    reference_status=excluded.reference_status,
                    dataset_status=excluded.dataset_status,
                    dataset_count=excluded.dataset_count,
                    successful_dataset_count=excluded.successful_dataset_count,
                    latest_dataset_id=excluded.latest_dataset_id,
                    latest_source=excluded.latest_source,
                    latest_download_at=excluded.latest_download_at,
                    latest_success_at=excluded.latest_success_at,
                    latest_failure_at=excluded.latest_failure_at,
                    latest_failure_reason=excluded.latest_failure_reason,
                    coverage_start=excluded.coverage_start,
                    coverage_end=excluded.coverage_end,
                    freshness_status=excluded.freshness_status,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    str(asset_id).strip(),
                    str(reference_status or "").strip() or None,
                    str(dataset_status or "").strip() or None,
                    None if dataset_count is None else int(dataset_count),
                    None if successful_dataset_count is None else int(successful_dataset_count),
                    str(latest_dataset_id or "").strip() or None,
                    str(latest_source or "").strip() or None,
                    str(latest_download_at or "").strip() or None,
                    str(latest_success_at or "").strip() or None,
                    str(latest_failure_at or "").strip() or None,
                    str(latest_failure_reason or "").strip() or None,
                    str(coverage_start or "").strip() or None,
                    str(coverage_end or "").strip() or None,
                    str(freshness_status or "").strip() or None,
                ),
            )

    def bootstrap_assets_from_acquisition_datasets(self) -> int:
        if not self.db_path.exists():
            return 0

        def _latest_timestamp(values: Sequence[str | None]) -> str | None:
            valid = [str(value).strip() for value in list(values or ()) if str(value or "").strip()]
            if not valid:
                return None
            parsed = pd.to_datetime(pd.Series(valid), utc=True, errors="coerce")
            non_missing = parsed.dropna()
            if non_missing.empty:
                return max(valid)
            return non_missing.max().isoformat()

        def _earliest_timestamp(values: Sequence[str | None]) -> str | None:
            valid = [str(value).strip() for value in list(values or ()) if str(value or "").strip()]
            if not valid:
                return None
            parsed = pd.to_datetime(pd.Series(valid), utc=True, errors="coerce")
            non_missing = parsed.dropna()
            if non_missing.empty:
                return min(valid)
            return non_missing.min().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT
                    dataset_id,
                    symbol,
                    source,
                    coverage_start,
                    coverage_end,
                    ingested,
                    last_download_attempt_at,
                    last_download_success_at,
                    last_status,
                    last_error
                FROM acquisition_datasets
                WHERE TRIM(COALESCE(symbol, '')) != ''
                """
            ).fetchall()

        grouped: dict[str, list[tuple]] = {}
        for row in rows:
            normalized_symbol = self._normalize_asset_symbol(row[1])
            if not normalized_symbol:
                continue
            grouped.setdefault(normalized_symbol, []).append(row)

        touched = 0
        asset_rows: list[tuple] = []
        status_rows: list[tuple] = []
        for normalized_symbol, symbol_rows in grouped.items():
            asset_id = self._asset_id_from_symbol(normalized_symbol)
            dataset_count = len(symbol_rows)
            successful_rows = [row for row in symbol_rows if int(row[5] or 0) == 1 or str(row[7] or "").strip()]
            error_rows = [
                row
                for row in symbol_rows
                if str(row[8] or "").strip().lower() in _ACQUISITION_ERROR_STATUSES or str(row[9] or "").strip()
            ]
            latest_attempt_row = max(
                symbol_rows,
                key=lambda row: str(row[6] or row[7] or ""),
                default=symbol_rows[0],
            )
            latest_success_row = max(
                successful_rows,
                key=lambda row: str(row[7] or ""),
                default=latest_attempt_row,
            )
            latest_failure_row = max(
                error_rows,
                key=lambda row: str(row[6] or row[7] or ""),
                default=None,
            )

            if successful_rows:
                dataset_status = "Ready"
            elif error_rows:
                dataset_status = "Error"
            else:
                dataset_status = "Tracked"

            asset_rows.append(
                (
                    asset_id,
                    normalized_symbol,
                    normalized_symbol,
                    normalized_symbol,
                )
            )
            status_rows.append(
                (
                    asset_id,
                    "Bootstrapped",
                    dataset_status,
                    int(dataset_count),
                    int(len(successful_rows)),
                    str((latest_success_row or latest_attempt_row)[0] or "").strip() or None,
                    str((latest_success_row or latest_attempt_row)[2] or "").strip() or None,
                    _latest_timestamp([row[6] for row in symbol_rows]),
                    _latest_timestamp([row[7] for row in successful_rows]),
                    _latest_timestamp([latest_failure_row[6] if latest_failure_row else None]),
                    str((latest_failure_row[9] if latest_failure_row else "") or "").strip() or None,
                    _earliest_timestamp([row[3] for row in symbol_rows]),
                    _latest_timestamp([row[4] for row in symbol_rows]),
                )
            )
            touched += 1
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT INTO asset_master (
                    asset_id, symbol, display_symbol, name, first_seen_at, last_seen_at, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(asset_id) DO UPDATE SET
                    symbol=excluded.symbol,
                    display_symbol=COALESCE(NULLIF(asset_master.display_symbol, ''), excluded.display_symbol),
                    name=COALESCE(NULLIF(asset_master.name, ''), excluded.name),
                    last_seen_at=CURRENT_TIMESTAMP,
                    updated_at=CURRENT_TIMESTAMP
                """,
                asset_rows,
            )
            conn.executemany(
                """
                INSERT INTO asset_status (
                    asset_id, reference_status, dataset_status, dataset_count, successful_dataset_count,
                    latest_dataset_id, latest_source, latest_download_at, latest_success_at,
                    latest_failure_at, latest_failure_reason, coverage_start, coverage_end, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(asset_id) DO UPDATE SET
                    reference_status=excluded.reference_status,
                    dataset_status=excluded.dataset_status,
                    dataset_count=excluded.dataset_count,
                    successful_dataset_count=excluded.successful_dataset_count,
                    latest_dataset_id=excluded.latest_dataset_id,
                    latest_source=excluded.latest_source,
                    latest_download_at=excluded.latest_download_at,
                    latest_success_at=excluded.latest_success_at,
                    latest_failure_at=excluded.latest_failure_at,
                    latest_failure_reason=excluded.latest_failure_reason,
                    coverage_start=excluded.coverage_start,
                    coverage_end=excluded.coverage_end,
                    updated_at=CURRENT_TIMESTAMP
                """,
                status_rows,
            )
        return touched

    def load_asset_catalog(self) -> list[AssetCatalogRecord]:
        if not self.db_path.exists():
            return []
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    m.asset_id,
                    m.symbol,
                    m.display_symbol,
                    m.name,
                    m.asset_class,
                    m.security_type,
                    COALESCE(c.exchange, m.exchange),
                    COALESCE(c.country, m.country),
                    m.currency,
                    c.sector,
                    c.industry,
                    s.dataset_status,
                    COALESCE(s.dataset_count, 0),
                    COALESCE(s.successful_dataset_count, 0),
                    s.latest_dataset_id,
                    s.latest_source,
                    s.latest_download_at,
                    s.latest_success_at,
                    s.latest_failure_at,
                    s.latest_failure_reason,
                    s.coverage_start,
                    s.coverage_end,
                    s.freshness_status,
                    m.first_seen_at,
                    m.last_seen_at,
                    m.created_at,
                    COALESCE(s.updated_at, m.updated_at)
                FROM asset_master m
                LEFT JOIN asset_classifications c ON c.asset_id = m.asset_id
                LEFT JOIN asset_status s ON s.asset_id = m.asset_id
                ORDER BY LOWER(COALESCE(m.symbol, '')) ASC, LOWER(COALESCE(m.name, '')) ASC
                """
            ).fetchall()
        return [
            AssetCatalogRecord(
                asset_id=str(row[0]),
                symbol=str(row[1] or ""),
                display_symbol=row[2],
                name=row[3],
                asset_class=row[4],
                security_type=row[5],
                exchange=row[6],
                country=row[7],
                currency=row[8],
                sector=row[9],
                industry=row[10],
                dataset_status=row[11],
                dataset_count=int(row[12] or 0),
                successful_dataset_count=int(row[13] or 0),
                latest_dataset_id=row[14],
                latest_source=row[15],
                latest_download_at=row[16],
                latest_success_at=row[17],
                latest_failure_at=row[18],
                latest_failure_reason=row[19],
                coverage_start=row[20],
                coverage_end=row[21],
                freshness_status=row[22],
                first_seen_at=row[23],
                last_seen_at=row[24],
                created_at=row[25],
                updated_at=row[26],
            )
            for row in rows
        ]

    def load_asset_identifiers(self, asset_id: str) -> dict[str, str]:
        if not str(asset_id or "").strip() or not self.db_path.exists():
            return {}
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT
                    provider,
                    provider_symbol,
                    exchange_symbol,
                    conid,
                    isin,
                    cusip,
                    figi,
                    composite_figi,
                    shareclass_figi,
                    composite_key
                FROM asset_identifiers
                WHERE asset_id=?
                ORDER BY CASE WHEN provider='financedatabase' THEN 0 ELSE 1 END, updated_at DESC
                LIMIT 1
                """,
                (str(asset_id).strip(),),
            ).fetchone()
        if not row:
            return {}
        return {
            "provider": str(row[0] or ""),
            "provider_symbol": str(row[1] or ""),
            "exchange_symbol": str(row[2] or ""),
            "conid": str(row[3] or ""),
            "isin": str(row[4] or ""),
            "cusip": str(row[5] or ""),
            "figi": str(row[6] or ""),
            "composite_figi": str(row[7] or ""),
            "shareclass_figi": str(row[8] or ""),
            "composite_key": str(row[9] or ""),
        }

    def load_asset_provider_payload_summary(self, asset_id: str) -> list[dict[str, object]]:
        if not str(asset_id or "").strip() or not self.db_path.exists():
            return []
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    provider,
                    dataset_code,
                    COUNT(*) AS record_count,
                    MAX(COALESCE(fetched_at, created_at)) AS latest_fetched_at
                FROM provider_raw_payloads
                WHERE asset_id=?
                GROUP BY provider, dataset_code
                ORDER BY LOWER(provider) ASC, LOWER(dataset_code) ASC
                """,
                (str(asset_id).strip(),),
            ).fetchall()
        return [
            {
                "provider": str(row[0] or ""),
                "dataset_code": str(row[1] or ""),
                "record_count": int(row[2] or 0),
                "latest_fetched_at": row[3],
            }
            for row in rows
        ]

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
        quality_snapshot: dict | None = None,
    ) -> None:
        existing = None
        with sqlite3.connect(self.db_path) as conn:
            existing = conn.execute(
                """
                SELECT
                    source, symbol, resolution, history_window, csv_path, parquet_path,
                    coverage_start, coverage_end, bar_count, ingested,
                    last_download_attempt_at, last_download_success_at, last_ingest_at,
                    last_status, last_error, last_run_id, last_task_id,
                    quality_state, quality_expected_interval, suspicious_gap_count,
                    max_suspicious_gap, suspicious_gap_ranges_json, repair_request_start,
                    repair_request_end, quality_analyzed_at
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
            quality_snapshot = dict(quality_snapshot or {})
            quality_state_val = (
                quality_snapshot.get("quality_state")
                if "quality_state" in quality_snapshot
                else (existing[17] if existing else None)
            )
            quality_expected_interval_val = (
                quality_snapshot.get("expected_interval")
                if "expected_interval" in quality_snapshot
                else (existing[18] if existing else None)
            )
            suspicious_gap_count_val = (
                int(quality_snapshot.get("suspicious_gap_count") or 0)
                if "suspicious_gap_count" in quality_snapshot
                else (existing[19] if existing else None)
            )
            max_suspicious_gap_val = (
                quality_snapshot.get("max_suspicious_gap")
                if "max_suspicious_gap" in quality_snapshot
                else (existing[20] if existing else None)
            )
            suspicious_gap_ranges_json_val = (
                json.dumps(list(quality_snapshot.get("suspicious_gap_ranges") or []))
                if "suspicious_gap_ranges" in quality_snapshot
                else (existing[21] if existing else None)
            )
            repair_request_start_val = (
                quality_snapshot.get("repair_request_start")
                if "repair_request_start" in quality_snapshot
                else (existing[22] if existing else None)
            )
            repair_request_end_val = (
                quality_snapshot.get("repair_request_end")
                if "repair_request_end" in quality_snapshot
                else (existing[23] if existing else None)
            )
            quality_analyzed_at_val = (
                quality_snapshot.get("quality_analyzed_at")
                if "quality_analyzed_at" in quality_snapshot
                else (existing[24] if existing else None)
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO acquisition_datasets
                (
                    dataset_id, created_at, updated_at, source, symbol, resolution, history_window,
                    csv_path, parquet_path, coverage_start, coverage_end, bar_count, ingested,
                    last_download_attempt_at, last_download_success_at, last_ingest_at,
                    last_status, last_error, last_run_id, last_task_id, quality_state,
                    quality_expected_interval, suspicious_gap_count, max_suspicious_gap,
                    suspicious_gap_ranges_json, repair_request_start, repair_request_end,
                    quality_analyzed_at
                )
                VALUES (
                    ?, COALESCE((SELECT created_at FROM acquisition_datasets WHERE dataset_id=?), CURRENT_TIMESTAMP),
                    CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
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
                    quality_state_val,
                    quality_expected_interval_val,
                    suspicious_gap_count_val,
                    max_suspicious_gap_val,
                    suspicious_gap_ranges_json_val,
                    repair_request_start_val,
                    repair_request_end_val,
                    quality_analyzed_at_val,
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
        quality_snapshot: dict | None = None,
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
                quality_snapshot=quality_snapshot,
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
                    last_status, last_error, last_run_id, last_task_id,
                    quality_state, quality_expected_interval, suspicious_gap_count,
                    max_suspicious_gap, suspicious_gap_ranges_json, repair_request_start,
                    repair_request_end, quality_analyzed_at, created_at, updated_at
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
                quality_state=row[18],
                quality_expected_interval=row[19],
                suspicious_gap_count=row[20],
                max_suspicious_gap=row[21],
                suspicious_gap_ranges_json=row[22],
                repair_request_start=row[23],
                repair_request_end=row[24],
                quality_analyzed_at=row[25],
                created_at=row[26],
                updated_at=row[27],
            )
            for row in rows
        ]

    def load_acquisition_dataset(self, dataset_id: str) -> AcquisitionDatasetRecord | None:
        if not self.db_path.exists():
            return None
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT
                    dataset_id, source, symbol, resolution, history_window, csv_path, parquet_path,
                    coverage_start, coverage_end, bar_count, ingested,
                    last_download_attempt_at, last_download_success_at, last_ingest_at,
                    last_status, last_error, last_run_id, last_task_id,
                    quality_state, quality_expected_interval, suspicious_gap_count,
                    max_suspicious_gap, suspicious_gap_ranges_json, repair_request_start,
                    repair_request_end, quality_analyzed_at, created_at, updated_at
                FROM acquisition_datasets
                WHERE dataset_id=?
                """,
                (str(dataset_id),),
            ).fetchone()
        if row is None:
            return None
        return AcquisitionDatasetRecord(
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
            quality_state=row[18],
            quality_expected_interval=row[19],
            suspicious_gap_count=row[20],
            max_suspicious_gap=row[21],
            suspicious_gap_ranges_json=row[22],
            repair_request_start=row[23],
            repair_request_end=row[24],
            quality_analyzed_at=row[25],
            created_at=row[26],
            updated_at=row[27],
        )

    def update_acquisition_dataset_quality(self, dataset_id: str, quality_snapshot: dict | None) -> None:
        if not self.db_path.exists():
            return
        snapshot = dict(quality_snapshot or {})
        suspicious_gap_ranges_json_val = (
            json.dumps(list(snapshot.get("suspicious_gap_ranges") or []))
            if "suspicious_gap_ranges" in snapshot
            else None
        )
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE acquisition_datasets
                SET
                    quality_state=?,
                    quality_expected_interval=?,
                    suspicious_gap_count=?,
                    max_suspicious_gap=?,
                    suspicious_gap_ranges_json=?,
                    repair_request_start=?,
                    repair_request_end=?,
                    quality_analyzed_at=?,
                    updated_at=CURRENT_TIMESTAMP
                WHERE dataset_id=?
                """,
                (
                    snapshot.get("quality_state"),
                    snapshot.get("expected_interval"),
                    int(snapshot.get("suspicious_gap_count") or 0),
                    snapshot.get("max_suspicious_gap"),
                    suspicious_gap_ranges_json_val,
                    snapshot.get("repair_request_start"),
                    snapshot.get("repair_request_end"),
                    snapshot.get("quality_analyzed_at"),
                    str(dataset_id),
                ),
            )

    def load_acquisition_runs(
        self,
        limit: int | None = None,
        *,
        task_id: str | None = None,
        universe_id: str | None = None,
        offset: int = 0,
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
            if offset:
                query += " OFFSET ?"
                params.append(max(0, int(offset)))
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

    def count_acquisition_runs(
        self,
        *,
        task_id: str | None = None,
        universe_id: str | None = None,
    ) -> int:
        if not self.db_path.exists():
            return 0
        query = "SELECT COUNT(*) FROM acquisition_runs"
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
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(query, tuple(params)).fetchone()
        return int((row or [0])[0] or 0)

    def load_acquisition_attempts(
        self,
        *,
        acquisition_run_id: str | None = None,
        dataset_id: str | None = None,
        dataset_ids: Sequence[str] | None = None,
        symbol: str | None = None,
        symbols: Sequence[str] | None = None,
        task_id: str | None = None,
        universe_id: str | None = None,
        statuses: Sequence[str] | None = None,
        limit: int | None = None,
        offset: int = 0,
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
        if task_id:
            clauses.append("task_id=?")
            params.append(task_id)
        membership_clauses: list[str] = []
        if universe_id:
            membership_clauses.append("universe_id=?")
            params.append(universe_id)
        if dataset_id:
            membership_clauses.append("dataset_id=?")
            params.append(dataset_id)
        normalized_dataset_ids = [str(item).strip() for item in list(dataset_ids or []) if str(item).strip()]
        if normalized_dataset_ids:
            placeholders = ",".join(["?"] * len(normalized_dataset_ids))
            membership_clauses.append(f"dataset_id IN ({placeholders})")
            params.extend(normalized_dataset_ids)
        if symbol:
            membership_clauses.append("UPPER(symbol)=?")
            params.append(str(symbol).strip().upper())
        normalized_symbols = [str(item).strip().upper() for item in list(symbols or []) if str(item).strip()]
        if normalized_symbols:
            placeholders = ",".join(["?"] * len(normalized_symbols))
            membership_clauses.append(f"UPPER(symbol) IN ({placeholders})")
            params.extend(normalized_symbols)
        if membership_clauses:
            clauses.append("(" + " OR ".join(membership_clauses) + ")")
        normalized_statuses = [str(item).strip() for item in list(statuses or []) if str(item).strip()]
        if normalized_statuses:
            placeholders = ",".join(["?"] * len(normalized_statuses))
            clauses.append(f"status IN ({placeholders})")
            params.extend(normalized_statuses)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY COALESCE(finished_at, started_at) DESC, seq DESC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(int(limit))
            if offset:
                query += " OFFSET ?"
                params.append(max(0, int(offset)))
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

    def count_acquisition_attempts(
        self,
        *,
        acquisition_run_id: str | None = None,
        dataset_id: str | None = None,
        dataset_ids: Sequence[str] | None = None,
        symbol: str | None = None,
        symbols: Sequence[str] | None = None,
        task_id: str | None = None,
        universe_id: str | None = None,
        statuses: Sequence[str] | None = None,
    ) -> int:
        if not self.db_path.exists():
            return 0
        query = "SELECT COUNT(*) FROM acquisition_attempts"
        clauses: list[str] = []
        params: list[object] = []
        if acquisition_run_id:
            clauses.append("acquisition_run_id=?")
            params.append(acquisition_run_id)
        if task_id:
            clauses.append("task_id=?")
            params.append(task_id)
        membership_clauses: list[str] = []
        if universe_id:
            membership_clauses.append("universe_id=?")
            params.append(universe_id)
        if dataset_id:
            membership_clauses.append("dataset_id=?")
            params.append(dataset_id)
        normalized_dataset_ids = [str(item).strip() for item in list(dataset_ids or []) if str(item).strip()]
        if normalized_dataset_ids:
            placeholders = ",".join(["?"] * len(normalized_dataset_ids))
            membership_clauses.append(f"dataset_id IN ({placeholders})")
            params.extend(normalized_dataset_ids)
        if symbol:
            membership_clauses.append("UPPER(symbol)=?")
            params.append(str(symbol).strip().upper())
        normalized_symbols = [str(item).strip().upper() for item in list(symbols or []) if str(item).strip()]
        if normalized_symbols:
            placeholders = ",".join(["?"] * len(normalized_symbols))
            membership_clauses.append(f"UPPER(symbol) IN ({placeholders})")
            params.extend(normalized_symbols)
        if membership_clauses:
            clauses.append("(" + " OR ".join(membership_clauses) + ")")
        normalized_statuses = [str(item).strip() for item in list(statuses or []) if str(item).strip()]
        if normalized_statuses:
            placeholders = ",".join(["?"] * len(normalized_statuses))
            clauses.append(f"status IN ({placeholders})")
            params.extend(normalized_statuses)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(query, tuple(params)).fetchone()
        return int((row or [0])[0] or 0)

    def load_matching_attempt_dataset_ids(
        self,
        *,
        task_id: str | None = None,
        universe_id: str | None = None,
        dataset_ids: Sequence[str] | None = None,
        symbols: Sequence[str] | None = None,
    ) -> list[str]:
        if not self.db_path.exists():
            return []
        query = "SELECT DISTINCT dataset_id FROM acquisition_attempts"
        clauses: list[str] = []
        params: list[object] = []
        if task_id:
            clauses.append("task_id=?")
            params.append(task_id)
        membership_clauses: list[str] = []
        if universe_id:
            membership_clauses.append("universe_id=?")
            params.append(universe_id)
        normalized_dataset_ids = [str(item).strip() for item in list(dataset_ids or []) if str(item).strip()]
        if normalized_dataset_ids:
            placeholders = ",".join(["?"] * len(normalized_dataset_ids))
            membership_clauses.append(f"dataset_id IN ({placeholders})")
            params.extend(normalized_dataset_ids)
        normalized_symbols = [str(item).strip().upper() for item in list(symbols or []) if str(item).strip()]
        if normalized_symbols:
            placeholders = ",".join(["?"] * len(normalized_symbols))
            membership_clauses.append(f"UPPER(symbol) IN ({placeholders})")
            params.extend(normalized_symbols)
        if membership_clauses:
            clauses.append("(" + " OR ".join(membership_clauses) + ")")
        clauses.append("dataset_id IS NOT NULL")
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY dataset_id ASC"
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        return [str(row[0]) for row in rows if row and str(row[0] or "").strip()]

    def load_task_runs(
        self,
        *,
        task_id: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[TaskRunRecord]:
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
            if offset:
                query += " OFFSET ?"
                params.append(max(0, int(offset)))
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

    def count_task_runs(self, *, task_id: str | None = None) -> int:
        if not self.db_path.exists():
            return 0
        query = "SELECT COUNT(*) FROM task_runs"
        clauses: list[str] = []
        params: list[object] = []
        if task_id:
            clauses.append("task_id=?")
            params.append(task_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(query, tuple(params)).fetchone()
        return int((row or [0])[0] or 0)

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

    def save_deployment_target(
        self,
        *,
        target_id: str,
        name: str,
        mode: str,
        broker_scope: str,
        transport_mode: str,
        base_url: str = "",
        webhook_path: str = "",
        status_path: str = "",
        dashboard_path: str = "",
        logs_path: str = "",
        project_root: str = "",
        db_path: str = "",
        log_db_path: str = "",
        secret_ref: str = "",
        is_active: bool = True,
    ) -> str:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO deployment_targets
                (
                    target_id, name, mode, broker_scope, transport_mode, base_url, webhook_path,
                    status_path, dashboard_path, logs_path, project_root, db_path, log_db_path,
                    secret_ref, is_active
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(target_id) DO UPDATE SET
                    name=excluded.name,
                    mode=excluded.mode,
                    broker_scope=excluded.broker_scope,
                    transport_mode=excluded.transport_mode,
                    base_url=excluded.base_url,
                    webhook_path=excluded.webhook_path,
                    status_path=excluded.status_path,
                    dashboard_path=excluded.dashboard_path,
                    logs_path=excluded.logs_path,
                    project_root=excluded.project_root,
                    db_path=excluded.db_path,
                    log_db_path=excluded.log_db_path,
                    secret_ref=excluded.secret_ref,
                    is_active=excluded.is_active,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    str(target_id),
                    str(name),
                    str(mode),
                    str(broker_scope),
                    str(transport_mode),
                    str(base_url or ""),
                    str(webhook_path or ""),
                    str(status_path or ""),
                    str(dashboard_path or ""),
                    str(logs_path or ""),
                    str(project_root or ""),
                    str(db_path or ""),
                    str(log_db_path or ""),
                    str(secret_ref or ""),
                    1 if is_active else 0,
                ),
            )
        return str(target_id)

    def load_deployment_targets(self) -> list[DeploymentTargetRecord]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT
                    target_id, name, mode, broker_scope, transport_mode, base_url, webhook_path,
                    status_path, dashboard_path, logs_path, project_root, db_path, log_db_path,
                    secret_ref, is_active, created_at, updated_at
                FROM deployment_targets
                ORDER BY is_active DESC, name ASC, target_id ASC
                """
            ).fetchall()
        return [
            DeploymentTargetRecord(
                target_id=row[0],
                name=row[1],
                mode=row[2],
                broker_scope=row[3],
                transport_mode=row[4],
                base_url=row[5],
                webhook_path=row[6],
                status_path=row[7],
                dashboard_path=row[8],
                logs_path=row[9],
                project_root=row[10],
                db_path=row[11],
                log_db_path=row[12],
                secret_ref=row[13],
                is_active=bool(row[14]),
                created_at=row[15],
                updated_at=row[16],
            )
            for row in rows
        ]

    def save_manual_deployment_definition(
        self,
        *,
        manual_definition_id: str = "",
        deployment_kind: str,
        strategy: str,
        strategy_version: str = "",
        dataset_id: str = "",
        symbol: str = "",
        dataset_scope_json: Sequence[str] | str | None = None,
        timeframe: str = "",
        params_json: Dict | str | None = None,
        structure_json: Dict | str | None = None,
        target_id: str = "",
        mode: str = "",
        sizing_json: Dict | str | None = None,
        notes: str = "",
    ) -> str:
        definition_id = str(manual_definition_id or uuid.uuid4().hex)
        encoded_dataset_scope = (
            dataset_scope_json
            if isinstance(dataset_scope_json, str)
            else json.dumps(list(dataset_scope_json or []), sort_keys=True)
        )
        encoded_params = params_json if isinstance(params_json, str) else json.dumps(params_json or {}, sort_keys=True)
        encoded_structure = (
            structure_json if isinstance(structure_json, str) else json.dumps(structure_json or {}, sort_keys=True)
        )
        encoded_sizing = sizing_json if isinstance(sizing_json, str) else json.dumps(sizing_json or {}, sort_keys=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO manual_deployment_definitions
                (
                    manual_definition_id, deployment_kind, strategy, strategy_version, dataset_id, symbol,
                    dataset_scope_json, timeframe, params_json, structure_json, target_id, mode,
                    sizing_json, notes
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(manual_definition_id) DO UPDATE SET
                    deployment_kind=excluded.deployment_kind,
                    strategy=excluded.strategy,
                    strategy_version=excluded.strategy_version,
                    dataset_id=excluded.dataset_id,
                    symbol=excluded.symbol,
                    dataset_scope_json=excluded.dataset_scope_json,
                    timeframe=excluded.timeframe,
                    params_json=excluded.params_json,
                    structure_json=excluded.structure_json,
                    target_id=excluded.target_id,
                    mode=excluded.mode,
                    sizing_json=excluded.sizing_json,
                    notes=excluded.notes,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    definition_id,
                    str(deployment_kind),
                    str(strategy),
                    str(strategy_version or ""),
                    str(dataset_id or ""),
                    str(symbol or ""),
                    str(encoded_dataset_scope),
                    str(timeframe or ""),
                    str(encoded_params),
                    str(encoded_structure),
                    str(target_id or ""),
                    str(mode or ""),
                    str(encoded_sizing),
                    str(notes or ""),
                ),
            )
        return definition_id

    def load_manual_deployment_definitions(self) -> list[ManualDeploymentDefinitionRecord]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT
                    manual_definition_id, deployment_kind, strategy, strategy_version, dataset_id, symbol,
                    dataset_scope_json, timeframe, params_json, structure_json, target_id, mode,
                    sizing_json, notes, created_at, updated_at
                FROM manual_deployment_definitions
                ORDER BY updated_at DESC, created_at DESC, manual_definition_id DESC
                """
            ).fetchall()
        return [
            ManualDeploymentDefinitionRecord(
                manual_definition_id=row[0],
                deployment_kind=row[1],
                strategy=row[2],
                strategy_version=row[3],
                dataset_id=row[4],
                symbol=row[5],
                dataset_scope_json=row[6],
                timeframe=row[7],
                params_json=row[8],
                structure_json=row[9],
                target_id=row[10],
                mode=row[11],
                sizing_json=row[12],
                notes=row[13],
                created_at=row[14],
                updated_at=row[15],
            )
            for row in rows
        ]

    def save_deployment(
        self,
        *,
        deployment_id: str = "",
        parent_deployment_id: str = "",
        deployment_kind: str,
        source_type: str,
        source_id: str,
        candidate_id: str = "",
        strategy: str,
        strategy_version: str = "",
        dataset_id: str = "",
        symbol: str = "",
        timeframe: str = "",
        params_json: Dict | str | None = None,
        structure_json: Dict | str | None = None,
        validation_refs_json: Dict | str | None = None,
        target_id: str = "",
        mode: str = "",
        sizing_json: Dict | str | None = None,
        status: str = "draft",
        status_reason: str = "",
        last_signal_at: str = "",
        last_sync_at: str = "",
        last_error_at: str = "",
        notes: str = "",
        armed_at: str = "",
        started_at: str = "",
        stopped_at: str = "",
    ) -> str:
        resolved_id = str(deployment_id or uuid.uuid4().hex)
        encoded_params = params_json if isinstance(params_json, str) else json.dumps(params_json or {}, sort_keys=True)
        encoded_structure = (
            structure_json if isinstance(structure_json, str) else json.dumps(structure_json or {}, sort_keys=True)
        )
        encoded_validation = (
            validation_refs_json
            if isinstance(validation_refs_json, str)
            else json.dumps(validation_refs_json or {}, sort_keys=True)
        )
        encoded_sizing = sizing_json if isinstance(sizing_json, str) else json.dumps(sizing_json or {}, sort_keys=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO deployments
                (
                    deployment_id, parent_deployment_id, deployment_kind, source_type, source_id, candidate_id,
                    strategy, strategy_version, dataset_id, symbol, timeframe, params_json, structure_json,
                    validation_refs_json, target_id, mode, sizing_json, status, status_reason,
                    last_signal_at, last_sync_at, last_error_at, notes, armed_at, started_at, stopped_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(deployment_id) DO UPDATE SET
                    parent_deployment_id=excluded.parent_deployment_id,
                    deployment_kind=excluded.deployment_kind,
                    source_type=excluded.source_type,
                    source_id=excluded.source_id,
                    candidate_id=excluded.candidate_id,
                    strategy=excluded.strategy,
                    strategy_version=excluded.strategy_version,
                    dataset_id=excluded.dataset_id,
                    symbol=excluded.symbol,
                    timeframe=excluded.timeframe,
                    params_json=excluded.params_json,
                    structure_json=excluded.structure_json,
                    validation_refs_json=excluded.validation_refs_json,
                    target_id=excluded.target_id,
                    mode=excluded.mode,
                    sizing_json=excluded.sizing_json,
                    status=excluded.status,
                    status_reason=excluded.status_reason,
                    last_signal_at=excluded.last_signal_at,
                    last_sync_at=excluded.last_sync_at,
                    last_error_at=excluded.last_error_at,
                    notes=excluded.notes,
                    armed_at=excluded.armed_at,
                    started_at=excluded.started_at,
                    stopped_at=excluded.stopped_at,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    resolved_id,
                    str(parent_deployment_id or ""),
                    str(deployment_kind),
                    str(source_type),
                    str(source_id),
                    str(candidate_id or ""),
                    str(strategy),
                    str(strategy_version or ""),
                    str(dataset_id or ""),
                    str(symbol or ""),
                    str(timeframe or ""),
                    str(encoded_params),
                    str(encoded_structure),
                    str(encoded_validation),
                    str(target_id or ""),
                    str(mode or ""),
                    str(encoded_sizing),
                    str(status or "draft"),
                    str(status_reason or ""),
                    str(last_signal_at or ""),
                    str(last_sync_at or ""),
                    str(last_error_at or ""),
                    str(notes or ""),
                    str(armed_at or ""),
                    str(started_at or ""),
                    str(stopped_at or ""),
                ),
            )
        return resolved_id

    def update_deployment_status(
        self,
        deployment_id: str,
        *,
        status: str,
        status_reason: str = "",
        armed_at: str = "",
        started_at: str = "",
        stopped_at: str = "",
        last_error_at: str = "",
        last_sync_at: str = "",
    ) -> None:
        if not deployment_id:
            return
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE deployments
                SET
                    status=?,
                    status_reason=?,
                    armed_at=CASE WHEN ? <> '' THEN ? ELSE armed_at END,
                    started_at=CASE WHEN ? <> '' THEN ? ELSE started_at END,
                    stopped_at=CASE WHEN ? <> '' THEN ? ELSE stopped_at END,
                    last_error_at=CASE WHEN ? <> '' THEN ? ELSE last_error_at END,
                    last_sync_at=CASE WHEN ? <> '' THEN ? ELSE last_sync_at END,
                    updated_at=CURRENT_TIMESTAMP
                WHERE deployment_id=?
                """,
                (
                    str(status or ""),
                    str(status_reason or ""),
                    str(armed_at or ""),
                    str(armed_at or ""),
                    str(started_at or ""),
                    str(started_at or ""),
                    str(stopped_at or ""),
                    str(stopped_at or ""),
                    str(last_error_at or ""),
                    str(last_error_at or ""),
                    str(last_sync_at or ""),
                    str(last_sync_at or ""),
                    str(deployment_id),
                ),
            )

    def load_deployments(self) -> list[DeploymentRecord]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT
                    deployment_id, parent_deployment_id, deployment_kind, source_type, source_id, candidate_id,
                    strategy, strategy_version, dataset_id, symbol, timeframe, params_json, structure_json,
                    validation_refs_json, target_id, mode, sizing_json, status, status_reason, last_signal_at,
                    last_sync_at, last_error_at, notes, created_at, updated_at, armed_at, started_at, stopped_at
                FROM deployments
                ORDER BY updated_at DESC, created_at DESC, deployment_id DESC
                """
            ).fetchall()
        return [
            DeploymentRecord(
                deployment_id=row[0],
                parent_deployment_id=row[1],
                deployment_kind=row[2],
                source_type=row[3],
                source_id=row[4],
                candidate_id=row[5],
                strategy=row[6],
                strategy_version=row[7],
                dataset_id=row[8],
                symbol=row[9],
                timeframe=row[10],
                params_json=row[11],
                structure_json=row[12],
                validation_refs_json=row[13],
                target_id=row[14],
                mode=row[15],
                sizing_json=row[16],
                status=row[17],
                status_reason=row[18],
                last_signal_at=row[19],
                last_sync_at=row[20],
                last_error_at=row[21],
                notes=row[22],
                created_at=row[23],
                updated_at=row[24],
                armed_at=row[25],
                started_at=row[26],
                stopped_at=row[27],
            )
            for row in rows
        ]

    def save_deployment_child_link(
        self,
        *,
        parent_deployment_id: str,
        child_deployment_id: str,
        child_role: str = "",
        dataset_id: str = "",
        symbol: str = "",
        strategy_block_id: str = "",
    ) -> None:
        if not parent_deployment_id or not child_deployment_id:
            return
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO deployment_child_links
                (
                    parent_deployment_id, child_deployment_id, child_role, dataset_id, symbol, strategy_block_id
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(parent_deployment_id),
                    str(child_deployment_id),
                    str(child_role or ""),
                    str(dataset_id or ""),
                    str(symbol or ""),
                    str(strategy_block_id or ""),
                ),
            )

    def load_deployment_child_links(self, parent_deployment_id: str = "") -> list[DeploymentChildLinkRecord]:
        with sqlite3.connect(self.db_path) as conn:
            if parent_deployment_id:
                rows = conn.execute(
                    """
                    SELECT
                        parent_deployment_id, child_deployment_id, child_role, dataset_id, symbol, strategy_block_id, created_at
                    FROM deployment_child_links
                    WHERE parent_deployment_id=?
                    ORDER BY created_at ASC, child_deployment_id ASC
                    """,
                    (str(parent_deployment_id),),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT
                        parent_deployment_id, child_deployment_id, child_role, dataset_id, symbol, strategy_block_id, created_at
                    FROM deployment_child_links
                    ORDER BY created_at ASC, child_deployment_id ASC
                    """
                ).fetchall()
        return [
            DeploymentChildLinkRecord(
                parent_deployment_id=row[0],
                child_deployment_id=row[1],
                child_role=row[2],
                dataset_id=row[3],
                symbol=row[4],
                strategy_block_id=row[5],
                created_at=row[6],
            )
            for row in rows
        ]

    def save_deployment_metric_snapshot(
        self,
        *,
        deployment_id: str,
        snapshot_ts: str,
        realized_pnl: float | None = None,
        open_pnl: float | None = None,
        trade_count: int | None = None,
        win_count: int | None = None,
        loss_count: int | None = None,
        win_rate: float | None = None,
        profit_factor: float | None = None,
        sharpe: float | None = None,
        current_position_json: Dict | str | None = None,
        health_json: Dict | str | None = None,
    ) -> None:
        if not deployment_id or not snapshot_ts:
            return
        encoded_position = (
            current_position_json
            if isinstance(current_position_json, str)
            else json.dumps(current_position_json or {}, sort_keys=True)
        )
        encoded_health = health_json if isinstance(health_json, str) else json.dumps(health_json or {}, sort_keys=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO deployment_metric_snapshots
                (
                    deployment_id, snapshot_ts, realized_pnl, open_pnl, trade_count, win_count,
                    loss_count, win_rate, profit_factor, sharpe, current_position_json, health_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(deployment_id),
                    str(snapshot_ts),
                    float(realized_pnl) if realized_pnl is not None else None,
                    float(open_pnl) if open_pnl is not None else None,
                    int(trade_count) if trade_count is not None else None,
                    int(win_count) if win_count is not None else None,
                    int(loss_count) if loss_count is not None else None,
                    float(win_rate) if win_rate is not None else None,
                    float(profit_factor) if profit_factor is not None else None,
                    float(sharpe) if sharpe is not None else None,
                    str(encoded_position),
                    str(encoded_health),
                ),
            )

    def load_latest_deployment_metric_snapshots(self) -> list[DeploymentMetricSnapshotRecord]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT
                    s.deployment_id,
                    s.snapshot_ts,
                    s.realized_pnl,
                    s.open_pnl,
                    s.trade_count,
                    s.win_count,
                    s.loss_count,
                    s.win_rate,
                    s.profit_factor,
                    s.sharpe,
                    s.current_position_json,
                    s.health_json
                FROM deployment_metric_snapshots s
                JOIN (
                    SELECT deployment_id, MAX(snapshot_ts) AS snapshot_ts
                    FROM deployment_metric_snapshots
                    GROUP BY deployment_id
                ) latest
                  ON latest.deployment_id = s.deployment_id
                 AND latest.snapshot_ts = s.snapshot_ts
                ORDER BY s.snapshot_ts DESC, s.deployment_id ASC
                """
            ).fetchall()
        return [
            DeploymentMetricSnapshotRecord(
                deployment_id=row[0],
                snapshot_ts=row[1],
                realized_pnl=float(row[2]) if row[2] is not None else None,
                open_pnl=float(row[3]) if row[3] is not None else None,
                trade_count=int(row[4]) if row[4] is not None else None,
                win_count=int(row[5]) if row[5] is not None else None,
                loss_count=int(row[6]) if row[6] is not None else None,
                win_rate=float(row[7]) if row[7] is not None else None,
                profit_factor=float(row[8]) if row[8] is not None else None,
                sharpe=float(row[9]) if row[9] is not None else None,
                current_position_json=row[10],
                health_json=row[11],
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
