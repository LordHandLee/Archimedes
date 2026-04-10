"""
PyQt dashboard for the backtest engine, reusing the styling of dashboard.html.

Features:
- Loads run metrics and heatmap references from SQLite (ResultCatalog tables).
- Displays run table, summary metrics, and latest heatmap preview.
- Optional refresh button; no automatic grid search is triggered (UI is optional).
"""

from __future__ import annotations

import os
import sys

# Set BLAS threads upfront; default to (CPU count - 1) to use most cores.
_CPU_THREADS = max(1, (os.cpu_count() or 2) - 1)
os.environ["OPENBLAS_NUM_THREADS"] = str(_CPU_THREADS)
os.environ["OMP_NUM_THREADS"] = str(_CPU_THREADS)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(_CPU_THREADS)
os.environ["NUMEXPR_MAX_THREADS"] = str(_CPU_THREADS)

import json
import time
import urllib.request
import signal
import re
import sqlite3
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, MaxNLocator
from mplfinance.original_flavor import candlestick_ohlc
from PyQt6 import QtCore, QtGui, QtWidgets, sip

from backtest_engine import (
    ALLOCATION_OWNERSHIP_HYBRID,
    ALLOCATION_OWNERSHIP_PORTFOLIO,
    ALLOCATION_OWNERSHIP_STRATEGY,
    DEFAULT_ACQUISITION_PROVIDER,
    BatchExecutionBenchmark,
    BacktestConfig,
    BacktestEngine,
    DuckDBStore,
    EngineBatchComparisonSummary,
    EngineComparisonSummary,
    ExecutionMode,
    ExecutionOrchestrator,
    ExecutionRequest,
    GridSearch,
    GridSpec,
    IndependentAssetTarget,
    ROBUST_SCORE_VERSION,
    PortfolioExecutionAsset,
    PortfolioExecutionRequest,
    PortfolioExecutionStrategyBlock,
    PortfolioExecutionStrategyBlockAsset,
    PortfolioConstructionConfig,
    PortfolioAssetTarget,
    PortfolioStrategyBlockTarget,
    PortfolioStrategyBlockAssetTarget,
    RANKING_MODE_NONE,
    RANKING_MODE_SCORE_THRESHOLD,
    RANKING_MODE_TOP_N,
    RANKING_MODE_TOP_N_OVER_SCORE_THRESHOLD,
    REBALANCE_MODE_ON_CHANGE,
    REBALANCE_MODE_ON_CHANGE_OR_DRIFT,
    REBALANCE_MODE_ON_CHANGE_OR_PERIODIC,
    REBALANCE_MODE_ON_CHANGE_OR_PERIODIC_OR_DRIFT,
    ResultCatalog,
    SMACrossStrategy,
    InverseTurtleStrategy,
    ZScoreMeanReversionStrategy,
    UnsupportedExecutionModeError,
    WEIGHTING_MODE_EQUAL_SELECTED,
    WEIGHTING_MODE_PRESERVE,
    WEIGHTING_MODE_SCORE_PROPORTIONAL,
    ACQUISITION_ACTION_GAP_FILL_SECONDARY,
    ACQUISITION_ACTION_DOWNLOAD,
    ACQUISITION_ACTION_INGEST_EXISTING,
    ACQUISITION_ACTION_SKIP_FRESH,
    available_acquisition_providers,
    build_provider_fetch_command,
    build_download_csv_path,
    build_download_dataset_id,
    build_horizons,
    build_asset_distribution_frame,
    build_optimization_study_artifacts,
    build_portfolio_chart_data,
    build_portfolio_trades_log_frame,
    candidate_param_sets_from_records,
    compute_freshness_state,
    decide_acquisition_policy,
    extract_walk_forward_trade_returns,
    gap_fill_dataset_from_secondary,
    ingest_csv_to_store,
    get_acquisition_provider,
    ibapi_install_command,
    interactive_brokers_api_status,
    interactive_brokers_head_timestamp_status,
    ibapi_package_status,
    interactive_brokers_socket_status,
    load_csv_prices,
    MONTE_CARLO_MODE_BOOTSTRAP,
    MONTE_CARLO_MODE_RESHUFFLE,
    MONTE_CARLO_SOURCE_WALK_FORWARD,
    provider_display_name,
    portfolio_drawdown_frame,
    run_walk_forward_study,
    run_walk_forward_portfolio_study,
    run_monte_carlo_study,
    run_independent_asset_grid_search,
    run_vectorized_strategy_block_portfolio_search,
    run_vectorized_portfolio_grid_search,
    resolve_acquisition_source,
    portfolio_report_frame,
    portfolio_strategy_report_frame,
    summarize_portfolio_result,
    summarize_engine_batch,
    summarize_engine_runs,
    WalkForwardPortfolioAssetDefinition,
    WalkForwardPortfolioStrategyBlockAssetDefinition,
    WalkForwardPortfolioStrategyBlockDefinition,
    WALK_FORWARD_SOURCE_FIXED_PORTFOLIO,
    WALK_FORWARD_SOURCE_FULL_GRID,
    WALK_FORWARD_SOURCE_REDUCED_CANDIDATES,
)
from backtest_engine.chart_snapshot import ChartSnapshotExporter
from backtest_engine.magellan import MagellanClient, MagellanError
from backtest_engine.optimization import compute_robust_score
from backtest_engine.provider_config import (
    build_provider_runtime_environment,
    load_provider_secrets,
    load_provider_settings,
    provider_settings_status,
    save_provider_secrets,
    save_provider_settings,
)
from backtest_engine.reporting import plot_param_heatmap
from backtest_engine.sample_strategies import compute_zscore_mean_reversion_features


# --- Palette (mirrors dashboard.html) ---------------------------------------
PALETTE = {
    "bg": "#0b1220",
    "panel": "#101a2e",
    "panel2": "#0e1730",
    "text": "#e7eefc",
    "muted": "#9ab0d0",
    "grid": "#3a455d",
    "green": "#27d07d",
    "red": "#ff4d6d",
    "amber": "#ffcc66",
    "blue": "#4da3ff",
    "border": "#e7eefc",
}
NASDAQ_SYMBOLS_PATH = Path("data/all_listed_symbols.txt")
AUTOMATION_TASKS_PATH = Path("data/automation_tasks.json")
NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"
EXPECTED_2Y_1M_EQUITY_ROWS = int(2 * 252 * 16 * 60)
SCHEDULER_SCRIPT = Path("scripts") / "scheduler_service.py"
DOWNLOAD_LOG_DIR = Path("data") / "download_logs"
STUDY_MODE_INDEPENDENT = "independent"
STUDY_MODE_PORTFOLIO = "portfolio"
PORTFOLIO_ALLOC_EQUAL = "equal"
PORTFOLIO_ALLOC_RELATIVE = "relative"
PORTFOLIO_ALLOC_FIXED = "fixed"


def _safe_inspect_dataset_quality(dataset_id: str, resolution: str) -> dict:
    if not str(dataset_id or "").strip():
        return {
            "quality_state": "unknown",
            "suspicious_gap_count": 0,
            "max_suspicious_gap": None,
            "repair_request_start": None,
            "repair_request_end": None,
        }
    store = DuckDBStore()
    try:
        return store.inspect_dataset_quality(str(dataset_id), str(resolution or ""))
    except Exception:
        return {
            "quality_state": "unknown",
            "suspicious_gap_count": 0,
            "max_suspicious_gap": None,
            "repair_request_start": None,
            "repair_request_end": None,
        }
    finally:
        store.close()


def _format_dataset_quality_label(quality: dict) -> str:
    state = str(quality.get("quality_state") or "unknown")
    gap_count = int(quality.get("suspicious_gap_count") or 0)
    max_gap = str(quality.get("max_suspicious_gap") or "").strip()
    repair_start = str(quality.get("repair_request_start") or "").strip()
    repair_end = str(quality.get("repair_request_end") or "").strip()
    if state == "gappy":
        repair = (
            f", repair {repair_start} -> {repair_end}"
            if repair_start and repair_end
            else ""
        )
        if max_gap:
            return f"gappy ({gap_count} suspicious gap(s){repair}, max {max_gap})"
        return f"gappy ({gap_count} suspicious gap(s){repair})"
    if state == "gap_free":
        return "gap free"
    return "unknown"


def load_stylesheet() -> str:
    return f"""
    * {{
        color: {PALETTE['text']};
        font-family: 'SF Pro Text', 'Segoe UI', Arial, sans-serif;
        font-size: 13px;
    }}
    QMainWindow {{
        background-color: {PALETTE['bg']};
    }}
    QWidget#Panel {{
        background: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 rgba(16,26,46,0.95), stop:1 rgba(14,23,48,0.85));
        border: 1px solid {PALETTE['border']};
        border-radius: 12px;
    }}
    QLabel#Title {{
        font-size: 16px;
        font-weight: 700;
    }}
    QLabel#Sub {{
        color: {PALETTE['muted']};
        font-size: 12px;
    }}
    QMessageBox {{
        background-color: {PALETTE['panel']};
        color: {PALETTE['text']};
    }}
    QMessageBox QLabel {{
        color: {PALETTE['text']};
    }}
    QMessageBox QPushButton {{
        background: {PALETTE['panel2']};
        color: {PALETTE['text']};
        border: 1px solid {PALETTE['border']};
        border-radius: 8px;
        padding: 6px 10px;
    }}
    QMessageBox QTextEdit, QMessageBox QPlainTextEdit {{
        background: {PALETTE['panel2']};
        color: {PALETTE['text']};
        border: 1px solid {PALETTE['border']};
    }}
    QTabBar::tab {{
        color: {PALETTE['text']};
        background: rgba(255,255,255,.08);
        border: 1px solid {PALETTE['border']};
        border-bottom: none;
        border-radius: 8px 8px 0 0;
        padding: 8px 14px;
        min-width: 90px;
    }}
    QTabBar::tab:selected {{
        color: {PALETTE['text']};
        background: {PALETTE['panel']};
        border-color: {PALETTE['blue']};
        font-weight: 700;
    }}
    QTabWidget::pane {{
        border-top: 1px solid {PALETTE['border']};
        top: -1px;
        background: {PALETTE['panel']};
    }}
    QComboBox {{
        background: rgba(255,255,255,.08);
        color: {PALETTE['text']};
        border: 1px solid {PALETTE['border']};
        border-radius: 8px;
        padding: 6px 8px;
    }}
    QComboBox QAbstractItemView {{
        background: {PALETTE['panel']};
        color: {PALETTE['text']};
        selection-background-color: rgba(77,163,255,.25);
    }}
    QLineEdit, QTextEdit, QPlainTextEdit, QAbstractSpinBox {{
        background: rgba(255,255,255,.08);
        color: {PALETTE['text']};
        border: 1px solid {PALETTE['border']};
        border-radius: 8px;
        padding: 6px 8px;
        selection-background-color: rgba(77,163,255,.25);
        selection-color: {PALETTE['text']};
    }}
    QLineEdit[readOnly=\"true\"], QTextEdit[readOnly=\"true\"], QPlainTextEdit[readOnly=\"true\"] {{
        background: rgba(255,255,255,.06);
        color: {PALETTE['text']};
    }}
    QLineEdit[placeholderText], QTextEdit, QPlainTextEdit {{
        color: {PALETTE['text']};
    }}
    QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{
        background: rgba(255,255,255,.08);
        border: none;
        width: 18px;
    }}
    QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {{
        background: rgba(77,163,255,.18);
    }}
    QCheckBox {{
        spacing: 8px;
    }}
    QCheckBox::indicator {{
        width: 18px;
        height: 18px;
        border-radius: 4px;
        border: 1px solid {PALETTE['border']};
        background: rgba(255,255,255,.06);
    }}
    QCheckBox::indicator:hover {{
        border-color: {PALETTE['blue']};
        background: rgba(77,163,255,.12);
    }}
    QCheckBox::indicator:checked {{
        border-color: {PALETTE['blue']};
        background: {PALETTE['blue']};
    }}
    QCheckBox::indicator:unchecked {{
        background: rgba(255,255,255,.03);
    }}
    QCheckBox::indicator:disabled {{
        background: rgba(255,255,255,.03);
        border-color: rgba(154,176,208,.30);
    }}
    QPushButton {{
        background: rgba(0,0,0,.2);
        border: 1px solid {PALETTE['border']};
        border-radius: 8px;
        padding: 6px 10px;
    }}
    QPushButton:hover {{
        border-color: {PALETTE['blue']};
    }}
    QTableView, QTableWidget {{
        background-color: {PALETTE['panel2']};
        color: {PALETTE['text']};
        border: 1px solid rgba(231, 238, 252, 0.45);
        gridline-color: rgba(154, 176, 208, 0.22);
        alternate-background-color: rgba(255,255,255,.04);
        selection-background-color: rgba(77,163,255,.25);
        selection-color: {PALETTE['text']};
    }}
    QHeaderView::section {{
        background-color: rgba(16, 26, 46, 0.98);
        color: {PALETTE['muted']};
        font-size: 11px;
        font-weight: 700;
        border: none;
        border-right: 1px solid rgba(154, 176, 208, 0.18);
        border-bottom: 1px solid rgba(154, 176, 208, 0.28);
        padding: 6px 8px;
    }}
    QTableCornerButton::section {{
        background-color: rgba(16, 26, 46, 0.98);
        border: none;
        border-right: 1px solid rgba(154, 176, 208, 0.18);
        border-bottom: 1px solid rgba(154, 176, 208, 0.28);
    }}
    QScrollArea {{
        background: transparent;
        border: none;
    }}
    QScrollArea > QWidget > QWidget {{
        background: transparent;
    }}
    QScrollBar:vertical, QScrollBar:horizontal {{
        background: rgba(0,0,0,.2);
    }}
    QLineEdit {{
        background: rgba(0,0,0,.25);
        border: 1px solid {PALETTE['border']};
        border-radius: 8px;
        padding: 6px 8px;
        color: {PALETTE['text']};
    }}
    QLineEdit:focus {{
        border-color: {PALETTE['blue']};
    }}
    QScrollArea {{
        background: transparent;
        border: none;
    }}
    QScrollArea > QWidget > QWidget {{
        background: transparent;
    }}
    QScrollBar:vertical, QScrollBar:horizontal {{
        background: rgba(0,0,0,.2);
    }}
    QToolTip {{
        color: {PALETTE['text']};
        background-color: {PALETTE['panel']};
        border: 1px solid {PALETTE['border']};
        padding: 4px 6px;
        border-radius: 6px;
        font-size: 12px;
    }}
    """


# --- Data access ------------------------------------------------------------
@dataclass
class RunRow:
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
    metrics: dict
    run_started_at: str
    run_finished_at: str | None
    status: str
    requested_execution_mode: str | None
    resolved_execution_mode: str | None
    engine_impl: str | None
    engine_version: str | None
    fallback_reason: str | None


@dataclass
class BatchRow:
    batch_id: str
    strategy: str
    dataset_id: str
    params: str
    timeframes: str
    horizons: str
    run_total: int | None
    run_started_at: str | None
    run_finished_at: str | None
    status: str
    metrics: dict
    run_count: int
    finished_count: int


class CatalogReader:
    """Read runs and heatmaps from the existing SQLite catalog."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)

    def load_runs(self, batch_id: str | None = None) -> List[RunRow]:
        if not self.db_path.exists():
            return []
        with sqlite3.connect(self.db_path) as conn:
            if batch_id:
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
                    FROM runs WHERE batch_id=? ORDER BY created_at DESC
                    """,
                    (batch_id,),
                ).fetchall()
            else:
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
                    FROM runs ORDER BY created_at DESC
                    """
                ).fetchall()
        result = []
        for r in rows:
            result.append(
                RunRow(
                    run_id=r[0],
                    logical_run_id=r[1],
                    batch_id=r[2],
                    strategy=r[3],
                    params=r[4],
                    timeframe=r[5],
                    start=r[6],
                    end=r[7],
                    dataset_id=r[8],
                    starting_cash=r[9],
                    metrics=json.loads(r[10]) if r[10] else {},
                    run_started_at=r[11],
                    run_finished_at=r[12],
                    status=r[13] or "finished",
                    requested_execution_mode=r[14],
                    resolved_execution_mode=r[15],
                    engine_impl=r[16],
                    engine_version=r[17],
                    fallback_reason=r[18],
                )
            )
        return result

    def load_runs_for_logical_run_id(self, logical_run_id: str) -> List[RunRow]:
        if not logical_run_id or not self.db_path.exists():
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
        result = []
        for r in rows:
            result.append(
                RunRow(
                    run_id=r[0],
                    logical_run_id=r[1],
                    batch_id=r[2],
                    strategy=r[3],
                    params=r[4],
                    timeframe=r[5],
                    start=r[6],
                    end=r[7],
                    dataset_id=r[8],
                    starting_cash=r[9],
                    metrics=json.loads(r[10]) if r[10] else {},
                    run_started_at=r[11],
                    run_finished_at=r[12],
                    status=r[13] or "finished",
                    requested_execution_mode=r[14],
                    resolved_execution_mode=r[15],
                    engine_impl=r[16],
                    engine_version=r[17],
                    fallback_reason=r[18],
                )
            )
        return result

    def load_batches(self) -> List[BatchRow]:
        if not self.db_path.exists():
            return []
        with sqlite3.connect(self.db_path) as conn:
            batch_rows = conn.execute(
                """
                SELECT batch_id,strategy,dataset_id,params,timeframes,horizons,run_total,status,started_at,finished_at
                FROM batches ORDER BY created_at DESC
                """
            ).fetchall()
        runs = self.load_runs()
        runs_by_batch: Dict[str, List[RunRow]] = {}
        for r in runs:
            runs_by_batch.setdefault(r.batch_id or "ad-hoc", []).append(r)
        batches: List[BatchRow] = []
        for b in batch_rows:
            batch_id = b[0]
            params = json.loads(b[3]) if b[3] else {}
            timeframes = ", ".join(json.loads(b[4]) if b[4] else [])
            horizons = ", ".join(json.loads(b[5]) if b[5] else [])
            run_total = b[6]
            status = b[7] or "running"
            started_at = b[8]
            finished_at = b[9]
            runs = runs_by_batch.get(batch_id, [])
            metrics_list = [r.metrics for r in runs if r.metrics]
            best = max(metrics_list, key=lambda m: (m.get("sharpe", 0), m.get("total_return", 0)), default={})
            run_count = len(runs)
            finished_count = len([r for r in runs if r.status == "finished"])
            if run_total:
                if finished_count >= run_total:
                    status = "finished"
                else:
                    status = "running"
            else:
                if run_count:
                    if finished_count == run_count:
                        status = "finished"
                    elif any(r.status != "finished" for r in runs):
                        status = "running"
            batches.append(
                BatchRow(
                    batch_id=batch_id,
                    strategy=b[1],
                    dataset_id=b[2],
                    params=json.dumps(params),
                    timeframes=timeframes,
                    horizons=horizons,
                    run_total=run_total,
                    run_started_at=started_at,
                    run_finished_at=finished_at,
                    status=status if finished_count < run_count else "finished",
                    metrics=best,
                    run_count=run_count,
                    finished_count=finished_count,
                )
            )
        return batches

    def load_batch_benchmarks(self, batch_id: str) -> tuple[BatchExecutionBenchmark, ...]:
        if not batch_id:
            return ()
        records = ResultCatalog(self.db_path).load_batch_benchmarks(batch_id)
        benchmarks: list[BatchExecutionBenchmark] = []
        for record in records:
            try:
                requested_mode = ExecutionMode.from_value(record.requested_execution_mode)
                resolved_mode = ExecutionMode.from_value(record.resolved_execution_mode)
            except Exception:
                continue
            benchmarks.append(
                BatchExecutionBenchmark(
                    dataset_id=record.dataset_id,
                    strategy=record.strategy,
                    timeframe=record.timeframe,
                    requested_execution_mode=requested_mode,
                    resolved_execution_mode=resolved_mode,
                    engine_impl=record.engine_impl,
                    engine_version=record.engine_version,
                    bars=record.bars,
                    total_params=record.total_params,
                    cached_runs=record.cached_runs,
                    uncached_runs=record.uncached_runs,
                    duration_seconds=record.duration_seconds,
                    chunk_count=record.chunk_count,
                    chunk_sizes=record.chunk_sizes,
                    effective_param_batch_size=record.effective_param_batch_size,
                    prepared_context_reused=record.prepared_context_reused,
                )
            )
        return tuple(benchmarks)

    def load_heatmaps(self) -> pd.DataFrame:
        if not self.db_path.exists():
            return pd.DataFrame(columns=["heatmap_id", "description", "file_path"])
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql("SELECT heatmap_id, description, params, file_path, created_at FROM heatmaps ORDER BY created_at DESC", conn)
        return df

    def load_optimization_studies(self) -> pd.DataFrame:
        rc = ResultCatalog(self.db_path)
        rows = rc.load_optimization_studies()
        if not rows:
            return pd.DataFrame(
                columns=[
                    "study_id",
                    "batch_id",
                    "strategy",
                    "dataset_scope",
                    "param_names",
                    "timeframes",
                    "horizons",
                    "score_version",
                    "aggregate_count",
                    "created_at",
                ]
            )
        return pd.DataFrame(
            [
                {
                    "study_id": row.study_id,
                    "batch_id": row.batch_id,
                    "strategy": row.strategy,
                    "dataset_scope": list(row.dataset_scope),
                    "param_names": list(row.param_names),
                    "timeframes": list(row.timeframes),
                    "horizons": list(row.horizons),
                    "score_version": row.score_version,
                    "aggregate_count": row.aggregate_count,
                    "created_at": row.created_at,
                }
                for row in rows
            ]
        )

    def load_optimization_aggregates(self, study_id: str) -> pd.DataFrame:
        return self._expand_param_columns(ResultCatalog(self.db_path).load_optimization_aggregates(study_id))

    def load_optimization_asset_results(self, study_id: str) -> pd.DataFrame:
        return self._expand_param_columns(ResultCatalog(self.db_path).load_optimization_asset_results(study_id))

    def load_optimization_candidates(self, study_id: str) -> pd.DataFrame:
        rc = ResultCatalog(self.db_path)
        rows = rc.load_optimization_candidates(study_id)
        if not rows:
            return pd.DataFrame(
                columns=[
                    "candidate_id",
                    "study_id",
                    "timeframe",
                    "start",
                    "end",
                    "param_key",
                    "params_json",
                    "source_type",
                    "promotion_reason",
                    "status",
                    "metrics_json",
                    "asset_results_json",
                    "artifact_refs_json",
                    "notes",
                    "created_at",
                    "updated_at",
                ]
            )
        frame = pd.DataFrame(
            [
                {
                    "candidate_id": row.candidate_id,
                    "study_id": row.study_id,
                    "timeframe": row.timeframe,
                    "start": row.start,
                    "end": row.end,
                    "param_key": row.param_key,
                    "params_json": row.params_json,
                    "source_type": row.source_type,
                    "promotion_reason": row.promotion_reason,
                    "status": row.status,
                    "metrics_json": row.metrics_json,
                    "asset_results_json": row.asset_results_json,
                    "artifact_refs_json": row.artifact_refs_json,
                    "notes": row.notes,
                    "created_at": row.created_at,
                    "updated_at": row.updated_at,
                }
                for row in rows
            ]
        )
        return self._expand_param_columns(frame)

    def load_walk_forward_studies(self) -> pd.DataFrame:
        rc = ResultCatalog(self.db_path)
        rows = rc.load_walk_forward_studies()
        if not rows:
            return pd.DataFrame(
                columns=[
                    "wf_study_id",
                    "batch_id",
                    "strategy",
                    "dataset_id",
                    "timeframe",
                    "candidate_source_mode",
                    "param_names",
                    "schedule_json",
                    "selection_rule",
                    "params_json",
                    "status",
                    "description",
                    "stitched_metrics_json",
                    "stitched_equity_json",
                    "fold_count",
                    "created_at",
                ]
            )
        return pd.DataFrame(
            [
                {
                    "wf_study_id": row.wf_study_id,
                    "batch_id": row.batch_id,
                    "strategy": row.strategy,
                    "dataset_id": row.dataset_id,
                    "timeframe": row.timeframe,
                    "candidate_source_mode": row.candidate_source_mode,
                    "param_names": list(row.param_names),
                    "schedule_json": row.schedule_json,
                    "selection_rule": row.selection_rule,
                    "params_json": row.params_json,
                    "status": row.status,
                    "description": row.description,
                    "stitched_metrics_json": row.stitched_metrics_json,
                    "stitched_equity_json": row.stitched_equity_json,
                    "fold_count": row.fold_count,
                    "created_at": row.created_at,
                }
                for row in rows
            ]
        )

    def load_walk_forward_folds(self, wf_study_id: str) -> pd.DataFrame:
        rc = ResultCatalog(self.db_path)
        rows = rc.load_walk_forward_folds(wf_study_id)
        if not rows:
            return pd.DataFrame(
                columns=[
                    "wf_study_id",
                    "fold_index",
                    "train_study_id",
                    "timeframe",
                    "train_start",
                    "train_end",
                    "test_start",
                    "test_end",
                    "selected_param_set_id",
                    "selected_params_json",
                    "train_rank",
                    "train_robust_score",
                    "test_run_id",
                    "status",
                ]
            )
        frame = pd.DataFrame(
            [
                {
                    "wf_study_id": row.wf_study_id,
                    "fold_index": row.fold_index,
                    "train_study_id": row.train_study_id,
                    "timeframe": row.timeframe,
                    "train_start": row.train_start,
                    "train_end": row.train_end,
                    "test_start": row.test_start,
                    "test_end": row.test_end,
                    "selected_param_set_id": row.selected_param_set_id,
                    "selected_params_json": row.selected_params_json,
                    "train_rank": row.train_rank,
                    "train_robust_score": row.train_robust_score,
                    "test_run_id": row.test_run_id,
                    "status": row.status,
                }
                for row in rows
            ]
        )
        return self._expand_param_columns(frame)

    def load_walk_forward_fold_metrics(self, wf_study_id: str) -> pd.DataFrame:
        rc = ResultCatalog(self.db_path)
        rows = rc.load_walk_forward_fold_metrics(wf_study_id)
        if not rows:
            return pd.DataFrame(
                columns=[
                    "wf_study_id",
                    "fold_index",
                    "train_metrics_json",
                    "test_metrics_json",
                    "degradation_json",
                    "param_drift_json",
                ]
            )
        return pd.DataFrame(
            [
                {
                    "wf_study_id": row.wf_study_id,
                    "fold_index": row.fold_index,
                    "train_metrics_json": row.train_metrics_json,
                    "test_metrics_json": row.test_metrics_json,
                    "degradation_json": row.degradation_json,
                    "param_drift_json": row.param_drift_json,
                }
                for row in rows
            ]
        )

    def load_monte_carlo_studies(self) -> pd.DataFrame:
        rc = ResultCatalog(self.db_path)
        rows = rc.load_monte_carlo_studies()
        if not rows:
            return pd.DataFrame(
                columns=[
                    "mc_study_id",
                    "source_type",
                    "source_id",
                    "resampling_mode",
                    "simulation_count",
                    "seed",
                    "cost_stress_json",
                    "status",
                    "description",
                    "source_trade_count",
                    "starting_equity",
                    "summary_json",
                    "fan_quantiles_json",
                    "terminal_returns_json",
                    "max_drawdowns_json",
                    "terminal_equities_json",
                    "original_path_json",
                    "created_at",
                ]
            )
        return pd.DataFrame(
            [
                {
                    "mc_study_id": row.mc_study_id,
                    "source_type": row.source_type,
                    "source_id": row.source_id,
                    "resampling_mode": row.resampling_mode,
                    "simulation_count": row.simulation_count,
                    "seed": row.seed,
                    "cost_stress_json": row.cost_stress_json,
                    "status": row.status,
                    "description": row.description,
                    "source_trade_count": row.source_trade_count,
                    "starting_equity": row.starting_equity,
                    "summary_json": row.summary_json,
                    "fan_quantiles_json": row.fan_quantiles_json,
                    "terminal_returns_json": row.terminal_returns_json,
                    "max_drawdowns_json": row.max_drawdowns_json,
                    "terminal_equities_json": row.terminal_equities_json,
                    "original_path_json": row.original_path_json,
                    "created_at": row.created_at,
                }
                for row in rows
            ]
        )

    def load_monte_carlo_paths(self, mc_study_id: str) -> pd.DataFrame:
        rc = ResultCatalog(self.db_path)
        rows = rc.load_monte_carlo_paths(mc_study_id)
        if not rows:
            return pd.DataFrame(columns=["mc_study_id", "path_id", "path_type", "path_json", "summary_json"])
        return pd.DataFrame(
            [
                {
                    "mc_study_id": row.mc_study_id,
                    "path_id": row.path_id,
                    "path_type": row.path_type,
                    "path_json": row.path_json,
                    "summary_json": row.summary_json,
                }
                for row in rows
            ]
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
        metrics: dict,
        asset_results: pd.DataFrame | None = None,
        artifact_refs: dict | None = None,
        notes: str = "",
    ) -> str:
        return ResultCatalog(self.db_path).save_optimization_candidate(
            study_id=study_id,
            timeframe=timeframe,
            start=start,
            end=end,
            param_key=param_key,
            params_json=params_json,
            source_type=source_type,
            promotion_reason=promotion_reason,
            status=status,
            metrics=metrics,
            asset_results=asset_results,
            artifact_refs=artifact_refs,
            notes=notes,
        )

    def delete_optimization_candidate(self, candidate_id: str) -> None:
        ResultCatalog(self.db_path).delete_optimization_candidate(candidate_id)

    @staticmethod
    def _expand_param_columns(frame: pd.DataFrame) -> pd.DataFrame:
        if frame is None or frame.empty or "params_json" not in frame.columns:
            return frame
        expanded = frame.copy()
        try:
            params_series = expanded["params_json"].apply(
                lambda value: json.loads(value) if isinstance(value, str) and value else {}
            )
        except Exception:
            return expanded
        param_frame = pd.DataFrame(list(params_series))
        if param_frame.empty:
            return expanded
        for col in param_frame.columns:
            if col not in expanded.columns:
                expanded[col] = param_frame[col]
        return expanded

    def load_scheduled_tasks(self) -> List[Dict]:
        if not self.db_path.exists():
            return []
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT task_id, created_at, updated_at, symbols, schedule, status,
                       last_run_at, last_run_status, last_run_message, next_run_at
                FROM scheduled_tasks ORDER BY created_at DESC
                """
            ).fetchall()
        tasks = []
        for r in rows:
            tasks.append(
                {
                    "task_id": r[0],
                    "created_at": r[1],
                    "updated_at": r[2],
                    "symbols": json.loads(r[3]) if r[3] else [],
                    "schedule": json.loads(r[4]) if r[4] else {},
                    "status": r[5],
                    "last_run_at": r[6],
                    "last_run_status": r[7],
                    "last_run_message": r[8],
                    "next_run_at": r[9],
                }
            )
        return tasks

    def load_universes(self) -> List[Dict]:
        records = ResultCatalog(self.db_path).load_universes()
        universes: list[dict] = []
        for record in records:
            try:
                symbols = json.loads(record.symbols_json) if record.symbols_json else []
            except Exception:
                symbols = []
            try:
                dataset_ids = json.loads(record.dataset_ids_json) if record.dataset_ids_json else []
            except Exception:
                dataset_ids = []
            universes.append(
                {
                    "universe_id": record.universe_id,
                    "name": record.name,
                    "description": record.description or "",
                    "symbols": [str(item) for item in list(symbols or []) if str(item).strip()],
                    "dataset_ids": [str(item) for item in list(dataset_ids or []) if str(item).strip()],
                    "source_preference": str(record.source_preference or ""),
                    "created_at": record.created_at,
                    "updated_at": record.updated_at,
                }
            )
        return universes

    def load_acquisition_datasets(self) -> pd.DataFrame:
        rows = ResultCatalog(self.db_path).load_acquisition_datasets()
        frame = pd.DataFrame(
            [
                {
                    "dataset_id": row.dataset_id,
                    "source": row.source,
                    "symbol": row.symbol,
                    "resolution": row.resolution,
                    "history_window": row.history_window,
                    "csv_path": row.csv_path,
                    "parquet_path": row.parquet_path,
                    "coverage_start": row.coverage_start,
                    "coverage_end": row.coverage_end,
                    "bar_count": row.bar_count,
                    "ingested": bool(row.ingested),
                    "last_download_attempt_at": row.last_download_attempt_at,
                    "last_download_success_at": row.last_download_success_at,
                    "last_ingest_at": row.last_ingest_at,
                    "last_status": row.last_status,
                    "last_error": row.last_error,
                    "last_run_id": row.last_run_id,
                    "last_task_id": row.last_task_id,
                    "created_at": row.created_at,
                    "updated_at": row.updated_at,
                }
                for row in rows
            ]
        )
        dataset_map = {str(row["dataset_id"]): row for _, row in frame.iterrows()} if not frame.empty else {}
        store = DuckDBStore()
        for path in sorted(store.data_dir.glob("*.parquet")):
            dataset_id = path.stem
            if dataset_id in dataset_map:
                mask = frame["dataset_id"] == dataset_id
                if not str(frame.loc[mask, "parquet_path"].iloc[0] or "").strip():
                    frame.loc[mask, "parquet_path"] = str(path)
                try:
                    meta = store.describe_dataset(dataset_id)
                except Exception:
                    continue
                if frame.loc[mask, "coverage_start"].isna().all():
                    frame.loc[mask, "coverage_start"] = meta.get("coverage_start")
                if frame.loc[mask, "coverage_end"].isna().all():
                    frame.loc[mask, "coverage_end"] = meta.get("coverage_end")
                if frame.loc[mask, "bar_count"].isna().all():
                    frame.loc[mask, "bar_count"] = meta.get("bar_count")
                if frame.loc[mask, "ingested"].isna().all():
                    frame.loc[mask, "ingested"] = True
                continue
            try:
                meta = store.describe_dataset(dataset_id)
            except Exception:
                continue
            extra = {
                "dataset_id": dataset_id,
                "source": None,
                "symbol": None,
                "resolution": None,
                "history_window": None,
                "csv_path": None,
                "parquet_path": meta.get("parquet_path"),
                "coverage_start": meta.get("coverage_start"),
                "coverage_end": meta.get("coverage_end"),
                "bar_count": meta.get("bar_count"),
                "ingested": True,
                "last_download_attempt_at": None,
                "last_download_success_at": None,
                "last_ingest_at": None,
                "last_status": "ingested",
                "last_error": None,
                "last_run_id": None,
                "last_task_id": None,
                "created_at": None,
                "updated_at": None,
            }
            frame = pd.concat([frame, pd.DataFrame([extra])], ignore_index=True)
        if frame.empty:
            return pd.DataFrame(
                columns=[
                    "dataset_id",
                    "source",
                    "symbol",
                    "resolution",
                    "history_window",
                    "csv_path",
                    "parquet_path",
                    "coverage_start",
                    "coverage_end",
                    "bar_count",
                    "ingested",
                    "last_download_attempt_at",
                    "last_download_success_at",
                    "last_ingest_at",
                    "last_status",
                    "last_error",
                    "last_run_id",
                    "last_task_id",
                    "created_at",
                    "updated_at",
                ]
            )
        frame = frame.sort_values(
            by=["last_download_attempt_at", "last_ingest_at", "dataset_id"],
            ascending=[False, False, True],
            na_position="last",
        ).reset_index(drop=True)
        return frame

    def load_acquisition_runs(
        self,
        limit: int | None = None,
        *,
        task_id: str | None = None,
        universe_id: str | None = None,
    ) -> pd.DataFrame:
        rows = ResultCatalog(self.db_path).load_acquisition_runs(limit=limit, task_id=task_id, universe_id=universe_id)
        if not rows:
            return pd.DataFrame(
                columns=[
                    "acquisition_run_id",
                    "trigger_type",
                    "source",
                    "universe_id",
                    "universe_name",
                    "task_id",
                    "started_at",
                    "finished_at",
                    "status",
                    "symbol_count",
                    "success_count",
                    "failed_count",
                    "ingested_count",
                    "notes",
                    "log_path",
                    "created_at",
                    "updated_at",
                ]
            )
        return pd.DataFrame(
            [
                {
                    "acquisition_run_id": row.acquisition_run_id,
                    "trigger_type": row.trigger_type,
                    "source": row.source,
                    "universe_id": row.universe_id,
                    "universe_name": row.universe_name,
                    "task_id": row.task_id,
                    "started_at": row.started_at,
                    "finished_at": row.finished_at,
                    "status": row.status,
                    "symbol_count": row.symbol_count,
                    "success_count": row.success_count,
                    "failed_count": row.failed_count,
                    "ingested_count": row.ingested_count,
                    "notes": row.notes,
                    "log_path": row.log_path,
                    "created_at": row.created_at,
                    "updated_at": row.updated_at,
                }
                for row in rows
            ]
        )

    def load_acquisition_attempts(
        self,
        *,
        acquisition_run_id: str | None = None,
        dataset_id: str | None = None,
        symbol: str | None = None,
        task_id: str | None = None,
        universe_id: str | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        rows = ResultCatalog(self.db_path).load_acquisition_attempts(
            acquisition_run_id=acquisition_run_id,
            dataset_id=dataset_id,
            symbol=symbol,
            task_id=task_id,
            universe_id=universe_id,
            limit=limit,
        )
        if not rows:
            return pd.DataFrame(
                columns=[
                    "attempt_id",
                    "acquisition_run_id",
                    "seq",
                    "source",
                    "symbol",
                    "dataset_id",
                    "status",
                    "started_at",
                    "finished_at",
                    "csv_path",
                    "parquet_path",
                    "coverage_start",
                    "coverage_end",
                    "bar_count",
                    "ingested",
                    "error_message",
                    "log_path",
                    "task_id",
                    "universe_id",
                ]
            )
        return pd.DataFrame(
            [
                {
                    "attempt_id": row.attempt_id,
                    "acquisition_run_id": row.acquisition_run_id,
                    "seq": row.seq,
                    "source": row.source,
                    "symbol": row.symbol,
                    "dataset_id": row.dataset_id,
                    "status": row.status,
                    "started_at": row.started_at,
                    "finished_at": row.finished_at,
                    "csv_path": row.csv_path,
                    "parquet_path": row.parquet_path,
                    "coverage_start": row.coverage_start,
                    "coverage_end": row.coverage_end,
                    "bar_count": row.bar_count,
                    "ingested": bool(row.ingested),
                    "error_message": row.error_message,
                    "log_path": row.log_path,
                    "task_id": row.task_id,
                    "universe_id": row.universe_id,
                }
                for row in rows
            ]
        )

    def load_task_runs(self, *, task_id: str | None = None, limit: int | None = None) -> pd.DataFrame:
        rows = ResultCatalog(self.db_path).load_task_runs(task_id=task_id, limit=limit)
        if not rows:
            return pd.DataFrame(
                columns=[
                    "run_id",
                    "task_id",
                    "started_at",
                    "finished_at",
                    "status",
                    "ticker_count",
                    "log_path",
                    "error_message",
                ]
            )
        return pd.DataFrame(
            [
                {
                    "run_id": row.run_id,
                    "task_id": row.task_id,
                    "started_at": row.started_at,
                    "finished_at": row.finished_at,
                    "status": row.status,
                    "ticker_count": row.ticker_count,
                    "log_path": row.log_path,
                    "error_message": row.error_message,
                }
                for row in rows
            ]
        )

    def ensure_catalog(self) -> None:
        # Initialize schema if missing.
        ResultCatalog(self.db_path)


# --- Qt Models --------------------------------------------------------------
def _parse_catalog_timestamp(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    try:
        ts = pd.to_datetime(value, utc=True, errors="coerce")
    except Exception:
        return None
    if ts is pd.NaT or pd.isna(ts):
        return None
    return ts


def _format_duration(start: str | None, end: str | None) -> str:
    start_ts = _parse_catalog_timestamp(start)
    end_ts = _parse_catalog_timestamp(end)
    if start_ts is None:
        return "—"
    if end_ts is None:
        return "Running…"
    seconds = max(0.0, float((end_ts - start_ts).total_seconds()))
    if seconds < 1.0:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60.0:
        return f"{seconds:.2f} s"
    if seconds < 3600.0:
        minutes = int(seconds // 60)
        rem = seconds - (minutes * 60)
        return f"{minutes}m {rem:.1f}s"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    rem_seconds = int(seconds % 60)
    return f"{hours}h {minutes}m {rem_seconds}s"


def _format_seconds_precise(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    try:
        seconds = float(seconds)
    except Exception:
        return "—"
    if seconds < 1.0:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60.0:
        return f"{seconds:.3f} s"
    if seconds < 3600.0:
        minutes = int(seconds // 60)
        rem = seconds - (minutes * 60)
        return f"{minutes}m {rem:.2f}s"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    rem = seconds % 60
    return f"{hours}h {minutes}m {rem:.1f}s"


def _summarize_batch_benchmarks(benchmarks: Sequence[BatchExecutionBenchmark]) -> str:
    if not benchmarks:
        return "No batch benchmark data is available for this batch in the current session."
    vectorized = [b for b in benchmarks if b.resolved_execution_mode == ExecutionMode.VECTORIZED]
    reference = [b for b in benchmarks if b.resolved_execution_mode == ExecutionMode.REFERENCE]
    total_duration = sum(b.duration_seconds for b in benchmarks)
    total_params = sum(b.total_params for b in benchmarks)
    total_chunks = sum(b.chunk_count for b in benchmarks)
    reused = sum(1 for b in benchmarks if b.prepared_context_reused)
    lines = [
        f"Batches: {len(benchmarks)}",
        f"Wall-clock total: {_format_seconds_precise(total_duration)}",
        f"Parameter evaluations: {total_params}",
        f"Vectorized batches: {len(vectorized)}",
        f"Reference batches: {len(reference)}",
    ]
    if vectorized:
        lines.append(f"Vectorized chunks: {total_chunks}")
        lines.append(f"Prepared-context reuse batches: {reused}")
    return " | ".join(lines)


class RunsTableModel(QtCore.QAbstractTableModel):
    def __init__(self, runs: Sequence[RunRow]) -> None:
        super().__init__()
        self._runs = list(runs)
        self._headers = [
            "Status",
            "Run ID",
            "Strategy",
            "Requested",
            "Resolved",
            "Engine",
            "Params",
            "Timeframe",
            "Data Start",
            "Data End",
            "Run Started",
            "Run Finished",
            "Duration",
            "Dataset",
            "Total Return",
            "Sharpe",
            "Rolling Sharpe",
            "Max DD",
            "Final Equity",
            "Trades Log",
            "Trades CSV",
        ]

    def rowCount(self, parent=None):
        return len(self._runs)

    def columnCount(self, parent=None):
        return len(self._headers)

    def data(self, index, role=None):
        if not index.isValid():
            return None
        run = self._runs[index.row()]
        col = index.column()
        metrics = run.metrics or {}
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            if col == 0:
                return "●"
            if col == 1:
                return run.run_id[:10] + "…"
            if col == 2:
                return run.strategy
            if col == 3:
                return (run.requested_execution_mode or "—").title()
            if col == 4:
                return (run.resolved_execution_mode or "—").title()
            if col == 5:
                return run.engine_impl or "—"
            if col == 6:
                return run.params
            if col == 7:
                return run.timeframe
            if col == 8:
                return run.start
            if col == 9:
                return run.end
            if col == 10:
                return run.run_started_at or ""
            if col == 11:
                return run.run_finished_at or ""
            if col == 12:
                return _format_duration(run.run_started_at, run.run_finished_at)
            if col == 13:
                return run.dataset_id
            if col == 14:
                return "—" if not metrics else f"{metrics.get('total_return', 0):.4f}"
            if col == 15:
                return "—" if not metrics else f"{metrics.get('sharpe', 0):.3f}"
            if col == 16:
                if not metrics:
                    return "—"
                roll = metrics.get("rolling_sharpe")
                if roll is None or (isinstance(roll, float) and roll != roll):
                    return "—"
                return f"{roll:.3f}"
            if col == 17:
                return "—" if not metrics else f"{metrics.get('max_drawdown', 0):.4f}"
            if col == 18:
                start_cash = run.starting_cash if run.starting_cash is not None else 100_000
                final_equity = start_cash * (1 + metrics.get("total_return", 0)) if metrics else start_cash
                return f"{final_equity:,.0f}"
            if col == 19:
                return ""
            if col == 20:
                return ""
        if role == QtCore.Qt.ItemDataRole.ForegroundRole and col == 0:
            color = PALETTE["green"] if run.status == "finished" else PALETTE["red"]
            return QtGui.QColor(color)
        if role == QtCore.Qt.ItemDataRole.ToolTipRole:
            fallback = f"\nFallback: {run.fallback_reason}" if run.fallback_reason else ""
            return (
                f"Requested: {run.requested_execution_mode or '—'}\n"
                f"Resolved: {run.resolved_execution_mode or '—'}\n"
                f"Engine: {run.engine_impl or '—'} v{run.engine_version or '—'}\n"
                f"Duration: {_format_duration(run.run_started_at, run.run_finished_at)}{fallback}"
            )
        if role == QtCore.Qt.ItemDataRole.TextAlignmentRole:
            return int(QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft)
        return None

    def headerData(self, section, orientation, role=None):
        if role == QtCore.Qt.ItemDataRole.DisplayRole and orientation == QtCore.Qt.Orientation.Horizontal:
            return self._headers[section]
        return None

    def set_runs(self, runs: Sequence[RunRow]) -> None:
        self.beginResetModel()
        self._runs = list(runs)
        self.endResetModel()


class BatchTableModel(QtCore.QAbstractTableModel):
    def __init__(self, batches: Sequence[BatchRow]) -> None:
        super().__init__()
        self._batches = list(batches)
        self._headers = [
            "Status",
            "Batch ID",
            "Strategy",
            "Dataset",
            "Params",
            "Timeframes",
            "Horizons",
            "Started",
            "Finished",
            "Duration",
            "Runs",
            "Best Return",
            "Sharpe",
            "Max DD",
            "Final Equity",
        ]

    def rowCount(self, parent=None):
        return len(self._batches)

    def columnCount(self, parent=None):
        return len(self._headers)

    def data(self, index, role=None):
        if not index.isValid():
            return None
        batch = self._batches[index.row()]
        col = index.column()
        metrics = batch.metrics or {}
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            if col == 0:
                return "●"
            if col == 1:
                return batch.batch_id
            if col == 2:
                return batch.strategy
            if col == 3:
                return batch.dataset_id
            if col == 4:
                return batch.params
            if col == 5:
                return batch.timeframes
            if col == 6:
                return batch.horizons
            if col == 7:
                return batch.run_started_at or ""
            if col == 8:
                return batch.run_finished_at or ""
            if col == 9:
                return _format_duration(batch.run_started_at, batch.run_finished_at)
            if col == 10:
                total = batch.run_total or batch.run_count
                return f"{batch.finished_count}/{total}" if total else "0"
            if col == 11:
                return "—" if not metrics else f"{metrics.get('total_return', 0):.4f}"
            if col == 12:
                return "—" if not metrics else f"{metrics.get('sharpe', 0):.3f}"
            if col == 13:
                return "—" if not metrics else f"{metrics.get('max_drawdown', 0):.4f}"
            if col == 14:
                start_cash = 100_000
                final_equity = start_cash * (1 + metrics.get("total_return", 0)) if metrics else start_cash
                return f"{final_equity:,.0f}"
        if role == QtCore.Qt.ItemDataRole.ForegroundRole and col == 0:
            color = PALETTE["green"] if batch.status == "finished" else PALETTE["red"]
            return QtGui.QColor(color)
        if role == QtCore.Qt.ItemDataRole.TextAlignmentRole:
            return int(QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft)
        return None

    def headerData(self, section, orientation, role=None):
        if role == QtCore.Qt.ItemDataRole.DisplayRole and orientation == QtCore.Qt.Orientation.Horizontal:
            return self._headers[section]
        return None

    def set_batches(self, batches: Sequence[BatchRow]) -> None:
        self.beginResetModel()
        self._batches = list(batches)
        self.endResetModel()

    def batch_at(self, row: int) -> BatchRow | None:
        if 0 <= row < len(self._batches):
            return self._batches[row]
        return None


# --- Worker thread for grid orchestration -----------------------------------
class GridWorker(QtCore.QThread):
    finished_signal = QtCore.pyqtSignal(object)  # payload dict with df/spec/message
    error_signal = QtCore.pyqtSignal(str)
    progress_signal = QtCore.pyqtSignal(int, int)

    def __init__(
        self,
        csv_path: Path,
        dataset_id: str,
        dataset_ids: Sequence[str],
        study_mode: str,
        timeframes: list[str],
        horizons: list[str],
        catalog_path: Path,
        strategy_factory: Callable,
        strategy_params: Dict[str, float],
        blas_threads: int,
        intrabar_sim: bool,
        sharpe_debug: bool,
        risk_free_rate: float,
        bt_settings: Dict[str, float | bool | dict | str],
        portfolio_strategy_blocks: Sequence[dict] | None = None,
    ) -> None:
        super().__init__()
        self.csv_path = csv_path
        self.dataset_id = dataset_id
        self.dataset_ids = [dataset for dataset in dataset_ids if str(dataset).strip()]
        self.study_mode = str(study_mode or STUDY_MODE_INDEPENDENT)
        self.timeframes = timeframes
        self.horizons = horizons
        self.catalog_path = catalog_path
        self.strategy_factory = strategy_factory
        self.strategy_params = strategy_params
        self.blas_threads = max(1, blas_threads)
        self._stop_requested = False
        self.batch_id = f"batch_{uuid.uuid4().hex[:8]}"
        self.intrabar_sim = intrabar_sim
        self.sharpe_debug = sharpe_debug
        self.risk_free_rate = risk_free_rate
        self.bt_settings = bt_settings
        self.portfolio_strategy_blocks = [dict(block) for block in list(portfolio_strategy_blocks or ())]

    @staticmethod
    def _dataset_label(dataset_ids: Sequence[str]) -> str:
        unique = [dataset_id for dataset_id in dataset_ids if dataset_id]
        if not unique:
            return "dataset"
        if len(unique) == 1:
            return unique[0]
        preview = ", ".join(unique[:3])
        if len(unique) <= 3:
            return preview
        return f"{len(unique)} datasets ({preview}, ...)"

    def run(self) -> None:
        try:
            # Allow worker to tune BLAS threads per run.
            os.environ["OPENBLAS_NUM_THREADS"] = str(self.blas_threads)
            os.environ["OMP_NUM_THREADS"] = str(self.blas_threads)
            os.environ["VECLIB_MAXIMUM_THREADS"] = str(self.blas_threads)
            os.environ["NUMEXPR_MAX_THREADS"] = str(self.blas_threads)

            duck = DuckDBStore()
            dataset_ids = list(dict.fromkeys(self.dataset_ids or [self.dataset_id]))
            raw_by_dataset: dict[str, pd.DataFrame] = {}
            for dataset_id in dataset_ids:
                try:
                    raw_by_dataset[dataset_id] = duck.load(dataset_id)
                except Exception as exc:
                    raise RuntimeError(
                        f"Dataset '{dataset_id}' not found in DuckDB/parquet store. Add it first."
                    ) from exc
            end_ts = min(raw.index[-1] for raw in raw_by_dataset.values())
            dataset_label = self._dataset_label(dataset_ids)
            study_mode = str(self.study_mode or STUDY_MODE_INDEPENDENT)
            batch_dataset_label = dataset_label
            batch_strategy_label = self.strategy_factory.__name__
            if study_mode == STUDY_MODE_PORTFOLIO:
                batch_dataset_label = f"Portfolio | {dataset_label}"
                if self.portfolio_strategy_blocks:
                    batch_strategy_label = f"Portfolio Blocks [{len(self.portfolio_strategy_blocks)}]"
                else:
                    batch_strategy_label = f"{self.strategy_factory.__name__} [Portfolio]"

            # Parse horizons (e.g., "7d,30d") -> timedeltas
            deltas = []
            for h in self.horizons:
                try:
                    deltas.append(pd.Timedelta(h))
                except Exception:
                    continue
            horizons = build_horizons(end_ts, deltas) if deltas else [(None, None)]

            catalog = ResultCatalog(self.catalog_path)
            started_at = pd.Timestamp.utcnow().isoformat()
            # Compute expected run_total for batch status.
            param_lists = list(self.strategy_params.values())
            param_combo = 1
            for lst in param_lists:
                param_combo *= max(1, len(lst))
            run_total = max(1, len(self.timeframes)) * max(1, len(horizons)) * param_combo
            if study_mode != STUDY_MODE_PORTFOLIO:
                run_total *= max(1, len(dataset_ids))
            elif self.portfolio_strategy_blocks:
                run_total = max(1, len(self.timeframes)) * max(1, len(horizons))
            base_config = BacktestConfig(
                timeframe=self.timeframes[0],
                starting_cash=float(self.bt_settings.get("starting_cash", 100_000)),
                fee_rate=float(self.bt_settings.get("fee_rate", 0.0002)),
                fee_schedule=self.bt_settings.get("fee_schedule", {"buy": 0.0003, "sell": 0.0005}),
                slippage=float(self.bt_settings.get("slippage", 0.0002)),
                slippage_schedule=self.bt_settings.get("slippage_schedule", {"buy": 0.0003, "sell": 0.0001}),
                borrow_rate=float(self.bt_settings.get("borrow_rate", 0.0)),
                fill_ratio=float(self.bt_settings.get("fill_ratio", 1.0)),
                fill_on_close=bool(self.bt_settings.get("fill_on_close", False)),
                recalc_on_fill=bool(self.bt_settings.get("recalc_on_fill", True)),
                allow_short=bool(self.bt_settings.get("allow_short", True)),
                use_cache=bool(self.bt_settings.get("use_cache", False)),
                intrabar_sim=self.intrabar_sim,
                prevent_scale_in=bool(self.bt_settings.get("prevent_scale_in", True)),
                one_order_per_signal=bool(self.bt_settings.get("one_order_per_signal", True)),
                sharpe_debug=self.sharpe_debug,
                risk_free_rate=self.risk_free_rate,
            )

            if not self.strategy_params:
                raise ValueError("No strategy parameters provided.")
            first_key = list(self.strategy_params.keys())[0]
            second_key = list(self.strategy_params.keys())[1] if len(self.strategy_params) > 1 else first_key
            spec = GridSpec(
                params=self.strategy_params,
                timeframes=self.timeframes,
                horizons=horizons,
                metric="total_return",
                heatmap_rows=second_key,
                heatmap_cols=first_key,
                description=f"{'Portfolio study' if study_mode == STUDY_MODE_PORTFOLIO else 'Grid'} for {batch_dataset_label}",
                batch_id=self.batch_id,
                execution_mode=str(self.bt_settings.get("execution_mode", ExecutionMode.AUTO.value)),
            )
            batch_params = dict(self.strategy_params)
            if study_mode == STUDY_MODE_PORTFOLIO:
                batch_params["_study_mode"] = STUDY_MODE_PORTFOLIO
                batch_params["_portfolio_dataset_ids"] = list(dataset_ids)
                if self.portfolio_strategy_blocks:
                    batch_params["_portfolio_strategy_blocks"] = list(self.portfolio_strategy_blocks)
                batch_params["_portfolio_allocation_mode"] = str(
                    self.bt_settings.get("portfolio_allocation_mode", PORTFOLIO_ALLOC_EQUAL)
                )
                batch_params["_portfolio_target_weights"] = dict(
                    self.bt_settings.get("portfolio_target_weights", {}) or {}
                )
                batch_params["_portfolio_ownership_mode"] = str(
                    self.bt_settings.get("portfolio_ownership_mode", ALLOCATION_OWNERSHIP_STRATEGY)
                )
                batch_params["_portfolio_ranking_mode"] = str(
                    self.bt_settings.get("portfolio_ranking_mode", RANKING_MODE_NONE)
                )
                batch_params["_portfolio_rank_count"] = int(self.bt_settings.get("portfolio_rank_count", 1) or 1)
                batch_params["_portfolio_score_threshold"] = float(
                    self.bt_settings.get("portfolio_score_threshold", 0.0) or 0.0
                )
                batch_params["_portfolio_weighting_mode"] = str(
                    self.bt_settings.get("portfolio_weighting_mode", WEIGHTING_MODE_PRESERVE)
                )
                batch_params["_portfolio_min_active_weight"] = float(
                    self.bt_settings.get("portfolio_min_active_weight", 0.0) or 0.0
                )
                batch_params["_portfolio_max_asset_weight"] = float(
                    self.bt_settings.get("portfolio_max_asset_weight", 0.0) or 0.0
                )
                batch_params["_portfolio_cash_reserve_weight"] = float(
                    self.bt_settings.get("portfolio_cash_reserve_weight", 0.0) or 0.0
                )
                batch_params["_portfolio_rebalance_mode"] = str(
                    self.bt_settings.get("portfolio_rebalance_mode", REBALANCE_MODE_ON_CHANGE)
                )
                batch_params["_portfolio_rebalance_every_n_bars"] = int(
                    self.bt_settings.get("portfolio_rebalance_every_n_bars", 20) or 20
                )
                batch_params["_portfolio_rebalance_drift_threshold"] = float(
                    self.bt_settings.get("portfolio_rebalance_drift_threshold", 0.05) or 0.05
                )
            catalog.save_batch(
                batch_id=self.batch_id,
                strategy=batch_strategy_label,
                dataset_id=batch_dataset_label,
                params=batch_params,
                timeframes=self.timeframes,
                horizons=[str(h) for h in self.horizons],
                run_total=run_total,
                status="running",
                started_at=started_at,
                finished_at=None,
            )
            if study_mode == STUDY_MODE_PORTFOLIO:
                allocation_mode = str(self.bt_settings.get("portfolio_allocation_mode", PORTFOLIO_ALLOC_EQUAL))
                target_weights = {
                    str(dataset_id): float(weight)
                    for dataset_id, weight in dict(self.bt_settings.get("portfolio_target_weights", {}) or {}).items()
                }
                construction_config = PortfolioConstructionConfig(
                    allocation_ownership=str(
                        self.bt_settings.get("portfolio_ownership_mode", ALLOCATION_OWNERSHIP_STRATEGY)
                    ),
                    ranking_mode=str(self.bt_settings.get("portfolio_ranking_mode", RANKING_MODE_NONE)),
                    max_ranked_assets=int(self.bt_settings.get("portfolio_rank_count", 1) or 1),
                    min_rank_score=float(self.bt_settings.get("portfolio_score_threshold", 0.0) or 0.0),
                    weighting_mode=str(self.bt_settings.get("portfolio_weighting_mode", WEIGHTING_MODE_PRESERVE)),
                    min_active_weight=(
                        float(self.bt_settings.get("portfolio_min_active_weight", 0.0) or 0.0) or None
                    ),
                    max_asset_weight=(
                        float(self.bt_settings.get("portfolio_max_asset_weight", 0.0) or 0.0) or None
                    ),
                    cash_reserve_weight=float(self.bt_settings.get("portfolio_cash_reserve_weight", 0.0) or 0.0),
                    rebalance_mode=str(
                        self.bt_settings.get("portfolio_rebalance_mode", REBALANCE_MODE_ON_CHANGE)
                    ),
                    rebalance_every_n_bars=int(
                        self.bt_settings.get("portfolio_rebalance_every_n_bars", 20) or 20
                    ),
                    rebalance_weight_drift_threshold=float(
                        self.bt_settings.get("portfolio_rebalance_drift_threshold", 0.05) or 0.05
                    ),
                )
                normalize_weights = allocation_mode != PORTFOLIO_ALLOC_FIXED
                if self.portfolio_strategy_blocks:
                    strategy_blocks = []
                    for block in self.portfolio_strategy_blocks:
                        strategy_name = str(block.get("strategy_name") or "").strip()
                        strategy_spec = getattr(self.parent(), "strategy_specs", None) if hasattr(self, "parent") else None
                        strategy_cls = None
                        if strategy_name == "SMACrossStrategy":
                            strategy_cls = SMACrossStrategy
                        elif strategy_name == "ZScoreMeanReversionStrategy":
                            strategy_cls = ZScoreMeanReversionStrategy
                        elif strategy_name == "InverseTurtleStrategy":
                            strategy_cls = InverseTurtleStrategy
                        else:
                            raise RuntimeError(f"Unknown strategy block strategy: {strategy_name}")
                        asset_target_weights = dict(block.get("asset_target_weights") or {})
                        strategy_blocks.append(
                            PortfolioStrategyBlockTarget(
                                block_id=str(block.get("block_id") or strategy_name),
                                strategy_cls=strategy_cls,
                                strategy_params=dict(block.get("strategy_params") or {}),
                                assets=[
                                    PortfolioStrategyBlockAssetTarget(
                                        dataset_id=asset_dataset_id,
                                        data_loader=(lambda tf, did=asset_dataset_id: duck.resample(did, tf)),
                                        target_weight=asset_target_weights.get(asset_dataset_id),
                                        display_name=asset_dataset_id,
                                    )
                                    for asset_dataset_id in list(block.get("asset_dataset_ids") or [])
                                ],
                                budget_weight=block.get("budget_weight"),
                                display_name=str(block.get("display_name") or block.get("block_id") or strategy_name),
                            )
                        )
                    df = run_vectorized_strategy_block_portfolio_search(
                        strategy_blocks=strategy_blocks,
                        base_config=base_config,
                        grid=spec,
                        catalog=catalog,
                        stop_cb=lambda: self._stop_requested,
                        progress_cb=lambda d, t: self.progress_signal.emit(d, t),
                        normalize_weights=normalize_weights,
                        construction_config=construction_config,
                    )
                else:
                    targets = [
                        PortfolioAssetTarget(
                            dataset_id=dataset_id,
                            data_loader=(lambda tf, did=dataset_id: duck.resample(did, tf)),
                            target_weight=(
                                None
                                if allocation_mode == PORTFOLIO_ALLOC_EQUAL
                                else float(target_weights.get(dataset_id, 0.0))
                            ),
                        )
                        for dataset_id in dataset_ids
                    ]
                    df = run_vectorized_portfolio_grid_search(
                        targets=targets,
                        strategy_cls=self.strategy_factory,
                        base_config=base_config,
                        grid=spec,
                        catalog=catalog,
                        stop_cb=lambda: self._stop_requested,
                        progress_cb=lambda d, t: self.progress_signal.emit(d, t),
                        normalize_weights=normalize_weights,
                        construction_config=construction_config,
                    )
                batch_benchmarks = tuple(df.attrs.get("batch_benchmarks") or ())
            elif len(dataset_ids) == 1:
                def loader(tf: str):
                    return duck.resample(dataset_ids[0], tf)

                grid = GridSearch(
                    dataset_id=dataset_ids[0],
                    data_loader=loader,
                    strategy_cls=self.strategy_factory,
                    base_config=base_config,
                    catalog=catalog,
                )
                df = grid.run(
                    spec,
                    make_heatmap=False,  # avoid matplotlib in worker thread to prevent crashes
                    stop_cb=lambda: self._stop_requested,
                    progress_cb=lambda d, t: self.progress_signal.emit(d, t),
                )
                batch_benchmarks = tuple(grid.last_batch_benchmarks)
            else:
                targets = [
                    IndependentAssetTarget(
                        dataset_id=dataset_id,
                        data_loader=(lambda tf, did=dataset_id: duck.resample(did, tf)),
                    )
                    for dataset_id in dataset_ids
                ]
                df = run_independent_asset_grid_search(
                    targets=targets,
                    strategy_cls=self.strategy_factory,
                    base_config=base_config,
                    grid=spec,
                    catalog=catalog,
                    make_heatmap=False,
                    stop_cb=lambda: self._stop_requested,
                    progress_cb=lambda d, t: self.progress_signal.emit(d, t),
                    share_batch_id=True,
                )
                batch_benchmarks = tuple(df.attrs.get("batch_benchmarks") or ())
            should_persist_optimization = (
                isinstance(df, pd.DataFrame)
                and not df.empty
                and (
                    study_mode != STUDY_MODE_PORTFOLIO
                    or not self.portfolio_strategy_blocks
                )
            )
            if should_persist_optimization:
                try:
                    optimization_artifacts = build_optimization_study_artifacts(
                        df=df,
                        study_id=self.batch_id,
                        batch_id=self.batch_id,
                        strategy=batch_strategy_label,
                        dataset_scope=dataset_ids,
                        param_names=list(self.strategy_params.keys()),
                        timeframes=list(self.timeframes),
                        horizons=[str(h) for h in self.horizons],
                        score_version=ROBUST_SCORE_VERSION,
                    )
                    catalog.save_optimization_study(
                        study_id=optimization_artifacts.study_id,
                        batch_id=optimization_artifacts.batch_id,
                        strategy=optimization_artifacts.strategy,
                        dataset_scope=optimization_artifacts.dataset_scope,
                        param_names=optimization_artifacts.param_names,
                        timeframes=optimization_artifacts.timeframes,
                        horizons=optimization_artifacts.horizons,
                        score_version=optimization_artifacts.score_version,
                        aggregates=optimization_artifacts.aggregates,
                        asset_results=optimization_artifacts.asset_results,
                    )
                except Exception:
                    # Optimization persistence is best-effort so a completed batch still succeeds
                    # even if the post-study aggregation step fails.
                    pass
            catalog.save_batch_benchmarks(self.batch_id, batch_benchmarks)
            if self._stop_requested:
                message = "Study stopped." if study_mode == STUDY_MODE_PORTFOLIO else "Grid stopped."
            elif study_mode == STUDY_MODE_PORTFOLIO:
                message = f"Portfolio study completed across {len(dataset_ids)} datasets."
            elif len(dataset_ids) > 1:
                message = f"Independent study completed across {len(dataset_ids)} datasets."
            else:
                message = "Grid completed."
            catalog.save_batch(
                batch_id=self.batch_id,
                strategy=batch_strategy_label,
                dataset_id=batch_dataset_label,
                params=batch_params,
                timeframes=self.timeframes,
                horizons=[str(h) for h in self.horizons],
                run_total=run_total,
                status="finished" if not self._stop_requested else "stopped",
                started_at=started_at,
                finished_at=pd.Timestamp.utcnow().isoformat(),
            )
            self.finished_signal.emit(
                {
                    "df": df,
                    "spec": spec,
                    "message": message,
                    "batch_id": self.batch_id,
                    "batch_benchmarks": batch_benchmarks,
                }
            )
        except UnsupportedExecutionModeError as exc:
            self.error_signal.emit(str(exc))
        except Exception as exc:
            tb = traceback.format_exc()
            print("GridWorker error:\n", tb)
            err_text = tb if tb else str(exc)
            self.error_signal.emit(err_text)

    def request_stop(self) -> None:
        self._stop_requested = True


class WalkForwardWorker(QtCore.QThread):
    finished_signal = QtCore.pyqtSignal(object)
    error_signal = QtCore.pyqtSignal(str)

    def __init__(
        self,
        *,
        catalog_path: Path,
        source_study_row: dict,
        strategy_cls: Callable,
        dataset_id: str,
        timeframe: str,
        first_test_start: str,
        test_window_bars: int,
        num_folds: int,
        min_train_bars: int,
        candidate_source_mode: str,
        execution_mode: str,
        bt_settings: Dict[str, float | bool | dict | str],
        description: str,
    ) -> None:
        super().__init__()
        self.catalog_path = Path(catalog_path)
        self.source_study_row = dict(source_study_row)
        self.strategy_cls = strategy_cls
        self.dataset_id = str(dataset_id)
        self.timeframe = str(timeframe)
        self.first_test_start = str(first_test_start)
        self.test_window_bars = int(test_window_bars)
        self.num_folds = int(num_folds)
        self.min_train_bars = int(min_train_bars)
        self.candidate_source_mode = str(candidate_source_mode)
        self.execution_mode = str(execution_mode or ExecutionMode.AUTO.value)
        self.bt_settings = dict(bt_settings)
        self.description = str(description or "")
        self.wf_study_id = f"wf_{uuid.uuid4().hex[:8]}"

    def run(self) -> None:
        try:
            duck = DuckDBStore()
            catalog = ResultCatalog(self.catalog_path)
            study_id = str(self.source_study_row.get("study_id", "") or "")
            if not study_id:
                raise RuntimeError("Walk-forward setup is missing a source optimization study.")
            bars = duck.resample(self.dataset_id, self.timeframe)
            if bars is None or bars.empty:
                raise RuntimeError(f"No bars were available for dataset '{self.dataset_id}' at timeframe '{self.timeframe}'.")

            base_config = BacktestConfig(
                timeframe=self.timeframe,
                starting_cash=float(self.bt_settings.get("starting_cash", 100_000)),
                fee_rate=float(self.bt_settings.get("fee_rate", 0.0002)),
                fee_schedule=self.bt_settings.get("fee_schedule", {"buy": 0.0003, "sell": 0.0005}),
                slippage=float(self.bt_settings.get("slippage", 0.0002)),
                slippage_schedule=self.bt_settings.get("slippage_schedule", {"buy": 0.0003, "sell": 0.0001}),
                borrow_rate=float(self.bt_settings.get("borrow_rate", 0.0)),
                fill_ratio=float(self.bt_settings.get("fill_ratio", 1.0)),
                fill_on_close=bool(self.bt_settings.get("fill_on_close", False)),
                recalc_on_fill=bool(self.bt_settings.get("recalc_on_fill", True)),
                allow_short=bool(self.bt_settings.get("allow_short", False)),
                use_cache=bool(self.bt_settings.get("use_cache", False)),
                intrabar_sim=bool(self.bt_settings.get("intrabar_sim", False)),
                prevent_scale_in=bool(self.bt_settings.get("prevent_scale_in", True)),
                one_order_per_signal=bool(self.bt_settings.get("one_order_per_signal", True)),
                sharpe_debug=bool(self.bt_settings.get("sharpe_debug", False)),
                risk_free_rate=float(self.bt_settings.get("risk_free_rate", 0.0)),
                sharpe_basis="period",
            )

            param_names = [str(name) for name in list(self.source_study_row.get("param_names") or []) if str(name)]
            param_grid = None
            candidate_params = None
            if self.candidate_source_mode == WALK_FORWARD_SOURCE_FULL_GRID:
                aggregates = catalog.load_optimization_aggregates(study_id)
                if aggregates.empty:
                    raise RuntimeError("The selected optimization study does not contain aggregate rows to build a train-fold parameter grid.")
                if "timeframe" in aggregates.columns:
                    aggregates = aggregates.loc[aggregates["timeframe"].fillna("") == self.timeframe].reset_index(drop=True)
                if aggregates.empty:
                    raise RuntimeError("No optimization aggregates were found for the selected timeframe.")
                param_grid = self._param_grid_from_aggregate_frame(aggregates, param_names)
            else:
                candidates = catalog.load_optimization_candidates(study_id)
                filtered_candidates = []
                for candidate in candidates:
                    candidate_timeframe = str(getattr(candidate, "timeframe", "") or "")
                    if candidate_timeframe and candidate_timeframe != self.timeframe:
                        continue
                    filtered_candidates.append(candidate)
                if not filtered_candidates:
                    raise RuntimeError(
                        "No promoted optimization candidates were available for the selected study/timeframe."
                    )
                candidate_params = candidate_param_sets_from_records(filtered_candidates)

            artifacts = run_walk_forward_study(
                wf_study_id=self.wf_study_id,
                dataset_id=self.dataset_id,
                data_loader=lambda tf: duck.resample(self.dataset_id, tf),
                strategy_cls=self.strategy_cls,
                base_config=base_config,
                timeframe=self.timeframe,
                first_test_start=self.first_test_start,
                test_window_bars=self.test_window_bars,
                num_folds=self.num_folds,
                param_grid=param_grid,
                candidate_params=candidate_params,
                candidate_source_mode=self.candidate_source_mode,
                catalog=catalog,
                requested_execution_mode=self.execution_mode,
                min_train_bars=self.min_train_bars,
                description=self.description or f"Walk-forward from optimization study {study_id}",
                source_study_id=study_id,
                source_batch_id=str(self.source_study_row.get("batch_id", "") or ""),
            )
            self.finished_signal.emit(
                {
                    "wf_study_id": self.wf_study_id,
                    "source_study_id": study_id,
                    "artifacts": artifacts,
                    "message": (
                        f"Walk-forward study completed with {len(artifacts.folds)} fold"
                        f"{'' if len(artifacts.folds) == 1 else 's'}."
                    ),
                }
            )
        except Exception as exc:
            tb = traceback.format_exc()
            print("WalkForwardWorker error:\n", tb)
            self.error_signal.emit(tb if tb else str(exc))

    @staticmethod
    def _param_grid_from_aggregate_frame(frame: pd.DataFrame, param_names: Sequence[str]) -> dict[str, list]:
        grid: dict[str, list] = {}
        for param_name in param_names:
            values: list[object] = []
            for params_json in frame["params_json"].dropna().tolist():
                try:
                    decoded = json.loads(str(params_json))
                except Exception:
                    decoded = {}
                if param_name in decoded:
                    values.append(decoded[param_name])
            deduped: list[object] = []
            seen: set[str] = set()
            for value in values:
                marker = json.dumps(value, sort_keys=True)
                if marker in seen:
                    continue
                seen.add(marker)
                deduped.append(value)
            if deduped:
                grid[str(param_name)] = deduped
        if not grid:
            raise RuntimeError("Could not derive a parameter grid from the selected optimization study.")
        return grid


class PortfolioWalkForwardWorker(QtCore.QThread):
    finished_signal = QtCore.pyqtSignal(object)
    error_signal = QtCore.pyqtSignal(str)

    def __init__(
        self,
        *,
        catalog_path: Path,
        source_batch_row: dict,
        timeframe: str,
        first_test_start: str,
        test_window_bars: int,
        num_folds: int,
        min_train_bars: int,
        candidate_source_mode: str,
        execution_mode: str,
        bt_settings: Dict[str, float | bool | dict | str],
        description: str,
    ) -> None:
        super().__init__()
        self.catalog_path = Path(catalog_path)
        self.source_batch_row = dict(source_batch_row)
        self.timeframe = str(timeframe)
        self.first_test_start = str(first_test_start)
        self.test_window_bars = int(test_window_bars)
        self.num_folds = int(num_folds)
        self.min_train_bars = int(min_train_bars)
        self.candidate_source_mode = str(candidate_source_mode or WALK_FORWARD_SOURCE_FULL_GRID)
        self.execution_mode = str(execution_mode or ExecutionMode.AUTO.value)
        self.bt_settings = dict(bt_settings)
        self.description = str(description or "")
        self.wf_study_id = f"wf_{uuid.uuid4().hex[:8]}"

    @staticmethod
    def _base_config_from_settings(
        *,
        timeframe: str,
        bt_settings: Dict[str, float | bool | dict | str],
    ) -> BacktestConfig:
        # Portfolio walk-forward still uses the vectorized portfolio backend, so
        # keep this config normalization local and explicit.
        return BacktestConfig(
            timeframe=str(timeframe),
            starting_cash=float(bt_settings.get("starting_cash", 100_000)),
            fee_rate=float(bt_settings.get("fee_rate", 0.0002)),
            fee_schedule=bt_settings.get("fee_schedule", {"buy": 0.0003, "sell": 0.0005}),
            slippage=float(bt_settings.get("slippage", 0.0002)),
            slippage_schedule=bt_settings.get("slippage_schedule", {"buy": 0.0003, "sell": 0.0001}),
            borrow_rate=float(bt_settings.get("borrow_rate", 0.0)),
            fill_ratio=float(bt_settings.get("fill_ratio", 1.0)),
            fill_on_close=bool(bt_settings.get("fill_on_close", False)),
            recalc_on_fill=bool(bt_settings.get("recalc_on_fill", True)),
            allow_short=bool(bt_settings.get("allow_short", False)),
            use_cache=bool(bt_settings.get("use_cache", False)),
            intrabar_sim=bool(bt_settings.get("intrabar_sim", False)),
            prevent_scale_in=bool(bt_settings.get("prevent_scale_in", True)),
            one_order_per_signal=bool(bt_settings.get("one_order_per_signal", True)),
            sharpe_debug=bool(bt_settings.get("sharpe_debug", False)),
            risk_free_rate=float(bt_settings.get("risk_free_rate", 0.0)),
            sharpe_basis="period",
        )

    def run(self) -> None:
        try:
            duck = DuckDBStore()
            catalog = ResultCatalog(self.catalog_path)
            batch_id = str(self.source_batch_row.get("batch_id", "") or "")
            if not batch_id:
                raise RuntimeError("Portfolio walk-forward setup is missing a source portfolio batch.")
            batch_params = self._decode_batch_params(self.source_batch_row)
            mode = str(self.source_batch_row.get("mode", "shared_strategy") or "shared_strategy")
            dataset_ids = [str(item) for item in list(self.source_batch_row.get("dataset_ids") or []) if str(item).strip()]
            if not dataset_ids:
                raise RuntimeError("The selected portfolio batch does not contain enough dataset information to rebuild folds.")

            base_config = self._base_config_from_settings(
                timeframe=self.timeframe,
                bt_settings=self.bt_settings,
            )

            construction_config = self._construction_config_from_batch_params(batch_params)
            normalize_weights = str(batch_params.get("_portfolio_allocation_mode", PORTFOLIO_ALLOC_EQUAL)) != PORTFOLIO_ALLOC_FIXED
            strategy_label = str(self.source_batch_row.get("strategy", "PortfolioExecution") or "PortfolioExecution")

            if mode == "strategy_blocks":
                strategy_blocks = self._strategy_blocks_from_batch_params(batch_params)
                if not strategy_blocks:
                    raise RuntimeError("The selected portfolio batch does not contain strategy blocks to revalidate.")
                effective_candidate_source_mode = WALK_FORWARD_SOURCE_FIXED_PORTFOLIO
                strategy_block_candidates = None
                if self.candidate_source_mode == WALK_FORWARD_SOURCE_REDUCED_CANDIDATES:
                    candidates = catalog.load_optimization_candidates(batch_id)
                    filtered_candidates = []
                    for candidate in candidates:
                        candidate_timeframe = str(getattr(candidate, "timeframe", "") or "")
                        if candidate_timeframe and candidate_timeframe != self.timeframe:
                            continue
                        filtered_candidates.append(candidate)
                    if not filtered_candidates:
                        raise RuntimeError(
                            "No promoted fixed portfolio definitions were available for the selected batch/timeframe."
                        )
                    strategy_block_candidates = self._strategy_block_candidates_from_candidate_records(filtered_candidates)
                    effective_candidate_source_mode = WALK_FORWARD_SOURCE_REDUCED_CANDIDATES
                artifacts = run_walk_forward_portfolio_study(
                    wf_study_id=self.wf_study_id,
                    portfolio_dataset_id=f"WalkForward Portfolio | {batch_id}",
                    data_loader=lambda dataset_id, tf: duck.resample(dataset_id, tf),
                    base_config=base_config,
                    timeframe=self.timeframe,
                    first_test_start=self.first_test_start,
                    test_window_bars=self.test_window_bars,
                    num_folds=self.num_folds,
                    strategy_blocks=strategy_blocks,
                    strategy_block_candidates=strategy_block_candidates,
                    strategy_label=strategy_label,
                    construction_config=construction_config,
                    normalize_weights=normalize_weights,
                    candidate_source_mode=effective_candidate_source_mode,
                    catalog=catalog,
                    requested_execution_mode=self.execution_mode,
                    min_train_bars=self.min_train_bars,
                    description=self.description or f"Portfolio walk-forward from batch {batch_id}",
                    source_batch_id=batch_id,
                )
            else:
                shared_strategy_cls = self._shared_strategy_cls(strategy_label)
                portfolio_assets = self._portfolio_assets_from_batch_params(batch_params, dataset_ids)
                param_grid = None
                candidate_params = None
                if self.candidate_source_mode == WALK_FORWARD_SOURCE_REDUCED_CANDIDATES:
                    candidates = catalog.load_optimization_candidates(batch_id)
                    filtered_candidates = []
                    for candidate in candidates:
                        candidate_timeframe = str(getattr(candidate, "timeframe", "") or "")
                        if candidate_timeframe and candidate_timeframe != self.timeframe:
                            continue
                        filtered_candidates.append(candidate)
                    if not filtered_candidates:
                        raise RuntimeError(
                            "No promoted portfolio candidates were available for the selected batch/timeframe."
                        )
                    candidate_params = candidate_param_sets_from_records(filtered_candidates)
                else:
                    param_grid = self._param_grid_from_batch_params(batch_params)
                    if not param_grid:
                        raise RuntimeError("The selected shared-strategy portfolio batch does not contain a parameter grid.")
                artifacts = run_walk_forward_portfolio_study(
                    wf_study_id=self.wf_study_id,
                    portfolio_dataset_id=f"WalkForward Portfolio | {batch_id}",
                    data_loader=lambda dataset_id, tf: duck.resample(dataset_id, tf),
                    base_config=base_config,
                    timeframe=self.timeframe,
                    first_test_start=self.first_test_start,
                    test_window_bars=self.test_window_bars,
                    num_folds=self.num_folds,
                    shared_strategy_cls=shared_strategy_cls,
                    portfolio_assets=portfolio_assets,
                    param_grid=param_grid,
                    candidate_params=candidate_params,
                    strategy_label=strategy_label,
                    construction_config=construction_config,
                    normalize_weights=normalize_weights,
                    candidate_source_mode=self.candidate_source_mode,
                    catalog=catalog,
                    requested_execution_mode=self.execution_mode,
                    min_train_bars=self.min_train_bars,
                    description=self.description or f"Portfolio walk-forward from batch {batch_id}",
                    source_batch_id=batch_id,
                    source_study_id=str(self.source_batch_row.get("source_study_id", "") or ""),
                )

            self.finished_signal.emit(
                {
                    "wf_study_id": self.wf_study_id,
                    "source_batch_id": batch_id,
                    "artifacts": artifacts,
                    "message": (
                        f"Portfolio walk-forward study completed with {len(artifacts.folds)} fold"
                        f"{'' if len(artifacts.folds) == 1 else 's'}."
                    ),
                }
            )
        except Exception as exc:
            tb = traceback.format_exc()
            print("PortfolioWalkForwardWorker error:\n", tb)
            self.error_signal.emit(tb if tb else str(exc))

    @staticmethod
    def _decode_batch_params(source_batch_row: dict) -> dict:
        params = source_batch_row.get("params_dict")
        if isinstance(params, dict):
            return dict(params)
        raw = source_batch_row.get("params")
        if isinstance(raw, dict):
            return dict(raw)
        if not raw:
            return {}
        try:
            return json.loads(str(raw))
        except Exception:
            return {}

    @staticmethod
    def _shared_strategy_cls(strategy_label: str):
        base_name = str(strategy_label or "").replace("[Portfolio]", "").strip()
        return RunChartDialog._strategy_class_static(base_name)

    @staticmethod
    def _param_grid_from_batch_params(batch_params: dict) -> dict[str, list]:
        param_grid: dict[str, list] = {}
        for key, value in dict(batch_params or {}).items():
            if str(key).startswith("_"):
                continue
            values = list(value) if isinstance(value, (list, tuple)) else [value]
            if values:
                param_grid[str(key)] = values
        return param_grid

    @staticmethod
    def _portfolio_assets_from_batch_params(
        batch_params: dict,
        dataset_ids: Sequence[str],
    ) -> list[WalkForwardPortfolioAssetDefinition]:
        target_weights = {
            str(dataset_id): float(weight)
            for dataset_id, weight in dict(batch_params.get("_portfolio_target_weights", {}) or {}).items()
        }
        allocation_mode = str(batch_params.get("_portfolio_allocation_mode", PORTFOLIO_ALLOC_EQUAL) or PORTFOLIO_ALLOC_EQUAL)
        assets: list[WalkForwardPortfolioAssetDefinition] = []
        for dataset_id in dataset_ids:
            assets.append(
                WalkForwardPortfolioAssetDefinition(
                    dataset_id=str(dataset_id),
                    target_weight=(target_weights.get(str(dataset_id)) if allocation_mode != PORTFOLIO_ALLOC_EQUAL else None),
                    display_name=str(dataset_id),
                )
            )
        return assets

    @staticmethod
    def _strategy_blocks_from_batch_params(
        batch_params: dict,
    ) -> list[WalkForwardPortfolioStrategyBlockDefinition]:
        blocks: list[WalkForwardPortfolioStrategyBlockDefinition] = []
        for raw_block in list(batch_params.get("_portfolio_strategy_blocks") or []):
            strategy_name = str(raw_block.get("strategy_name") or raw_block.get("strategy") or "").strip()
            strategy_cls = RunChartDialog._strategy_class_static(strategy_name)
            params = dict(raw_block.get("strategy_params") or raw_block.get("params") or {})
            asset_target_weights = dict(raw_block.get("asset_target_weights") or {})
            raw_assets = list(raw_block.get("assets") or [])
            if raw_assets:
                assets = [
                    WalkForwardPortfolioStrategyBlockAssetDefinition(
                        dataset_id=str(asset.get("dataset_id") or ""),
                        target_weight=(
                            None
                            if asset.get("target_weight") in (None, "")
                            else float(asset.get("target_weight"))
                        ),
                        display_name=str(asset.get("display_name") or asset.get("dataset_id") or ""),
                    )
                    for asset in raw_assets
                    if str(asset.get("dataset_id") or "").strip()
                ]
            else:
                assets = [
                    WalkForwardPortfolioStrategyBlockAssetDefinition(
                        dataset_id=str(dataset_id),
                        target_weight=(
                            None
                            if asset_target_weights.get(dataset_id) in (None, "")
                            else float(asset_target_weights.get(dataset_id))
                        ),
                        display_name=str(dataset_id),
                    )
                    for dataset_id in list(raw_block.get("asset_dataset_ids") or [])
                    if str(dataset_id).strip()
                ]
            blocks.append(
                WalkForwardPortfolioStrategyBlockDefinition(
                    block_id=str(raw_block.get("block_id") or strategy_name or f"block_{len(blocks) + 1}"),
                    strategy_cls=strategy_cls,
                    strategy_params=params,
                    assets=assets,
                    budget_weight=(
                        None
                        if raw_block.get("budget_weight") in (None, "")
                        else float(raw_block.get("budget_weight"))
                    ),
                    display_name=str(raw_block.get("display_name") or raw_block.get("block_id") or strategy_name),
                )
            )
        return blocks

    @classmethod
    def _strategy_block_candidates_from_candidate_records(
        cls,
        records: Sequence,
    ) -> list[list[WalkForwardPortfolioStrategyBlockDefinition]]:
        candidates: list[list[WalkForwardPortfolioStrategyBlockDefinition]] = []
        for record in list(records or ()):
            params_json = getattr(record, "params_json", None)
            if not params_json and isinstance(record, dict):
                params_json = record.get("params_json")
            if not params_json:
                continue
            try:
                payload = json.loads(str(params_json))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            strategy_blocks = cls._strategy_blocks_from_batch_params(
                {"_portfolio_strategy_blocks": list(payload.get("strategy_blocks") or [])}
            )
            if strategy_blocks:
                candidates.append(strategy_blocks)
        return candidates

    @staticmethod
    def _construction_config_from_batch_params(batch_params: dict) -> PortfolioConstructionConfig:
        return PortfolioConstructionConfig(
            allocation_ownership=str(
                batch_params.get("_portfolio_ownership_mode", ALLOCATION_OWNERSHIP_STRATEGY)
                or ALLOCATION_OWNERSHIP_STRATEGY
            ),
            ranking_mode=str(batch_params.get("_portfolio_ranking_mode", RANKING_MODE_NONE) or RANKING_MODE_NONE),
            max_ranked_assets=int(batch_params.get("_portfolio_rank_count", 1) or 1),
            min_rank_score=float(batch_params.get("_portfolio_score_threshold", 0.0) or 0.0),
            weighting_mode=str(batch_params.get("_portfolio_weighting_mode", WEIGHTING_MODE_PRESERVE) or WEIGHTING_MODE_PRESERVE),
            min_active_weight=(float(batch_params.get("_portfolio_min_active_weight", 0.0) or 0.0) or None),
            max_asset_weight=(float(batch_params.get("_portfolio_max_asset_weight", 0.0) or 0.0) or None),
            cash_reserve_weight=float(batch_params.get("_portfolio_cash_reserve_weight", 0.0) or 0.0),
            rebalance_mode=str(batch_params.get("_portfolio_rebalance_mode", REBALANCE_MODE_ON_CHANGE) or REBALANCE_MODE_ON_CHANGE),
            rebalance_every_n_bars=int(batch_params.get("_portfolio_rebalance_every_n_bars", 20) or 20),
            rebalance_weight_drift_threshold=float(
                batch_params.get("_portfolio_rebalance_drift_threshold", 0.05) or 0.05
            ),
        )


class MonteCarloWorker(QtCore.QThread):
    finished_signal = QtCore.pyqtSignal(object)
    error_signal = QtCore.pyqtSignal(str)

    def __init__(
        self,
        *,
        catalog_path: Path,
        source_study_row: dict,
        resampling_mode: str,
        simulation_count: int,
        seed: int | None,
        cost_stress_bps: float,
        description: str,
    ) -> None:
        super().__init__()
        self.catalog_path = Path(catalog_path)
        self.source_study_row = dict(source_study_row)
        self.resampling_mode = str(resampling_mode or MONTE_CARLO_MODE_BOOTSTRAP)
        self.simulation_count = int(simulation_count)
        self.seed = int(seed) if seed is not None else None
        self.cost_stress_bps = float(cost_stress_bps)
        self.description = str(description or "")
        self.mc_study_id = f"mc_{uuid.uuid4().hex[:8]}"

    def run(self) -> None:
        try:
            catalog = ResultCatalog(self.catalog_path)
            source_id = str(self.source_study_row.get("wf_study_id", "") or "")
            if not source_id:
                raise RuntimeError("Monte Carlo setup is missing a source walk-forward study.")
            artifacts = run_monte_carlo_study(
                mc_study_id=self.mc_study_id,
                source_type=MONTE_CARLO_SOURCE_WALK_FORWARD,
                source_id=source_id,
                catalog=catalog,
                resampling_mode=self.resampling_mode,
                simulation_count=self.simulation_count,
                seed=self.seed,
                cost_stress_bps=self.cost_stress_bps,
                description=self.description or f"Monte Carlo from walk-forward study {source_id}",
            )
            self.finished_signal.emit(
                {
                    "mc_study_id": self.mc_study_id,
                    "source_id": source_id,
                    "artifacts": artifacts,
                    "message": (
                        f"Monte Carlo study completed with {artifacts.simulation_count} simulation"
                        f"{'' if artifacts.simulation_count == 1 else 's'}."
                    ),
                }
            )
        except Exception as exc:
            tb = traceback.format_exc()
            print("MonteCarloWorker error:\n", tb)
            self.error_signal.emit(tb if tb else str(exc))


class DashboardDialog(QtWidgets.QDialog):
    def __init__(self, parent=None) -> None:
        self._logical_parent = parent
        # Keep these as normal decorated dialogs, but do not make them transient
        # children of intermediate dialogs. That way they get proper title bars
        # and can be moved independently by the window manager.
        super().__init__(None)
        self._geometry_clamped_once = False

    def logical_parent(self):
        return self._logical_parent if self._logical_parent is not None else self.parent()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        if self._geometry_clamped_once:
            return
        self._geometry_clamped_once = True
        self._clamp_to_available_screen()

    def _clamp_to_available_screen(self) -> None:
        screen = self.screen()
        if screen is None:
            parent_widget = self.logical_parent()
            if parent_widget is not None and parent_widget.windowHandle() is not None:
                screen = parent_widget.windowHandle().screen()
        if screen is None:
            screen = QtWidgets.QApplication.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry().adjusted(12, 12, -12, -12)
        if available.width() <= 0 or available.height() <= 0:
            return

        width = min(self.width(), available.width())
        height = min(self.height(), available.height())
        if width != self.width() or height != self.height():
            self.resize(width, height)

        geom = self.frameGeometry()
        x = geom.x()
        y = geom.y()
        parent_widget = self.logical_parent()
        if isinstance(parent_widget, QtWidgets.QWidget):
            anchor = parent_widget.frameGeometry().center()
            x = anchor.x() - (geom.width() // 2)
            y = anchor.y() - (geom.height() // 2)
        else:
            x = available.center().x() - (geom.width() // 2)
            y = available.center().y() - (geom.height() // 2)
        if geom.width() > available.width():
            x = available.left()
        else:
            x = max(available.left(), min(x, available.right() - geom.width() + 1))
        if geom.height() > available.height():
            y = available.top()
        else:
            y = max(available.top(), min(y, available.bottom() - geom.height() + 1))
        self.move(x, y)


class DatasetSelectionDialog(DashboardDialog):
    def __init__(self, datasets: Sequence[str], selected: Sequence[str], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Choose Study Datasets")
        self.resize(420, 520)
        self.setMinimumSize(420, 520)
        self.setObjectName("Panel")
        self.setStyleSheet(
            f"""
            QDialog#Panel {{
                background: {PALETTE['panel']};
                color: {PALETTE['text']};
            }}
            QLabel {{
                color: {PALETTE['text']};
            }}
            QLabel#Sub {{
                color: {PALETTE['muted']};
            }}
            QListWidget {{
                background: {PALETTE['panel2']};
                color: {PALETTE['text']};
                border: 1px solid rgba(231, 238, 252, 0.45);
                border-radius: 10px;
                padding: 6px;
                outline: 0;
            }}
            QListWidget::item {{
                color: {PALETTE['text']};
                background: transparent;
                padding: 8px 10px;
                border-radius: 8px;
            }}
            QListWidget::item:selected {{
                background: rgba(77, 163, 255, 0.24);
                color: {PALETTE['text']};
            }}
            QListWidget::item:hover {{
                background: rgba(255, 255, 255, 0.08);
            }}
            QPushButton {{
                background: rgba(255, 255, 255, 0.08);
                color: {PALETTE['text']};
                border: 1px solid rgba(231, 238, 252, 0.45);
                border-radius: 8px;
                padding: 7px 12px;
            }}
            QPushButton:hover {{
                border-color: {PALETTE['blue']};
                background: rgba(77, 163, 255, 0.15);
            }}
            QPushButton:disabled {{
                color: rgba(231, 238, 252, 0.45);
                border-color: rgba(231, 238, 252, 0.25);
                background: rgba(255, 255, 255, 0.03);
            }}
            QDialogButtonBox QPushButton {{
                min-width: 88px;
            }}
            """
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)
        label = QtWidgets.QLabel(
            "Select one or more datasets for an independent multi-dataset study.\n"
            "Each dataset will be backtested separately under the same study."
        )
        label.setObjectName("Sub")
        label.setWordWrap(True)
        layout.addWidget(label)

        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.MultiSelection)
        selected_set = set(selected)
        for dataset_id in datasets:
            item = QtWidgets.QListWidgetItem(dataset_id)
            self.list_widget.addItem(item)
            if dataset_id in selected_set:
                item.setSelected(True)
        layout.addWidget(self.list_widget)

        actions = QtWidgets.QHBoxLayout()
        select_all_btn = QtWidgets.QPushButton("Select All")
        select_all_btn.clicked.connect(self._select_all)
        clear_btn = QtWidgets.QPushButton("Clear")
        clear_btn.clicked.connect(self.list_widget.clearSelection)
        actions.addWidget(select_all_btn)
        actions.addWidget(clear_btn)
        actions.addStretch(1)
        layout.addLayout(actions)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _select_all(self) -> None:
        for row in range(self.list_widget.count()):
            item = self.list_widget.item(row)
            if item is not None:
                item.setSelected(True)

    def selected_datasets(self) -> list[str]:
        return [item.text() for item in self.list_widget.selectedItems()]

    def accept(self) -> None:
        if not self.selected_datasets():
            QtWidgets.QMessageBox.information(self, "No datasets selected", "Select at least one dataset.")
            return
        super().accept()


class DatasetPickerDialog(DashboardDialog):
    def __init__(self, datasets: Sequence[str], selected_dataset: str = "", parent=None, *, title: str = "Choose Dataset") -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(420, 520)
        self.setMinimumSize(420, 520)
        self.setObjectName("Panel")
        self.setStyleSheet(
            f"""
            QDialog#Panel {{
                background: {PALETTE['panel']};
                color: {PALETTE['text']};
            }}
            QLabel {{
                color: {PALETTE['text']};
            }}
            QLabel#Sub {{
                color: {PALETTE['muted']};
            }}
            QListWidget {{
                background: {PALETTE['panel2']};
                color: {PALETTE['text']};
                border: 1px solid rgba(231, 238, 252, 0.45);
                border-radius: 10px;
                padding: 6px;
                outline: 0;
            }}
            QListWidget::item {{
                color: {PALETTE['text']};
                background: transparent;
                padding: 8px 10px;
                border-radius: 8px;
            }}
            QListWidget::item:selected {{
                background: rgba(77, 163, 255, 0.24);
                color: {PALETTE['text']};
            }}
            QListWidget::item:hover {{
                background: rgba(255, 255, 255, 0.08);
            }}
            QPushButton {{
                background: rgba(255, 255, 255, 0.08);
                color: {PALETTE['text']};
                border: 1px solid rgba(231, 238, 252, 0.45);
                border-radius: 8px;
                padding: 7px 12px;
            }}
            QPushButton:hover {{
                border-color: {PALETTE['blue']};
                background: rgba(77, 163, 255, 0.15);
            }}
            QPushButton:disabled {{
                color: rgba(231, 238, 252, 0.45);
                border-color: rgba(231, 238, 252, 0.25);
                background: rgba(255, 255, 255, 0.03);
            }}
            """
        )
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)
        label = QtWidgets.QLabel("Choose the dataset whose acquisition metadata and history you want to inspect.")
        label.setObjectName("Sub")
        label.setWordWrap(True)
        layout.addWidget(label)

        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        current_row = 0
        for idx, dataset_id in enumerate(datasets):
            item = QtWidgets.QListWidgetItem(dataset_id)
            self.list_widget.addItem(item)
            if dataset_id == selected_dataset:
                current_row = idx
        if self.list_widget.count() > 0:
            self.list_widget.setCurrentRow(current_row)
        layout.addWidget(self.list_widget)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_dataset(self) -> str:
        item = self.list_widget.currentItem()
        return item.text().strip() if item is not None else ""

    def accept(self) -> None:
        if not self.selected_dataset():
            QtWidgets.QMessageBox.information(self, "No dataset selected", "Select one dataset.")
            return
        super().accept()


class ProviderSettingsDialog(DashboardDialog):
    def __init__(self, catalog: ResultCatalog, parent=None) -> None:
        super().__init__(parent)
        self.catalog = catalog
        self.setWindowTitle("Provider Settings")
        self.resize(760, 520)
        self.setMinimumSize(700, 500)
        self.setObjectName("Panel")
        self.setStyleSheet(
            f"""
            QDialog#Panel {{
                background: {PALETTE['panel']};
                color: {PALETTE['text']};
            }}
            QLabel {{
                color: {PALETTE['text']};
            }}
            QLabel#Sub {{
                color: {PALETTE['muted']};
            }}
            QGroupBox {{
                border: 1px solid rgba(231, 238, 252, 0.35);
                border-radius: 10px;
                margin-top: 12px;
                padding: 10px 12px 12px 12px;
                font-weight: 700;
                color: {PALETTE['muted']};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
            }}
            QLineEdit, QSpinBox {{
                background: {PALETTE['panel2']};
                color: {PALETTE['text']};
                border: 1px solid rgba(231, 238, 252, 0.45);
                border-radius: 8px;
                padding: 6px 8px;
            }}
            QPushButton {{
                background: rgba(255,255,255,.08);
                color: {PALETTE['text']};
                border: 1px solid rgba(231, 238, 252, 0.55);
                border-radius: 8px;
                padding: 6px 12px;
            }}
            QPushButton:hover {{
                border-color: {PALETTE['blue']};
                background: rgba(77,163,255,.15);
            }}
            """
        )

        massive_secrets = load_provider_secrets("massive")
        ib_settings = load_provider_settings("interactive_brokers", catalog=self.catalog)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        title = QtWidgets.QLabel("Provider Settings")
        title.setObjectName("Title")
        subtitle = QtWidgets.QLabel(
            "Configure saved provider credentials and connection settings. "
            "Non-secret settings are saved in SQLite. Secrets are saved locally in the data directory."
        )
        subtitle.setObjectName("Sub")
        subtitle.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(subtitle)

        massive_box = QtWidgets.QGroupBox("Massive / Polygon")
        massive_form = QtWidgets.QFormLayout(massive_box)
        self.massive_api_key_edit = QtWidgets.QLineEdit(str(massive_secrets.get("api_key") or ""))
        self.massive_api_key_edit.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self.massive_api_key_edit.setPlaceholderText("Paste Massive API key")
        massive_form.addRow("API Key", self.massive_api_key_edit)
        massive_note = QtWidgets.QLabel("Used for interactive and scheduled Massive downloads.")
        massive_note.setObjectName("Sub")
        massive_note.setWordWrap(True)
        massive_form.addRow("", massive_note)
        layout.addWidget(massive_box)

        ib_box = QtWidgets.QGroupBox("Interactive Brokers")
        ib_form = QtWidgets.QFormLayout(ib_box)
        self.ib_host_edit = QtWidgets.QLineEdit(str(ib_settings.get("host") or "127.0.0.1"))
        self.ib_port_spin = QtWidgets.QSpinBox()
        self.ib_port_spin.setRange(1, 65535)
        self.ib_port_spin.setValue(int(ib_settings.get("port") or 7497))
        self.ib_client_id_spin = QtWidgets.QSpinBox()
        self.ib_client_id_spin.setRange(1, 999999)
        self.ib_client_id_spin.setValue(int(ib_settings.get("client_id") or 9301))
        ib_form.addRow("Host", self.ib_host_edit)
        ib_form.addRow("Port", self.ib_port_spin)
        ib_form.addRow("Client ID", self.ib_client_id_spin)
        ib_note = QtWidgets.QLabel(
            "These values are used to connect to a running TWS / IB Gateway session."
        )
        ib_note.setObjectName("Sub")
        ib_note.setWordWrap(True)
        ib_form.addRow("", ib_note)
        ib_helper_row = QtWidgets.QHBoxLayout()
        ib_helper_btn = QtWidgets.QPushButton("Install / Validate IB")
        ib_helper_btn.clicked.connect(self._open_ib_setup_dialog)
        ib_helper_row.addWidget(ib_helper_btn)
        ib_helper_row.addStretch(1)
        ib_form.addRow("", ib_helper_row)
        layout.addWidget(ib_box)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Save | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def accept(self) -> None:
        save_provider_secrets(
            "massive",
            {"api_key": self.massive_api_key_edit.text().strip()},
        )
        save_provider_settings(
            "interactive_brokers",
            {
                "host": self.ib_host_edit.text().strip(),
                "port": int(self.ib_port_spin.value()),
                "client_id": int(self.ib_client_id_spin.value()),
            },
            catalog=self.catalog,
        )
        super().accept()

    def _open_ib_setup_dialog(self) -> None:
        dlg = IBSetupDialog(
            host=self.ib_host_edit.text().strip(),
            port=int(self.ib_port_spin.value()),
            client_id=int(self.ib_client_id_spin.value()),
            parent=self,
        )
        dlg.exec()


class IBSetupDialog(DashboardDialog):
    def __init__(self, *, host: str, port: int, client_id: int, parent=None) -> None:
        super().__init__(parent)
        self._host = str(host or "").strip() or "127.0.0.1"
        self._port = int(port or 7497)
        self._client_id = int(client_id or 9301)
        self.setWindowTitle("Install / Validate Interactive Brokers")
        self.resize(760, 560)
        self.setMinimumSize(700, 520)
        self.setObjectName("Panel")
        self.setStyleSheet(
            f"""
            QDialog#Panel {{
                background: {PALETTE['panel']};
                color: {PALETTE['text']};
            }}
            QLabel {{
                color: {PALETTE['text']};
            }}
            QLabel#Sub {{
                color: {PALETTE['muted']};
            }}
            QPlainTextEdit, QLineEdit {{
                background: {PALETTE['panel2']};
                color: {PALETTE['text']};
                border: 1px solid rgba(231, 238, 252, 0.45);
                border-radius: 8px;
                padding: 6px 8px;
            }}
            QPushButton {{
                background: rgba(255,255,255,.08);
                color: {PALETTE['text']};
                border: 1px solid rgba(231, 238, 252, 0.55);
                border-radius: 8px;
                padding: 6px 12px;
            }}
            QPushButton:hover {{
                border-color: {PALETTE['blue']};
                background: rgba(77,163,255,.15);
            }}
            """
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        title = QtWidgets.QLabel("Interactive Brokers Setup")
        title.setObjectName("Title")
        subtitle = QtWidgets.QLabel(
            "On Ubuntu, install the Python package in this project venv, install IB Gateway separately from IB, "
            "then validate that the saved host/port are reachable."
        )
        subtitle.setObjectName("Sub")
        subtitle.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(subtitle)

        install_cmd = ibapi_install_command()
        self.command_edit = QtWidgets.QLineEdit(install_cmd)
        self.command_edit.setReadOnly(True)
        layout.addWidget(QtWidgets.QLabel("Install Command"))
        layout.addWidget(self.command_edit)

        helper_text = QtWidgets.QPlainTextEdit()
        helper_text.setReadOnly(True)
        helper_text.setMaximumHeight(180)
        helper_text.setPlainText(
            "\n".join(
                [
                    "Ubuntu checklist:",
                    "1. Install the Python package in the project venv.",
                    f"   {install_cmd}",
                    "2. Install IB Gateway separately from Interactive Brokers and launch it.",
                    "3. In IB Gateway or TWS, enable API socket access.",
                    f"4. Use host {self._host}, port {self._port}, client_id {self._client_id} in Provider Settings.",
                    "5. Press 'Validate Current Settings' below to test local reachability.",
                ]
            )
        )
        layout.addWidget(helper_text)

        self.package_status_label = QtWidgets.QLabel()
        self.package_status_label.setWordWrap(True)
        self.connection_status_label = QtWidgets.QLabel()
        self.connection_status_label.setWordWrap(True)
        self.connection_status_label.setObjectName("Sub")
        self.probe_symbol_edit = QtWidgets.QLineEdit("AAPL")
        self.probe_symbol_edit.setPlaceholderText("Probe symbol for head timestamp, e.g. AAPL")
        probe_row = QtWidgets.QHBoxLayout()
        probe_row.addWidget(QtWidgets.QLabel("Probe Symbol"))
        probe_row.addWidget(self.probe_symbol_edit, 1)
        layout.addWidget(self.package_status_label)
        layout.addWidget(self.connection_status_label)
        layout.addLayout(probe_row)

        buttons = QtWidgets.QHBoxLayout()
        copy_btn = QtWidgets.QPushButton("Copy Install Command")
        copy_btn.clicked.connect(self._copy_install_command)
        validate_btn = QtWidgets.QPushButton("Validate Current Settings")
        validate_btn.clicked.connect(self._validate)
        deep_validate_btn = QtWidgets.QPushButton("Deep Validate (API + Head Timestamp)")
        deep_validate_btn.clicked.connect(self._deep_validate)
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        buttons.addWidget(copy_btn)
        buttons.addWidget(validate_btn)
        buttons.addWidget(deep_validate_btn)
        buttons.addStretch(1)
        buttons.addWidget(close_btn)
        layout.addLayout(buttons)

        self._refresh_package_status()
        self.connection_status_label.setText(
            f"Saved connection: {self._host}:{self._port} with client_id {self._client_id}. Validation has not been run yet."
        )

    def _copy_install_command(self) -> None:
        QtWidgets.QApplication.clipboard().setText(self.command_edit.text())
        self.connection_status_label.setText("Install command copied to clipboard.")

    def _refresh_package_status(self) -> None:
        status = ibapi_package_status()
        prefix = "Package Status: "
        self.package_status_label.setText(prefix + status.message)
        color = PALETTE["green"] if status.ok else PALETTE["amber"]
        self.package_status_label.setStyleSheet(f"color: {color};")

    def _validate(self) -> None:
        self._refresh_package_status()
        socket_status = interactive_brokers_socket_status(self._host, self._port)
        color = PALETTE["green"] if socket_status.ok else PALETTE["red"]
        self.connection_status_label.setStyleSheet(f"color: {color};")
        self.connection_status_label.setText(
            f"Connection Status: {socket_status.message} Client ID configured: {self._client_id}."
        )

    def _deep_validate(self) -> None:
        self._refresh_package_status()
        api_status = interactive_brokers_api_status(self._host, self._port, self._client_id)
        if not api_status.ok:
            self.connection_status_label.setStyleSheet(f"color: {PALETTE['red']};")
            self.connection_status_label.setText(f"Deep Validation: {api_status.message}")
            return
        probe_symbol = self.probe_symbol_edit.text().strip().upper() or "AAPL"
        head_status = interactive_brokers_head_timestamp_status(
            self._host,
            self._port,
            self._client_id,
            symbol=probe_symbol,
        )
        color = PALETTE["green"] if head_status.ok else PALETTE["amber"]
        self.connection_status_label.setStyleSheet(f"color: {color};")
        self.connection_status_label.setText(
            f"Deep Validation: {api_status.message} {head_status.message}"
        )


class ScheduledTaskEditorDialog(DashboardDialog):
    def __init__(self, task: dict, universes: Sequence[dict], symbols: Sequence[str], parent=None) -> None:
        super().__init__(parent)
        self.task = dict(task or {})
        self.universes = [dict(item) for item in list(universes or ())]
        self.symbols = [str(item) for item in list(symbols or ()) if str(item).strip()]
        payload = self.task.get("symbols") if isinstance(self.task.get("symbols"), dict) else {}
        schedule = self.task.get("schedule") if isinstance(self.task.get("schedule"), dict) else {}
        self._selected_symbols = [
            str(item).upper()
            for item in list(payload.get("symbols") or [])
            if str(item).strip()
        ]

        self.setWindowTitle("Edit Scheduled Task")
        self.resize(760, 620)
        self.setMinimumSize(700, 560)
        self.setObjectName("Panel")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        intro = QtWidgets.QLabel(
            "Edit the saved universe/manual symbols and the recurring schedule for this automated download task."
        )
        intro.setObjectName("Sub")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(10)

        self.universe_combo = QtWidgets.QComboBox()
        self.universe_combo.addItem("Manual Symbols", "")
        for universe in self.universes:
            label = str(universe.get("name") or universe.get("universe_id") or "Universe")
            self.universe_combo.addItem(label, str(universe.get("universe_id") or ""))
        existing_universe_id = str(payload.get("universe_id") or "")
        if existing_universe_id:
            idx = self.universe_combo.findData(existing_universe_id)
            self.universe_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.universe_combo.currentIndexChanged.connect(self._update_symbol_summary)
        form.addRow("Universe", self.universe_combo)

        symbol_wrap = QtWidgets.QWidget()
        symbol_layout = QtWidgets.QVBoxLayout(symbol_wrap)
        symbol_layout.setContentsMargins(0, 0, 0, 0)
        symbol_layout.setSpacing(8)
        self.symbol_summary = QtWidgets.QLineEdit()
        self.symbol_summary.setReadOnly(True)
        btn_row = QtWidgets.QHBoxLayout()
        choose_symbols_btn = QtWidgets.QPushButton("Choose Symbols")
        choose_symbols_btn.clicked.connect(self._choose_symbols)
        clear_symbols_btn = QtWidgets.QPushButton("Clear")
        clear_symbols_btn.clicked.connect(self._clear_symbols)
        btn_row.addWidget(choose_symbols_btn)
        btn_row.addWidget(clear_symbols_btn)
        btn_row.addStretch(1)
        self.manual_symbol_buttons = (choose_symbols_btn, clear_symbols_btn)
        symbol_layout.addWidget(self.symbol_summary)
        symbol_layout.addLayout(btn_row)
        form.addRow("Symbols", symbol_wrap)

        self.source_combo = QtWidgets.QComboBox()
        self.source_combo.addItem("Auto (Use Universe Preference)", "")
        for provider in available_acquisition_providers():
            self.source_combo.addItem(provider.label, provider.provider_id)
        source_idx = self.source_combo.findData(str(payload.get("source") or ""))
        self.source_combo.setCurrentIndex(source_idx if source_idx >= 0 else 0)
        form.addRow("Source", self.source_combo)

        self.force_refresh_chk = QtWidgets.QCheckBox("Force refresh even if local data looks fresh")
        self.force_refresh_chk.setChecked(bool(payload.get("force_refresh", False)))
        form.addRow("Refresh Policy", self.force_refresh_chk)

        self.frequency_combo = QtWidgets.QComboBox()
        self.frequency_combo.addItems(["Nightly", "Weekly", "Monthly"])
        freq_idx = self.frequency_combo.findText(str(schedule.get("frequency") or "Nightly"))
        self.frequency_combo.setCurrentIndex(freq_idx if freq_idx >= 0 else 0)
        form.addRow("Frequency", self.frequency_combo)

        self.time_edit = QtWidgets.QTimeEdit()
        self.time_edit.setDisplayFormat("hh:mm AP")
        schedule_time = QtCore.QTime.fromString(str(schedule.get("time") or "00:00"), "HH:mm")
        if not schedule_time.isValid():
            schedule_time = QtCore.QTime.currentTime()
        self.time_edit.setTime(schedule_time)
        form.addRow("Start Time", self.time_edit)

        self.weekday_checks = []
        weekday_wrap = QtWidgets.QWidget()
        weekday_layout = QtWidgets.QHBoxLayout(weekday_wrap)
        weekday_layout.setContentsMargins(0, 0, 0, 0)
        weekday_layout.setSpacing(6)
        selected_days = {str(item) for item in list(schedule.get("days") or [])}
        for label in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]:
            chk = QtWidgets.QCheckBox(label)
            chk.setChecked(label in selected_days)
            self.weekday_checks.append(chk)
            weekday_layout.addWidget(chk)
        weekday_layout.addStretch(1)
        form.addRow("Days", weekday_wrap)

        self.week_of_month_checks = []
        weeks_wrap = QtWidgets.QWidget()
        weeks_layout = QtWidgets.QHBoxLayout(weeks_wrap)
        weeks_layout.setContentsMargins(0, 0, 0, 0)
        weeks_layout.setSpacing(6)
        selected_weeks = {str(item) for item in list(schedule.get("weeks") or [])}
        for label in ["1st", "2nd", "3rd", "4th", "5th"]:
            chk = QtWidgets.QCheckBox(label)
            chk.setChecked(label in selected_weeks)
            self.week_of_month_checks.append(chk)
            weeks_layout.addWidget(chk)
        weeks_layout.addStretch(1)
        form.addRow("Weeks", weeks_wrap)

        self.month_checks = []
        months_wrap = QtWidgets.QWidget()
        months_layout = QtWidgets.QGridLayout(months_wrap)
        months_layout.setContentsMargins(0, 0, 0, 0)
        months_layout.setHorizontalSpacing(6)
        months_layout.setVerticalSpacing(4)
        selected_months = {str(item) for item in list(schedule.get("months") or [])}
        month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        for idx, label in enumerate(month_labels):
            chk = QtWidgets.QCheckBox(label)
            chk.setChecked(label in selected_months)
            self.month_checks.append(chk)
            months_layout.addWidget(chk, idx // 4, idx % 4)
        form.addRow("Months", months_wrap)

        layout.addLayout(form)
        self.status_note = QtWidgets.QLabel()
        self.status_note.setObjectName("Sub")
        self.status_note.setWordWrap(True)
        layout.addWidget(self.status_note)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addStretch(1)
        layout.addWidget(buttons)

        self._update_symbol_summary()

    def _selected_universe(self) -> dict | None:
        universe_id = str(self.universe_combo.currentData() or "").strip()
        if not universe_id:
            return None
        for universe in self.universes:
            if str(universe.get("universe_id") or "") == universe_id:
                return universe
        return None

    def _update_symbol_summary(self) -> None:
        universe = self._selected_universe()
        manual_mode = universe is None
        for button in self.manual_symbol_buttons:
            button.setEnabled(manual_mode)
        if universe is not None:
            symbols = [str(item).upper() for item in list(universe.get("symbols") or []) if str(item).strip()]
            datasets = [str(item) for item in list(universe.get("dataset_ids") or []) if str(item).strip()]
            self.symbol_summary.setText(f"{len(symbols)} symbols from universe '{universe.get('name') or universe.get('universe_id')}'")
            self.symbol_summary.setToolTip("\n".join(symbols))
            self.status_note.setText(
                f"Universe mode is active. This task will use {len(symbols)} symbol(s) and {len(datasets)} linked dataset binding(s) from the saved universe."
            )
            return
        if not self._selected_symbols:
            self.symbol_summary.setText("No manual symbols selected")
            self.symbol_summary.setToolTip("")
        elif len(self._selected_symbols) <= 6:
            self.symbol_summary.setText(", ".join(self._selected_symbols))
            self.symbol_summary.setToolTip("\n".join(self._selected_symbols))
        else:
            self.symbol_summary.setText(f"{len(self._selected_symbols)} manual symbols selected")
            self.symbol_summary.setToolTip("\n".join(self._selected_symbols))
        self.status_note.setText("Manual symbol mode is active.")

    def _choose_symbols(self) -> None:
        if not self.symbols:
            QtWidgets.QMessageBox.warning(self, "Symbols missing", f"Symbols file not found or empty: {NASDAQ_SYMBOLS_PATH}")
            return
        dlg = TickerPickerDialog(self.symbols, set(self._selected_symbols), self)
        if dlg.exec() != int(QtWidgets.QDialog.DialogCode.Accepted):
            return
        self._selected_symbols = [str(item).upper() for item in list(dlg.selected) if str(item).strip()]
        self._update_symbol_summary()

    def _clear_symbols(self) -> None:
        self._selected_symbols = []
        self._update_symbol_summary()

    def result_payload(self) -> tuple[dict, dict]:
        universe = self._selected_universe()
        if universe is not None:
            symbols = [str(item).upper() for item in list(universe.get("symbols") or []) if str(item).strip()]
            universe_id = str(universe.get("universe_id") or "")
            universe_name = str(universe.get("name") or "")
        else:
            symbols = [str(item).upper() for item in self._selected_symbols if str(item).strip()]
            universe_id = ""
            universe_name = ""
        schedule = {
            "frequency": self.frequency_combo.currentText(),
            "time": self.time_edit.time().toString("HH:mm"),
            "days": [chk.text() for chk in self.weekday_checks if chk.isChecked()],
            "weeks": [chk.text() for chk in self.week_of_month_checks if chk.isChecked()],
            "months": [chk.text() for chk in self.month_checks if chk.isChecked()],
        }
        payload = {
            "symbols": symbols,
            "universe_id": universe_id,
            "universe_name": universe_name,
            "source": resolve_acquisition_source(
                str(self.source_combo.currentData() or ""),
                str((universe or {}).get("source_preference") or ""),
            ),
            "force_refresh": bool(self.force_refresh_chk.isChecked()),
            "resolution": str((self.task.get("symbols") or {}).get("resolution") or "1m"),
            "history": str((self.task.get("symbols") or {}).get("history") or "2y"),
            "schedule": schedule,
        }
        return payload, schedule

    def accept(self) -> None:
        payload, _schedule = self.result_payload()
        symbols = [str(item) for item in list(payload.get("symbols") or []) if str(item).strip()]
        if not symbols:
            QtWidgets.QMessageBox.warning(self, "No symbols", "Choose at least one manual symbol or a universe with symbols.")
            return
        super().accept()


class AcquisitionCatalogDialog(DashboardDialog):
    def __init__(
        self,
        catalog: CatalogReader,
        universes: Sequence[Dict],
        parent=None,
        *,
        selected_universe_id: str = "",
        selected_task_id: str = "",
        selected_dataset_id: str = "",
    ) -> None:
        super().__init__(parent)
        self.catalog = catalog
        self.universes = list(universes or [])
        self.selected_universe_id = str(selected_universe_id or "")
        self.selected_task_id = str(selected_task_id or "")
        self.selected_dataset_id = str(selected_dataset_id or "")
        self.dataset_frame_raw = pd.DataFrame()
        self.run_frame_raw = pd.DataFrame()
        self.attempt_frame_raw = pd.DataFrame()
        self.task_run_frame_raw = pd.DataFrame()
        self.dataset_frame = pd.DataFrame()
        self.run_frame = pd.DataFrame()
        self.attempt_frame = pd.DataFrame()
        self.task_run_frame = pd.DataFrame()

        self.setWindowTitle("Acquisition Catalog And History")
        self.resize(1480, 920)
        self.setMinimumSize(1240, 780)
        self.setObjectName("Panel")
        self.setStyleSheet(
            f"""
            QDialog#Panel {{
                background: {PALETTE['panel']};
                color: {PALETTE['text']};
            }}
            QLabel {{
                color: {PALETTE['text']};
            }}
            QLabel#Sub {{
                color: {PALETTE['muted']};
            }}
            QTableWidget {{
                background: {PALETTE['panel2']};
                color: {PALETTE['text']};
                border: 1px solid {PALETTE['border']};
                gridline-color: {PALETTE['grid']};
            }}
            QHeaderView::section {{
                background: {PALETTE['panel']};
                color: {PALETTE['muted']};
                border: 1px solid {PALETTE['border']};
                padding: 6px 8px;
                font-weight: 600;
            }}
            QPlainTextEdit {{
                background: {PALETTE['panel2']};
                color: {PALETTE['text']};
                border: 1px solid {PALETTE['border']};
                border-radius: 10px;
            }}
            """
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        title = QtWidgets.QLabel("Acquisition Catalog")
        title.setObjectName("Title")
        intro = QtWidgets.QLabel(
            "Review downloaded symbols, dataset coverage, last ingest state, interactive download batches, and scheduled refresh history."
        )
        intro.setObjectName("Sub")
        intro.setWordWrap(True)
        self.summary_label = QtWidgets.QLabel("Loading acquisition metadata…")
        self.summary_label.setObjectName("Sub")
        self.summary_label.setWordWrap(True)

        actions = QtWidgets.QHBoxLayout()
        refresh_btn = QtWidgets.QPushButton("Refresh")
        refresh_btn.clicked.connect(self._reload)
        self.universe_filter_combo = QtWidgets.QComboBox()
        self.universe_filter_combo.addItem("All Universes", "")
        for universe in self.universes:
            self.universe_filter_combo.addItem(str(universe.get("name") or ""), str(universe.get("universe_id") or ""))
        if self.selected_universe_id:
            self.universe_filter_combo.blockSignals(True)
            idx = self.universe_filter_combo.findData(self.selected_universe_id)
            self.universe_filter_combo.setCurrentIndex(idx if idx >= 0 else 0)
            self.universe_filter_combo.blockSignals(False)
        self.universe_filter_combo.currentIndexChanged.connect(self._apply_filters)
        self.task_filter_edit = QtWidgets.QLineEdit(self.selected_task_id)
        self.task_filter_edit.setPlaceholderText("Optional task id filter")
        self.task_filter_edit.textChanged.connect(self._apply_filters)
        clear_filters_btn = QtWidgets.QPushButton("Clear Filters")
        clear_filters_btn.clicked.connect(self._clear_filters)
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        actions.addWidget(QtWidgets.QLabel("Universe"))
        actions.addWidget(self.universe_filter_combo, 1)
        actions.addWidget(QtWidgets.QLabel("Task ID"))
        actions.addWidget(self.task_filter_edit, 1)
        actions.addWidget(clear_filters_btn)
        actions.addWidget(refresh_btn)
        actions.addStretch(1)
        actions.addWidget(close_btn)

        layout.addWidget(title)
        layout.addWidget(intro)
        layout.addWidget(self.summary_label)
        layout.addLayout(actions)

        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs, 1)

        self._build_dataset_tab()
        self._build_runs_tab()
        self._build_attempts_tab()
        self._build_task_runs_tab()
        self._reload()

    def _make_table(self, columns: Sequence[str]) -> QtWidgets.QTableWidget:
        table = QtWidgets.QTableWidget(0, len(columns))
        table.setHorizontalHeaderLabels(list(columns))
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        table.setAlternatingRowColors(True)
        table.setTextElideMode(QtCore.Qt.TextElideMode.ElideNone)
        table.setWordWrap(False)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)
        table.horizontalHeader().setStretchLastSection(True)
        table.setObjectName("Panel")
        return table

    def _build_dataset_tab(self) -> None:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        dataset_actions = QtWidgets.QHBoxLayout()
        self.dataset_detail_btn = QtWidgets.QPushButton("Open Selected Dataset Detail")
        self.dataset_detail_btn.clicked.connect(self._open_selected_dataset_detail)
        dataset_actions.addWidget(self.dataset_detail_btn)
        dataset_actions.addStretch(1)
        self.dataset_table = self._make_table(
            [
                "Dataset",
                "Source",
                "Symbol",
                "Coverage",
                "Bars",
                "Freshness",
                "Last Download",
                "Last Ingest",
                "Status",
                "Ingested",
                "Last Error",
            ]
        )
        self.dataset_table.itemSelectionChanged.connect(self._on_dataset_selected)
        self.dataset_table.itemDoubleClicked.connect(lambda _item: self._open_selected_dataset_detail())
        self.dataset_detail = QtWidgets.QPlainTextEdit()
        self.dataset_detail.setReadOnly(True)
        self.dataset_detail.setMinimumHeight(140)
        self.dataset_attempt_table = self._make_table(
            ["Finished", "Run", "Symbol", "Dataset", "Status", "Bars", "Coverage", "Ingested", "Error"]
        )
        self.dataset_table.setColumnWidth(10, 420)
        self.dataset_attempt_table.setColumnWidth(8, 420)
        layout.addLayout(dataset_actions)
        layout.addWidget(self.dataset_table, 3)
        layout.addWidget(self.dataset_detail, 1)
        layout.addWidget(self.dataset_attempt_table, 2)
        self.tabs.addTab(panel, "Datasets")

    def _build_runs_tab(self) -> None:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        run_actions = QtWidgets.QHBoxLayout()
        self.run_log_btn = QtWidgets.QPushButton("Open Selected Log")
        self.run_log_btn.clicked.connect(self._open_selected_run_log)
        run_actions.addWidget(self.run_log_btn)
        run_actions.addStretch(1)
        self.run_table = self._make_table(
            [
                "Run ID",
                "Type",
                "Source",
                "Universe",
                "Task",
                "Started",
                "Finished",
                "Status",
                "Symbols",
                "Success",
                "Failed",
                "Ingested",
            ]
        )
        self.run_table.itemSelectionChanged.connect(self._on_run_selected)
        self.run_notes = QtWidgets.QPlainTextEdit()
        self.run_notes.setReadOnly(True)
        self.run_notes.setMinimumHeight(100)
        self.run_attempt_table = self._make_table(
            ["Seq", "Symbol", "Dataset", "Status", "Bars", "Coverage", "Ingested", "Error"]
        )
        self.run_attempt_table.setColumnWidth(7, 420)
        layout.addLayout(run_actions)
        layout.addWidget(self.run_table, 3)
        layout.addWidget(self.run_notes, 1)
        layout.addWidget(self.run_attempt_table, 2)
        self.tabs.addTab(panel, "Run History")

    def _build_attempts_tab(self) -> None:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.attempt_table = self._make_table(
            ["Finished", "Run", "Symbol", "Dataset", "Source", "Status", "Bars", "Coverage", "Ingested", "Error"]
        )
        self.attempt_table.setColumnWidth(9, 420)
        layout.addWidget(self.attempt_table)
        self.tabs.addTab(panel, "Attempt History")

    def _build_task_runs_tab(self) -> None:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        run_actions = QtWidgets.QHBoxLayout()
        self.task_run_log_btn = QtWidgets.QPushButton("Open Selected Scheduler Log")
        self.task_run_log_btn.clicked.connect(self._open_selected_task_run_log)
        run_actions.addWidget(self.task_run_log_btn)
        run_actions.addStretch(1)
        self.task_run_table = self._make_table(
            ["Run ID", "Task", "Started", "Finished", "Status", "Tickers", "Log", "Error"]
        )
        self.task_run_table.setColumnWidth(7, 420)
        self.task_run_table.itemSelectionChanged.connect(self._on_task_run_selected)
        self.task_run_notes = QtWidgets.QPlainTextEdit()
        self.task_run_notes.setReadOnly(True)
        self.task_run_notes.setMinimumHeight(120)
        layout.addLayout(run_actions)
        layout.addWidget(self.task_run_table, 3)
        layout.addWidget(self.task_run_notes, 1)
        self.tabs.addTab(panel, "Scheduler Runs")

    def _reload(self) -> None:
        self.dataset_frame_raw = self.catalog.load_acquisition_datasets()
        self.run_frame_raw = self.catalog.load_acquisition_runs(limit=250)
        self.attempt_frame_raw = self.catalog.load_acquisition_attempts(limit=500)
        self.task_run_frame_raw = self.catalog.load_task_runs(limit=250)
        self._apply_filters()

    def _clear_filters(self) -> None:
        self.selected_dataset_id = ""
        self.task_filter_edit.blockSignals(True)
        self.task_filter_edit.clear()
        self.task_filter_edit.blockSignals(False)
        self.universe_filter_combo.blockSignals(True)
        self.universe_filter_combo.setCurrentIndex(0)
        self.universe_filter_combo.blockSignals(False)
        self._apply_filters()

    def _apply_filters(self) -> None:
        selected_universe_id = str(self.universe_filter_combo.currentData() or "").strip() if hasattr(self, "universe_filter_combo") else ""
        task_id = str(self.task_filter_edit.text() or "").strip() if hasattr(self, "task_filter_edit") else ""
        dataset_id = str(self.selected_dataset_id or "").strip()

        self.dataset_frame = self.dataset_frame_raw.copy()
        self.run_frame = self.run_frame_raw.copy()
        self.attempt_frame = self.attempt_frame_raw.copy()
        self.task_run_frame = self.task_run_frame_raw.copy()

        universe = None
        if selected_universe_id:
            universe = next(
                (item for item in self.universes if str(item.get("universe_id") or "") == selected_universe_id),
                None,
            )
        universe_dataset_ids = {
            str(item)
            for item in list((universe or {}).get("dataset_ids") or [])
            if str(item).strip()
        }
        universe_symbols = {
            str(item).upper()
            for item in list((universe or {}).get("symbols") or [])
            if str(item).strip()
        }

        if selected_universe_id:
            if not self.run_frame.empty and "universe_id" in self.run_frame.columns:
                self.run_frame = self.run_frame[self.run_frame["universe_id"].astype(str) == selected_universe_id].reset_index(drop=True)
            if not self.attempt_frame.empty:
                attempt_mask = pd.Series(False, index=self.attempt_frame.index)
                if "universe_id" in self.attempt_frame.columns:
                    attempt_mask = attempt_mask | (self.attempt_frame["universe_id"].astype(str) == selected_universe_id)
                if universe_dataset_ids and "dataset_id" in self.attempt_frame.columns:
                    attempt_mask = attempt_mask | self.attempt_frame["dataset_id"].astype(str).isin(universe_dataset_ids)
                if universe_symbols and "symbol" in self.attempt_frame.columns:
                    attempt_mask = attempt_mask | self.attempt_frame["symbol"].astype(str).str.upper().isin(universe_symbols)
                self.attempt_frame = self.attempt_frame[attempt_mask].reset_index(drop=True)
            if not self.dataset_frame.empty:
                dataset_mask = pd.Series(False, index=self.dataset_frame.index)
                if universe_dataset_ids and "dataset_id" in self.dataset_frame.columns:
                    dataset_mask = dataset_mask | self.dataset_frame["dataset_id"].astype(str).isin(universe_dataset_ids)
                if universe_symbols and "symbol" in self.dataset_frame.columns:
                    dataset_mask = dataset_mask | self.dataset_frame["symbol"].astype(str).str.upper().isin(universe_symbols)
                self.dataset_frame = self.dataset_frame[dataset_mask].reset_index(drop=True)

        if task_id:
            if not self.run_frame.empty and "task_id" in self.run_frame.columns:
                self.run_frame = self.run_frame[self.run_frame["task_id"].astype(str) == task_id].reset_index(drop=True)
            if not self.attempt_frame.empty and "task_id" in self.attempt_frame.columns:
                self.attempt_frame = self.attempt_frame[self.attempt_frame["task_id"].astype(str) == task_id].reset_index(drop=True)
            if not self.task_run_frame.empty and "task_id" in self.task_run_frame.columns:
                self.task_run_frame = self.task_run_frame[self.task_run_frame["task_id"].astype(str) == task_id].reset_index(drop=True)
            if not self.dataset_frame.empty and not self.attempt_frame.empty:
                task_dataset_ids = {
                    str(item)
                    for item in list(self.attempt_frame.get("dataset_id", pd.Series(dtype=object)).dropna().astype(str))
                    if item.strip()
                }
                if task_dataset_ids:
                    self.dataset_frame = self.dataset_frame[
                        self.dataset_frame["dataset_id"].astype(str).isin(task_dataset_ids)
                    ].reset_index(drop=True)

        if dataset_id and not self.dataset_frame.empty:
            self.dataset_frame = self.dataset_frame[self.dataset_frame["dataset_id"].astype(str) == dataset_id].reset_index(drop=True)
        if dataset_id and not self.attempt_frame.empty:
            self.attempt_frame = self.attempt_frame[self.attempt_frame["dataset_id"].astype(str) == dataset_id].reset_index(drop=True)

        dataset_count = len(self.dataset_frame)
        ingested_count = int(self.dataset_frame["ingested"].fillna(False).astype(bool).sum()) if not self.dataset_frame.empty else 0
        failed_attempts = (
            int(self.attempt_frame["status"].isin(["download_error", "ingest_error", "failed"]).sum())
            if not self.attempt_frame.empty
            else 0
        )
        self.summary_label.setText(
            f"Datasets: {dataset_count} | Ingested datasets: {ingested_count} | "
            f"Recorded runs: {len(self.run_frame)} | Recorded attempts: {len(self.attempt_frame)} | "
            f"Scheduler runs: {len(self.task_run_frame)} | Failed attempts: {failed_attempts}"
        )
        self._refresh_dataset_table()
        self._refresh_run_table()
        self._refresh_attempt_table()
        self._refresh_task_run_table()

    def _refresh_dataset_table(self) -> None:
        table = self.dataset_table
        table.setRowCount(0)
        for _, row in self.dataset_frame.iterrows():
            idx = table.rowCount()
            table.insertRow(idx)
            coverage = "—"
            if row.get("coverage_start") and row.get("coverage_end"):
                coverage = f"{str(row['coverage_start'])[:10]} → {str(row['coverage_end'])[:10]}"
            freshness = compute_freshness_state(row.get("coverage_end"), str(row.get("resolution") or ""))
            values = [
                str(row.get("dataset_id") or ""),
                str(row.get("source") or "—"),
                str(row.get("symbol") or "—"),
                coverage,
                str(int(row.get("bar_count") or 0)) if pd.notna(row.get("bar_count")) else "—",
                freshness,
                self._fmt_ts(row.get("last_download_attempt_at")),
                self._fmt_ts(row.get("last_ingest_at")),
                str(row.get("last_status") or "—"),
                "Yes" if bool(row.get("ingested")) else "No",
                str(row.get("last_error") or "—"),
            ]
            for col, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                if col == 0:
                    item.setData(QtCore.Qt.ItemDataRole.UserRole, str(row.get("dataset_id") or ""))
                if col == 10:
                    item.setToolTip(str(row.get("last_error") or ""))
                table.setItem(idx, col, item)
        if table.rowCount() > 0:
            table.selectRow(0)

    def _refresh_run_table(self) -> None:
        table = self.run_table
        table.setRowCount(0)
        for _, row in self.run_frame.iterrows():
            idx = table.rowCount()
            table.insertRow(idx)
            values = [
                str(row.get("acquisition_run_id") or ""),
                str(row.get("trigger_type") or ""),
                str(row.get("source") or "—"),
                str(row.get("universe_name") or row.get("universe_id") or "—"),
                str(row.get("task_id") or "—"),
                self._fmt_ts(row.get("started_at")),
                self._fmt_ts(row.get("finished_at")),
                str(row.get("status") or ""),
                str(int(row.get("symbol_count") or 0)),
                str(int(row.get("success_count") or 0)),
                str(int(row.get("failed_count") or 0)),
                str(int(row.get("ingested_count") or 0)),
            ]
            for col, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                if col == 0:
                    item.setData(QtCore.Qt.ItemDataRole.UserRole, str(row.get("acquisition_run_id") or ""))
                table.setItem(idx, col, item)
        if table.rowCount() > 0:
            table.selectRow(0)

    def _refresh_attempt_table(self) -> None:
        self._populate_attempt_table(self.attempt_table, self.attempt_frame, include_source=True)

    def _refresh_task_run_table(self) -> None:
        table = self.task_run_table
        table.setRowCount(0)
        for _, row in self.task_run_frame.iterrows():
            idx = table.rowCount()
            table.insertRow(idx)
            values = [
                str(row.get("run_id") or ""),
                str(row.get("task_id") or "—"),
                self._fmt_ts(row.get("started_at")),
                self._fmt_ts(row.get("finished_at")),
                str(row.get("status") or "—"),
                str(int(row.get("ticker_count") or 0)) if pd.notna(row.get("ticker_count")) else "—",
                str(Path(str(row.get("log_path") or "")).name or "—"),
                str(row.get("error_message") or "—"),
            ]
            for col, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                if col == 0:
                    item.setData(QtCore.Qt.ItemDataRole.UserRole, str(row.get("run_id") or ""))
                if col == 7:
                    item.setToolTip(str(row.get("error_message") or ""))
                table.setItem(idx, col, item)
        if table.rowCount() > 0:
            table.selectRow(0)

    def _populate_attempt_table(
        self,
        table: QtWidgets.QTableWidget,
        frame: pd.DataFrame,
        *,
        include_source: bool,
    ) -> None:
        table.setRowCount(0)
        for _, row in frame.iterrows():
            idx = table.rowCount()
            table.insertRow(idx)
            coverage = "—"
            if row.get("coverage_start") and row.get("coverage_end"):
                coverage = f"{str(row['coverage_start'])[:10]} → {str(row['coverage_end'])[:10]}"
            values = [
                self._fmt_ts(row.get("finished_at")),
                str(row.get("acquisition_run_id") or ""),
                str(row.get("symbol") or "—"),
                str(row.get("dataset_id") or "—"),
            ]
            if include_source:
                values.append(str(row.get("source") or "—"))
            values.extend(
                [
                    str(row.get("status") or ""),
                    str(int(row.get("bar_count") or 0)) if pd.notna(row.get("bar_count")) else "—",
                    coverage,
                    "Yes" if bool(row.get("ingested")) else "No",
                    str(row.get("error_message") or "—"),
                ]
            )
            for col, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                if col == len(values) - 1:
                    item.setToolTip(str(row.get("error_message") or ""))
                table.setItem(idx, col, item)

    def _on_dataset_selected(self) -> None:
        current = self.dataset_table.currentRow()
        if current < 0 or self.dataset_frame.empty:
            self.dataset_detail.setPlainText("")
            self.dataset_attempt_table.setRowCount(0)
            return
        dataset_item = self.dataset_table.item(current, 0)
        dataset_id = str(dataset_item.data(QtCore.Qt.ItemDataRole.UserRole) or "") if dataset_item else ""
        match = self.dataset_frame[self.dataset_frame["dataset_id"] == dataset_id]
        if match.empty:
            return
        row = match.iloc[0]
        quality = _safe_inspect_dataset_quality(dataset_id, str(row.get("resolution") or ""))
        related_universes = [
            str(item.get("name") or "")
            for item in self.universes
            if dataset_id in list(item.get("dataset_ids") or [])
            or (row.get("symbol") and str(row.get("symbol")) in list(item.get("symbols") or []))
        ]
        next_action = None
        if row.get("symbol") and row.get("source") and row.get("resolution") and row.get("history_window"):
            try:
                next_action = decide_acquisition_policy(
                    str(row.get("symbol")),
                    source=str(row.get("source")),
                    resolution=str(row.get("resolution")),
                    history_window=str(row.get("history_window")),
                    catalog=self.catalog,
                )
            except Exception:
                next_action = None
        details = [
            f"Dataset: {dataset_id}",
            f"Source: {row.get('source') or '—'}",
            f"Symbol: {row.get('symbol') or '—'}",
            f"Resolution: {row.get('resolution') or '—'}",
            f"History Window: {row.get('history_window') or '—'}",
            f"Coverage: {(row.get('coverage_start') or '—')} -> {(row.get('coverage_end') or '—')}",
            f"Freshness: {compute_freshness_state(row.get('coverage_end'), str(row.get('resolution') or ''))}",
            f"Quality: {_format_dataset_quality_label(quality)}",
            f"Suggested Gap Repair Window: {(quality.get('repair_request_start') or '—')} -> {(quality.get('repair_request_end') or '—')}",
            f"Recommended Action: {next_action.action if next_action else '—'}",
            f"Acquisition Plan: {next_action.plan_type if next_action else '—'}",
            f"Planned Request Window: {(next_action.request_start or 'provider default') if next_action else '—'} -> {(next_action.request_end or 'provider default') if next_action else '—'}",
            f"Request Windows: {list(next_action.request_windows) if next_action and next_action.request_windows else '—'}",
            f"Secondary Repair Windows: {list(next_action.secondary_request_windows) if next_action and next_action.secondary_request_windows else '—'}",
            f"Merge With Existing: {'Yes' if next_action and next_action.merge_with_existing else 'No'}",
            f"Secondary Gap-Fill Source: {f'{next_action.secondary_source} / {next_action.secondary_dataset_id}' if next_action and next_action.secondary_dataset_id else '—'}",
            f"Secondary Parity: {f'{next_action.parity_state} | overlap {next_action.parity_overlap_bars} | mean abs {next_action.parity_close_mean_abs_bps:.2f} bps' if next_action and next_action.secondary_dataset_id and next_action.parity_close_mean_abs_bps is not None else '—'}",
            f"Policy Reason: {next_action.reason if next_action else '—'}",
            f"Bars: {int(row.get('bar_count') or 0) if pd.notna(row.get('bar_count')) else '—'}",
            f"Last Download Attempt: {self._fmt_ts(row.get('last_download_attempt_at'))}",
            f"Last Download Success: {self._fmt_ts(row.get('last_download_success_at'))}",
            f"Last Ingest: {self._fmt_ts(row.get('last_ingest_at'))}",
            f"Status: {row.get('last_status') or '—'}",
            f"Ingested: {'Yes' if bool(row.get('ingested')) else 'No'}",
            f"CSV Path: {row.get('csv_path') or '—'}",
            f"Parquet Path: {row.get('parquet_path') or '—'}",
            f"Last Error: {row.get('last_error') or '—'}",
            f"Linked Universes: {', '.join(related_universes) if related_universes else '—'}",
        ]
        self.dataset_detail.setPlainText("\n".join(details))
        attempts = self.catalog.load_acquisition_attempts(dataset_id=dataset_id, limit=100)
        self._populate_attempt_table(self.dataset_attempt_table, attempts, include_source=False)

    def _on_run_selected(self) -> None:
        current = self.run_table.currentRow()
        if current < 0 or self.run_frame.empty:
            self.run_notes.setPlainText("")
            self.run_attempt_table.setRowCount(0)
            return
        run_item = self.run_table.item(current, 0)
        run_id = str(run_item.data(QtCore.Qt.ItemDataRole.UserRole) or "") if run_item else ""
        match = self.run_frame[self.run_frame["acquisition_run_id"] == run_id]
        if match.empty:
            return
        row = match.iloc[0]
        details = [
            f"Run ID: {run_id}",
            f"Trigger: {row.get('trigger_type') or '—'}",
            f"Source: {row.get('source') or '—'}",
            f"Universe: {row.get('universe_name') or row.get('universe_id') or '—'}",
            f"Task ID: {row.get('task_id') or '—'}",
            f"Started: {self._fmt_ts(row.get('started_at'))}",
            f"Finished: {self._fmt_ts(row.get('finished_at'))}",
            f"Status: {row.get('status') or '—'}",
            f"Symbols: {int(row.get('symbol_count') or 0)}",
            f"Success: {int(row.get('success_count') or 0)} | Failed: {int(row.get('failed_count') or 0)} | Ingested: {int(row.get('ingested_count') or 0)}",
            f"Log Path: {row.get('log_path') or '—'}",
            f"Notes: {row.get('notes') or '—'}",
        ]
        self.run_notes.setPlainText("\n".join(details))
        attempts = self.catalog.load_acquisition_attempts(acquisition_run_id=run_id, limit=500)
        self.run_attempt_table.setRowCount(0)
        for _, attempt in attempts.iterrows():
            idx = self.run_attempt_table.rowCount()
            self.run_attempt_table.insertRow(idx)
            coverage = "—"
            if attempt.get("coverage_start") and attempt.get("coverage_end"):
                coverage = f"{str(attempt['coverage_start'])[:10]} → {str(attempt['coverage_end'])[:10]}"
            values = [
                str(int(attempt.get("seq") or 0)),
                str(attempt.get("symbol") or "—"),
                str(attempt.get("dataset_id") or "—"),
                str(attempt.get("status") or ""),
                str(int(attempt.get("bar_count") or 0)) if pd.notna(attempt.get("bar_count")) else "—",
                coverage,
                "Yes" if bool(attempt.get("ingested")) else "No",
                str(attempt.get("error_message") or "—"),
            ]
            for col, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                if col == 7:
                    item.setToolTip(str(attempt.get("error_message") or ""))
                self.run_attempt_table.setItem(idx, col, item)

    def _open_selected_dataset_detail(self) -> None:
        current = self.dataset_table.currentRow()
        if current < 0 or self.dataset_frame.empty:
            return
        dataset_item = self.dataset_table.item(current, 0)
        dataset_id = str(dataset_item.data(QtCore.Qt.ItemDataRole.UserRole) or "") if dataset_item else ""
        if not dataset_id:
            return
        dlg = AcquisitionDatasetDetailDialog(
            self.catalog,
            self.universes,
            [dataset_id],
            self,
            initial_dataset_id=dataset_id,
        )
        dlg.exec()

    def _open_selected_run_log(self) -> None:
        current = self.run_table.currentRow()
        if current < 0 or self.run_frame.empty:
            return
        run_item = self.run_table.item(current, 0)
        run_id = str(run_item.data(QtCore.Qt.ItemDataRole.UserRole) or "") if run_item else ""
        match = self.run_frame[self.run_frame["acquisition_run_id"] == run_id]
        if match.empty:
            return
        log_path = Path(str(match.iloc[0].get("log_path") or ""))
        if not log_path.exists():
            QtWidgets.QMessageBox.information(self, "Log", "No log file is available for the selected run.")
            return
        try:
            content = log_path.read_text(encoding="utf-8")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Log", f"Unable to read log: {exc}")
            return
        dlg = DashboardDialog(self)
        dlg.setWindowTitle(f"Acquisition Log {run_id}")
        dlg.resize(960, 640)
        dlg.setObjectName("Panel")
        layout = QtWidgets.QVBoxLayout(dlg)
        path_label = QtWidgets.QLabel(str(log_path))
        path_label.setObjectName("Sub")
        layout.addWidget(path_label)
        text = QtWidgets.QPlainTextEdit()
        text.setObjectName("Panel")
        text.setReadOnly(True)
        text.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        text.setPlainText(content)
        layout.addWidget(text)
        dlg.exec()

    def _on_task_run_selected(self) -> None:
        current = self.task_run_table.currentRow()
        if current < 0 or self.task_run_frame.empty:
            self.task_run_notes.setPlainText("")
            return
        run_item = self.task_run_table.item(current, 0)
        run_id = str(run_item.data(QtCore.Qt.ItemDataRole.UserRole) or "") if run_item else ""
        match = self.task_run_frame[self.task_run_frame["run_id"] == run_id]
        if match.empty:
            self.task_run_notes.setPlainText("")
            return
        row = match.iloc[0]
        details = [
            f"Run ID: {run_id}",
            f"Task ID: {row.get('task_id') or '—'}",
            f"Started: {self._fmt_ts(row.get('started_at'))}",
            f"Finished: {self._fmt_ts(row.get('finished_at'))}",
            f"Status: {row.get('status') or '—'}",
            f"Ticker Count: {int(row.get('ticker_count') or 0) if pd.notna(row.get('ticker_count')) else '—'}",
            f"Log Path: {row.get('log_path') or '—'}",
            f"Error: {row.get('error_message') or '—'}",
        ]
        self.task_run_notes.setPlainText("\n".join(details))

    def _open_selected_task_run_log(self) -> None:
        current = self.task_run_table.currentRow()
        if current < 0 or self.task_run_frame.empty:
            return
        run_item = self.task_run_table.item(current, 0)
        run_id = str(run_item.data(QtCore.Qt.ItemDataRole.UserRole) or "") if run_item else ""
        match = self.task_run_frame[self.task_run_frame["run_id"] == run_id]
        if match.empty:
            return
        log_path = Path(str(match.iloc[0].get("log_path") or ""))
        if not log_path.exists():
            QtWidgets.QMessageBox.information(self, "Log", "No scheduler log file is available for the selected run.")
            return
        try:
            content = log_path.read_text(encoding="utf-8")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Log", f"Unable to read log: {exc}")
            return
        dlg = DashboardDialog(self)
        dlg.setWindowTitle(f"Scheduler Log {run_id}")
        dlg.resize(960, 640)
        dlg.setObjectName("Panel")
        layout = QtWidgets.QVBoxLayout(dlg)
        path_label = QtWidgets.QLabel(str(log_path))
        path_label.setObjectName("Sub")
        layout.addWidget(path_label)
        text = QtWidgets.QPlainTextEdit()
        text.setObjectName("Panel")
        text.setReadOnly(True)
        text.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        text.setPlainText(content)
        layout.addWidget(text)
        dlg.exec()

    @staticmethod
    def _fmt_ts(value: object) -> str:
        if not value:
            return "—"
        try:
            ts = pd.to_datetime(value, utc=True)
            return ts.tz_convert("America/New_York").strftime("%Y-%m-%d %I:%M %p ET")
        except Exception:
            return str(value)

    @staticmethod
    def _clip_text(text: str, max_chars: int) -> str:
        text = str(text or "")
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 1] + "…"


class AcquisitionDatasetDetailDialog(DashboardDialog):
    def __init__(
        self,
        catalog: CatalogReader,
        universes: Sequence[Dict],
        dataset_ids: Sequence[str],
        parent=None,
        *,
        initial_dataset_id: str = "",
    ) -> None:
        super().__init__(parent)
        self.catalog = catalog
        self.universes = list(universes or [])
        self.dataset_ids = [str(item).strip() for item in dataset_ids if str(item).strip()]
        self.initial_dataset_id = str(initial_dataset_id or "").strip()
        self.dataset_frame = pd.DataFrame(self.catalog.load_acquisition_datasets())
        self.attempt_frame = pd.DataFrame(self.catalog.load_acquisition_attempts(limit=500))

        self.setWindowTitle("Dataset Acquisition Detail")
        self.resize(1180, 860)
        self.setMinimumSize(980, 720)
        self.setObjectName("Panel")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        title = QtWidgets.QLabel("Dataset Acquisition Detail")
        title.setObjectName("Title")
        subtitle = QtWidgets.QLabel(
            "Review freshness, coverage, last success/failure state, linked universes, and recent acquisition attempts for a dataset."
        )
        subtitle.setObjectName("Sub")
        subtitle.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(subtitle)

        selector_row = QtWidgets.QHBoxLayout()
        selector_row.addWidget(QtWidgets.QLabel("Dataset"))
        self.dataset_combo = QtWidgets.QComboBox()
        for dataset_id in self.dataset_ids:
            self.dataset_combo.addItem(dataset_id, dataset_id)
        if self.initial_dataset_id:
            idx = self.dataset_combo.findData(self.initial_dataset_id)
            if idx >= 0:
                self.dataset_combo.setCurrentIndex(idx)
        self.dataset_combo.currentIndexChanged.connect(self._refresh_view)
        selector_row.addWidget(self.dataset_combo, 1)
        refresh_btn = QtWidgets.QPushButton("Refresh")
        refresh_btn.clicked.connect(self._reload)
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        selector_row.addWidget(refresh_btn)
        selector_row.addWidget(close_btn)
        layout.addLayout(selector_row)

        self.summary_text = QtWidgets.QPlainTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMinimumHeight(220)
        layout.addWidget(self.summary_text)

        self.attempt_table = QtWidgets.QTableWidget(0, 10)
        self.attempt_table.setHorizontalHeaderLabels(
            ["Finished", "Run", "Symbol", "Source", "Status", "Bars", "Coverage", "Ingested", "Task", "Error"]
        )
        self.attempt_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.attempt_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.attempt_table.setAlternatingRowColors(True)
        self.attempt_table.setTextElideMode(QtCore.Qt.TextElideMode.ElideNone)
        self.attempt_table.setWordWrap(False)
        self.attempt_table.verticalHeader().setVisible(False)
        self.attempt_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)
        self.attempt_table.horizontalHeader().setStretchLastSection(True)
        self.attempt_table.setObjectName("Panel")
        self.attempt_table.setColumnWidth(9, 420)
        layout.addWidget(self.attempt_table, 1)

        self._refresh_view()

    def _reload(self) -> None:
        self.dataset_frame = pd.DataFrame(self.catalog.load_acquisition_datasets())
        self.attempt_frame = pd.DataFrame(self.catalog.load_acquisition_attempts(limit=500))
        self._refresh_view()

    def _refresh_view(self) -> None:
        dataset_id = str(self.dataset_combo.currentData() or "").strip()
        if not dataset_id:
            self.summary_text.setPlainText("No dataset selected.")
            self.attempt_table.setRowCount(0)
            return
        match = self.dataset_frame[self.dataset_frame["dataset_id"].astype(str) == dataset_id] if not self.dataset_frame.empty else pd.DataFrame()
        if match.empty:
            self.summary_text.setPlainText(
                f"Dataset '{dataset_id}' does not currently have acquisition metadata in the catalog."
            )
            self.attempt_table.setRowCount(0)
            return
        row = match.iloc[0]
        symbol = str(row.get("symbol") or "").strip()
        linked_universes = [
            str(item.get("name") or "")
            for item in self.universes
            if dataset_id in list(item.get("dataset_ids") or [])
            or (symbol and symbol in [str(entry).upper() for entry in list(item.get("symbols") or []) if str(entry).strip()])
        ]
        freshness = compute_freshness_state(row.get("coverage_end"), str(row.get("resolution") or ""))
        quality = _safe_inspect_dataset_quality(dataset_id, str(row.get("resolution") or ""))
        next_action = None
        if row.get("symbol") and row.get("source") and row.get("resolution") and row.get("history_window"):
            try:
                next_action = decide_acquisition_policy(
                    str(row.get("symbol")),
                    source=str(row.get("source")),
                    resolution=str(row.get("resolution")),
                    history_window=str(row.get("history_window")),
                    catalog=self.catalog,
                )
            except Exception:
                next_action = None

        def _fmt_ts(value: object) -> str:
            if not value:
                return "—"
            try:
                ts = pd.to_datetime(value, utc=True)
                return ts.tz_convert("America/New_York").strftime("%Y-%m-%d %I:%M %p ET")
            except Exception:
                return str(value)

        details = [
            f"Dataset: {dataset_id}",
            f"Source: {row.get('source') or '—'}",
            f"Symbol: {symbol or '—'}",
            f"Resolution: {row.get('resolution') or '—'} | History Window: {row.get('history_window') or '—'}",
            f"Freshness: {freshness}",
            f"Quality: {_format_dataset_quality_label(quality)}",
            f"Suggested Gap Repair Window: {(quality.get('repair_request_start') or '—')} -> {(quality.get('repair_request_end') or '—')}",
            f"Recommended Action: {next_action.action if next_action else '—'}",
            f"Acquisition Plan: {next_action.plan_type if next_action else '—'}",
            f"Planned Request Window: {(next_action.request_start or 'provider default') if next_action else '—'} -> {(next_action.request_end or 'provider default') if next_action else '—'}",
            f"Request Windows: {list(next_action.request_windows) if next_action and next_action.request_windows else '—'}",
            f"Secondary Repair Windows: {list(next_action.secondary_request_windows) if next_action and next_action.secondary_request_windows else '—'}",
            f"Merge With Existing: {'Yes' if next_action and next_action.merge_with_existing else 'No'}",
            f"Secondary Gap-Fill Source: {f'{next_action.secondary_source} / {next_action.secondary_dataset_id}' if next_action and next_action.secondary_dataset_id else '—'}",
            f"Secondary Parity: {f'{next_action.parity_state} | overlap {next_action.parity_overlap_bars} | mean abs {next_action.parity_close_mean_abs_bps:.2f} bps' if next_action and next_action.secondary_dataset_id and next_action.parity_close_mean_abs_bps is not None else '—'}",
            f"Policy Reason: {next_action.reason if next_action else '—'}",
            f"Coverage: {(row.get('coverage_start') or '—')} -> {(row.get('coverage_end') or '—')}",
            f"Bars: {int(row.get('bar_count') or 0) if pd.notna(row.get('bar_count')) else '—'} | Ingested: {'Yes' if bool(row.get('ingested')) else 'No'}",
            f"Last Download Attempt: {_fmt_ts(row.get('last_download_attempt_at'))}",
            f"Last Download Success: {_fmt_ts(row.get('last_download_success_at'))}",
            f"Last Ingest: {_fmt_ts(row.get('last_ingest_at'))}",
            f"Last Status: {row.get('last_status') or '—'}",
            f"Last Error: {row.get('last_error') or '—'}",
            f"CSV Path: {row.get('csv_path') or '—'}",
            f"Parquet Path: {row.get('parquet_path') or '—'}",
            f"Linked Universes: {', '.join(linked_universes) if linked_universes else '—'}",
        ]
        self.summary_text.setPlainText("\n".join(details))

        attempt_frame = (
            self.attempt_frame[self.attempt_frame["dataset_id"].astype(str) == dataset_id].copy()
            if not self.attempt_frame.empty
            else pd.DataFrame()
        )
        self.attempt_table.setRowCount(0)
        for _, attempt in attempt_frame.iterrows():
            idx = self.attempt_table.rowCount()
            self.attempt_table.insertRow(idx)
            coverage = "—"
            if attempt.get("coverage_start") and attempt.get("coverage_end"):
                coverage = f"{str(attempt['coverage_start'])[:10]} → {str(attempt['coverage_end'])[:10]}"
            values = [
                _fmt_ts(attempt.get("finished_at")),
                str(attempt.get("acquisition_run_id") or "—"),
                str(attempt.get("symbol") or "—"),
                str(attempt.get("source") or "—"),
                str(attempt.get("status") or "—"),
                str(int(attempt.get("bar_count") or 0)) if pd.notna(attempt.get("bar_count")) else "—",
                coverage,
                "Yes" if bool(attempt.get("ingested")) else "No",
                str(attempt.get("task_id") or "—"),
                str(attempt.get("error_message") or "—"),
            ]
            for col_idx, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                if col_idx == 9:
                    item.setToolTip(str(attempt.get("error_message") or ""))
                self.attempt_table.setItem(idx, col_idx, item)


class StrategyBlockEditorDialog(DashboardDialog):
    def __init__(
        self,
        datasets: Sequence[str],
        strategy_specs: dict,
        blocks: Sequence[dict],
        universes: Sequence[dict] | None = None,
        default_universe_id: str = "",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit Portfolio Strategy Blocks")
        self.resize(1240, 780)
        self.setMinimumSize(1120, 720)
        self.setObjectName("Panel")
        self.setStyleSheet(
            f"""
            QDialog#Panel {{
                background: {PALETTE['panel']};
                color: {PALETTE['text']};
            }}
            QWidget#Panel {{
                background: {PALETTE['panel2']};
                border: 1px solid rgba(231, 238, 252, 0.35);
                border-radius: 12px;
            }}
            QLabel {{
                color: {PALETTE['text']};
            }}
            QLabel#Sub {{
                color: {PALETTE['muted']};
            }}
            QGroupBox {{
                color: {PALETTE['text']};
                border: 1px solid rgba(231, 238, 252, 0.38);
                border-radius: 10px;
                margin-top: 12px;
                padding-top: 10px;
                background: transparent;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: {PALETTE['muted']};
                font-weight: 700;
            }}
            QListWidget {{
                background: rgba(0, 0, 0, 0.16);
                color: {PALETTE['text']};
                border: 1px solid rgba(231, 238, 252, 0.35);
                border-radius: 10px;
                padding: 6px;
                outline: 0;
            }}
            QListWidget::item {{
                padding: 8px 10px;
                border-radius: 8px;
            }}
            QListWidget::item:selected {{
                background: rgba(77, 163, 255, 0.24);
                color: {PALETTE['text']};
            }}
            QLineEdit, QComboBox, QAbstractSpinBox {{
                background: rgba(0, 0, 0, 0.18);
                color: {PALETTE['text']};
                border: 1px solid rgba(231, 238, 252, 0.45);
                border-radius: 8px;
                padding: 6px 8px;
                min-height: 22px;
            }}
            QLineEdit[readOnly=\"true\"] {{
                background: rgba(255, 255, 255, 0.08);
                color: {PALETTE['text']};
            }}
            QComboBox QAbstractItemView {{
                background: {PALETTE['panel']};
                color: {PALETTE['text']};
                selection-background-color: rgba(77, 163, 255, 0.24);
                border: 1px solid {PALETTE['border']};
            }}
            QPushButton {{
                background: rgba(255, 255, 255, 0.08);
                color: {PALETTE['text']};
                border: 1px solid rgba(231, 238, 252, 0.45);
                border-radius: 8px;
                padding: 7px 12px;
            }}
            QPushButton:hover {{
                border-color: {PALETTE['blue']};
            }}
            QScrollArea {{
                background: transparent;
                border: none;
            }}
            QScrollArea > QWidget > QWidget {{
                background: transparent;
            }}
            QSplitter::handle {{
                background: rgba(231, 238, 252, 0.08);
                width: 8px;
            }}
            """
        )
        self.datasets = list(datasets)
        self.strategy_specs = strategy_specs
        self.universes = [dict(item) for item in list(universes or ())]
        self.default_universe_id = str(default_universe_id or "")
        self.blocks: list[dict] = [self._normalize_block(block) for block in blocks]
        self._updating = False
        self._current_index = -1

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)
        intro = QtWidgets.QLabel(
            "Define one or more strategy blocks. Each block has its own strategy, fixed parameter set, attached assets, and optional budget cap."
        )
        intro.setObjectName("Sub")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        layout.addWidget(splitter, 1)

        left_panel = QtWidgets.QWidget()
        left_panel.setObjectName("Panel")
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(12, 12, 12, 12)
        left_layout.setSpacing(10)
        left_title = QtWidgets.QLabel("Strategy Blocks")
        left_title.setObjectName("Title")
        left_layout.addWidget(left_title)
        self.block_list = QtWidgets.QListWidget()
        self.block_list.setAlternatingRowColors(True)
        self.block_list.setMinimumWidth(300)
        self.block_list.currentRowChanged.connect(self._on_block_selection_changed)
        left_layout.addWidget(self.block_list, 1)
        block_actions = QtWidgets.QHBoxLayout()
        add_btn = QtWidgets.QPushButton("Add Block")
        add_btn.clicked.connect(self._add_block)
        remove_btn = QtWidgets.QPushButton("Remove Block")
        remove_btn.clicked.connect(self._remove_current_block)
        block_actions.addWidget(add_btn)
        block_actions.addWidget(remove_btn)
        left_layout.addLayout(block_actions)
        splitter.addWidget(left_panel)

        right_panel = QtWidgets.QWidget()
        right_panel.setObjectName("Panel")
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(12, 12, 12, 12)
        right_layout.setSpacing(10)
        form_scroll = QtWidgets.QScrollArea()
        form_scroll.setWidgetResizable(True)
        form_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        form_inner = QtWidgets.QWidget()
        form_layout = QtWidgets.QVBoxLayout(form_inner)
        form_layout.setContentsMargins(0, 0, 0, 0)
        form_layout.setSpacing(10)

        details_box = QtWidgets.QGroupBox("Block Details")
        details_form = QtWidgets.QFormLayout(details_box)
        details_form.setContentsMargins(12, 12, 12, 12)
        details_form.setSpacing(8)
        self.block_name_edit = QtWidgets.QLineEdit()
        self.block_name_edit.textChanged.connect(self._save_current_block_state)
        details_form.addRow("Display Name", self.block_name_edit)
        self.block_strategy_combo = QtWidgets.QComboBox()
        for name in self.strategy_specs:
            self.block_strategy_combo.addItem(name)
        self.block_strategy_combo.currentTextChanged.connect(self._on_block_strategy_changed)
        details_form.addRow("Strategy", self.block_strategy_combo)
        self.block_budget_spin = QtWidgets.QDoubleSpinBox()
        self.block_budget_spin.setDecimals(4)
        self.block_budget_spin.setRange(0.0, 1.0)
        self.block_budget_spin.setSingleStep(0.05)
        self.block_budget_spin.valueChanged.connect(self._save_current_block_state)
        self.block_budget_spin.setToolTip("Optional strategy budget cap. Set 0 to leave uncapped at the block level.")
        details_form.addRow("Budget Weight", self.block_budget_spin)
        self.block_universe_combo = QtWidgets.QComboBox()
        self.block_universe_combo.addItem("No Universe", "")
        for universe in self.universes:
            label = str(universe.get("name") or universe.get("universe_id") or "Universe")
            dataset_count = len([str(item) for item in list(universe.get("dataset_ids") or []) if str(item).strip()])
            self.block_universe_combo.addItem(f"{label} ({dataset_count} datasets)", str(universe.get("universe_id") or ""))
        if self.default_universe_id:
            idx = self.block_universe_combo.findData(self.default_universe_id)
            self.block_universe_combo.setCurrentIndex(idx if idx >= 0 else 0)
        apply_universe_btn = QtWidgets.QPushButton("Apply To Block")
        apply_universe_btn.clicked.connect(self._apply_selected_universe_to_current_block)
        apply_all_universe_btn = QtWidgets.QPushButton("Apply To All Blocks")
        apply_all_universe_btn.clicked.connect(self._apply_selected_universe_to_all_blocks)
        append_dataset_blocks_btn = QtWidgets.QPushButton("Append Block Per Dataset")
        append_dataset_blocks_btn.clicked.connect(self._append_blocks_from_selected_universe)
        expand_strategy_matrix_btn = QtWidgets.QPushButton("Expand All Blocks Across Universe")
        expand_strategy_matrix_btn.clicked.connect(self._expand_existing_blocks_across_selected_universe)
        universe_row = QtWidgets.QHBoxLayout()
        universe_row.addWidget(self.block_universe_combo, 2)
        universe_row.addWidget(apply_universe_btn, 1)
        universe_row.addWidget(apply_all_universe_btn, 1)
        universe_row.addWidget(append_dataset_blocks_btn, 1)
        universe_row.addWidget(expand_strategy_matrix_btn, 1)
        universe_wrap = QtWidgets.QWidget()
        universe_wrap.setLayout(universe_row)
        details_form.addRow("Seed From Universe", universe_wrap)
        self.block_assets_summary = QtWidgets.QLineEdit()
        self.block_assets_summary.setReadOnly(True)
        choose_assets_btn = QtWidgets.QPushButton("Choose Assets")
        choose_assets_btn.clicked.connect(self._choose_block_assets)
        assets_row = QtWidgets.QHBoxLayout()
        assets_row.addWidget(self.block_assets_summary, 1)
        assets_row.addWidget(choose_assets_btn, 0)
        assets_wrap = QtWidgets.QWidget()
        assets_wrap.setLayout(assets_row)
        details_form.addRow("Assets", assets_wrap)
        self.block_asset_weights_edit = QtWidgets.QLineEdit()
        self.block_asset_weights_edit.setPlaceholderText("Optional. SPY=0.6,QQQ=0.4 or 0.6,0.4 in block asset order")
        self.block_asset_weights_edit.textChanged.connect(self._save_current_block_state)
        details_form.addRow("Asset Weights", self.block_asset_weights_edit)
        self.block_note = QtWidgets.QLabel(
            "These block parameters are fixed values for portfolio construction. Grid sweeps across multiple strategy blocks are not part of this first UI pass."
        )
        self.block_note.setObjectName("Sub")
        self.block_note.setWordWrap(True)
        details_form.addRow("", self.block_note)
        form_layout.addWidget(details_box)

        self.param_group = QtWidgets.QGroupBox("Strategy Parameters")
        self.param_form = QtWidgets.QFormLayout(self.param_group)
        self.param_form.setContentsMargins(12, 12, 12, 12)
        self.param_form.setSpacing(8)
        form_layout.addWidget(self.param_group)
        form_layout.addStretch(1)
        form_scroll.setWidget(form_inner)
        right_layout.addWidget(form_scroll, 1)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([340, 820])

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.block_param_inputs: dict[str, QtWidgets.QLineEdit] = {}
        self._refresh_block_list()
        if not self.blocks:
            self._add_block()
        elif self.block_list.count() > 0:
            self.block_list.setCurrentRow(0)

    @staticmethod
    def _slugify(value: str, fallback: str) -> str:
        cleaned = re.sub(r"[^0-9A-Za-z_]+", "_", value.strip())
        cleaned = re.sub(r"_+", "_", cleaned).strip("_")
        return cleaned or fallback

    def _normalize_block(self, block: dict) -> dict:
        strategy_name = str(block.get("strategy_name") or next(iter(self.strategy_specs.keys()), "SMACrossStrategy"))
        params = dict(block.get("strategy_params") or {})
        datasets = [str(dataset_id) for dataset_id in list(block.get("asset_dataset_ids") or []) if str(dataset_id).strip()]
        return {
            "block_id": str(block.get("block_id") or ""),
            "display_name": str(block.get("display_name") or strategy_name),
            "strategy_name": strategy_name,
            "strategy_params": params,
            "asset_dataset_ids": datasets,
            "asset_universe_id": str(block.get("asset_universe_id") or ""),
            "asset_universe_name": str(block.get("asset_universe_name") or ""),
            "asset_weights_text": str(block.get("asset_weights_text") or ""),
            "budget_weight": float(block.get("budget_weight") or 0.0),
        }

    def _default_block(self) -> dict:
        strategy_name = next(iter(self.strategy_specs.keys()), "SMACrossStrategy")
        spec = self.strategy_specs.get(strategy_name, {})
        params = {
            key: default
            for key, (_ptype, default) in dict(spec.get("params") or {}).items()
        }
        block_number = len(self.blocks) + 1
        display_name = f"Strategy Block {block_number}"
        default_universe = self._find_universe(self.default_universe_id)
        default_dataset_ids = self._datasets_from_universe(default_universe) if default_universe else []
        return {
            "block_id": self._slugify(display_name, f"block_{block_number}"),
            "display_name": display_name,
            "strategy_name": strategy_name,
            "strategy_params": params,
            "asset_dataset_ids": default_dataset_ids,
            "asset_universe_id": str((default_universe or {}).get("universe_id") or ""),
            "asset_universe_name": str((default_universe or {}).get("name") or ""),
            "asset_weights_text": "",
            "budget_weight": 0.0,
        }

    def _find_universe(self, universe_id: str) -> dict | None:
        universe_id = str(universe_id or "").strip()
        if not universe_id:
            return None
        for universe in self.universes:
            if str(universe.get("universe_id") or "") == universe_id:
                return universe
        return None

    def _datasets_from_universe(self, universe: dict | None) -> list[str]:
        if universe is None:
            return []
        allowed = set(self.datasets)
        return [
            dataset_id
            for dataset_id in [str(item) for item in list(universe.get("dataset_ids") or []) if str(item).strip()]
            if dataset_id in allowed
        ]

    def _refresh_block_list(self) -> None:
        self.block_list.blockSignals(True)
        self.block_list.clear()
        for block in self.blocks:
            label = str(block.get("display_name") or block.get("block_id") or "Strategy Block")
            strategy_name = str(block.get("strategy_name") or "")
            assets = list(block.get("asset_dataset_ids") or [])
            suffix = f" ({strategy_name}, {len(assets)} asset{'s' if len(assets) != 1 else ''})"
            self.block_list.addItem(label + suffix)
        self.block_list.blockSignals(False)

    def _update_asset_summary(self, dataset_ids: Sequence[str]) -> None:
        if not dataset_ids:
            self.block_assets_summary.setText("No assets selected")
            self.block_assets_summary.setToolTip("")
            return
        if len(dataset_ids) <= 3:
            self.block_assets_summary.setText(", ".join(dataset_ids))
            self.block_assets_summary.setToolTip("\n".join(dataset_ids))
            return
        self.block_assets_summary.setText(f"{len(dataset_ids)} assets selected")
        self.block_assets_summary.setToolTip("\n".join(dataset_ids))

    def _render_block_params(self, strategy_name: str, values: dict | None = None) -> None:
        while self.param_form.rowCount():
            self.param_form.removeRow(0)
        self.block_param_inputs.clear()
        spec = self.strategy_specs.get(strategy_name)
        if not spec:
            return
        params = dict(spec.get("params") or {})
        values = values or {}
        for param_name, (_ptype, default) in params.items():
            edit = QtWidgets.QLineEdit(str(values.get(param_name, default)))
            edit.textChanged.connect(self._save_current_block_state)
            self.param_form.addRow(QtWidgets.QLabel(param_name), edit)
            self.block_param_inputs[param_name] = edit

    def _save_current_block_state(self) -> None:
        if self._updating:
            return
        row = self._current_index
        if row < 0 or row >= len(self.blocks):
            return
        block = dict(self.blocks[row])
        block["display_name"] = self.block_name_edit.text().strip() or f"Strategy Block {row + 1}"
        block["strategy_name"] = self.block_strategy_combo.currentText().strip()
        block["budget_weight"] = float(self.block_budget_spin.value())
        block["asset_weights_text"] = self.block_asset_weights_edit.text().strip()
        block["strategy_params"] = {
            key: edit.text().strip()
            for key, edit in self.block_param_inputs.items()
        }
        block["block_id"] = self._slugify(
            str(block.get("block_id") or block["display_name"]),
            f"block_{row + 1}",
        )
        self.blocks[row] = block
        self._refresh_block_list()
        self.block_list.setCurrentRow(row)

    def _on_block_selection_changed(self, row: int) -> None:
        self._current_index = row
        self._updating = True
        try:
            enabled = 0 <= row < len(self.blocks)
            self.block_name_edit.setEnabled(enabled)
            self.block_strategy_combo.setEnabled(enabled)
            self.block_budget_spin.setEnabled(enabled)
            self.block_assets_summary.setEnabled(enabled)
            self.block_asset_weights_edit.setEnabled(enabled)
            if not enabled:
                self.block_name_edit.clear()
                self.block_strategy_combo.setCurrentIndex(0 if self.block_strategy_combo.count() else -1)
                self.block_budget_spin.setValue(0.0)
                self.block_assets_summary.clear()
                self.block_asset_weights_edit.clear()
                self._render_block_params("", {})
                return
            block = self.blocks[row]
            self.block_name_edit.setText(str(block.get("display_name") or ""))
            idx = self.block_strategy_combo.findText(str(block.get("strategy_name") or ""))
            self.block_strategy_combo.setCurrentIndex(idx if idx >= 0 else 0)
            self.block_budget_spin.setValue(float(block.get("budget_weight") or 0.0))
            self._update_asset_summary(list(block.get("asset_dataset_ids") or []))
            self.block_asset_weights_edit.setText(str(block.get("asset_weights_text") or ""))
            self._render_block_params(str(block.get("strategy_name") or ""), dict(block.get("strategy_params") or {}))
        finally:
            self._updating = False

    def _on_block_strategy_changed(self, strategy_name: str) -> None:
        if self._updating:
            return
        row = self._current_index
        if row < 0 or row >= len(self.blocks):
            return
        spec = self.strategy_specs.get(strategy_name, {})
        defaults = {
            key: default
            for key, (_ptype, default) in dict(spec.get("params") or {}).items()
        }
        self._updating = True
        try:
            self._render_block_params(strategy_name, defaults)
        finally:
            self._updating = False
        self.blocks[row]["strategy_name"] = strategy_name
        self.blocks[row]["strategy_params"] = defaults
        self._save_current_block_state()

    def _add_block(self) -> None:
        block = self._default_block()
        self.blocks.append(block)
        self._refresh_block_list()
        self.block_list.setCurrentRow(len(self.blocks) - 1)

    def _remove_current_block(self) -> None:
        row = self.block_list.currentRow()
        if row < 0 or row >= len(self.blocks):
            return
        self.blocks.pop(row)
        self._refresh_block_list()
        if self.blocks:
            self.block_list.setCurrentRow(min(row, len(self.blocks) - 1))
        else:
            self._current_index = -1
            self._on_block_selection_changed(-1)

    def _choose_block_assets(self) -> None:
        row = self._current_index
        if row < 0 or row >= len(self.blocks):
            return
        initial = list(self.blocks[row].get("asset_dataset_ids") or [])
        dlg = DatasetSelectionDialog(self.datasets, initial, self)
        dlg.setWindowTitle("Choose Strategy Block Assets")
        if dlg.exec() != int(QtWidgets.QDialog.DialogCode.Accepted):
            return
        selected = dlg.selected_datasets()
        self.blocks[row]["asset_dataset_ids"] = selected
        self.blocks[row]["asset_universe_id"] = ""
        self.blocks[row]["asset_universe_name"] = ""
        self._update_asset_summary(selected)
        self._save_current_block_state()

    def _apply_universe_to_block(self, row: int, universe: dict) -> bool:
        dataset_ids = self._datasets_from_universe(universe)
        if row < 0 or row >= len(self.blocks):
            return False
        if not dataset_ids:
            QtWidgets.QMessageBox.information(
                self,
                "Universe has no local datasets",
                "The selected universe does not have any local dataset bindings available for strategy blocks yet.",
            )
            return False
        self.blocks[row]["asset_dataset_ids"] = dataset_ids
        self.blocks[row]["asset_universe_id"] = str(universe.get("universe_id") or "")
        self.blocks[row]["asset_universe_name"] = str(universe.get("name") or "")
        self.blocks[row]["asset_weights_text"] = ""
        return True

    def _apply_selected_universe_to_current_block(self) -> None:
        row = self._current_index
        universe = self._find_universe(str(self.block_universe_combo.currentData() or ""))
        if row < 0 or row >= len(self.blocks):
            return
        if universe is None:
            QtWidgets.QMessageBox.information(self, "No universe selected", "Choose a saved universe first.")
            return
        if not self._apply_universe_to_block(row, universe):
            return
        self._update_asset_summary(list(self.blocks[row].get("asset_dataset_ids") or []))
        self.block_asset_weights_edit.clear()
        self._save_current_block_state()

    def _apply_selected_universe_to_all_blocks(self) -> None:
        universe = self._find_universe(str(self.block_universe_combo.currentData() or ""))
        if universe is None:
            QtWidgets.QMessageBox.information(self, "No universe selected", "Choose a saved universe first.")
            return
        applied = 0
        for row in range(len(self.blocks)):
            if self._apply_universe_to_block(row, universe):
                applied += 1
        if applied <= 0:
            return
        current = self._current_index
        self._refresh_block_list()
        if 0 <= current < len(self.blocks):
            self.block_list.setCurrentRow(current)
        QtWidgets.QMessageBox.information(
            self,
            "Universe applied",
            f"Applied universe '{universe.get('name') or universe.get('universe_id') or 'Universe'}' to {applied} strategy block(s).",
        )

    def _build_dataset_scoped_block(self, template_block: dict, dataset_id: str, universe: dict, ordinal: int) -> dict:
        dataset_label = str(dataset_id).strip() or f"asset_{ordinal + 1}"
        base_display_name = str(template_block.get("display_name") or template_block.get("strategy_name") or "Strategy Block").strip()
        display_name = f"{base_display_name} - {dataset_label}"
        template_key = self._slugify(
            str(template_block.get("block_id") or base_display_name),
            f"template_{ordinal + 1}",
        )
        dataset_key = self._slugify(dataset_label, f"asset_{ordinal + 1}")
        return {
            "block_id": self._slugify(f"{template_key}_{dataset_key}", f"block_{len(self.blocks) + ordinal + 1}"),
            "display_name": display_name,
            "strategy_name": str(template_block.get("strategy_name") or next(iter(self.strategy_specs.keys()), "SMACrossStrategy")),
            "strategy_params": dict(template_block.get("strategy_params") or {}),
            "asset_dataset_ids": [dataset_label],
            "asset_universe_id": str(universe.get("universe_id") or ""),
            "asset_universe_name": str(universe.get("name") or ""),
            "asset_weights_text": "",
            "budget_weight": float(template_block.get("budget_weight") or 0.0),
        }

    def _append_blocks_from_selected_universe(self) -> None:
        universe = self._find_universe(str(self.block_universe_combo.currentData() or ""))
        if universe is None:
            QtWidgets.QMessageBox.information(self, "No universe selected", "Choose a saved universe first.")
            return
        dataset_ids = self._datasets_from_universe(universe)
        if not dataset_ids:
            QtWidgets.QMessageBox.information(
                self,
                "Universe has no local datasets",
                "The selected universe does not have any local dataset bindings available for strategy blocks yet.",
            )
            return
        row = self._current_index
        if row >= 0 and row < len(self.blocks):
            self._save_current_block_state()
            template_block = dict(self.blocks[row])
        else:
            template_block = self._default_block()

        replace_blank_template = bool(
            row >= 0
            and row < len(self.blocks)
            and len(self.blocks) == 1
            and not list(self.blocks[row].get("asset_dataset_ids") or [])
        )
        new_blocks = [
            self._build_dataset_scoped_block(template_block, dataset_id, universe, ordinal)
            for ordinal, dataset_id in enumerate(dataset_ids)
        ]
        if replace_blank_template:
            self.blocks = new_blocks
            selected_row = 0
            message = (
                f"Created {len(new_blocks)} strategy block(s) from universe "
                f"'{universe.get('name') or universe.get('universe_id') or 'Universe'}' using the current block as a template."
            )
        else:
            start_row = len(self.blocks)
            self.blocks.extend(new_blocks)
            selected_row = start_row
            message = (
                f"Appended {len(new_blocks)} strategy block(s) from universe "
                f"'{universe.get('name') or universe.get('universe_id') or 'Universe'}'."
            )
        self._refresh_block_list()
        if self.blocks:
            self.block_list.setCurrentRow(min(selected_row, len(self.blocks) - 1))
        QtWidgets.QMessageBox.information(self, "Blocks created", message)

    def _expand_existing_blocks_across_selected_universe(self) -> None:
        universe = self._find_universe(str(self.block_universe_combo.currentData() or ""))
        if universe is None:
            QtWidgets.QMessageBox.information(self, "No universe selected", "Choose a saved universe first.")
            return
        dataset_ids = self._datasets_from_universe(universe)
        if not dataset_ids:
            QtWidgets.QMessageBox.information(
                self,
                "Universe has no local datasets",
                "The selected universe does not have any local dataset bindings available for strategy blocks yet.",
            )
            return
        if self._current_index >= 0:
            self._save_current_block_state()
        template_blocks = [dict(block) for block in self.blocks] if self.blocks else [self._default_block()]
        replace_blank_template = bool(
            len(template_blocks) == 1
            and not list(template_blocks[0].get("asset_dataset_ids") or [])
        )
        expanded_blocks: list[dict] = []
        for template_idx, template_block in enumerate(template_blocks):
            for dataset_idx, dataset_id in enumerate(dataset_ids):
                ordinal = template_idx * len(dataset_ids) + dataset_idx
                expanded_blocks.append(
                    self._build_dataset_scoped_block(template_block, dataset_id, universe, ordinal)
                )
        if replace_blank_template:
            self.blocks = expanded_blocks
            selected_row = 0
        else:
            start_row = len(self.blocks)
            self.blocks.extend(expanded_blocks)
            selected_row = start_row
        self._refresh_block_list()
        if self.blocks:
            self.block_list.setCurrentRow(min(selected_row, len(self.blocks) - 1))
        QtWidgets.QMessageBox.information(
            self,
            "Blocks expanded",
            (
                f"Expanded {len(template_blocks)} strategy template(s) across {len(dataset_ids)} universe dataset(s) "
                f"into {len(expanded_blocks)} block(s)."
            ),
        )

    @staticmethod
    def _parse_single_strategy_param_value(ptype: type, token: str):
        text = token.strip()
        if ptype is bool:
            lowered = text.lower()
            if lowered in {"1", "true", "yes", "y", "on"}:
                return True
            if lowered in {"0", "false", "no", "n", "off"}:
                return False
            raise ValueError(f"invalid boolean literal: {token}")
        if ptype is str:
            return text
        return ptype(text)

    @staticmethod
    def _parse_asset_weights(dataset_ids: Sequence[str], weight_text: str) -> tuple[dict[str, float], list[str]]:
        datasets = [dataset_id for dataset_id in dataset_ids if dataset_id]
        if not datasets:
            return {}, []
        text = (weight_text or "").strip()
        if not text:
            return {}, []
        tokens = [token.strip() for token in text.split(",") if token.strip()]
        if not tokens:
            return {}, []
        weights: dict[str, float] = {}
        errors: list[str] = []
        if any("=" in token for token in tokens):
            for token in tokens:
                if "=" not in token:
                    errors.append(f"Invalid asset weight token '{token}'. Use dataset=weight format.")
                    continue
                dataset_id, raw_weight = token.split("=", 1)
                dataset_id = dataset_id.strip()
                if dataset_id not in datasets:
                    errors.append(f"Asset weight dataset '{dataset_id}' is not attached to this strategy block.")
                    continue
                try:
                    weight = float(raw_weight.strip())
                except Exception:
                    errors.append(f"Asset weight for '{dataset_id}' is not a valid number.")
                    continue
                if weight < 0:
                    errors.append(f"Asset weight for '{dataset_id}' must be non-negative.")
                    continue
                weights[dataset_id] = weight
        else:
            if len(tokens) != len(datasets):
                errors.append(
                    f"Positional asset weights must provide exactly {len(datasets)} value(s) for this block."
                )
                return {}, errors
            for dataset_id, token in zip(datasets, tokens):
                try:
                    weight = float(token)
                except Exception:
                    errors.append(f"Asset weight '{token}' is not a valid number.")
                    continue
                if weight < 0:
                    errors.append(f"Asset weight for '{dataset_id}' must be non-negative.")
                    continue
                weights[dataset_id] = weight
        if not errors and weights and sum(weights.values()) <= 0:
            errors.append("Asset weights must sum to more than zero when provided.")
        return weights, errors

    def strategy_blocks(self) -> list[dict]:
        if self._current_index >= 0:
            self._save_current_block_state()
        return [dict(block) for block in self.blocks]

    def accept(self) -> None:
        blocks = self.strategy_blocks()
        if not blocks:
            QtWidgets.QMessageBox.information(self, "No blocks", "Add at least one strategy block.")
            return
        cleaned_blocks: list[dict] = []
        seen_block_ids: set[str] = set()
        for idx, block in enumerate(blocks, start=1):
            display_name = str(block.get("display_name") or f"Strategy Block {idx}").strip()
            block_id = self._slugify(str(block.get("block_id") or display_name), f"block_{idx}")
            if block_id in seen_block_ids:
                QtWidgets.QMessageBox.warning(self, "Duplicate block", f"Duplicate strategy block id '{block_id}'.")
                return
            seen_block_ids.add(block_id)
            strategy_name = str(block.get("strategy_name") or "").strip()
            spec = self.strategy_specs.get(strategy_name)
            if not spec:
                QtWidgets.QMessageBox.warning(self, "Invalid strategy", f"Unknown strategy for block '{display_name}'.")
                return
            dataset_ids = [str(dataset_id) for dataset_id in list(block.get("asset_dataset_ids") or []) if str(dataset_id).strip()]
            if not dataset_ids:
                QtWidgets.QMessageBox.warning(self, "Missing assets", f"Block '{display_name}' must include at least one asset.")
                return
            asset_weights, weight_errors = self._parse_asset_weights(dataset_ids, str(block.get("asset_weights_text") or ""))
            if weight_errors:
                QtWidgets.QMessageBox.warning(self, "Invalid asset weights", weight_errors[0])
                return
            parsed_params: dict[str, object] = {}
            raw_params = dict(block.get("strategy_params") or {})
            for param_name, (ptype, default) in dict(spec.get("params") or {}).items():
                raw_value = str(raw_params.get(param_name, default)).strip()
                try:
                    parsed_params[param_name] = self._parse_single_strategy_param_value(ptype, raw_value)
                except Exception:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Invalid parameter",
                        f"Block '{display_name}' has an invalid value for '{param_name}'.",
                    )
                    return
            cleaned_blocks.append(
                {
                    "block_id": block_id,
                    "display_name": display_name,
                    "strategy_name": strategy_name,
                    "strategy_params": parsed_params,
                    "asset_dataset_ids": dataset_ids,
                    "asset_universe_id": str(block.get("asset_universe_id") or ""),
                    "asset_universe_name": str(block.get("asset_universe_name") or ""),
                    "asset_target_weights": asset_weights,
                    "budget_weight": (float(block.get("budget_weight") or 0.0) or None),
                }
            )
        self.blocks = cleaned_blocks
        super().accept()


class WalkForwardSetupDialog(DashboardDialog):
    def __init__(self, study_row: dict, catalog: CatalogReader, parent=None) -> None:
        super().__init__(parent)
        self.study_row = dict(study_row)
        self.catalog = catalog
        self.study_id = str(self.study_row.get("study_id", "") or "")
        self.setWindowTitle("New Walk-Forward Study")
        self.resize(860, 720)
        self.setObjectName("Panel")
        self.setStyleSheet(
            f"""
            QDialog#Panel {{
                background: {PALETTE['panel']};
                color: {PALETTE['text']};
            }}
            QGroupBox {{
                color: {PALETTE['text']};
                border: 1px solid rgba(231, 238, 252, 0.38);
                border-radius: 10px;
                margin-top: 12px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: {PALETTE['muted']};
                font-weight: 700;
            }}
            """
        )
        self._duck = DuckDBStore()
        self._candidate_frame = self.catalog.load_optimization_candidates(self.study_id)
        self._last_suggested_test_window_bars: int | None = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        intro = QtWidgets.QLabel(
            "Seed a walk-forward validation study from an existing optimization study. "
            "Version 1 stays single-strategy and single-dataset."
        )
        intro.setObjectName("Sub")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        summary = QtWidgets.QPlainTextEdit()
        summary.setReadOnly(True)
        summary.setMaximumHeight(112)
        summary.setObjectName("Panel")
        summary.setPlainText(
            "\n".join(
                [
                    f"Study: {self.study_id}",
                    f"Strategy: {self.study_row.get('strategy', '')}",
                    f"Datasets: {', '.join(list(self.study_row.get('dataset_scope') or [])) or '—'}",
                    f"Timeframes: {', '.join(list(self.study_row.get('timeframes') or [])) or '—'}",
                    f"Promoted candidates: {len(self._candidate_frame)}",
                ]
            )
        )
        layout.addWidget(summary)

        form_box = QtWidgets.QGroupBox("Walk-Forward Setup")
        form = QtWidgets.QFormLayout(form_box)
        form.setContentsMargins(12, 12, 12, 12)
        form.setSpacing(10)

        self.dataset_combo = QtWidgets.QComboBox()
        for dataset_id in list(self.study_row.get("dataset_scope") or []):
            self.dataset_combo.addItem(str(dataset_id), str(dataset_id))
        form.addRow("Dataset", self.dataset_combo)

        self.timeframe_combo = QtWidgets.QComboBox()
        for timeframe in list(self.study_row.get("timeframes") or []):
            self.timeframe_combo.addItem(str(timeframe), str(timeframe))
        form.addRow("Timeframe", self.timeframe_combo)

        self.source_mode_combo = QtWidgets.QComboBox()
        self.source_mode_combo.addItem("Full Grid Per Fold", WALK_FORWARD_SOURCE_FULL_GRID)
        self.source_mode_combo.addItem("Reduced Candidate Set", WALK_FORWARD_SOURCE_REDUCED_CANDIDATES)
        if not self._candidate_frame.empty:
            self.source_mode_combo.setCurrentIndex(1)
        form.addRow("Candidate Source", self.source_mode_combo)

        self.first_test_start_edit = QtWidgets.QLineEdit()
        self.first_test_start_edit.setPlaceholderText("2024-01-01T14:30:00+00:00")
        form.addRow("First Test Start", self.first_test_start_edit)

        self.test_window_spin = QtWidgets.QSpinBox()
        self.test_window_spin.setRange(5, 500000)
        self.test_window_spin.setValue(60)
        form.addRow("Test Window Bars", self.test_window_spin)

        self.num_folds_spin = QtWidgets.QSpinBox()
        self.num_folds_spin.setRange(1, 250)
        self.num_folds_spin.setValue(4)
        form.addRow("Number Of Folds", self.num_folds_spin)

        self.min_train_spin = QtWidgets.QSpinBox()
        self.min_train_spin.setRange(10, 500000)
        self.min_train_spin.setValue(120)
        form.addRow("Min Train Bars", self.min_train_spin)

        self.execution_mode_combo = QtWidgets.QComboBox()
        self.execution_mode_combo.addItem("Auto (Recommended)", ExecutionMode.AUTO.value)
        self.execution_mode_combo.addItem("Reference", ExecutionMode.REFERENCE.value)
        self.execution_mode_combo.addItem("Vectorized", ExecutionMode.VECTORIZED.value)
        form.addRow("Execution Mode", self.execution_mode_combo)

        self.description_edit = QtWidgets.QPlainTextEdit()
        self.description_edit.setMaximumHeight(96)
        self.description_edit.setPlaceholderText("Optional notes for this walk-forward study.")
        form.addRow("Description", self.description_edit)

        self.preview_label = QtWidgets.QLabel("")
        self.preview_label.setObjectName("Sub")
        self.preview_label.setWordWrap(True)
        form.addRow("Data Preview", self.preview_label)

        self.mode_note = QtWidgets.QLabel("")
        self.mode_note.setObjectName("Sub")
        self.mode_note.setWordWrap(True)
        form.addRow("", self.mode_note)

        layout.addWidget(form_box, 1)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.dataset_combo.currentIndexChanged.connect(self._refresh_preview)
        self.timeframe_combo.currentIndexChanged.connect(self._refresh_preview)
        self.source_mode_combo.currentIndexChanged.connect(self._refresh_preview)
        self.min_train_spin.valueChanged.connect(self._refresh_preview)
        self.num_folds_spin.valueChanged.connect(self._refresh_preview)
        self.test_window_spin.valueChanged.connect(self._refresh_preview)
        self._refresh_preview()

    def _refresh_preview(self) -> None:
        dataset_id = str(self.dataset_combo.currentData() or "")
        timeframe = str(self.timeframe_combo.currentData() or "")
        source_mode = str(self.source_mode_combo.currentData() or WALK_FORWARD_SOURCE_FULL_GRID)
        if not dataset_id or not timeframe:
            self.preview_label.setText("Choose a dataset and timeframe to preview the available bar range.")
            return
        try:
            bars = self._duck.resample(dataset_id, timeframe)
        except Exception as exc:
            self.preview_label.setText(f"Failed to load bars for preview: {exc}")
            return
        if bars is None or bars.empty:
            self.preview_label.setText("No bars are available for the selected dataset/timeframe.")
            return
        min_train = int(self.min_train_spin.value())
        suggested_pos = min(len(bars) - 1, max(min_train, int(len(bars) * 0.65)))
        suggested_test_window = self._suggest_test_window_bars(
            bars=bars,
            suggested_test_start_pos=suggested_pos,
            num_folds=int(self.num_folds_spin.value()),
        )
        current_test_window = int(self.test_window_spin.value())
        if (
            self._last_suggested_test_window_bars is None
            or current_test_window in {60, self._last_suggested_test_window_bars}
        ):
            self.test_window_spin.blockSignals(True)
            self.test_window_spin.setValue(int(suggested_test_window))
            self.test_window_spin.blockSignals(False)
            current_test_window = int(suggested_test_window)
        self._last_suggested_test_window_bars = int(suggested_test_window)
        suggested_start = bars.index[suggested_pos].isoformat()
        self.first_test_start_edit.setText(suggested_start)
        bars_per_day = self._bars_per_trading_day(bars)
        test_window_days = (float(current_test_window) / bars_per_day) if bars_per_day > 0 else None
        suggested_days = (float(suggested_test_window) / bars_per_day) if bars_per_day > 0 else None
        range_text = (
            f"Bars: {len(bars)} | Range: {self._fmt_timestamp(bars.index[0])} -> {self._fmt_timestamp(bars.index[-1])}\n"
            f"Suggested first test start: {self._fmt_timestamp(bars.index[suggested_pos])}\n"
            f"Current test window: {current_test_window} bars"
            + (f" (~{test_window_days:.1f} trading days)" if test_window_days is not None else "")
            + "\n"
            f"Suggested test window: {suggested_test_window} bars"
            + (f" (~{suggested_days:.1f} trading days)" if suggested_days is not None else "")
        )
        candidate_count = len(self._candidate_frame.loc[self._candidate_frame["timeframe"].fillna("") == timeframe]) if not self._candidate_frame.empty else 0
        if source_mode == WALK_FORWARD_SOURCE_REDUCED_CANDIDATES:
            range_text += f"\nPromoted candidates for this timeframe: {candidate_count}"
        else:
            range_text += "\nFull-grid mode will rebuild the train-fold parameter universe from the optimization aggregates."
        self.preview_label.setText(range_text)
        short_window_warning = ""
        if test_window_days is not None and test_window_days < 5.0:
            short_window_warning = (
                "Warning: the current out-of-sample window is only "
                f"{test_window_days:.1f} trading days. That is usually too short to produce meaningful walk-forward validation."
            )
        if source_mode == WALK_FORWARD_SOURCE_REDUCED_CANDIDATES and candidate_count == 0:
            note = (
                "No promoted candidates exist for this timeframe yet. Queue candidates from the optimization study first, or switch to Full Grid Per Fold."
            )
            if short_window_warning:
                note += f"\n\n{short_window_warning}"
            self.mode_note.setText(note)
        else:
            note = "Walk-forward v1 uses anchored folds with expanding train windows and a stitched out-of-sample equity path."
            if short_window_warning:
                note += f"\n\n{short_window_warning}"
            self.mode_note.setText(note)

    @staticmethod
    def _fmt_timestamp(value) -> str:
        ts = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.isna(ts):
            return str(value or "")
        return ts.tz_convert("America/New_York").strftime("%Y-%m-%d %H:%M")

    @staticmethod
    def _bars_per_trading_day(bars: pd.DataFrame) -> float:
        if bars is None or bars.empty:
            return 0.0
        index = pd.DatetimeIndex(bars.index)
        if index.tz is None:
            index = index.tz_localize("UTC")
        local_days = index.tz_convert("America/New_York").normalize()
        counts = pd.Series(1, index=local_days).groupby(level=0).sum()
        if counts.empty:
            return float(len(index))
        return float(counts.median())

    def _suggest_test_window_bars(
        self,
        *,
        bars: pd.DataFrame,
        suggested_test_start_pos: int,
        num_folds: int,
    ) -> int:
        bars_per_day = self._bars_per_trading_day(bars)
        base_target = max(20, int(round(bars_per_day * 20.0))) if bars_per_day > 0 else 60
        remaining = max(5, int(len(bars) - suggested_test_start_pos - 1))
        max_fit = max(5, int(remaining // max(1, num_folds)))
        return max(5, min(base_target, max_fit))

    def settings(self) -> dict:
        return {
            "dataset_id": str(self.dataset_combo.currentData() or ""),
            "timeframe": str(self.timeframe_combo.currentData() or ""),
            "candidate_source_mode": str(self.source_mode_combo.currentData() or WALK_FORWARD_SOURCE_FULL_GRID),
            "first_test_start": self.first_test_start_edit.text().strip(),
            "test_window_bars": int(self.test_window_spin.value()),
            "num_folds": int(self.num_folds_spin.value()),
            "min_train_bars": int(self.min_train_spin.value()),
            "execution_mode": str(self.execution_mode_combo.currentData() or ExecutionMode.AUTO.value),
            "description": self.description_edit.toPlainText().strip(),
        }

    def accept(self) -> None:
        settings = self.settings()
        if not settings["dataset_id"] or not settings["timeframe"]:
            QtWidgets.QMessageBox.warning(self, "Missing Selection", "Choose a dataset and timeframe first.")
            return
        try:
            pd.Timestamp(settings["first_test_start"])
        except Exception:
            QtWidgets.QMessageBox.warning(self, "Invalid Start", "First Test Start must be a valid timestamp.")
            return
        if settings["candidate_source_mode"] == WALK_FORWARD_SOURCE_REDUCED_CANDIDATES:
            candidate_count = 0
            if not self._candidate_frame.empty:
                candidate_count = int(
                    (
                        self._candidate_frame["timeframe"].fillna("")
                        == str(settings["timeframe"])
                    ).sum()
                )
            if candidate_count <= 0:
                QtWidgets.QMessageBox.warning(
                    self,
                    "No Candidates",
                    "No promoted optimization candidates are available for the selected timeframe.",
                )
                return
        super().accept()


class PortfolioWalkForwardSetupDialog(DashboardDialog):
    def __init__(self, source_row: dict, catalog: CatalogReader, parent=None) -> None:
        super().__init__(parent)
        self.source_row = dict(source_row)
        self.catalog = catalog
        self.batch_id = str(self.source_row.get("batch_id", "") or "")
        self.dataset_ids = [str(item) for item in list(self.source_row.get("dataset_ids") or []) if str(item).strip()]
        self.timeframes = [str(item) for item in list(self.source_row.get("timeframes") or []) if str(item).strip()]
        self.mode = str(self.source_row.get("mode", "shared_strategy") or "shared_strategy")
        self._candidate_frame = self.catalog.load_optimization_candidates(self.batch_id)
        self.setWindowTitle("New Portfolio Walk-Forward Study")
        self.resize(920, 760)
        self.setObjectName("Panel")
        self.setStyleSheet(
            f"""
            QDialog#Panel {{
                background: {PALETTE['panel']};
                color: {PALETTE['text']};
            }}
            QGroupBox {{
                color: {PALETTE['text']};
                border: 1px solid rgba(231, 238, 252, 0.38);
                border-radius: 10px;
                margin-top: 12px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: {PALETTE['muted']};
                font-weight: 700;
            }}
            """
        )
        self._duck = DuckDBStore()
        self._last_suggested_test_window_bars: int | None = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        intro = QtWidgets.QLabel(
            "Validate a saved portfolio batch out of sample. Shared-strategy portfolios re-run a train-fold grid, "
            "while fixed strategy-block portfolios revalidate the saved block definition across anchored folds."
        )
        intro.setObjectName("Sub")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        summary = QtWidgets.QPlainTextEdit()
        summary.setReadOnly(True)
        summary.setMaximumHeight(136)
        summary.setObjectName("Panel")
        summary.setPlainText(
            "\n".join(
                [
                    f"Batch: {self.batch_id}",
                    f"Strategy: {self.source_row.get('strategy', '')}",
                    f"Mode: {'Fixed strategy blocks' if self.mode == 'strategy_blocks' else 'Shared strategy portfolio'}",
                    f"Datasets: {', '.join(self.dataset_ids) or '—'}",
                    f"Timeframes: {', '.join(self.timeframes) or '—'}",
                    f"Runs: {int(self.source_row.get('run_count', 0) or 0)}",
                ]
            )
        )
        layout.addWidget(summary)

        form_box = QtWidgets.QGroupBox("Walk-Forward Setup")
        form = QtWidgets.QFormLayout(form_box)
        form.setContentsMargins(12, 12, 12, 12)
        form.setSpacing(10)

        self.timeframe_combo = QtWidgets.QComboBox()
        for timeframe in self.timeframes:
            self.timeframe_combo.addItem(str(timeframe), str(timeframe))
        form.addRow("Timeframe", self.timeframe_combo)

        self.source_mode_combo = QtWidgets.QComboBox()
        if self.mode == "strategy_blocks":
            self.source_mode_combo.addItem("Fixed Portfolio Definition", WALK_FORWARD_SOURCE_FIXED_PORTFOLIO)
            self.source_mode_combo.addItem("Reduced Candidate Set", WALK_FORWARD_SOURCE_REDUCED_CANDIDATES)
        else:
            self.source_mode_combo.addItem("Full Grid Per Fold", WALK_FORWARD_SOURCE_FULL_GRID)
            self.source_mode_combo.addItem("Reduced Candidate Set", WALK_FORWARD_SOURCE_REDUCED_CANDIDATES)
        form.addRow("Candidate Source", self.source_mode_combo)

        self.first_test_start_edit = QtWidgets.QLineEdit()
        self.first_test_start_edit.setPlaceholderText("2024-01-01T14:30:00+00:00")
        form.addRow("First Test Start", self.first_test_start_edit)

        self.test_window_spin = QtWidgets.QSpinBox()
        self.test_window_spin.setRange(5, 500000)
        self.test_window_spin.setValue(120)
        form.addRow("Test Window Bars", self.test_window_spin)

        self.num_folds_spin = QtWidgets.QSpinBox()
        self.num_folds_spin.setRange(1, 250)
        self.num_folds_spin.setValue(4)
        form.addRow("Number Of Folds", self.num_folds_spin)

        self.min_train_spin = QtWidgets.QSpinBox()
        self.min_train_spin.setRange(10, 500000)
        self.min_train_spin.setValue(240)
        form.addRow("Min Train Bars", self.min_train_spin)

        self.execution_mode_combo = QtWidgets.QComboBox()
        self.execution_mode_combo.addItem("Auto (Recommended)", ExecutionMode.AUTO.value)
        self.execution_mode_combo.addItem("Vectorized", ExecutionMode.VECTORIZED.value)
        form.addRow("Execution Mode", self.execution_mode_combo)

        self.description_edit = QtWidgets.QPlainTextEdit()
        self.description_edit.setMaximumHeight(96)
        self.description_edit.setPlaceholderText("Optional notes for this portfolio walk-forward study.")
        form.addRow("Description", self.description_edit)

        self.preview_label = QtWidgets.QLabel("")
        self.preview_label.setObjectName("Sub")
        self.preview_label.setWordWrap(True)
        form.addRow("Data Preview", self.preview_label)

        self.mode_note = QtWidgets.QLabel("")
        self.mode_note.setObjectName("Sub")
        self.mode_note.setWordWrap(True)
        form.addRow("", self.mode_note)

        layout.addWidget(form_box, 1)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.timeframe_combo.currentIndexChanged.connect(self._refresh_preview)
        self.source_mode_combo.currentIndexChanged.connect(self._refresh_preview)
        self.min_train_spin.valueChanged.connect(self._refresh_preview)
        self.num_folds_spin.valueChanged.connect(self._refresh_preview)
        self.test_window_spin.valueChanged.connect(self._refresh_preview)
        self._refresh_preview()

    def _load_common_index(self, timeframe: str) -> pd.DatetimeIndex:
        frames: list[pd.DataFrame] = []
        for dataset_id in self.dataset_ids:
            bars = self._duck.resample(dataset_id, timeframe)
            if bars is None or bars.empty:
                raise RuntimeError(f"No bars were available for dataset '{dataset_id}' at timeframe '{timeframe}'.")
            frame = bars.copy()
            frame.index = pd.to_datetime(frame.index, utc=True)
            frame = frame.sort_index()
            frames.append(frame)
        if not frames:
            raise RuntimeError("No datasets are attached to this portfolio source.")
        common_index = pd.DatetimeIndex(frames[0].index)
        for frame in frames[1:]:
            common_index = common_index.intersection(pd.DatetimeIndex(frame.index))
        if common_index.empty:
            raise RuntimeError("The selected portfolio datasets do not share a non-empty common timestamp range.")
        return common_index

    def _refresh_preview(self) -> None:
        timeframe = str(self.timeframe_combo.currentData() or "")
        source_mode = str(self.source_mode_combo.currentData() or WALK_FORWARD_SOURCE_FULL_GRID)
        if not timeframe:
            self.preview_label.setText("Choose a timeframe to preview the shared portfolio history.")
            return
        try:
            common_index = self._load_common_index(timeframe)
        except Exception as exc:
            self.preview_label.setText(f"Failed to build a shared portfolio preview: {exc}")
            return
        min_train = int(self.min_train_spin.value())
        suggested_pos = min(len(common_index) - 1, max(min_train, int(len(common_index) * 0.65)))
        suggested_test_window = self._suggest_test_window_bars(
            index=common_index,
            suggested_test_start_pos=suggested_pos,
            num_folds=int(self.num_folds_spin.value()),
        )
        current_test_window = int(self.test_window_spin.value())
        if (
            self._last_suggested_test_window_bars is None
            or current_test_window in {120, self._last_suggested_test_window_bars}
        ):
            self.test_window_spin.blockSignals(True)
            self.test_window_spin.setValue(int(suggested_test_window))
            self.test_window_spin.blockSignals(False)
            current_test_window = int(suggested_test_window)
        self._last_suggested_test_window_bars = int(suggested_test_window)
        suggested_start = common_index[suggested_pos].isoformat()
        self.first_test_start_edit.setText(suggested_start)
        bars_per_day = self._bars_per_trading_day(common_index)
        test_window_days = (float(current_test_window) / bars_per_day) if bars_per_day > 0 else None
        suggested_days = (float(suggested_test_window) / bars_per_day) if bars_per_day > 0 else None
        preview_lines = [
            f"Common bars across {len(self.dataset_ids)} dataset(s): {len(common_index)}",
            f"Shared range: {WalkForwardSetupDialog._fmt_timestamp(common_index[0])} -> {WalkForwardSetupDialog._fmt_timestamp(common_index[-1])}",
            f"Suggested first test start: {WalkForwardSetupDialog._fmt_timestamp(common_index[suggested_pos])}",
            (
                f"Current test window: {current_test_window} bars"
                + (f" (~{test_window_days:.1f} trading days)" if test_window_days is not None else "")
            ),
            (
                f"Suggested test window: {suggested_test_window} bars"
                + (f" (~{suggested_days:.1f} trading days)" if suggested_days is not None else "")
            ),
        ]
        candidate_count = len(
            self._candidate_frame.loc[self._candidate_frame["timeframe"].fillna("") == timeframe]
        ) if not self._candidate_frame.empty else 0
        if self.mode != "strategy_blocks":
            if source_mode == WALK_FORWARD_SOURCE_REDUCED_CANDIDATES:
                preview_lines.append(f"Promoted candidates for this timeframe: {candidate_count}")
            else:
                preview_lines.append("Full-grid mode will rebuild the train-fold portfolio parameter universe from the saved study.")
        elif source_mode == WALK_FORWARD_SOURCE_REDUCED_CANDIDATES:
            preview_lines.append(f"Promoted fixed portfolio definitions for this timeframe: {candidate_count}")
        self.preview_label.setText("\n".join(preview_lines))
        short_window_warning = ""
        if test_window_days is not None and test_window_days < 5.0:
            short_window_warning = (
                "Warning: the current out-of-sample window is only "
                f"{test_window_days:.1f} trading days across the shared portfolio index. "
                "That is usually too short to validate a portfolio study meaningfully."
            )
        if self.mode == "strategy_blocks":
            if source_mode == WALK_FORWARD_SOURCE_REDUCED_CANDIDATES:
                note = (
                    "This source uses fixed strategy blocks. Each fold will revalidate one of the promoted fixed "
                    "portfolio definitions saved on the source batch."
                )
            else:
                note = "This source uses fixed strategy blocks. Each fold will revalidate the saved multi-strategy portfolio definition out of sample."
        else:
            if source_mode == WALK_FORWARD_SOURCE_REDUCED_CANDIDATES:
                note = "This source uses a shared strategy across multiple assets. Each train fold will choose from the promoted portfolio candidates saved on the source study."
            else:
                note = "This source uses a shared strategy across multiple assets. Each train fold will rebuild the portfolio parameter surface from the saved grid."
            if source_mode == WALK_FORWARD_SOURCE_REDUCED_CANDIDATES and candidate_count <= 0:
                note += "\n\nNo promoted candidates exist for this timeframe yet. Promote candidates from the portfolio optimization study or switch to Full Grid Per Fold."
        if self.mode == "strategy_blocks" and source_mode == WALK_FORWARD_SOURCE_REDUCED_CANDIDATES and candidate_count <= 0:
            note += "\n\nNo promoted fixed portfolio definitions exist for this timeframe yet. Promote the current strategy-block definition from the batch detail window or switch to Fixed Portfolio Definition."
        if short_window_warning:
            note += f"\n\n{short_window_warning}"
        self.mode_note.setText(note)

    @staticmethod
    def _bars_per_trading_day(index: pd.DatetimeIndex) -> float:
        normalized = pd.DatetimeIndex(index)
        if normalized.tz is None:
            normalized = normalized.tz_localize("UTC")
        local_days = normalized.tz_convert("America/New_York").normalize()
        counts = pd.Series(1, index=local_days).groupby(level=0).sum()
        if counts.empty:
            return float(len(normalized))
        return float(counts.median())

    def _suggest_test_window_bars(
        self,
        *,
        index: pd.DatetimeIndex,
        suggested_test_start_pos: int,
        num_folds: int,
    ) -> int:
        bars_per_day = self._bars_per_trading_day(index)
        base_target = max(20, int(round(bars_per_day * 20.0))) if bars_per_day > 0 else 120
        remaining = max(5, int(len(index) - suggested_test_start_pos - 1))
        max_fit = max(5, int(remaining // max(1, num_folds)))
        return max(5, min(base_target, max_fit))

    def settings(self) -> dict:
        return {
            "timeframe": str(self.timeframe_combo.currentData() or ""),
            "candidate_source_mode": str(self.source_mode_combo.currentData() or WALK_FORWARD_SOURCE_FULL_GRID),
            "first_test_start": self.first_test_start_edit.text().strip(),
            "test_window_bars": int(self.test_window_spin.value()),
            "num_folds": int(self.num_folds_spin.value()),
            "min_train_bars": int(self.min_train_spin.value()),
            "execution_mode": str(self.execution_mode_combo.currentData() or ExecutionMode.AUTO.value),
            "description": self.description_edit.toPlainText().strip(),
        }

    def _portfolio_support_issues(self) -> list[str]:
        parent = self.logical_parent()
        if parent is None or not hasattr(parent, "_collect_backtest_settings"):
            return []
        try:
            bt_settings = parent._collect_backtest_settings()
        except Exception:
            return []
        timeframe = str(self.timeframe_combo.currentData() or "")
        if self.mode == "strategy_blocks":
            if not hasattr(parent, "_portfolio_strategy_block_support_issues"):
                return []
            batch_params = self.source_row.get("params_dict")
            if not isinstance(batch_params, dict):
                try:
                    batch_params = json.loads(str(self.source_row.get("params") or "{}"))
                except Exception:
                    batch_params = {}
            strategy_blocks = PortfolioWalkForwardWorker._strategy_blocks_from_batch_params(batch_params or {})
            issues, _ = parent._portfolio_strategy_block_support_issues(strategy_blocks, bt_settings)
            return list(dict.fromkeys(str(issue) for issue in issues if str(issue).strip()))
        if not hasattr(parent, "_portfolio_vectorized_support_issues"):
            return []
        strategy_cls = PortfolioWalkForwardWorker._shared_strategy_cls(str(self.source_row.get("strategy", "") or ""))
        issues, _ = parent._portfolio_vectorized_support_issues(strategy_cls, [timeframe], bt_settings)
        return list(dict.fromkeys(str(issue) for issue in issues if str(issue).strip()))

    def accept(self) -> None:
        settings = self.settings()
        if not settings["timeframe"]:
            QtWidgets.QMessageBox.warning(self, "Missing Timeframe", "Choose a timeframe first.")
            return
        try:
            pd.Timestamp(settings["first_test_start"])
        except Exception:
            QtWidgets.QMessageBox.warning(self, "Invalid Start", "First Test Start must be a valid timestamp.")
            return
        if settings["candidate_source_mode"] == WALK_FORWARD_SOURCE_REDUCED_CANDIDATES:
            candidate_count = 0
            if not self._candidate_frame.empty:
                candidate_count = int(
                    (self._candidate_frame["timeframe"].fillna("") == str(settings["timeframe"])).sum()
                )
            if candidate_count <= 0:
                QtWidgets.QMessageBox.warning(
                    self,
                    "No Candidates",
                    (
                        "No promoted fixed portfolio definitions are available for the selected timeframe."
                        if self.mode == "strategy_blocks"
                        else "No promoted portfolio candidates are available for the selected timeframe."
                    ),
                )
                return
        support_issues = self._portfolio_support_issues()
        if support_issues:
            QtWidgets.QMessageBox.warning(
                self,
                "Unsupported Portfolio Settings",
                "\n".join(support_issues),
            )
            return
        super().accept()


class WalkForwardStudyDialog(DashboardDialog):
    def __init__(self, study_row: dict, catalog: CatalogReader, parent=None) -> None:
        super().__init__(parent)
        self.study_row = dict(study_row)
        self.catalog = catalog
        self.wf_study_id = str(self.study_row.get("wf_study_id", "") or "")
        self.schedule_payload = self._parse_json_text(self.study_row.get("schedule_json"))
        self.params_payload = self._parse_json_text(self.study_row.get("params_json"))
        self.is_portfolio_study = str(self.params_payload.get("source_kind", "")) == "portfolio"
        self.source_study_id = str(self.params_payload.get("source_study_id", "") or "")
        self.source_batch_id = str(self.params_payload.get("source_batch_id", "") or "")
        self.portfolio_mode = str(self.schedule_payload.get("portfolio_mode", "shared_strategy") or "shared_strategy")
        self.portfolio_dataset_ids = [
            str(item) for item in list(self.schedule_payload.get("dataset_ids") or []) if str(item).strip()
        ]
        self.folds = self.catalog.load_walk_forward_folds(self.wf_study_id)
        self.fold_metrics = self.catalog.load_walk_forward_fold_metrics(self.wf_study_id)
        self.combined_folds = self._combined_fold_frame()
        self.setWindowTitle(f"Walk-Forward Study | {self.wf_study_id}")
        self.resize(1620, 1040)
        self.setObjectName("Panel")
        self.setStyleSheet(
            f"""
            QDialog#Panel {{
                background: {PALETTE['panel']};
                color: {PALETTE['text']};
            }}
            QGroupBox {{
                color: {PALETTE['text']};
                border: 1px solid rgba(231, 238, 252, 0.38);
                border-radius: 10px;
                margin-top: 12px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: {PALETTE['muted']};
                font-weight: 700;
            }}
            """
        )

        layout = QtWidgets.QVBoxLayout(self)
        summary = QtWidgets.QPlainTextEdit()
        summary.setReadOnly(True)
        summary.setMaximumHeight(116)
        summary.setObjectName("Panel")
        stitched_metrics = self._parse_json_text(self.study_row.get("stitched_metrics_json"))
        fold_count = int(self.study_row.get("fold_count", 0) or 0)
        summary_lines = [
            f"Study: {self.wf_study_id}",
            f"Strategy: {self.study_row.get('strategy', '')}",
            f"Dataset: {self.study_row.get('dataset_id', '')}",
            f"Timeframe: {self.study_row.get('timeframe', '')}",
            f"Source Mode: {self.study_row.get('candidate_source_mode', '')}",
            f"Folds: {fold_count} | "
            f"Stitched Return: {self._format_numeric(stitched_metrics.get('total_return'))} | "
            f"Stitched Sharpe: {self._format_numeric(stitched_metrics.get('sharpe'))} | "
            f"Stitched Max DD: {self._format_numeric(stitched_metrics.get('max_drawdown'))}",
        ]
        if self.is_portfolio_study:
            summary_lines.extend(
                [
                    f"Portfolio Mode: {'Fixed Strategy Blocks' if self.portfolio_mode == 'strategy_blocks' else 'Shared Strategy Portfolio'}",
                    f"Underlying Assets: {', '.join(self.portfolio_dataset_ids) or '—'}",
                    f"Selection Rule: {self.study_row.get('selection_rule', '')}",
                ]
            )
        if self.source_study_id:
            summary_lines.append(f"Source Optimization Study: {self.source_study_id}")
        if self.source_batch_id and self.is_portfolio_study:
            summary_lines.append(f"Source Portfolio Batch: {self.source_batch_id}")
        summary.setPlainText("\n".join(summary_lines))
        layout.addWidget(summary)
        self.study_summary_label = QtWidgets.QLabel("")
        self.study_summary_label.setObjectName("Sub")
        self.study_summary_label.setWordWrap(True)
        self.study_summary_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.study_summary_label)
        relationship_row = QtWidgets.QHBoxLayout()
        self.open_source_optimization_btn = QtWidgets.QPushButton("Open Source Optimization Study")
        self.open_source_optimization_btn.clicked.connect(self._open_source_optimization_study)
        self.open_related_mc_btn = QtWidgets.QPushButton("Open Latest Monte Carlo Study")
        self.open_related_mc_btn.clicked.connect(self._open_latest_related_monte_carlo)
        relationship_row.addWidget(self.open_source_optimization_btn)
        relationship_row.addWidget(self.open_related_mc_btn)
        if self.is_portfolio_study:
            self.open_validation_chain_btn = QtWidgets.QPushButton("Portfolio Validation Chain")
            self.open_validation_chain_btn.clicked.connect(self._open_portfolio_validation_chain)
            relationship_row.addWidget(self.open_validation_chain_btn)
        relationship_row.addStretch(1)
        layout.addLayout(relationship_row)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        split.setChildrenCollapsible(False)
        layout.addWidget(split, 1)

        chart_panel = QtWidgets.QWidget()
        chart_panel.setObjectName("Panel")
        chart_layout = QtWidgets.QVBoxLayout(chart_panel)
        chart_layout.setContentsMargins(8, 8, 8, 8)
        chart_title = QtWidgets.QLabel("Stitched OOS Equity")
        chart_title.setObjectName("Title")
        chart_layout.addWidget(chart_title)
        self.wf_figure = Figure(figsize=(13.5, 5.0), tight_layout=True, facecolor=PALETTE["panel"])
        self.wf_canvas = FigureCanvasQTAgg(self.wf_figure)
        chart_layout.addWidget(self.wf_canvas, 1)
        split.addWidget(chart_panel)

        table_panel = QtWidgets.QWidget()
        table_panel.setObjectName("Panel")
        table_layout = QtWidgets.QVBoxLayout(table_panel)
        table_layout.setContentsMargins(8, 8, 8, 8)
        table_layout.setSpacing(8)
        table_title = QtWidgets.QLabel("Fold Summary")
        table_title.setObjectName("Title")
        table_layout.addWidget(table_title)
        lower_split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        lower_split.setChildrenCollapsible(False)
        table_layout.addWidget(lower_split, 1)

        fold_panel = QtWidgets.QWidget()
        fold_panel.setObjectName("Panel")
        fold_layout = QtWidgets.QVBoxLayout(fold_panel)
        fold_layout.setContentsMargins(0, 0, 0, 0)
        fold_layout.setSpacing(6)

        self.fold_table = QtWidgets.QTableWidget(0, 10)
        self.fold_table.setHorizontalHeaderLabels(
            [
                "Fold",
                "Train Period",
                "Test Period",
                "Selected Params",
                "Train Score",
                "Test Return",
                "Test Sharpe",
                "Test Max DD",
                "Trade Count",
                "Param Switches",
            ]
        )
        self.fold_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.fold_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.fold_table.setAlternatingRowColors(True)
        self.fold_table.horizontalHeader().setStretchLastSection(True)
        self.fold_table.verticalHeader().setVisible(False)
        self.fold_table.setObjectName("Panel")
        fold_layout.addWidget(self.fold_table, 1)
        lower_split.addWidget(fold_panel)

        analysis_tabs = QtWidgets.QTabWidget()
        lower_split.addWidget(analysis_tabs)

        detail_panel = QtWidgets.QWidget()
        detail_panel.setObjectName("Panel")
        detail_layout = QtWidgets.QVBoxLayout(detail_panel)
        detail_layout.setContentsMargins(8, 8, 8, 8)
        detail_layout.setSpacing(8)
        self.fold_analysis_figure = Figure(figsize=(7.2, 4.3), tight_layout=True, facecolor=PALETTE["panel"])
        self.fold_analysis_canvas = FigureCanvasQTAgg(self.fold_analysis_figure)
        detail_layout.addWidget(self.fold_analysis_canvas, 1)

        self.detail_notes = QtWidgets.QPlainTextEdit()
        self.detail_notes.setReadOnly(True)
        self.detail_notes.setMaximumHeight(150)
        self.detail_notes.setObjectName("Panel")
        detail_layout.addWidget(self.detail_notes)

        action_row = QtWidgets.QHBoxLayout()
        self.open_train_study_btn = QtWidgets.QPushButton("Open Selected Train Study")
        self.open_train_study_btn.clicked.connect(self._open_selected_train_study)
        self.open_test_run_btn = QtWidgets.QPushButton("Open Selected Test Run")
        self.open_test_run_btn.clicked.connect(self._open_selected_test_run)
        action_row.addWidget(self.open_train_study_btn)
        action_row.addWidget(self.open_test_run_btn)
        self.open_test_report_btn = QtWidgets.QPushButton("Open Selected Portfolio Report")
        self.open_test_report_btn.clicked.connect(self._open_selected_test_portfolio_report)
        self.open_test_report_btn.setVisible(self.is_portfolio_study)
        action_row.addWidget(self.open_test_report_btn)
        action_row.addStretch(1)
        detail_layout.addLayout(action_row)
        analysis_tabs.addTab(detail_panel, "Fold Analysis")

        drift_panel = QtWidgets.QWidget()
        drift_panel.setObjectName("Panel")
        drift_layout = QtWidgets.QVBoxLayout(drift_panel)
        drift_layout.setContentsMargins(8, 8, 8, 8)
        drift_layout.setSpacing(8)
        self.drift_figure = Figure(figsize=(7.2, 4.2), tight_layout=True, facecolor=PALETTE["panel"])
        self.drift_canvas = FigureCanvasQTAgg(self.drift_figure)
        drift_layout.addWidget(self.drift_canvas, 1)
        self.param_history_table = QtWidgets.QTableWidget(0, 0)
        self.param_history_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.param_history_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.param_history_table.setAlternatingRowColors(True)
        self.param_history_table.horizontalHeader().setStretchLastSection(True)
        self.param_history_table.verticalHeader().setVisible(False)
        self.param_history_table.setObjectName("Panel")
        drift_layout.addWidget(self.param_history_table, 1)
        analysis_tabs.addTab(drift_panel, "Parameter Drift")

        if self.is_portfolio_study:
            structure_panel = QtWidgets.QWidget()
            structure_panel.setObjectName("Panel")
            structure_layout = QtWidgets.QVBoxLayout(structure_panel)
            structure_layout.setContentsMargins(8, 8, 8, 8)
            structure_layout.setSpacing(8)
            self.portfolio_structure_table = QtWidgets.QTableWidget(0, 4)
            self.portfolio_structure_table.setAlternatingRowColors(True)
            self.portfolio_structure_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
            self.portfolio_structure_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
            self.portfolio_structure_table.horizontalHeader().setStretchLastSection(True)
            self.portfolio_structure_table.verticalHeader().setVisible(False)
            self.portfolio_structure_table.setObjectName("Panel")
            structure_layout.addWidget(self.portfolio_structure_table, 1)
            self.portfolio_structure_notes = QtWidgets.QPlainTextEdit()
            self.portfolio_structure_notes.setReadOnly(True)
            self.portfolio_structure_notes.setMaximumHeight(180)
            self.portfolio_structure_notes.setObjectName("Panel")
            structure_layout.addWidget(self.portfolio_structure_notes)
            analysis_tabs.addTab(structure_panel, "Portfolio Structure")

        lower_split.setStretchFactor(0, 6)
        lower_split.setStretchFactor(1, 5)
        split.addWidget(table_panel)
        split.setSizes([360, 560])

        close_row = QtWidgets.QHBoxLayout()
        close_row.addStretch(1)
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        close_row.addWidget(close_btn)
        layout.addLayout(close_row)

        self.fold_table.itemSelectionChanged.connect(self._refresh_selected_fold_detail)
        self._populate_fold_table()
        self._refresh_study_summary()
        self._refresh_relationship_actions()
        self._draw_stitched_equity()
        self._draw_drift_overview()
        self._populate_param_history_table()
        if self.is_portfolio_study:
            self._refresh_portfolio_structure()

    @staticmethod
    def _parse_json_text(value) -> dict:
        if not value:
            return {}
        if isinstance(value, dict):
            return dict(value)
        try:
            decoded = json.loads(str(value))
            return decoded if isinstance(decoded, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _format_numeric(value) -> str:
        try:
            numeric = float(value)
        except Exception:
            return "—"
        if not np.isfinite(numeric):
            return "—"
        return f"{numeric:.4f}"

    @staticmethod
    def _format_window(start, end) -> str:
        start_ts = pd.to_datetime(start, utc=True, errors="coerce")
        end_ts = pd.to_datetime(end, utc=True, errors="coerce")
        if pd.isna(start_ts) or pd.isna(end_ts):
            return f"{start} -> {end}"
        return (
            f"{start_ts.tz_convert('America/New_York').strftime('%Y-%m-%d %H:%M')} -> "
            f"{end_ts.tz_convert('America/New_York').strftime('%Y-%m-%d %H:%M')}"
        )

    def _combined_fold_frame(self) -> pd.DataFrame:
        folds = self.folds.copy()
        metrics = self.fold_metrics.copy()
        if folds.empty:
            return folds
        if metrics.empty:
            return folds
        frame = folds.merge(metrics, on=["wf_study_id", "fold_index"], how="left")
        train_frame = frame["train_metrics_json"].apply(self._parse_json_text).apply(pd.Series)
        test_frame = frame["test_metrics_json"].apply(self._parse_json_text).apply(pd.Series)
        degr_frame = frame["degradation_json"].apply(self._parse_json_text).apply(pd.Series)
        drift_frame = frame["param_drift_json"].apply(self._parse_json_text).apply(pd.Series)
        for source_frame, prefix in (
            (train_frame, "train_"),
            (test_frame, "test_"),
            (degr_frame, "degradation_"),
            (drift_frame, "drift_"),
        ):
            for col in source_frame.columns:
                if f"{prefix}{col}" not in frame.columns:
                    frame[f"{prefix}{col}"] = source_frame[col]
        return frame

    def _refresh_study_summary(self) -> None:
        frame = self.combined_folds.copy()
        if frame.empty:
            self.study_summary_label.setText("No fold rows are available for this walk-forward study yet.")
            return
        avg_test_sharpe = pd.to_numeric(frame.get("test_sharpe"), errors="coerce").mean()
        median_test_return = pd.to_numeric(frame.get("test_total_return"), errors="coerce").median()
        worst_test_dd = pd.to_numeric(frame.get("test_max_drawdown"), errors="coerce").min()
        avg_return_delta = pd.to_numeric(frame.get("degradation_total_return_delta"), errors="coerce").mean()
        avg_sharpe_delta = pd.to_numeric(frame.get("degradation_sharpe_delta"), errors="coerce").mean()
        total_switches = int(pd.to_numeric(frame.get("drift_switch_count"), errors="coerce").fillna(0).sum())
        summary_prefix = "Portfolio Fold Summary" if self.is_portfolio_study else "Fold Summary"
        extra = ""
        if self.is_portfolio_study:
            extra = (
                f" | Mode: {'Fixed Strategy Blocks' if self.portfolio_mode == 'strategy_blocks' else 'Shared Strategy'}"
                f" | Assets: {len(self.portfolio_dataset_ids)}"
            )
        related_mc_count = int(len(self._related_monte_carlo_studies()))
        if related_mc_count:
            extra += f" | Related Monte Carlo: {related_mc_count}"
        elif self.source_study_id:
            extra += " | Related Monte Carlo: 0"
        self.study_summary_label.setText(
            f"{summary_prefix}: "
            f"Avg test Sharpe {self._format_numeric(avg_test_sharpe)} | "
            f"Median test return {self._format_numeric(median_test_return)} | "
            f"Worst test max DD {self._format_numeric(worst_test_dd)} | "
            f"Avg return delta {self._format_numeric(avg_return_delta)} | "
            f"Avg Sharpe delta {self._format_numeric(avg_sharpe_delta)} | "
            f"Total parameter switches {total_switches}"
            f"{extra}"
        )

    def _related_monte_carlo_studies(self) -> pd.DataFrame:
        studies = self.catalog.load_monte_carlo_studies()
        if studies.empty:
            return studies
        return studies.loc[studies["source_id"].fillna("") == self.wf_study_id].reset_index(drop=True)

    def _refresh_relationship_actions(self) -> None:
        if hasattr(self, "open_source_optimization_btn"):
            self.open_source_optimization_btn.setEnabled(bool(self.source_study_id))
        if hasattr(self, "open_related_mc_btn"):
            self.open_related_mc_btn.setEnabled(not self._related_monte_carlo_studies().empty)

    def _open_source_optimization_study(self) -> None:
        if not self.source_study_id:
            QtWidgets.QMessageBox.information(
                self,
                "Source Study Unavailable",
                "This walk-forward study does not have a saved source optimization study ID.",
            )
            return
        parent = self.logical_parent()
        if parent is not None and hasattr(parent, "_open_optimization_study"):
            parent._open_optimization_study(self.source_study_id)
            return
        QtWidgets.QMessageBox.information(
            self,
            "Source Study Unavailable",
            f"The source optimization study '{self.source_study_id}' could not be opened from this dialog.",
        )

    def _open_latest_related_monte_carlo(self) -> None:
        related = self._related_monte_carlo_studies()
        if related.empty:
            QtWidgets.QMessageBox.information(
                self,
                "Monte Carlo Unavailable",
                "No Monte Carlo studies are linked to this walk-forward study yet.",
            )
            return
        mc_study_id = str(related.iloc[0].get("mc_study_id", "") or "")
        if not mc_study_id:
            return
        parent = self.logical_parent()
        if parent is not None and hasattr(parent, "_open_monte_carlo_study"):
            parent._open_monte_carlo_study(mc_study_id)
            return
        QtWidgets.QMessageBox.information(
            self,
            "Monte Carlo Unavailable",
            f"The related Monte Carlo study '{mc_study_id}' could not be opened from this dialog.",
        )

    def _portfolio_chain_walk_forward_studies(self) -> pd.DataFrame:
        studies = self.catalog.load_walk_forward_studies()
        if studies.empty:
            return studies
        matches: list[bool] = []
        for _, row in studies.iterrows():
            params_payload = self._parse_json_text(row.get("params_json"))
            matches.append(
                (
                    bool(self.source_study_id)
                    and str(params_payload.get("source_study_id", "") or "") == self.source_study_id
                )
                or (
                    bool(self.source_batch_id)
                    and str(params_payload.get("source_batch_id", "") or "") == self.source_batch_id
                )
                or str(row.get("wf_study_id", "") or "") == self.wf_study_id
            )
        return studies.loc[matches].reset_index(drop=True)

    def _portfolio_chain_monte_carlo_studies(self) -> pd.DataFrame:
        walk_forward = self._portfolio_chain_walk_forward_studies()
        if walk_forward.empty:
            return pd.DataFrame()
        studies = self.catalog.load_monte_carlo_studies()
        if studies.empty:
            return studies
        wf_ids = {str(item) for item in walk_forward["wf_study_id"].tolist() if str(item).strip()}
        return studies.loc[studies["source_id"].fillna("").isin(wf_ids)].reset_index(drop=True)

    def _portfolio_chain_batch_params(self) -> dict:
        if self.portfolio_mode == "strategy_blocks":
            return {
                "strategy_blocks": list(self.params_payload.get("strategy_blocks") or []),
            }
        construction = dict(self.params_payload.get("construction_config") or {})
        return {
            "portfolio_assets": list(self.params_payload.get("portfolio_assets") or []),
            "construction_config": construction,
            "_portfolio_dataset_ids": list(self.portfolio_dataset_ids),
            "_portfolio_allocation_ownership": construction.get("allocation_ownership", ALLOCATION_OWNERSHIP_STRATEGY),
            "_portfolio_ranking_mode": construction.get("ranking_mode", RANKING_MODE_NONE),
            "_portfolio_rebalance_mode": construction.get("rebalance_mode", REBALANCE_MODE_ON_CHANGE),
        }

    def _open_portfolio_validation_chain(self) -> None:
        if not self.is_portfolio_study:
            return
        source_study_row = None
        if self.source_study_id:
            studies = self.catalog.load_optimization_studies()
            match = studies.loc[studies["study_id"].fillna("") == self.source_study_id]
            if not match.empty:
                source_study_row = match.iloc[0].to_dict()
        dlg = PortfolioValidationChainDialog(
            title_context=self.wf_study_id,
            catalog=self.catalog,
            portfolio_mode=self.portfolio_mode,
            dataset_ids=self.portfolio_dataset_ids,
            batch_params=self._portfolio_chain_batch_params(),
            source_study_id=self.source_study_id,
            source_batch_id=self.source_batch_id,
            optimization_study_row=source_study_row,
            candidates=self.catalog.load_optimization_candidates(self.source_study_id or self.source_batch_id),
            walk_forward_studies=self._portfolio_chain_walk_forward_studies(),
            monte_carlo_studies=self._portfolio_chain_monte_carlo_studies(),
            initial_tab="walk_forward",
            parent=self,
        )
        dlg.exec()

    def _populate_fold_table(self) -> None:
        frame = self.combined_folds.copy()
        self.fold_table.setRowCount(len(frame))
        for row_idx, row in frame.reset_index(drop=True).iterrows():
            params_json = self._parse_json_text(row.get("selected_params_json"))
            params_label = ", ".join(f"{key}={value}" for key, value in params_json.items())
            values = [
                str(int(row.get("fold_index", 0) or 0)),
                self._format_window(row.get("train_start"), row.get("train_end")),
                self._format_window(row.get("test_start"), row.get("test_end")),
                params_label,
                self._format_numeric(row.get("train_robust_score", row.get("train_robust_score"))),
                self._format_numeric(row.get("test_total_return")),
                self._format_numeric(row.get("test_sharpe")),
                self._format_numeric(row.get("test_max_drawdown")),
                str(int(row.get("test_trade_count", 0) or 0)),
                str(int(row.get("drift_switch_count", 0) or 0)),
            ]
            for col_idx, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                if col_idx == 0:
                    item.setData(QtCore.Qt.ItemDataRole.UserRole, row.to_dict())
                self.fold_table.setItem(row_idx, col_idx, item)
        if self.fold_table.rowCount() > 0:
            self.fold_table.selectRow(0)

    def _refresh_selected_fold_detail(self) -> None:
        selection_model = self.fold_table.selectionModel()
        if selection_model is None:
            self.detail_notes.clear()
            self._draw_fold_analysis()
            if self.is_portfolio_study:
                self._refresh_portfolio_structure()
            return
        rows = selection_model.selectedRows()
        if not rows:
            self.detail_notes.clear()
            self._draw_fold_analysis()
            if self.is_portfolio_study:
                self._refresh_portfolio_structure()
            return
        item = self.fold_table.item(rows[0].row(), 0)
        if item is None:
            self.detail_notes.clear()
            self._draw_fold_analysis()
            if self.is_portfolio_study:
                self._refresh_portfolio_structure()
            return
        payload = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if not isinstance(payload, dict):
            self.detail_notes.clear()
            self._draw_fold_analysis()
            if self.is_portfolio_study:
                self._refresh_portfolio_structure()
            return
        self.open_train_study_btn.setEnabled(bool(str(payload.get("train_study_id", "") or "").strip()))
        if self.is_portfolio_study:
            self.open_test_report_btn.setEnabled(bool(str(payload.get("test_run_id", "") or "").strip()))
        drift = self._parse_json_text(payload.get("param_drift_json"))
        degradation = self._parse_json_text(payload.get("degradation_json"))
        selected_params = self._parse_json_text(payload.get("selected_params_json"))
        drift_lines: list[str] = []
        for key, detail in sorted((drift.get("params") or {}).items()):
            if not isinstance(detail, dict):
                continue
            changed = bool(detail.get("changed"))
            marker = "*" if changed else "-"
            drift_lines.append(
                f"{marker} {key}: {detail.get('previous')} -> {detail.get('current')}"
                + (
                    f" | abs change {self._format_numeric(detail.get('absolute_change'))}"
                    if detail.get("absolute_change") is not None
                    else ""
                )
            )
        notes = [
            f"Fold {payload.get('fold_index', '')}",
            f"Selected Params: {json.dumps(selected_params, sort_keys=True)}",
            f"Train Study ID: {payload.get('train_study_id', '') or '—'}",
            f"Test Run ID: {payload.get('test_run_id', '') or '—'}",
            "",
            "Degradation",
            f"Return Delta: {self._format_numeric(degradation.get('total_return_delta'))}",
            f"Sharpe Delta: {self._format_numeric(degradation.get('sharpe_delta'))}",
            f"Max DD Delta: {self._format_numeric(degradation.get('max_drawdown_delta'))}",
            "",
            f"Parameter Switches: {int(drift.get('switch_count', 0) or 0)}",
        ]
        if drift_lines:
            notes.extend(["", "Parameter Drift", *drift_lines])
        self.detail_notes.setPlainText("\n".join(notes))
        try:
            selected_fold = int(payload.get("fold_index", 0) or 0)
        except Exception:
            selected_fold = None
        self._draw_fold_analysis(selected_fold)
        if self.is_portfolio_study:
            self._refresh_portfolio_structure(payload)

    def _open_selected_train_study(self) -> None:
        selection_model = self.fold_table.selectionModel()
        if selection_model is None:
            return
        rows = selection_model.selectedRows()
        if not rows:
            QtWidgets.QMessageBox.information(self, "No fold selected", "Select a fold first.")
            return
        item = self.fold_table.item(rows[0].row(), 0)
        if item is None:
            return
        payload = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if not isinstance(payload, dict):
            return
        train_study_id = str(payload.get("train_study_id", "") or "")
        if not train_study_id:
            QtWidgets.QMessageBox.information(
                self,
                "Study unavailable",
                "This fold does not have a saved train-fold optimization study ID.",
            )
            return
        parent = self.logical_parent()
        if parent is not None and hasattr(parent, "_open_optimization_study"):
            parent._open_optimization_study(train_study_id)
            return
        QtWidgets.QMessageBox.information(
            self,
            "Study unavailable",
            f"The train study '{train_study_id}' could not be opened from this dialog.",
        )

    def _open_selected_test_run(self) -> None:
        selection_model = self.fold_table.selectionModel()
        if selection_model is None:
            return
        rows = selection_model.selectedRows()
        if not rows:
            QtWidgets.QMessageBox.information(self, "No fold selected", "Select a fold first.")
            return
        item = self.fold_table.item(rows[0].row(), 0)
        if item is None:
            return
        payload = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if not isinstance(payload, dict):
            return
        test_run_id = str(payload.get("test_run_id", "") or "")
        if not test_run_id:
            QtWidgets.QMessageBox.information(
                self,
                "Run unavailable",
                "This fold does not have a saved out-of-sample test run ID.",
            )
            return
        run_match = next((run for run in self.catalog.load_runs() if run.run_id == test_run_id), None)
        if run_match is None:
            QtWidgets.QMessageBox.information(
                self,
                "Run unavailable",
                f"The test run '{test_run_id}' could not be found in the run catalog.",
            )
            return
        parent = self.logical_parent()
        if parent is not None and hasattr(parent, "_open_run_chart_for_run"):
            parent._open_run_chart_for_run(run_match)
            return
        dlg = RunChartDialog(run_match, self.catalog.db_path, {}, self)
        dlg.exec()

    def _open_selected_test_portfolio_report(self) -> None:
        if not self.is_portfolio_study:
            return
        selection_model = self.fold_table.selectionModel()
        if selection_model is None:
            return
        rows = selection_model.selectedRows()
        if not rows:
            QtWidgets.QMessageBox.information(self, "No fold selected", "Select a fold first.")
            return
        item = self.fold_table.item(rows[0].row(), 0)
        if item is None:
            return
        payload = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if not isinstance(payload, dict):
            return
        test_run_id = str(payload.get("test_run_id", "") or "")
        if not test_run_id:
            QtWidgets.QMessageBox.information(self, "Run unavailable", "This fold does not have a saved out-of-sample test run ID.")
            return
        run_match = next((run for run in self.catalog.load_runs() if run.run_id == test_run_id), None)
        if run_match is None:
            QtWidgets.QMessageBox.information(self, "Run unavailable", f"The test run '{test_run_id}' could not be found in the run catalog.")
            return
        parent = self.logical_parent()
        if parent is None or not hasattr(parent, "_build_portfolio_report_for_run"):
            QtWidgets.QMessageBox.information(
                self,
                "Portfolio Report Unavailable",
                "The parent window cannot build a portfolio report for this fold.",
            )
            return
        try:
            report = parent._build_portfolio_report_for_run(run_match)  # type: ignore[attr-defined]
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Portfolio Report Error", str(exc))
            return
        dlg = PortfolioReportDialog(run_match, report, self)
        dlg.exec()

    def _refresh_portfolio_structure(self, payload: dict | None = None) -> None:
        if not self.is_portfolio_study or not hasattr(self, "portfolio_structure_table"):
            return
        data = dict(payload or {})
        selected_params = self._parse_json_text(data.get("selected_params_json"))
        if not selected_params:
            selected_params = dict(self.params_payload)
        if self.portfolio_mode == "strategy_blocks":
            blocks = list(selected_params.get("strategy_blocks") or self.params_payload.get("strategy_blocks") or [])
            self.portfolio_structure_table.setColumnCount(4)
            self.portfolio_structure_table.setHorizontalHeaderLabels(["Block", "Strategy", "Budget", "Assets"])
            self.portfolio_structure_table.setRowCount(len(blocks))
            note_lines: list[str] = []
            for row_idx, block in enumerate(blocks):
                assets = list(block.get("assets") or [])
                asset_ids = ", ".join(
                    str(asset.get("dataset_id") or "")
                    for asset in assets
                    if str(asset.get("dataset_id") or "").strip()
                ) or "—"
                values = [
                    str(block.get("display_name") or block.get("block_id") or f"Block {row_idx + 1}"),
                    str(block.get("strategy") or block.get("strategy_name") or "—"),
                    str(block.get("budget_weight") if block.get("budget_weight") not in (None, "") else "auto"),
                    asset_ids,
                ]
                for col_idx, value in enumerate(values):
                    self.portfolio_structure_table.setItem(row_idx, col_idx, QtWidgets.QTableWidgetItem(value))
                note_lines.append(f"{values[0]} | Strategy={values[1]} | Budget={values[2]} | Assets={values[3]}")
            self.portfolio_structure_notes.setPlainText(
                "\n".join(note_lines) if note_lines else "No strategy-block structure was stored for this fold."
            )
            return

        target_weights = {
            str(asset.get("dataset_id") or ""): asset.get("target_weight")
            for asset in list(self.params_payload.get("portfolio_assets") or [])
            if str(asset.get("dataset_id") or "").strip()
        }
        dataset_ids = self.portfolio_dataset_ids or [
            str(asset.get("dataset_id") or "")
            for asset in list(self.params_payload.get("portfolio_assets") or [])
            if str(asset.get("dataset_id") or "").strip()
        ]
        self.portfolio_structure_table.setColumnCount(4)
        self.portfolio_structure_table.setHorizontalHeaderLabels(["Dataset", "Target Weight", "Mode", "Window"])
        self.portfolio_structure_table.setRowCount(len(dataset_ids))
        window_label = (
            f"{PortfolioReportDialog._fmt_timestamp(data.get('test_start'))} -> {PortfolioReportDialog._fmt_timestamp(data.get('test_end'))}"
            if data
            else "Study-level portfolio definition"
        )
        for row_idx, dataset_id in enumerate(dataset_ids):
            values = [
                str(dataset_id),
                self._format_numeric(target_weights.get(dataset_id)),
                "Shared Strategy Portfolio",
                window_label,
            ]
            for col_idx, value in enumerate(values):
                self.portfolio_structure_table.setItem(row_idx, col_idx, QtWidgets.QTableWidgetItem(value))
        self.portfolio_structure_notes.setPlainText(
            "Shared-strategy portfolio walk-forward keeps the same underlying asset universe each fold while the selected strategy parameters may change."
        )

    def _draw_fold_analysis(self, selected_fold: int | None = None) -> None:
        self.fold_analysis_figure.clear()
        ax_top = self.fold_analysis_figure.add_subplot(211)
        ax_bottom = self.fold_analysis_figure.add_subplot(212)
        for ax in (ax_top, ax_bottom):
            ax.set_facecolor(PALETTE["bg"])
            ax.grid(alpha=0.15, color=PALETTE["grid"])
            ax.tick_params(axis="x", colors=PALETTE["text"])
            ax.tick_params(axis="y", colors=PALETTE["text"])
            for spine in ax.spines.values():
                spine.set_color(PALETTE["border"])
                spine.set_alpha(0.25)
        self.fold_analysis_figure.set_facecolor(PALETTE["panel"])

        frame = self.combined_folds.copy()
        if frame.empty:
            for ax in (ax_top, ax_bottom):
                ax.text(0.5, 0.5, "No fold data is available yet.", ha="center", va="center", color=PALETTE["muted"])
                ax.set_xticks([])
                ax.set_yticks([])
            self.fold_analysis_canvas.draw_idle()
            return

        ordered = frame.sort_values("fold_index").reset_index(drop=True)
        x = ordered["fold_index"].to_numpy(dtype=int)
        train_scores = pd.to_numeric(ordered.get("train_robust_score"), errors="coerce").to_numpy(dtype=float)
        test_sharpes = pd.to_numeric(ordered.get("test_sharpe"), errors="coerce").to_numpy(dtype=float)
        return_delta = pd.to_numeric(ordered.get("degradation_total_return_delta"), errors="coerce").to_numpy(dtype=float)
        sharpe_delta = pd.to_numeric(ordered.get("degradation_sharpe_delta"), errors="coerce").to_numpy(dtype=float)

        ax_top.plot(x, train_scores, marker="o", color=PALETTE["amber"], linewidth=1.6, label="Train Robust Score")
        ax_top.plot(x, test_sharpes, marker="o", color=PALETTE["blue"], linewidth=1.6, label="Test Sharpe")
        ax_top.set_title("Train Ranking vs Test Quality", color=PALETTE["text"], fontsize=13, pad=8)
        ax_top.legend(loc="upper left")

        bar_width = 0.34
        ax_bottom.bar(x - (bar_width / 2.0), return_delta, width=bar_width, color=PALETTE["green"], alpha=0.85, label="Return Delta")
        ax_bottom.bar(x + (bar_width / 2.0), sharpe_delta, width=bar_width, color=PALETTE["red"], alpha=0.85, label="Sharpe Delta")
        ax_bottom.axhline(0.0, color=PALETTE["border"], alpha=0.35, linewidth=1.0)
        ax_bottom.set_title("Train-To-Test Degradation", color=PALETTE["text"], fontsize=13, pad=8)
        ax_bottom.legend(loc="upper left")
        ax_bottom.set_xlabel("Fold", color=PALETTE["text"])

        if selected_fold is not None:
            for ax in (ax_top, ax_bottom):
                ax.axvline(float(selected_fold), color=PALETTE["border"], linestyle="--", alpha=0.45, linewidth=1.2)

        self.fold_analysis_figure.tight_layout()
        self.fold_analysis_canvas.draw_idle()

    def _draw_drift_overview(self) -> None:
        self.drift_figure.clear()
        ax_left = self.drift_figure.add_subplot(121)
        ax_right = self.drift_figure.add_subplot(122)
        for ax in (ax_left, ax_right):
            ax.set_facecolor(PALETTE["bg"])
            ax.grid(alpha=0.15, color=PALETTE["grid"])
            ax.tick_params(axis="x", colors=PALETTE["text"])
            ax.tick_params(axis="y", colors=PALETTE["text"])
            for spine in ax.spines.values():
                spine.set_color(PALETTE["border"])
                spine.set_alpha(0.25)
        self.drift_figure.set_facecolor(PALETTE["panel"])

        frame = self.combined_folds.copy()
        if frame.empty:
            for ax in (ax_left, ax_right):
                ax.text(0.5, 0.5, "No drift data is available yet.", ha="center", va="center", color=PALETTE["muted"])
                ax.set_xticks([])
                ax.set_yticks([])
            self.drift_canvas.draw_idle()
            return

        ordered = frame.sort_values("fold_index").reset_index(drop=True)
        switch_counts = pd.to_numeric(ordered.get("drift_switch_count"), errors="coerce").fillna(0.0)
        numeric_drift_totals: list[float] = []
        for _, row in ordered.iterrows():
            drift = self._parse_json_text(row.get("param_drift_json"))
            total_abs_change = 0.0
            for detail in (drift.get("params") or {}).values():
                if not isinstance(detail, dict):
                    continue
                abs_change = detail.get("absolute_change")
                if abs_change is None:
                    continue
                try:
                    total_abs_change += abs(float(abs_change))
                except Exception:
                    continue
            numeric_drift_totals.append(total_abs_change)
        x = ordered["fold_index"].to_numpy(dtype=int)
        ax_left.bar(x, switch_counts.to_numpy(dtype=float), color=PALETTE["amber"], alpha=0.9)
        ax_left.set_title("Parameter Switch Count By Fold", color=PALETTE["text"], fontsize=12, pad=8)
        ax_left.set_xlabel("Fold", color=PALETTE["text"])

        ax_right.bar(x, np.asarray(numeric_drift_totals, dtype=float), color=PALETTE["blue"], alpha=0.9)
        ax_right.set_title("Total Numeric Drift By Fold", color=PALETTE["text"], fontsize=12, pad=8)
        ax_right.set_xlabel("Fold", color=PALETTE["text"])

        self.drift_figure.tight_layout()
        self.drift_canvas.draw_idle()

    def _populate_param_history_table(self) -> None:
        frame = self.combined_folds.copy()
        if frame.empty:
            self.param_history_table.setColumnCount(0)
            self.param_history_table.setRowCount(0)
            return
        param_names: list[str] = []
        for _, row in frame.iterrows():
            params_json = self._parse_json_text(row.get("selected_params_json"))
            for key in params_json.keys():
                if key not in param_names:
                    param_names.append(str(key))
        headers = ["Fold", "Train Score", "Switches", *param_names]
        self.param_history_table.setColumnCount(len(headers))
        self.param_history_table.setHorizontalHeaderLabels(headers)
        ordered = frame.sort_values("fold_index").reset_index(drop=True)
        self.param_history_table.setRowCount(len(ordered))
        for row_idx, row in ordered.iterrows():
            params_json = self._parse_json_text(row.get("selected_params_json"))
            values: list[str] = [
                str(int(row.get("fold_index", 0) or 0)),
                self._format_numeric(row.get("train_robust_score")),
                str(int(row.get("drift_switch_count", 0) or 0)),
            ]
            for param_name in param_names:
                values.append(str(params_json.get(param_name, "—")))
            for col_idx, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                if col_idx == 0:
                    item.setData(QtCore.Qt.ItemDataRole.UserRole, row.to_dict())
                self.param_history_table.setItem(row_idx, col_idx, item)

    def _draw_stitched_equity(self) -> None:
        self.wf_figure.clear()
        ax = self.wf_figure.add_subplot(111)
        ax.set_facecolor(PALETTE["bg"])
        self.wf_figure.set_facecolor(PALETTE["panel"])
        series = self._series_from_json(self.study_row.get("stitched_equity_json"))
        if series.empty:
            ax.text(
                0.5,
                0.5,
                "No stitched OOS equity curve is stored for this walk-forward study.",
                ha="center",
                va="center",
                color=PALETTE["muted"],
                fontsize=11,
            )
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            display_index = series.index.tz_convert("America/New_York").tz_localize(None)
            ax.plot(display_index, series.to_numpy(dtype=float), color=PALETTE["blue"], linewidth=1.8)
            ax.set_title("Stitched Out-Of-Sample Equity", color=PALETTE["text"], fontsize=14, pad=10)
            ordered = self.combined_folds.sort_values("fold_index").reset_index(drop=True)
            for _, fold_row in ordered.iterrows():
                test_start = pd.to_datetime(fold_row.get("test_start"), utc=True, errors="coerce")
                test_end = pd.to_datetime(fold_row.get("test_end"), utc=True, errors="coerce")
                fold_index = int(fold_row.get("fold_index", 0) or 0)
                if not pd.isna(test_start):
                    start_local = test_start.tz_convert("America/New_York").tz_localize(None)
                    ax.axvline(start_local, color=PALETTE["border"], alpha=0.12, linewidth=1.0)
                    ax.text(
                        start_local,
                        float(np.nanmax(series.to_numpy(dtype=float))) if len(series) else 0.0,
                        f"F{fold_index}",
                        color=PALETTE["muted"],
                        fontsize=8,
                        rotation=90,
                        va="top",
                        ha="right",
                    )
                if not pd.isna(test_start) and not pd.isna(test_end):
                    end_local = test_end.tz_convert("America/New_York").tz_localize(None)
                    ax.axvspan(start_local, end_local, color=PALETTE["blue"], alpha=0.035)
            ax.grid(alpha=0.15, color=PALETTE["grid"])
            ax.tick_params(axis="x", colors=PALETTE["text"])
            ax.tick_params(axis="y", colors=PALETTE["text"])
            for spine in ax.spines.values():
                spine.set_color(PALETTE["border"])
                spine.set_alpha(0.25)
        self.wf_figure.tight_layout()
        self.wf_canvas.draw_idle()

    @staticmethod
    def _series_from_json(value) -> pd.Series:
        if not value:
            return pd.Series(dtype=float)
        try:
            decoded = json.loads(str(value))
        except Exception:
            return pd.Series(dtype=float)
        if not isinstance(decoded, dict) or not decoded:
            return pd.Series(dtype=float)
        index = pd.to_datetime(list(decoded.keys()), utc=True, errors="coerce")
        values = pd.to_numeric(pd.Series(list(decoded.values())), errors="coerce")
        frame = pd.Series(values.to_numpy(dtype=float), index=index)
        frame = frame.dropna()
        frame.name = "stitched_oos_equity"
        return frame.sort_index()


class MonteCarloSetupDialog(DashboardDialog):
    def __init__(self, study_row: dict, catalog: CatalogReader, parent=None) -> None:
        super().__init__(parent)
        self.study_row = dict(study_row)
        self.catalog = catalog
        self.wf_study_id = str(self.study_row.get("wf_study_id", "") or "")
        self._preview_trade_returns = np.asarray([], dtype=float)
        self._preview_meta: dict[str, object] = {}
        self.setWindowTitle("New Monte Carlo Study")
        self.resize(820, 620)
        self.setObjectName("Panel")
        self.setStyleSheet(
            f"""
            QDialog#Panel {{
                background: {PALETTE['panel']};
                color: {PALETTE['text']};
            }}
            QGroupBox {{
                color: {PALETTE['text']};
                border: 1px solid rgba(231, 238, 252, 0.38);
                border-radius: 10px;
                margin-top: 12px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: {PALETTE['muted']};
                font-weight: 700;
            }}
            """
        )

        layout = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            "Seed a Monte Carlo robustness study from a completed walk-forward study. "
            "Version 1 uses the stitched out-of-sample trade sequence as the default source."
        )
        intro.setObjectName("Sub")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        summary = QtWidgets.QPlainTextEdit()
        summary.setReadOnly(True)
        summary.setMaximumHeight(116)
        summary.setObjectName("Panel")
        summary.setPlainText("\n".join(self._source_summary_lines()))
        layout.addWidget(summary)

        form_box = QtWidgets.QGroupBox("Monte Carlo Setup")
        form = QtWidgets.QFormLayout(form_box)
        form.setContentsMargins(12, 12, 12, 12)
        form.setSpacing(10)

        self.resampling_combo = QtWidgets.QComboBox()
        self.resampling_combo.addItem("Trade Bootstrap", MONTE_CARLO_MODE_BOOTSTRAP)
        self.resampling_combo.addItem("Trade Reshuffle", MONTE_CARLO_MODE_RESHUFFLE)
        form.addRow("Resampling Mode", self.resampling_combo)

        self.simulation_spin = QtWidgets.QSpinBox()
        self.simulation_spin.setRange(10, 100000)
        self.simulation_spin.setValue(500)
        form.addRow("Simulation Count", self.simulation_spin)

        self.seed_spin = QtWidgets.QSpinBox()
        self.seed_spin.setRange(0, 2_147_483_647)
        self.seed_spin.setValue(7)
        form.addRow("Random Seed", self.seed_spin)

        self.cost_drag_spin = QtWidgets.QDoubleSpinBox()
        self.cost_drag_spin.setDecimals(2)
        self.cost_drag_spin.setRange(0.0, 1000.0)
        self.cost_drag_spin.setSingleStep(1.0)
        self.cost_drag_spin.setValue(0.0)
        self.cost_drag_spin.setToolTip("Optional fixed return drag in basis points applied to each resampled trade return.")
        form.addRow("Return Drag (bps)", self.cost_drag_spin)

        self.description_edit = QtWidgets.QPlainTextEdit()
        self.description_edit.setMaximumHeight(96)
        self.description_edit.setPlaceholderText("Optional notes for this Monte Carlo study.")
        form.addRow("Description", self.description_edit)

        self.preview_label = QtWidgets.QLabel(self._preview_guardrail_text())
        self.preview_label.setObjectName("Sub")
        self.preview_label.setWordWrap(True)
        form.addRow("Method Note", self.preview_label)
        layout.addWidget(form_box, 1)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _source_preview_text(self) -> str:
        try:
            trade_returns, meta = extract_walk_forward_trade_returns(ResultCatalog(self.catalog.db_path), self.wf_study_id)
            self._preview_trade_returns = np.asarray(trade_returns, dtype=float)
            self._preview_meta = dict(meta)
            source_type = self._source_type_label(meta)
            assets = ", ".join(str(item) for item in list(meta.get("portfolio_dataset_ids") or []) if str(item).strip())
            preview = (
                f"Source Trade Count: {len(trade_returns)} | "
                f"Starting Equity: {float(meta.get('starting_equity', 0.0)):.2f} | "
                f"Unit Mode: {meta.get('unit_mode', '—')}"
            )
            if bool(meta.get("is_portfolio", False)):
                preview += f" | Source Type: {source_type}"
                if assets:
                    preview += f" | Assets: {assets}"
            return preview
        except Exception as exc:
            return f"Source Preview: unavailable ({exc})"

    def _source_summary_lines(self) -> list[str]:
        preview_line = self._source_preview_text()
        lines = [
            f"Walk-Forward Study: {self.wf_study_id}",
            f"Strategy: {self.study_row.get('strategy', '')}",
            f"Dataset: {self.study_row.get('dataset_id', '')}",
            f"Timeframe: {self.study_row.get('timeframe', '')}",
        ]
        if self._preview_meta.get("is_portfolio", False):
            lines.append(f"Source Type: {self._source_type_label(self._preview_meta)}")
            assets = ", ".join(
                str(item)
                for item in list(self._preview_meta.get("portfolio_dataset_ids") or [])
                if str(item).strip()
            )
            if assets:
                lines.append(f"Underlying Assets: {assets}")
        lines.append(preview_line)
        return lines

    @staticmethod
    def _source_type_label(meta: dict[str, object]) -> str:
        if bool(meta.get("is_portfolio", False)):
            portfolio_mode = str(meta.get("portfolio_mode", "") or "")
            if portfolio_mode == "strategy_blocks":
                return "Fixed Strategy-Block Portfolio"
            return "Shared Strategy Portfolio"
        return "Single Strategy"

    def _preview_guardrail_text(self) -> str:
        trade_count = int(len(self._preview_trade_returns))
        prefix = "Trade bootstrap is the default distribution view. Trade reshuffle is useful as a sequence-risk companion."
        if trade_count <= 0:
            return prefix
        notes: list[str] = [prefix]
        if trade_count < 10:
            notes.append(
                f"This source only has {trade_count} completed out-of-sample trade cycles. Monte Carlo will still run, but the distribution will be very thin."
            )
        elif trade_count < 25:
            notes.append(
                f"This source has {trade_count} completed out-of-sample trade cycles. Treat the resulting distribution as an early stress test rather than a mature robustness estimate."
            )
        if self._preview_meta.get("is_portfolio", False):
            notes.append(
                "For portfolio sources, Monte Carlo v1 resamples the stitched out-of-sample portfolio trade sequence rather than rebuilding asset-level portfolio paths."
            )
        return "\n\n".join(notes)

    def settings(self) -> dict[str, object]:
        return {
            "resampling_mode": str(self.resampling_combo.currentData() or MONTE_CARLO_MODE_BOOTSTRAP),
            "simulation_count": int(self.simulation_spin.value()),
            "seed": int(self.seed_spin.value()),
            "cost_stress_bps": float(self.cost_drag_spin.value()),
            "description": self.description_edit.toPlainText().strip(),
        }

    def accept(self) -> None:
        trade_count = int(len(self._preview_trade_returns))
        if 0 < trade_count < 10:
            box = QtWidgets.QMessageBox(self)
            box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
            box.setWindowTitle("Thin Monte Carlo Source")
            box.setText(
                f"This walk-forward source only has {trade_count} completed trade cycles. "
                "Monte Carlo will run, but the distribution may be unstable."
            )
            box.setInformativeText("Choose Continue to run anyway, or Cancel to revisit the source study first.")
            continue_btn = box.addButton("Continue", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
            cancel_btn = box.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
            box.setDefaultButton(cancel_btn)
            box.exec()
            if box.clickedButton() is not continue_btn:
                return
        super().accept()


class MonteCarloStudyDialog(DashboardDialog):
    def __init__(self, study_row: dict, catalog: CatalogReader, parent=None) -> None:
        super().__init__(parent)
        self.study_row = dict(study_row)
        self.catalog = catalog
        self.mc_study_id = str(self.study_row.get("mc_study_id", "") or "")
        self.path_rows = self.catalog.load_monte_carlo_paths(self.mc_study_id)
        self.setWindowTitle(f"Monte Carlo Study | {self.mc_study_id}")
        self.resize(1620, 1020)
        self.setObjectName("Panel")
        self.setStyleSheet(
            f"""
            QDialog#Panel {{
                background: {PALETTE['panel']};
                color: {PALETTE['text']};
            }}
            """
        )

        layout = QtWidgets.QVBoxLayout(self)
        summary = self._parse_json_dict(self.study_row.get("summary_json"))
        summary_box = QtWidgets.QPlainTextEdit()
        summary_box.setReadOnly(True)
        summary_box.setMaximumHeight(120)
        summary_box.setObjectName("Panel")
        summary_lines = [
            f"Study: {self.mc_study_id}",
            f"Source: {self.study_row.get('source_type', '')} | {self.study_row.get('source_id', '')}",
            f"Mode: {self.study_row.get('resampling_mode', '')}",
            f"Simulations: {int(self.study_row.get('simulation_count', 0) or 0)} | Seed: {self.study_row.get('seed', '—')}",
            f"Median Return: {self._fmt(summary.get('terminal_return_p50'))} | "
            f"P05 Return: {self._fmt(summary.get('terminal_return_p05'))} | "
            f"P95 Return: {self._fmt(summary.get('terminal_return_p95'))}",
            f"Median Max DD: {self._fmt(summary.get('max_drawdown_p50'))} | "
            f"P95 Max DD: {self._fmt(summary.get('max_drawdown_p95'))} | "
            f"Loss Probability: {self._fmt_pct(summary.get('loss_probability'))}",
        ]
        if bool(summary.get("source_is_portfolio", False)):
            source_type = "Fixed Strategy-Block Portfolio" if str(summary.get("source_portfolio_mode", "") or "") == "strategy_blocks" else "Shared Strategy Portfolio"
            summary_lines.append(f"Source Type: {source_type}")
            assets = ", ".join(str(item) for item in list(summary.get("source_portfolio_assets") or []) if str(item).strip())
            if assets:
                summary_lines.append(f"Underlying Assets: {assets}")
        if str(summary.get("source_study_id", "") or "").strip():
            summary_lines.append(f"Source Optimization Study: {summary.get('source_study_id')}")
        if str(summary.get("source_batch_id", "") or "").strip():
            summary_lines.append(f"Source Portfolio Batch: {summary.get('source_batch_id')}")
        summary_box.setPlainText("\n".join(summary_lines))
        layout.addWidget(summary_box)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        split.setChildrenCollapsible(False)
        layout.addWidget(split, 1)

        fan_panel = QtWidgets.QWidget()
        fan_panel.setObjectName("Panel")
        fan_layout = QtWidgets.QVBoxLayout(fan_panel)
        fan_layout.setContentsMargins(8, 8, 8, 8)
        fan_title = QtWidgets.QLabel("Equity Fan Chart")
        fan_title.setObjectName("Title")
        fan_layout.addWidget(fan_title)
        self.mc_figure = Figure(figsize=(13.5, 5.4), tight_layout=True, facecolor=PALETTE["panel"])
        self.mc_canvas = FigureCanvasQTAgg(self.mc_figure)
        fan_layout.addWidget(self.mc_canvas, 1)
        split.addWidget(fan_panel)

        lower = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        lower.setChildrenCollapsible(False)
        split.addWidget(lower)

        hist_panel = QtWidgets.QWidget()
        hist_panel.setObjectName("Panel")
        hist_layout = QtWidgets.QVBoxLayout(hist_panel)
        hist_layout.setContentsMargins(8, 8, 8, 8)
        hist_layout.setSpacing(8)
        hist_title = QtWidgets.QLabel("Distribution Views")
        hist_title.setObjectName("Title")
        hist_layout.addWidget(hist_title)
        self.hist_figure = Figure(figsize=(8.0, 4.5), tight_layout=True, facecolor=PALETTE["panel"])
        self.hist_canvas = FigureCanvasQTAgg(self.hist_figure)
        hist_layout.addWidget(self.hist_canvas, 1)
        lower.addWidget(hist_panel)

        side_panel = QtWidgets.QWidget()
        side_panel.setObjectName("Panel")
        side_layout = QtWidgets.QVBoxLayout(side_panel)
        side_layout.setContentsMargins(8, 8, 8, 8)
        side_layout.setSpacing(8)
        threshold_title = QtWidgets.QLabel("Risk Thresholds")
        threshold_title.setObjectName("Title")
        side_layout.addWidget(threshold_title)
        self.threshold_table = QtWidgets.QTableWidget(0, 2)
        self.threshold_table.setHorizontalHeaderLabels(["Threshold", "Probability"])
        self.threshold_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.threshold_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.threshold_table.setAlternatingRowColors(True)
        self.threshold_table.horizontalHeader().setStretchLastSection(True)
        self.threshold_table.verticalHeader().setVisible(False)
        self.threshold_table.setObjectName("Panel")
        side_layout.addWidget(self.threshold_table, 1)
        path_title = QtWidgets.QLabel("Representative Paths")
        path_title.setObjectName("Title")
        side_layout.addWidget(path_title)
        self.path_table = QtWidgets.QTableWidget(0, 4)
        self.path_table.setHorizontalHeaderLabels(["Path Type", "Terminal Return", "Max DD", "Path ID"])
        self.path_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.path_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.path_table.setAlternatingRowColors(True)
        self.path_table.horizontalHeader().setStretchLastSection(True)
        self.path_table.verticalHeader().setVisible(False)
        self.path_table.setObjectName("Panel")
        side_layout.addWidget(self.path_table, 1)
        lower.addWidget(side_panel)

        split.setSizes([430, 420])
        lower.setSizes([980, 500])

        action_row = QtWidgets.QHBoxLayout()
        self.open_source_walk_forward_btn = QtWidgets.QPushButton("Open Source Walk-Forward Study")
        self.open_source_walk_forward_btn.clicked.connect(self._open_source_walk_forward_study)
        self.open_source_optimization_btn = QtWidgets.QPushButton("Open Source Optimization Study")
        self.open_source_optimization_btn.clicked.connect(self._open_source_optimization_study)
        action_row.addWidget(self.open_source_walk_forward_btn)
        action_row.addWidget(self.open_source_optimization_btn)
        action_row.addStretch(1)
        layout.addLayout(action_row)

        close_row = QtWidgets.QHBoxLayout()
        close_row.addStretch(1)
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        close_row.addWidget(close_btn)
        layout.addLayout(close_row)

        self._populate_threshold_table()
        self._populate_path_table()
        self._draw_fan_chart()
        self._draw_histograms()
        self.open_source_optimization_btn.setEnabled(bool(self._source_optimization_study_id()))

    @staticmethod
    def _parse_json_dict(value) -> dict:
        if not value:
            return {}
        if isinstance(value, dict):
            return dict(value)
        try:
            decoded = json.loads(str(value))
            return decoded if isinstance(decoded, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _parse_json_list(value) -> list[float]:
        if not value:
            return []
        if isinstance(value, list):
            return [float(item) for item in value]
        try:
            decoded = json.loads(str(value))
        except Exception:
            return []
        if not isinstance(decoded, list):
            return []
        result: list[float] = []
        for item in decoded:
            try:
                result.append(float(item))
            except Exception:
                continue
        return result

    @staticmethod
    def _finite_numeric_array(value) -> np.ndarray:
        raw = np.asarray(MonteCarloStudyDialog._parse_json_list(value), dtype=float)
        if raw.size == 0:
            return raw
        return raw[np.isfinite(raw)]

    @staticmethod
    def _fmt(value, precision: int = 4) -> str:
        try:
            numeric = float(value)
        except Exception:
            return "—"
        if not np.isfinite(numeric):
            return "—"
        return f"{numeric:.{precision}f}"

    @staticmethod
    def _fmt_pct(value) -> str:
        try:
            numeric = float(value)
        except Exception:
            return "—"
        if not np.isfinite(numeric):
            return "—"
        return f"{numeric * 100:.1f}%"

    def _populate_threshold_table(self) -> None:
        summary = self._parse_json_dict(self.study_row.get("summary_json"))
        rows: list[tuple[str, str]] = []
        for key, value in dict(summary.get("drawdown_thresholds") or {}).items():
            rows.append((f"Drawdown >= {key}", self._fmt_pct(value)))
        for key, value in dict(summary.get("return_thresholds") or {}).items():
            rows.append((f"Return threshold {key}", self._fmt_pct(value)))
        self.threshold_table.setRowCount(len(rows))
        for row_idx, (label, value) in enumerate(rows):
            self.threshold_table.setItem(row_idx, 0, QtWidgets.QTableWidgetItem(label))
            self.threshold_table.setItem(row_idx, 1, QtWidgets.QTableWidgetItem(value))

    def _populate_path_table(self) -> None:
        self.path_table.setRowCount(len(self.path_rows))
        for row_idx, (_, row) in enumerate(self.path_rows.iterrows()):
            summary = self._parse_json_dict(row.get("summary_json"))
            values = [
                str(row.get("path_type", "")),
                self._fmt(summary.get("terminal_return")),
                self._fmt(summary.get("max_drawdown")),
                str(row.get("path_id", "")),
            ]
            for col_idx, value in enumerate(values):
                self.path_table.setItem(row_idx, col_idx, QtWidgets.QTableWidgetItem(value))

    def _draw_fan_chart(self) -> None:
        self.mc_figure.clear()
        ax = self.mc_figure.add_subplot(111)
        ax.set_facecolor(PALETTE["bg"])
        self.mc_figure.set_facecolor(PALETTE["panel"])
        fan = self._parse_json_dict(self.study_row.get("fan_quantiles_json"))
        original_path = np.asarray(self._parse_json_list(self.study_row.get("original_path_json")), dtype=float)
        if not fan:
            ax.text(0.5, 0.5, "No fan chart data is stored for this Monte Carlo study.", ha="center", va="center", color=PALETTE["muted"])
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            x = np.arange(len(next(iter(fan.values()))), dtype=int)
            p05 = np.asarray(fan.get("p05", []), dtype=float)
            p25 = np.asarray(fan.get("p25", []), dtype=float)
            p50 = np.asarray(fan.get("p50", []), dtype=float)
            p75 = np.asarray(fan.get("p75", []), dtype=float)
            p95 = np.asarray(fan.get("p95", []), dtype=float)
            if len(p05) == len(x):
                ax.fill_between(x, p05, p95, color=PALETTE["blue"], alpha=0.12, label="5-95% band")
            if len(p25) == len(x):
                ax.fill_between(x, p25, p75, color=PALETTE["blue"], alpha=0.24, label="25-75% band")
            if len(p50) == len(x):
                ax.plot(x, p50, color=PALETTE["amber"], linewidth=1.7, label="Median path")
            if len(original_path) == len(x):
                ax.plot(x, original_path, color=PALETTE["green"], linewidth=1.5, label="Original validated path")
            ax.set_title("Monte Carlo Equity Fan", color=PALETTE["text"], fontsize=14, pad=10)
            ax.set_xlabel("Trade Number", color=PALETTE["text"])
            ax.tick_params(axis="x", colors=PALETTE["text"])
            ax.tick_params(axis="y", colors=PALETTE["text"])
            ax.grid(alpha=0.15, color=PALETTE["grid"])
            ax.legend(loc="upper left")
            for spine in ax.spines.values():
                spine.set_color(PALETTE["border"])
                spine.set_alpha(0.25)
        self.mc_figure.tight_layout()
        self.mc_canvas.draw_idle()

    def _draw_histograms(self) -> None:
        self.hist_figure.clear()
        ax_left = self.hist_figure.add_subplot(121)
        ax_right = self.hist_figure.add_subplot(122)
        for ax in (ax_left, ax_right):
            ax.set_facecolor(PALETTE["bg"])
            ax.grid(alpha=0.15, color=PALETTE["grid"])
            ax.tick_params(axis="x", colors=PALETTE["text"])
            ax.tick_params(axis="y", colors=PALETTE["text"])
            for spine in ax.spines.values():
                spine.set_color(PALETTE["border"])
                spine.set_alpha(0.25)
        self.hist_figure.set_facecolor(PALETTE["panel"])
        terminal_returns = self._finite_numeric_array(self.study_row.get("terminal_returns_json"))
        max_drawdowns = self._finite_numeric_array(self.study_row.get("max_drawdowns_json"))
        self._draw_distribution_panel(
            ax_left,
            terminal_returns,
            title="Terminal Return Distribution",
            color=PALETTE["blue"],
            empty_message="No return distribution available.",
        )
        self._draw_distribution_panel(
            ax_right,
            max_drawdowns,
            title="Max Drawdown Distribution",
            color=PALETTE["red"],
            empty_message="No drawdown distribution available.",
        )
        self.hist_figure.tight_layout()
        self.hist_canvas.draw_idle()

    def _draw_distribution_panel(
        self,
        ax,
        values: np.ndarray,
        *,
        title: str,
        color: str,
        empty_message: str,
    ) -> None:
        ax.set_title(title, color=PALETTE["text"], fontsize=12, pad=8)
        if values.size == 0:
            ax.text(0.5, 0.5, empty_message, ha="center", va="center", color=PALETTE["muted"])
            return
        value_min = float(np.min(values))
        value_max = float(np.max(values))
        same_value = np.isclose(
            value_min,
            value_max,
            rtol=1e-12,
            atol=max(1e-12, abs(value_min) * 1e-12),
        )
        if same_value:
            ax.bar([0], [int(values.size)], width=0.6, color=color, alpha=0.85)
            ax.set_xticks([0])
            ax.set_xticklabels([f"{value_min:.6g}"], color=PALETTE["text"])
            ax.set_ylabel("Count", color=PALETTE["text"])
            ax.text(
                0,
                float(values.size),
                f"n={values.size}",
                ha="center",
                va="bottom",
                color=PALETTE["text"],
                fontsize=10,
            )
            return
        bins = min(40, max(12, int(np.sqrt(values.size))))
        ax.hist(values, bins=bins, color=color, alpha=0.85)

    def _source_optimization_study_id(self) -> str:
        summary = self._parse_json_dict(self.study_row.get("summary_json"))
        return str(summary.get("source_study_id", "") or "")

    def _open_source_walk_forward_study(self) -> None:
        source_id = str(self.study_row.get("source_id", "") or "")
        if not source_id:
            QtWidgets.QMessageBox.information(
                self,
                "Source Unavailable",
                "This Monte Carlo study does not have a saved source walk-forward study ID.",
            )
            return
        parent = self.logical_parent()
        if parent is not None and hasattr(parent, "_open_walk_forward_study"):
            parent._open_walk_forward_study(source_id)
            return
        QtWidgets.QMessageBox.information(
            self,
            "Source Unavailable",
            f"The source walk-forward study '{source_id}' could not be opened from this dialog.",
        )

    def _open_source_optimization_study(self) -> None:
        study_id = self._source_optimization_study_id()
        if not study_id:
            QtWidgets.QMessageBox.information(
                self,
                "Source Study Unavailable",
                "This Monte Carlo study does not have a saved upstream optimization study ID.",
            )
            return
        parent = self.logical_parent()
        if parent is not None and hasattr(parent, "_open_optimization_study"):
            parent._open_optimization_study(study_id)
            return
        QtWidgets.QMessageBox.information(
            self,
            "Source Study Unavailable",
            f"The source optimization study '{study_id}' could not be opened from this dialog.",
        )


# --- UI ---------------------------------------------------------------------
class DashboardWindow(QtWidgets.QMainWindow):
    def __init__(self, catalog_path: Path) -> None:
        super().__init__()
        self.setWindowTitle("Backtest Dashboard")
        self.setMinimumSize(1400, 900)
        icon_path = Path("assets/app_icon.png")
        if icon_path.exists():
            self.setWindowIcon(QtGui.QIcon(str(icon_path)))

        self.catalog = CatalogReader(catalog_path)
        self.catalog.ensure_catalog()
        self.worker: GridWorker | None = None
        self.nasdaq_symbols: list[str] = []
        self.selected_tickers: list[str] = []
        self.select_all_tickers = False
        self.download_queue: list[str] = []
        self.download_proc: QtCore.QProcess | None = None
        self.download_procs: list[QtCore.QProcess] = []
        self.download_paused = False
        self.download_active_ticker: str | None = None
        self.download_progress_rows: dict[str, dict] = {}
        self.download_proc_meta: dict[int, dict] = {}
        self.current_acquisition_run_id: str | None = None
        self.current_acquisition_attempts: dict[str, str] = {}
        self.scheduled_tasks: list[dict] = []
        self.universes: list[dict] = []
        self.study_universe_id: str = ""
        self.download_universe_id: str = ""
        self._editing_universe_id: str = ""
        self._editing_universe_dataset_ids: list[str] = []
        self.magellan = MagellanClient(self)
        self.snapshot_exporter = ChartSnapshotExporter()
        self._closing = False
        self._current_grid_started_at: float | None = None
        self._current_walk_forward_started_at: float | None = None
        self._current_monte_carlo_started_at: float | None = None
        self._batch_benchmark_cache: dict[str, tuple[BatchExecutionBenchmark, ...]] = {}
        self.study_dataset_ids: list[str] = []
        self.portfolio_target_weights: dict[str, float] = {}
        self.portfolio_strategy_blocks: list[dict] = []
        self.walk_forward_worker: QtCore.QThread | None = None
        self.monte_carlo_worker: MonteCarloWorker | None = None
        self._magellan_warm_timer = QtCore.QTimer(self)
        self._magellan_warm_timer.setSingleShot(True)
        self._magellan_warm_timer.timeout.connect(self._warm_magellan)

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        layout.addWidget(self._build_header())

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._build_home_tab(), "Home")
        self.tabs.addTab(self._build_universe_tab(), "Universe Builder")
        self.tabs.addTab(self._build_optimization_tab(), "Optimization")
        self.tabs.addTab(self._build_walk_forward_tab(), "Walk Forward")
        self.tabs.addTab(self._build_monte_carlo_tab(), "Monte Carlo")
        self.tabs.addTab(self._build_heatmap_tab(), "Heatmaps")
        self.tabs.addTab(self._build_control_panel(), "Orchestrate")
        self.tabs.addTab(self._build_automate_tab(), "Automate")

        layout.addWidget(self.tabs)

        self.setCentralWidget(central)
        self.refresh_timer = QtCore.QTimer(self)
        self.refresh_timer.setInterval(2000)
        self.refresh_timer.timeout.connect(self._refresh_batches_live)
        self.refresh_timer.start()
        self._load_universes()
        self.refresh()
        self._magellan_warm_timer.start(0)

    # -- sections ------------------------------------------------------------
    def _build_header(self) -> QtWidgets.QWidget:
        box = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(box)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(12)

        brand = QtWidgets.QLabel("Backtest Dashboard")
        brand.setObjectName("Title")
        dot = QtWidgets.QLabel()
        dot.setFixedSize(12, 12)
        dot.setStyleSheet(f"background:{PALETTE['green']}; border-radius:6px;")

        left = QtWidgets.QHBoxLayout()
        left.setSpacing(8)
        left.addWidget(dot)
        left.addWidget(brand)

        left_box = QtWidgets.QWidget()
        left_box.setLayout(left)

        self.status_label = QtWidgets.QLabel("DB: …")
        self.status_label.setObjectName("Sub")

        refresh_btn = QtWidgets.QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh)

        h.addWidget(left_box, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
        h.addStretch(1)
        h.addWidget(self.status_label)
        h.addWidget(refresh_btn)
        return box

    def _build_control_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        panel.setObjectName("Panel")
        outer = QtWidgets.QVBoxLayout(panel)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(12)

        title = QtWidgets.QLabel("Orchestrate Backtests")
        title.setObjectName("Title")
        subtitle = QtWidgets.QLabel("Load CSV → DuckDB → run grid/backtests; DB created if missing.")
        subtitle.setObjectName("Sub")
        outer.addWidget(title)
        outer.addWidget(subtitle)

        # Two-column area: left form, right params.
        split = QtWidgets.QHBoxLayout()
        split.setSpacing(16)

        # Left column tabs (Setup + Backtest Settings) without QTabWidget pane.
        left_column = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_column)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        left_tabs = QtWidgets.QTabBar()
        left_tabs.setExpanding(False)
        left_tabs.setElideMode(QtCore.Qt.TextElideMode.ElideRight)
        left_tabs.setDrawBase(False)
        left_tabs.setStyleSheet(
            f"""
            QTabBar::tab {{
                color: {PALETTE['text']};
                background: {PALETTE['panel']};
                border: 1px solid {PALETTE['border']};
                border-bottom: none;
                border-radius: 8px 8px 0 0;
                padding: 6px 12px;
                min-width: 90px;
                margin-right: 6px;
            }}
            QTabBar::tab:selected {{
                background: {PALETTE['panel2']};
                border-color: {PALETTE['blue']};
                font-weight: 700;
            }}
            """
        )

        main_scroll = QtWidgets.QScrollArea()
        main_scroll.setWidgetResizable(True)
        main_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        main_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        main_tab = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout(main_tab)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(12)

        def add_setup_section(title_text: str, detail_text: str | None = None) -> None:
            title_label = QtWidgets.QLabel(title_text)
            title_label.setStyleSheet("font-weight: 700; font-size: 13px; padding-top: 4px;")
            main_layout.addWidget(title_label)
            if detail_text:
                detail_label = QtWidgets.QLabel(detail_text)
                detail_label.setObjectName("Sub")
                detail_label.setWordWrap(True)
                main_layout.addWidget(detail_label)

        self.strategy_combo = QtWidgets.QComboBox()
        self.execution_mode_combo = QtWidgets.QComboBox()
        self.execution_mode_combo.addItem("Auto (Recommended)", ExecutionMode.AUTO.value)
        self.execution_mode_combo.addItem("Reference", ExecutionMode.REFERENCE.value)
        self.execution_mode_combo.addItem("Vectorized", ExecutionMode.VECTORIZED.value)
        self.execution_mode_combo.setToolTip(
            "Choose which execution backend to use for new runs.\n"
            "Auto selects vectorized when supported and falls back to reference.\n"
            "On supported higher-timeframe studies, Auto may choose same-timeframe resampled vectorized execution.\n"
            "Higher-timeframe vectorized execution currently uses same-timeframe resampled bars; "
            "full base-execution parity is still future work."
        )
        self.study_mode_combo = QtWidgets.QComboBox()
        self.study_mode_combo.addItem("Independent Assets", STUDY_MODE_INDEPENDENT)
        self.study_mode_combo.addItem("Portfolio (Vectorized v1)", STUDY_MODE_PORTFOLIO)
        self.study_mode_combo.setToolTip(
            "Independent Assets runs the same backtest separately on each selected dataset.\n"
            "Portfolio (Vectorized v1) uses shared cash across the selected datasets with the vectorized portfolio backend.\n"
            "Portfolio mode is currently same-timeframe only, requires a vectorized-supported strategy, and still does not have a reference-engine fallback."
        )
        self.portfolio_allocation_combo = QtWidgets.QComboBox()
        self.portfolio_allocation_combo.addItem("Equal Weight", PORTFOLIO_ALLOC_EQUAL)
        self.portfolio_allocation_combo.addItem("Relative Weights", PORTFOLIO_ALLOC_RELATIVE)
        self.portfolio_allocation_combo.addItem("Fixed Weights", PORTFOLIO_ALLOC_FIXED)
        self.portfolio_allocation_combo.setToolTip(
            "Equal Weight splits capital evenly across active assets.\n"
            "Relative Weights normalizes your weights across the selected assets.\n"
            "Fixed Weights uses your weights as absolute targets and then caps gross exposure if needed."
        )
        add_setup_section(
            "Core Setup",
            "Choose the strategy, execution backend, and overall study mode before configuring datasets and portfolio rules.",
        )
        main_layout.addWidget(QtWidgets.QLabel("Strategy"))
        main_layout.addWidget(self.strategy_combo)
        main_layout.addWidget(QtWidgets.QLabel("Execution Mode"))
        main_layout.addWidget(self.execution_mode_combo)
        main_layout.addWidget(QtWidgets.QLabel("Study Mode"))
        main_layout.addWidget(self.study_mode_combo)

        add_setup_section(
            "Data And Study Scope",
            "Add data, choose the working dataset, and optionally expand the study across multiple datasets.",
        )
        main_layout.addWidget(QtWidgets.QLabel("CSV Path"))
        csv_row = QtWidgets.QHBoxLayout()
        self.csv_path_edit = QtWidgets.QLineEdit("AAPL.USUSD_Candlestick_1_M_BID_12.01.2026-17.01.2026.csv")
        browse = QtWidgets.QPushButton("Browse")
        browse.clicked.connect(self._browse_csv)
        csv_row.addWidget(self.csv_path_edit, 3)
        csv_row.addWidget(browse, 1)
        main_layout.addLayout(csv_row)
        add_btn = QtWidgets.QPushButton("Add CSV to Database")
        add_btn.clicked.connect(self._add_csv_clicked)
        main_layout.addWidget(add_btn)

        main_layout.addWidget(QtWidgets.QLabel("Dataset ID"))
        dataset_id_row = QtWidgets.QHBoxLayout()
        self.dataset_combo = QtWidgets.QComboBox()
        self.dataset_combo.setEditable(True)
        self.dataset_combo.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
        self.dataset_combo.setMinimumContentsLength(12)
        self.dataset_combo.currentTextChanged.connect(lambda _: self._update_study_dataset_summary())
        current_dataset_acq_btn = QtWidgets.QPushButton("View Acquisition")
        current_dataset_acq_btn.clicked.connect(self._open_current_dataset_acquisition_detail)
        dataset_id_row.addWidget(self.dataset_combo, 3)
        dataset_id_row.addWidget(current_dataset_acq_btn, 1)
        main_layout.addLayout(dataset_id_row)
        main_layout.addWidget(QtWidgets.QLabel("Study Datasets"))
        dataset_row = QtWidgets.QHBoxLayout()
        self.study_dataset_summary = QtWidgets.QLineEdit()
        self.study_dataset_summary.setReadOnly(True)
        self.study_dataset_summary.setPlaceholderText("Current dataset only")
        choose_datasets_btn = QtWidgets.QPushButton("Choose Datasets")
        choose_datasets_btn.clicked.connect(self._choose_study_datasets)
        current_only_btn = QtWidgets.QPushButton("Current Only")
        current_only_btn.clicked.connect(self._reset_study_datasets_to_current)
        study_datasets_acq_btn = QtWidgets.QPushButton("Dataset Detail")
        study_datasets_acq_btn.clicked.connect(self._open_selected_study_dataset_acquisition_detail)
        dataset_row.addWidget(self.study_dataset_summary, 3)
        dataset_row.addWidget(choose_datasets_btn, 1)
        dataset_row.addWidget(current_only_btn, 1)
        dataset_row.addWidget(study_datasets_acq_btn, 1)
        main_layout.addLayout(dataset_row)
        main_layout.addWidget(QtWidgets.QLabel("Study Universe"))
        universe_row = QtWidgets.QHBoxLayout()
        self.study_universe_combo = QtWidgets.QComboBox()
        self.study_universe_combo.currentIndexChanged.connect(self._on_study_universe_changed)
        open_universe_btn = QtWidgets.QPushButton("Open Builder")
        open_universe_btn.clicked.connect(self._open_universe_builder_tab)
        view_acq_btn = QtWidgets.QPushButton("View Acquisition")
        view_acq_btn.clicked.connect(self._open_study_universe_acquisition_catalog)
        clear_universe_btn = QtWidgets.QPushButton("Manual")
        clear_universe_btn.clicked.connect(lambda: self.study_universe_combo.setCurrentIndex(0))
        universe_row.addWidget(self.study_universe_combo, 3)
        universe_row.addWidget(open_universe_btn, 1)
        universe_row.addWidget(view_acq_btn, 1)
        universe_row.addWidget(clear_universe_btn, 1)
        main_layout.addLayout(universe_row)
        self.study_universe_note = QtWidgets.QLabel()
        self.study_universe_note.setObjectName("Sub")
        self.study_universe_note.setWordWrap(True)
        main_layout.addWidget(self.study_universe_note)
        self.study_mode_note = QtWidgets.QLabel()
        self.study_mode_note.setObjectName("Sub")
        self.study_mode_note.setWordWrap(True)
        main_layout.addWidget(self.study_mode_note)
        self.study_mode_combo.currentTextChanged.connect(lambda _: self._update_study_mode_note())

        add_setup_section(
            "Portfolio Construction",
            "These controls matter mainly for portfolio mode and remain scrollable so they do not crush the rest of the setup form.",
        )
        main_layout.addWidget(QtWidgets.QLabel("Strategy Blocks"))
        block_row = QtWidgets.QHBoxLayout()
        self.portfolio_blocks_summary = QtWidgets.QLineEdit()
        self.portfolio_blocks_summary.setReadOnly(True)
        self.portfolio_blocks_summary.setPlaceholderText("No explicit strategy blocks configured")
        edit_blocks_btn = QtWidgets.QPushButton("Edit Blocks")
        edit_blocks_btn.clicked.connect(self._edit_portfolio_strategy_blocks)
        clear_blocks_btn = QtWidgets.QPushButton("Clear Blocks")
        clear_blocks_btn.clicked.connect(self._clear_portfolio_strategy_blocks)
        block_row.addWidget(self.portfolio_blocks_summary, 3)
        block_row.addWidget(edit_blocks_btn, 1)
        block_row.addWidget(clear_blocks_btn, 1)
        main_layout.addLayout(block_row)
        self.portfolio_blocks_note = QtWidgets.QLabel()
        self.portfolio_blocks_note.setObjectName("Sub")
        self.portfolio_blocks_note.setWordWrap(True)
        main_layout.addWidget(self.portfolio_blocks_note)
        main_layout.addWidget(QtWidgets.QLabel("Portfolio Allocation"))
        main_layout.addWidget(self.portfolio_allocation_combo)
        self.portfolio_weights_edit = QtWidgets.QLineEdit()
        self.portfolio_weights_edit.setPlaceholderText("Optional. SPY=0.6,QQQ=0.4 or 0.6,0.4 in selected order")
        self.portfolio_weights_edit.setToolTip(
            "Used for Relative Weights and Fixed Weights.\n"
            "Examples:\n"
            "- SPY=0.6,QQQ=0.4\n"
            "- 0.6,0.4 (applies in the selected dataset order)\n"
            "Named weights may omit datasets to set them to zero."
        )
        main_layout.addWidget(QtWidgets.QLabel("Portfolio Weights"))
        main_layout.addWidget(self.portfolio_weights_edit)
        self.portfolio_allocation_summary = QtWidgets.QLabel()
        self.portfolio_allocation_summary.setObjectName("Sub")
        self.portfolio_allocation_summary.setWordWrap(True)
        main_layout.addWidget(self.portfolio_allocation_summary)
        self.portfolio_allocation_combo.currentTextChanged.connect(lambda _: self._update_portfolio_allocation_summary())
        self.portfolio_weights_edit.textChanged.connect(lambda _: self._update_portfolio_allocation_summary())
        self.portfolio_ownership_combo = QtWidgets.QComboBox()
        self.portfolio_ownership_combo.addItem("Strategy-Owned", ALLOCATION_OWNERSHIP_STRATEGY)
        self.portfolio_ownership_combo.addItem("Portfolio-Owned", ALLOCATION_OWNERSHIP_PORTFOLIO)
        self.portfolio_ownership_combo.addItem("Hybrid", ALLOCATION_OWNERSHIP_HYBRID)
        self.portfolio_ownership_combo.setToolTip(
            "Strategy-Owned keeps strategy sizing in control.\n"
            "Portfolio-Owned lets the portfolio layer decide allocation from strategy candidates.\n"
            "Hybrid keeps strategy weights but allows explicit portfolio filters such as ranking."
        )
        main_layout.addWidget(QtWidgets.QLabel("Allocation Ownership"))
        main_layout.addWidget(self.portfolio_ownership_combo)
        self.portfolio_ranking_combo = QtWidgets.QComboBox()
        self.portfolio_ranking_combo.addItem("No Ranking", RANKING_MODE_NONE)
        self.portfolio_ranking_combo.addItem("Top N", RANKING_MODE_TOP_N)
        self.portfolio_ranking_combo.addItem("Score Threshold", RANKING_MODE_SCORE_THRESHOLD)
        self.portfolio_ranking_combo.addItem("Top N Over Threshold", RANKING_MODE_TOP_N_OVER_SCORE_THRESHOLD)
        self.portfolio_ranking_combo.setToolTip(
            "Top N keeps only the strongest ranked assets among the currently eligible portfolio candidates.\n"
            "Score Threshold keeps only assets whose score is at or above the configured minimum score.\n"
            "Top N Over Threshold first filters by minimum score, then keeps the strongest N survivors."
        )
        main_layout.addWidget(QtWidgets.QLabel("Portfolio Ranking"))
        main_layout.addWidget(self.portfolio_ranking_combo)
        self.portfolio_rank_count_spin = QtWidgets.QSpinBox()
        self.portfolio_rank_count_spin.setRange(1, 9999)
        self.portfolio_rank_count_spin.setValue(1)
        main_layout.addWidget(QtWidgets.QLabel("Max Ranked Assets"))
        main_layout.addWidget(self.portfolio_rank_count_spin)
        self.portfolio_score_threshold_spin = QtWidgets.QDoubleSpinBox()
        self.portfolio_score_threshold_spin.setDecimals(4)
        self.portfolio_score_threshold_spin.setRange(-999999.0, 999999.0)
        self.portfolio_score_threshold_spin.setSingleStep(0.05)
        self.portfolio_score_threshold_spin.setValue(0.0)
        self.portfolio_score_threshold_spin.setToolTip(
            "Minimum portfolio score required for Score Threshold ranking."
        )
        main_layout.addWidget(QtWidgets.QLabel("Min Rank Score"))
        main_layout.addWidget(self.portfolio_score_threshold_spin)
        self.portfolio_weighting_combo = QtWidgets.QComboBox()
        self.portfolio_weighting_combo.addItem("Preserve Strategy Weights", WEIGHTING_MODE_PRESERVE)
        self.portfolio_weighting_combo.addItem("Equal Weight Selected", WEIGHTING_MODE_EQUAL_SELECTED)
        self.portfolio_weighting_combo.addItem("Score-Proportional", WEIGHTING_MODE_SCORE_PROPORTIONAL)
        self.portfolio_weighting_combo.setToolTip(
            "Preserve Strategy Weights keeps the surviving candidate weights as-is.\n"
            "Equal Weight Selected distributes portfolio weight evenly across the selected assets.\n"
            "Score-Proportional weights selected assets by their current portfolio score."
        )
        main_layout.addWidget(QtWidgets.QLabel("Portfolio Weighting"))
        main_layout.addWidget(self.portfolio_weighting_combo)
        self.portfolio_min_active_weight_spin = QtWidgets.QDoubleSpinBox()
        self.portfolio_min_active_weight_spin.setDecimals(4)
        self.portfolio_min_active_weight_spin.setRange(0.0, 1.0)
        self.portfolio_min_active_weight_spin.setSingleStep(0.01)
        self.portfolio_min_active_weight_spin.setValue(0.0)
        self.portfolio_min_active_weight_spin.setToolTip(
            "Optional minimum target weight for any active asset after portfolio construction. Set 0 to disable."
        )
        main_layout.addWidget(QtWidgets.QLabel("Min Active Weight"))
        main_layout.addWidget(self.portfolio_min_active_weight_spin)
        self.portfolio_max_asset_weight_spin = QtWidgets.QDoubleSpinBox()
        self.portfolio_max_asset_weight_spin.setDecimals(4)
        self.portfolio_max_asset_weight_spin.setRange(0.0, 1.0)
        self.portfolio_max_asset_weight_spin.setSingleStep(0.01)
        self.portfolio_max_asset_weight_spin.setValue(0.0)
        self.portfolio_max_asset_weight_spin.setToolTip(
            "Optional cap on any single asset target weight after portfolio construction. Set 0 to disable."
        )
        main_layout.addWidget(QtWidgets.QLabel("Max Asset Weight"))
        main_layout.addWidget(self.portfolio_max_asset_weight_spin)
        self.portfolio_cash_reserve_spin = QtWidgets.QDoubleSpinBox()
        self.portfolio_cash_reserve_spin.setDecimals(4)
        self.portfolio_cash_reserve_spin.setRange(0.0, 0.99)
        self.portfolio_cash_reserve_spin.setSingleStep(0.01)
        self.portfolio_cash_reserve_spin.setValue(0.0)
        self.portfolio_cash_reserve_spin.setToolTip(
            "Optional fraction of capital to keep unallocated as cash reserve."
        )
        main_layout.addWidget(QtWidgets.QLabel("Cash Reserve Weight"))
        main_layout.addWidget(self.portfolio_cash_reserve_spin)
        self.portfolio_rebalance_combo = QtWidgets.QComboBox()
        self.portfolio_rebalance_combo.addItem("On Change", REBALANCE_MODE_ON_CHANGE)
        self.portfolio_rebalance_combo.addItem("On Change + Periodic", REBALANCE_MODE_ON_CHANGE_OR_PERIODIC)
        self.portfolio_rebalance_combo.addItem("On Change + Drift Threshold", REBALANCE_MODE_ON_CHANGE_OR_DRIFT)
        self.portfolio_rebalance_combo.addItem(
            "On Change + Periodic + Drift Threshold",
            REBALANCE_MODE_ON_CHANGE_OR_PERIODIC_OR_DRIFT,
        )
        self.portfolio_rebalance_combo.setToolTip(
            "On Change only rebalances when desired target weights change.\n"
            "On Change + Periodic also refreshes target quantities every N bars to manage drift.\n"
            "On Change + Drift Threshold also rebalances when actual weights drift too far from target weights.\n"
            "On Change + Periodic + Drift Threshold uses both periodic and drift-triggered refreshes."
        )
        main_layout.addWidget(QtWidgets.QLabel("Portfolio Rebalance"))
        main_layout.addWidget(self.portfolio_rebalance_combo)
        self.portfolio_rebalance_every_spin = QtWidgets.QSpinBox()
        self.portfolio_rebalance_every_spin.setRange(1, 999999)
        self.portfolio_rebalance_every_spin.setValue(20)
        main_layout.addWidget(QtWidgets.QLabel("Rebalance Every N Bars"))
        main_layout.addWidget(self.portfolio_rebalance_every_spin)
        self.portfolio_rebalance_drift_spin = QtWidgets.QDoubleSpinBox()
        self.portfolio_rebalance_drift_spin.setDecimals(4)
        self.portfolio_rebalance_drift_spin.setRange(0.0001, 1.0)
        self.portfolio_rebalance_drift_spin.setSingleStep(0.01)
        self.portfolio_rebalance_drift_spin.setValue(0.05)
        self.portfolio_rebalance_drift_spin.setToolTip(
            "Maximum absolute asset-weight drift allowed before a drift-triggered rebalance occurs."
        )
        main_layout.addWidget(QtWidgets.QLabel("Rebalance Drift Threshold"))
        main_layout.addWidget(self.portfolio_rebalance_drift_spin)
        self.portfolio_ownership_combo.currentTextChanged.connect(lambda _: self._update_portfolio_allocation_summary())
        self.portfolio_ranking_combo.currentTextChanged.connect(lambda _: self._update_portfolio_allocation_summary())
        self.portfolio_rank_count_spin.valueChanged.connect(lambda _: self._update_portfolio_allocation_summary())
        self.portfolio_score_threshold_spin.valueChanged.connect(lambda _: self._update_portfolio_allocation_summary())
        self.portfolio_weighting_combo.currentTextChanged.connect(lambda _: self._update_portfolio_allocation_summary())
        self.portfolio_min_active_weight_spin.valueChanged.connect(lambda _: self._update_portfolio_allocation_summary())
        self.portfolio_max_asset_weight_spin.valueChanged.connect(lambda _: self._update_portfolio_allocation_summary())
        self.portfolio_cash_reserve_spin.valueChanged.connect(lambda _: self._update_portfolio_allocation_summary())
        self.portfolio_rebalance_combo.currentTextChanged.connect(lambda _: self._update_portfolio_allocation_summary())
        self.portfolio_rebalance_every_spin.valueChanged.connect(lambda _: self._update_portfolio_allocation_summary())
        self.portfolio_rebalance_drift_spin.valueChanged.connect(lambda _: self._update_portfolio_allocation_summary())

        add_setup_section(
            "Run Controls",
            "Timeframes, horizons, risk settings, and the final run actions live here.",
        )
        main_layout.addWidget(QtWidgets.QLabel("Timeframes"))
        self.timeframes_combo = QtWidgets.QComboBox()
        self.timeframes_combo.setEditable(True)
        self.timeframes_combo.addItems(
            [
                "1 minutes",
                "5 minutes",
                "15 minutes",
                "1 hours",
                "1 minutes,5 minutes",
                "1 minutes,5 minutes,15 minutes",
                "1 minutes,5 minutes,15 minutes,1 hours",
            ]
        )
        main_layout.addWidget(self.timeframes_combo)

        main_layout.addWidget(QtWidgets.QLabel("Horizons"))
        self.horizons_combo = QtWidgets.QComboBox()
        self.horizons_combo.setEditable(True)
        self.horizons_combo.addItems(
            [
                "7d",
                "30d",
                "7d,30d",
            ]
        )
        main_layout.addWidget(self.horizons_combo)

        main_layout.addWidget(QtWidgets.QLabel("Risk-free rate (annual, e.g. 0.02)"))
        self.risk_free_edit = QtWidgets.QLineEdit("0.0")
        main_layout.addWidget(self.risk_free_edit)

        self.intrabar_chk = QtWidgets.QCheckBox("Intrabar simulation (multi-fills per bar)")
        self.intrabar_chk.setChecked(True)
        main_layout.addWidget(self.intrabar_chk)
        self.sharpe_debug_chk = QtWidgets.QCheckBox("Sharpe debug (print mean/std/periods)")
        self.sharpe_debug_chk.setChecked(False)
        main_layout.addWidget(self.sharpe_debug_chk)

        self.run_btn = QtWidgets.QPushButton("Run Grid")
        self.run_btn.clicked.connect(self._run_grid_clicked)
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(8)
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_grid_clicked)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setFormat("%p%")
        self.progress.setMinimumHeight(16)
        btn_row.addWidget(self.run_btn, 1)
        btn_row.addWidget(self.stop_btn, 0)
        btn_row.addWidget(self.progress, 2)
        main_layout.addLayout(btn_row)
        main_layout.addStretch(1)
        main_scroll.setWidget(main_tab)

        settings_tab = QtWidgets.QWidget()
        settings_layout = QtWidgets.QVBoxLayout(settings_tab)
        settings_layout.setContentsMargins(6, 6, 6, 6)
        settings_layout.setSpacing(6)

        settings_scroll = QtWidgets.QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setMinimumHeight(260)
        settings_inner = QtWidgets.QWidget()
        settings_form = QtWidgets.QFormLayout(settings_inner)
        settings_form.setContentsMargins(8, 8, 8, 8)
        settings_form.setSpacing(6)

        self.starting_cash_edit = QtWidgets.QLineEdit("100000")
        settings_form.addRow("Starting Cash", self.starting_cash_edit)
        self.fee_rate_edit = QtWidgets.QLineEdit("0.0002")
        settings_form.addRow("Fee Rate", self.fee_rate_edit)
        self.fee_buy_edit = QtWidgets.QLineEdit("0.0003")
        settings_form.addRow("Fee Buy", self.fee_buy_edit)
        self.fee_sell_edit = QtWidgets.QLineEdit("0.0005")
        settings_form.addRow("Fee Sell", self.fee_sell_edit)
        self.slippage_edit = QtWidgets.QLineEdit("0.0002")
        settings_form.addRow("Slippage", self.slippage_edit)
        self.slip_buy_edit = QtWidgets.QLineEdit("0.0003")
        settings_form.addRow("Slippage Buy", self.slip_buy_edit)
        self.slip_sell_edit = QtWidgets.QLineEdit("0.0001")
        settings_form.addRow("Slippage Sell", self.slip_sell_edit)
        self.borrow_rate_edit = QtWidgets.QLineEdit("0.0")
        settings_form.addRow("Borrow Rate", self.borrow_rate_edit)
        self.fill_ratio_edit = QtWidgets.QLineEdit("1.0")
        settings_form.addRow("Fill Ratio", self.fill_ratio_edit)

        self.fill_on_close_chk = QtWidgets.QCheckBox()
        self.fill_on_close_chk.setChecked(False)
        settings_form.addRow("Fill On Close", self.fill_on_close_chk)
        self.recalc_on_fill_chk = QtWidgets.QCheckBox()
        self.recalc_on_fill_chk.setChecked(True)
        settings_form.addRow("Recalc On Fill", self.recalc_on_fill_chk)
        self.allow_short_chk = QtWidgets.QCheckBox()
        self.allow_short_chk.setChecked(True)
        settings_form.addRow("Allow Short", self.allow_short_chk)
        self.use_cache_chk = QtWidgets.QCheckBox()
        self.use_cache_chk.setChecked(False)
        settings_form.addRow("Use Cache", self.use_cache_chk)
        self.prevent_scale_in_chk = QtWidgets.QCheckBox()
        self.prevent_scale_in_chk.setChecked(True)
        settings_form.addRow("Prevent Scale In", self.prevent_scale_in_chk)
        self.one_order_chk = QtWidgets.QCheckBox()
        self.one_order_chk.setChecked(True)
        settings_form.addRow("One Order/Signal", self.one_order_chk)

        settings_scroll.setWidget(settings_inner)
        settings_layout.addWidget(settings_scroll)
        settings_layout.addStretch(1)

        left_stack = QtWidgets.QStackedWidget()
        left_stack.addWidget(main_scroll)
        left_stack.addWidget(settings_tab)

        left_tabs.addTab("Setup")
        left_tabs.addTab("Backtest Settings")
        left_tabs.setCurrentIndex(0)
        left_stack.setCurrentIndex(0)
        left_tabs.currentChanged.connect(left_stack.setCurrentIndex)

        # Right column params scroll
        right = QtWidgets.QVBoxLayout()
        right.setSpacing(8)
        right.addWidget(QtWidgets.QLabel("Strategy Params"))
        params_scroll = QtWidgets.QScrollArea()
        params_scroll.setWidgetResizable(True)
        params_scroll.setMinimumWidth(320)
        params_scroll.setMinimumHeight(220)
        params_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        params_widget = QtWidgets.QWidget()
        self.strategy_params_box = QtWidgets.QFormLayout(params_widget)
        self.strategy_params_box.setContentsMargins(10, 10, 10, 10)
        self.strategy_params_box.setSpacing(10)
        params_scroll.setWidget(params_widget)
        right.addWidget(params_scroll)
        right.addStretch(1)

        left_layout.addWidget(left_tabs, 0)
        left_layout.addWidget(left_stack, 1)
        split.addWidget(left_column, 2)
        split.addLayout(right, 1)
        outer.addLayout(split)

        self._init_strategy_selector()
        self._refresh_dataset_options()
        self._update_study_mode_note()
        self._update_portfolio_strategy_block_summary()
        self._update_portfolio_allocation_summary()

        return panel

    def _build_home_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setSpacing(12)

        top_row = QtWidgets.QHBoxLayout()
        top_row.setSpacing(12)

        self.metric_equity = self._metric_card("Total Runs", "—")
        self.metric_sharpe = self._metric_card("Best Sharpe", "—")
        self.metric_return = self._metric_card("Best Return", "—")
        metrics_box = QtWidgets.QWidget()
        metrics_box.setObjectName("Panel")
        metrics_layout = QtWidgets.QHBoxLayout(metrics_box)
        metrics_layout.setContentsMargins(14, 12, 14, 12)
        metrics_layout.setSpacing(10)
        for w in (self.metric_equity, self.metric_sharpe, self.metric_return):
            metrics_layout.addWidget(w)
        metrics_layout.addStretch(1)

        self.batch_model = BatchTableModel([])
        table = QtWidgets.QTableView()
        table.setModel(self.batch_model)
        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        table.horizontalHeader().setStretchLastSection(True)
        table.verticalHeader().setVisible(False)
        table.setObjectName("Panel")
        table.setStyleSheet(
            f"""
            QTableView {{
                background: {PALETTE['panel2']};
                color: {PALETTE['text']};
                border: 1px solid {PALETTE['border']};
                gridline-color: {PALETTE['grid']};
            }}
            QHeaderView::section {{
                background: {PALETTE['panel']};
                color: {PALETTE['muted']};
                border: 1px solid {PALETTE['border']};
                padding: 6px 8px;
                font-weight: 600;
            }}
            QTableView::item:selected {{
                background: rgba(77, 163, 255, 0.2);
            }}
            """
        )
        table.doubleClicked.connect(self._open_batch_detail)
        self.batches_table = table

        runs_panel = QtWidgets.QWidget()
        runs_panel.setObjectName("Panel")
        runs_layout = QtWidgets.QVBoxLayout(runs_panel)
        runs_layout.setContentsMargins(10, 10, 10, 10)
        runs_layout.setSpacing(8)
        title = QtWidgets.QLabel("Backtest Runs")
        title.setObjectName("Title")
        subtitle = QtWidgets.QLabel("Cached results from SQLite catalog")
        subtitle.setObjectName("Sub")
        runs_layout.addWidget(title)
        runs_layout.addWidget(subtitle)
        runs_layout.addWidget(table)

        top_row.addWidget(metrics_box, 1)
        top_row.addWidget(runs_panel, 2)

        layout.addLayout(top_row)
        return tab

    def _build_universe_tab(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        self.universe_tab = panel
        panel.setObjectName("Panel")
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        title = QtWidgets.QLabel("Universe Builder")
        title.setObjectName("Title")
        subtitle = QtWidgets.QLabel(
            "Create reusable universes of provider symbols and local datasets. "
            "Universes can be reused in Orchestrate and Automate."
        )
        subtitle.setObjectName("Sub")
        subtitle.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(subtitle)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        layout.addWidget(split, 1)

        left_panel = QtWidgets.QWidget()
        left_panel.setObjectName("Panel")
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(8)

        left_title = QtWidgets.QLabel("Saved Universes")
        left_title.setObjectName("Title")
        left_note = QtWidgets.QLabel("Pick a universe to edit it, or create a new one.")
        left_note.setObjectName("Sub")
        left_note.setWordWrap(True)
        left_layout.addWidget(left_title)
        left_layout.addWidget(left_note)

        self.universe_table = QtWidgets.QTableWidget(0, 4)
        self.universe_table.setHorizontalHeaderLabels(["Universe", "Symbols", "Datasets", "Source"])
        self.universe_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.universe_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.universe_table.setAlternatingRowColors(True)
        self.universe_table.horizontalHeader().setStretchLastSection(True)
        self.universe_table.verticalHeader().setVisible(False)
        self.universe_table.itemSelectionChanged.connect(self._on_universe_table_selection_changed)
        left_layout.addWidget(self.universe_table, 1)

        left_buttons = QtWidgets.QHBoxLayout()
        new_btn = QtWidgets.QPushButton("New Universe")
        new_btn.clicked.connect(self._new_universe)
        refresh_btn = QtWidgets.QPushButton("Refresh")
        refresh_btn.clicked.connect(self._load_universes)
        acq_btn = QtWidgets.QPushButton("Acquisition View")
        acq_btn.clicked.connect(self._open_selected_universe_acquisition_catalog)
        delete_btn = QtWidgets.QPushButton("Delete")
        delete_btn.clicked.connect(self._delete_current_universe)
        left_buttons.addWidget(new_btn)
        left_buttons.addWidget(refresh_btn)
        left_buttons.addWidget(acq_btn)
        left_buttons.addStretch(1)
        left_buttons.addWidget(delete_btn)
        left_layout.addLayout(left_buttons)

        right_panel = QtWidgets.QWidget()
        right_panel.setObjectName("Panel")
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(10)

        editor_title = QtWidgets.QLabel("Universe Editor")
        editor_title.setObjectName("Title")
        editor_note = QtWidgets.QLabel(
            "Symbols drive download automation. Local dataset ids drive backtest and portfolio launch."
        )
        editor_note.setObjectName("Sub")
        editor_note.setWordWrap(True)
        right_layout.addWidget(editor_title)
        right_layout.addWidget(editor_note)

        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        form.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(10)

        self.universe_name_edit = QtWidgets.QLineEdit()
        self.universe_source_combo = QtWidgets.QComboBox()
        self.universe_source_combo.addItem("Any Source", "")
        for provider in available_acquisition_providers():
            self.universe_source_combo.addItem(provider.label, provider.provider_id)
        self.universe_source_combo.currentIndexChanged.connect(self._refresh_universe_acquisition_summary)

        self.universe_desc_edit = QtWidgets.QPlainTextEdit()
        self.universe_desc_edit.setPlaceholderText("Optional notes for this universe.")
        self.universe_desc_edit.setFixedHeight(90)

        self.universe_symbols_edit = QtWidgets.QPlainTextEdit()
        self.universe_symbols_edit.setPlaceholderText("One symbol per line or comma-separated, e.g. SPY\\nQQQ\\nAAPL")
        self.universe_symbols_edit.setFixedHeight(180)
        self.universe_symbols_edit.textChanged.connect(self._refresh_universe_acquisition_summary)
        symbols_wrap = QtWidgets.QWidget()
        symbols_layout = QtWidgets.QVBoxLayout(symbols_wrap)
        symbols_layout.setContentsMargins(0, 0, 0, 0)
        symbols_layout.setSpacing(8)
        symbols_header_row = QtWidgets.QHBoxLayout()
        symbols_hint = QtWidgets.QLabel("Choose from the listed symbol universe or paste symbols manually.")
        symbols_hint.setObjectName("Sub")
        symbols_hint.setWordWrap(True)
        choose_symbols_btn = QtWidgets.QPushButton("Choose Symbols From List")
        choose_symbols_btn.clicked.connect(self._choose_universe_symbols)
        symbols_header_row.addWidget(symbols_hint, 1)
        symbols_header_row.addWidget(choose_symbols_btn, 0)
        symbols_layout.addLayout(symbols_header_row)
        symbols_layout.addWidget(self.universe_symbols_edit)
        symbol_btn_row = QtWidgets.QHBoxLayout()
        use_tickers_btn = QtWidgets.QPushButton("Use Selected Tickers")
        use_tickers_btn.clicked.connect(self._use_selected_tickers_for_universe)
        clear_symbols_btn = QtWidgets.QPushButton("Clear Symbols")
        clear_symbols_btn.clicked.connect(lambda: self.universe_symbols_edit.setPlainText(""))
        symbol_btn_row.addWidget(use_tickers_btn)
        symbol_btn_row.addWidget(clear_symbols_btn)
        symbol_btn_row.addStretch(1)
        symbols_layout.addLayout(symbol_btn_row)

        self.universe_dataset_summary = QtWidgets.QLineEdit()
        self.universe_dataset_summary.setReadOnly(True)
        self.universe_dataset_summary.setPlaceholderText("No local datasets selected")
        dataset_wrap = QtWidgets.QWidget()
        dataset_layout = QtWidgets.QVBoxLayout(dataset_wrap)
        dataset_layout.setContentsMargins(0, 0, 0, 0)
        dataset_layout.setSpacing(8)
        dataset_layout.addWidget(self.universe_dataset_summary)
        dataset_btn_row = QtWidgets.QHBoxLayout()
        choose_datasets_btn = QtWidgets.QPushButton("Choose Datasets")
        choose_datasets_btn.clicked.connect(self._choose_universe_datasets)
        use_study_btn = QtWidgets.QPushButton("Use Current Study Datasets")
        use_study_btn.clicked.connect(self._use_current_study_datasets_for_universe)
        dataset_acq_btn = QtWidgets.QPushButton("View Dataset Acquisition")
        dataset_acq_btn.clicked.connect(self._open_current_universe_dataset_acquisition_detail)
        clear_datasets_btn = QtWidgets.QPushButton("Clear Datasets")
        clear_datasets_btn.clicked.connect(self._clear_universe_datasets)
        dataset_btn_row.addWidget(choose_datasets_btn)
        dataset_btn_row.addWidget(use_study_btn)
        dataset_btn_row.addWidget(dataset_acq_btn)
        dataset_btn_row.addWidget(clear_datasets_btn)
        dataset_btn_row.addStretch(1)
        dataset_layout.addLayout(dataset_btn_row)

        form.addRow("Universe Name", self.universe_name_edit)
        form.addRow("Preferred Source", self.universe_source_combo)
        form.addRow("Description", self.universe_desc_edit)
        form.addRow("Symbols", symbols_wrap)
        form.addRow("Local Datasets", dataset_wrap)
        right_layout.addLayout(form)

        self.universe_editor_status = QtWidgets.QLabel("Create a new universe and save it when ready.")
        self.universe_editor_status.setObjectName("Sub")
        self.universe_editor_status.setWordWrap(True)
        right_layout.addWidget(self.universe_editor_status)

        acquisition_box = QtWidgets.QGroupBox("Universe Acquisition Summary")
        acquisition_layout = QtWidgets.QVBoxLayout(acquisition_box)
        acquisition_layout.setContentsMargins(10, 10, 10, 10)
        acquisition_layout.setSpacing(8)
        self.universe_acquisition_summary = QtWidgets.QPlainTextEdit()
        self.universe_acquisition_summary.setReadOnly(True)
        self.universe_acquisition_summary.setMinimumHeight(150)
        acquisition_layout.addWidget(self.universe_acquisition_summary)
        right_layout.addWidget(acquisition_box)

        save_row = QtWidgets.QHBoxLayout()
        clear_btn = QtWidgets.QPushButton("Reset Editor")
        clear_btn.clicked.connect(self._new_universe)
        save_btn = QtWidgets.QPushButton("Save Universe")
        save_btn.clicked.connect(self._save_current_universe)
        save_row.addStretch(1)
        save_row.addWidget(clear_btn)
        save_row.addWidget(save_btn)
        right_layout.addLayout(save_row)

        split.addWidget(left_panel)
        split.addWidget(right_panel)
        split.setSizes([420, 920])

        self._new_universe()
        return panel

    def _load_universes(self) -> None:
        self.universes = self.catalog.load_universes()
        self._refresh_universe_table()
        self._refresh_universe_options()

    def _refresh_universe_table(self) -> None:
        if not hasattr(self, "universe_table"):
            return
        current_id = self._editing_universe_id
        self.universe_table.setRowCount(len(self.universes))
        selected_row = -1
        for row_idx, universe in enumerate(self.universes):
            values = [
                str(universe.get("name", "")),
                str(len(list(universe.get("symbols") or []))),
                str(len(list(universe.get("dataset_ids") or []))),
                str(universe.get("source_preference") or "Any"),
            ]
            for col_idx, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                if col_idx == 0:
                    item.setData(QtCore.Qt.ItemDataRole.UserRole, str(universe.get("universe_id", "")))
                    tooltip_lines = [
                        f"Universe: {universe.get('name', '')}",
                        f"Symbols: {', '.join(list(universe.get('symbols') or [])) or '—'}",
                        f"Datasets: {', '.join(list(universe.get('dataset_ids') or [])) or '—'}",
                    ]
                    if str(universe.get("description") or "").strip():
                        tooltip_lines.append(f"Description: {universe.get('description')}")
                    item.setToolTip("\n".join(tooltip_lines))
                self.universe_table.setItem(row_idx, col_idx, item)
            if str(universe.get("universe_id", "")) == current_id:
                selected_row = row_idx
        if selected_row >= 0:
            self.universe_table.selectRow(selected_row)
        elif self.universe_table.rowCount() > 0 and not self.universe_table.selectedItems():
            self.universe_table.selectRow(0)

    def _refresh_universe_options(self) -> None:
        options = [(str(item.get("universe_id", "")), str(item.get("name", ""))) for item in self.universes]
        for combo_name, selected_id in (
            ("study_universe_combo", self.study_universe_id),
            ("download_universe_combo", self.download_universe_id),
        ):
            combo = getattr(self, combo_name, None)
            if combo is None:
                continue
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("Manual Selection", "")
            for universe_id, name in options:
                combo.addItem(name, universe_id)
            idx = combo.findData(selected_id)
            combo.setCurrentIndex(idx if idx >= 0 else 0)
            combo.blockSignals(False)
            resolved_id = str(combo.currentData() or "")
            if combo_name == "study_universe_combo":
                self.study_universe_id = resolved_id
            elif combo_name == "download_universe_combo":
                self.download_universe_id = resolved_id
        self._update_study_dataset_summary()
        self._update_ticker_summary()

    def _find_universe(self, universe_id: str) -> dict | None:
        universe_id = str(universe_id or "").strip()
        if not universe_id:
            return None
        for universe in self.universes:
            if str(universe.get("universe_id", "")) == universe_id:
                return universe
        return None

    def _new_universe(self) -> None:
        self._editing_universe_id = ""
        self._editing_universe_dataset_ids = []
        if hasattr(self, "universe_name_edit"):
            self.universe_name_edit.clear()
            self.universe_source_combo.setCurrentIndex(0)
            self.universe_desc_edit.setPlainText("")
            self.universe_symbols_edit.setPlainText("")
            self._clear_universe_datasets()
            self.universe_editor_status.setText("Create a new universe and save it when ready.")
            self._refresh_universe_acquisition_summary()

    def _on_universe_table_selection_changed(self) -> None:
        if not hasattr(self, "universe_table"):
            return
        selection_model = self.universe_table.selectionModel()
        if selection_model is None:
            return
        rows = selection_model.selectedRows()
        if not rows:
            return
        item = self.universe_table.item(rows[0].row(), 0)
        if item is None:
            return
        universe_id = str(item.data(QtCore.Qt.ItemDataRole.UserRole) or "").strip()
        universe = self._find_universe(universe_id)
        if universe is not None:
            self._load_universe_into_editor(universe)

    def _load_universe_into_editor(self, universe: dict) -> None:
        self._editing_universe_id = str(universe.get("universe_id", ""))
        self.universe_name_edit.setText(str(universe.get("name", "")))
        idx = self.universe_source_combo.findData(str(universe.get("source_preference") or ""))
        self.universe_source_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.universe_desc_edit.setPlainText(str(universe.get("description", "")))
        self.universe_symbols_edit.setPlainText("\n".join(list(universe.get("symbols") or [])))
        self._editing_universe_dataset_ids = [
            str(item) for item in list(universe.get("dataset_ids") or []) if str(item).strip()
        ]
        self._refresh_universe_dataset_summary()
        self._refresh_universe_acquisition_summary()
        self.universe_editor_status.setText(
            f"Editing universe '{universe.get('name', '')}' with "
            f"{len(list(universe.get('symbols') or []))} symbols and {len(self._editing_universe_dataset_ids)} datasets."
        )

    def _parse_text_tokens(self, text: str, *, uppercase: bool = False) -> list[str]:
        tokens = [part.strip() for part in re.split(r"[\n,]+", str(text or "")) if part.strip()]
        if uppercase:
            return sorted(set(token.upper() for token in tokens))
        return sorted(set(tokens))

    def _choose_universe_symbols(self) -> None:
        if not self.nasdaq_symbols:
            self._load_nasdaq_symbols()
        if not self.nasdaq_symbols:
            QtWidgets.QMessageBox.warning(
                self,
                "Symbols missing",
                f"Symbols file not found or empty: {NASDAQ_SYMBOLS_PATH}",
            )
            return
        current_symbols = set(self._parse_text_tokens(self.universe_symbols_edit.toPlainText(), uppercase=True))
        dlg = TickerPickerDialog(self.nasdaq_symbols, current_symbols, self)
        if dlg.exec() != int(QtWidgets.QDialog.DialogCode.Accepted):
            return
        self.universe_symbols_edit.setPlainText("\n".join(sorted(set(dlg.selected))))
        self.universe_editor_status.setText(
            f"Selected {len(dlg.selected)} symbol(s) for the current universe draft."
        )

    def _refresh_universe_dataset_summary(self) -> None:
        if not hasattr(self, "universe_dataset_summary"):
            return
        dataset_ids = [str(item) for item in self._editing_universe_dataset_ids if str(item).strip()]
        if not dataset_ids:
            self.universe_dataset_summary.clear()
            self.universe_dataset_summary.setPlaceholderText("No local datasets selected")
            self.universe_dataset_summary.setToolTip("")
            return
        label = ", ".join(dataset_ids) if len(dataset_ids) <= 3 else f"{len(dataset_ids)} datasets selected"
        self.universe_dataset_summary.setText(label)
        self.universe_dataset_summary.setToolTip("\n".join(dataset_ids))

    def _refresh_universe_acquisition_summary(self) -> None:
        if not hasattr(self, "universe_acquisition_summary"):
            return
        symbols = self._parse_text_tokens(self.universe_symbols_edit.toPlainText(), uppercase=True)
        dataset_ids = [str(item) for item in self._editing_universe_dataset_ids if str(item).strip()]
        preferred_source = str(self.universe_source_combo.currentData() or "") if hasattr(
            self, "universe_source_combo"
        ) else ""
        if not symbols and not dataset_ids:
            self.universe_acquisition_summary.setPlainText(
                "No symbols or local datasets are in this universe draft yet.\n\n"
                "Add symbols and/or dataset bindings to see freshness, coverage, recent downloads, and failure state."
            )
            return

        tracked_rows = self.catalog.load_acquisition_datasets()
        matched_rows = [
            row
            for row in tracked_rows
            if row.dataset_id in dataset_ids or (row.symbol and str(row.symbol).upper() in symbols)
        ]
        matched_dataset_ids = {row.dataset_id for row in matched_rows}
        matched_symbols = {str(row.symbol).upper() for row in matched_rows if row.symbol}
        fresh_count = 0
        stale_count = 0
        unknown_count = 0
        ingested_count = 0
        error_count = 0
        skipped_count = 0
        sources: set[str] = set()
        coverage_starts: list[pd.Timestamp] = []
        coverage_ends: list[pd.Timestamp] = []
        last_success_times: list[pd.Timestamp] = []
        last_attempt_times: list[pd.Timestamp] = []
        recent_failures: list[str] = []

        for row in matched_rows:
            if row.source:
                sources.add(str(row.source))
            if row.ingested:
                ingested_count += 1
            freshness = compute_freshness_state(row.coverage_end, row.resolution)
            if freshness == "fresh":
                fresh_count += 1
            elif freshness == "stale":
                stale_count += 1
            else:
                unknown_count += 1
            if row.last_status in {"download_error", "ingest_error", "failed"}:
                error_count += 1
                recent_failures.append(
                    f"- {row.dataset_id}: {row.last_status} | {row.last_error or 'No details recorded.'}"
                )
            elif row.last_status == "skipped":
                skipped_count += 1

            start_ts = pd.to_datetime(row.coverage_start, utc=True, errors="coerce")
            end_ts = pd.to_datetime(row.coverage_end, utc=True, errors="coerce")
            success_ts = pd.to_datetime(row.last_download_success_at, utc=True, errors="coerce")
            attempt_ts = pd.to_datetime(row.last_download_attempt_at, utc=True, errors="coerce")
            if pd.notna(start_ts):
                coverage_starts.append(start_ts)
            if pd.notna(end_ts):
                coverage_ends.append(end_ts)
            if pd.notna(success_ts):
                last_success_times.append(success_ts)
            if pd.notna(attempt_ts):
                last_attempt_times.append(attempt_ts)

        unresolved_symbols = [symbol for symbol in symbols if symbol not in matched_symbols]
        missing_dataset_ids = [dataset_id for dataset_id in dataset_ids if dataset_id not in matched_dataset_ids]

        def _fmt_timestamp(value: pd.Timestamp | None) -> str:
            if value is None or pd.isna(value):
                return "—"
            return value.tz_convert("America/New_York").strftime("%Y-%m-%d %H:%M ET")

        coverage_window = "—"
        if coverage_starts and coverage_ends:
            coverage_window = (
                f"{min(coverage_starts).strftime('%Y-%m-%d')} -> {max(coverage_ends).strftime('%Y-%m-%d')}"
            )

        lines = [
            f"Preferred Source: {provider_display_name(preferred_source) if preferred_source else 'Any Source'}",
            f"Symbols In Draft: {len(symbols)} | Dataset Bindings: {len(dataset_ids)}",
            f"Tracked Acquisition Datasets: {len(matched_rows)} | Ingested: {ingested_count}",
            f"Fresh: {fresh_count} | Stale: {stale_count} | Unknown: {unknown_count} | Recent Failures: {error_count} | Recent Skips: {skipped_count}",
            f"Coverage Window: {coverage_window}",
            f"Last Successful Download: {_fmt_timestamp(max(last_success_times) if last_success_times else None)}",
            f"Last Attempt: {_fmt_timestamp(max(last_attempt_times) if last_attempt_times else None)}",
            f"Sources Seen: {', '.join(provider_display_name(source) for source in sorted(sources)) or '—'}",
        ]
        if unresolved_symbols:
            preview = ", ".join(unresolved_symbols[:10])
            if len(unresolved_symbols) > 10:
                preview += f" (+{len(unresolved_symbols) - 10} more)"
            lines.append(f"Symbols Without Acquisition Metadata: {preview}")
        if missing_dataset_ids:
            preview = ", ".join(missing_dataset_ids[:6])
            if len(missing_dataset_ids) > 6:
                preview += f" (+{len(missing_dataset_ids) - 6} more)"
            lines.append(f"Dataset Bindings Without Catalog Entry: {preview}")
        if recent_failures:
            lines.append("")
            lines.append("Recent Failures:")
            lines.extend(recent_failures[:5])
        self.universe_acquisition_summary.setPlainText("\n".join(lines))

    def _choose_universe_datasets(self) -> None:
        available = self._available_dataset_ids()
        if not available:
            QtWidgets.QMessageBox.information(self, "No datasets", "No datasets are available in the local store yet.")
            return
        dlg = DatasetSelectionDialog(available, list(self._editing_universe_dataset_ids), self)
        if dlg.exec() != int(QtWidgets.QDialog.DialogCode.Accepted):
            return
        self._editing_universe_dataset_ids = dlg.selected_datasets()
        self._refresh_universe_dataset_summary()
        self._refresh_universe_acquisition_summary()

    def _clear_universe_datasets(self) -> None:
        self._editing_universe_dataset_ids = []
        self._refresh_universe_dataset_summary()
        self._refresh_universe_acquisition_summary()

    def _use_current_study_datasets_for_universe(self) -> None:
        self._editing_universe_dataset_ids = list(self._selected_study_dataset_ids(manual_only=True))
        self._refresh_universe_dataset_summary()
        self._refresh_universe_acquisition_summary()

    def _use_selected_tickers_for_universe(self) -> None:
        symbols = self.selected_tickers if not self.select_all_tickers else self.nasdaq_symbols
        clean = sorted(set(str(sym).upper() for sym in symbols if str(sym).strip()))
        self.universe_symbols_edit.setPlainText("\n".join(clean))

    def _save_current_universe(self) -> None:
        name = self.universe_name_edit.text().strip()
        symbols = self._parse_text_tokens(self.universe_symbols_edit.toPlainText(), uppercase=True)
        dataset_ids = [str(item) for item in self._editing_universe_dataset_ids if str(item).strip()]
        if not name:
            QtWidgets.QMessageBox.warning(self, "Universe name", "Provide a universe name.")
            return
        if not symbols and not dataset_ids:
            QtWidgets.QMessageBox.warning(self, "Universe empty", "Add at least one symbol or one dataset.")
            return
        universe_id = self._editing_universe_id or f"universe_{uuid.uuid4().hex[:8]}"
        try:
            ResultCatalog(self.catalog.db_path).save_universe(
                universe_id=universe_id,
                name=name,
                description=self.universe_desc_edit.toPlainText().strip(),
                symbols=symbols,
                dataset_ids=dataset_ids,
                source_preference=str(self.universe_source_combo.currentData() or ""),
            )
        except Exception as exc:
            self._show_error_dialog("Save Universe Error", str(exc), details=traceback.format_exc())
            return
        self._editing_universe_id = universe_id
        self._load_universes()
        self._refresh_universe_acquisition_summary()
        self.universe_editor_status.setText(
            f"Saved universe '{name}' with {len(symbols)} symbols and {len(dataset_ids)} datasets."
        )

    def _delete_current_universe(self) -> None:
        universe = self._find_universe(self._editing_universe_id)
        if universe is None:
            QtWidgets.QMessageBox.information(self, "No universe selected", "Select a universe first.")
            return
        name = str(universe.get("name", "this universe"))
        if (
            QtWidgets.QMessageBox.question(
                self,
                "Delete Universe",
                f"Delete universe '{name}'?",
            )
            != QtWidgets.QMessageBox.StandardButton.Yes
        ):
            return
        try:
            ResultCatalog(self.catalog.db_path).delete_universe(str(universe.get("universe_id", "")))
        except Exception as exc:
            self._show_error_dialog("Delete Universe Error", str(exc), details=traceback.format_exc())
            return
        if self.study_universe_id == str(universe.get("universe_id", "")):
            self.study_universe_id = ""
        if self.download_universe_id == str(universe.get("universe_id", "")):
            self.download_universe_id = ""
        self._new_universe()
        self._load_universes()

    def _open_universe_builder_tab(self) -> None:
        if hasattr(self, "tabs") and hasattr(self, "universe_tab"):
            index = self.tabs.indexOf(self.universe_tab)
            if index >= 0:
                self.tabs.setCurrentIndex(index)

    def _selected_study_universe(self) -> dict | None:
        return self._find_universe(self.study_universe_id)

    def _selected_download_universe(self) -> dict | None:
        return self._find_universe(self.download_universe_id)

    def _selected_download_source_choice(self) -> str:
        if not hasattr(self, "download_source_combo"):
            return ""
        return str(self.download_source_combo.currentData() or "").strip().lower()

    def _resolved_download_source(self) -> str:
        universe = self._selected_download_universe() or {}
        return resolve_acquisition_source(
            self._selected_download_source_choice(),
            str(universe.get("source_preference") or ""),
            default_source=DEFAULT_ACQUISITION_PROVIDER,
        )

    def _provider_runtime_environment(self, provider_id: str) -> dict[str, str]:
        return build_provider_runtime_environment(provider_id, catalog=self.catalog)

    def _provider_settings_status_text(self, provider_id: str) -> str:
        return provider_settings_status(provider_id, catalog=self.catalog)

    def _open_provider_settings_dialog(self) -> None:
        dlg = ProviderSettingsDialog(self.catalog, self)
        if dlg.exec() == int(QtWidgets.QDialog.DialogCode.Accepted):
            self._update_ticker_summary()

    def _build_heatmap_tab(self) -> QtWidgets.QWidget:
        return self._build_heatmap_panel()

    def _build_optimization_tab(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        panel.setObjectName("Panel")
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        title = QtWidgets.QLabel("Parameter Optimization")
        title.setObjectName("Title")
        subtitle = QtWidgets.QLabel(
            "Durable optimization studies ranked by robustness across the selected dataset universe."
        )
        subtitle.setObjectName("Sub")
        self.optimization_summary_label = QtWidgets.QLabel("No optimization studies saved yet.")
        self.optimization_summary_label.setObjectName("Sub")
        self.optimization_summary_label.setWordWrap(True)
        self.optimization_summary_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)

        actions = QtWidgets.QHBoxLayout()
        self.optimization_walk_forward_btn = QtWidgets.QPushButton("Create Walk-Forward")
        self.optimization_walk_forward_btn.clicked.connect(self._launch_walk_forward_from_selected_optimization)
        actions.addWidget(self.optimization_walk_forward_btn)
        self.optimization_open_btn = QtWidgets.QPushButton("Open Selected Study")
        self.optimization_open_btn.clicked.connect(self._open_selected_optimization_study)
        actions.addWidget(self.optimization_open_btn)
        actions.addStretch(1)

        self.optimization_table = QtWidgets.QTableWidget(0, 7)
        self.optimization_table.setHorizontalHeaderLabels(
            ["Study ID", "Strategy", "Batch", "Datasets", "Params", "Timeframes", "Aggregates"]
        )
        self.optimization_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.optimization_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.optimization_table.setAlternatingRowColors(True)
        self.optimization_table.horizontalHeader().setStretchLastSection(True)
        self.optimization_table.verticalHeader().setVisible(False)
        self.optimization_table.setObjectName("Panel")
        self.optimization_table.itemDoubleClicked.connect(lambda _item: self._open_selected_optimization_study())

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(self.optimization_summary_label)
        layout.addLayout(actions)
        layout.addWidget(self.optimization_table, 1)
        return panel

    def _build_walk_forward_tab(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        panel.setObjectName("Panel")
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        title = QtWidgets.QLabel("Walk-Forward Validation")
        title.setObjectName("Title")
        subtitle = QtWidgets.QLabel(
            "Validate optimized parameter sets out of sample with anchored folds and a stitched OOS equity curve."
        )
        subtitle.setObjectName("Sub")
        self.walk_forward_summary_label = QtWidgets.QLabel("No walk-forward studies saved yet.")
        self.walk_forward_summary_label.setObjectName("Sub")
        self.walk_forward_summary_label.setWordWrap(True)
        self.walk_forward_summary_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(self.walk_forward_summary_label)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        layout.addWidget(split, 1)

        source_panel = QtWidgets.QWidget()
        source_panel.setObjectName("Panel")
        source_layout = QtWidgets.QVBoxLayout(source_panel)
        source_layout.setContentsMargins(8, 8, 8, 8)
        source_layout.setSpacing(8)
        source_title = QtWidgets.QLabel("Walk-Forward Sources")
        source_title.setObjectName("Title")
        source_note = QtWidgets.QLabel(
            "Seed walk-forward from either a saved optimization study or a saved portfolio batch."
        )
        source_note.setObjectName("Sub")
        source_note.setWordWrap(True)
        source_layout.addWidget(source_title)
        source_layout.addWidget(source_note)

        source_stack = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        source_stack.setChildrenCollapsible(False)
        source_layout.addWidget(source_stack, 1)

        optimization_source_panel = QtWidgets.QWidget()
        optimization_source_panel.setObjectName("Panel")
        optimization_source_layout = QtWidgets.QVBoxLayout(optimization_source_panel)
        optimization_source_layout.setContentsMargins(4, 4, 4, 4)
        optimization_source_layout.setSpacing(8)
        optimization_source_title = QtWidgets.QLabel("Source Optimization Studies")
        optimization_source_title.setObjectName("Title")
        optimization_source_note = QtWidgets.QLabel(
            "Select an optimization study to seed single-strategy walk-forward validation."
        )
        optimization_source_note.setObjectName("Sub")
        optimization_source_note.setWordWrap(True)
        self.walk_forward_source_summary = QtWidgets.QLabel("No optimization studies available yet.")
        self.walk_forward_source_summary.setObjectName("Sub")
        self.walk_forward_source_summary.setWordWrap(True)
        source_actions = QtWidgets.QHBoxLayout()
        self.walk_forward_launch_btn = QtWidgets.QPushButton("New From Selected Optimization Study")
        self.walk_forward_launch_btn.clicked.connect(self._launch_walk_forward_from_selected_optimization)
        source_actions.addWidget(self.walk_forward_launch_btn)
        source_actions.addStretch(1)
        self.walk_forward_source_table = QtWidgets.QTableWidget(0, 5)
        self.walk_forward_source_table.setHorizontalHeaderLabels(
            ["Study ID", "Strategy", "Datasets", "Timeframes", "Aggregates"]
        )
        self.walk_forward_source_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.walk_forward_source_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.walk_forward_source_table.setAlternatingRowColors(True)
        self.walk_forward_source_table.horizontalHeader().setStretchLastSection(True)
        self.walk_forward_source_table.verticalHeader().setVisible(False)
        self.walk_forward_source_table.setObjectName("Panel")
        self.walk_forward_source_table.itemDoubleClicked.connect(
            lambda _item: self._launch_walk_forward_from_selected_optimization()
        )
        optimization_source_layout.addWidget(optimization_source_title)
        optimization_source_layout.addWidget(optimization_source_note)
        optimization_source_layout.addWidget(self.walk_forward_source_summary)
        optimization_source_layout.addLayout(source_actions)
        optimization_source_layout.addWidget(self.walk_forward_source_table, 1)
        source_stack.addWidget(optimization_source_panel)

        portfolio_source_panel = QtWidgets.QWidget()
        portfolio_source_panel.setObjectName("Panel")
        portfolio_source_layout = QtWidgets.QVBoxLayout(portfolio_source_panel)
        portfolio_source_layout.setContentsMargins(4, 4, 4, 4)
        portfolio_source_layout.setSpacing(8)
        portfolio_source_title = QtWidgets.QLabel("Source Portfolio Batches")
        portfolio_source_title.setObjectName("Title")
        portfolio_source_note = QtWidgets.QLabel(
            "Select a saved portfolio batch to validate a shared-strategy or fixed strategy-block portfolio out of sample."
        )
        portfolio_source_note.setObjectName("Sub")
        portfolio_source_note.setWordWrap(True)
        self.walk_forward_portfolio_source_summary = QtWidgets.QLabel("No eligible portfolio batches available yet.")
        self.walk_forward_portfolio_source_summary.setObjectName("Sub")
        self.walk_forward_portfolio_source_summary.setWordWrap(True)
        portfolio_actions = QtWidgets.QHBoxLayout()
        self.walk_forward_portfolio_launch_btn = QtWidgets.QPushButton("New From Selected Portfolio Batch")
        self.walk_forward_portfolio_launch_btn.clicked.connect(self._launch_walk_forward_from_selected_portfolio_batch)
        portfolio_actions.addWidget(self.walk_forward_portfolio_launch_btn)
        portfolio_actions.addStretch(1)
        self.walk_forward_portfolio_source_table = QtWidgets.QTableWidget(0, 6)
        self.walk_forward_portfolio_source_table.setHorizontalHeaderLabels(
            ["Batch ID", "Strategy", "Mode", "Datasets", "Timeframes", "Runs"]
        )
        self.walk_forward_portfolio_source_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.walk_forward_portfolio_source_table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self.walk_forward_portfolio_source_table.setAlternatingRowColors(True)
        self.walk_forward_portfolio_source_table.horizontalHeader().setStretchLastSection(True)
        self.walk_forward_portfolio_source_table.verticalHeader().setVisible(False)
        self.walk_forward_portfolio_source_table.setObjectName("Panel")
        self.walk_forward_portfolio_source_table.itemDoubleClicked.connect(
            lambda _item: self._launch_walk_forward_from_selected_portfolio_batch()
        )
        portfolio_source_layout.addWidget(portfolio_source_title)
        portfolio_source_layout.addWidget(portfolio_source_note)
        portfolio_source_layout.addWidget(self.walk_forward_portfolio_source_summary)
        portfolio_source_layout.addLayout(portfolio_actions)
        portfolio_source_layout.addWidget(self.walk_forward_portfolio_source_table, 1)
        source_stack.addWidget(portfolio_source_panel)

        source_stack.setStretchFactor(0, 1)
        source_stack.setStretchFactor(1, 1)
        split.addWidget(source_panel)

        studies_panel = QtWidgets.QWidget()
        studies_panel.setObjectName("Panel")
        studies_layout = QtWidgets.QVBoxLayout(studies_panel)
        studies_layout.setContentsMargins(8, 8, 8, 8)
        studies_layout.setSpacing(8)
        studies_title = QtWidgets.QLabel("Saved Walk-Forward Studies")
        studies_title.setObjectName("Title")
        studies_note = QtWidgets.QLabel(
            "Open a saved study to inspect folds, train/test degradation, and stitched OOS performance."
        )
        studies_note.setObjectName("Sub")
        studies_note.setWordWrap(True)
        self.walk_forward_table = QtWidgets.QTableWidget(0, 8)
        self.walk_forward_table.setHorizontalHeaderLabels(
            ["WF Study ID", "Strategy", "Dataset", "Timeframe", "Candidate Source", "Folds", "Status", "Created"]
        )
        self.walk_forward_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.walk_forward_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.walk_forward_table.setAlternatingRowColors(True)
        self.walk_forward_table.horizontalHeader().setStretchLastSection(True)
        self.walk_forward_table.verticalHeader().setVisible(False)
        self.walk_forward_table.setObjectName("Panel")
        self.walk_forward_table.itemDoubleClicked.connect(lambda _item: self._open_selected_walk_forward_study())
        studies_actions = QtWidgets.QHBoxLayout()
        self.walk_forward_open_btn = QtWidgets.QPushButton("Open Selected Study")
        self.walk_forward_open_btn.clicked.connect(self._open_selected_walk_forward_study)
        studies_actions.addWidget(self.walk_forward_open_btn)
        studies_actions.addStretch(1)
        studies_layout.addWidget(studies_title)
        studies_layout.addWidget(studies_note)
        studies_layout.addLayout(studies_actions)
        studies_layout.addWidget(self.walk_forward_table, 1)
        split.addWidget(studies_panel)

        split.setStretchFactor(0, 5)
        split.setStretchFactor(1, 6)
        return panel

    def _build_monte_carlo_tab(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        panel.setObjectName("Panel")
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        title = QtWidgets.QLabel("Monte Carlo Robustness")
        title.setObjectName("Title")
        subtitle = QtWidgets.QLabel(
            "Stress-test validated walk-forward results with trade bootstrap and trade reshuffle distributions."
        )
        subtitle.setObjectName("Sub")
        self.monte_carlo_summary_label = QtWidgets.QLabel("No Monte Carlo studies saved yet.")
        self.monte_carlo_summary_label.setObjectName("Sub")
        self.monte_carlo_summary_label.setWordWrap(True)
        self.monte_carlo_summary_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(self.monte_carlo_summary_label)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        layout.addWidget(split, 1)

        source_panel = QtWidgets.QWidget()
        source_panel.setObjectName("Panel")
        source_layout = QtWidgets.QVBoxLayout(source_panel)
        source_layout.setContentsMargins(8, 8, 8, 8)
        source_layout.setSpacing(8)
        source_title = QtWidgets.QLabel("Source Walk-Forward Studies")
        source_title.setObjectName("Title")
        source_note = QtWidgets.QLabel(
            "Start from a completed walk-forward study. Monte Carlo v1 uses the stitched OOS trade sequence as the default source."
        )
        source_note.setObjectName("Sub")
        source_note.setWordWrap(True)
        self.monte_carlo_source_summary = QtWidgets.QLabel("No walk-forward studies available yet.")
        self.monte_carlo_source_summary.setObjectName("Sub")
        self.monte_carlo_source_summary.setWordWrap(True)
        source_actions = QtWidgets.QHBoxLayout()
        self.monte_carlo_launch_btn = QtWidgets.QPushButton("New From Selected Walk-Forward Study")
        self.monte_carlo_launch_btn.clicked.connect(self._launch_monte_carlo_from_selected_walk_forward)
        source_actions.addWidget(self.monte_carlo_launch_btn)
        source_actions.addStretch(1)
        self.monte_carlo_source_table = QtWidgets.QTableWidget(0, 7)
        self.monte_carlo_source_table.setHorizontalHeaderLabels(
            ["WF Study ID", "Type", "Strategy", "Dataset", "Timeframe", "Folds", "Status"]
        )
        self.monte_carlo_source_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.monte_carlo_source_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.monte_carlo_source_table.setAlternatingRowColors(True)
        self.monte_carlo_source_table.horizontalHeader().setStretchLastSection(True)
        self.monte_carlo_source_table.verticalHeader().setVisible(False)
        self.monte_carlo_source_table.setObjectName("Panel")
        self.monte_carlo_source_table.itemDoubleClicked.connect(
            lambda _item: self._launch_monte_carlo_from_selected_walk_forward()
        )
        source_layout.addWidget(source_title)
        source_layout.addWidget(source_note)
        source_layout.addWidget(self.monte_carlo_source_summary)
        source_layout.addLayout(source_actions)
        source_layout.addWidget(self.monte_carlo_source_table, 1)
        split.addWidget(source_panel)

        studies_panel = QtWidgets.QWidget()
        studies_panel.setObjectName("Panel")
        studies_layout = QtWidgets.QVBoxLayout(studies_panel)
        studies_layout.setContentsMargins(8, 8, 8, 8)
        studies_layout.setSpacing(8)
        studies_title = QtWidgets.QLabel("Saved Monte Carlo Studies")
        studies_title.setObjectName("Title")
        studies_note = QtWidgets.QLabel(
            "Review fan charts, return distributions, drawdown distributions, and downside probability summaries."
        )
        studies_note.setObjectName("Sub")
        studies_note.setWordWrap(True)
        studies_actions = QtWidgets.QHBoxLayout()
        self.monte_carlo_open_btn = QtWidgets.QPushButton("Open Selected Study")
        self.monte_carlo_open_btn.clicked.connect(self._open_selected_monte_carlo_study)
        studies_actions.addWidget(self.monte_carlo_open_btn)
        studies_actions.addStretch(1)
        self.monte_carlo_table = QtWidgets.QTableWidget(0, 9)
        self.monte_carlo_table.setHorizontalHeaderLabels(
            ["MC Study ID", "Source", "Type", "Mode", "Sims", "Trades", "Median Return", "Loss Prob", "Created"]
        )
        self.monte_carlo_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.monte_carlo_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.monte_carlo_table.setAlternatingRowColors(True)
        self.monte_carlo_table.horizontalHeader().setStretchLastSection(True)
        self.monte_carlo_table.verticalHeader().setVisible(False)
        self.monte_carlo_table.setObjectName("Panel")
        self.monte_carlo_table.itemDoubleClicked.connect(lambda _item: self._open_selected_monte_carlo_study())
        studies_layout.addWidget(studies_title)
        studies_layout.addWidget(studies_note)
        studies_layout.addLayout(studies_actions)
        studies_layout.addWidget(self.monte_carlo_table, 1)
        split.addWidget(studies_panel)

        split.setStretchFactor(0, 5)
        split.setStretchFactor(1, 6)
        return panel

    def _build_heatmap_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        panel.setObjectName("Panel")
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        title = QtWidgets.QLabel("Heatmaps")
        title.setObjectName("Title")
        subtitle = QtWidgets.QLabel("Saved grid-search heatmaps (latest shown)")
        subtitle.setObjectName("Sub")

        self.heatmap_label = QtWidgets.QLabel("No heatmap saved yet.")
        self.heatmap_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.heatmap_label.setMinimumHeight(320)
        self.heatmap_label.setStyleSheet("color: #9ab0d0;")

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(self.heatmap_label)
        return panel

    def _build_automate_tab(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        panel.setObjectName("Panel")
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        title = QtWidgets.QLabel("Automate")
        title.setObjectName("Title")
        subtitle = QtWidgets.QLabel(
            "Schedule data acquisition using the selected provider or a saved universe source preference."
        )
        subtitle.setObjectName("Sub")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        content_split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        content_split.setChildrenCollapsible(False)
        layout.addWidget(content_split, 1)

        left_scroll = QtWidgets.QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        left_container = QtWidgets.QWidget()
        left_container.setObjectName("Panel")
        left_layout = QtWidgets.QVBoxLayout(left_container)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(10)
        left_scroll.setWidget(left_container)
        content_split.addWidget(left_scroll)

        right_split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        right_split.setChildrenCollapsible(False)
        content_split.addWidget(right_split)

        universe_label = QtWidgets.QLabel("Download Universe")
        universe_label.setObjectName("Title")
        left_layout.addWidget(universe_label)
        self.download_universe_combo = QtWidgets.QComboBox()
        self.download_universe_combo.currentIndexChanged.connect(self._on_download_universe_changed)
        open_universe_btn = QtWidgets.QPushButton("Open Builder")
        open_universe_btn.clicked.connect(self._open_universe_builder_tab)
        clear_universe_btn = QtWidgets.QPushButton("Manual")
        clear_universe_btn.clicked.connect(lambda: self.download_universe_combo.setCurrentIndex(0))
        universe_row = QtWidgets.QHBoxLayout()
        universe_row.addWidget(self.download_universe_combo, 3)
        universe_row.addWidget(open_universe_btn, 1)
        universe_row.addWidget(clear_universe_btn, 1)
        left_layout.addLayout(universe_row)
        self.download_universe_note = QtWidgets.QLabel()
        self.download_universe_note.setObjectName("Sub")
        self.download_universe_note.setWordWrap(True)
        left_layout.addWidget(self.download_universe_note)

        source_label = QtWidgets.QLabel("Download Source")
        source_label.setObjectName("Title")
        left_layout.addWidget(source_label)
        self.download_source_combo = QtWidgets.QComboBox()
        self.download_source_combo.addItem("Auto (Use Universe Preference)", "")
        for provider in available_acquisition_providers():
            self.download_source_combo.addItem(provider.label, provider.provider_id)
        self.download_source_combo.currentIndexChanged.connect(self._update_ticker_summary)
        source_row = QtWidgets.QHBoxLayout()
        source_row.addWidget(self.download_source_combo, 2)
        provider_settings_btn = QtWidgets.QPushButton("Provider Settings")
        provider_settings_btn.clicked.connect(self._open_provider_settings_dialog)
        source_row.addWidget(provider_settings_btn, 1)
        source_row.addStretch(1)
        left_layout.addLayout(source_row)
        self.download_source_note = QtWidgets.QLabel()
        self.download_source_note.setObjectName("Sub")
        self.download_source_note.setWordWrap(True)
        left_layout.addWidget(self.download_source_note)

        self.force_refresh_chk = QtWidgets.QCheckBox("Force refresh even if local data looks fresh")
        self.force_refresh_chk.setChecked(False)
        self.force_refresh_chk.toggled.connect(self._update_ticker_summary)
        left_layout.addWidget(self.force_refresh_chk)

        ticker_label = QtWidgets.QLabel("Tickers (NASDAQ + Other Listed)")
        ticker_label.setObjectName("Title")
        left_layout.addWidget(ticker_label)
        self.ticker_summary = QtWidgets.QLineEdit()
        self.ticker_summary.setReadOnly(True)
        self.ticker_summary.setPlaceholderText("No tickers selected")
        self._load_nasdaq_symbols()
        select_all_btn = QtWidgets.QPushButton("Select All")
        select_all_btn.clicked.connect(self._select_all_tickers)
        choose_btn = QtWidgets.QPushButton("Choose Ticker")
        choose_btn.clicked.connect(self._open_ticker_picker)
        refresh_btn = QtWidgets.QPushButton("Update Symbols")
        refresh_btn.clicked.connect(self._update_nasdaq_symbols)
        ticker_row = QtWidgets.QHBoxLayout()
        ticker_row.addWidget(self.ticker_summary, 3)
        ticker_row.addWidget(select_all_btn, 1)
        ticker_row.addWidget(choose_btn, 1)
        ticker_row.addWidget(refresh_btn, 1)
        left_layout.addLayout(ticker_row)

        schedule_label = QtWidgets.QLabel("Scheduling")
        schedule_label.setObjectName("Title")
        left_layout.addWidget(schedule_label)
        self.schedule_combo = QtWidgets.QComboBox()
        self.schedule_combo.addItems(["Nightly", "Weekly", "Monthly"])
        schedule_row = QtWidgets.QHBoxLayout()
        schedule_row.addWidget(self.schedule_combo, 1)
        schedule_row.addStretch(2)
        left_layout.addWidget(QtWidgets.QLabel("Download Frequency"))
        left_layout.addLayout(schedule_row)

        schedule_grid = QtWidgets.QGridLayout()
        schedule_grid.setHorizontalSpacing(10)
        schedule_grid.setVerticalSpacing(6)

        schedule_grid.addWidget(QtWidgets.QLabel("Start Time"), 0, 0)
        time_row = QtWidgets.QHBoxLayout()
        time_row.setSpacing(6)
        self.schedule_time = QtWidgets.QTimeEdit()
        self.schedule_time.setDisplayFormat("hh:mm AP")
        self.schedule_time.setTime(QtCore.QTime.currentTime())
        self.schedule_time.setFixedWidth(110)
        self.schedule_time.setFixedHeight(30)
        self.schedule_time.setStyleSheet(
            f"""
            QTimeEdit {{
                background: {PALETTE['panel2']};
                color: {PALETTE['text']};
                border: 1px solid {PALETTE['border']};
                border-radius: 8px;
                padding: 2px 8px;
            }}
            QTimeEdit::up-button, QTimeEdit::down-button {{
                width: 12px;
                border: none;
            }}
            """
        )
        tz_label = QtWidgets.QLabel("ET")
        tz_label.setObjectName("Sub")
        time_row.addWidget(self.schedule_time)
        time_row.addWidget(tz_label)
        time_wrap = QtWidgets.QWidget()
        time_wrap.setLayout(time_row)
        time_wrap.setFixedWidth(150)
        schedule_grid.addWidget(time_wrap, 0, 1)

        schedule_grid.addWidget(QtWidgets.QLabel("Days of Week"), 1, 0)
        self.weekday_checks = []
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        days_row = QtWidgets.QHBoxLayout()
        for d in days:
            chk = QtWidgets.QCheckBox(d)
            chk.setChecked(d in ["Mon", "Tue", "Wed", "Thu", "Fri"])
            self.weekday_checks.append(chk)
            days_row.addWidget(chk)
        day_wrap = QtWidgets.QWidget()
        day_wrap.setLayout(days_row)
        schedule_grid.addWidget(day_wrap, 1, 1)

        schedule_grid.addWidget(QtWidgets.QLabel("Weeks of Month"), 2, 0)
        self.week_of_month_checks = []
        week_labels = ["1", "2", "3", "4", "Last"]
        weeks_row = QtWidgets.QHBoxLayout()
        for w in week_labels:
            chk = QtWidgets.QCheckBox(w)
            chk.setChecked(w == "1")
            self.week_of_month_checks.append(chk)
            weeks_row.addWidget(chk)
        weeks_wrap = QtWidgets.QWidget()
        weeks_wrap.setLayout(weeks_row)
        schedule_grid.addWidget(weeks_wrap, 2, 1)

        schedule_grid.addWidget(QtWidgets.QLabel("Months"), 3, 0)
        self.month_checks = []
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        months_row = QtWidgets.QHBoxLayout()
        for m in months:
            chk = QtWidgets.QCheckBox(m)
            chk.setChecked(True)
            self.month_checks.append(chk)
            months_row.addWidget(chk)
        months_wrap = QtWidgets.QWidget()
        months_wrap.setLayout(months_row)
        schedule_grid.addWidget(months_wrap, 3, 1)

        left_layout.addLayout(schedule_grid)
        left_layout.addStretch(1)

        downloads_panel = QtWidgets.QWidget()
        downloads_panel.setObjectName("Panel")
        downloads_layout = QtWidgets.QVBoxLayout(downloads_panel)
        downloads_layout.setContentsMargins(10, 10, 10, 10)
        downloads_layout.setSpacing(10)
        downloads_title = QtWidgets.QLabel("Current Downloads")
        downloads_title.setObjectName("Title")
        downloads_sub = QtWidgets.QLabel("Monitor active acquisition jobs, logs, and per-ticker progress.")
        downloads_sub.setObjectName("Sub")
        downloads_sub.setWordWrap(True)
        downloads_layout.addWidget(downloads_title)
        downloads_layout.addWidget(downloads_sub)

        controls_row = QtWidgets.QHBoxLayout()
        self.download_start_btn = QtWidgets.QPushButton("Start Download")
        self.download_start_btn.clicked.connect(self._start_download)
        self.download_pause_btn = QtWidgets.QPushButton("Pause")
        self.download_pause_btn.clicked.connect(self._pause_download)
        self.download_resume_btn = QtWidgets.QPushButton("Resume")
        self.download_resume_btn.clicked.connect(self._resume_download)
        self.download_stop_btn = QtWidgets.QPushButton("Stop")
        self.download_stop_btn.clicked.connect(self._stop_download)
        controls_row.addWidget(self.download_start_btn)
        controls_row.addWidget(self.download_pause_btn)
        controls_row.addWidget(self.download_resume_btn)
        controls_row.addWidget(self.download_stop_btn)
        downloads_layout.addLayout(controls_row)

        self.download_status = QtWidgets.QLabel("Idle")
        self.download_status.setObjectName("Sub")
        downloads_layout.addWidget(self.download_status)
        self.download_progress = QtWidgets.QProgressBar()
        self.download_progress.setRange(0, 0)
        self.download_progress.setVisible(False)
        downloads_layout.addWidget(self.download_progress)

        concurrency_row = QtWidgets.QHBoxLayout()
        self.concurrency_spin = QtWidgets.QSpinBox()
        self.concurrency_spin.setRange(1, 50)
        self.concurrency_spin.setValue(1)
        self.concurrency_spin.setToolTip(
            "Controls how many ticker downloads run at the same time across all providers."
        )
        concurrency_row.addWidget(QtWidgets.QLabel("Concurrent Downloads"))
        concurrency_row.addWidget(self.concurrency_spin)
        concurrency_row.addStretch(1)
        downloads_layout.addLayout(concurrency_row)

        self.resume_chk = QtWidgets.QCheckBox("Resume if previously interrupted")
        self.resume_chk.setChecked(True)
        downloads_layout.addWidget(self.resume_chk)

        self.progress_table = QtWidgets.QTableWidget(0, 6)
        self.progress_table.setHorizontalHeaderLabels(["Ticker", "Status", "Pages", "Rows", "Progress", "Log"])
        self.progress_table.horizontalHeader().setStretchLastSection(True)
        self.progress_table.verticalHeader().setVisible(False)
        self.progress_table.setAlternatingRowColors(True)
        self.progress_table.setObjectName("Panel")
        self.progress_table.setStyleSheet(
            f"""
            QTableWidget {{
                background: {PALETTE['panel2']};
                color: {PALETTE['text']};
                border: 1px solid {PALETTE['border']};
                gridline-color: {PALETTE['grid']};
            }}
            QHeaderView::section {{
                background: {PALETTE['panel']};
                color: {PALETTE['muted']};
                border: 1px solid {PALETTE['border']};
                padding: 6px 8px;
                font-weight: 600;
            }}
            QTableWidget::item {{
                padding: 6px 8px;
            }}
            QTableWidget::item:selected {{
                background: rgba(77, 163, 255, 0.2);
            }}
            """
        )
        self.progress_table.setMinimumHeight(260)
        downloads_layout.addWidget(self.progress_table, 1)
        right_split.addWidget(downloads_panel)

        tasks_panel = QtWidgets.QWidget()
        tasks_panel.setObjectName("Panel")
        tasks_layout = QtWidgets.QVBoxLayout(tasks_panel)
        tasks_layout.setContentsMargins(10, 10, 10, 10)
        tasks_layout.setSpacing(10)
        tasks_title = QtWidgets.QLabel("Scheduled Tasks")
        tasks_title.setObjectName("Title")
        tasks_sub = QtWidgets.QLabel("Review automation schedules, run history, and acquisition catalog links.")
        tasks_sub.setObjectName("Sub")
        tasks_sub.setWordWrap(True)
        tasks_layout.addWidget(tasks_title)
        tasks_layout.addWidget(tasks_sub)

        autostart_row = QtWidgets.QHBoxLayout()
        self.autostart_chk = QtWidgets.QCheckBox("Auto-start scheduler on login (macOS/Windows/Linux)")
        self.autostart_chk.stateChanged.connect(self._toggle_autostart)
        self.autostart_status = QtWidgets.QLabel("Status: unknown")
        self.autostart_status.setObjectName("Sub")
        autostart_row.addWidget(self.autostart_chk)
        autostart_row.addStretch(1)
        autostart_row.addWidget(self.autostart_status)
        tasks_layout.addLayout(autostart_row)
        self._refresh_autostart_status()

        task_actions = QtWidgets.QHBoxLayout()
        self.schedule_btn = QtWidgets.QPushButton("Schedule Task")
        self.schedule_btn.clicked.connect(self._schedule_task)
        self.acquisition_catalog_btn = QtWidgets.QPushButton("Acquisition Catalog")
        self.acquisition_catalog_btn.clicked.connect(self._open_acquisition_catalog)
        task_actions.addWidget(self.schedule_btn)
        task_actions.addWidget(self.acquisition_catalog_btn)
        task_actions.addStretch(1)
        tasks_layout.addLayout(task_actions)

        self.tasks_table = QtWidgets.QTableWidget(0, 13)
        self.tasks_table.setHorizontalHeaderLabels(
            [
                "Created",
                "Universe",
                "Frequency",
                "Tickers",
                "Schedule",
                "Last Run",
                "Next Run",
                "Countdown",
                "Run Status",
                "Log",
                "History",
                "Task State",
                "Actions",
            ]
        )
        self.tasks_table.horizontalHeader().setStretchLastSection(True)
        self.tasks_table.verticalHeader().setVisible(False)
        self.tasks_table.setAlternatingRowColors(True)
        self.tasks_table.setObjectName("Panel")
        self.tasks_table.setStyleSheet(
            f"""
            QTableWidget {{
                background: {PALETTE['panel2']};
                color: {PALETTE['text']};
                border: 1px solid {PALETTE['border']};
                gridline-color: {PALETTE['grid']};
            }}
            QHeaderView::section {{
                background: {PALETTE['panel']};
                color: {PALETTE['muted']};
                border: 1px solid {PALETTE['border']};
                padding: 6px 8px;
                font-weight: 600;
            }}
            QTableWidget::item {{
                padding: 6px 8px;
            }}
            QTableWidget::item:selected {{
                background: rgba(77, 163, 255, 0.2);
            }}
            """
        )
        self.tasks_table.setMinimumHeight(260)
        tasks_layout.addWidget(self.tasks_table, 1)
        right_split.addWidget(tasks_panel)

        content_split.setSizes([620, 820])
        right_split.setSizes([380, 340])
        self._load_tasks()
        self._refresh_tasks_table()
        if not hasattr(self, "tasks_timer"):
            self.tasks_timer = QtCore.QTimer(self)
            self.tasks_timer.setInterval(1000)
            self.tasks_timer.timeout.connect(self._refresh_tasks_table)
            self.tasks_timer.start()
        return panel

    def _metric_card(self, title: str, value: str) -> QtWidgets.QWidget:
        card = QtWidgets.QWidget()
        card.setObjectName("Panel")
        lay = QtWidgets.QVBoxLayout(card)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(4)
        t = QtWidgets.QLabel(title)
        t.setObjectName("Sub")
        v = QtWidgets.QLabel(value)
        v.setObjectName("Title")
        lay.addWidget(t)
        lay.addWidget(v)
        card.value_label = v  # type: ignore[attr-defined]
        return card

    def _load_nasdaq_symbols(self) -> None:
        if NASDAQ_SYMBOLS_PATH.exists():
            try:
                data = NASDAQ_SYMBOLS_PATH.read_text(encoding="utf-8")
            except Exception:
                data = ""
            self.nasdaq_symbols = self._parse_symbols(data)
        else:
            self.nasdaq_symbols = []
        self._update_ticker_summary()

    def _parse_symbols(self, data: str) -> list[str]:
        symbols: list[str] = []
        for line in data.splitlines():
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            if raw.startswith("Symbol|"):
                continue
            if raw.startswith("File Creation Time|"):
                continue
            if "|" in raw:
                sym = raw.split("|", 1)[0].strip()
            else:
                sym = raw
            if sym:
                symbols.append(sym)
        return symbols

    def _update_nasdaq_symbols(self) -> None:
        try:
            NASDAQ_SYMBOLS_PATH.parent.mkdir(parents=True, exist_ok=True)
            if NASDAQ_SYMBOLS_PATH.exists():
                backup_dir = NASDAQ_SYMBOLS_PATH.parent / "nasdaq_symbols_backups"
                backup_dir.mkdir(parents=True, exist_ok=True)
                backup_id = f"{pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
                backup_path = backup_dir / f"all_symbols_{backup_id}.txt"
                backup_path.write_text(NASDAQ_SYMBOLS_PATH.read_text(encoding="utf-8"), encoding="utf-8")
            with urllib.request.urlopen(NASDAQ_LISTED_URL, timeout=20) as resp:
                nasdaq_raw = resp.read().decode("utf-8", errors="ignore")
            with urllib.request.urlopen(OTHER_LISTED_URL, timeout=20) as resp:
                other_raw = resp.read().decode("utf-8", errors="ignore")
            raw_dir = NASDAQ_SYMBOLS_PATH.parent / "nasdaq_symbols_raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            raw_id = f"{pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            (raw_dir / f"nasdaqlisted_{raw_id}.txt").write_text(nasdaq_raw, encoding="utf-8")
            (raw_dir / f"otherlisted_{raw_id}.txt").write_text(other_raw, encoding="utf-8")
            symbols = sorted(set(self._parse_symbols(nasdaq_raw) + self._parse_symbols(other_raw)))
            if not symbols:
                raise RuntimeError("No symbols parsed from NASDAQ list.")
            NASDAQ_SYMBOLS_PATH.write_text("\n".join(symbols) + "\n", encoding="utf-8")
            self.nasdaq_symbols = symbols
            if self.select_all_tickers:
                self.selected_tickers = []
            self._update_ticker_summary()
            QtWidgets.QMessageBox.information(
                self, "Symbols Updated", f"Updated NASDAQ symbols ({len(symbols)} tickers)."
            )
        except Exception as exc:
            self._show_error_dialog("Update Failed", str(exc), details=traceback.format_exc())

    def _update_ticker_summary(self) -> None:
        resolved_source = self._resolved_download_source()
        source_label = provider_display_name(resolved_source)
        provider = get_acquisition_provider(resolved_source)
        settings_status = self._provider_settings_status_text(resolved_source)
        refresh_note = " Force refresh is enabled." if getattr(self, "force_refresh_chk", None) and self.force_refresh_chk.isChecked() else ""
        source_details = (
            f"{source_label} | defaults {provider.default_resolution}, {provider.default_history_window} | "
            f"{provider.description or 'No provider description.'} Settings: {settings_status}.{refresh_note}"
        )
        universe = self._selected_download_universe()
        if universe is not None:
            symbols = [str(sym) for sym in list(universe.get("symbols") or []) if str(sym).strip()]
            self.ticker_summary.setText(f"Universe: {universe.get('name', '')} ({len(symbols)} symbols)")
            if hasattr(self, "download_universe_note"):
                dataset_count = len(list(universe.get("dataset_ids") or []))
                self.download_universe_note.setText(
                    f"Using saved universe '{universe.get('name', '')}' with {len(symbols)} symbol(s) and {dataset_count} dataset binding(s)."
                )
            if hasattr(self, "download_source_note"):
                preferred = str(universe.get("source_preference") or "").strip()
                if preferred:
                    self.download_source_note.setText(
                        f"Resolved source: {source_details}. Universe preference is {provider_display_name(preferred)}."
                    )
                else:
                    self.download_source_note.setText(
                        f"Resolved source: {source_details}. This universe does not set a preferred source."
                    )
            return
        if hasattr(self, "download_universe_note"):
            self.download_universe_note.setText("Manual ticker selection is active.")
        if hasattr(self, "download_source_note"):
            if self._selected_download_source_choice():
                self.download_source_note.setText(f"Manual source selection is active: {source_details}.")
            else:
                self.download_source_note.setText(
                    f"Automatic source selection is using the default provider: {source_details}."
                )
        if self.select_all_tickers:
            self.ticker_summary.setText(f"All NASDAQ ({len(self.nasdaq_symbols)})")
            return
        if not self.selected_tickers:
            self.ticker_summary.setText("")
            return
        shown = ", ".join(self.selected_tickers[:6])
        extra = f" (+{len(self.selected_tickers) - 6})" if len(self.selected_tickers) > 6 else ""
        self.ticker_summary.setText(f"{shown}{extra}")

    def _select_all_tickers(self) -> None:
        if not self.nasdaq_symbols:
            QtWidgets.QMessageBox.warning(self, "Symbols missing", f"Symbols file not found: {NASDAQ_SYMBOLS_PATH}")
            return
        self.download_universe_id = ""
        if hasattr(self, "download_universe_combo"):
            self.download_universe_combo.blockSignals(True)
            self.download_universe_combo.setCurrentIndex(0)
            self.download_universe_combo.blockSignals(False)
        self.select_all_tickers = True
        self.selected_tickers = []
        self._update_ticker_summary()

    def _open_ticker_picker(self) -> None:
        if not self.nasdaq_symbols:
            QtWidgets.QMessageBox.warning(self, "Symbols missing", f"Symbols file not found: {NASDAQ_SYMBOLS_PATH}")
            return
        dlg = TickerPickerDialog(self.nasdaq_symbols, set(self.selected_tickers), self)
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.download_universe_id = ""
            if hasattr(self, "download_universe_combo"):
                self.download_universe_combo.blockSignals(True)
                self.download_universe_combo.setCurrentIndex(0)
                self.download_universe_combo.blockSignals(False)
            self.select_all_tickers = False
            self.selected_tickers = dlg.selected
            self._update_ticker_summary()

    def _on_download_universe_changed(self) -> None:
        if not hasattr(self, "download_universe_combo"):
            return
        self.download_universe_id = str(self.download_universe_combo.currentData() or "")
        self._update_ticker_summary()

    def _selected_download_symbols(self) -> list[str]:
        universe = self._selected_download_universe()
        if universe is not None:
            return [
                str(symbol).upper()
                for symbol in list(universe.get("symbols") or [])
                if str(symbol).strip()
            ]
        if self.select_all_tickers:
            return list(self.nasdaq_symbols)
        return list(self.selected_tickers)

    def _schedule_task(self) -> None:
        symbols = self._selected_download_symbols()
        if not symbols:
            QtWidgets.QMessageBox.warning(self, "No tickers", "Select at least one ticker or a universe with symbols.")
            return
        source = self._resolved_download_source()
        provider = get_acquisition_provider(source)
        frequency = self.schedule_combo.currentText()
        start_time = self.schedule_time.time().toString("HH:mm")
        days = [chk.text() for chk in self.weekday_checks if chk.isChecked()]
        weeks = [chk.text() for chk in self.week_of_month_checks if chk.isChecked()]
        months = [chk.text() for chk in self.month_checks if chk.isChecked()]
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        schedule = {
            "frequency": frequency,
            "time": start_time,
            "days": days,
            "weeks": weeks,
            "months": months,
        }
        payload = {
            "symbols": symbols,
            "universe_id": self.download_universe_id,
            "universe_name": (self._selected_download_universe() or {}).get("name", ""),
            "source": source,
            "resolution": provider.default_resolution,
            "history": provider.default_history_window,
            "force_refresh": bool(self.force_refresh_chk.isChecked()),
            "schedule": schedule,
        }
        rc = ResultCatalog(self.catalog.db_path)
        rc.upsert_task(task_id, payload, schedule, status="active")
        self._load_tasks()
        self._refresh_tasks_table()
        scope_label = (
            f"universe '{(self._selected_download_universe() or {}).get('name', '')}'"
            if self._selected_download_universe() is not None
            else f"{len(symbols)} tickers"
        )
        QtWidgets.QMessageBox.information(
            self,
            "Scheduled",
            f"Scheduled {scope_label} ({frequency}) using {provider_display_name(source)}."
            f"{' Force refresh is enabled.' if self.force_refresh_chk.isChecked() else ''}",
        )

    def _load_tasks(self) -> None:
        self.scheduled_tasks = self.catalog.load_scheduled_tasks()

    def _save_tasks(self) -> None:
        # Deprecated: tasks are stored in SQLite.
        return

    def _refresh_tasks_table(self) -> None:
        self.tasks_table.setRowCount(0)
        now = pd.Timestamp.utcnow()
        for idx, task in enumerate(self.scheduled_tasks):
            row = self.tasks_table.rowCount()
            self.tasks_table.insertRow(row)
            created = self._format_timestamp(task.get("created_at"))
            schedule = task.get("schedule") or {}
            frequency = schedule.get("frequency", task.get("frequency", ""))
            symbols = (task.get("symbols") or {}).get("symbols") if isinstance(task.get("symbols"), dict) else task.get("symbols", [])
            if symbols is None:
                symbols = task.get("symbols", [])
            universe_name = (
                str((task.get("symbols") or {}).get("universe_name", "")).strip()
                if isinstance(task.get("symbols"), dict)
                else ""
            )
            schedule_desc = self._format_schedule(schedule)
            last_run = self._format_timestamp(task.get("last_run_at"))
            next_run = self._parse_timestamp(task.get("next_run_at")) or self._compute_next_run(schedule, now)
            countdown = self._format_countdown(next_run, now) if next_run else "—"
            task_state = str(task.get("status") or "active")
            status = task.get("last_run_status") or task_state or "—"
            message = task.get("last_run_message") or ""
            if message:
                status = f"{status}: {message}"
            self.tasks_table.setItem(row, 0, QtWidgets.QTableWidgetItem(created))
            self.tasks_table.setItem(row, 1, QtWidgets.QTableWidgetItem(universe_name or "—"))
            self.tasks_table.setItem(row, 2, QtWidgets.QTableWidgetItem(frequency))
            self.tasks_table.setItem(row, 3, QtWidgets.QTableWidgetItem(str(len(symbols))))
            self.tasks_table.setItem(row, 4, QtWidgets.QTableWidgetItem(schedule_desc))
            self.tasks_table.setItem(row, 5, QtWidgets.QTableWidgetItem(last_run))
            self.tasks_table.setItem(row, 6, QtWidgets.QTableWidgetItem(self._format_timestamp(next_run)))
            self.tasks_table.setItem(row, 7, QtWidgets.QTableWidgetItem(countdown))
            self.tasks_table.setItem(row, 8, QtWidgets.QTableWidgetItem(status))
            log_btn = QtWidgets.QPushButton("Log")
            log_btn.clicked.connect(lambda _, i=idx: self._open_task_log(i))
            self.tasks_table.setCellWidget(row, 9, log_btn)
            history_btn = QtWidgets.QPushButton("History")
            history_btn.clicked.connect(lambda _, i=idx: self._open_task_history(i))
            self.tasks_table.setCellWidget(row, 10, history_btn)
            self.tasks_table.setItem(row, 11, QtWidgets.QTableWidgetItem(task_state))
            actions_wrap = QtWidgets.QWidget()
            actions_layout = QtWidgets.QHBoxLayout(actions_wrap)
            actions_layout.setContentsMargins(4, 0, 4, 0)
            actions_layout.setSpacing(6)
            toggle_btn = QtWidgets.QPushButton("Pause" if task_state == "active" else "Activate")
            toggle_btn.clicked.connect(lambda _, i=idx: self._toggle_task_status(i))
            edit_btn = QtWidgets.QPushButton("Edit")
            edit_btn.clicked.connect(lambda _, i=idx: self._edit_task(i))
            remove_btn = QtWidgets.QPushButton("Delete")
            remove_btn.clicked.connect(lambda _, i=idx: self._unschedule_task(i))
            actions_layout.addWidget(toggle_btn)
            actions_layout.addWidget(edit_btn)
            actions_layout.addWidget(remove_btn)
            actions_layout.addStretch(1)
            self.tasks_table.setCellWidget(row, 12, actions_wrap)

    def _toggle_task_status(self, index: int) -> None:
        if index < 0 or index >= len(self.scheduled_tasks):
            return
        task = self.scheduled_tasks[index]
        task_id = str(task.get("task_id") or "")
        if not task_id:
            return
        current = str(task.get("status") or "active")
        next_status = "paused" if current == "active" else "active"
        ResultCatalog(self.catalog.db_path).update_task_status(task_id, next_status)
        self._load_tasks()
        self._refresh_tasks_table()

    def _edit_task(self, index: int) -> None:
        if index < 0 or index >= len(self.scheduled_tasks):
            return
        task = dict(self.scheduled_tasks[index] or {})
        dlg = ScheduledTaskEditorDialog(task, self.universes, self.nasdaq_symbols, self)
        if dlg.exec() != int(QtWidgets.QDialog.DialogCode.Accepted):
            return
        task_id = str(task.get("task_id") or "")
        if not task_id:
            return
        payload, schedule = dlg.result_payload()
        status = str(task.get("status") or "active")
        rc = ResultCatalog(self.catalog.db_path)
        rc.upsert_task(task_id, payload, schedule, status=status)
        next_run = self._compute_next_run(schedule, pd.Timestamp.utcnow())
        rc.update_task_run_info(
            task_id,
            task.get("last_run_at"),
            task.get("last_run_status"),
            task.get("last_run_message"),
            next_run.isoformat() if next_run is not None else None,
        )
        self._load_tasks()
        self._refresh_tasks_table()

    def _open_task_history(self, index: int) -> None:
        if index < 0 or index >= len(self.scheduled_tasks):
            return
        task = dict(self.scheduled_tasks[index] or {})
        task_id = str(task.get("task_id") or "")
        if not task_id:
            QtWidgets.QMessageBox.information(self, "Task history", "No task id is available for the selected scheduled task.")
            return
        selected_universe_id = (
            str((task.get("symbols") or {}).get("universe_id") or "")
            if isinstance(task.get("symbols"), dict)
            else ""
        )
        dlg = AcquisitionCatalogDialog(
            self.catalog,
            self.universes,
            self,
            selected_universe_id=selected_universe_id,
            selected_task_id=task_id,
        )
        if hasattr(dlg, "tabs"):
            dlg.tabs.setCurrentIndex(3)
        dlg.exec()

    def _unschedule_task(self, index: int) -> None:
        if index < 0 or index >= len(self.scheduled_tasks):
            return
        removed = self.scheduled_tasks.pop(index)
        task_id = removed.get("task_id")
        if task_id:
            rc = ResultCatalog(self.catalog.db_path)
            rc.delete_task(task_id)
        self._refresh_tasks_table()
        QtWidgets.QMessageBox.information(
            self,
            "Task removed",
            f"Unschedule task ({(removed.get('schedule') or {}).get('frequency', '')})",
        )

    def _open_task_log(self, index: int) -> None:
        if index < 0 or index >= len(self.scheduled_tasks):
            return
        task = self.scheduled_tasks[index]
        task_id = task.get("task_id")
        if not task_id:
            QtWidgets.QMessageBox.information(self, "Log", "No task id available.")
            return
        log_dir = Path("data") / "scheduler_logs" / task_id
        if not log_dir.exists():
            QtWidgets.QMessageBox.information(self, "Log", "No logs available yet.")
            return
        logs = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not logs:
            QtWidgets.QMessageBox.information(self, "Log", "No logs available yet.")
            return
        path = logs[0]
        try:
            content = path.read_text(encoding="utf-8")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Log", f"Unable to read log: {exc}")
            return
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(f"Task Log {task_id}")
        dlg.resize(900, 600)
        layout = QtWidgets.QVBoxLayout(dlg)
        text = QtWidgets.QPlainTextEdit()
        text.setReadOnly(True)
        text.setPlainText(content)
        layout.addWidget(text)
        dlg.exec()

    def _refresh_autostart_status(self) -> None:
        enabled = False
        try:
            enabled = self._is_autostart_enabled()
        except Exception:
            enabled = False
        self.autostart_chk.blockSignals(True)
        self.autostart_chk.setChecked(enabled)
        self.autostart_chk.blockSignals(False)
        self.autostart_status.setText("Status: enabled" if enabled else "Status: disabled")

    def _toggle_autostart(self) -> None:
        want = self.autostart_chk.isChecked()
        try:
            if want:
                self._enable_autostart()
            else:
                self._disable_autostart()
        except Exception as exc:
            self._show_error_dialog("Auto-start error", str(exc), details=traceback.format_exc())
        self._refresh_autostart_status()

    def _is_autostart_enabled(self) -> bool:
        if sys.platform == "darwin":
            plist = Path.home() / "Library" / "LaunchAgents" / "com.quantdata.scheduler.plist"
            return plist.exists()
        if sys.platform.startswith("win"):
            try:
                result = subprocess.run(
                    ["schtasks", "/Query", "/TN", "QuantDataScheduler"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                return result.returncode == 0
            except Exception:
                return False
        # linux
        unit = Path.home() / ".config" / "systemd" / "user" / "quantdata-scheduler.service"
        return unit.exists()

    def _enable_autostart(self) -> None:
        if sys.platform == "darwin":
            plist = Path.home() / "Library" / "LaunchAgents" / "com.quantdata.scheduler.plist"
            plist.parent.mkdir(parents=True, exist_ok=True)
            content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.quantdata.scheduler</string>
  <key>ProgramArguments</key>
  <array>
    <string>{sys.executable}</string>
    <string>{SCHEDULER_SCRIPT.resolve()}</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>WorkingDirectory</key><string>{Path.cwd()}</string>
  <key>StandardOutPath</key><string>{(Path.cwd()/ "data" / "scheduler_stdout.log").resolve()}</string>
  <key>StandardErrorPath</key><string>{(Path.cwd()/ "data" / "scheduler_stderr.log").resolve()}</string>
</dict>
</plist>
"""
            plist.write_text(content, encoding="utf-8")
            subprocess.run(["launchctl", "load", "-w", str(plist)], check=False)
            return
        if sys.platform.startswith("win"):
            subprocess.run(
                [
                    "schtasks",
                    "/Create",
                    "/TN",
                    "QuantDataScheduler",
                    "/SC",
                    "ONLOGON",
                    "/TR",
                    f"\"{sys.executable}\" \"{SCHEDULER_SCRIPT.resolve()}\"",
                ],
                check=False,
            )
            return
        # linux systemd --user
        unit_dir = Path.home() / ".config" / "systemd" / "user"
        unit_dir.mkdir(parents=True, exist_ok=True)
        unit_path = unit_dir / "quantdata-scheduler.service"
        unit_text = f"""[Unit]
Description=Quant Data Scheduler

[Service]
ExecStart={sys.executable} {SCHEDULER_SCRIPT.resolve()}
WorkingDirectory={Path.cwd()}
Restart=always

[Install]
WantedBy=default.target
"""
        unit_path.write_text(unit_text, encoding="utf-8")
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
        subprocess.run(["systemctl", "--user", "enable", "--now", "quantdata-scheduler.service"], check=False)

    def _disable_autostart(self) -> None:
        if sys.platform == "darwin":
            plist = Path.home() / "Library" / "LaunchAgents" / "com.quantdata.scheduler.plist"
            if plist.exists():
                subprocess.run(["launchctl", "unload", "-w", str(plist)], check=False)
                plist.unlink(missing_ok=True)
            return
        if sys.platform.startswith("win"):
            subprocess.run(["schtasks", "/Delete", "/TN", "QuantDataScheduler", "/F"], check=False)
            return
        unit_path = Path.home() / ".config" / "systemd" / "user" / "quantdata-scheduler.service"
        if unit_path.exists():
            subprocess.run(["systemctl", "--user", "disable", "--now", "quantdata-scheduler.service"], check=False)
            unit_path.unlink(missing_ok=True)
    def _format_schedule(self, schedule: dict) -> str:
        time_str = schedule.get("time", "00:00")
        days = ",".join(schedule.get("days", []))
        weeks = ",".join(schedule.get("weeks", []))
        months = ",".join(schedule.get("months", []))
        return f"{time_str} | D:{days or 'all'} W:{weeks or 'all'} M:{months or 'all'}"

    def _parse_timestamp(self, ts: str | pd.Timestamp | None) -> pd.Timestamp | None:
        if not ts:
            return None
        try:
            return pd.to_datetime(ts, utc=True)
        except Exception:
            return None

    def _format_timestamp(self, ts: str | pd.Timestamp | None) -> str:
        if not ts:
            return "—"
        try:
            stamp = pd.to_datetime(ts, utc=True)
            stamp = stamp.tz_convert("America/New_York")
            return stamp.strftime("%Y-%m-%d %I:%M %p ET")
        except Exception:
            return str(ts)

    def _format_countdown(self, target: pd.Timestamp | None, now: pd.Timestamp) -> str:
        if target is None:
            return "—"
        delta = target - now
        if delta.total_seconds() < 0:
            return "—"
        total = int(delta.total_seconds())
        days = total // 86400
        hours = (total % 86400) // 3600
        minutes = (total % 3600) // 60
        seconds = total % 60
        return f"{days}d {hours:02d}:{minutes:02d}:{seconds:02d}"

    def _compute_next_run(self, schedule: dict, now: pd.Timestamp) -> pd.Timestamp | None:
        freq = (schedule.get("frequency") or "Nightly").lower()
        time_str = schedule.get("time", "00:00")
        try:
            hour, minute = [int(x) for x in time_str.split(":")]
        except Exception:
            hour, minute = 0, 0
        days = schedule.get("days", [])
        weeks = schedule.get("weeks", [])
        months = schedule.get("months", [])

        weekday_map = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}
        month_map = {m: i + 1 for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}
        valid_weekdays = {weekday_map[d] for d in days if d in weekday_map} if days else set(range(7))
        valid_months = {month_map[m] for m in months if m in month_map} if months else set(range(1, 13))
        valid_weeks = set()
        for w in weeks or ["1"]:
            if w == "Last":
                valid_weeks.add("Last")
            else:
                try:
                    valid_weeks.add(int(w))
                except Exception:
                    continue

        start = now + pd.Timedelta(minutes=1)
        for i in range(0, 365 * 2):
            day = (start.normalize() + pd.Timedelta(days=i)).tz_convert("UTC")
            candidate = day.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if candidate <= now:
                continue
            if candidate.weekday() not in valid_weekdays:
                continue
            if candidate.month not in valid_months:
                continue
            if freq == "weekly" and candidate.weekday() not in valid_weekdays:
                continue
            if freq == "monthly":
                week_number = ((candidate.day - 1) // 7) + 1
                last_week = (candidate + pd.offsets.MonthEnd(0)).day
                is_last = candidate.day + 7 > last_week
                if "Last" in valid_weeks and is_last:
                    return candidate
                if any(isinstance(w, int) and w == week_number for w in valid_weeks):
                    return candidate
                continue
            if freq == "nightly":
                return candidate
            if freq == "weekly":
                return candidate
        return None

    def _create_download_log_path(self, ticker: str) -> Path:
        DOWNLOAD_LOG_DIR.mkdir(parents=True, exist_ok=True)
        stamp = pd.Timestamp.now("UTC").strftime("%Y%m%d_%H%M%S")
        return DOWNLOAD_LOG_DIR / f"{ticker.upper()}_{stamp}_{uuid.uuid4().hex[:8]}.log"

    def _append_download_log(self, log_path: Path | None, text: str) -> None:
        if log_path is None or not text:
            return
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(text)

    def _download_meta(self, proc: QtCore.QProcess | None) -> dict | None:
        if proc is None:
            return None
        return self.download_proc_meta.get(id(proc))

    def _open_download_log(self, ticker: str) -> None:
        info = self.download_progress_rows.get(ticker)
        if not info:
            QtWidgets.QMessageBox.information(self, "Log", f"No download row exists for {ticker}.")
            return
        log_path = info.get("log_path")
        if not log_path:
            QtWidgets.QMessageBox.information(self, "Log", f"No log path recorded for {ticker}.")
            return
        path = Path(log_path)
        if not path.exists():
            QtWidgets.QMessageBox.information(self, "Log", f"No log available yet for {ticker}.")
            return
        try:
            content = path.read_text(encoding="utf-8")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Log", f"Unable to read log: {exc}")
            return
        dlg = DashboardDialog(self)
        dlg.setWindowTitle(f"Download Log {ticker}")
        dlg.resize(1080, 760)
        dlg.setObjectName("Panel")
        layout = QtWidgets.QVBoxLayout(dlg)
        subtitle = QtWidgets.QLabel(str(path))
        subtitle.setObjectName("Sub")
        subtitle.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(subtitle)
        text = QtWidgets.QPlainTextEdit()
        text.setObjectName("Panel")
        text.setReadOnly(True)
        text.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        text.setPlainText(content)
        layout.addWidget(text)
        dlg.exec()

    def _open_acquisition_catalog(
        self,
        *,
        selected_universe_id: str = "",
        selected_task_id: str = "",
        selected_dataset_id: str = "",
        initial_tab: int | None = None,
    ) -> None:
        dlg = AcquisitionCatalogDialog(
            self.catalog,
            self.universes,
            self,
            selected_universe_id=selected_universe_id,
            selected_task_id=selected_task_id,
            selected_dataset_id=selected_dataset_id,
        )
        if initial_tab is not None and hasattr(dlg, "tabs"):
            dlg.tabs.setCurrentIndex(max(0, min(initial_tab, dlg.tabs.count() - 1)))
        dlg.exec()

    def _prompt_for_dataset_id(self, dataset_ids: Sequence[str], *, title: str) -> str:
        clean = [str(item).strip() for item in dataset_ids if str(item).strip()]
        if not clean:
            return ""
        if len(clean) == 1:
            return clean[0]
        dlg = DatasetPickerDialog(clean, clean[0], self, title=title)
        if dlg.exec() != int(QtWidgets.QDialog.DialogCode.Accepted):
            return ""
        return dlg.selected_dataset()

    def _open_dataset_acquisition_detail(self, dataset_ids: Sequence[str], *, title: str = "Dataset Acquisition Detail") -> None:
        clean = [str(item).strip() for item in dataset_ids if str(item).strip()]
        if not clean:
            QtWidgets.QMessageBox.information(self, "No dataset", "Choose at least one dataset first.")
            return
        initial_dataset_id = self._prompt_for_dataset_id(clean, title=title)
        if not initial_dataset_id:
            return
        dlg = AcquisitionDatasetDetailDialog(
            self.catalog,
            self.universes,
            clean,
            self,
            initial_dataset_id=initial_dataset_id,
        )
        dlg.setWindowTitle(title)
        dlg.exec()

    def _open_current_dataset_acquisition_detail(self) -> None:
        dataset_id = str(self.dataset_combo.currentText() or "").strip() if hasattr(self, "dataset_combo") else ""
        if not dataset_id:
            dataset_ids = self._selected_study_dataset_ids(manual_only=False)
            self._open_dataset_acquisition_detail(dataset_ids, title="Study Dataset Acquisition Detail")
            return
        self._open_dataset_acquisition_detail([dataset_id], title="Current Dataset Acquisition Detail")

    def _open_selected_study_dataset_acquisition_detail(self) -> None:
        dataset_ids = self._selected_study_dataset_ids(manual_only=False)
        if not dataset_ids:
            dataset_id = str(self.dataset_combo.currentText() or "").strip() if hasattr(self, "dataset_combo") else ""
            dataset_ids = [dataset_id] if dataset_id else []
        self._open_dataset_acquisition_detail(dataset_ids, title="Study Dataset Acquisition Detail")

    def _open_current_universe_dataset_acquisition_detail(self) -> None:
        self._open_dataset_acquisition_detail(
            list(self._editing_universe_dataset_ids),
            title="Universe Dataset Acquisition Detail",
        )

    def _open_selected_universe_acquisition_catalog(self) -> None:
        universe_id = str(self._editing_universe_id or "").strip()
        if not universe_id:
            QtWidgets.QMessageBox.information(self, "No universe selected", "Select a saved universe first.")
            return
        self._open_acquisition_catalog(selected_universe_id=universe_id)

    def _open_study_universe_acquisition_catalog(self) -> None:
        universe_id = str(self.study_universe_id or "").strip()
        if not universe_id:
            dataset_ids = self._selected_study_dataset_ids(manual_only=False)
            if len(dataset_ids) == 1:
                self._open_acquisition_catalog(selected_dataset_id=dataset_ids[0], initial_tab=0)
                return
            QtWidgets.QMessageBox.information(
                self,
                "No study universe",
                "Choose a saved study universe first, or narrow the study to a single dataset to open its acquisition view.",
            )
            return
        self._open_acquisition_catalog(selected_universe_id=universe_id)

    def _auto_ingest_downloaded_csv(
        self,
        *,
        ticker: str,
        out_path: Path,
        source: str,
        history_window: str,
        resolution: str,
        merge_with_existing: bool = False,
    ) -> tuple[str, object | None, str | None]:
        dataset_id = build_download_dataset_id(
            ticker,
            source=source,
            history_window=history_window,
            resolution=resolution,
        )
        try:
            artifact = ingest_csv_to_store(out_path, dataset_id=dataset_id, merge_existing=merge_with_existing)
            return "ingested", artifact, None
        except Exception as exc:
            return "ingest_error", None, str(exc)

    @staticmethod
    def _decision_request_windows(decision) -> list[tuple[str | None, str | None]]:
        windows = [(str(start or "").strip() or None, str(end or "").strip() or None) for start, end in list(decision.request_windows or [])]
        if windows:
            return windows
        return [(decision.request_start, decision.request_end)]

    @staticmethod
    def _decision_secondary_request_windows(decision) -> list[tuple[str | None, str | None]]:
        return [
            (str(start or "").strip() or None, str(end or "").strip() or None)
            for start, end in list(decision.secondary_request_windows or [])
        ]

    def _launch_download_process_for_meta(self, meta: dict) -> None:
        source = str(meta.get("source") or DEFAULT_ACQUISITION_PROVIDER)
        ticker = str(meta.get("ticker") or "")
        request_windows = list(meta.get("request_windows") or [(None, None)])
        window_index = int(meta.get("window_index") or 0)
        window_total = len(request_windows)
        window_start, window_end = request_windows[window_index]
        meta["current_window_start"] = window_start
        meta["current_window_end"] = window_end
        proc = QtCore.QProcess(self)
        proc.setProgram(sys.executable)
        extra_args: list[str] = []
        if window_start:
            extra_args.extend(["--start", window_start])
        if window_end:
            extra_args.extend(["--end", window_end])
        args = build_provider_fetch_command(
            source,
            python_executable=sys.executable,
            ticker=ticker,
            out_path=Path(str(meta.get("out_path") or "")),
            resolution=str(meta.get("resolution") or ""),
            history_window=str(meta.get("history_window") or ""),
            progress=True,
            resume=self.resume_chk.isChecked(),
            extra_args=extra_args or None,
        )
        proc.setArguments(args[1:])
        proc.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
        proc.readyReadStandardOutput.connect(self._handle_download_output)
        proc.finished.connect(self._download_finished)
        proc.errorOccurred.connect(self._download_process_error)
        process_env = QtCore.QProcessEnvironment.systemEnvironment()
        for key, value in self._provider_runtime_environment(source).items():
            process_env.insert(key, value)
        proc.setProcessEnvironment(process_env)
        self.download_proc_meta[id(proc)] = meta
        launch_text = (
            f"[{pd.Timestamp.now('UTC').isoformat()}] Launching download for {ticker}"
            f" window {window_index + 1}/{window_total}\n"
            f"Request Window: {window_start or 'provider default'} -> {window_end or 'provider default'}\n"
            f"Command: {' '.join(args)}\n\n"
        )
        self._append_download_log(Path(str(meta["log_path"])), launch_text)
        proc.start()
        self.download_proc = proc
        self.download_procs.append(proc)
        self.download_status.setText(
            f"Downloading {ticker} window {window_index + 1}/{window_total}…"
        )
        self._update_progress_row(
            ticker,
            status="running",
            tooltip=(
                f"Window {window_index + 1}/{window_total}\n"
                f"Request: {window_start or 'provider default'} -> {window_end or 'provider default'}\n"
                f"Log: {meta['log_path']}"
            ),
        )

    def _record_acquisition_attempt_result(
        self,
        *,
        ticker: str,
        meta: dict,
        status: str,
        summary: str | None = None,
        parquet_path: str | None = None,
        coverage_start: str | None = None,
        coverage_end: str | None = None,
        bar_count: int | None = None,
        ingested: bool = False,
    ) -> None:
        if not self.current_acquisition_run_id or not ticker:
            return
        source = str(meta.get("source") or DEFAULT_ACQUISITION_PROVIDER)
        provider = get_acquisition_provider(source)
        dataset_id = build_download_dataset_id(
            ticker,
            source=source,
            history_window=str(meta.get("history_window") or provider.default_history_window),
            resolution=str(meta.get("resolution") or provider.default_resolution),
        )
        ResultCatalog(self.catalog.db_path).record_acquisition_attempt(
            attempt_id=f"{self.current_acquisition_run_id}_{len(self.current_acquisition_attempts) + 1:04d}",
            acquisition_run_id=self.current_acquisition_run_id,
            seq=len(self.current_acquisition_attempts) + 1,
            source=source,
            symbol=ticker,
            dataset_id=dataset_id,
            status=status,
            started_at=str(meta.get("attempt_started_at") or pd.Timestamp.utcnow().isoformat()),
            finished_at=pd.Timestamp.utcnow().isoformat(),
            csv_path=str(meta.get("out_path") or ""),
            parquet_path=parquet_path,
            coverage_start=coverage_start,
            coverage_end=coverage_end,
            bar_count=bar_count,
            ingested=ingested,
            error_message=summary or None,
            log_path=str(meta.get("log_path") or ""),
            universe_id=self.download_universe_id or None,
            resolution=str(meta.get("resolution") or provider.default_resolution),
            history_window=str(meta.get("history_window") or provider.default_history_window),
        )
        self.current_acquisition_attempts[ticker] = status
        if ingested and parquet_path:
            self._refresh_dataset_options()

    def _finish_current_acquisition_run(self, *, stopped: bool = False) -> None:
        run_id = self.current_acquisition_run_id
        if not run_id:
            return
        attempts = list(self.current_acquisition_attempts.values())
        success_count = len([status for status in attempts if status in {"ingested", "downloaded", "gap_filled"}])
        failed_count = len([status for status in attempts if status in {"download_error", "failed", "ingest_error", "gap_fill_error"}])
        ingested_count = len([status for status in attempts if status in {"ingested", "gap_filled"}])
        skipped_count = len([status for status in attempts if status == "skipped"])
        if stopped:
            status = "stopped"
        elif skipped_count and not success_count and not failed_count:
            status = "skipped"
        elif failed_count and success_count:
            status = "partial"
        elif failed_count and not success_count:
            status = "failed"
        else:
            status = "success"
        ResultCatalog(self.catalog.db_path).finish_acquisition_run(
            run_id,
            finished_at=pd.Timestamp.utcnow().isoformat(),
            status=status,
            success_count=success_count,
            failed_count=failed_count,
            ingested_count=ingested_count,
            notes=(
                f"{success_count} succeeded, {failed_count} failed, {ingested_count} ingested, {skipped_count} skipped."
                if attempts
                else ("Stopped before any ticker completed." if stopped else "No recorded attempts.")
            ),
        )
        self.current_acquisition_run_id = None
        self.current_acquisition_attempts = {}

    def _start_download(self) -> None:
        if self.download_procs:
            QtWidgets.QMessageBox.information(self, "Download running", "A download process is already running.")
            return
        symbols = self._selected_download_symbols()
        if not symbols:
            QtWidgets.QMessageBox.warning(self, "No tickers", "Select at least one ticker or a universe with symbols.")
            return
        source = self._resolved_download_source()
        provider = get_acquisition_provider(source)
        self.download_queue = list(symbols)
        self.download_progress_rows = {}
        self.download_proc_meta = {}
        self.progress_table.setRowCount(0)
        self.download_paused = False
        self.download_progress.setVisible(True)
        self.download_progress.setRange(0, 0)
        self.download_status.setText("Starting downloads…")
        self.download_procs = []
        self.current_acquisition_run_id = f"acq_{uuid.uuid4().hex[:8]}"
        self.current_acquisition_attempts = {}
        selected_universe = self._selected_download_universe() or {}
        ResultCatalog(self.catalog.db_path).start_acquisition_run(
            acquisition_run_id=self.current_acquisition_run_id,
            trigger_type="interactive_download",
            source=source,
            universe_id=str(selected_universe.get("universe_id", "") or "") or None,
            universe_name=str(selected_universe.get("name", "") or "") or None,
            started_at=pd.Timestamp.utcnow().isoformat(),
            status="running",
            symbol_count=len(self.download_queue),
            notes=f"Interactive download request launched from Automate tab using {provider_display_name(source)}.",
        )
        for _ in range(min(self.concurrency_spin.value(), len(self.download_queue))):
            self._start_next_download()

    def _start_next_download(self) -> None:
        if not self.download_queue:
            if not self.download_procs:
                self._finish_current_acquisition_run()
                self.download_status.setText("Downloads complete.")
                self.download_progress.setVisible(False)
                self.download_active_ticker = None
            return
        ticker = self.download_queue.pop(0)
        self.download_active_ticker = ticker
        source = self._resolved_download_source()
        provider = get_acquisition_provider(source)
        out_path = build_download_csv_path(
            ticker,
            source=source,
            history_window=provider.default_history_window,
            resolution=provider.default_resolution,
        )
        log_path = self._create_download_log_path(ticker)
        self._ensure_progress_row(ticker, log_path)
        decision = decide_acquisition_policy(
            ticker,
            source=source,
            resolution=provider.default_resolution,
            history_window=provider.default_history_window,
            catalog=ResultCatalog(self.catalog.db_path),
            force_refresh=bool(self.force_refresh_chk.isChecked()),
        )
        meta = {
            "ticker": ticker,
            "log_path": str(log_path),
            "buffer": "",
            "last_error": "",
            "rows": int(decision.bar_count or 0),
            "out_path": str(out_path),
            "attempt_started_at": pd.Timestamp.utcnow().isoformat(),
            "source": source,
            "history_window": provider.default_history_window,
            "resolution": provider.default_resolution,
            "policy_reason": decision.reason,
            "policy_action": decision.action,
            "policy_plan_type": decision.plan_type,
            "request_start": decision.request_start,
            "request_end": decision.request_end,
            "request_windows": self._decision_request_windows(decision),
            "secondary_request_windows": self._decision_secondary_request_windows(decision),
            "merge_with_existing": bool(decision.merge_with_existing),
            "window_index": 0,
            "secondary_source": decision.secondary_source,
            "secondary_dataset_id": decision.secondary_dataset_id,
            "parity_state": decision.parity_state,
            "parity_overlap_bars": decision.parity_overlap_bars,
            "parity_close_mae": decision.parity_close_mae,
            "parity_close_mean_abs_bps": decision.parity_close_mean_abs_bps,
        }
        launch_text = (
            f"[{pd.Timestamp.now('UTC').isoformat()}] Acquisition policy for {ticker}: {decision.action}\n"
            f"Plan: {decision.plan_type}\n"
            f"Reason: {decision.reason}\n"
            f"Request Window: {decision.request_start or 'provider default'} -> {decision.request_end or 'provider default'}\n"
            f"Merge With Existing Dataset: {'yes' if decision.merge_with_existing else 'no'}\n"
            f"{f'Request Windows: {list(decision.request_windows)}\\n' if decision.request_windows else ''}"
            f"{f'Secondary Repair Windows: {list(decision.secondary_request_windows)}\\n' if decision.secondary_request_windows else ''}"
            f"{f'Secondary Source: {decision.secondary_source} / {decision.secondary_dataset_id}\\n' if decision.secondary_dataset_id else ''}"
            f"{f'Parity: {decision.parity_state} ({decision.parity_overlap_bars} overlap bars, mean abs {decision.parity_close_mean_abs_bps:.2f} bps)\\n' if decision.secondary_dataset_id and decision.parity_close_mean_abs_bps is not None else ''}"
        )
        self._append_download_log(log_path, launch_text + "\n")
        if decision.action == ACQUISITION_ACTION_SKIP_FRESH:
            tooltip = (
                f"{decision.reason}\nPlan: {decision.plan_type}\n"
                f"Coverage: {decision.coverage_start or '—'} → {decision.coverage_end or '—'}\nLog: {log_path}"
            )
            self.download_status.setText(f"Skipping {ticker}: dataset is already fresh.")
            self._update_progress_row(ticker, status="skipped", rows=decision.bar_count, done=True, tooltip=tooltip)
            self._record_acquisition_attempt_result(
                ticker=ticker,
                meta=meta,
                status="skipped",
                summary=decision.reason,
                parquet_path=decision.parquet_path,
                coverage_start=decision.coverage_start,
                coverage_end=decision.coverage_end,
                bar_count=decision.bar_count,
                ingested=decision.ingested,
            )
            QtCore.QTimer.singleShot(0, self._start_next_download)
            return
        if decision.action == ACQUISITION_ACTION_INGEST_EXISTING:
            ingest_status, artifact, ingest_error = self._auto_ingest_downloaded_csv(
                ticker=ticker,
                out_path=Path(decision.csv_path),
                source=source,
                history_window=provider.default_history_window,
                resolution=provider.default_resolution,
                merge_with_existing=bool(decision.merge_with_existing),
            )
            if artifact is not None:
                tooltip = f"Ingested existing CSV → {artifact.dataset_id}\nPlan: {decision.plan_type}\nLog: {log_path}"
                self.download_status.setText(f"Ingested existing CSV for {ticker}.")
                self._update_progress_row(
                    ticker,
                    status="ingested",
                    rows=artifact.bar_count,
                    done=True,
                    tooltip=tooltip,
                )
                self._record_acquisition_attempt_result(
                    ticker=ticker,
                    meta=meta,
                    status=ingest_status,
                    summary=decision.reason,
                    parquet_path=artifact.parquet_path,
                    coverage_start=artifact.start,
                    coverage_end=artifact.end,
                    bar_count=artifact.bar_count,
                    ingested=True,
                )
            else:
                summary = ingest_error or "Existing CSV ingestion failed."
                self.download_status.setText(f"{ticker} CSV found but ingest failed: {summary}")
                self._update_progress_row(ticker, status="ingest_error", tooltip=summary)
                self._record_acquisition_attempt_result(
                    ticker=ticker,
                    meta=meta,
                    status=ingest_status,
                    summary=summary,
                    parquet_path=decision.parquet_path,
                    coverage_start=decision.coverage_start,
                    coverage_end=decision.coverage_end,
                    bar_count=decision.bar_count,
                    ingested=False,
                )
            QtCore.QTimer.singleShot(0, self._start_next_download)
            return
        if decision.secondary_dataset_id and self._decision_secondary_request_windows(decision):
            try:
                artifact = None
                for window_start, window_end in self._decision_secondary_request_windows(decision):
                    artifact = gap_fill_dataset_from_secondary(
                        build_download_dataset_id(
                            ticker,
                            source=source,
                            history_window=provider.default_history_window,
                            resolution=provider.default_resolution,
                        ),
                        str(decision.secondary_dataset_id or ""),
                        start=window_start,
                        end=window_end,
                    )
                meta["prefill_parquet_path"] = artifact.parquet_path if artifact is not None else None
                meta["prefill_coverage_start"] = artifact.start if artifact is not None else None
                meta["prefill_coverage_end"] = artifact.end if artifact is not None else None
                meta["prefill_bar_count"] = artifact.bar_count if artifact is not None else None
                self._append_download_log(
                    log_path,
                    (
                        f"[{pd.Timestamp.now('UTC').isoformat()}] Secondary gap fill complete for {ticker}\n"
                        f"Source: {decision.secondary_source or '—'} / {decision.secondary_dataset_id or '—'}\n"
                        f"Windows: {list(decision.secondary_request_windows)}\n\n"
                    ),
                )
            except Exception as exc:
                summary = f"Secondary gap-fill pre-step failed: {exc}"
                self.download_status.setText(f"{ticker} secondary gap fill failed: {exc}")
                self._update_progress_row(ticker, status="gap_fill_error", tooltip=summary)
                self._record_acquisition_attempt_result(
                    ticker=ticker,
                    meta=meta,
                    status="gap_fill_error",
                    summary=summary,
                    parquet_path=decision.parquet_path,
                    coverage_start=decision.coverage_start,
                    coverage_end=decision.coverage_end,
                    bar_count=decision.bar_count,
                    ingested=False,
                )
                QtCore.QTimer.singleShot(0, self._start_next_download)
                return
        if decision.action == ACQUISITION_ACTION_GAP_FILL_SECONDARY:
            try:
                artifact = None
                windows = self._decision_request_windows(decision)
                for window_start, window_end in windows:
                    artifact = gap_fill_dataset_from_secondary(
                        build_download_dataset_id(
                            ticker,
                            source=source,
                            history_window=provider.default_history_window,
                            resolution=provider.default_resolution,
                        ),
                        str(decision.secondary_dataset_id or ""),
                        start=window_start,
                        end=window_end,
                    )
                summary = (
                    f"{decision.reason} Secondary source: {decision.secondary_source or '—'} "
                    f"dataset={decision.secondary_dataset_id or '—'} "
                    f"parity={decision.parity_state} overlap={decision.parity_overlap_bars}"
                )
                self.download_status.setText(f"Gap-filled {ticker} from {decision.secondary_source or 'secondary source'}.")
                self._update_progress_row(
                    ticker,
                    status="gap_filled",
                    rows=artifact.bar_count if artifact is not None else decision.bar_count,
                    done=True,
                    tooltip=(
                        f"Gap-filled from {decision.secondary_source or 'secondary source'}\n"
                        f"Plan: {decision.plan_type}\n"
                        f"Log: {log_path}"
                    ),
                )
                self._record_acquisition_attempt_result(
                    ticker=ticker,
                    meta=meta,
                    status="gap_filled",
                    summary=summary,
                    parquet_path=artifact.parquet_path if artifact is not None else decision.parquet_path,
                    coverage_start=artifact.start if artifact is not None else decision.coverage_start,
                    coverage_end=artifact.end if artifact is not None else decision.coverage_end,
                    bar_count=artifact.bar_count if artifact is not None else decision.bar_count,
                    ingested=True,
                )
            except Exception as exc:
                summary = f"Cross-source gap fill failed: {exc}"
                self.download_status.setText(f"{ticker} gap fill failed: {exc}")
                self._update_progress_row(ticker, status="gap_fill_error", tooltip=summary)
                self._record_acquisition_attempt_result(
                    ticker=ticker,
                    meta=meta,
                    status="gap_fill_error",
                    summary=summary,
                    parquet_path=decision.parquet_path,
                    coverage_start=decision.coverage_start,
                    coverage_end=decision.coverage_end,
                    bar_count=decision.bar_count,
                    ingested=False,
                )
            QtCore.QTimer.singleShot(0, self._start_next_download)
            return
        self._launch_download_process_for_meta(meta)

    def _handle_download_output(self) -> None:
        proc = self.sender()
        if not isinstance(proc, QtCore.QProcess):
            return
        meta = self._download_meta(proc)
        data = bytes(proc.readAllStandardOutput()).decode("utf-8", errors="ignore")
        if meta is None:
            return
        self._append_download_log(Path(meta["log_path"]), data)
        buffer = str(meta.get("buffer", "")) + data
        lines = buffer.split("\n")
        meta["buffer"] = lines.pop() if lines else ""
        for line in lines:
            line = line.strip()
            if not line:
                continue
            self._process_download_output_line(meta, line)

    def _process_download_output_line(self, meta: dict, line: str) -> None:
        ticker = str(meta.get("ticker", ""))
        try:
            payload = json.loads(line)
        except Exception:
            if "Traceback" in line or "Error" in line or "Exception" in line or "No module named" in line:
                meta["last_error"] = line
                self._update_progress_row(ticker, status="error", tooltip=line)
            return
        if payload.get("type") == "progress":
            pages = payload.get("pages")
            rows = payload.get("rows")
            meta["rows"] = int(rows or 0)
            self.download_status.setText(
                f"Downloading {payload.get('ticker')}… pages={pages} rows={rows}"
            )
            self._update_progress_row(payload.get("ticker"), pages=pages, rows=rows)
        elif payload.get("type") == "start":
            self.download_status.setText(
                f"Downloading {payload.get('ticker')}… {payload.get('start')} → {payload.get('end')}"
            )
            self._update_progress_row(payload.get("ticker"), status="running")
        elif payload.get("type") == "done":
            meta["rows"] = int(payload.get("rows") or 0)
            self.download_status.setText(
                f"Finished {payload.get('ticker')} ({payload.get('rows')} bars)"
            )
            self._update_progress_row(payload.get("ticker"), status="done", rows=payload.get("rows"), done=True)
        elif payload.get("type") == "error":
            message = str(payload.get("message") or payload.get("details") or "Unknown download error")
            meta["last_error"] = message
            self.download_status.setText(f"{payload.get('ticker')} failed: {message}")
            self._update_progress_row(payload.get("ticker"), status="error", tooltip=message)

    def _download_process_error(self, error: QtCore.QProcess.ProcessError) -> None:
        proc = self.sender()
        if not isinstance(proc, QtCore.QProcess):
            return
        meta = self._download_meta(proc)
        if meta is None:
            return
        ticker = str(meta.get("ticker", ""))
        message = f"Process error: {getattr(error, 'name', str(error))}"
        meta["last_error"] = message
        self.download_status.setText(f"{ticker} failed: {message}")
        self._update_progress_row(ticker, status="error", tooltip=message)

    def _download_finished(self, exit_code: int, exit_status: QtCore.QProcess.ExitStatus) -> None:
        proc = self.sender()
        meta = self._download_meta(proc) if isinstance(proc, QtCore.QProcess) else None
        ticker = str(meta.get("ticker", "")) if meta else ""
        continue_same_ticker = False
        if meta:
            source = str(meta.get("source") or DEFAULT_ACQUISITION_PROVIDER)
            provider = get_acquisition_provider(source)
            request_windows = list(meta.get("request_windows") or [(None, None)])
            window_total = len(request_windows)
            window_index = int(meta.get("window_index") or 0)
            remainder = str(meta.get("buffer", ""))
            if remainder.strip():
                self._append_download_log(Path(meta["log_path"]), remainder + "\n")
                self._process_download_output_line(meta, remainder.strip())
                meta["buffer"] = ""
            log_path = Path(meta["log_path"])
            summary = str(meta.get("last_error") or "")
            attempt_status = "ingested"
            parquet_path = None
            coverage_start = None
            coverage_end = None
            bar_count = int(meta.get("rows") or 0)
            ingested = False
            if (exit_code != 0 or exit_status != QtCore.QProcess.ExitStatus.NormalExit) and not summary:
                try:
                    summary = self._summarize_error(log_path.read_text(encoding="utf-8"))
                except Exception:
                    summary = f"Process exited with code {exit_code}"
            if exit_code != 0 or exit_status != QtCore.QProcess.ExitStatus.NormalExit:
                if not summary:
                    summary = f"Process exited with code {exit_code}"
                attempt_status = "download_error"
                self.download_status.setText(f"{ticker} failed: {summary}")
                self._update_progress_row(ticker, status="error", tooltip=summary)
            elif ticker:
                ingest_status, artifact, ingest_error = self._auto_ingest_downloaded_csv(
                    ticker=ticker,
                    out_path=Path(str(meta.get("out_path") or "")),
                    source=source,
                    history_window=str(meta.get("history_window") or provider.default_history_window),
                    resolution=str(meta.get("resolution") or provider.default_resolution),
                    merge_with_existing=bool(meta.get("merge_with_existing") or window_index > 0 or window_total > 1),
                )
                attempt_status = ingest_status
                if artifact is not None:
                    parquet_path = artifact.parquet_path
                    coverage_start = artifact.start
                    coverage_end = artifact.end
                    bar_count = artifact.bar_count
                    ingested = True
                    meta["latest_parquet_path"] = artifact.parquet_path
                    meta["latest_coverage_start"] = artifact.start
                    meta["latest_coverage_end"] = artifact.end
                    meta["latest_bar_count"] = artifact.bar_count
                    if window_index + 1 < window_total:
                        meta["window_index"] = window_index + 1
                        meta["last_error"] = ""
                        continue_same_ticker = True
                        self.download_status.setText(
                            f"Finished {ticker} window {window_index + 1}/{window_total}; continuing…"
                        )
                        self._update_progress_row(
                            ticker,
                            status="running",
                            rows=artifact.bar_count,
                            tooltip=(
                                f"Completed window {window_index + 1}/{window_total}\n"
                                f"Plan: {meta.get('policy_plan_type') or '—'}\n"
                                f"Log: {log_path}"
                            ),
                        )
                    else:
                        summary = summary or str(meta.get("policy_reason") or "")
                        if window_total > 1:
                            summary = f"{summary} Completed {window_total} request windows.".strip()
                        self.download_status.setText(f"Finished {ticker} and ingested {artifact.bar_count} bars.")
                        self._update_progress_row(
                            ticker,
                            status="ingested",
                            rows=artifact.bar_count,
                            done=True,
                            tooltip=(
                                f"Ingested → {artifact.dataset_id}\n"
                                f"Plan: {meta.get('policy_plan_type') or '—'}\n"
                                f"Merge With Existing: {'yes' if meta.get('merge_with_existing') else 'no'}\n"
                                f"Windows: {window_total}\n"
                                f"Log: {log_path}"
                            ),
                        )
                else:
                    summary = ingest_error or "Download completed but ingestion failed."
                    self.download_status.setText(f"{ticker} downloaded but ingest failed: {summary}")
                    self._update_progress_row(ticker, status="ingest_error", tooltip=summary)
            if self.current_acquisition_run_id and ticker and not continue_same_ticker:
                self._record_acquisition_attempt_result(
                    ticker=ticker,
                    meta=meta,
                    status=attempt_status,
                    summary=summary,
                    parquet_path=parquet_path,
                    coverage_start=coverage_start,
                    coverage_end=coverage_end,
                    bar_count=bar_count,
                    ingested=ingested,
                )
        if isinstance(proc, QtCore.QProcess) and proc in self.download_procs:
            self.download_procs.remove(proc)
        if isinstance(proc, QtCore.QProcess):
            self.download_proc_meta.pop(id(proc), None)
        self.download_proc = self.download_procs[0] if self.download_procs else None
        if continue_same_ticker and meta is not None:
            if self.download_paused:
                return
            self._launch_download_process_for_meta(meta)
            return
        if self.download_paused:
            return
        if self.download_queue:
            self._start_next_download()
        elif not self.download_procs:
            self._finish_current_acquisition_run()
            if self.download_status.text().startswith(f"{ticker} failed:"):
                pass
            else:
                self.download_status.setText("Downloads complete.")
            self.download_progress.setVisible(False)

    def _pause_download(self) -> None:
        if not self.download_procs:
            return
        for proc in list(self.download_procs):
            if proc.state() == QtCore.QProcess.ProcessState.NotRunning:
                continue
            pid = proc.processId()
            if pid:
                os.kill(pid, signal.SIGSTOP)
        self.download_paused = True
        self.download_status.setText("Download paused.")
        for ticker in list(self.download_progress_rows.keys()):
            self._update_progress_row(ticker, status="paused")

    def _resume_download(self) -> None:
        if not self.download_procs:
            return
        for proc in list(self.download_procs):
            if proc.state() == QtCore.QProcess.ProcessState.NotRunning:
                continue
            pid = proc.processId()
            if pid:
                os.kill(pid, signal.SIGCONT)
        self.download_paused = False
        self.download_status.setText("Download resumed.")
        for ticker in list(self.download_progress_rows.keys()):
            self._update_progress_row(ticker, status="running")

    def _stop_download(self) -> None:
        for proc in list(self.download_procs):
            if proc.state() != QtCore.QProcess.ProcessState.NotRunning:
                proc.kill()
        self.download_procs = []
        self.download_proc = None
        self.download_queue = []
        self.download_active_ticker = None
        self.download_paused = False
        self.download_progress.setVisible(False)
        self.download_status.setText("Download stopped.")
        self._finish_current_acquisition_run(stopped=True)
        for ticker in list(self.download_progress_rows.keys()):
            self._update_progress_row(ticker, status="stopped")

    def _ensure_progress_row(self, ticker: str, log_path: Path | None = None) -> None:
        if ticker in self.download_progress_rows:
            if log_path is not None:
                self.download_progress_rows[ticker]["log_path"] = str(log_path)
            return
        row = self.progress_table.rowCount()
        self.progress_table.insertRow(row)
        self.progress_table.setItem(row, 0, QtWidgets.QTableWidgetItem(ticker))
        self.progress_table.setItem(row, 1, QtWidgets.QTableWidgetItem("queued"))
        self.progress_table.setItem(row, 2, QtWidgets.QTableWidgetItem("0"))
        self.progress_table.setItem(row, 3, QtWidgets.QTableWidgetItem("0"))
        bar = QtWidgets.QProgressBar()
        bar.setRange(0, EXPECTED_2Y_1M_EQUITY_ROWS)
        bar.setValue(0)
        self.progress_table.setCellWidget(row, 4, bar)
        log_btn = QtWidgets.QPushButton("Log")
        log_btn.clicked.connect(lambda _, sym=ticker: self._open_download_log(sym))
        self.progress_table.setCellWidget(row, 5, log_btn)
        self.download_progress_rows[ticker] = {"row": row, "bar": bar, "log_path": str(log_path) if log_path else None}

    def _update_progress_row(
        self,
        ticker: str | None,
        status: str | None = None,
        pages: int | None = None,
        rows: int | None = None,
        done: bool = False,
        tooltip: str | None = None,
    ) -> None:
        if not ticker or ticker not in self.download_progress_rows:
            return
        row = self.download_progress_rows[ticker]["row"]
        bar: QtWidgets.QProgressBar = self.download_progress_rows[ticker]["bar"]
        if status:
            item = self.progress_table.item(row, 1)
            if item:
                item.setText(status)
                if tooltip:
                    item.setToolTip(tooltip)
        if pages is not None:
            item = self.progress_table.item(row, 2)
            if item:
                item.setText(str(pages))
        if rows is not None:
            item = self.progress_table.item(row, 3)
            if item:
                item.setText(str(rows))
            if bar.maximum() > 0:
                bar.setValue(min(int(rows), bar.maximum()))
        if done:
            bar.setRange(0, 100)
            bar.setValue(100)
        if tooltip:
            for col in range(self.progress_table.columnCount()):
                item = self.progress_table.item(row, col)
                if item is not None:
                    item.setToolTip(tooltip)

    def _render_batches(self, batches: List[BatchRow]) -> None:
        self.batch_model.set_batches(batches)
        self.batches_table.resizeColumnsToContents()

    def _refresh_batches_live(self) -> None:
        worker = getattr(self, "worker", None)
        if not worker:
            return
        try:
            running = worker.isRunning()
        except RuntimeError:
            return
        if running:
            batches = self.catalog.load_batches()
            runs = self.catalog.load_runs()
            self._render_batches(batches)
            self._update_metrics(runs)

    # -- actions -------------------------------------------------------------
    def refresh(self, refresh_heatmap: bool = True) -> None:
        self._load_tasks()
        if hasattr(self, "tasks_table"):
            self._refresh_tasks_table()
        runs = self.catalog.load_runs()
        batches = self.catalog.load_batches()
        self._render_batches(batches)
        self._update_metrics(runs)
        self._update_optimization_panel()
        self._update_walk_forward_panel()
        self._update_monte_carlo_panel()
        if refresh_heatmap:
            self._update_heatmap()
        status = f"DB: {self.catalog.db_path} ({len(runs)} runs, {len(batches)} batches)"
        self.status_label.setText(status)

    def _update_metrics(self, runs: List[RunRow]) -> None:
        total_runs = len(runs)
        best_sharpe = max((r.metrics.get("sharpe", 0) for r in runs), default=0)
        best_return = max((r.metrics.get("total_return", 0) for r in runs), default=0)
        self.metric_equity.value_label.setText(str(total_runs))
        self.metric_sharpe.value_label.setText(f"{best_sharpe:.3f}")
        self.metric_return.value_label.setText(f"{best_return:.3f}")

    def _update_heatmap(self) -> None:
        df = self.catalog.load_heatmaps()
        if df.empty:
            self.heatmap_label.setText("No heatmap saved yet.")
            self.heatmap_label.setPixmap(QtGui.QPixmap())
            return
        row = df.iloc[0]
        file_path = Path(row["file_path"])
        if file_path.exists():
            pixmap = QtGui.QPixmap(str(file_path))
            self.heatmap_label.setPixmap(pixmap.scaledToWidth(640, QtCore.Qt.TransformationMode.SmoothTransformation))
        else:
            self.heatmap_label.setText(f"Heatmap file missing: {file_path}")

    def _update_optimization_panel(self) -> None:
        if not hasattr(self, "optimization_table"):
            return
        studies = self.catalog.load_optimization_studies()
        self.optimization_table.setRowCount(len(studies))
        if studies.empty:
            self.optimization_summary_label.setText("No optimization studies saved yet.")
            return
        for row_idx, (_, row) in enumerate(studies.iterrows()):
            dataset_scope = list(row.get("dataset_scope") or [])
            param_names = list(row.get("param_names") or [])
            timeframes = list(row.get("timeframes") or [])
            values = [
                str(row.get("study_id", "")),
                str(row.get("strategy", "")),
                str(row.get("batch_id", "")),
                str(len(dataset_scope)),
                ", ".join(param_names),
                ", ".join(timeframes),
                str(int(row.get("aggregate_count", 0) or 0)),
            ]
            for col_idx, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                if col_idx == 0:
                    item.setData(QtCore.Qt.ItemDataRole.UserRole, str(row.get("study_id", "")))
                self.optimization_table.setItem(row_idx, col_idx, item)
        if self.optimization_table.rowCount() > 0 and not self.optimization_table.selectedItems():
            self.optimization_table.selectRow(0)
        latest = studies.iloc[0]
        summary = (
            f"Latest study: {latest['study_id']} | Strategy: {latest['strategy']} | "
            f"Datasets: {len(latest['dataset_scope'])} | Params: {', '.join(latest['param_names'])} | "
            f"Aggregates: {int(latest['aggregate_count'])}"
        )
        self.optimization_summary_label.setText(summary)

    def _selected_optimization_study_id(self) -> str | None:
        if not hasattr(self, "optimization_table"):
            return None
        selection_model = self.optimization_table.selectionModel()
        if selection_model is None:
            return None
        rows = selection_model.selectedRows()
        if not rows:
            return None
        item = self.optimization_table.item(rows[0].row(), 0)
        if item is None:
            return None
        study_id = item.data(QtCore.Qt.ItemDataRole.UserRole) or item.text()
        return str(study_id) if study_id else None

    def _open_selected_optimization_study(self) -> None:
        study_id = self._selected_optimization_study_id()
        if not study_id:
            QtWidgets.QMessageBox.information(self, "No study selected", "Select an optimization study first.")
            return
        self._open_optimization_study(study_id)

    def _open_optimization_study(self, study_id: str) -> None:
        studies = self.catalog.load_optimization_studies()
        if studies.empty:
            QtWidgets.QMessageBox.information(self, "No study selected", "No optimization study is available yet.")
            return
        match = studies.loc[studies["study_id"] == study_id]
        if match.empty:
            QtWidgets.QMessageBox.information(
                self,
                "Study unavailable",
                f"Optimization study '{study_id}' could not be found in the catalog.",
            )
            return
        dlg = OptimizationStudyDialog(match.iloc[0].to_dict(), self.catalog, self)
        dlg.exec()

    def _update_walk_forward_panel(self) -> None:
        if (
            not hasattr(self, "walk_forward_table")
            or not hasattr(self, "walk_forward_source_table")
            or not hasattr(self, "walk_forward_portfolio_source_table")
        ):
            return
        all_optimization_studies = self.catalog.load_optimization_studies()
        optimization_studies = self._non_portfolio_optimization_studies(all_optimization_studies)
        self.walk_forward_source_table.setRowCount(len(optimization_studies))
        if optimization_studies.empty:
            self.walk_forward_source_summary.setText("No optimization studies available yet.")
        else:
            for row_idx, (_, row) in enumerate(optimization_studies.iterrows()):
                dataset_scope = list(row.get("dataset_scope") or [])
                timeframes = list(row.get("timeframes") or [])
                values = [
                    str(row.get("study_id", "")),
                    str(row.get("strategy", "")),
                    str(len(dataset_scope)),
                    ", ".join(timeframes),
                    str(int(row.get("aggregate_count", 0) or 0)),
                ]
                for col_idx, value in enumerate(values):
                    item = QtWidgets.QTableWidgetItem(value)
                    if col_idx == 0:
                        item.setData(QtCore.Qt.ItemDataRole.UserRole, str(row.get("study_id", "")))
                    self.walk_forward_source_table.setItem(row_idx, col_idx, item)
            if self.walk_forward_source_table.rowCount() > 0 and not self.walk_forward_source_table.selectedItems():
                self.walk_forward_source_table.selectRow(0)
            latest_source = optimization_studies.iloc[0]
            self.walk_forward_source_summary.setText(
                f"Latest source study: {latest_source['study_id']} | Strategy: {latest_source['strategy']} | "
                f"Datasets: {len(latest_source['dataset_scope'])} | Timeframes: {', '.join(latest_source['timeframes'])}"
            )

        portfolio_sources = self._build_portfolio_walk_forward_sources()
        self._walk_forward_portfolio_sources = portfolio_sources
        self.walk_forward_portfolio_source_table.setRowCount(len(portfolio_sources))
        if not portfolio_sources:
            self.walk_forward_portfolio_source_summary.setText("No eligible portfolio batches available yet.")
        else:
            for row_idx, row in enumerate(portfolio_sources):
                mode_label = "Strategy Blocks" if row.get("mode") == "strategy_blocks" else "Shared Strategy"
                values = [
                    str(row.get("batch_id", "")),
                    str(row.get("strategy", "")),
                    mode_label,
                    str(len(list(row.get("dataset_ids") or []))),
                    ", ".join(list(row.get("timeframes") or [])),
                    str(int(row.get("run_count", 0) or 0)),
                ]
                tooltip = (
                    f"Datasets: {', '.join(list(row.get('dataset_ids') or []))}\n"
                    f"Timeframes: {', '.join(list(row.get('timeframes') or []))}\n"
                    f"Mode: {mode_label}"
                )
                for col_idx, value in enumerate(values):
                    item = QtWidgets.QTableWidgetItem(value)
                    if col_idx == 0:
                        item.setData(QtCore.Qt.ItemDataRole.UserRole, str(row.get("batch_id", "")))
                    item.setToolTip(tooltip)
                    self.walk_forward_portfolio_source_table.setItem(row_idx, col_idx, item)
            if (
                self.walk_forward_portfolio_source_table.rowCount() > 0
                and not self.walk_forward_portfolio_source_table.selectedItems()
            ):
                self.walk_forward_portfolio_source_table.selectRow(0)
            latest_portfolio = portfolio_sources[0]
            self.walk_forward_portfolio_source_summary.setText(
                f"Latest portfolio source: {latest_portfolio['batch_id']} | Strategy: {latest_portfolio['strategy']} | "
                f"Datasets: {len(latest_portfolio['dataset_ids'])} | Mode: "
                f"{'Strategy Blocks' if latest_portfolio['mode'] == 'strategy_blocks' else 'Shared Strategy'}"
            )

        wf_studies = self.catalog.load_walk_forward_studies()
        self.walk_forward_table.setRowCount(len(wf_studies))
        if wf_studies.empty:
            self.walk_forward_summary_label.setText("No walk-forward studies saved yet.")
            return
        for row_idx, (_, row) in enumerate(wf_studies.iterrows()):
            created = WalkForwardSetupDialog._fmt_timestamp(row.get("created_at"))
            source_mode = str(row.get("candidate_source_mode", "") or "").replace("_", " ").title()
            fold_count = int(row["fold_count"]) if pd.notna(row.get("fold_count")) else 0
            values = [
                str(row.get("wf_study_id", "")),
                str(row.get("strategy", "")),
                str(row.get("dataset_id", "")),
                str(row.get("timeframe", "")),
                source_mode,
                str(fold_count),
                str(row.get("status", "")),
                created,
            ]
            stitched_metrics = WalkForwardStudyDialog._parse_json_text(row.get("stitched_metrics_json"))
            tooltip = (
                f"Stitched Return: {WalkForwardStudyDialog._format_numeric(stitched_metrics.get('total_return'))}\n"
                f"Stitched Sharpe: {WalkForwardStudyDialog._format_numeric(stitched_metrics.get('sharpe'))}\n"
                f"Stitched Max DD: {WalkForwardStudyDialog._format_numeric(stitched_metrics.get('max_drawdown'))}"
            )
            for col_idx, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                if col_idx == 0:
                    item.setData(QtCore.Qt.ItemDataRole.UserRole, str(row.get("wf_study_id", "")))
                item.setToolTip(tooltip)
                self.walk_forward_table.setItem(row_idx, col_idx, item)
        if self.walk_forward_table.rowCount() > 0 and not self.walk_forward_table.selectedItems():
            self.walk_forward_table.selectRow(0)
        latest = wf_studies.iloc[0]
        latest_metrics = WalkForwardStudyDialog._parse_json_text(latest.get("stitched_metrics_json"))
        latest_fold_count = int(latest["fold_count"]) if pd.notna(latest.get("fold_count")) else 0
        self.walk_forward_summary_label.setText(
            f"Latest walk-forward: {latest['wf_study_id']} | Strategy: {latest['strategy']} | "
            f"Dataset: {latest['dataset_id']} | Folds: {latest_fold_count} | "
            f"Stitched Sharpe: {WalkForwardStudyDialog._format_numeric(latest_metrics.get('sharpe'))} | "
            f"Stitched Return: {WalkForwardStudyDialog._format_numeric(latest_metrics.get('total_return'))}"
        )

    def _build_portfolio_walk_forward_sources(self) -> list[dict]:
        sources: list[dict] = []
        for batch in self.catalog.load_batches():
            try:
                params = json.loads(batch.params) if batch.params else {}
            except Exception:
                params = {}
            if str(params.get("_study_mode", "")) != STUDY_MODE_PORTFOLIO:
                continue
            if str(batch.status or "").lower() not in {"finished", "completed"}:
                continue
            mode = "strategy_blocks" if list(params.get("_portfolio_strategy_blocks") or []) else "shared_strategy"
            dataset_ids = self._portfolio_batch_dataset_ids(batch, params)
            if not dataset_ids:
                continue
            timeframes = [item.strip() for item in str(batch.timeframes or "").split(",") if item.strip()]
            sources.append(
                {
                    "batch_id": str(batch.batch_id),
                    "strategy": str(batch.strategy),
                    "dataset_id": str(batch.dataset_id),
                    "params": str(batch.params),
                    "params_dict": params,
                    "dataset_ids": dataset_ids,
                    "timeframes": timeframes,
                    "mode": mode,
                    "run_count": int(batch.run_count or 0),
                    "finished_count": int(batch.finished_count or 0),
                }
            )
        return sources

    def _portfolio_batch_dataset_ids(self, batch: BatchRow, params: dict) -> list[str]:
        dataset_ids = [
            str(dataset_id)
            for dataset_id in list(params.get("_portfolio_dataset_ids") or [])
            if str(dataset_id).strip()
        ]
        if not dataset_ids:
            dataset_ids = list(
                dict.fromkeys(
                    str(dataset_id)
                    for block in list(params.get("_portfolio_strategy_blocks") or [])
                    for dataset_id in list(block.get("asset_dataset_ids") or [])
                    if str(dataset_id).strip()
                )
            )
        if dataset_ids:
            return dataset_ids
        runs = self.catalog.load_runs(batch.batch_id)
        if not runs:
            return []
        try:
            run_params = json.loads(runs[0].params) if runs[0].params else {}
        except Exception:
            run_params = {}
        dataset_ids = list(
            dict.fromkeys(
                str(asset.get("dataset_id") or "")
                for asset in list(run_params.get("assets") or [])
                if str(asset.get("dataset_id") or "").strip()
            )
        )
        if dataset_ids:
            return dataset_ids
        return list(
            dict.fromkeys(
                str(asset.get("dataset_id") or "")
                for block in list(run_params.get("strategy_blocks") or [])
                for asset in list(block.get("assets") or [])
                if str(asset.get("dataset_id") or "").strip()
            )
        )

    def _batch_params_dict_by_batch_id(self) -> dict[str, dict]:
        mapping: dict[str, dict] = {}
        for batch in self.catalog.load_batches():
            try:
                mapping[str(batch.batch_id)] = json.loads(batch.params) if batch.params else {}
            except Exception:
                mapping[str(batch.batch_id)] = {}
        return mapping

    def _is_portfolio_optimization_study(self, study_row: dict, batch_params_map: dict[str, dict] | None = None) -> bool:
        batch_id = str(study_row.get("batch_id", "") or "")
        if not batch_id:
            return False
        params_map = batch_params_map or self._batch_params_dict_by_batch_id()
        params = params_map.get(batch_id, {})
        return str(params.get("_study_mode", "")) == STUDY_MODE_PORTFOLIO

    def _non_portfolio_optimization_studies(self, studies: pd.DataFrame) -> pd.DataFrame:
        if studies.empty:
            return studies
        batch_params_map = self._batch_params_dict_by_batch_id()
        mask = [
            not self._is_portfolio_optimization_study(row.to_dict(), batch_params_map)
            for _, row in studies.iterrows()
        ]
        return studies.loc[mask].reset_index(drop=True)

    def _selected_walk_forward_source_study_id(self) -> str | None:
        if not hasattr(self, "walk_forward_source_table"):
            return None
        selection_model = self.walk_forward_source_table.selectionModel()
        if selection_model is None:
            return None
        rows = selection_model.selectedRows()
        if not rows:
            return None
        item = self.walk_forward_source_table.item(rows[0].row(), 0)
        if item is None:
            return None
        study_id = item.data(QtCore.Qt.ItemDataRole.UserRole) or item.text()
        return str(study_id) if study_id else None

    def _selected_walk_forward_portfolio_batch_id(self) -> str | None:
        if not hasattr(self, "walk_forward_portfolio_source_table"):
            return None
        selection_model = self.walk_forward_portfolio_source_table.selectionModel()
        if selection_model is None:
            return None
        rows = selection_model.selectedRows()
        if not rows:
            return None
        item = self.walk_forward_portfolio_source_table.item(rows[0].row(), 0)
        if item is None:
            return None
        batch_id = item.data(QtCore.Qt.ItemDataRole.UserRole) or item.text()
        return str(batch_id) if batch_id else None

    def _selected_walk_forward_study_id(self) -> str | None:
        if not hasattr(self, "walk_forward_table"):
            return None
        selection_model = self.walk_forward_table.selectionModel()
        if selection_model is None:
            return None
        rows = selection_model.selectedRows()
        if not rows:
            return None
        item = self.walk_forward_table.item(rows[0].row(), 0)
        if item is None:
            return None
        study_id = item.data(QtCore.Qt.ItemDataRole.UserRole) or item.text()
        return str(study_id) if study_id else None

    def _launch_walk_forward_from_selected_optimization(self) -> None:
        if self.worker and self.worker.isRunning():
            QtWidgets.QMessageBox.information(self, "Busy", "A grid study is already running.")
            return
        if self.walk_forward_worker and self.walk_forward_worker.isRunning():
            QtWidgets.QMessageBox.information(self, "Busy", "A walk-forward study is already running.")
            return
        sender = self.sender()
        if sender is getattr(self, "optimization_walk_forward_btn", None):
            study_id = self._selected_optimization_study_id() or self._selected_walk_forward_source_study_id()
        else:
            study_id = self._selected_walk_forward_source_study_id() or self._selected_optimization_study_id()
        if not study_id:
            QtWidgets.QMessageBox.information(
                self,
                "No source study selected",
                "Select an optimization study first, either in the Optimization tab or the Walk Forward source list.",
            )
            return
        studies = self.catalog.load_optimization_studies()
        if studies.empty:
            QtWidgets.QMessageBox.information(self, "No source study selected", "No optimization study is available yet.")
            return
        match = studies.loc[studies["study_id"] == study_id]
        if match.empty:
            QtWidgets.QMessageBox.information(
                self,
                "Study unavailable",
                f"Optimization study '{study_id}' could not be found in the catalog.",
            )
            return
        source_study_row = match.iloc[0].to_dict()
        if self._is_portfolio_optimization_study(source_study_row):
            batch_id = str(source_study_row.get("batch_id", "") or "")
            source_rows = getattr(self, "_walk_forward_portfolio_sources", None) or self._build_portfolio_walk_forward_sources()
            source_row = next((row for row in source_rows if str(row.get("batch_id", "")) == batch_id), None)
            if source_row is None:
                QtWidgets.QMessageBox.information(
                    self,
                    "Portfolio Source Required",
                    "This optimization study comes from a portfolio batch. Use the portfolio source section in the Walk Forward tab.",
                )
                return
            source_row = dict(source_row)
            source_row["source_study_id"] = study_id
            dlg = PortfolioWalkForwardSetupDialog(source_row, self.catalog, self)
            if dlg.exec() != int(QtWidgets.QDialog.DialogCode.Accepted):
                return
            settings = dlg.settings()
            self.walk_forward_launch_btn.setEnabled(False)
            if hasattr(self, "walk_forward_portfolio_launch_btn"):
                self.walk_forward_portfolio_launch_btn.setEnabled(False)
            if hasattr(self, "optimization_walk_forward_btn"):
                self.optimization_walk_forward_btn.setEnabled(False)
            self.walk_forward_open_btn.setEnabled(False)
            self._current_walk_forward_started_at = time.perf_counter()
            self.status_label.setText("Running portfolio walk-forward study…")
            self.walk_forward_worker = PortfolioWalkForwardWorker(
                catalog_path=self.catalog.db_path,
                source_batch_row=source_row,
                timeframe=str(settings["timeframe"]),
                first_test_start=str(settings["first_test_start"]),
                test_window_bars=int(settings["test_window_bars"]),
                num_folds=int(settings["num_folds"]),
                min_train_bars=int(settings["min_train_bars"]),
                candidate_source_mode=str(settings["candidate_source_mode"]),
                execution_mode=str(settings["execution_mode"]),
                bt_settings=self._collect_backtest_settings(),
                description=str(settings["description"]),
            )
            self.walk_forward_worker.finished_signal.connect(self._walk_forward_finished)
            self.walk_forward_worker.error_signal.connect(self._walk_forward_error)
            self.walk_forward_worker.finished.connect(self.walk_forward_worker.deleteLater)
            self.walk_forward_worker.start()
            return
        strategy_cls = RunChartDialog._strategy_class_static(str(source_study_row.get("strategy", "")))
        dlg = WalkForwardSetupDialog(source_study_row, self.catalog, self)
        if dlg.exec() != int(QtWidgets.QDialog.DialogCode.Accepted):
            return
        settings = dlg.settings()
        self.walk_forward_launch_btn.setEnabled(False)
        if hasattr(self, "optimization_walk_forward_btn"):
            self.optimization_walk_forward_btn.setEnabled(False)
        self.walk_forward_open_btn.setEnabled(False)
        self._current_walk_forward_started_at = time.perf_counter()
        self.status_label.setText("Running walk-forward study…")
        self.walk_forward_worker = WalkForwardWorker(
            catalog_path=self.catalog.db_path,
            source_study_row=source_study_row,
            strategy_cls=strategy_cls,
            dataset_id=str(settings["dataset_id"]),
            timeframe=str(settings["timeframe"]),
            first_test_start=str(settings["first_test_start"]),
            test_window_bars=int(settings["test_window_bars"]),
            num_folds=int(settings["num_folds"]),
            min_train_bars=int(settings["min_train_bars"]),
            candidate_source_mode=str(settings["candidate_source_mode"]),
            execution_mode=str(settings["execution_mode"]),
            bt_settings=self._collect_backtest_settings(),
            description=str(settings["description"]),
        )
        self.walk_forward_worker.finished_signal.connect(self._walk_forward_finished)
        self.walk_forward_worker.error_signal.connect(self._walk_forward_error)
        self.walk_forward_worker.finished.connect(self.walk_forward_worker.deleteLater)
        self.walk_forward_worker.start()

    def _launch_walk_forward_from_selected_portfolio_batch(self) -> None:
        if self.worker and self.worker.isRunning():
            QtWidgets.QMessageBox.information(self, "Busy", "A grid study is already running.")
            return
        if self.walk_forward_worker and self.walk_forward_worker.isRunning():
            QtWidgets.QMessageBox.information(self, "Busy", "A walk-forward study is already running.")
            return
        batch_id = self._selected_walk_forward_portfolio_batch_id()
        if not batch_id:
            QtWidgets.QMessageBox.information(
                self,
                "No portfolio batch selected",
                "Select a portfolio batch first in the Walk Forward source list.",
            )
            return
        source_rows = getattr(self, "_walk_forward_portfolio_sources", None) or self._build_portfolio_walk_forward_sources()
        source_row = next((row for row in source_rows if str(row.get("batch_id", "")) == batch_id), None)
        if not source_row:
            QtWidgets.QMessageBox.information(
                self,
                "Batch unavailable",
                f"Portfolio batch '{batch_id}' could not be found in the current source list.",
            )
            return
        dlg = PortfolioWalkForwardSetupDialog(source_row, self.catalog, self)
        if dlg.exec() != int(QtWidgets.QDialog.DialogCode.Accepted):
            return
        settings = dlg.settings()
        self.walk_forward_launch_btn.setEnabled(False)
        if hasattr(self, "walk_forward_portfolio_launch_btn"):
            self.walk_forward_portfolio_launch_btn.setEnabled(False)
        if hasattr(self, "optimization_walk_forward_btn"):
            self.optimization_walk_forward_btn.setEnabled(False)
        self.walk_forward_open_btn.setEnabled(False)
        self._current_walk_forward_started_at = time.perf_counter()
        self.status_label.setText("Running portfolio walk-forward study…")
        self.walk_forward_worker = PortfolioWalkForwardWorker(
            catalog_path=self.catalog.db_path,
            source_batch_row=source_row,
            timeframe=str(settings["timeframe"]),
            first_test_start=str(settings["first_test_start"]),
            test_window_bars=int(settings["test_window_bars"]),
            num_folds=int(settings["num_folds"]),
            min_train_bars=int(settings["min_train_bars"]),
            candidate_source_mode=str(settings["candidate_source_mode"]),
            execution_mode=str(settings["execution_mode"]),
            bt_settings=self._collect_backtest_settings(),
            description=str(settings["description"]),
        )
        self.walk_forward_worker.finished_signal.connect(self._walk_forward_finished)
        self.walk_forward_worker.error_signal.connect(self._walk_forward_error)
        self.walk_forward_worker.finished.connect(self.walk_forward_worker.deleteLater)
        self.walk_forward_worker.start()

    def _open_selected_walk_forward_study(self) -> None:
        study_id = self._selected_walk_forward_study_id()
        if not study_id:
            QtWidgets.QMessageBox.information(self, "No study selected", "Select a walk-forward study first.")
            return
        self._open_walk_forward_study(study_id)

    def _open_walk_forward_study(self, wf_study_id: str) -> None:
        studies = self.catalog.load_walk_forward_studies()
        if studies.empty:
            QtWidgets.QMessageBox.information(self, "No study selected", "No walk-forward study is available yet.")
            return
        match = studies.loc[studies["wf_study_id"] == wf_study_id]
        if match.empty:
            QtWidgets.QMessageBox.information(
                self,
                "Study unavailable",
                f"Walk-forward study '{wf_study_id}' could not be found in the catalog.",
            )
            return
        dlg = WalkForwardStudyDialog(match.iloc[0].to_dict(), self.catalog, self)
        dlg.exec()

    def _walk_forward_finished(self, payload) -> None:
        message = "Walk-forward study completed."
        wf_study_id = None
        if isinstance(payload, dict):
            message = str(payload.get("message") or message)
            wf_study_id = payload.get("wf_study_id")
        elapsed_text = ""
        if self._current_walk_forward_started_at is not None:
            elapsed_text = f" in {time.perf_counter() - self._current_walk_forward_started_at:.2f}s"
            self._current_walk_forward_started_at = None
        if elapsed_text and message.endswith("."):
            message = f"{message[:-1]}{elapsed_text}."
        elif elapsed_text:
            message = f"{message}{elapsed_text}"
        self.walk_forward_launch_btn.setEnabled(True)
        if hasattr(self, "walk_forward_portfolio_launch_btn"):
            self.walk_forward_portfolio_launch_btn.setEnabled(True)
        if hasattr(self, "optimization_walk_forward_btn"):
            self.optimization_walk_forward_btn.setEnabled(True)
        self.walk_forward_open_btn.setEnabled(True)
        self.walk_forward_worker = None
        self.refresh(refresh_heatmap=False)
        self.status_label.setText(message)
        if wf_study_id:
            self._open_walk_forward_study(str(wf_study_id))

    def _walk_forward_error(self, message: str) -> None:
        self.walk_forward_launch_btn.setEnabled(True)
        if hasattr(self, "walk_forward_portfolio_launch_btn"):
            self.walk_forward_portfolio_launch_btn.setEnabled(True)
        if hasattr(self, "optimization_walk_forward_btn"):
            self.optimization_walk_forward_btn.setEnabled(True)
        self.walk_forward_open_btn.setEnabled(True)
        if self._current_walk_forward_started_at is not None:
            elapsed = time.perf_counter() - self._current_walk_forward_started_at
            self.status_label.setText(f"Walk-forward error after {elapsed:.2f}s")
            self._current_walk_forward_started_at = None
        else:
            self.status_label.setText("Walk-forward error")
        print("Walk-forward error:\n", message or "Unknown error")
        summary = self._summarize_error(message)
        self._show_error_dialog("Walk-Forward Error", summary, details=message)
        self.walk_forward_worker = None

    def _update_monte_carlo_panel(self) -> None:
        if not hasattr(self, "monte_carlo_table") or not hasattr(self, "monte_carlo_source_table"):
            return
        wf_studies = self.catalog.load_walk_forward_studies()
        self.monte_carlo_source_table.setRowCount(len(wf_studies))
        if wf_studies.empty:
            self.monte_carlo_source_summary.setText("No walk-forward studies available yet.")
        else:
            for row_idx, (_, row) in enumerate(wf_studies.iterrows()):
                schedule_payload = WalkForwardStudyDialog._parse_json_text(row.get("schedule_json"))
                params_payload = WalkForwardStudyDialog._parse_json_text(row.get("params_json"))
                is_portfolio = str(params_payload.get("source_kind", "")) == "portfolio"
                portfolio_mode = str(schedule_payload.get("portfolio_mode", "") or "")
                if is_portfolio:
                    type_label = "Fixed Strategy Blocks" if portfolio_mode == "strategy_blocks" else "Shared Strategy Portfolio"
                else:
                    type_label = "Single Strategy"
                fold_count = int(row["fold_count"]) if pd.notna(row.get("fold_count")) else 0
                values = [
                    str(row.get("wf_study_id", "")),
                    type_label,
                    str(row.get("strategy", "")),
                    str(row.get("dataset_id", "")),
                    str(row.get("timeframe", "")),
                    str(fold_count),
                    str(row.get("status", "")),
                ]
                for col_idx, value in enumerate(values):
                    item = QtWidgets.QTableWidgetItem(value)
                    if col_idx == 0:
                        item.setData(QtCore.Qt.ItemDataRole.UserRole, str(row.get("wf_study_id", "")))
                    self.monte_carlo_source_table.setItem(row_idx, col_idx, item)
            if self.monte_carlo_source_table.rowCount() > 0 and not self.monte_carlo_source_table.selectedItems():
                self.monte_carlo_source_table.selectRow(0)
            latest = wf_studies.iloc[0]
            stitched = WalkForwardStudyDialog._parse_json_text(latest.get("stitched_metrics_json"))
            latest_schedule = WalkForwardStudyDialog._parse_json_text(latest.get("schedule_json"))
            latest_params = WalkForwardStudyDialog._parse_json_text(latest.get("params_json"))
            latest_type = "Single Strategy"
            if str(latest_params.get("source_kind", "")) == "portfolio":
                latest_type = (
                    "Fixed Strategy Blocks"
                    if str(latest_schedule.get("portfolio_mode", "") or "") == "strategy_blocks"
                    else "Shared Strategy Portfolio"
                )
            self.monte_carlo_source_summary.setText(
                f"Latest walk-forward source: {latest['wf_study_id']} | Type: {latest_type} | "
                f"Strategy: {latest['strategy']} | Dataset: {latest['dataset_id']} | "
                f"Stitched Sharpe: {WalkForwardStudyDialog._format_numeric(stitched.get('sharpe'))}"
            )

        mc_studies = self.catalog.load_monte_carlo_studies()
        self.monte_carlo_table.setRowCount(len(mc_studies))
        if mc_studies.empty:
            self.monte_carlo_summary_label.setText("No Monte Carlo studies saved yet.")
            return
        for row_idx, (_, row) in enumerate(mc_studies.iterrows()):
            summary = MonteCarloStudyDialog._parse_json_dict(row.get("summary_json"))
            mc_type = "Single Strategy"
            if bool(summary.get("source_is_portfolio", False)):
                mc_type = (
                    "Fixed Strategy Blocks"
                    if str(summary.get("source_portfolio_mode", "") or "") == "strategy_blocks"
                    else "Shared Strategy Portfolio"
                )
            values = [
                str(row.get("mc_study_id", "")),
                str(row.get("source_id", "")),
                mc_type,
                str(row.get("resampling_mode", "")),
                str(int(row.get("simulation_count", 0) or 0)),
                str(int(row.get("source_trade_count", 0) or 0)),
                MonteCarloStudyDialog._fmt(summary.get("terminal_return_p50")),
                MonteCarloStudyDialog._fmt_pct(summary.get("loss_probability")),
                WalkForwardSetupDialog._fmt_timestamp(row.get("created_at")),
            ]
            tooltip = (
                f"P05 Return: {MonteCarloStudyDialog._fmt(summary.get('terminal_return_p05'))}\n"
                f"P95 Return: {MonteCarloStudyDialog._fmt(summary.get('terminal_return_p95'))}\n"
                f"Median Max DD: {MonteCarloStudyDialog._fmt(summary.get('max_drawdown_p50'))}\n"
                f"P95 Max DD: {MonteCarloStudyDialog._fmt(summary.get('max_drawdown_p95'))}"
            )
            for col_idx, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                if col_idx == 0:
                    item.setData(QtCore.Qt.ItemDataRole.UserRole, str(row.get("mc_study_id", "")))
                item.setToolTip(tooltip)
                self.monte_carlo_table.setItem(row_idx, col_idx, item)
        if self.monte_carlo_table.rowCount() > 0 and not self.monte_carlo_table.selectedItems():
            self.monte_carlo_table.selectRow(0)
        latest = mc_studies.iloc[0]
        summary = MonteCarloStudyDialog._parse_json_dict(latest.get("summary_json"))
        latest_type = "Single Strategy"
        if bool(summary.get("source_is_portfolio", False)):
            latest_type = (
                "Fixed Strategy Blocks"
                if str(summary.get("source_portfolio_mode", "") or "") == "strategy_blocks"
                else "Shared Strategy Portfolio"
            )
        self.monte_carlo_summary_label.setText(
            f"Latest Monte Carlo: {latest['mc_study_id']} | Source: {latest['source_id']} | Type: {latest_type} | "
            f"Median Return: {MonteCarloStudyDialog._fmt(summary.get('terminal_return_p50'))} | "
            f"Loss Probability: {MonteCarloStudyDialog._fmt_pct(summary.get('loss_probability'))} | "
            f"P95 Max DD: {MonteCarloStudyDialog._fmt(summary.get('max_drawdown_p95'))}"
        )

    def _selected_walk_forward_source_for_monte_carlo(self) -> str | None:
        if not hasattr(self, "monte_carlo_source_table"):
            return None
        selection_model = self.monte_carlo_source_table.selectionModel()
        if selection_model is None:
            return None
        rows = selection_model.selectedRows()
        if not rows:
            return None
        item = self.monte_carlo_source_table.item(rows[0].row(), 0)
        if item is None:
            return None
        study_id = item.data(QtCore.Qt.ItemDataRole.UserRole) or item.text()
        return str(study_id) if study_id else None

    def _selected_monte_carlo_study_id(self) -> str | None:
        if not hasattr(self, "monte_carlo_table"):
            return None
        selection_model = self.monte_carlo_table.selectionModel()
        if selection_model is None:
            return None
        rows = selection_model.selectedRows()
        if not rows:
            return None
        item = self.monte_carlo_table.item(rows[0].row(), 0)
        if item is None:
            return None
        study_id = item.data(QtCore.Qt.ItemDataRole.UserRole) or item.text()
        return str(study_id) if study_id else None

    def _launch_monte_carlo_from_selected_walk_forward(self) -> None:
        if self.worker and self.worker.isRunning():
            QtWidgets.QMessageBox.information(self, "Busy", "A grid study is already running.")
            return
        if self.walk_forward_worker and self.walk_forward_worker.isRunning():
            QtWidgets.QMessageBox.information(self, "Busy", "A walk-forward study is already running.")
            return
        if self.monte_carlo_worker and self.monte_carlo_worker.isRunning():
            QtWidgets.QMessageBox.information(self, "Busy", "A Monte Carlo study is already running.")
            return
        wf_study_id = self._selected_walk_forward_source_for_monte_carlo() or self._selected_walk_forward_study_id()
        if not wf_study_id:
            QtWidgets.QMessageBox.information(
                self,
                "No source study selected",
                "Select a walk-forward study first, either in the Walk Forward tab or the Monte Carlo source list.",
            )
            return
        studies = self.catalog.load_walk_forward_studies()
        if studies.empty:
            QtWidgets.QMessageBox.information(self, "No source study selected", "No walk-forward study is available yet.")
            return
        match = studies.loc[studies["wf_study_id"] == wf_study_id]
        if match.empty:
            QtWidgets.QMessageBox.information(
                self,
                "Study unavailable",
                f"Walk-forward study '{wf_study_id}' could not be found in the catalog.",
            )
            return
        source_row = match.iloc[0].to_dict()
        dlg = MonteCarloSetupDialog(source_row, self.catalog, self)
        if dlg.exec() != int(QtWidgets.QDialog.DialogCode.Accepted):
            return
        settings = dlg.settings()
        self.monte_carlo_launch_btn.setEnabled(False)
        self.monte_carlo_open_btn.setEnabled(False)
        self._current_monte_carlo_started_at = time.perf_counter()
        self.status_label.setText("Running Monte Carlo study…")
        self.monte_carlo_worker = MonteCarloWorker(
            catalog_path=self.catalog.db_path,
            source_study_row=source_row,
            resampling_mode=str(settings["resampling_mode"]),
            simulation_count=int(settings["simulation_count"]),
            seed=int(settings["seed"]),
            cost_stress_bps=float(settings["cost_stress_bps"]),
            description=str(settings["description"]),
        )
        self.monte_carlo_worker.finished_signal.connect(self._monte_carlo_finished)
        self.monte_carlo_worker.error_signal.connect(self._monte_carlo_error)
        self.monte_carlo_worker.finished.connect(self.monte_carlo_worker.deleteLater)
        self.monte_carlo_worker.start()

    def _open_selected_monte_carlo_study(self) -> None:
        study_id = self._selected_monte_carlo_study_id()
        if not study_id:
            QtWidgets.QMessageBox.information(self, "No study selected", "Select a Monte Carlo study first.")
            return
        self._open_monte_carlo_study(study_id)

    def _open_monte_carlo_study(self, mc_study_id: str) -> None:
        studies = self.catalog.load_monte_carlo_studies()
        if studies.empty:
            QtWidgets.QMessageBox.information(self, "No study selected", "No Monte Carlo study is available yet.")
            return
        match = studies.loc[studies["mc_study_id"] == mc_study_id]
        if match.empty:
            QtWidgets.QMessageBox.information(
                self,
                "Study unavailable",
                f"Monte Carlo study '{mc_study_id}' could not be found in the catalog.",
            )
            return
        dlg = MonteCarloStudyDialog(match.iloc[0].to_dict(), self.catalog, self)
        dlg.exec()

    def _monte_carlo_finished(self, payload) -> None:
        message = "Monte Carlo study completed."
        mc_study_id = None
        if isinstance(payload, dict):
            message = str(payload.get("message") or message)
            mc_study_id = payload.get("mc_study_id")
        elapsed_text = ""
        if self._current_monte_carlo_started_at is not None:
            elapsed_text = f" in {time.perf_counter() - self._current_monte_carlo_started_at:.2f}s"
            self._current_monte_carlo_started_at = None
        if elapsed_text and message.endswith("."):
            message = f"{message[:-1]}{elapsed_text}."
        elif elapsed_text:
            message = f"{message}{elapsed_text}"
        self.monte_carlo_launch_btn.setEnabled(True)
        self.monte_carlo_open_btn.setEnabled(True)
        self.monte_carlo_worker = None
        self.refresh(refresh_heatmap=False)
        self.status_label.setText(message)
        if mc_study_id:
            self._open_monte_carlo_study(str(mc_study_id))

    def _monte_carlo_error(self, message: str) -> None:
        self.monte_carlo_launch_btn.setEnabled(True)
        self.monte_carlo_open_btn.setEnabled(True)
        if self._current_monte_carlo_started_at is not None:
            elapsed = time.perf_counter() - self._current_monte_carlo_started_at
            self.status_label.setText(f"Monte Carlo error after {elapsed:.2f}s")
            self._current_monte_carlo_started_at = None
        else:
            self.status_label.setText("Monte Carlo error")
        print("Monte Carlo error:\n", message or "Unknown error")
        summary = self._summarize_error(message)
        self._show_error_dialog("Monte Carlo Error", summary, details=message)
        self.monte_carlo_worker = None

    # -- orchestration actions -----------------------------------------------
    def _browse_csv(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select CSV", "", "CSV Files (*.csv);;All Files (*)")
        if path:
            self.csv_path_edit.setText(path)

    def _add_csv_clicked(self) -> None:
        csv_path = Path(self.csv_path_edit.text().strip())
        dataset_id = self.dataset_combo.currentText().strip() or "dataset"
        if not csv_path.exists():
            QtWidgets.QMessageBox.warning(self, "Missing CSV", f"CSV not found: {csv_path}")
            return
        try:
            artifact = ingest_csv_to_store(csv_path, dataset_id=dataset_id)
            acq_run_id = f"import_{uuid.uuid4().hex[:8]}"
            rc = ResultCatalog(self.catalog.db_path)
            started_at = pd.Timestamp.utcnow().isoformat()
            rc.start_acquisition_run(
                acquisition_run_id=acq_run_id,
                trigger_type="manual_import",
                source="manual_csv",
                started_at=started_at,
                status="running",
                symbol_count=1,
                notes=f"Manual CSV import for {dataset_id}.",
            )
            rc.record_acquisition_attempt(
                attempt_id=f"{acq_run_id}_0001",
                acquisition_run_id=acq_run_id,
                seq=1,
                source="manual_csv",
                symbol=dataset_id,
                dataset_id=artifact.dataset_id,
                status="ingested",
                started_at=started_at,
                finished_at=pd.Timestamp.utcnow().isoformat(),
                csv_path=artifact.csv_path,
                parquet_path=artifact.parquet_path,
                coverage_start=artifact.start,
                coverage_end=artifact.end,
                bar_count=artifact.bar_count,
                ingested=True,
            )
            rc.finish_acquisition_run(
                acq_run_id,
                finished_at=pd.Timestamp.utcnow().isoformat(),
                status="success",
                success_count=1,
                failed_count=0,
                ingested_count=1,
                notes=f"Imported {artifact.bar_count} bars into {artifact.dataset_id}.",
            )
            self._refresh_dataset_options(select_id=dataset_id)
            self.status_label.setText(f"Added {csv_path.name} → {dataset_id} and cataloged it.")
        except Exception as exc:
            self._show_error_dialog("Import Error", str(exc), details=traceback.format_exc())

    def _collect_backtest_settings(self) -> Dict[str, float | bool | dict | str]:
        def _float(edit: QtWidgets.QLineEdit, default: float) -> float:
            try:
                val = float(edit.text().strip())
                return val
            except Exception:
                return default

        starting_cash = _float(self.starting_cash_edit, 100_000)
        fee_rate = _float(self.fee_rate_edit, 0.0002)
        fee_buy = _float(self.fee_buy_edit, fee_rate)
        fee_sell = _float(self.fee_sell_edit, fee_rate)
        slippage = _float(self.slippage_edit, 0.0002)
        slip_buy = _float(self.slip_buy_edit, slippage)
        slip_sell = _float(self.slip_sell_edit, slippage)
        borrow_rate = _float(self.borrow_rate_edit, 0.0)
        fill_ratio = _float(self.fill_ratio_edit, 1.0)
        return {
            "starting_cash": starting_cash,
            "fee_rate": fee_rate,
            "fee_schedule": {"buy": fee_buy, "sell": fee_sell},
            "slippage": slippage,
            "slippage_schedule": {"buy": slip_buy, "sell": slip_sell},
            "borrow_rate": borrow_rate,
            "fill_ratio": fill_ratio,
            "fill_on_close": self.fill_on_close_chk.isChecked(),
            "recalc_on_fill": self.recalc_on_fill_chk.isChecked(),
            "allow_short": self.allow_short_chk.isChecked(),
            "use_cache": self.use_cache_chk.isChecked(),
            "prevent_scale_in": self.prevent_scale_in_chk.isChecked(),
            "one_order_per_signal": self.one_order_chk.isChecked(),
            "execution_mode": str(self.execution_mode_combo.currentData() or ExecutionMode.AUTO.value),
            "study_mode": str(self.study_mode_combo.currentData() or STUDY_MODE_INDEPENDENT),
            "portfolio_allocation_mode": str(self.portfolio_allocation_combo.currentData() or PORTFOLIO_ALLOC_EQUAL),
            "portfolio_weights_text": self.portfolio_weights_edit.text().strip(),
            "portfolio_strategy_blocks": [dict(block) for block in self.portfolio_strategy_blocks],
            "portfolio_ownership_mode": str(
                self.portfolio_ownership_combo.currentData() or ALLOCATION_OWNERSHIP_STRATEGY
            ),
            "portfolio_ranking_mode": str(self.portfolio_ranking_combo.currentData() or RANKING_MODE_NONE),
            "portfolio_rank_count": int(self.portfolio_rank_count_spin.value()),
            "portfolio_score_threshold": float(self.portfolio_score_threshold_spin.value()),
            "portfolio_weighting_mode": str(self.portfolio_weighting_combo.currentData() or WEIGHTING_MODE_PRESERVE),
            "portfolio_min_active_weight": float(self.portfolio_min_active_weight_spin.value()),
            "portfolio_max_asset_weight": float(self.portfolio_max_asset_weight_spin.value()),
            "portfolio_cash_reserve_weight": float(self.portfolio_cash_reserve_spin.value()),
            "portfolio_rebalance_mode": str(self.portfolio_rebalance_combo.currentData() or REBALANCE_MODE_ON_CHANGE),
            "portfolio_rebalance_every_n_bars": int(self.portfolio_rebalance_every_spin.value()),
            "portfolio_rebalance_drift_threshold": float(self.portfolio_rebalance_drift_spin.value()),
        }

    @staticmethod
    def _study_mode_label(mode: str | None) -> str:
        if mode == STUDY_MODE_PORTFOLIO:
            return "Portfolio"
        return "Independent"

    def _vectorized_support_issues(
        self,
        strategy_factory: Callable,
        timeframes: list[str],
        bt_settings: Dict[str, float | bool | dict | str],
    ) -> tuple[list[str], bool]:
        issues: list[str] = []
        supported_strategies = {"SMACrossStrategy", "ZScoreMeanReversionStrategy"}
        if strategy_factory.__name__ not in supported_strategies:
            issues.append(
                f"{strategy_factory.__name__} does not have a vectorized adapter yet. "
                "Current vectorized support is limited to SMACrossStrategy and ZScoreMeanReversionStrategy."
            )
            return issues, False

        if self.intrabar_chk.isChecked():
            issues.append("Intrabar simulation must be turned off for vectorized v1.")
        return issues, True

    def _portfolio_strategy_block_support_issues(
        self,
        portfolio_strategy_blocks: Sequence[dict],
        bt_settings: Dict[str, float | bool | dict | str],
    ) -> tuple[list[str], bool]:
        issues: list[str] = []
        can_apply_defaults = True
        supported_strategies = {"SMACrossStrategy", "ZScoreMeanReversionStrategy"}
        if not portfolio_strategy_blocks:
            issues.append("No strategy blocks are configured.")
            return issues, False
        for block in portfolio_strategy_blocks:
            strategy_name = str(block.get("strategy_name") or "").strip()
            if strategy_name not in supported_strategies:
                issues.append(
                    f"{strategy_name or 'Unknown strategy'} does not have a vectorized portfolio adapter yet."
                )
                can_apply_defaults = False
            if not list(block.get("asset_dataset_ids") or []):
                issues.append(
                    f"{str(block.get('display_name') or block.get('block_id') or 'A strategy block')} has no attached assets."
                )
                can_apply_defaults = False
        if self.intrabar_chk.isChecked():
            issues.append("Intrabar simulation must be turned off for vectorized portfolio v1.")
        return issues, can_apply_defaults

    def _apply_vectorized_defaults(self) -> None:
        self.intrabar_chk.setChecked(False)

    def _portfolio_vectorized_support_issues(
        self,
        strategy_factory: Callable,
        timeframes: list[str],
        bt_settings: Dict[str, float | bool | dict | str],
    ) -> tuple[list[str], bool]:
        issues, can_apply_defaults = self._vectorized_support_issues(strategy_factory, timeframes, bt_settings)
        requested_mode = str(bt_settings.get("execution_mode", ExecutionMode.AUTO.value))
        ownership_mode = str(bt_settings.get("portfolio_ownership_mode", ALLOCATION_OWNERSHIP_STRATEGY))
        ranking_mode = str(bt_settings.get("portfolio_ranking_mode", RANKING_MODE_NONE))
        weighting_mode = str(bt_settings.get("portfolio_weighting_mode", WEIGHTING_MODE_PRESERVE))
        min_active_weight = float(bt_settings.get("portfolio_min_active_weight", 0.0) or 0.0)
        max_asset_weight = float(bt_settings.get("portfolio_max_asset_weight", 0.0) or 0.0)
        cash_reserve_weight = float(bt_settings.get("portfolio_cash_reserve_weight", 0.0) or 0.0)
        if requested_mode == ExecutionMode.REFERENCE.value:
            issues.append("Portfolio study mode does not have a reference-engine fallback yet.")
        if ownership_mode == ALLOCATION_OWNERSHIP_STRATEGY and ranking_mode != RANKING_MODE_NONE:
            issues.append(
                "Strategy-Owned allocation cannot be combined with portfolio ranking. Use Hybrid or Portfolio-Owned allocation."
            )
            can_apply_defaults = False
        if ownership_mode != ALLOCATION_OWNERSHIP_PORTFOLIO and weighting_mode != WEIGHTING_MODE_PRESERVE:
            issues.append(
                "Portfolio weighting overrides require Portfolio-Owned allocation. Strategy-Owned and Hybrid modes must preserve strategy sizing."
            )
            can_apply_defaults = False
        if min_active_weight < 0.0:
            issues.append("Min Active Weight must be >= 0.")
            can_apply_defaults = False
        if max_asset_weight < 0.0:
            issues.append("Max Asset Weight must be >= 0.")
            can_apply_defaults = False
        if min_active_weight > 0.0 and max_asset_weight > 0.0 and max_asset_weight < min_active_weight:
            issues.append("Max Asset Weight must be >= Min Active Weight when both are enabled.")
            can_apply_defaults = False
        if not (0.0 <= cash_reserve_weight < 1.0):
            issues.append("Cash Reserve Weight must be between 0 and 1.")
            can_apply_defaults = False
        if (
            str(bt_settings.get("portfolio_rebalance_mode", REBALANCE_MODE_ON_CHANGE))
            in {REBALANCE_MODE_ON_CHANGE_OR_PERIODIC, REBALANCE_MODE_ON_CHANGE_OR_PERIODIC_OR_DRIFT}
            and int(bt_settings.get("portfolio_rebalance_every_n_bars", 0) or 0) <= 0
        ):
            issues.append("Periodic portfolio rebalancing requires Rebalance Every N Bars > 0.")
            can_apply_defaults = False
        if (
            str(bt_settings.get("portfolio_rebalance_mode", REBALANCE_MODE_ON_CHANGE))
            in {REBALANCE_MODE_ON_CHANGE_OR_DRIFT, REBALANCE_MODE_ON_CHANGE_OR_PERIODIC_OR_DRIFT}
            and float(bt_settings.get("portfolio_rebalance_drift_threshold", 0.0) or 0.0) <= 0.0
        ):
            issues.append("Drift-threshold portfolio rebalancing requires a drift threshold > 0.")
            can_apply_defaults = False
        return issues, can_apply_defaults

    def _apply_portfolio_vectorized_defaults(self) -> None:
        self._apply_vectorized_defaults()
        if str(self.portfolio_ranking_combo.currentData() or RANKING_MODE_NONE) != RANKING_MODE_NONE and (
            str(self.portfolio_ownership_combo.currentData() or ALLOCATION_OWNERSHIP_STRATEGY)
            == ALLOCATION_OWNERSHIP_STRATEGY
        ):
            idx = self.portfolio_ownership_combo.findData(ALLOCATION_OWNERSHIP_HYBRID)
            if idx >= 0:
                self.portfolio_ownership_combo.setCurrentIndex(idx)
        if str(self.execution_mode_combo.currentData() or "") == ExecutionMode.REFERENCE.value:
            idx = self.execution_mode_combo.findData(ExecutionMode.AUTO.value)
            if idx >= 0:
                self.execution_mode_combo.setCurrentIndex(idx)
        if (
            str(self.portfolio_ownership_combo.currentData() or ALLOCATION_OWNERSHIP_STRATEGY)
            != ALLOCATION_OWNERSHIP_PORTFOLIO
            and str(self.portfolio_weighting_combo.currentData() or WEIGHTING_MODE_PRESERVE)
            != WEIGHTING_MODE_PRESERVE
        ):
            idx = self.portfolio_weighting_combo.findData(WEIGHTING_MODE_PRESERVE)
            if idx >= 0:
                self.portfolio_weighting_combo.setCurrentIndex(idx)
        if self.portfolio_rebalance_drift_spin.value() <= 0.0:
            self.portfolio_rebalance_drift_spin.setValue(0.05)

    def _ensure_portfolio_vectorized_compatible(
        self,
        strategy_factory: Callable,
        timeframes: list[str],
        bt_settings: Dict[str, float | bool | dict | str],
        portfolio_strategy_blocks: Sequence[dict] | None = None,
    ) -> tuple[list[str], Dict[str, float | bool | dict | str]] | None:
        issues, can_apply_defaults = self._portfolio_vectorized_support_issues(strategy_factory, timeframes, bt_settings)
        if portfolio_strategy_blocks:
            block_issues, block_defaults = self._portfolio_strategy_block_support_issues(
                portfolio_strategy_blocks,
                bt_settings,
            )
            issues.extend(block_issues)
            can_apply_defaults = can_apply_defaults and block_defaults
        if not issues:
            return timeframes, bt_settings

        if not can_apply_defaults:
            self._show_error_dialog(
                "Portfolio Mode Unsupported",
                issues[0],
                details="\n".join(f"- {issue}" for issue in issues),
            )
            return None

        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        box.setWindowTitle("Portfolio Mode Requirements")
        box.setText("Portfolio vectorized v1 cannot run with the current settings.")
        box.setInformativeText(
            "Blockers:\n" + "\n".join(f"- {issue}" for issue in issues) + "\n\n"
            "Choose 'Apply Compatible Defaults' to switch to a portfolio-compatible setup."
        )
        apply_btn = box.addButton("Apply Compatible Defaults", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        cancel_btn = box.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(cancel_btn)
        box.exec()
        if box.clickedButton() is not apply_btn:
            return None

        self._apply_portfolio_vectorized_defaults()
        new_timeframes = [tf.strip() for tf in self.timeframes_combo.currentText().split(",") if tf.strip()]
        new_settings = self._collect_backtest_settings()
        follow_up_issues, _ = self._portfolio_vectorized_support_issues(strategy_factory, new_timeframes, new_settings)
        if portfolio_strategy_blocks:
            block_issues, _ = self._portfolio_strategy_block_support_issues(portfolio_strategy_blocks, new_settings)
            follow_up_issues.extend(block_issues)
        if follow_up_issues:
            self._show_error_dialog(
                "Portfolio Mode Still Unsupported",
                follow_up_issues[0],
                details="\n".join(f"- {issue}" for issue in follow_up_issues),
            )
            return None
        return new_timeframes, new_settings

    def _parse_portfolio_weights(
        self,
        dataset_ids: Sequence[str],
        allocation_mode: str,
        weight_text: str,
    ) -> tuple[dict[str, float], list[str]]:
        datasets = [dataset_id for dataset_id in dataset_ids if dataset_id]
        if not datasets:
            return {}, []
        if allocation_mode == PORTFOLIO_ALLOC_EQUAL:
            return {dataset_id: 1.0 for dataset_id in datasets}, []

        text = (weight_text or "").strip()
        if not text:
            return {dataset_id: 1.0 for dataset_id in datasets}, []

        tokens = [token.strip() for token in text.split(",") if token.strip()]
        if not tokens:
            return {dataset_id: 1.0 for dataset_id in datasets}, []

        weights: dict[str, float] = {}
        errors: list[str] = []
        if any("=" in token for token in tokens):
            weights = {dataset_id: 0.0 for dataset_id in datasets}
            seen: set[str] = set()
            for token in tokens:
                if "=" not in token:
                    errors.append(f"Invalid portfolio weight token '{token}'. Use dataset=weight format.")
                    continue
                dataset_id, raw_weight = token.split("=", 1)
                dataset_id = dataset_id.strip()
                if dataset_id not in datasets:
                    errors.append(f"Portfolio weight dataset '{dataset_id}' is not in the selected study datasets.")
                    continue
                if dataset_id in seen:
                    errors.append(f"Duplicate portfolio weight provided for '{dataset_id}'.")
                    continue
                try:
                    weight = float(raw_weight.strip())
                except Exception:
                    errors.append(f"Portfolio weight for '{dataset_id}' is not a valid number.")
                    continue
                if weight < 0:
                    errors.append(f"Portfolio weight for '{dataset_id}' must be non-negative.")
                    continue
                weights[dataset_id] = weight
                seen.add(dataset_id)
        else:
            if len(tokens) != len(datasets):
                return {}, [
                    "Positional portfolio weights must provide exactly one value per selected dataset, "
                    f"but received {len(tokens)} value(s) for {len(datasets)} dataset(s)."
                ]
            for dataset_id, token in zip(datasets, tokens):
                try:
                    weight = float(token)
                except Exception:
                    errors.append(f"Portfolio weight '{token}' is not a valid number.")
                    continue
                if weight < 0:
                    errors.append(f"Portfolio weight for '{dataset_id}' must be non-negative.")
                    continue
                weights[dataset_id] = weight

        if not errors and sum(weights.values()) <= 0:
            errors.append("Portfolio weights must sum to more than zero.")
        return weights, errors

    def _ensure_vectorized_compatible(
        self,
        strategy_factory: Callable,
        timeframes: list[str],
        bt_settings: Dict[str, float | bool | dict | str],
    ) -> tuple[list[str], Dict[str, float | bool | dict | str]] | None:
        issues, can_apply_defaults = self._vectorized_support_issues(strategy_factory, timeframes, bt_settings)
        if not issues:
            return timeframes, bt_settings

        if not can_apply_defaults:
            self._show_error_dialog(
                "Vectorized Mode Unsupported",
                issues[0],
                details="\n".join(f"- {issue}" for issue in issues),
            )
            return None

        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        box.setWindowTitle("Vectorized Mode Requirements")
        box.setText("Vectorized v1 cannot run with the current settings.")
        timeframe_note = ""
        if any(tf != "1 minutes" for tf in timeframes):
            timeframe_note = (
                "\nNote: explicit vectorized runs on higher timeframes currently use "
                "same-timeframe resampled bars, not full base-execution parity."
            )
        box.setInformativeText(
            "Blockers:\n" + "\n".join(f"- {issue}" for issue in issues) + "\n\n"
            "Choose 'Apply Compatible Defaults' to switch to a vectorized-compatible setup."
            + timeframe_note
        )
        apply_btn = box.addButton("Apply Compatible Defaults", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        cancel_btn = box.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(cancel_btn)
        box.exec()
        if box.clickedButton() is not apply_btn:
            return None

        self._apply_vectorized_defaults()
        new_timeframes = [tf.strip() for tf in self.timeframes_combo.currentText().split(",") if tf.strip()]
        new_settings = self._collect_backtest_settings()
        follow_up_issues, _ = self._vectorized_support_issues(strategy_factory, new_timeframes, new_settings)
        if follow_up_issues:
            self._show_error_dialog(
                "Vectorized Mode Still Unsupported",
                follow_up_issues[0],
                details="\n".join(f"- {issue}" for issue in follow_up_issues),
            )
            return None
        return new_timeframes, new_settings

    def _run_grid_clicked(self) -> None:
        csv_path = Path(self.csv_path_edit.text().strip())
        portfolio_strategy_blocks = [dict(block) for block in self.portfolio_strategy_blocks]
        dataset_ids = (
            self._portfolio_strategy_block_dataset_ids()
            if portfolio_strategy_blocks and str(self.study_mode_combo.currentData() or STUDY_MODE_INDEPENDENT) == STUDY_MODE_PORTFOLIO
            else self._selected_study_dataset_ids()
        )
        dataset_id = dataset_ids[0] if dataset_ids else (self.dataset_combo.currentText().strip() or "dataset")
        study_mode = str(self.study_mode_combo.currentData() or STUDY_MODE_INDEPENDENT)
        timeframes = [tf.strip() for tf in self.timeframes_combo.currentText().split(",") if tf.strip()]
        horizons_raw = [h.strip() for h in self.horizons_combo.currentText().split(",") if h.strip()]
        risk_free_raw = self.risk_free_edit.text().strip()
        try:
            risk_free_rate = float(risk_free_raw) if risk_free_raw else 0.0
        except Exception:
            risk_free_rate = 0.0
        bt_settings = self._collect_backtest_settings()
        if self.worker and self.worker.isRunning():
            QtWidgets.QMessageBox.information(self, "Busy", "Grid already running.")
            return
        if not dataset_ids:
            QtWidgets.QMessageBox.warning(self, "Dataset missing", "Select at least one dataset.")
            return
        # Ensure dataset already imported.
        duck = DuckDBStore()
        missing_datasets = []
        for selected_dataset in dataset_ids:
            try:
                duck.load(selected_dataset)
            except Exception:
                missing_datasets.append(selected_dataset)
        if missing_datasets:
            QtWidgets.QMessageBox.warning(
                self,
                "Dataset missing",
                "These datasets were not found in the local store:\n" + "\n".join(missing_datasets),
            )
            return
        if not timeframes:
            QtWidgets.QMessageBox.warning(self, "Timeframes", "Provide at least one timeframe.")
            return

        strategy_factory, strat_params, param_errors = self._collect_strategy_params()
        if param_errors:
            QtWidgets.QMessageBox.warning(self, "Parameter warnings", "Some values were invalid and replaced with defaults:\n" + "\n".join(param_errors))

        requested_mode = str(bt_settings.get("execution_mode", ExecutionMode.AUTO.value))
        if study_mode == STUDY_MODE_PORTFOLIO:
            compatible = self._ensure_portfolio_vectorized_compatible(
                SMACrossStrategy if portfolio_strategy_blocks else strategy_factory,
                timeframes,
                bt_settings,
                portfolio_strategy_blocks=portfolio_strategy_blocks,
            )
            if compatible is None:
                self.status_label.setText("Portfolio study cancelled")
                return
            timeframes, bt_settings = compatible
            allocation_mode = str(bt_settings.get("portfolio_allocation_mode", PORTFOLIO_ALLOC_EQUAL))
            target_weights, weight_errors = self._parse_portfolio_weights(
                dataset_ids,
                allocation_mode,
                str(bt_settings.get("portfolio_weights_text", "")),
            )
            if weight_errors:
                self._show_error_dialog(
                    "Portfolio Weights Invalid",
                    weight_errors[0],
                    details="\n".join(f"- {issue}" for issue in weight_errors),
                )
                self.status_label.setText("Portfolio study cancelled")
                return
            self.portfolio_target_weights = dict(target_weights)
            bt_settings["portfolio_target_weights"] = target_weights
            bt_settings["portfolio_strategy_blocks"] = portfolio_strategy_blocks
        elif requested_mode == ExecutionMode.VECTORIZED.value:
            compatible = self._ensure_vectorized_compatible(strategy_factory, timeframes, bt_settings)
            if compatible is None:
                self.status_label.setText("Vectorized run cancelled")
                return
            timeframes, bt_settings = compatible

        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress.setValue(0)
        mode_label = self._execution_mode_label(str(bt_settings.get("execution_mode", ExecutionMode.AUTO.value)))
        self._current_grid_started_at = time.perf_counter()
        if study_mode == STUDY_MODE_PORTFOLIO:
            self.status_label.setText(f"Running portfolio study ({mode_label}) across {len(dataset_ids)} datasets…")
        elif len(dataset_ids) > 1:
            self.status_label.setText(f"Running independent study ({mode_label}) across {len(dataset_ids)} datasets…")
        else:
            self.status_label.setText(f"Running grid ({mode_label})…")

        self.worker = GridWorker(
            csv_path=csv_path,
            dataset_id=dataset_id,
            dataset_ids=dataset_ids,
            study_mode=study_mode,
            timeframes=timeframes,
            horizons=horizons_raw,
            catalog_path=self.catalog.db_path,
            strategy_factory=strategy_factory,
            strategy_params=strat_params,
            blas_threads=self._desired_blas_threads(),
            intrabar_sim=self.intrabar_chk.isChecked(),
            sharpe_debug=self.sharpe_debug_chk.isChecked(),
            risk_free_rate=risk_free_rate,
            bt_settings=bt_settings,
            portfolio_strategy_blocks=portfolio_strategy_blocks,
        )
        self.worker.finished_signal.connect(self._grid_finished)
        self.worker.error_signal.connect(self._grid_error)
        self.worker.progress_signal.connect(self._grid_progress)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.start()

    def _grid_finished(self, payload) -> None:
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        message = "Grid completed."
        df = None
        spec = None
        batch_id = None
        batch_benchmarks: tuple[BatchExecutionBenchmark, ...] = ()
        if isinstance(payload, dict):
            message = payload.get("message", message)
            df = payload.get("df")
            spec = payload.get("spec")
            batch_id = payload.get("batch_id")
            batch_benchmarks = tuple(payload.get("batch_benchmarks") or ())
        elapsed_text = ""
        if self._current_grid_started_at is not None:
            elapsed_text = f" in {time.perf_counter() - self._current_grid_started_at:.2f}s"
            self._current_grid_started_at = None
        if elapsed_text and message.endswith("."):
            message = f"{message[:-1]}{elapsed_text}."
        elif elapsed_text:
            message = f"{message}{elapsed_text}"
        if isinstance(df, pd.DataFrame) and not df.empty and "resolved_execution_mode" in df.columns:
            mode_counts = df["resolved_execution_mode"].fillna("unknown").value_counts().to_dict()
            mode_summary = ", ".join(
                f"{self._execution_mode_label(mode)}: {count}" for mode, count in sorted(mode_counts.items())
            )
            if mode_summary:
                message = f"{message} Engines used -> {mode_summary}"
            if "dataset_id" in df.columns:
                dataset_count = int(df["dataset_id"].nunique())
                if dataset_count > 1:
                    message = f"{message} Datasets -> {dataset_count}"
        if batch_id and batch_benchmarks:
            self._batch_benchmark_cache[str(batch_id)] = batch_benchmarks
            message = f"{message} {_summarize_batch_benchmarks(batch_benchmarks)}"
        self.status_label.setText(message)
        self.progress.setValue(100)
        if df is not None and spec is not None:
            self._generate_heatmap_from_results(df, spec)
        self.refresh()
        self.worker = None

    def _grid_error(self, message: str) -> None:
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress.setValue(0)
        if self._current_grid_started_at is not None:
            elapsed = time.perf_counter() - self._current_grid_started_at
            self.status_label.setText(f"Grid error after {elapsed:.2f}s")
            self._current_grid_started_at = None
        else:
            self.status_label.setText("Grid error")
        # Echo to console for debugging heavy workloads.
        print("Grid error:\n", message or "Unknown error")
        summary = self._summarize_error(message)
        self._show_error_dialog("Grid Error", summary, details=message)
        self.worker = None

    def _grid_progress(self, done: int, total: int) -> None:
        if total <= 0:
            self.progress.setValue(0)
            return
        pct = int((done / total) * 100)
        self.progress.setValue(min(100, max(0, pct)))
        mode_label = "Auto"
        study_prefix = "Running grid"
        if self.worker is not None:
            mode_label = self._execution_mode_label(str(self.worker.bt_settings.get("execution_mode", ExecutionMode.AUTO.value)))
            if getattr(self.worker, "study_mode", STUDY_MODE_INDEPENDENT) == STUDY_MODE_PORTFOLIO:
                study_prefix = f"Running portfolio study across {len(self.worker.dataset_ids)} datasets"
            elif len(getattr(self.worker, "dataset_ids", [])) > 1:
                study_prefix = f"Running independent study across {len(self.worker.dataset_ids)} datasets"
        self.status_label.setText(f"{study_prefix} ({mode_label})… {done}/{total}")

    def _generate_heatmap_from_results(self, df: pd.DataFrame, spec: GridSpec) -> None:
        try:
            if df is None or df.empty:
                return
            param_keys = list(spec.params.keys()) if getattr(spec, "params", None) else []
            if len(param_keys) < 2:
                return
            col = spec.heatmap_cols or param_keys[0]
            row = spec.heatmap_rows or param_keys[1]
            metric = spec.metric or "total_return"
            if col not in df.columns or row not in df.columns or metric not in df.columns:
                return
            heatmap_df = df
            if heatmap_df.duplicated(subset=[row, col]).any():
                heatmap_df = (
                    heatmap_df.groupby([row, col], as_index=False)[metric]
                    .median()
                    .sort_values([row, col])
                    .reset_index(drop=True)
                )
            fig = plot_param_heatmap(heatmap_df, value_col=metric, row=row, col=col, title=f"{metric} heatmap")
            heatmap_id = f"heatmap_{uuid.uuid4().hex[:8]}"
            heatmap_dir = Path("heatmaps")
            heatmap_dir.mkdir(parents=True, exist_ok=True)
            file_path = heatmap_dir / f"{heatmap_id}.png"
            fig.savefig(file_path)
            # Persist heatmap metadata
            rc = ResultCatalog(self.catalog.db_path)
            rc.save_heatmap(
                heatmap_id=heatmap_id,
                params={"params": spec.params, "metric": metric},
                file_path=str(file_path),
                description=spec.description or "",
            )
            plt = __import__("matplotlib.pyplot", fromlist=["close"])
            plt.close(fig)
        except Exception:
            # Silent failure to avoid crashing UI; heatmap is optional post-step.
            pass

    def _stop_grid_clicked(self) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.request_stop()
            self.status_label.setText("Stop requested…")
            self.stop_btn.setEnabled(False)

    def _desired_blas_threads(self) -> int:
        # Use cpu_count() - 1 to leave a little headroom for UI/OS.
        return max(1, (os.cpu_count() or 2) - 1)

    @staticmethod
    def _execution_mode_label(mode: str | None) -> str:
        if not mode:
            return "—"
        try:
            return ExecutionMode.from_value(mode).value.title()
        except Exception:
            return str(mode).replace("_", " ").title()

    def _portfolio_strategy_block_dataset_ids(self) -> list[str]:
        dataset_ids: list[str] = []
        seen: set[str] = set()
        for block in self.portfolio_strategy_blocks:
            for dataset_id in list(block.get("asset_dataset_ids") or []):
                if dataset_id and dataset_id not in seen:
                    dataset_ids.append(str(dataset_id))
                    seen.add(str(dataset_id))
        return dataset_ids

    def _edit_portfolio_strategy_blocks(self) -> None:
        available = self._available_dataset_ids()
        if not available:
            QtWidgets.QMessageBox.information(self, "No datasets", "No datasets are available in the local store yet.")
            return
        dlg = StrategyBlockEditorDialog(
            available,
            self.strategy_specs,
            self.portfolio_strategy_blocks,
            self.universes,
            self.study_universe_id,
            self,
        )
        if dlg.exec() != int(QtWidgets.QDialog.DialogCode.Accepted):
            return
        self.portfolio_strategy_blocks = dlg.strategy_blocks()
        block_dataset_ids = self._portfolio_strategy_block_dataset_ids()
        self.study_dataset_ids = block_dataset_ids
        self._update_study_dataset_summary()
        self._update_portfolio_strategy_block_summary()
        self._update_portfolio_allocation_summary()

    def _clear_portfolio_strategy_blocks(self) -> None:
        self.portfolio_strategy_blocks = []
        self._update_portfolio_strategy_block_summary()
        self._update_portfolio_allocation_summary()

    def _update_portfolio_strategy_block_summary(self) -> None:
        if not hasattr(self, "portfolio_blocks_summary"):
            return
        if not self.portfolio_strategy_blocks:
            self.portfolio_blocks_summary.setText("")
            self.portfolio_blocks_summary.setPlaceholderText("No explicit strategy blocks configured")
            self.portfolio_blocks_summary.setToolTip("")
            self.portfolio_blocks_note.setText(
                "Leave this empty to use the current single-strategy portfolio flow across the selected datasets. "
                "Add blocks to build a true multi-strategy portfolio."
            )
            return
        names = [str(block.get("display_name") or block.get("block_id") or "Strategy Block") for block in self.portfolio_strategy_blocks]
        if len(names) <= 2:
            label = ", ".join(names)
        else:
            label = f"{len(names)} blocks configured"
        self.portfolio_blocks_summary.setText(label)
        self.portfolio_blocks_summary.setToolTip("\n".join(names))
        datasets = self._portfolio_strategy_block_dataset_ids()
        dataset_note = f" Assets: {len(datasets)} attached dataset(s)." if datasets else ""
        self.portfolio_blocks_note.setText(
            "Explicit strategy blocks are configured. In portfolio mode, these blocks override the single-strategy controls for run execution."
            + dataset_note
        )

    def _update_study_mode_note(self) -> None:
        if not hasattr(self, "study_mode_note"):
            return
        mode = str(self.study_mode_combo.currentData() or STUDY_MODE_INDEPENDENT)
        self.portfolio_allocation_combo.setEnabled(mode == STUDY_MODE_PORTFOLIO)
        self.portfolio_ownership_combo.setEnabled(mode == STUDY_MODE_PORTFOLIO)
        self.portfolio_ranking_combo.setEnabled(mode == STUDY_MODE_PORTFOLIO)
        self.portfolio_rank_count_spin.setEnabled(mode == STUDY_MODE_PORTFOLIO)
        self.portfolio_score_threshold_spin.setEnabled(mode == STUDY_MODE_PORTFOLIO)
        self.portfolio_weighting_combo.setEnabled(mode == STUDY_MODE_PORTFOLIO)
        self.portfolio_min_active_weight_spin.setEnabled(mode == STUDY_MODE_PORTFOLIO)
        self.portfolio_max_asset_weight_spin.setEnabled(mode == STUDY_MODE_PORTFOLIO)
        self.portfolio_cash_reserve_spin.setEnabled(mode == STUDY_MODE_PORTFOLIO)
        self.portfolio_rebalance_combo.setEnabled(mode == STUDY_MODE_PORTFOLIO)
        self.portfolio_rebalance_every_spin.setEnabled(mode == STUDY_MODE_PORTFOLIO)
        self.portfolio_rebalance_drift_spin.setEnabled(mode == STUDY_MODE_PORTFOLIO)
        if mode == STUDY_MODE_PORTFOLIO:
            block_note = ""
            if self.portfolio_strategy_blocks:
                block_note = " Explicit strategy blocks are configured and will override the single-strategy controls."
            self.study_mode_note.setText(
                "Portfolio mode uses shared cash across the selected datasets with the vectorized portfolio backend. "
                "Current scope is same-timeframe only, and unsupported semantics still fail clearly because portfolio reference fallback is not built yet."
                + block_note
            )
        else:
            self.study_mode_note.setText(
                "Optional. Choose multiple datasets to run one independent multi-dataset study. "
                "Each asset is backtested separately; this is not portfolio backtesting."
            )
        self._update_portfolio_allocation_summary()

    def _update_portfolio_allocation_summary(self) -> None:
        if not hasattr(self, "portfolio_allocation_summary"):
            return
        study_mode = str(self.study_mode_combo.currentData() or STUDY_MODE_INDEPENDENT)
        allocation_mode = str(self.portfolio_allocation_combo.currentData() or PORTFOLIO_ALLOC_EQUAL)
        ownership_mode = str(self.portfolio_ownership_combo.currentData() or ALLOCATION_OWNERSHIP_STRATEGY)
        ranking_mode = str(self.portfolio_ranking_combo.currentData() or RANKING_MODE_NONE)
        weighting_mode = str(self.portfolio_weighting_combo.currentData() or WEIGHTING_MODE_PRESERVE)
        min_active_weight = float(self.portfolio_min_active_weight_spin.value())
        max_asset_weight = float(self.portfolio_max_asset_weight_spin.value())
        cash_reserve_weight = float(self.portfolio_cash_reserve_spin.value())
        rebalance_mode = str(self.portfolio_rebalance_combo.currentData() or REBALANCE_MODE_ON_CHANGE)
        self.portfolio_rank_count_spin.setEnabled(
            study_mode == STUDY_MODE_PORTFOLIO
            and ranking_mode in {RANKING_MODE_TOP_N, RANKING_MODE_TOP_N_OVER_SCORE_THRESHOLD}
        )
        self.portfolio_score_threshold_spin.setEnabled(
            study_mode == STUDY_MODE_PORTFOLIO
            and ranking_mode in {RANKING_MODE_SCORE_THRESHOLD, RANKING_MODE_TOP_N_OVER_SCORE_THRESHOLD}
        )
        self.portfolio_rebalance_every_spin.setEnabled(
            study_mode == STUDY_MODE_PORTFOLIO
            and rebalance_mode in {
                REBALANCE_MODE_ON_CHANGE_OR_PERIODIC,
                REBALANCE_MODE_ON_CHANGE_OR_PERIODIC_OR_DRIFT,
            }
        )
        self.portfolio_rebalance_drift_spin.setEnabled(
            study_mode == STUDY_MODE_PORTFOLIO
            and rebalance_mode in {
                REBALANCE_MODE_ON_CHANGE_OR_DRIFT,
                REBALANCE_MODE_ON_CHANGE_OR_PERIODIC_OR_DRIFT,
            }
        )
        dataset_ids = self._selected_study_dataset_ids()
        self.portfolio_weights_edit.setEnabled(
            study_mode == STUDY_MODE_PORTFOLIO and allocation_mode != PORTFOLIO_ALLOC_EQUAL
        )
        if study_mode != STUDY_MODE_PORTFOLIO:
            self.portfolio_allocation_summary.setText(
                "Portfolio allocation settings apply only when Study Mode is set to Portfolio."
            )
            return
        if self.portfolio_strategy_blocks:
            self.portfolio_allocation_summary.setText(
                "Explicit strategy blocks are configured. Allocation, ranking, weighting, and rebalance settings will apply at the portfolio layer on top of those blocks."
            )
            return
        if not dataset_ids:
            self.portfolio_allocation_summary.setText("Select one or more datasets for the portfolio study.")
            return
        if ownership_mode == ALLOCATION_OWNERSHIP_STRATEGY and ranking_mode != RANKING_MODE_NONE:
            self.portfolio_allocation_summary.setText(
                "Strategy-Owned allocation cannot be combined with portfolio ranking. Switch to Hybrid or Portfolio-Owned."
            )
            return
        if ownership_mode != ALLOCATION_OWNERSHIP_PORTFOLIO and weighting_mode != WEIGHTING_MODE_PRESERVE:
            self.portfolio_allocation_summary.setText(
                "Portfolio weighting overrides require Portfolio-Owned allocation. Strategy-Owned and Hybrid modes must preserve strategy sizing."
            )
            return
        if min_active_weight > 0.0 and max_asset_weight > 0.0 and max_asset_weight < min_active_weight:
            self.portfolio_allocation_summary.setText(
                "Max Asset Weight must be greater than or equal to Min Active Weight."
            )
            return
        if allocation_mode == PORTFOLIO_ALLOC_EQUAL:
            allocation_text = f"Equal-weight portfolio across {len(dataset_ids)} selected dataset(s)."
        else:
            weights, errors = self._parse_portfolio_weights(
                dataset_ids,
                allocation_mode,
                self.portfolio_weights_edit.text().strip(),
            )
            if errors:
                self.portfolio_allocation_summary.setText(errors[0])
                return
            nonzero = [(dataset_id, weight) for dataset_id, weight in weights.items() if weight > 0]
            preview = ", ".join(f"{dataset_id}={weight:g}" for dataset_id, weight in nonzero[:4]) or "all zero"
            if len(nonzero) > 4:
                preview = f"{preview}, ..."
            if allocation_mode == PORTFOLIO_ALLOC_RELATIVE:
                allocation_text = (
                    f"Relative weights will be normalized across the selected datasets. Current weights: {preview}"
                )
            else:
                allocation_text = (
                    f"Fixed weights will be applied as absolute targets before gross-exposure capping. Current weights: {preview}"
                )
        ranking_text = "No cross-asset ranking."
        if ranking_mode == RANKING_MODE_TOP_N:
            ranking_text = f"Top-N ranking keeps the strongest {int(self.portfolio_rank_count_spin.value())} asset(s)."
        elif ranking_mode == RANKING_MODE_SCORE_THRESHOLD:
            ranking_text = (
                f"Score-threshold ranking keeps only assets with score >= {float(self.portfolio_score_threshold_spin.value()):g}."
            )
        elif ranking_mode == RANKING_MODE_TOP_N_OVER_SCORE_THRESHOLD:
            ranking_text = (
                f"Top-N-over-threshold ranking keeps up to {int(self.portfolio_rank_count_spin.value())} asset(s) "
                f"with score >= {float(self.portfolio_score_threshold_spin.value()):g}."
            )
        weighting_text = "Selected assets preserve their underlying strategy or target weights."
        if weighting_mode == WEIGHTING_MODE_EQUAL_SELECTED:
            weighting_text = "Selected assets are equal-weighted by the portfolio layer."
        elif weighting_mode == WEIGHTING_MODE_SCORE_PROPORTIONAL:
            weighting_text = "Selected assets are weighted in proportion to their portfolio score."
        constraint_parts: list[str] = []
        if min_active_weight > 0.0:
            constraint_parts.append(f"drop allocations below {min_active_weight:g}")
        if max_asset_weight > 0.0:
            constraint_parts.append(f"cap any single asset at {max_asset_weight:g}")
        if cash_reserve_weight > 0.0:
            constraint_parts.append(f"hold {cash_reserve_weight:g} as cash reserve")
        constraint_text = "No additional portfolio construction constraints."
        if constraint_parts:
            constraint_text = "Construction constraints: " + ", ".join(constraint_parts) + "."
        rebalance_text = "Rebalance on target change only."
        if rebalance_mode == REBALANCE_MODE_ON_CHANGE_OR_PERIODIC:
            rebalance_text = (
                f"Also rebalance every {int(self.portfolio_rebalance_every_spin.value())} bar(s) to manage drift."
            )
        elif rebalance_mode == REBALANCE_MODE_ON_CHANGE_OR_DRIFT:
            rebalance_text = (
                f"Also rebalance when any asset drifts by at least {float(self.portfolio_rebalance_drift_spin.value()):g} weight."
            )
        elif rebalance_mode == REBALANCE_MODE_ON_CHANGE_OR_PERIODIC_OR_DRIFT:
            rebalance_text = (
                f"Also rebalance every {int(self.portfolio_rebalance_every_spin.value())} bar(s) and "
                f"whenever any asset drifts by at least {float(self.portfolio_rebalance_drift_spin.value()):g} weight."
            )
        ownership_text = {
            ALLOCATION_OWNERSHIP_STRATEGY: "Strategy-Owned allocation preserves strategy sizing.",
            ALLOCATION_OWNERSHIP_PORTFOLIO: "Portfolio-Owned allocation uses strategy candidates and portfolio controls.",
            ALLOCATION_OWNERSHIP_HYBRID: "Hybrid allocation keeps strategy weights but applies explicit portfolio filters.",
        }.get(ownership_mode, "Unknown ownership mode.")
        self.portfolio_allocation_summary.setText(
            f"{ownership_text} {allocation_text} {ranking_text} {weighting_text} {constraint_text} {rebalance_text}"
        )

    def _refresh_dataset_options(self, select_id: str | None = None) -> None:
        try:
            opts = self._available_dataset_ids()
            current_text = self.dataset_combo.currentText().strip() if hasattr(self, "dataset_combo") else ""
            self.dataset_combo.blockSignals(True)
            self.dataset_combo.clear()
            self.dataset_combo.addItems(opts)
            self.dataset_combo.blockSignals(False)
            target_text = (select_id or current_text).strip()
            if target_text:
                self.dataset_combo.setCurrentText(target_text)
            elif opts:
                self.dataset_combo.setCurrentIndex(0)
            self.study_dataset_ids = [dataset_id for dataset_id in self.study_dataset_ids if dataset_id in opts]
            self.portfolio_target_weights = {
                dataset_id: weight
                for dataset_id, weight in self.portfolio_target_weights.items()
                if dataset_id in opts
            }
            self._update_study_dataset_summary()
            self._refresh_universe_dataset_summary()
        except Exception:
            # Best-effort; leave combo as-is if listing fails.
            pass

    def _available_dataset_ids(self) -> list[str]:
        store = DuckDBStore()
        opts = []
        for path in store.data_dir.glob("*.parquet"):
            if path.is_file():
                opts.append(path.stem)
        return sorted(set(opts))

    def _selected_study_dataset_ids(self, *, manual_only: bool = False) -> list[str]:
        if not manual_only:
            universe = self._selected_study_universe()
            if universe is not None:
                dataset_ids = [
                    str(dataset_id)
                    for dataset_id in list(universe.get("dataset_ids") or [])
                    if str(dataset_id).strip()
                ]
                if dataset_ids:
                    return dataset_ids
        explicit = [dataset_id for dataset_id in self.study_dataset_ids if dataset_id]
        if explicit:
            return explicit
        current = self.dataset_combo.currentText().strip()
        return [current] if current else []

    def _update_study_dataset_summary(self) -> None:
        if not hasattr(self, "study_dataset_summary"):
            return
        universe = self._selected_study_universe()
        if hasattr(self, "study_universe_note"):
            if universe is None:
                self.study_universe_note.setText("Manual dataset selection is active.")
            else:
                self.study_universe_note.setText(
                    f"Universe '{universe.get('name', '')}' contributes "
                    f"{len(list(universe.get('dataset_ids') or []))} dataset(s) and "
                    f"{len(list(universe.get('symbols') or []))} symbol(s)."
                )
        dataset_ids = self._selected_study_dataset_ids()
        if not dataset_ids:
            self.study_dataset_summary.setText("")
            self.study_dataset_summary.setToolTip("")
            self._update_portfolio_allocation_summary()
            return
        if universe is not None:
            label = f"Universe: {universe.get('name', '')} ({len(dataset_ids)} datasets)"
        elif len(dataset_ids) == 1 and not self.study_dataset_ids:
            label = f"Current dataset only: {dataset_ids[0]}"
        elif len(dataset_ids) <= 3:
            label = ", ".join(dataset_ids)
        else:
            label = f"{len(dataset_ids)} datasets selected"
        self.study_dataset_summary.setText(label)
        self.study_dataset_summary.setToolTip("\n".join(dataset_ids))
        self._update_portfolio_allocation_summary()

    def _choose_study_datasets(self) -> None:
        available = self._available_dataset_ids()
        if not available:
            QtWidgets.QMessageBox.information(self, "No datasets", "No datasets are available in the local store yet.")
            return
        initial = self._selected_study_dataset_ids()
        dlg = DatasetSelectionDialog(available, initial, self)
        if dlg.exec() != int(QtWidgets.QDialog.DialogCode.Accepted):
            return
        selected = dlg.selected_datasets()
        current = self.dataset_combo.currentText().strip()
        self.study_universe_id = ""
        if hasattr(self, "study_universe_combo"):
            self.study_universe_combo.blockSignals(True)
            self.study_universe_combo.setCurrentIndex(0)
            self.study_universe_combo.blockSignals(False)
        if len(selected) == 1 and selected[0] == current:
            self.study_dataset_ids = []
        else:
            self.study_dataset_ids = selected
        self._update_study_dataset_summary()

    def _reset_study_datasets_to_current(self) -> None:
        self.study_universe_id = ""
        if hasattr(self, "study_universe_combo"):
            self.study_universe_combo.blockSignals(True)
            self.study_universe_combo.setCurrentIndex(0)
            self.study_universe_combo.blockSignals(False)
        self.study_dataset_ids = []
        self._update_study_dataset_summary()

    def _on_study_universe_changed(self) -> None:
        if not hasattr(self, "study_universe_combo"):
            return
        self.study_universe_id = str(self.study_universe_combo.currentData() or "")
        self._update_study_dataset_summary()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._closing = True
        if self._magellan_warm_timer.isActive():
            self._magellan_warm_timer.stop()
        if self.worker and self.worker.isRunning():
            self.worker.quit()
            self.worker.wait(2000)
        self.magellan.shutdown()
        event.accept()

    def _warm_magellan(self) -> None:
        if self._closing:
            return
        try:
            self.magellan.ensure_running(timeout_ms=2500)
        except Exception:
            # Keep warmup best-effort so the dashboard still opens even if Magellan is unavailable.
            pass

    def _snapshot_root_for_run(self, run: "RunRow") -> Path:
        return self.snapshot_exporter.root_dir / str(run.run_id)

    def _existing_snapshot_root_for_run(self, run: "RunRow") -> Path | None:
        snapshot_root = self._snapshot_root_for_run(run)
        manifest_path = snapshot_root / "manifest.json"
        if manifest_path.exists():
            return snapshot_root.resolve()
        return None

    def _existing_portfolio_snapshot_roots_for_run(self, run: "RunRow") -> list[Path]:
        snapshot_root = self._snapshot_root_for_run(run)
        assets_root = snapshot_root / "assets"
        if not assets_root.exists():
            return []
        roots = [
            child.resolve()
            for child in sorted(assets_root.iterdir(), key=lambda item: item.name.lower())
            if child.is_dir() and (child / "manifest.json").exists()
        ]
        return roots

    @staticmethod
    def _portfolio_source_bars_from_request(request: PortfolioExecutionRequest) -> dict[str, pd.DataFrame]:
        source_bars: dict[str, pd.DataFrame] = {}
        for asset in list(request.assets or []):
            dataset_id = str(asset.dataset_id)
            if dataset_id not in source_bars:
                source_bars[dataset_id] = asset.data
        for block in list(request.strategy_blocks or []):
            for asset in list(block.assets or []):
                dataset_id = str(asset.dataset_id)
                if dataset_id not in source_bars:
                    source_bars[dataset_id] = asset.data
        return source_bars

    def _rebuild_portfolio_request_for_run(self, run: "RunRow") -> PortfolioExecutionRequest:
        params = json.loads(run.params) if isinstance(run.params, str) else (run.params or {})
        assets_payload = params.get("assets") or []
        strategy_blocks_payload = params.get("strategy_blocks") or []
        if not assets_payload and not strategy_blocks_payload:
            raise ValueError(
                "This portfolio run does not contain asset or strategy-block definitions needed to rebuild a chart snapshot."
            )

        execution_cfg = dict(params.get("execution_config") or {})
        construction_cfg = dict(params.get("construction_config") or {})
        timeframe = str(run.timeframe or "1 minutes")
        duck = DuckDBStore()
        assets: list[PortfolioExecutionAsset] = []
        strategy_blocks: list[PortfolioExecutionStrategyBlock] = []
        if strategy_blocks_payload:
            for block_payload in strategy_blocks_payload:
                block_id = str(block_payload.get("block_id") or "").strip()
                if not block_id:
                    raise ValueError("A saved portfolio strategy block is missing block_id.")
                strategy_cls = RunChartDialog._strategy_class_static(str(block_payload.get("strategy") or ""))
                block_assets: list[PortfolioExecutionStrategyBlockAsset] = []
                for asset_payload in list(block_payload.get("assets") or []):
                    dataset_id = str(asset_payload.get("dataset_id") or "").strip()
                    if not dataset_id:
                        raise ValueError(f"Strategy block '{block_id}' is missing an asset dataset_id.")
                    data = duck.resample(dataset_id, timeframe)
                    if data is None or data.empty:
                        raise ValueError(
                            f"Historical data is unavailable for portfolio asset '{dataset_id}' at timeframe '{timeframe}'."
                        )
                    block_assets.append(
                        PortfolioExecutionStrategyBlockAsset(
                            dataset_id=dataset_id,
                            data=data,
                            target_weight=asset_payload.get("target_weight"),
                            display_name=asset_payload.get("display_name"),
                        )
                    )
                strategy_blocks.append(
                    PortfolioExecutionStrategyBlock(
                        block_id=block_id,
                        strategy_cls=strategy_cls,
                        strategy_params=dict(block_payload.get("params") or {}),
                        assets=tuple(block_assets),
                        budget_weight=block_payload.get("budget_weight"),
                        display_name=block_payload.get("display_name"),
                    )
                )
        else:
            for asset_payload in assets_payload:
                dataset_id = str(asset_payload.get("dataset_id") or "").strip()
                if not dataset_id:
                    raise ValueError("A saved portfolio asset is missing dataset_id.")
                data = duck.resample(dataset_id, timeframe)
                if data is None or data.empty:
                    raise ValueError(
                        f"Historical data is unavailable for portfolio asset '{dataset_id}' at timeframe '{timeframe}'."
                    )
                strategy_cls = RunChartDialog._strategy_class_static(str(asset_payload.get("strategy") or ""))
                assets.append(
                    PortfolioExecutionAsset(
                        dataset_id=dataset_id,
                        data=data,
                        strategy_cls=strategy_cls,
                        strategy_params=dict(asset_payload.get("params") or {}),
                        target_weight=asset_payload.get("target_weight"),
                        display_name=asset_payload.get("display_name"),
                    )
                )

        current_bt_settings = self._collect_backtest_settings()
        config = BacktestConfig(
            timeframe=timeframe,
            starting_cash=(
                float(run.starting_cash)
                if getattr(run, "starting_cash", None) is not None
                else float(execution_cfg.get("starting_cash", 100_000.0))
            ),
            fee_rate=float(execution_cfg.get("fee_rate", current_bt_settings.get("fee_rate", 0.0002))),
            fee_schedule=execution_cfg.get(
                "fee_schedule",
                current_bt_settings.get("fee_schedule", {"buy": 0.0003, "sell": 0.0005}),
            ),
            slippage=float(execution_cfg.get("slippage", current_bt_settings.get("slippage", 0.0002))),
            slippage_schedule=execution_cfg.get(
                "slippage_schedule",
                current_bt_settings.get("slippage_schedule", {"buy": 0.0003, "sell": 0.0001}),
            ),
            borrow_rate=float(execution_cfg.get("borrow_rate", current_bt_settings.get("borrow_rate", 0.0))),
            fill_ratio=float(execution_cfg.get("fill_ratio", current_bt_settings.get("fill_ratio", 1.0))),
            fill_on_close=bool(execution_cfg.get("fill_on_close", current_bt_settings.get("fill_on_close", False))),
            recalc_on_fill=bool(execution_cfg.get("recalc_on_fill", current_bt_settings.get("recalc_on_fill", True))),
            allow_short=bool(execution_cfg.get("allow_short", current_bt_settings.get("allow_short", False))),
            use_cache=False,
            base_execution=False,
            time_horizon_start=str(run.start) if getattr(run, "start", None) else None,
            time_horizon_end=str(run.end) if getattr(run, "end", None) else None,
            prevent_scale_in=bool(
                execution_cfg.get("prevent_scale_in", current_bt_settings.get("prevent_scale_in", True))
            ),
            one_order_per_signal=bool(
                execution_cfg.get("one_order_per_signal", current_bt_settings.get("one_order_per_signal", True))
            ),
            risk_free_rate=float(execution_cfg.get("risk_free_rate", current_bt_settings.get("risk_free_rate", 0.0))),
        )
        return PortfolioExecutionRequest(
            assets=assets,
            config=config,
            strategy_blocks=tuple(strategy_blocks),
            catalog=None,
            requested_execution_mode=ExecutionMode.VECTORIZED,
            normalize_weights=bool(params.get("normalize_weights", True)),
            portfolio_dataset_id=str(run.dataset_id),
            construction_config=PortfolioConstructionConfig(**construction_cfg),
            logical_run_id=str(getattr(run, "logical_run_id", "") or ""),
        )

    def _rebuild_portfolio_result_for_run(self, run: "RunRow"):
        request = self._rebuild_portfolio_request_for_run(run)
        orchestrator = ExecutionOrchestrator()
        portfolio_result = orchestrator.execute_portfolio(
            request
        )
        return portfolio_result.result

    def _ensure_portfolio_snapshot_roots(self, run: "RunRow") -> list[Path]:
        existing = self._existing_portfolio_snapshot_roots_for_run(run)
        if existing:
            return existing
        request = self._rebuild_portfolio_request_for_run(run)
        orchestrator = ExecutionOrchestrator()
        portfolio_result = orchestrator.execute_portfolio(request).result
        artifacts = self.snapshot_exporter.export_portfolio_asset_snapshots(
            run=run,
            portfolio_result=portfolio_result,
            source_bars=self._portfolio_source_bars_from_request(request),
            strategy_contexts=ChartSnapshotExporter.build_portfolio_strategy_contexts(request),
            overwrite=True,
        )
        return [artifact.snapshot_root for artifact in artifacts]

    def _build_portfolio_chart_payload_for_run(self, run: "RunRow"):
        portfolio_result = self._rebuild_portfolio_result_for_run(run)
        report = summarize_portfolio_result(
            portfolio_result,
            starting_cash=(
                float(run.starting_cash) if getattr(run, "starting_cash", None) is not None else None
            ),
        )
        return portfolio_result, report

    def _build_portfolio_report_for_run(self, run: "RunRow"):
        _, report = self._build_portfolio_chart_payload_for_run(run)
        return report

    def _open_run_chart_in_magellan(self, run: "RunRow") -> bool:
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            if (run.engine_impl or "").lower() == "vectorized_portfolio":
                snapshot_roots = self._ensure_portfolio_snapshot_roots(run)
            else:
                bars = RunChartDialog._load_bars_utc_static(run)
                if bars is None or bars.empty:
                    raise MagellanError(RunChartDialog._missing_run_data_message_static(run))

                overlays, panes, series_styles = RunChartDialog._build_snapshot_series_static(run, bars)
                trades_df = ChartSnapshotExporter.build_trade_frame(run, self.catalog.db_path, bars)
                equity_curve = ChartSnapshotExporter.build_equity_curve(run, bars, trades_df)
                snapshot = self.snapshot_exporter.export_backtest_snapshot(
                    run=run,
                    bars=bars,
                    overlays=overlays,
                    panes=panes,
                    series_styles=series_styles,
                    equity_curve=equity_curve,
                    trades_df=trades_df,
                )
                snapshot_root = snapshot.snapshot_root

            if not self.magellan.ensure_running(timeout_ms=5000):
                raise MagellanError(self.magellan.last_error or "Magellan is unavailable.")

            if (run.engine_impl or "").lower() == "vectorized_portfolio":
                for snapshot_root in snapshot_roots:
                    self.magellan.open_snapshot(snapshot_root)
            else:
                self.magellan.open_snapshot(snapshot_root)
            return True
        except Exception as exc:
            self.magellan._last_error = str(exc)
            return False
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _open_batch_detail(self, index: QtCore.QModelIndex) -> None:
        if not index.isValid():
            return
        batch = self.batch_model.batch_at(index.row())
        if not batch:
            return
        runs = self.catalog.load_runs(batch.batch_id)
        benchmarks = self._batch_benchmark_cache.get(batch.batch_id)
        if benchmarks is None:
            benchmarks = self.catalog.load_batch_benchmarks(batch.batch_id)
            self._batch_benchmark_cache[batch.batch_id] = benchmarks
        dlg = BatchDetailDialog(batch, runs, self.catalog.db_path, self, batch_benchmarks=benchmarks)
        dlg.exec()

    # -- strategies ----------------------------------------------------------
    def _init_strategy_selector(self) -> None:
        self.strategy_specs: Dict[str, Dict[str, object]] = {
            "SMA Crossover": {
                "class": SMACrossStrategy,
                "params": {
                    "fast": (int, 10),
                    "slow": (int, 30),
                    "target": (float, 1.0),
                },
            },
            "Inverse Turtle": {
                "class": InverseTurtleStrategy,
                "params": {
                    "entry_len": (int, 20),
                    "exit_len": (int, 10),
                    "atr_len": (int, 14),
                    "atr_mult": (float, 2.0),
                    "target": (float, 1.0),
                },
            },
            "Z-Score Mean Reversion": {
                "class": ZScoreMeanReversionStrategy,
                "params": {
                    "half_life_lookback": (int, 100),
                    "half_life_factor": (float, 1.5),
                    "std_len": (int, 20),
                    "z_smooth_type": (str, "None"),
                    "z_smooth_len": (int, 5),
                    "use_vol_norm": (bool, True),
                    "vol_type": (str, "ATR"),
                    "vol_len": (int, 14),
                    "atr_mult": (float, 1.0),
                    "long_entry_z": (float, -1.0),
                    "long_exit_z": (float, 0.0),
                    "target": (float, 1.0),
                },
            },
        }
        self.strategy_combo.clear()
        for name in self.strategy_specs:
            self.strategy_combo.addItem(name)
        self.strategy_combo.currentTextChanged.connect(self._render_strategy_params)
        self.param_inputs: Dict[str, QtWidgets.QLineEdit] = {}
        self._render_strategy_params(self.strategy_combo.currentText())

    def _render_strategy_params(self, name: str) -> None:
        # Clear existing rows
        while self.strategy_params_box.rowCount():
            self.strategy_params_box.removeRow(0)
        self.param_inputs.clear()
        spec = self.strategy_specs.get(name)
        if not spec:
            return
        for param, (_, default) in spec["params"].items():
            edit = QtWidgets.QLineEdit(str(default))
            edit.setPlaceholderText("comma list or start:end:step (e.g., 5,10,15 or 5:15:5)")
            edit.setToolTip(
                "Enter comma-separated values.\n"
                "Numeric params also support range notation start:end:step.\n"
                "String params accept literals like None,SMA,EMA.\n"
                "Boolean params accept true/false."
            )
            self.strategy_params_box.addRow(QtWidgets.QLabel(param), edit)
            self.param_inputs[param] = edit

    @staticmethod
    def _parse_strategy_param_value(ptype: type, token: str):
        if ptype is bool:
            lowered = token.strip().lower()
            if lowered in {"1", "true", "yes", "y", "on"}:
                return True
            if lowered in {"0", "false", "no", "n", "off"}:
                return False
            raise ValueError(f"invalid boolean literal: {token}")
        if ptype is str:
            return token.strip()
        return ptype(token)

    def _collect_strategy_params(self):
        name = self.strategy_combo.currentText()
        spec = self.strategy_specs.get(name)
        if not spec:
            return SMACrossStrategy, {"fast": [5, 10], "slow": [20, 30]}, []
        cls = spec["class"]
        params_meta = spec["params"]
        grid_params: Dict[str, List] = {}
        errors = []
        for key, (ptype, default) in params_meta.items():
            text = self.param_inputs[key].text().strip()
            vals: List = []
            tokens = [t.strip() for t in text.split(",")] if text else []
            if not tokens:
                vals = [default]
            else:
                for token in tokens:
                    if not token:
                        continue
                    try:
                        if ":" in token and ptype in {int, float}:
                            start_s, end_s, step_s = token.split(":")
                            start_v = self._parse_strategy_param_value(ptype, start_s)
                            end_v = self._parse_strategy_param_value(ptype, end_s)
                            step_v = self._parse_strategy_param_value(ptype, step_s)
                            if step_v == 0:
                                raise ValueError("step cannot be 0")
                            current = start_v
                            while (current <= end_v) if step_v > 0 else (current >= end_v):
                                vals.append(current)
                                current = ptype(current + step_v)
                        else:
                            vals.append(self._parse_strategy_param_value(ptype, token))
                    except Exception:
                        errors.append(f"{key}: '{token}'")
                if not vals:
                    vals = [default]
            grid_params[key] = vals
        return cls, grid_params, errors

    def _show_error_dialog(self, title: str, text: str, details: str | None = None) -> None:
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Icon.Critical)
        box.setWindowTitle(title)
        box.setText(text or "Unknown error")
        if details:
            box.setDetailedText(details)
        box.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        box.setStyleSheet(
            f"""
            QMessageBox {{
                background-color: {PALETTE['panel']};
                color: {PALETTE['text']};
            }}
            QLabel {{
                color: {PALETTE['text']};
            }}
            QPushButton {{
                background: {PALETTE['panel2']};
                color: {PALETTE['text']};
                border: 1px solid {PALETTE['border']};
                border-radius: 8px;
                padding: 6px 10px;
            }}
            """
        )
        box.exec()

    def _summarize_error(self, message: str | None) -> str:
        """Return a concise error line (typically the last line of a traceback)."""
        if not message:
            return "Unknown error"
        lines = [ln.strip() for ln in message.splitlines() if ln.strip()]
        if not lines:
            return "Unknown error"
        return lines[-1]


class BatchDetailDialog(DashboardDialog):
    def __init__(
        self,
        batch: BatchRow,
        runs: List[RunRow],
        catalog_path: Path,
        parent=None,
        batch_benchmarks: Sequence[BatchExecutionBenchmark] = (),
    ) -> None:
        super().__init__(parent)
        self.batch = batch
        self.setObjectName("Panel")
        self.setWindowTitle(f"Batch {batch.batch_id}")
        screen = QtWidgets.QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            self.resize(min(1100, max(820, int(available.width() * 0.82))), min(720, max(560, int(available.height() * 0.72))))
            self.setMaximumSize(int(available.width() * 0.96), int(available.height() * 0.94))
        else:
            self.resize(980, 680)
        self.setStyleSheet(
            f"""
            QDialog#Panel {{
                background: {PALETTE['panel']};
                color: {PALETTE['text']};
            }}
            QLabel#Title {{
                color: {PALETTE['text']};
                font-size: 18px;
                font-weight: 700;
            }}
            QLabel#Sub {{
                color: {PALETTE['muted']};
                font-size: 12px;
            }}
            QPlainTextEdit#Panel, QTableView#Panel {{
                background: {PALETTE['panel2']};
                color: {PALETTE['text']};
                border: 1px solid rgba(231, 238, 252, 0.55);
                border-radius: 10px;
            }}
            QPushButton {{
                background: rgba(255,255,255,.08);
                color: {PALETTE['text']};
                border: 1px solid rgba(231, 238, 252, 0.55);
                border-radius: 8px;
                padding: 6px 10px;
            }}
            QPushButton:hover {{
                border-color: {PALETTE['blue']};
                background: rgba(77,163,255,.15);
            }}
            QPushButton:disabled {{
                color: rgba(231, 238, 252, 0.45);
                border-color: rgba(231, 238, 252, 0.25);
                background: rgba(255,255,255,.03);
            }}
            """
        )
        self.catalog_path = catalog_path
        self.runs = list(runs)
        self.batch_benchmarks = tuple(batch_benchmarks)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)
        header_title = QtWidgets.QLabel(f"Batch {batch.batch_id}")
        header_title.setObjectName("Title")
        layout.addWidget(header_title)

        header_meta = QtWidgets.QPlainTextEdit()
        header_meta.setReadOnly(True)
        header_meta.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.WidgetWidth)
        header_meta.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        header_meta.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        header_meta.setMaximumHeight(110)
        header_meta.setObjectName("Panel")
        header_meta.setPlainText(
            "\n".join(
                [
                    f"Strategy: {batch.strategy}",
                    f"Dataset: {batch.dataset_id}",
                    f"Timeframes: {batch.timeframes}",
                    f"Horizons: {batch.horizons}",
                    f"Params: {batch.params}",
                ]
            )
        )
        layout.addWidget(header_meta)

        self.benchmark_summary = QtWidgets.QLabel(_summarize_batch_benchmarks(self.batch_benchmarks))
        self.benchmark_summary.setObjectName("Sub")
        self.benchmark_summary.setWordWrap(True)
        self.benchmark_summary.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.benchmark_summary)

        actions = QtWidgets.QHBoxLayout()
        self.open_chart_btn = QtWidgets.QPushButton("Open Selected Chart")
        self.open_chart_btn.clicked.connect(self._open_selected_run_chart)
        self.compare_btn = QtWidgets.QPushButton("Compare Engines")
        self.compare_btn.clicked.connect(self._compare_selected_run)
        self.compare_batch_btn = QtWidgets.QPushButton("Compare Batch")
        self.compare_batch_btn.clicked.connect(self._compare_batch_runs)
        self.optimization_btn = QtWidgets.QPushButton("Optimization")
        self.optimization_btn.clicked.connect(self._open_batch_optimization)
        self.portfolio_report_btn = QtWidgets.QPushButton("Portfolio Report")
        self.portfolio_report_btn.clicked.connect(self._open_selected_portfolio_report)
        self.benchmark_btn = QtWidgets.QPushButton("Benchmarks")
        self.benchmark_btn.clicked.connect(self._show_batch_benchmarks)
        actions.addWidget(self.open_chart_btn)
        actions.addWidget(self.compare_btn)
        actions.addWidget(self.compare_batch_btn)
        actions.addWidget(self.optimization_btn)
        actions.addWidget(self.portfolio_report_btn)
        actions.addWidget(self.benchmark_btn)
        actions.addStretch(1)
        layout.addLayout(actions)

        self.model = RunsTableModel(runs)
        self.table = QtWidgets.QTableView()
        self.table.setModel(self.model)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setObjectName("Panel")
        self.table.doubleClicked.connect(lambda idx: self._open_run_chart(self.model, idx))
        batch_params_dict = {}
        try:
            batch_params_dict = json.loads(batch.params) if batch.params else {}
        except Exception:
            batch_params_dict = {}
        self._batch_params_dict = dict(batch_params_dict)
        if (
            str(batch_params_dict.get("_study_mode", "")) == STUDY_MODE_PORTFOLIO
            and list(batch_params_dict.get("_portfolio_strategy_blocks") or [])
        ):
            self.optimization_btn.setText("Portfolio Candidates")
            self.optimization_btn.setToolTip(
                "Manage promoted fixed portfolio definitions for reduced-candidate portfolio walk-forward."
            )
        # Add log/download buttons per row for trades.
        for r_idx, run in enumerate(runs):
            log_btn = QtWidgets.QPushButton("Log")
            log_btn.clicked.connect(lambda _, rr=run: self._open_trades_log(rr))
            self.table.setIndexWidget(self.model.index(r_idx, self.model.columnCount() - 2), log_btn)
            btn = QtWidgets.QPushButton("Download")
            btn.clicked.connect(lambda _, rr=run: self._download_trades(rr))
            self.table.setIndexWidget(self.model.index(r_idx, self.model.columnCount() - 1), btn)
        if self.model.rowCount() > 0:
            self.table.selectRow(0)
        layout.addWidget(self.table)

    def _collect_backtest_settings(self) -> Dict[str, float | bool | dict | str]:
        parent = self.logical_parent()
        if parent and hasattr(parent, "_collect_backtest_settings"):
            try:
                return parent._collect_backtest_settings()  # type: ignore[attr-defined]
            except Exception:
                pass
        return {
            "starting_cash": 100_000,
            "fee_rate": 0.0002,
            "fee_schedule": {"buy": 0.0003, "sell": 0.0005},
            "slippage": 0.0002,
            "slippage_schedule": {"buy": 0.0003, "sell": 0.0001},
            "borrow_rate": 0.0,
            "fill_ratio": 1.0,
            "fill_on_close": False,
            "recalc_on_fill": True,
            "allow_short": True,
            "use_cache": False,
            "prevent_scale_in": True,
            "one_order_per_signal": True,
            "execution_mode": ExecutionMode.AUTO.value,
            "study_mode": STUDY_MODE_INDEPENDENT,
            "portfolio_allocation_mode": PORTFOLIO_ALLOC_EQUAL,
            "portfolio_weights_text": "",
            "portfolio_target_weights": {},
            "portfolio_ownership_mode": ALLOCATION_OWNERSHIP_STRATEGY,
            "portfolio_ranking_mode": RANKING_MODE_NONE,
            "portfolio_rank_count": 1,
            "portfolio_score_threshold": 0.0,
            "portfolio_weighting_mode": WEIGHTING_MODE_PRESERVE,
            "portfolio_min_active_weight": 0.0,
            "portfolio_max_asset_weight": 0.0,
            "portfolio_cash_reserve_weight": 0.0,
            "portfolio_rebalance_mode": REBALANCE_MODE_ON_CHANGE,
            "portfolio_rebalance_every_n_bars": 20,
            "portfolio_rebalance_drift_threshold": 0.05,
        }

    def _open_run_chart(self, model: RunsTableModel, index: QtCore.QModelIndex) -> None:
        if not index.isValid():
            return
        row = index.row()
        if row < 0 or row >= model.rowCount():
            return
        run = model._runs[row]
        self._open_run_chart_for_run(run)

    def _selected_run(self) -> RunRow | None:
        selection_model = self.table.selectionModel()
        if selection_model is None:
            return self.runs[0] if self.runs else None
        rows = selection_model.selectedRows()
        if not rows:
            return self.runs[0] if self.runs else None
        row = rows[0].row()
        if row < 0 or row >= len(self.runs):
            return None
        return self.runs[row]

    def _open_selected_run_chart(self) -> None:
        run = self._selected_run()
        if run is None:
            QtWidgets.QMessageBox.information(self, "No run selected", "Select a run first.")
            return
        self._open_run_chart_for_run(run)

    def _open_run_chart_for_run(self, run: RunRow) -> None:
        is_portfolio = (run.engine_impl or "").lower() == "vectorized_portfolio"
        if not is_portfolio:
            bars_utc = RunChartDialog._load_bars_utc_static(run)
            if bars_utc is None or bars_utc.empty:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Historical Data Unavailable",
                    RunChartDialog._missing_run_data_message_static(run),
                )
                return
        parent = self.logical_parent()
        portfolio_fallback_reason = ""
        if parent and hasattr(parent, "_open_run_chart_in_magellan"):
            try:
                if parent._open_run_chart_in_magellan(run):  # type: ignore[attr-defined]
                    return
                magellan = getattr(parent, "magellan", None)
                magellan_error = getattr(magellan, "last_error", "")
                if magellan_error:
                    if is_portfolio:
                        portfolio_fallback_reason = str(magellan_error)
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Magellan unavailable",
                        f"{magellan_error}\n\nFalling back to the built-in chart viewer.",
                    )
            except Exception as exc:
                if is_portfolio:
                    portfolio_fallback_reason = str(exc)
                QtWidgets.QMessageBox.warning(
                    self,
                    "Magellan unavailable",
                    f"{exc}\n\nFalling back to the built-in chart viewer.",
                )
        if is_portfolio:
            if parent is None or not hasattr(parent, "_build_portfolio_chart_payload_for_run"):
                QtWidgets.QMessageBox.information(
                    self,
                    "Portfolio Chart Unavailable",
                    "This portfolio run could not be opened because the built-in fallback viewer is unavailable.",
                )
                return
            try:
                portfolio_result, report = parent._build_portfolio_chart_payload_for_run(run)  # type: ignore[attr-defined]
            except Exception as exc:
                detail = f"{portfolio_fallback_reason}\n\n" if portfolio_fallback_reason else ""
                QtWidgets.QMessageBox.warning(
                    self,
                    "Portfolio Chart Error",
                    f"{detail}{exc}",
                )
                return
            try:
                dlg = PortfolioRunChartDialog(run, portfolio_result, report, self)
                dlg.exec()
            except Exception as exc:
                detail = f"{portfolio_fallback_reason}\n\n" if portfolio_fallback_reason else ""
                QtWidgets.QMessageBox.warning(
                    self,
                    "Portfolio Chart Error",
                    f"{detail}{exc}",
                )
            return
        bt_settings = self._collect_backtest_settings()
        dlg = RunChartDialog(run, self.catalog_path, bt_settings, self)
        dlg.exec()

    def _compare_selected_run(self) -> None:
        run = self._selected_run()
        if run is None:
            QtWidgets.QMessageBox.information(self, "No run selected", "Select a run first.")
            return
        if not run.logical_run_id:
            QtWidgets.QMessageBox.information(
                self,
                "Comparison Unavailable",
                "This run does not have a logical run id, so engine peers cannot be matched automatically.",
            )
            return
        peers = CatalogReader(self.catalog_path).load_runs_for_logical_run_id(run.logical_run_id)
        if not peers:
            QtWidgets.QMessageBox.information(
                self,
                "Comparison Unavailable",
                "No peer runs were found for this logical run id.",
            )
            return
        dlg = EngineComparisonDialog(run, peers, self.catalog_path, self._collect_backtest_settings(), self)
        dlg.exec()

    def _compare_batch_runs(self) -> None:
        if not self.runs:
            QtWidgets.QMessageBox.information(self, "Comparison Unavailable", "This batch has no runs to compare.")
            return
        dlg = BatchEngineComparisonDialog(
            self.runs,
            self.catalog_path,
            self._collect_backtest_settings(),
            self,
            batch_benchmarks=self.batch_benchmarks,
        )
        dlg.exec()

    def _show_batch_benchmarks(self) -> None:
        dlg = BatchBenchmarkDialog(self.batch_benchmarks, self)
        dlg.exec()

    def _open_batch_optimization(self) -> None:
        if (
            str(self._batch_params_dict.get("_study_mode", "")) == STUDY_MODE_PORTFOLIO
            and list(self._batch_params_dict.get("_portfolio_strategy_blocks") or [])
        ):
            dlg = FixedPortfolioCandidateDialog(self.batch, CatalogReader(self.catalog_path), self)
            dlg.exec()
            return
        parent = self.logical_parent()
        if parent is None or not hasattr(parent, "_open_optimization_study"):
            QtWidgets.QMessageBox.information(
                self,
                "Optimization Unavailable",
                "Optimization study viewing is unavailable in this context.",
            )
            return
        try:
            parent._open_optimization_study(self.batch.batch_id)  # type: ignore[attr-defined]
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Optimization Unavailable", str(exc))

    def _open_selected_portfolio_report(self) -> None:
        run = self._selected_run()
        if run is None:
            QtWidgets.QMessageBox.information(self, "No run selected", "Select a run first.")
            return
        if (run.engine_impl or "").lower() != "vectorized_portfolio":
            QtWidgets.QMessageBox.information(
                self,
                "Portfolio Report Unavailable",
                "Portfolio attribution reports are currently available only for vectorized portfolio runs.",
            )
            return
        parent = self.logical_parent()
        if parent is None or not hasattr(parent, "_build_portfolio_report_for_run"):
            QtWidgets.QMessageBox.warning(
                self,
                "Portfolio Report Unavailable",
                "The parent window cannot build a portfolio report for this run.",
            )
            return
        try:
            report = parent._build_portfolio_report_for_run(run)  # type: ignore[attr-defined]
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Portfolio Report Error",
                str(exc),
            )
            return
        dlg = PortfolioReportDialog(run, report, self)
        dlg.exec()

    def _open_trades_log(self, run: RunRow) -> None:
        trades_df = self._build_trades_dataframe(run)
        dlg = TradesLogDialog(run.run_id, trades_df, self)
        dlg.exec()

    def _build_trades_dataframe(self, run: RunRow) -> pd.DataFrame:
        if (run.engine_impl or "").lower() == "vectorized_portfolio":
            try:
                portfolio_result = self._rebuild_portfolio_result_for_run(run)
                return build_portfolio_trades_log_frame(portfolio_result)
            except Exception:
                # Fall back to stored generic trade rows if portfolio reconstruction fails.
                pass
        rc = ResultCatalog(self.catalog_path)
        trades = rc.load_trades(run.run_id) or []
        if not trades:
            return pd.DataFrame()
        rows = []
        pos = 0.0
        prev_realized = 0.0
        for i, t in enumerate(trades, start=1):
            position_before = pos
            qty = t["qty"]
            pos += qty
            net_pnl = t["realized_pnl"] - prev_realized
            prev_realized = t["realized_pnl"]
            if position_before * pos < 0:
                trade_type = "flip"
            elif abs(pos) > abs(position_before):
                trade_type = "entry"
            elif pos == position_before:
                trade_type = "adjust"
            else:
                trade_type = "exit"
            side = "buy" if qty > 0 else "sell"
            if trade_type == "entry":
                signal = "long" if side == "buy" else "short"
            elif trade_type == "flip":
                signal = "flip"
            else:
                signal = "open"
            rows.append(
                {
                    "trade_number": i,
                    "type": trade_type,
                    "timestamp": t["timestamp"],
                    "signal": signal,
                    "side": side,
                    "price": t["price"],
                    "qty": qty,
                    "position_after": pos,
                    "net_pnl": net_pnl,
                    "realized_pnl_cum": t["realized_pnl"],
                }
            )
        return pd.DataFrame(rows)

    def _download_trades(self, run: RunRow) -> None:
        try:
            df = self._build_trades_dataframe(run)
            if df.empty:
                QtWidgets.QMessageBox.information(self, "No trades", "No stored trades for this run.")
                return
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Trades CSV", f"{run.run_id[:10]}_trades.csv", "CSV Files (*.csv)")
            if path:
                df.to_csv(path, index=False)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Export error", str(exc))


class BatchBenchmarkDialog(DashboardDialog):
    def __init__(self, benchmarks: Sequence[BatchExecutionBenchmark], parent=None) -> None:
        super().__init__(parent)
        self.benchmarks = list(benchmarks)
        self.setWindowTitle("Batch Benchmarks")
        self.resize(1140, 720)

        layout = QtWidgets.QVBoxLayout(self)
        summary = QtWidgets.QLabel(_summarize_batch_benchmarks(self.benchmarks))
        summary.setObjectName("Sub")
        summary.setWordWrap(True)
        summary.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(summary)

        if not self.benchmarks:
            empty = QtWidgets.QLabel(
                "No benchmark data is available for this batch yet."
            )
            empty.setObjectName("Sub")
            empty.setWordWrap(True)
            layout.addWidget(empty)
            close_btn = QtWidgets.QPushButton("Close")
            close_btn.clicked.connect(self.accept)
            layout.addWidget(close_btn, 0, QtCore.Qt.AlignmentFlag.AlignRight)
            return

        self.figure = Figure(figsize=(10.5, 3.4), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas)
        self._draw_charts()

        self.table = QtWidgets.QTableWidget(len(self.benchmarks), 13)
        self.table.setHorizontalHeaderLabels(
            [
                "Engine",
                "Dataset",
                "Requested",
                "Resolved",
                "Timeframe",
                "Bars",
                "Params",
                "Cached",
                "Uncached",
                "Duration",
                "Chunks",
                "Chunk Sizes",
                "Prepared Reuse",
            ]
        )
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setObjectName("Panel")
        for row_idx, benchmark in enumerate(self.benchmarks):
            self._populate_row(row_idx, benchmark)
        if self.table.rowCount() > 0:
            self.table.selectRow(0)
        layout.addWidget(self.table)

        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn, 0, QtCore.Qt.AlignmentFlag.AlignRight)

    def _populate_row(self, row_idx: int, benchmark: BatchExecutionBenchmark) -> None:
        values = [
            benchmark.engine_impl,
            benchmark.dataset_id,
            benchmark.requested_execution_mode.value.title(),
            benchmark.resolved_execution_mode.value.title(),
            benchmark.timeframe,
            str(benchmark.bars),
            str(benchmark.total_params),
            str(benchmark.cached_runs),
            str(benchmark.uncached_runs),
            _format_seconds_precise(benchmark.duration_seconds),
            str(benchmark.chunk_count),
            ", ".join(str(size) for size in benchmark.chunk_sizes) if benchmark.chunk_sizes else "—",
            "Yes" if benchmark.prepared_context_reused else "No",
        ]
        for col_idx, value in enumerate(values):
            item = QtWidgets.QTableWidgetItem(value)
            if col_idx == 10:
                detail = [
                    f"Dataset: {benchmark.dataset_id}",
                    f"Strategy: {benchmark.strategy}",
                    f"Engine: {benchmark.engine_impl} v{benchmark.engine_version}",
                ]
                if benchmark.effective_param_batch_size is not None:
                    detail.append(f"Effective batch size: {benchmark.effective_param_batch_size}")
                item.setToolTip("\n".join(detail))
            self.table.setItem(row_idx, col_idx, item)

    def _draw_charts(self) -> None:
        self.figure.clear()
        if not self.benchmarks:
            self.canvas.draw_idle()
            return
        labels = [
            f"{benchmark.dataset_id}\n{benchmark.timeframe}\n{benchmark.engine_impl}"
            for benchmark in self.benchmarks
        ]
        x = np.arange(len(self.benchmarks))
        duration_ax = self.figure.add_subplot(1, 2, 1)
        throughput_ax = self.figure.add_subplot(1, 2, 2)

        duration_colors = [
            PALETTE["green"] if benchmark.resolved_execution_mode == ExecutionMode.VECTORIZED else PALETTE["blue"]
            for benchmark in self.benchmarks
        ]
        duration_values = [benchmark.duration_seconds for benchmark in self.benchmarks]
        duration_bars = duration_ax.bar(x, duration_values, color=duration_colors, alpha=0.9)
        duration_ax.set_title("Batch Duration")
        duration_ax.set_ylabel("Seconds")
        duration_ax.set_xticks(x)
        duration_ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        duration_ax.grid(axis="y", alpha=0.2)
        for bar, seconds in zip(duration_bars, duration_values):
            duration_ax.text(
                bar.get_x() + (bar.get_width() / 2),
                bar.get_height(),
                _format_seconds_precise(seconds),
                ha="center",
                va="bottom",
                fontsize=8,
                color=PALETTE["text"],
            )

        cached = np.array([benchmark.cached_runs for benchmark in self.benchmarks], dtype=float)
        uncached = np.array([benchmark.uncached_runs for benchmark in self.benchmarks], dtype=float)
        throughput_ax.bar(x, cached, label="Cached", color=PALETTE["blue"], alpha=0.75)
        throughput_ax.bar(x, uncached, bottom=cached, label="Uncached", color=PALETTE["amber"], alpha=0.85)
        throughput_ax.set_title("Run Composition")
        throughput_ax.set_ylabel("Run Count")
        throughput_ax.set_xticks(x)
        throughput_ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        throughput_ax.grid(axis="y", alpha=0.2)
        chunk_ax = throughput_ax.twinx()
        chunk_values = [benchmark.chunk_count for benchmark in self.benchmarks]
        chunk_ax.plot(x, chunk_values, color=PALETTE["red"], marker="o", linewidth=1.8, label="Chunks")
        chunk_ax.set_ylabel("Chunks")

        handles_1, labels_1 = throughput_ax.get_legend_handles_labels()
        handles_2, labels_2 = chunk_ax.get_legend_handles_labels()
        throughput_ax.legend(handles_1 + handles_2, labels_1 + labels_2, loc="upper right", fontsize=8)
        self.canvas.draw_idle()


class EngineComparisonDialog(DashboardDialog):
    def __init__(
        self,
        anchor_run: RunRow,
        peer_runs: List[RunRow],
        catalog_path: Path,
        bt_settings: Dict[str, float | bool | dict | str],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.anchor_run = anchor_run
        self.runs = sorted(peer_runs, key=self._row_sort_key, reverse=True)
        self.catalog_path = Path(catalog_path)
        self.bt_settings = bt_settings
        self.summary = summarize_engine_runs(self.runs)

        self.setWindowTitle(f"Compare Engines | {anchor_run.strategy}")
        self.resize(1050, 620)

        layout = QtWidgets.QVBoxLayout(self)
        header = QtWidgets.QLabel(
            f"Strategy: {anchor_run.strategy} | Dataset: {anchor_run.dataset_id} | Timeframe: {anchor_run.timeframe}\n"
            f"Logical Run ID: {anchor_run.logical_run_id or '—'}\n"
            f"Params: {anchor_run.params}"
        )
        header.setObjectName("Sub")
        header.setWordWrap(True)
        header.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(header)

        summary_label = QtWidgets.QLabel(self._summary_text(self.summary, anchor_run))
        summary_label.setObjectName("Sub")
        summary_label.setWordWrap(True)
        summary_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(summary_label)

        self.table = QtWidgets.QTableWidget(len(self.runs), 11)
        self.table.setHorizontalHeaderLabels(
            [
                "Engine",
                "Requested",
                "Resolved",
                "Version",
                "Duration",
                "Total Return",
                "Sharpe",
                "Rolling Sharpe",
                "Max DD",
                "Status",
                "Run ID",
            ]
        )
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setObjectName("Panel")
        self.table.doubleClicked.connect(lambda _: self._open_selected_chart())
        for row_idx, run in enumerate(self.runs):
            self._populate_row(row_idx, run)
            if run.run_id == anchor_run.run_id:
                self.table.selectRow(row_idx)
        if self.table.rowCount() > 0 and not self.table.selectionModel().hasSelection():
            self.table.selectRow(0)
        layout.addWidget(self.table)

        button_row = QtWidgets.QHBoxLayout()
        open_btn = QtWidgets.QPushButton("Open Selected Chart")
        open_btn.clicked.connect(self._open_selected_chart)
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        button_row.addWidget(open_btn)
        button_row.addStretch(1)
        button_row.addWidget(close_btn)
        layout.addLayout(button_row)

    @staticmethod
    def _row_sort_key(run: RunRow) -> tuple[int, int, int, str]:
        engine = (run.engine_impl or run.resolved_execution_mode or run.requested_execution_mode or "").lower()
        engine_rank = 2
        if engine == "vectorized":
            engine_rank = 1
        elif engine == "reference":
            engine_rank = 0
        finished = _parse_catalog_timestamp(run.run_finished_at)
        started = _parse_catalog_timestamp(run.run_started_at)
        finished_score = finished.value if finished is not None else -1
        started_score = started.value if started is not None else -1
        return (engine_rank, finished_score, started_score, run.run_id)

    def _populate_row(self, row_idx: int, run: RunRow) -> None:
        metrics = run.metrics or {}
        values = [
            (run.engine_impl or "—").title(),
            (run.requested_execution_mode or "—").title(),
            (run.resolved_execution_mode or "—").title(),
            run.engine_version or "—",
            _format_duration(run.run_started_at, run.run_finished_at),
            self._format_metric(metrics.get("total_return"), precision=4),
            self._format_metric(metrics.get("sharpe"), precision=4),
            self._format_metric(metrics.get("rolling_sharpe"), precision=4),
            self._format_metric(metrics.get("max_drawdown"), precision=4),
            run.status or "—",
            run.run_id,
        ]
        for col_idx, value in enumerate(values):
            item = QtWidgets.QTableWidgetItem(value)
            if col_idx == 10:
                item.setToolTip(
                    f"Requested: {run.requested_execution_mode or '—'}\n"
                    f"Resolved: {run.resolved_execution_mode or '—'}\n"
                    f"Engine: {run.engine_impl or '—'} v{run.engine_version or '—'}\n"
                    f"Fallback: {run.fallback_reason or '—'}"
                )
            self.table.setItem(row_idx, col_idx, item)

    def _selected_run(self) -> RunRow | None:
        selection_model = self.table.selectionModel()
        if selection_model is None:
            return self.runs[0] if self.runs else None
        rows = selection_model.selectedRows()
        if not rows:
            return self.runs[0] if self.runs else None
        row = rows[0].row()
        if row < 0 or row >= len(self.runs):
            return None
        return self.runs[row]

    def _open_selected_chart(self) -> None:
        run = self._selected_run()
        if run is None:
            QtWidgets.QMessageBox.information(self, "No run selected", "Select a run first.")
            return
        if (run.engine_impl or "").lower() != "vectorized_portfolio":
            bars_utc = RunChartDialog._load_bars_utc_static(run)
            if bars_utc is None or bars_utc.empty:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Historical Data Unavailable",
                    RunChartDialog._missing_run_data_message_static(run),
                )
                return
        parent = self.logical_parent()
        if parent and hasattr(parent, "_open_run_chart_for_run"):
            try:
                parent._open_run_chart_for_run(run)  # type: ignore[attr-defined]
                return
            except Exception:
                pass
        dlg = RunChartDialog(run, self.catalog_path, self.bt_settings, self)
        dlg.exec()

    @staticmethod
    def _format_metric(value, precision: int = 4) -> str:
        try:
            numeric = float(value)
        except Exception:
            return "—"
        if np.isnan(numeric):
            return "—"
        return f"{numeric:.{precision}f}"

    def _summary_text(self, summary: EngineComparisonSummary, anchor_run: RunRow) -> str:
        ref = summary.latest_reference
        vec = summary.latest_vectorized
        lines: list[str] = []
        if ref is not None:
            lines.append(f"Reference duration: {_format_duration(ref.run_started_at, ref.run_finished_at)}")
        if vec is not None:
            lines.append(f"Vectorized duration: {_format_duration(vec.run_started_at, vec.run_finished_at)}")
        if summary.speedup_vs_reference is not None:
            lines.append(f"Vectorized speedup vs reference: {summary.speedup_vs_reference:.2f}x")
        if summary.total_return_delta is not None:
            lines.append(f"Return delta (vectorized - reference): {summary.total_return_delta:+.6f}")
        if summary.sharpe_delta is not None:
            lines.append(f"Sharpe delta (vectorized - reference): {summary.sharpe_delta:+.6f}")
        if summary.max_drawdown_delta is not None:
            lines.append(f"Max DD delta (vectorized - reference): {summary.max_drawdown_delta:+.6f}")
        if not lines:
            lines.append("Only one engine variant is available for this logical run so far.")
        if anchor_run.timeframe != "1 minutes" and ref is not None and vec is not None:
            lines.append(
                "Note: higher-timeframe vectorized runs currently use same-timeframe resampled bars; "
                "full base-execution parity is still future work."
            )
        return "\n".join(lines)


class BatchEngineComparisonDialog(DashboardDialog):
    def __init__(
        self,
        runs: List[RunRow],
        catalog_path: Path,
        bt_settings: Dict[str, float | bool | dict | str],
        parent=None,
        batch_benchmarks: Sequence[BatchExecutionBenchmark] = (),
    ) -> None:
        super().__init__(parent)
        self.runs = list(runs)
        self.catalog_path = Path(catalog_path)
        self.bt_settings = bt_settings
        self.batch_benchmarks = tuple(batch_benchmarks)
        self.summary = summarize_engine_batch(self.runs)
        self.group_runs = self._build_group_runs(self.runs)
        self.group_keys = sorted(self.group_runs.keys())

        self.setWindowTitle("Batch Engine Comparison")
        self.resize(1220, 680)

        layout = QtWidgets.QVBoxLayout(self)
        summary_label = QtWidgets.QLabel(self._summary_text(self.summary))
        summary_label.setObjectName("Sub")
        summary_label.setWordWrap(True)
        summary_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(summary_label)
        benchmark_label = QtWidgets.QLabel(_summarize_batch_benchmarks(self.batch_benchmarks))
        benchmark_label.setObjectName("Sub")
        benchmark_label.setWordWrap(True)
        benchmark_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(benchmark_label)

        self.table = QtWidgets.QTableWidget(len(self.summary.groups), 12)
        self.table.setHorizontalHeaderLabels(
            [
                "Timeframe",
                "Params",
                "Reference",
                "Vectorized",
                "Speedup",
                "Return Δ",
                "Sharpe Δ",
                "Max DD Δ",
                "Runs",
                "Engines",
                "Logical Run ID",
                "Example Run",
            ]
        )
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setObjectName("Panel")
        self.table.doubleClicked.connect(lambda _: self._inspect_selected_pair())
        for row_idx, group in enumerate(self.summary.groups):
            self._populate_row(row_idx, group)
        if self.table.rowCount() > 0:
            self.table.selectRow(0)
        layout.addWidget(self.table)

        buttons = QtWidgets.QHBoxLayout()
        inspect_btn = QtWidgets.QPushButton("Inspect Selected Pair")
        inspect_btn.clicked.connect(self._inspect_selected_pair)
        export_btn = QtWidgets.QPushButton("Export CSV")
        export_btn.clicked.connect(self._export_csv)
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        buttons.addWidget(inspect_btn)
        buttons.addWidget(export_btn)
        buttons.addStretch(1)
        buttons.addWidget(close_btn)
        layout.addLayout(buttons)

    @staticmethod
    def _build_group_runs(runs: List[RunRow]) -> dict[str, list[RunRow]]:
        grouped: dict[str, list[RunRow]] = {}
        for run in runs:
            key = run.logical_run_id or f"run::{run.run_id}"
            grouped.setdefault(key, []).append(run)
        return grouped

    def _populate_row(self, row_idx: int, group: EngineComparisonSummary) -> None:
        group_key = self.group_keys[row_idx] if row_idx < len(self.group_keys) else ""
        runs = self.group_runs.get(group_key, [])
        example = runs[0] if runs else None
        ref = group.latest_reference
        vec = group.latest_vectorized
        values = [
            example.timeframe if example else "—",
            example.params if example else "—",
            _format_duration(ref.run_started_at, ref.run_finished_at) if ref is not None else "—",
            _format_duration(vec.run_started_at, vec.run_finished_at) if vec is not None else "—",
            self._format_float(group.speedup_vs_reference, precision=2, suffix="x"),
            self._format_float(group.total_return_delta, precision=6, signed=True),
            self._format_float(group.sharpe_delta, precision=6, signed=True),
            self._format_float(group.max_drawdown_delta, precision=6, signed=True),
            str(group.compared_run_count),
            ", ".join(group.available_engines) if group.available_engines else "—",
            (group.logical_run_id or "—")[:12] + ("…" if group.logical_run_id and len(group.logical_run_id) > 12 else ""),
            example.run_id[:12] + "…" if example else "—",
        ]
        for col_idx, value in enumerate(values):
            item = QtWidgets.QTableWidgetItem(value)
            if col_idx == 1 and example is not None:
                item.setToolTip(example.params)
            if col_idx == 10:
                item.setToolTip(group.logical_run_id or "—")
            self.table.setItem(row_idx, col_idx, item)

    @staticmethod
    def _format_float(
        value: float | None,
        *,
        precision: int = 4,
        signed: bool = False,
        suffix: str = "",
    ) -> str:
        if value is None:
            return "—"
        if np.isnan(value):
            return "—"
        fmt = f"{{:{'+' if signed else ''}.{precision}f}}"
        return f"{fmt.format(value)}{suffix}"

    def _summary_text(self, summary: EngineBatchComparisonSummary) -> str:
        lines = [
            f"Logical run groups: {summary.total_groups}",
            f"Paired groups: {summary.paired_groups}",
            f"Reference-only groups: {summary.reference_only_groups}",
            f"Vectorized-only groups: {summary.vectorized_only_groups}",
        ]
        if summary.median_speedup_vs_reference is not None:
            lines.append(f"Median vectorized speedup: {summary.median_speedup_vs_reference:.2f}x")
        if summary.mean_speedup_vs_reference is not None:
            lines.append(f"Mean vectorized speedup: {summary.mean_speedup_vs_reference:.2f}x")
        if summary.max_abs_total_return_delta is not None:
            lines.append(f"Max |return delta|: {summary.max_abs_total_return_delta:.6f}")
        if summary.max_abs_sharpe_delta is not None:
            lines.append(f"Max |sharpe delta|: {summary.max_abs_sharpe_delta:.6f}")
        if summary.max_abs_max_drawdown_delta is not None:
            lines.append(f"Max |max DD delta|: {summary.max_abs_max_drawdown_delta:.6f}")
        lines.append(
            "Higher-timeframe vectorized rows may still reflect same-timeframe resampled semantics rather than "
            "full base-execution parity."
        )
        return " | ".join(lines)

    def _selected_group(self) -> tuple[EngineComparisonSummary | None, list[RunRow]]:
        selection_model = self.table.selectionModel()
        if selection_model is None:
            return None, []
        rows = selection_model.selectedRows()
        if not rows:
            return None, []
        row_idx = rows[0].row()
        if row_idx < 0 or row_idx >= len(self.summary.groups):
            return None, []
        group = self.summary.groups[row_idx]
        group_key = self.group_keys[row_idx] if row_idx < len(self.group_keys) else ""
        runs = self.group_runs.get(group_key, [])
        return group, runs

    def _inspect_selected_pair(self) -> None:
        group, runs = self._selected_group()
        if group is None or not runs:
            QtWidgets.QMessageBox.information(self, "No pair selected", "Select a comparison row first.")
            return
        anchor = group.latest_vectorized or group.latest_reference
        if anchor is None:
            QtWidgets.QMessageBox.information(self, "Comparison Unavailable", "No engine runs are available for this row.")
            return
        anchor_run = next((run for run in runs if run.run_id == anchor.run_id), runs[0])
        dlg = EngineComparisonDialog(anchor_run, runs, self.catalog_path, self.bt_settings, self)
        dlg.exec()

    def _export_csv(self) -> None:
        frame = self._to_frame()
        if frame.empty:
            QtWidgets.QMessageBox.information(self, "No data", "There is nothing to export yet.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Batch Comparison CSV",
            "batch_engine_comparison.csv",
            "CSV Files (*.csv)",
        )
        if not path:
            return
        try:
            frame.to_csv(path, index=False)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Export error", str(exc))

    def _to_frame(self) -> pd.DataFrame:
        rows = []
        for row_idx, group in enumerate(self.summary.groups):
            group_key = self.group_keys[row_idx] if row_idx < len(self.group_keys) else ""
            runs = self.group_runs.get(group_key, [])
            example = runs[0] if runs else None
            rows.append(
                {
                    "logical_run_id": group.logical_run_id,
                    "timeframe": example.timeframe if example else None,
                    "params": example.params if example else None,
                    "reference_run_id": group.latest_reference.run_id if group.latest_reference is not None else None,
                    "vectorized_run_id": group.latest_vectorized.run_id if group.latest_vectorized is not None else None,
                    "reference_duration_seconds": group.latest_reference.duration_seconds if group.latest_reference is not None else None,
                    "vectorized_duration_seconds": group.latest_vectorized.duration_seconds if group.latest_vectorized is not None else None,
                    "speedup_vs_reference": group.speedup_vs_reference,
                    "total_return_delta": group.total_return_delta,
                    "sharpe_delta": group.sharpe_delta,
                    "rolling_sharpe_delta": group.rolling_sharpe_delta,
                    "max_drawdown_delta": group.max_drawdown_delta,
                    "available_engines": ",".join(group.available_engines),
                    "compared_run_count": group.compared_run_count,
                }
            )
        return pd.DataFrame(rows)


class CandidatePromotionDialog(DashboardDialog):
    def __init__(self, reasons: Sequence[str], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Promote Candidate")
        self.resize(560, 320)
        self.setObjectName("Panel")
        self.setStyleSheet(
            f"""
            QDialog#Panel {{
                background: {PALETTE['panel']};
                color: {PALETTE['text']};
            }}
            QLabel#Title {{
                color: {PALETTE['text']};
                font-size: 18px;
                font-weight: 700;
            }}
            QLabel#Sub {{
                color: {PALETTE['muted']};
                font-size: 12px;
            }}
            QTextEdit, QPlainTextEdit, QComboBox {{
                background: {PALETTE['panel2']};
                color: {PALETTE['text']};
                border: 1px solid rgba(231, 238, 252, 0.55);
                border-radius: 8px;
                padding: 6px 8px;
            }}
            QPushButton {{
                background: rgba(255,255,255,.08);
                color: {PALETTE['text']};
                border: 1px solid rgba(231, 238, 252, 0.55);
                border-radius: 8px;
                padding: 6px 10px;
            }}
            QPushButton:hover {{
                border-color: {PALETTE['blue']};
                background: rgba(77,163,255,.15);
            }}
            """
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        title = QtWidgets.QLabel("Promote Selected Candidate")
        title.setObjectName("Title")
        layout.addWidget(title)

        subtitle = QtWidgets.QLabel(
            "Choose why you are promoting this parameter set and add any notes you want to carry into walk-forward review."
        )
        subtitle.setWordWrap(True)
        subtitle.setObjectName("Sub")
        layout.addWidget(subtitle)

        form = QtWidgets.QFormLayout()
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        form.setFormAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(10)

        self.reason_combo = QtWidgets.QComboBox()
        for reason in reasons:
            self.reason_combo.addItem(str(reason))
        form.addRow("Promotion Reason", self.reason_combo)

        self.notes_edit = QtWidgets.QPlainTextEdit()
        self.notes_edit.setPlaceholderText("Optional notes for later walk-forward review.")
        self.notes_edit.setMinimumHeight(140)
        form.addRow("Notes", self.notes_edit)
        layout.addLayout(form)

        button_row = QtWidgets.QHBoxLayout()
        button_row.addStretch(1)
        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        ok_btn = QtWidgets.QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        ok_btn.setDefault(True)
        button_row.addWidget(cancel_btn)
        button_row.addWidget(ok_btn)
        layout.addLayout(button_row)

    def values(self) -> tuple[str, str]:
        return self.reason_combo.currentText().strip(), self.notes_edit.toPlainText().strip()


class PortfolioValidationChainDialog(DashboardDialog):
    def __init__(
        self,
        *,
        title_context: str,
        catalog: CatalogReader,
        portfolio_mode: str,
        dataset_ids: Sequence[str],
        batch_params: dict | None = None,
        source_study_id: str = "",
        source_batch_id: str = "",
        optimization_study_row: dict | None = None,
        candidates: pd.DataFrame | None = None,
        walk_forward_studies: pd.DataFrame | None = None,
        monte_carlo_studies: pd.DataFrame | None = None,
        focus_candidate_id: str = "",
        initial_tab: str = "candidates",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.catalog = catalog
        self.title_context = str(title_context or "Portfolio")
        self.portfolio_mode = str(portfolio_mode or "shared_strategy")
        self.dataset_ids = [str(item) for item in list(dataset_ids or ()) if str(item).strip()]
        self.batch_params = dict(batch_params or {})
        self.source_study_id = str(source_study_id or "")
        self.source_batch_id = str(source_batch_id or "")
        self.optimization_study_row = dict(optimization_study_row or {})
        self.focus_candidate_id = str(focus_candidate_id or "")
        self._candidate_relation_cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}

        self.candidates = self._prepare_candidates(
            candidates if candidates is not None else self._load_candidates()
        )
        self.walk_forward_studies = (
            walk_forward_studies.copy()
            if walk_forward_studies is not None
            else self._load_related_walk_forward_studies()
        )
        self.monte_carlo_studies = (
            monte_carlo_studies.copy()
            if monte_carlo_studies is not None
            else self._load_related_monte_carlo_studies(self.walk_forward_studies)
        )

        self.setWindowTitle(f"Portfolio Validation Chain | {self.title_context}")
        self.resize(1560, 980)
        self.setMinimumSize(1240, 760)
        self.setObjectName("Panel")
        self.setStyleSheet(
            f"""
            QDialog#Panel {{
                background: {PALETTE['panel']};
                color: {PALETTE['text']};
            }}
            QLabel#Title {{
                color: {PALETTE['text']};
                font-size: 20px;
                font-weight: 700;
            }}
            QLabel#Sub {{
                color: {PALETTE['muted']};
            }}
            QPlainTextEdit#Panel, QTableWidget#Panel {{
                background: {PALETTE['panel2']};
                color: {PALETTE['text']};
                border: 1px solid rgba(231, 238, 252, 0.55);
                border-radius: 10px;
            }}
            QHeaderView::section {{
                background: rgba(255, 255, 255, 0.04);
                color: {PALETTE['muted']};
                border: none;
                border-right: 1px solid rgba(231, 238, 252, 0.18);
                border-bottom: 1px solid rgba(231, 238, 252, 0.18);
                padding: 7px 10px;
                font-weight: 700;
            }}
            QTabWidget::pane {{
                border: 1px solid rgba(231, 238, 252, 0.35);
                border-radius: 10px;
                top: -1px;
                background: {PALETTE['panel']};
            }}
            QTabBar::tab {{
                background: rgba(255,255,255,.05);
                color: {PALETTE['muted']};
                border: 1px solid rgba(231, 238, 252, 0.35);
                border-bottom: none;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
                padding: 8px 16px;
                min-width: 110px;
            }}
            QTabBar::tab:selected {{
                background: {PALETTE['panel2']};
                color: {PALETTE['text']};
                font-weight: 700;
            }}
            QPushButton {{
                background: rgba(255,255,255,.08);
                color: {PALETTE['text']};
                border: 1px solid rgba(231, 238, 252, 0.55);
                border-radius: 8px;
                padding: 6px 12px;
            }}
            QPushButton:hover {{
                border-color: {PALETTE['blue']};
                background: rgba(77,163,255,.15);
            }}
            QPushButton:disabled {{
                color: rgba(231, 238, 252, 0.45);
                border-color: rgba(231, 238, 252, 0.25);
                background: rgba(255,255,255,.03);
            }}
            """
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        title = QtWidgets.QLabel("Portfolio Validation Chain")
        title.setObjectName("Title")
        layout.addWidget(title)

        summary = QtWidgets.QPlainTextEdit()
        summary.setReadOnly(True)
        summary.setMaximumHeight(132)
        summary.setObjectName("Panel")
        summary.setPlainText("\n".join(self._summary_lines()))
        layout.addWidget(summary)

        self.relationship_summary = QtWidgets.QLabel("")
        self.relationship_summary.setObjectName("Sub")
        self.relationship_summary.setWordWrap(True)
        self.relationship_summary.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.relationship_summary)

        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs, 1)

        self.candidate_table = QtWidgets.QTableWidget(0, 8)
        self.candidate_table.setHorizontalHeaderLabels(
            ["Candidate", "Timeframe", "Reason", "Status", "Robust Score", "Median Sharpe", "Median Return", "Updated"]
        )
        self.candidate_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.candidate_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.candidate_table.setAlternatingRowColors(True)
        self.candidate_table.verticalHeader().setVisible(False)
        self.candidate_table.setObjectName("Panel")
        self.candidate_table.itemSelectionChanged.connect(self._refresh_candidate_details)
        self.candidate_notes = QtWidgets.QPlainTextEdit()
        self.candidate_notes.setReadOnly(True)
        self.candidate_notes.setMaximumHeight(180)
        self.candidate_notes.setObjectName("Panel")
        self.candidate_summary = QtWidgets.QLabel("")
        self.candidate_summary.setObjectName("Sub")
        self.candidate_summary.setWordWrap(True)
        self.candidate_summary.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        candidate_panel = QtWidgets.QWidget()
        candidate_layout = QtWidgets.QVBoxLayout(candidate_panel)
        candidate_layout.setContentsMargins(8, 8, 8, 8)
        candidate_layout.setSpacing(8)
        candidate_layout.addWidget(self.candidate_summary)
        candidate_layout.addWidget(self.candidate_table, 1)
        candidate_layout.addWidget(self.candidate_notes)
        candidate_actions = QtWidgets.QHBoxLayout()
        self.open_candidate_wf_btn = QtWidgets.QPushButton("Open Latest Candidate Walk-Forward")
        self.open_candidate_wf_btn.clicked.connect(self._open_selected_candidate_walk_forward)
        self.open_candidate_mc_btn = QtWidgets.QPushButton("Open Latest Candidate Monte Carlo")
        self.open_candidate_mc_btn.clicked.connect(self._open_selected_candidate_monte_carlo)
        candidate_actions.addWidget(self.open_candidate_wf_btn)
        candidate_actions.addWidget(self.open_candidate_mc_btn)
        candidate_actions.addStretch(1)
        candidate_layout.addLayout(candidate_actions)
        self.tabs.addTab(candidate_panel, "Candidates")

        self.wf_table = QtWidgets.QTableWidget(0, 9)
        self.wf_table.setHorizontalHeaderLabels(
            ["Study", "Mode", "Candidate Source", "Selection", "Folds", "Return", "Sharpe", "Max DD", "Created"]
        )
        self.wf_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.wf_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.wf_table.setAlternatingRowColors(True)
        self.wf_table.verticalHeader().setVisible(False)
        self.wf_table.setObjectName("Panel")
        self.wf_summary = QtWidgets.QLabel("")
        self.wf_summary.setObjectName("Sub")
        self.wf_summary.setWordWrap(True)
        self.wf_summary.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        wf_panel = QtWidgets.QWidget()
        wf_layout = QtWidgets.QVBoxLayout(wf_panel)
        wf_layout.setContentsMargins(8, 8, 8, 8)
        wf_layout.setSpacing(8)
        wf_layout.addWidget(self.wf_summary)
        wf_layout.addWidget(self.wf_table, 1)
        wf_actions = QtWidgets.QHBoxLayout()
        self.open_wf_btn = QtWidgets.QPushButton("Open Selected Walk-Forward")
        self.open_wf_btn.clicked.connect(self._open_selected_walk_forward)
        wf_actions.addWidget(self.open_wf_btn)
        wf_actions.addStretch(1)
        wf_layout.addLayout(wf_actions)
        self.tabs.addTab(wf_panel, "Walk-Forward")

        self.mc_table = QtWidgets.QTableWidget(0, 7)
        self.mc_table.setHorizontalHeaderLabels(
            ["Study", "Mode", "Simulations", "Median Return", "P95 Max DD", "Loss Prob", "Created"]
        )
        self.mc_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.mc_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.mc_table.setAlternatingRowColors(True)
        self.mc_table.verticalHeader().setVisible(False)
        self.mc_table.setObjectName("Panel")
        self.mc_summary = QtWidgets.QLabel("")
        self.mc_summary.setObjectName("Sub")
        self.mc_summary.setWordWrap(True)
        self.mc_summary.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        mc_panel = QtWidgets.QWidget()
        mc_layout = QtWidgets.QVBoxLayout(mc_panel)
        mc_layout.setContentsMargins(8, 8, 8, 8)
        mc_layout.setSpacing(8)
        mc_layout.addWidget(self.mc_summary)
        mc_layout.addWidget(self.mc_table, 1)
        mc_actions = QtWidgets.QHBoxLayout()
        self.open_mc_btn = QtWidgets.QPushButton("Open Selected Monte Carlo")
        self.open_mc_btn.clicked.connect(self._open_selected_monte_carlo)
        mc_actions.addWidget(self.open_mc_btn)
        mc_actions.addStretch(1)
        mc_layout.addLayout(mc_actions)
        self.tabs.addTab(mc_panel, "Monte Carlo")

        self.structure_table = QtWidgets.QTableWidget(0, 0)
        self.structure_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.structure_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.structure_table.setAlternatingRowColors(True)
        self.structure_table.verticalHeader().setVisible(False)
        self.structure_table.setObjectName("Panel")
        self.structure_notes = QtWidgets.QPlainTextEdit()
        self.structure_notes.setReadOnly(True)
        self.structure_notes.setMaximumHeight(180)
        self.structure_notes.setObjectName("Panel")
        structure_panel = QtWidgets.QWidget()
        structure_layout = QtWidgets.QVBoxLayout(structure_panel)
        structure_layout.setContentsMargins(8, 8, 8, 8)
        structure_layout.setSpacing(8)
        structure_layout.addWidget(self.structure_table, 1)
        structure_layout.addWidget(self.structure_notes)
        self.tabs.addTab(structure_panel, "Portfolio Structure")

        close_row = QtWidgets.QHBoxLayout()
        close_row.addStretch(1)
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        close_row.addWidget(close_btn)
        layout.addLayout(close_row)

        self._configure_table(self.candidate_table)
        self._configure_table(self.wf_table)
        self._configure_table(self.mc_table)
        self._configure_table(self.structure_table)
        self._populate_candidate_table()
        self._populate_walk_forward_table()
        self._populate_monte_carlo_table()
        self._populate_structure_table()
        self._apply_focus(initial_tab=initial_tab)

    @staticmethod
    def _configure_table(table: QtWidgets.QTableWidget) -> None:
        header = table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)
        table.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        table.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        table.setWordWrap(False)
        table.setTextElideMode(QtCore.Qt.TextElideMode.ElideRight)

    @staticmethod
    def _parse_json_dict(value) -> dict:
        if isinstance(value, dict):
            return dict(value)
        if not value:
            return {}
        try:
            decoded = json.loads(str(value))
        except Exception:
            return {}
        return decoded if isinstance(decoded, dict) else {}

    @staticmethod
    def _fmt(value, precision: int = 4) -> str:
        try:
            numeric = float(value)
        except Exception:
            return "—"
        if not np.isfinite(numeric):
            return "—"
        return f"{numeric:.{precision}f}"

    @classmethod
    def _fmt_pct(cls, value, precision: int = 2) -> str:
        try:
            numeric = float(value)
        except Exception:
            return "—"
        if not np.isfinite(numeric):
            return "—"
        return f"{numeric * 100:.{precision}f}%"

    @staticmethod
    def _fmt_timestamp(value) -> str:
        ts = pd.to_datetime(value, errors="coerce", utc=True)
        if pd.isna(ts):
            return "—"
        return ts.tz_convert("America/New_York").strftime("%Y-%m-%d %H:%M")

    @staticmethod
    def _portfolio_mode_label(mode: str) -> str:
        return "Fixed Strategy Blocks" if str(mode or "") == "strategy_blocks" else "Shared Strategy Portfolio"

    def _controller(self):
        parent = self.logical_parent()
        visited: set[int] = set()
        while parent is not None and id(parent) not in visited:
            visited.add(id(parent))
            if hasattr(parent, "_open_optimization_study") or hasattr(parent, "_open_walk_forward_study") or hasattr(parent, "_open_monte_carlo_study"):
                return parent
            if hasattr(parent, "logical_parent"):
                parent = parent.logical_parent()
            else:
                break
        return self.logical_parent()

    def _summary_lines(self) -> list[str]:
        line = [
            f"Context: {self.title_context}",
            f"Portfolio Mode: {self._portfolio_mode_label(self.portfolio_mode)}",
            f"Underlying Assets: {', '.join(self.dataset_ids) or '—'}",
            f"Promoted Candidates: {len(self.candidates)}",
            f"Linked Walk-Forward Studies: {len(self.walk_forward_studies)}",
            f"Linked Monte Carlo Studies: {len(self.monte_carlo_studies)}",
        ]
        if self.source_study_id:
            line.append(f"Source Optimization Study: {self.source_study_id}")
        if self.source_batch_id:
            line.append(f"Source Portfolio Batch: {self.source_batch_id}")
        if self.optimization_study_row:
            line.append(
                f"Optimization Strategy: {self.optimization_study_row.get('strategy', '')} | "
                f"Score Version: {self.optimization_study_row.get('score_version', '')}"
            )
        return line

    def _load_candidates(self) -> pd.DataFrame:
        if self.source_study_id:
            return self.catalog.load_optimization_candidates(self.source_study_id)
        if self.source_batch_id:
            return self.catalog.load_optimization_candidates(self.source_batch_id)
        return pd.DataFrame()

    def _load_related_walk_forward_studies(self) -> pd.DataFrame:
        studies = self.catalog.load_walk_forward_studies()
        if studies.empty:
            return studies
        matches: list[bool] = []
        for _, row in studies.iterrows():
            params_payload = self._parse_json_dict(row.get("params_json"))
            matches.append(
                (
                    bool(self.source_study_id)
                    and str(params_payload.get("source_study_id", "") or "") == self.source_study_id
                )
                or (
                    bool(self.source_batch_id)
                    and str(params_payload.get("source_batch_id", "") or "") == self.source_batch_id
                )
            )
        frame = studies.loc[matches].reset_index(drop=True)
        if "created_at" in frame.columns:
            frame = frame.sort_values(by=["created_at", "wf_study_id"], ascending=[False, False], kind="mergesort").reset_index(drop=True)
        return frame

    def _load_related_monte_carlo_studies(self, walk_forward_studies: pd.DataFrame) -> pd.DataFrame:
        if walk_forward_studies.empty:
            return pd.DataFrame()
        studies = self.catalog.load_monte_carlo_studies()
        if studies.empty:
            return studies
        wf_ids = {str(item) for item in walk_forward_studies["wf_study_id"].tolist() if str(item).strip()}
        if not wf_ids:
            return pd.DataFrame()
        frame = studies.loc[studies["source_id"].fillna("").isin(wf_ids)].reset_index(drop=True)
        if "created_at" in frame.columns:
            frame = frame.sort_values(by=["created_at", "mc_study_id"], ascending=[False, False], kind="mergesort").reset_index(drop=True)
        return frame

    def _prepare_candidates(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame()
        expanded = frame.copy()
        metric_frame = expanded["metrics_json"].apply(self._parse_json_dict).apply(pd.Series)
        for col in metric_frame.columns:
            if col not in expanded.columns:
                expanded[col] = metric_frame[col]
        params_frame = expanded["params_json"].apply(self._parse_json_dict).apply(pd.Series)
        for col in params_frame.columns:
            if col not in expanded.columns:
                expanded[col] = params_frame[col]
        expanded = expanded.sort_values(by=["updated_at", "created_at"], ascending=[False, False], kind="mergesort")
        return expanded.reset_index(drop=True)

    def _populate_candidate_table(self) -> None:
        frame = self.candidates.copy()
        if frame.empty:
            self.candidate_summary.setText("No promoted portfolio candidates have been queued for this portfolio context yet.")
            self.candidate_table.setRowCount(0)
            self.candidate_notes.clear()
            self.relationship_summary.setText(
                f"No promoted candidates | {len(self.walk_forward_studies)} walk-forward stud{'y' if len(self.walk_forward_studies) == 1 else 'ies'} | "
                f"{len(self.monte_carlo_studies)} Monte Carlo stud{'y' if len(self.monte_carlo_studies) == 1 else 'ies'}"
            )
            return
        best_robust = pd.to_numeric(frame.get("robust_score"), errors="coerce").max()
        self.candidate_summary.setText(
            f"Queued portfolio candidates: {len(frame)} | Best robust score: {self._fmt(best_robust)} | "
            f"Latest update: {self._fmt_timestamp(frame.iloc[0].get('updated_at'))}"
        )
        self.relationship_summary.setText(
            f"Validation chain: {len(frame)} promoted candidates | "
            f"{len(self.walk_forward_studies)} linked walk-forward stud{'y' if len(self.walk_forward_studies) == 1 else 'ies'} | "
            f"{len(self.monte_carlo_studies)} linked Monte Carlo stud{'y' if len(self.monte_carlo_studies) == 1 else 'ies'}"
        )
        self.candidate_table.setRowCount(len(frame))
        for row_idx, row in frame.iterrows():
            values = [
                str(row.get("candidate_id", "") or "")[:10],
                str(row.get("timeframe", "") or "—"),
                str(row.get("promotion_reason", "") or "Manual review"),
                str(row.get("status", "") or "queued"),
                self._fmt(row.get("robust_score")),
                self._fmt(row.get("median_sharpe")),
                self._fmt(row.get("median_total_return")),
                self._fmt_timestamp(row.get("updated_at")),
            ]
            for col_idx, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                if col_idx == 0:
                    item.setData(QtCore.Qt.ItemDataRole.UserRole, row.to_dict())
                self.candidate_table.setItem(row_idx, col_idx, item)
        if self.candidate_table.rowCount() > 0:
            self.candidate_table.selectRow(0)
        self._refresh_candidate_details()

    def _selected_candidate(self) -> dict | None:
        selection_model = self.candidate_table.selectionModel()
        if selection_model is None:
            return None
        rows = selection_model.selectedRows()
        if not rows:
            return None
        item = self.candidate_table.item(rows[0].row(), 0)
        if item is None:
            return None
        payload = item.data(QtCore.Qt.ItemDataRole.UserRole)
        return payload if isinstance(payload, dict) else None

    def _candidate_related_walk_forward(self, candidate_row: dict) -> pd.DataFrame:
        candidate_id = str(candidate_row.get("candidate_id", "") or "")
        cached = self._candidate_relation_cache.get(candidate_id)
        if cached is not None:
            return cached[0]
        param_key = str(candidate_row.get("param_key", "") or "")
        if not param_key or self.walk_forward_studies.empty:
            frame = pd.DataFrame()
            self._candidate_relation_cache[candidate_id] = (frame, pd.DataFrame())
            return frame
        rows: list[dict] = []
        for _, study_row in self.walk_forward_studies.iterrows():
            wf_study_id = str(study_row.get("wf_study_id", "") or "")
            if not wf_study_id:
                continue
            folds = self.catalog.load_walk_forward_folds(wf_study_id)
            if folds.empty:
                continue
            if folds["selected_param_set_id"].fillna("").astype(str).eq(param_key).any():
                rows.append(study_row.to_dict())
        frame = pd.DataFrame(rows)
        mc_frame = self._load_related_monte_carlo_studies(frame) if not frame.empty else pd.DataFrame()
        self._candidate_relation_cache[candidate_id] = (frame, mc_frame)
        return frame

    def _candidate_related_monte_carlo(self, candidate_row: dict) -> pd.DataFrame:
        candidate_id = str(candidate_row.get("candidate_id", "") or "")
        cached = self._candidate_relation_cache.get(candidate_id)
        if cached is not None:
            return cached[1]
        self._candidate_related_walk_forward(candidate_row)
        return self._candidate_relation_cache.get(candidate_id, (pd.DataFrame(), pd.DataFrame()))[1]

    def _candidate_structure_lines(self, candidate_row: dict) -> list[str]:
        params_payload = self._parse_json_dict(candidate_row.get("params_json"))
        if self.portfolio_mode == "strategy_blocks":
            blocks = list(params_payload.get("strategy_blocks") or self.batch_params.get("strategy_blocks") or self.batch_params.get("_portfolio_strategy_blocks") or [])
            if not blocks:
                return ["No strategy blocks were stored for this candidate."]
            return [
                f"{str(block.get('display_name') or block.get('block_id') or f'Block {idx + 1}')}: "
                f"{str(block.get('strategy') or block.get('strategy_name') or '—')} | "
                f"Budget={block.get('budget_weight') if block.get('budget_weight') not in (None, '') else 'auto'} | "
                f"Assets={', '.join(str(asset.get('dataset_id') or '') for asset in list(block.get('assets') or []) if str(asset.get('dataset_id') or '').strip()) or '—'}"
                for idx, block in enumerate(blocks)
            ]
        construction = self._parse_json_dict(self.batch_params.get("construction_config"))
        allocation_mode = str(
            self.batch_params.get("_portfolio_allocation_mode")
            or self.batch_params.get("allocation_mode")
            or PORTFOLIO_ALLOC_EQUAL
        )
        ownership = str(
            construction.get("allocation_ownership")
            or self.batch_params.get("_portfolio_allocation_ownership")
            or ALLOCATION_OWNERSHIP_STRATEGY
        )
        ranking = str(
            construction.get("ranking_mode")
            or self.batch_params.get("_portfolio_ranking_mode")
            or RANKING_MODE_NONE
        )
        rebalance = str(
            construction.get("rebalance_mode")
            or self.batch_params.get("_portfolio_rebalance_mode")
            or REBALANCE_MODE_ON_CHANGE
        )
        portfolio_assets = list(self.batch_params.get("portfolio_assets") or [])
        target_weights = {
            str(asset.get("dataset_id") or ""): asset.get("target_weight")
            for asset in portfolio_assets
            if str(asset.get("dataset_id") or "").strip()
        }
        if not target_weights:
            target_weights = {
                str(key): value
                for key, value in dict(self.batch_params.get("_portfolio_target_weights", {}) or {}).items()
                if str(key).strip()
            }
        if not target_weights and self.dataset_ids and allocation_mode == PORTFOLIO_ALLOC_EQUAL:
            equal_weight = 1.0 / max(1, len(self.dataset_ids))
            target_weights = {dataset_id: equal_weight for dataset_id in self.dataset_ids}
        return [
            f"Underlying Assets: {', '.join(self.dataset_ids) or '—'}",
            f"Allocation: {allocation_mode.replace('_', ' ').title()}",
            f"Ownership: {ownership.replace('_', ' ').title()}",
            f"Ranking: {ranking.replace('_', ' ').title()}",
            f"Rebalance: {rebalance.replace('_', ' ').title()}",
            (
                "Target Weights: "
                + ", ".join(f"{dataset_id}={self._fmt(weight)}" for dataset_id, weight in target_weights.items())
                if target_weights
                else "Target Weights: auto"
            ),
        ]

    def _refresh_candidate_details(self) -> None:
        selected = self._selected_candidate()
        if not selected:
            self.candidate_notes.clear()
            self.open_candidate_wf_btn.setEnabled(False)
            self.open_candidate_mc_btn.setEnabled(False)
            return
        related_wf = self._candidate_related_walk_forward(selected)
        related_mc = self._candidate_related_monte_carlo(selected)
        metrics = self._parse_json_dict(selected.get("metrics_json"))
        lines = [
            f"Candidate: {str(selected.get('candidate_id', '') or '')[:10]}",
            f"Status: {selected.get('status', '') or 'queued'}",
            f"Reason: {selected.get('promotion_reason', '') or 'Manual review'}",
            f"Source Type: {selected.get('source_type', '') or 'manual_selection'}",
            f"Timeframe: {selected.get('timeframe', '') or '—'}",
            f"Window: {self._fmt_timestamp(selected.get('start'))} -> {self._fmt_timestamp(selected.get('end'))}",
            f"Robust Score: {self._fmt(metrics.get('robust_score'))}",
            f"Median Sharpe: {self._fmt(metrics.get('median_sharpe'))}",
            f"Median Return: {self._fmt(metrics.get('median_total_return'))}",
            f"Worst Max DD: {self._fmt(metrics.get('worst_max_drawdown'))}",
            f"Linked Walk-Forward Studies: {len(related_wf)}",
            f"Linked Monte Carlo Studies: {len(related_mc)}",
            "",
            "Portfolio Structure",
            *self._candidate_structure_lines(selected),
            "",
            str(selected.get("notes", "") or "No analyst notes."),
        ]
        self.candidate_notes.setPlainText("\n".join(lines))
        self.open_candidate_wf_btn.setEnabled(not related_wf.empty)
        self.open_candidate_mc_btn.setEnabled(not related_mc.empty)

    def _populate_walk_forward_table(self) -> None:
        frame = self.walk_forward_studies.copy()
        if frame.empty:
            self.wf_summary.setText("No linked portfolio walk-forward studies have been saved yet.")
            self.wf_table.setRowCount(0)
            self.open_wf_btn.setEnabled(False)
            return
        self.wf_summary.setText(
            f"Linked walk-forward studies: {len(frame)} | "
            f"Candidate modes: {', '.join(sorted(set(str(item) for item in frame['candidate_source_mode'].fillna('') if str(item).strip()))) or '—'}"
        )
        self.wf_table.setRowCount(len(frame))
        for row_idx, row in frame.reset_index(drop=True).iterrows():
            schedule_payload = self._parse_json_dict(row.get("schedule_json"))
            stitched = self._parse_json_dict(row.get("stitched_metrics_json"))
            values = [
                str(row.get("wf_study_id", "") or ""),
                self._portfolio_mode_label(str(schedule_payload.get("portfolio_mode", self.portfolio_mode))),
                str(row.get("candidate_source_mode", "") or "—"),
                str(row.get("selection_rule", "") or "—"),
                str(int(row.get("fold_count", 0) or 0)),
                self._fmt(stitched.get("total_return")),
                self._fmt(stitched.get("sharpe")),
                self._fmt(stitched.get("max_drawdown")),
                self._fmt_timestamp(row.get("created_at")),
            ]
            for col_idx, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                if col_idx == 0:
                    item.setData(QtCore.Qt.ItemDataRole.UserRole, row.to_dict())
                self.wf_table.setItem(row_idx, col_idx, item)
        if self.wf_table.rowCount() > 0:
            self.wf_table.selectRow(0)
        self.open_wf_btn.setEnabled(True)

    def _populate_monte_carlo_table(self) -> None:
        frame = self.monte_carlo_studies.copy()
        if frame.empty:
            self.mc_summary.setText("No linked portfolio Monte Carlo studies have been saved yet.")
            self.mc_table.setRowCount(0)
            self.open_mc_btn.setEnabled(False)
            return
        self.mc_summary.setText(
            f"Linked Monte Carlo studies: {len(frame)} | "
            f"Modes: {', '.join(sorted(set(str(item) for item in frame['resampling_mode'].fillna('') if str(item).strip()))) or '—'}"
        )
        self.mc_table.setRowCount(len(frame))
        for row_idx, row in frame.reset_index(drop=True).iterrows():
            summary = self._parse_json_dict(row.get("summary_json"))
            values = [
                str(row.get("mc_study_id", "") or ""),
                str(row.get("resampling_mode", "") or "—"),
                str(int(row.get("simulation_count", 0) or 0)),
                self._fmt(summary.get("terminal_return_p50")),
                self._fmt(summary.get("max_drawdown_p95")),
                self._fmt_pct(summary.get("loss_probability")),
                self._fmt_timestamp(row.get("created_at")),
            ]
            for col_idx, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                if col_idx == 0:
                    item.setData(QtCore.Qt.ItemDataRole.UserRole, row.to_dict())
                self.mc_table.setItem(row_idx, col_idx, item)
        if self.mc_table.rowCount() > 0:
            self.mc_table.selectRow(0)
        self.open_mc_btn.setEnabled(True)

    def _populate_structure_table(self) -> None:
        if self.portfolio_mode == "strategy_blocks":
            blocks = list(
                self.batch_params.get("strategy_blocks")
                or self.batch_params.get("_portfolio_strategy_blocks")
                or []
            )
            self.structure_table.setColumnCount(5)
            self.structure_table.setHorizontalHeaderLabels(["Block", "Strategy", "Budget", "Assets", "Asset Weights"])
            self.structure_table.setRowCount(len(blocks))
            note_lines: list[str] = []
            for row_idx, block in enumerate(blocks):
                assets = list(block.get("assets") or [])
                asset_ids = [str(asset.get("dataset_id") or "") for asset in assets if str(asset.get("dataset_id") or "").strip()]
                asset_weight_text = ", ".join(
                    f"{asset_id}={asset.get('target_weight') if asset.get('target_weight') not in (None, '') else 'auto'}"
                    for asset_id, asset in zip(asset_ids, assets)
                ) or "auto"
                values = [
                    str(block.get("display_name") or block.get("block_id") or f"Block {row_idx + 1}"),
                    str(block.get("strategy") or block.get("strategy_name") or "—"),
                    str(block.get("budget_weight") if block.get("budget_weight") not in (None, "") else "auto"),
                    ", ".join(asset_ids) or "—",
                    asset_weight_text,
                ]
                for col_idx, value in enumerate(values):
                    self.structure_table.setItem(row_idx, col_idx, QtWidgets.QTableWidgetItem(value))
                note_lines.append(
                    f"{values[0]} | Strategy={values[1]} | Budget={values[2]} | Assets={values[3]}"
                )
            self.structure_notes.setPlainText(
                "\n".join(note_lines) if note_lines else "No fixed strategy blocks were stored for this portfolio context."
            )
            return

        portfolio_assets = list(self.batch_params.get("portfolio_assets") or [])
        dataset_ids = list(self.dataset_ids)
        if not dataset_ids:
            dataset_ids = [
                str(asset.get("dataset_id") or "")
                for asset in portfolio_assets
                if str(asset.get("dataset_id") or "").strip()
            ]
        construction = self._parse_json_dict(self.batch_params.get("construction_config"))
        allocation_mode = str(
            self.batch_params.get("_portfolio_allocation_mode")
            or self.batch_params.get("allocation_mode")
            or PORTFOLIO_ALLOC_EQUAL
        )
        ownership = str(
            construction.get("allocation_ownership")
            or self.batch_params.get("_portfolio_allocation_ownership")
            or ALLOCATION_OWNERSHIP_STRATEGY
        )
        ranking = str(
            construction.get("ranking_mode")
            or self.batch_params.get("_portfolio_ranking_mode")
            or RANKING_MODE_NONE
        )
        rebalance = str(
            construction.get("rebalance_mode")
            or self.batch_params.get("_portfolio_rebalance_mode")
            or REBALANCE_MODE_ON_CHANGE
        )
        target_weights = {
            str(asset.get("dataset_id") or ""): asset.get("target_weight")
            for asset in portfolio_assets
            if str(asset.get("dataset_id") or "").strip()
        }
        if not target_weights:
            target_weights = {
                str(key): value
                for key, value in dict(self.batch_params.get("_portfolio_target_weights", {}) or {}).items()
                if str(key).strip()
            }
        if not target_weights and dataset_ids and allocation_mode == PORTFOLIO_ALLOC_EQUAL:
            equal_weight = 1.0 / max(1, len(dataset_ids))
            target_weights = {dataset_id: equal_weight for dataset_id in dataset_ids}

        self.structure_table.setColumnCount(6)
        self.structure_table.setHorizontalHeaderLabels(["Dataset", "Target Weight", "Allocation", "Ownership", "Ranking", "Rebalance"])
        self.structure_table.setRowCount(len(dataset_ids))
        for row_idx, dataset_id in enumerate(dataset_ids):
            values = [
                str(dataset_id),
                self._fmt(target_weights.get(dataset_id)),
                allocation_mode.replace("_", " ").title(),
                ownership.replace("_", " ").title(),
                ranking.replace("_", " ").title(),
                rebalance.replace("_", " ").title(),
            ]
            for col_idx, value in enumerate(values):
                self.structure_table.setItem(row_idx, col_idx, QtWidgets.QTableWidgetItem(value))
        self.structure_notes.setPlainText(
            "Shared-strategy portfolio validation compares promoted portfolio candidates, linked walk-forward studies, and downstream Monte Carlo studies across the same underlying asset universe."
        )

    def _selected_table_row(self, table: QtWidgets.QTableWidget) -> dict | None:
        selection_model = table.selectionModel()
        if selection_model is None:
            return None
        rows = selection_model.selectedRows()
        if not rows:
            return None
        item = table.item(rows[0].row(), 0)
        if item is None:
            return None
        payload = item.data(QtCore.Qt.ItemDataRole.UserRole)
        return payload if isinstance(payload, dict) else None

    def _open_selected_walk_forward(self) -> None:
        selected = self._selected_table_row(self.wf_table)
        if not selected:
            QtWidgets.QMessageBox.information(self, "No study selected", "Select a walk-forward study first.")
            return
        controller = self._controller()
        wf_study_id = str(selected.get("wf_study_id", "") or "")
        if controller is not None and hasattr(controller, "_open_walk_forward_study"):
            controller._open_walk_forward_study(wf_study_id)  # type: ignore[attr-defined]
            return
        QtWidgets.QMessageBox.information(self, "Study unavailable", f"The walk-forward study '{wf_study_id}' could not be opened from this dialog.")

    def _open_selected_monte_carlo(self) -> None:
        selected = self._selected_table_row(self.mc_table)
        if not selected:
            QtWidgets.QMessageBox.information(self, "No study selected", "Select a Monte Carlo study first.")
            return
        controller = self._controller()
        mc_study_id = str(selected.get("mc_study_id", "") or "")
        if controller is not None and hasattr(controller, "_open_monte_carlo_study"):
            controller._open_monte_carlo_study(mc_study_id)  # type: ignore[attr-defined]
            return
        QtWidgets.QMessageBox.information(self, "Study unavailable", f"The Monte Carlo study '{mc_study_id}' could not be opened from this dialog.")

    def _open_selected_candidate_walk_forward(self) -> None:
        selected = self._selected_candidate()
        if not selected:
            QtWidgets.QMessageBox.information(self, "No candidate selected", "Select a portfolio candidate first.")
            return
        related = self._candidate_related_walk_forward(selected)
        if related.empty:
            QtWidgets.QMessageBox.information(self, "No linked study", "No walk-forward study has selected this portfolio candidate yet.")
            return
        controller = self._controller()
        wf_study_id = str(related.iloc[0].get("wf_study_id", "") or "")
        if controller is not None and hasattr(controller, "_open_walk_forward_study"):
            controller._open_walk_forward_study(wf_study_id)  # type: ignore[attr-defined]
            return
        QtWidgets.QMessageBox.information(self, "Study unavailable", f"The walk-forward study '{wf_study_id}' could not be opened from this dialog.")

    def _open_selected_candidate_monte_carlo(self) -> None:
        selected = self._selected_candidate()
        if not selected:
            QtWidgets.QMessageBox.information(self, "No candidate selected", "Select a portfolio candidate first.")
            return
        related = self._candidate_related_monte_carlo(selected)
        if related.empty:
            QtWidgets.QMessageBox.information(self, "No linked study", "No Monte Carlo study is linked to a walk-forward study that selected this portfolio candidate yet.")
            return
        controller = self._controller()
        mc_study_id = str(related.iloc[0].get("mc_study_id", "") or "")
        if controller is not None and hasattr(controller, "_open_monte_carlo_study"):
            controller._open_monte_carlo_study(mc_study_id)  # type: ignore[attr-defined]
            return
        QtWidgets.QMessageBox.information(self, "Study unavailable", f"The Monte Carlo study '{mc_study_id}' could not be opened from this dialog.")

    def _apply_focus(self, *, initial_tab: str) -> None:
        tab_name = str(initial_tab or "candidates").strip().lower()
        if tab_name == "walk_forward":
            self.tabs.setCurrentIndex(1)
        elif tab_name == "monte_carlo":
            self.tabs.setCurrentIndex(2)
        elif tab_name == "structure":
            self.tabs.setCurrentIndex(3)
        else:
            self.tabs.setCurrentIndex(0)
        if not self.focus_candidate_id or self.candidate_table.rowCount() <= 0:
            return
        for row_idx in range(self.candidate_table.rowCount()):
            item = self.candidate_table.item(row_idx, 0)
            if item is None:
                continue
            payload = item.data(QtCore.Qt.ItemDataRole.UserRole)
            if not isinstance(payload, dict):
                continue
            if str(payload.get("candidate_id", "") or "") == self.focus_candidate_id:
                self.tabs.setCurrentIndex(0)
                self.candidate_table.selectRow(row_idx)
                break


class FixedPortfolioCandidateDialog(DashboardDialog):
    _ALL_TIMEFRAMES = "__all_timeframes__"

    def __init__(self, batch: BatchRow, catalog: CatalogReader, parent=None) -> None:
        super().__init__(parent)
        self.batch = batch
        self.catalog = catalog
        self.batch_id = str(batch.batch_id)
        self.batch_params = self._decode_json_dict(batch.params)
        self.strategy_blocks = list(self.batch_params.get("_portfolio_strategy_blocks") or [])
        self.timeframes = [item.strip() for item in str(batch.timeframes or "").split(",") if item.strip()]
        self.dataset_ids = self._dataset_ids_from_blocks(self.strategy_blocks)
        self.candidates = self.catalog.load_optimization_candidates(self.batch_id)

        self.setWindowTitle(f"Portfolio Candidates | {self.batch_id}")
        self.resize(1240, 820)
        self.setObjectName("Panel")
        self.setStyleSheet(
            f"""
            QDialog#Panel {{
                background: {PALETTE['panel']};
                color: {PALETTE['text']};
            }}
            QGroupBox {{
                color: {PALETTE['text']};
                border: 1px solid rgba(231, 238, 252, 0.42);
                border-radius: 10px;
                margin-top: 12px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: {PALETTE['muted']};
                font-weight: 700;
            }}
            """
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        intro = QtWidgets.QLabel(
            "Promote fixed multi-strategy portfolio definitions into a validation queue. "
            "Reduced-candidate portfolio walk-forward can then require one of these promoted definitions."
        )
        intro.setObjectName("Sub")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        summary = QtWidgets.QPlainTextEdit()
        summary.setReadOnly(True)
        summary.setMaximumHeight(120)
        summary.setObjectName("Panel")
        summary.setPlainText(
            "\n".join(
                [
                    f"Batch: {self.batch_id}",
                    f"Strategy: {self.batch.strategy}",
                    f"Mode: Fixed strategy blocks",
                    f"Datasets: {', '.join(self.dataset_ids) or '—'}",
                    f"Timeframes: {', '.join(self.timeframes) or '—'}",
                    f"Strategy blocks: {len(self.strategy_blocks)}",
                ]
            )
        )
        layout.addWidget(summary)

        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        layout.addWidget(split, 1)

        left_panel = QtWidgets.QWidget()
        left_panel.setObjectName("Panel")
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(8)

        block_group = QtWidgets.QGroupBox("Strategy Block Definition")
        block_layout = QtWidgets.QVBoxLayout(block_group)
        block_layout.setContentsMargins(10, 10, 10, 10)
        block_layout.setSpacing(8)
        self.block_table = QtWidgets.QTableWidget(0, 5)
        self.block_table.setHorizontalHeaderLabels(["Block", "Strategy", "Budget", "Assets", "Asset Weights"])
        self.block_table.setAlternatingRowColors(True)
        self.block_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.block_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.block_table.horizontalHeader().setStretchLastSection(True)
        self.block_table.verticalHeader().setVisible(False)
        self.block_table.setObjectName("Panel")
        block_layout.addWidget(self.block_table, 1)
        self.block_notes = QtWidgets.QPlainTextEdit()
        self.block_notes.setReadOnly(True)
        self.block_notes.setMaximumHeight(120)
        self.block_notes.setObjectName("Panel")
        block_layout.addWidget(self.block_notes)
        left_layout.addWidget(block_group, 1)
        split.addWidget(left_panel)

        right_panel = QtWidgets.QWidget()
        right_panel.setObjectName("Panel")
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(8)

        queue_group = QtWidgets.QGroupBox("Validation Queue")
        queue_layout = QtWidgets.QVBoxLayout(queue_group)
        queue_layout.setContentsMargins(10, 10, 10, 10)
        queue_layout.setSpacing(8)

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Promotion Scope"))
        self.timeframe_scope_combo = QtWidgets.QComboBox()
        self.timeframe_scope_combo.addItem("All Batch Timeframes", self._ALL_TIMEFRAMES)
        for timeframe in self.timeframes:
            self.timeframe_scope_combo.addItem(str(timeframe), str(timeframe))
        controls.addWidget(self.timeframe_scope_combo)
        controls.addStretch(1)
        queue_layout.addLayout(controls)

        self.queue_summary = QtWidgets.QLabel("")
        self.queue_summary.setObjectName("Sub")
        self.queue_summary.setWordWrap(True)
        self.queue_summary.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        queue_layout.addWidget(self.queue_summary)

        self.candidate_table = QtWidgets.QTableWidget(0, 6)
        self.candidate_table.setHorizontalHeaderLabels(
            ["Timeframe", "Status", "Reason", "Robust Score", "Updated", "Notes"]
        )
        self.candidate_table.setAlternatingRowColors(True)
        self.candidate_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.candidate_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.candidate_table.horizontalHeader().setStretchLastSection(True)
        self.candidate_table.verticalHeader().setVisible(False)
        self.candidate_table.setObjectName("Panel")
        self.candidate_table.itemSelectionChanged.connect(self._refresh_candidate_notes)
        queue_layout.addWidget(self.candidate_table, 1)

        self.candidate_notes = QtWidgets.QPlainTextEdit()
        self.candidate_notes.setReadOnly(True)
        self.candidate_notes.setMaximumHeight(120)
        self.candidate_notes.setObjectName("Panel")
        queue_layout.addWidget(self.candidate_notes)

        action_row = QtWidgets.QHBoxLayout()
        self.promote_btn = QtWidgets.QPushButton("Promote Current Definition")
        self.promote_btn.clicked.connect(self._promote_current_definition)
        self.review_btn = QtWidgets.QPushButton("Review Candidate")
        self.review_btn.clicked.connect(self._review_selected_candidate)
        self.review_btn.setEnabled(False)
        self.validation_chain_btn = QtWidgets.QPushButton("Validation Chain")
        self.validation_chain_btn.clicked.connect(self._open_validation_chain)
        self.remove_btn = QtWidgets.QPushButton("Remove Candidate")
        self.remove_btn.clicked.connect(self._remove_selected_candidate)
        action_row.addWidget(self.promote_btn)
        action_row.addWidget(self.review_btn)
        action_row.addWidget(self.validation_chain_btn)
        action_row.addWidget(self.remove_btn)
        action_row.addStretch(1)
        queue_layout.addLayout(action_row)
        right_layout.addWidget(queue_group, 1)
        split.addWidget(right_panel)
        split.setSizes([560, 620])

        close_row = QtWidgets.QHBoxLayout()
        close_row.addStretch(1)
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        close_row.addWidget(close_btn)
        layout.addLayout(close_row)

        self._populate_block_definition()
        self._refresh_candidate_queue()

    @staticmethod
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

    @staticmethod
    def _dataset_ids_from_blocks(strategy_blocks: Sequence[dict]) -> list[str]:
        return list(
            dict.fromkeys(
                str(asset.get("dataset_id") or "")
                for block in list(strategy_blocks or ())
                for asset in list(block.get("assets") or [])
                if str(asset.get("dataset_id") or "").strip()
            )
        )

    def _construction_config_payload(self) -> dict:
        params = self.batch_params
        return {
            "allocation_ownership": str(params.get("_portfolio_allocation_ownership", ALLOCATION_OWNERSHIP_STRATEGY)),
            "ranking_mode": str(params.get("_portfolio_ranking_mode", RANKING_MODE_NONE)),
            "max_ranked_assets": int(params.get("_portfolio_rank_count", 1) or 1),
            "min_rank_score": float(params.get("_portfolio_score_threshold", 0.0) or 0.0),
            "weighting_mode": str(params.get("_portfolio_weighting_mode", WEIGHTING_MODE_PRESERVE)),
            "min_active_weight": float(params.get("_portfolio_min_active_weight", 0.0) or 0.0),
            "max_asset_weight": float(params.get("_portfolio_max_asset_weight", 0.0) or 0.0),
            "cash_reserve_weight": float(params.get("_portfolio_cash_reserve_weight", 0.0) or 0.0),
            "rebalance_mode": str(params.get("_portfolio_rebalance_mode", REBALANCE_MODE_ON_CHANGE)),
            "rebalance_every_n_bars": int(params.get("_portfolio_rebalance_every_n_bars", 20) or 20),
            "rebalance_weight_drift_threshold": float(params.get("_portfolio_rebalance_drift_threshold", 0.05) or 0.05),
        }

    def _strategy_block_payload(self) -> dict:
        return {
            "strategy_blocks": list(self.strategy_blocks),
            "portfolio_dataset_ids": list(self.dataset_ids),
            "construction_config": self._construction_config_payload(),
            "source_kind": "portfolio_fixed_blocks",
            "source_batch_id": self.batch_id,
        }

    def _populate_block_definition(self) -> None:
        rows = list(self.strategy_blocks)
        self.block_table.setRowCount(len(rows))
        note_lines: list[str] = []
        for row_idx, block in enumerate(rows):
            assets = list(block.get("assets") or [])
            asset_ids = [str(asset.get("dataset_id") or "") for asset in assets if str(asset.get("dataset_id") or "").strip()]
            asset_weight_text = ", ".join(
                f"{asset_id}={asset.get('target_weight') if asset.get('target_weight') not in (None, '') else 'auto'}"
                for asset_id, asset in zip(asset_ids, assets)
            ) or "auto"
            values = [
                str(block.get("display_name") or block.get("block_id") or f"Block {row_idx + 1}"),
                str(block.get("strategy_name") or block.get("strategy") or "—"),
                str(block.get("budget_weight") if block.get("budget_weight") not in (None, "") else "auto"),
                ", ".join(asset_ids) or "—",
                asset_weight_text,
            ]
            for col_idx, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                if col_idx == 0:
                    item.setData(QtCore.Qt.ItemDataRole.UserRole, dict(block))
                self.block_table.setItem(row_idx, col_idx, item)
            note_lines.append(
                f"{values[0]} | Strategy={values[1]} | Budget={values[2]} | Assets={values[3]}"
            )
        if self.block_table.rowCount() > 0:
            self.block_table.selectRow(0)
        self.block_notes.setPlainText(
            "\n".join(note_lines) if note_lines else "No fixed strategy blocks were stored on this batch."
        )

    def _selected_candidate(self) -> dict | None:
        selection_model = self.candidate_table.selectionModel()
        if selection_model is None:
            return None
        rows = selection_model.selectedRows()
        if not rows:
            return None
        item = self.candidate_table.item(rows[0].row(), 0)
        if item is None:
            return None
        payload = item.data(QtCore.Qt.ItemDataRole.UserRole)
        return payload if isinstance(payload, dict) else None

    def _candidate_metrics_for_timeframe(self, timeframe: str) -> dict:
        runs = [
            run
            for run in self.catalog.load_runs(self.batch_id)
            if str(run.timeframe) == str(timeframe)
            and str(run.engine_impl or "").lower() == "vectorized_portfolio"
        ]
        if not runs:
            return {
                "robust_score": 0.0,
                "median_sharpe": 0.0,
                "median_total_return": 0.0,
                "worst_max_drawdown": 0.0,
                "sharpe_std": 0.0,
                "profitable_asset_ratio": 0.0,
                "dataset_count": len(self.dataset_ids),
                "run_count": 0,
            }
        returns = pd.Series([float(run.metrics.get("total_return", 0.0) or 0.0) for run in runs], dtype=float)
        sharpes = pd.Series([float(run.metrics.get("sharpe", 0.0) or 0.0) for run in runs], dtype=float)
        drawdowns = pd.Series([abs(float(run.metrics.get("max_drawdown", 0.0) or 0.0)) for run in runs], dtype=float)
        profitable_ratio = float((returns > 0.0).mean()) if not returns.empty else 0.0
        sharpe_std = float(sharpes.std(ddof=0) or 0.0) if not sharpes.empty else 0.0
        median_sharpe = float(sharpes.median()) if not sharpes.empty else 0.0
        median_return = float(returns.median()) if not returns.empty else 0.0
        worst_drawdown = float(drawdowns.max()) if not drawdowns.empty else 0.0
        return {
            "robust_score": float(
                compute_robust_score(
                    median_sharpe=median_sharpe,
                    sharpe_std=sharpe_std,
                    worst_max_drawdown=worst_drawdown,
                    profitable_asset_ratio=profitable_ratio,
                )
            ),
            "median_sharpe": median_sharpe,
            "median_total_return": median_return,
            "worst_max_drawdown": worst_drawdown,
            "sharpe_std": sharpe_std,
            "profitable_asset_ratio": profitable_ratio,
            "dataset_count": len(self.dataset_ids),
            "run_count": len(runs),
        }

    def _refresh_candidate_queue(self) -> None:
        self.candidates = self.catalog.load_optimization_candidates(self.batch_id)
        frame = self.candidates.copy()
        if frame.empty:
            self.queue_summary.setText(
                "No promoted fixed portfolio definitions yet. Promote the current strategy-block definition to require it in reduced-candidate walk-forward."
            )
            self.candidate_table.setRowCount(0)
            self.candidate_notes.clear()
            if hasattr(self, "review_btn"):
                self.review_btn.setEnabled(False)
            return
        metric_frame = frame["metrics_json"].apply(OptimizationStudyDialog._parse_json_text).apply(pd.Series)
        for col in metric_frame.columns:
            if col not in frame.columns:
                frame[col] = metric_frame[col]
        frame = frame.sort_values(by=["updated_at", "created_at"], ascending=[False, False], kind="mergesort")
        self.queue_summary.setText(
            f"Queued definitions: {len(frame)} | "
            f"Timeframes covered: {frame['timeframe'].nunique()} | "
            f"Latest update: {PortfolioReportDialog._fmt_timestamp(frame.iloc[0].get('updated_at'))}"
        )
        self.candidate_table.setRowCount(len(frame))
        for row_idx, row in frame.reset_index(drop=True).iterrows():
            values = [
                str(row.get("timeframe", "") or "—"),
                str(row.get("status", "") or "queued"),
                str(row.get("promotion_reason", "") or "Manual review"),
                OptimizationStudyDialog._format_numeric_value(row.get("robust_score")),
                PortfolioReportDialog._fmt_timestamp(row.get("updated_at")),
                str(row.get("notes", "") or ""),
            ]
            for col_idx, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                if col_idx == 0:
                    item.setData(QtCore.Qt.ItemDataRole.UserRole, row.to_dict())
                self.candidate_table.setItem(row_idx, col_idx, item)
        if self.candidate_table.rowCount() > 0:
            self.candidate_table.selectRow(0)
        self._refresh_candidate_notes()

    def _refresh_candidate_notes(self) -> None:
        selected = self._selected_candidate()
        if not selected:
            self.candidate_notes.clear()
            if hasattr(self, "review_btn"):
                self.review_btn.setEnabled(False)
            return
        related_wf = self._related_walk_forward_studies_for_candidate(selected)
        related_mc = self._related_monte_carlo_studies_for_candidate(selected)
        params = OptimizationStudyDialog._parse_json_text(selected.get("params_json"))
        metrics = OptimizationStudyDialog._parse_json_text(selected.get("metrics_json"))
        blocks = list(params.get("strategy_blocks") or [])
        lines = [
            f"Timeframe: {selected.get('timeframe', '') or '—'}",
            f"Reason: {selected.get('promotion_reason', '') or 'Manual review'}",
            f"Robust Score: {OptimizationStudyDialog._format_numeric_value(metrics.get('robust_score'))}",
            f"Median Sharpe: {OptimizationStudyDialog._format_numeric_value(metrics.get('median_sharpe'))}",
            f"Median Return: {OptimizationStudyDialog._format_numeric_value(metrics.get('median_total_return'))}",
            f"Candidate Blocks: {len(blocks)}",
            f"Linked Walk-Forward Studies: {len(related_wf)}",
            f"Linked Monte Carlo Studies: {len(related_mc)}",
            "",
            str(selected.get("notes", "") or "No analyst notes."),
        ]
        self.candidate_notes.setPlainText("\n".join(lines))
        if hasattr(self, "review_btn"):
            self.review_btn.setEnabled(True)

    def _related_walk_forward_studies(self) -> pd.DataFrame:
        studies = self.catalog.load_walk_forward_studies()
        if studies.empty:
            return studies
        matches: list[bool] = []
        for _, row in studies.iterrows():
            params_payload = OptimizationStudyDialog._parse_json_text(row.get("params_json"))
            matches.append(str(params_payload.get("source_batch_id", "") or "") == self.batch_id)
        return studies.loc[matches].reset_index(drop=True)

    def _related_walk_forward_studies_for_candidate(self, candidate_row: dict) -> pd.DataFrame:
        param_key = str(candidate_row.get("param_key", "") or "")
        if not param_key:
            return pd.DataFrame()
        rows: list[dict] = []
        for _, study_row in self._related_walk_forward_studies().iterrows():
            wf_study_id = str(study_row.get("wf_study_id", "") or "")
            if not wf_study_id:
                continue
            folds = self.catalog.load_walk_forward_folds(wf_study_id)
            if folds.empty:
                continue
            if folds["selected_param_set_id"].fillna("").astype(str).eq(param_key).any():
                rows.append(study_row.to_dict())
        return pd.DataFrame(rows)

    def _related_monte_carlo_studies_for_candidate(self, candidate_row: dict) -> pd.DataFrame:
        related_wf = self._related_walk_forward_studies_for_candidate(candidate_row)
        if related_wf.empty:
            return pd.DataFrame()
        studies = self.catalog.load_monte_carlo_studies()
        if studies.empty:
            return studies
        wf_ids = {str(item) for item in related_wf["wf_study_id"].tolist() if str(item).strip()}
        return studies.loc[studies["source_id"].fillna("").isin(wf_ids)].reset_index(drop=True)

    def _open_validation_chain(self, *, focus_candidate_id: str = "", initial_tab: str = "candidates") -> None:
        dlg = PortfolioValidationChainDialog(
            title_context=self.batch_id,
            catalog=self.catalog,
            portfolio_mode="strategy_blocks",
            dataset_ids=self.dataset_ids,
            batch_params=self.batch_params,
            source_batch_id=self.batch_id,
            candidates=self.candidates.copy(),
            walk_forward_studies=self._related_walk_forward_studies(),
            focus_candidate_id=str(focus_candidate_id or ""),
            initial_tab=initial_tab,
            parent=self,
        )
        dlg.exec()

    def _review_selected_candidate(self) -> None:
        selected = self._selected_candidate()
        if not selected:
            QtWidgets.QMessageBox.information(self, "No candidate selected", "Select a queued definition first.")
            return
        self._open_validation_chain(
            focus_candidate_id=str(selected.get("candidate_id", "") or ""),
            initial_tab="candidates",
        )

    def _promote_current_definition(self) -> None:
        if not self.strategy_blocks:
            QtWidgets.QMessageBox.warning(
                self,
                "No Strategy Blocks",
                "This batch does not contain fixed strategy blocks to promote.",
            )
            return
        reasons = [
            "Manual review",
            "Portfolio structure conviction",
            "Clear block diversification",
            "Budget-weight thesis",
            "Walk-forward shortlist",
        ]
        dlg = CandidatePromotionDialog(reasons, self)
        if dlg.exec() != int(QtWidgets.QDialog.DialogCode.Accepted):
            return
        reason, notes = dlg.values()
        scope = str(self.timeframe_scope_combo.currentData() or self._ALL_TIMEFRAMES)
        timeframes = self.timeframes if scope == self._ALL_TIMEFRAMES else [scope]
        if not timeframes:
            QtWidgets.QMessageBox.warning(self, "No timeframe", "This batch does not list any timeframes to promote.")
            return
        params_json = json.dumps(self._strategy_block_payload(), sort_keys=True)
        artifact_refs = {
            "source_kind": "portfolio_fixed_blocks",
            "batch_id": self.batch_id,
            "timeframes": list(timeframes),
        }
        for timeframe in timeframes:
            self.catalog.save_optimization_candidate(
                study_id=self.batch_id,
                timeframe=str(timeframe),
                start="",
                end="",
                param_key=params_json,
                params_json=params_json,
                source_type="fixed_portfolio_definition",
                promotion_reason=str(reason or "Manual review"),
                status="queued",
                metrics=self._candidate_metrics_for_timeframe(str(timeframe)),
                asset_results=None,
                artifact_refs=artifact_refs,
                notes=str(notes or ""),
            )
        self._refresh_candidate_queue()

    def _remove_selected_candidate(self) -> None:
        selected = self._selected_candidate()
        if not selected:
            QtWidgets.QMessageBox.information(self, "No candidate selected", "Select a queued definition first.")
            return
        confirm = QtWidgets.QMessageBox.question(
            self,
            "Remove Candidate",
            "Remove this fixed portfolio definition from the validation queue?",
        )
        if confirm != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self.catalog.delete_optimization_candidate(str(selected.get("candidate_id", "")))
        self._refresh_candidate_queue()


class OptimizationStudyDialog(DashboardDialog):
    _METRIC_LABELS = {
        "robust_score": "Robust Score",
        "median_sharpe": "Median Sharpe",
        "median_total_return": "Median Return",
        "worst_max_drawdown": "Worst Max Drawdown",
        "sharpe_std": "Sharpe Std",
        "profitable_asset_ratio": "Profitable Asset Ratio",
    }

    def __init__(self, study_row: dict, catalog: CatalogReader, parent=None) -> None:
        super().__init__(parent)
        self.study_row = dict(study_row)
        self.catalog = catalog
        self.study_id = str(self.study_row.get("study_id", ""))
        self.param_names = [str(name) for name in list(self.study_row.get("param_names") or [])]
        self.aggregates = self.catalog.load_optimization_aggregates(self.study_id)
        self.asset_results = self.catalog.load_optimization_asset_results(self.study_id)
        self.candidates = self.catalog.load_optimization_candidates(self.study_id)
        self.batch_params = self._batch_params_dict()
        self.is_portfolio_study = str(self.batch_params.get("_study_mode", "")) == STUDY_MODE_PORTFOLIO
        self.portfolio_mode = (
            "strategy_blocks"
            if list(self.batch_params.get("_portfolio_strategy_blocks") or [])
            else ("shared_strategy" if self.is_portfolio_study else "none")
        )
        self.portfolio_dataset_ids = self._portfolio_dataset_scope()
        self.setWindowTitle(f"Optimization Study | {self.study_id}")
        screen = QtWidgets.QApplication.primaryScreen()
        available = screen.availableGeometry() if screen is not None else QtCore.QRect(0, 0, 1720, 1040)
        self.resize(min(1760, max(1380, available.width() - 80)), min(1040, max(900, available.height() - 80)))
        self.setObjectName("Panel")
        self.setStyleSheet(
            f"""
            QDialog#Panel {{
                background: {PALETTE['panel']};
                color: {PALETTE['text']};
            }}
            QGroupBox {{
                color: {PALETTE['text']};
                border: 1px solid rgba(231, 238, 252, 0.45);
                border-radius: 10px;
                margin-top: 12px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: {PALETTE['muted']};
                font-weight: 700;
            }}
            """
        )

        layout = QtWidgets.QVBoxLayout(self)
        dataset_scope = list(self.study_row.get("dataset_scope") or [])
        summary_lines = [
            f"Strategy: {self.study_row.get('strategy', '')}",
            f"Batch: {self.study_row.get('batch_id', '')}",
            f"Datasets: {len(dataset_scope)}",
            f"Params: {', '.join(self.param_names)}",
            f"Score: {self.study_row.get('score_version', '')}",
        ]
        if self.is_portfolio_study:
            summary_lines.extend(
                [
                    f"Study Mode: {'Fixed Strategy Blocks' if self.portfolio_mode == 'strategy_blocks' else 'Shared Strategy Portfolio'}",
                    f"Underlying Assets: {', '.join(self.portfolio_dataset_ids) or '—'}",
                    f"Allocation: {str(self.batch_params.get('_portfolio_allocation_mode', PORTFOLIO_ALLOC_EQUAL)).replace('_', ' ').title()}",
                    f"Ownership: {str(self.batch_params.get('_portfolio_allocation_ownership', ALLOCATION_OWNERSHIP_STRATEGY)).replace('_', ' ').title()}",
                ]
            )
        summary_text = "\n".join(summary_lines)
        summary = QtWidgets.QPlainTextEdit()
        summary.setReadOnly(True)
        summary.setPlainText(summary_text)
        summary.setMaximumHeight(108)
        summary.setObjectName("Panel")
        self.validation_summary = QtWidgets.QLabel("")
        self.validation_summary.setObjectName("Sub")
        self.validation_summary.setWordWrap(True)
        self.validation_summary.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        validation_actions = QtWidgets.QHBoxLayout()
        self.open_latest_wf_btn = QtWidgets.QPushButton("Open Latest Walk-Forward")
        self.open_latest_wf_btn.clicked.connect(self._open_latest_related_walk_forward)
        self.open_latest_mc_btn = QtWidgets.QPushButton("Open Latest Monte Carlo")
        self.open_latest_mc_btn.clicked.connect(self._open_latest_related_monte_carlo)
        validation_actions.addWidget(self.open_latest_wf_btn)
        validation_actions.addWidget(self.open_latest_mc_btn)
        if self.is_portfolio_study:
            self.open_validation_chain_btn = QtWidgets.QPushButton("Portfolio Validation Chain")
            self.open_validation_chain_btn.clicked.connect(self._open_portfolio_validation_chain)
            validation_actions.addWidget(self.open_validation_chain_btn)
        validation_actions.addStretch(1)

        controls_scroll = QtWidgets.QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        controls_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        controls_scroll.setMaximumHeight(320)
        controls_scroll.setObjectName("Panel")

        controls_host = QtWidgets.QWidget()
        controls_host.setObjectName("Panel")
        controls_scroll.setWidget(controls_host)
        controls_layout = QtWidgets.QVBoxLayout(controls_host)
        controls_layout.setContentsMargins(12, 12, 12, 12)
        controls_layout.setSpacing(12)

        core_group = QtWidgets.QGroupBox("Study Controls")
        core_layout = QtWidgets.QGridLayout(core_group)
        core_layout.setHorizontalSpacing(12)
        core_layout.setVerticalSpacing(10)

        self.slice_combo = QtWidgets.QComboBox()
        self.slice_keys = self._build_slice_keys()
        for key, label in self.slice_keys:
            self.slice_combo.addItem(label, key)
        self.slice_combo.currentIndexChanged.connect(self._refresh_slice_view)
        core_layout.addWidget(QtWidgets.QLabel("Time Slice"), 0, 0)
        core_layout.addWidget(self.slice_combo, 0, 1, 1, 3)

        self.metric_combo = QtWidgets.QComboBox()
        for metric_key, label in self._METRIC_LABELS.items():
            self.metric_combo.addItem(label, metric_key)
        self.metric_combo.currentIndexChanged.connect(self._refresh_slice_view)
        core_layout.addWidget(QtWidgets.QLabel("Heatmap Metric"), 1, 0)
        core_layout.addWidget(self.metric_combo, 1, 1)

        self.row_param_combo = QtWidgets.QComboBox()
        for name in self.param_names:
            self.row_param_combo.addItem(name, name)
        self.row_param_combo.currentIndexChanged.connect(self._refresh_slice_view)
        core_layout.addWidget(QtWidgets.QLabel("Heatmap Rows"), 1, 2)
        core_layout.addWidget(self.row_param_combo, 1, 3)

        self.col_param_combo = QtWidgets.QComboBox()
        for name in self.param_names:
            self.col_param_combo.addItem(name, name)
        self.col_param_combo.currentIndexChanged.connect(self._refresh_slice_view)
        core_layout.addWidget(QtWidgets.QLabel("Heatmap Cols"), 2, 0)
        core_layout.addWidget(self.col_param_combo, 2, 1)
        controls_layout.addWidget(core_group)

        filter_group = QtWidgets.QGroupBox("Fixed Parameter Filters")
        filter_layout = QtWidgets.QGridLayout(filter_group)
        filter_layout.setContentsMargins(10, 10, 10, 10)
        filter_layout.setHorizontalSpacing(12)
        filter_layout.setVerticalSpacing(10)
        self.param_filter_controls: dict[str, tuple[QtWidgets.QLabel, QtWidgets.QComboBox]] = {}
        for idx, param_name in enumerate(self.param_names):
            label = QtWidgets.QLabel(param_name)
            combo = QtWidgets.QComboBox()
            combo.currentIndexChanged.connect(self._refresh_slice_view)
            row = idx // 2
            col = (idx % 2) * 2
            filter_layout.addWidget(label, row, col)
            filter_layout.addWidget(combo, row, col + 1)
            self.param_filter_controls[param_name] = (label, combo)
        controls_layout.addWidget(filter_group)

        if len(self.param_names) >= 2:
            self.row_param_combo.blockSignals(True)
            self.col_param_combo.blockSignals(True)
            self.row_param_combo.setCurrentIndex(1)
            self.col_param_combo.setCurrentIndex(0)
            self.row_param_combo.blockSignals(False)
            self.col_param_combo.blockSignals(False)

        self.slice_summary = QtWidgets.QLabel("")
        self.slice_summary.setObjectName("Sub")
        self.slice_summary.setWordWrap(True)
        self.slice_summary.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)

        self.main_split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.main_split.setChildrenCollapsible(False)
        layout.addWidget(self.main_split, 1)

        left_panel = QtWidgets.QWidget()
        left_panel.setObjectName("Panel")
        left_panel.setMinimumWidth(640)
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(8)

        left_title = QtWidgets.QLabel(
            "Portfolio Study Setup And Candidate Table" if self.is_portfolio_study else "Study Setup And Candidate Table"
        )
        left_title.setObjectName("Title")
        left_layout.addWidget(left_title)
        left_layout.addWidget(summary)
        left_layout.addWidget(self.validation_summary)
        left_layout.addLayout(validation_actions)
        left_layout.addWidget(controls_scroll)
        left_layout.addWidget(self.slice_summary)

        left_table_title = QtWidgets.QLabel("Candidate Table")
        left_table_title.setObjectName("Title")
        left_layout.addWidget(left_table_title)

        self.top_table = QtWidgets.QTableWidget(0, 7)
        self.top_table.setHorizontalHeaderLabels(
            [
                "Rank",
                "Candidate",
                "Params",
                "Robust Score",
                "Median Sharpe",
                "Median Return",
                "Worst DD",
            ]
        )
        self.top_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.top_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.top_table.setAlternatingRowColors(True)
        self.top_table.verticalHeader().setVisible(False)
        self.top_table.setObjectName("Panel")
        self.top_table.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        top_header = self.top_table.horizontalHeader()
        top_header.setStretchLastSection(False)
        top_header.setSectionsClickable(True)
        top_header.setCascadingSectionResizes(True)
        for idx in range(self.top_table.columnCount()):
            top_header.setSectionResizeMode(idx, QtWidgets.QHeaderView.ResizeMode.Interactive)
        top_header.resizeSection(0, 58)
        top_header.resizeSection(1, 92)
        top_header.resizeSection(2, 270)
        for idx, width in ((3, 110), (4, 118), (5, 118), (6, 100)):
            top_header.resizeSection(idx, width)
        self.top_table.itemSelectionChanged.connect(self._refresh_asset_distribution)
        left_layout.addWidget(self.top_table, 1)

        left_buttons = QtWidgets.QHBoxLayout()
        export_btn = QtWidgets.QPushButton("Export CSV")
        export_btn.clicked.connect(self._export_csv)
        left_buttons.addWidget(export_btn)
        left_buttons.addStretch(1)
        left_layout.addLayout(left_buttons)
        self.main_split.addWidget(left_panel)

        self.right_stack = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.right_stack.setChildrenCollapsible(False)
        self.main_split.addWidget(self.right_stack)

        heatmap_panel = QtWidgets.QWidget()
        heatmap_panel.setObjectName("Panel")
        heatmap_panel.setMinimumWidth(900)
        heatmap_layout = QtWidgets.QVBoxLayout(heatmap_panel)
        heatmap_layout.setContentsMargins(8, 8, 8, 8)
        heatmap_layout.setSpacing(8)
        heatmap_title = QtWidgets.QLabel("Parameter Surface Heatmap")
        heatmap_title.setObjectName("Title")
        heatmap_layout.addWidget(heatmap_title)
        self.figure = Figure(figsize=(14.5, 8.2), tight_layout=True, facecolor=PALETTE["panel"])
        self.canvas = FigureCanvasQTAgg(self.figure)
        heatmap_layout.addWidget(self.canvas, 1)
        self.right_stack.addWidget(heatmap_panel)

        right_tabs = QtWidgets.QTabWidget()
        self.right_stack.addWidget(right_tabs)

        asset_panel = QtWidgets.QWidget()
        asset_panel.setObjectName("Panel")
        right_layout = QtWidgets.QVBoxLayout(asset_panel)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(8)
        asset_controls = QtWidgets.QHBoxLayout()
        asset_controls.addWidget(QtWidgets.QLabel("Asset Metric"))
        self.asset_metric_combo = QtWidgets.QComboBox()
        self.asset_metric_combo.addItem("Sharpe", "sharpe")
        self.asset_metric_combo.addItem("Total Return", "total_return")
        self.asset_metric_combo.addItem("Max Drawdown", "max_drawdown")
        self.asset_metric_combo.currentIndexChanged.connect(self._refresh_asset_distribution)
        if self.is_portfolio_study:
            self.asset_metric_combo.setEnabled(False)
            self.asset_metric_combo.setToolTip(
                "Shared-strategy portfolio optimization currently stores portfolio-level candidate metrics and underlying asset configuration, not per-asset run metrics."
            )
        asset_controls.addWidget(self.asset_metric_combo)
        asset_controls.addStretch(1)
        right_layout.addLayout(asset_controls)
        self.asset_summary_label = QtWidgets.QLabel("")
        self.asset_summary_label.setObjectName("Sub")
        self.asset_summary_label.setWordWrap(True)
        self.asset_summary_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        right_layout.addWidget(self.asset_summary_label)

        self.asset_figure = Figure(figsize=(5.2, 3.8), tight_layout=True, facecolor=PALETTE["panel"])
        self.asset_canvas = FigureCanvasQTAgg(self.asset_figure)
        right_layout.addWidget(self.asset_canvas, 1)

        self.asset_table = QtWidgets.QTableWidget(0, 6)
        self.asset_table.setHorizontalHeaderLabels(self._asset_table_headers())
        self.asset_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.asset_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.asset_table.setAlternatingRowColors(True)
        self.asset_table.horizontalHeader().setStretchLastSection(True)
        self.asset_table.verticalHeader().setVisible(False)
        self.asset_table.setObjectName("Panel")
        right_layout.addWidget(self.asset_table, 1)
        right_tabs.addTab(asset_panel, "Portfolio Snapshot" if self.is_portfolio_study else "Asset Distribution")

        candidate_panel = QtWidgets.QWidget()
        candidate_panel.setObjectName("Panel")
        candidate_layout = QtWidgets.QVBoxLayout(candidate_panel)
        candidate_layout.setContentsMargins(8, 8, 8, 8)
        candidate_layout.setSpacing(8)
        self.candidate_summary = QtWidgets.QLabel("")
        self.candidate_summary.setObjectName("Sub")
        self.candidate_summary.setWordWrap(True)
        self.candidate_summary.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        candidate_layout.addWidget(self.candidate_summary)

        self.candidate_table = QtWidgets.QTableWidget(0, 7)
        self.candidate_table.setHorizontalHeaderLabels(
            ["Status", "Reason", "Params", "Robust Score", "Median Sharpe", "Updated", "Notes"]
        )
        self.candidate_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.candidate_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.candidate_table.setAlternatingRowColors(True)
        self.candidate_table.horizontalHeader().setStretchLastSection(True)
        self.candidate_table.verticalHeader().setVisible(False)
        self.candidate_table.setObjectName("Panel")
        self.candidate_table.itemSelectionChanged.connect(self._refresh_candidate_details)
        candidate_layout.addWidget(self.candidate_table, 1)

        self.candidate_notes = QtWidgets.QPlainTextEdit()
        self.candidate_notes.setReadOnly(True)
        self.candidate_notes.setMaximumHeight(120)
        self.candidate_notes.setObjectName("Panel")
        candidate_layout.addWidget(self.candidate_notes)

        candidate_actions = QtWidgets.QHBoxLayout()
        self.promote_btn = QtWidgets.QPushButton("Promote Selected Candidate")
        self.promote_btn.clicked.connect(self._promote_selected_candidate)
        if self.is_portfolio_study:
            self.review_candidate_btn = QtWidgets.QPushButton("Review Candidate")
            self.review_candidate_btn.clicked.connect(self._review_selected_portfolio_candidate)
            self.review_candidate_btn.setEnabled(False)
            candidate_actions.addWidget(self.review_candidate_btn)
        self.remove_candidate_btn = QtWidgets.QPushButton("Remove Candidate")
        self.remove_candidate_btn.clicked.connect(self._remove_selected_candidate)
        candidate_actions.addWidget(self.promote_btn)
        candidate_actions.addWidget(self.remove_candidate_btn)
        candidate_actions.addStretch(1)
        candidate_layout.addLayout(candidate_actions)
        right_tabs.addTab(candidate_panel, "Portfolio Candidate Queue" if self.is_portfolio_study else "Candidate Queue")

        self.main_split.setStretchFactor(0, 4)
        self.main_split.setStretchFactor(1, 7)
        self.right_stack.setStretchFactor(0, 7)
        self.right_stack.setStretchFactor(1, 4)

        close_row = QtWidgets.QHBoxLayout()
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        close_row.addStretch(1)
        close_row.addWidget(close_btn)
        layout.addLayout(close_row)

        QtCore.QTimer.singleShot(0, self._apply_default_layout_sizes)
        self._refresh_validation_summary()
        self._refresh_slice_view()

    def _apply_default_layout_sizes(self) -> None:
        total_width = max(self.width() - 48, 1200)
        left_width = max(640, int(total_width * 0.39))
        right_width = max(920, total_width - left_width)
        self.main_split.setSizes([left_width, right_width])
        total_height = max(self.main_split.height(), 640)
        heatmap_height = max(420, int(total_height * 0.62))
        lower_height = max(260, total_height - heatmap_height)
        self.right_stack.setSizes([heatmap_height, lower_height])

    def _batch_params_dict(self) -> dict:
        batch_id = str(self.study_row.get("batch_id", "") or "")
        if not batch_id:
            return {}
        for batch in self.catalog.load_batches():
            if str(batch.batch_id) != batch_id:
                continue
            try:
                return json.loads(batch.params) if batch.params else {}
            except Exception:
                return {}
        return {}

    def _related_walk_forward_studies(self) -> pd.DataFrame:
        studies = self.catalog.load_walk_forward_studies()
        if studies.empty:
            return studies
        batch_id = str(self.study_row.get("batch_id", "") or "")
        matches: list[bool] = []
        for _, row in studies.iterrows():
            params_payload = WalkForwardStudyDialog._parse_json_text(row.get("params_json"))
            source_study_id = str(params_payload.get("source_study_id", "") or "")
            source_batch_id = str(params_payload.get("source_batch_id", "") or "")
            matches.append(
                source_study_id == self.study_id
                or (self.is_portfolio_study and batch_id and source_batch_id == batch_id)
            )
        return studies.loc[matches].reset_index(drop=True)

    def _related_monte_carlo_studies(self) -> pd.DataFrame:
        walk_forward = self._related_walk_forward_studies()
        if walk_forward.empty:
            return pd.DataFrame()
        wf_ids = {str(item) for item in walk_forward["wf_study_id"].tolist() if str(item).strip()}
        if not wf_ids:
            return pd.DataFrame()
        studies = self.catalog.load_monte_carlo_studies()
        if studies.empty:
            return studies
        return studies.loc[studies["source_id"].fillna("").isin(wf_ids)].reset_index(drop=True)

    def _refresh_validation_summary(self) -> None:
        related_wf = self._related_walk_forward_studies()
        related_mc = self._related_monte_carlo_studies()
        wf_count = int(len(related_wf))
        mc_count = int(len(related_mc))
        if wf_count <= 0 and mc_count <= 0:
            self.validation_summary.setText(
                "Validation chain: no linked walk-forward or Monte Carlo studies have been saved for this optimization study yet."
            )
        else:
            latest_wf = str(related_wf.iloc[0].get("wf_study_id", "") or "") if wf_count else "—"
            latest_mc = str(related_mc.iloc[0].get("mc_study_id", "") or "") if mc_count else "—"
            self.validation_summary.setText(
                f"Validation chain: {wf_count} linked walk-forward stud{'y' if wf_count == 1 else 'ies'} | "
                f"{mc_count} linked Monte Carlo stud{'y' if mc_count == 1 else 'ies'} | "
                f"Latest WF: {latest_wf} | Latest MC: {latest_mc}"
            )
        self.open_latest_wf_btn.setEnabled(wf_count > 0)
        self.open_latest_mc_btn.setEnabled(mc_count > 0)

    def _open_latest_related_walk_forward(self) -> None:
        related = self._related_walk_forward_studies()
        if related.empty:
            QtWidgets.QMessageBox.information(
                self,
                "No Walk-Forward Linked",
                "No linked walk-forward studies have been saved for this optimization study yet.",
            )
            return
        wf_study_id = str(related.iloc[0].get("wf_study_id", "") or "")
        parent = self.logical_parent()
        if parent is not None and hasattr(parent, "_open_walk_forward_study"):
            parent._open_walk_forward_study(wf_study_id)
            return
        QtWidgets.QMessageBox.information(
            self,
            "Study Unavailable",
            f"The linked walk-forward study '{wf_study_id}' could not be opened from this dialog.",
        )

    def _open_latest_related_monte_carlo(self) -> None:
        related = self._related_monte_carlo_studies()
        if related.empty:
            QtWidgets.QMessageBox.information(
                self,
                "No Monte Carlo Linked",
                "No linked Monte Carlo studies have been saved for this optimization study yet.",
            )
            return
        mc_study_id = str(related.iloc[0].get("mc_study_id", "") or "")
        parent = self.logical_parent()
        if parent is not None and hasattr(parent, "_open_monte_carlo_study"):
            parent._open_monte_carlo_study(mc_study_id)
            return
        QtWidgets.QMessageBox.information(
            self,
            "Study Unavailable",
            f"The linked Monte Carlo study '{mc_study_id}' could not be opened from this dialog.",
        )

    def _open_portfolio_validation_chain(self, *, focus_candidate_id: str = "", initial_tab: str = "candidates") -> None:
        if not self.is_portfolio_study:
            return
        dlg = PortfolioValidationChainDialog(
            title_context=self.study_id,
            catalog=self.catalog,
            portfolio_mode=self.portfolio_mode,
            dataset_ids=self.portfolio_dataset_ids,
            batch_params=self.batch_params,
            source_study_id=self.study_id,
            source_batch_id=str(self.study_row.get("batch_id", "") or ""),
            optimization_study_row=self.study_row,
            candidates=self.candidates.copy() if self.candidates is not None else pd.DataFrame(),
            walk_forward_studies=self._related_walk_forward_studies(),
            monte_carlo_studies=self._related_monte_carlo_studies(),
            focus_candidate_id=str(focus_candidate_id or ""),
            initial_tab=initial_tab,
            parent=self,
        )
        dlg.exec()

    def _review_selected_portfolio_candidate(self) -> None:
        if not self.is_portfolio_study:
            return
        selected = self._selected_candidate_row()
        if not selected:
            QtWidgets.QMessageBox.information(self, "No candidate selected", "Select a queued portfolio candidate first.")
            return
        self._open_portfolio_validation_chain(
            focus_candidate_id=str(selected.get("candidate_id", "") or ""),
            initial_tab="candidates",
        )

    def _portfolio_dataset_scope(self) -> list[str]:
        if not self.is_portfolio_study:
            return [str(item) for item in list(self.study_row.get("dataset_scope") or []) if str(item).strip()]
        dataset_ids = [
            str(item)
            for item in list(self.batch_params.get("_portfolio_dataset_ids") or [])
            if str(item).strip()
        ]
        if dataset_ids:
            return dataset_ids
        if self.portfolio_mode == "strategy_blocks":
            return list(
                dict.fromkeys(
                    str(asset.get("dataset_id") or "")
                    for block in list(self.batch_params.get("_portfolio_strategy_blocks") or [])
                    for asset in list(block.get("assets") or [])
                    if str(asset.get("dataset_id") or "").strip()
                )
            )
        return [str(item) for item in list(self.study_row.get("dataset_scope") or []) if str(item).strip()]

    def _asset_table_headers(self) -> list[str]:
        if not self.is_portfolio_study:
            return ["Dataset", "Total Return", "Sharpe", "Rolling Sharpe", "Max DD", "Run ID"]
        return ["Dataset", "Target Weight", "Allocation", "Ownership", "Ranking", "Rebalance"]

    def _build_slice_keys(self) -> list[tuple[tuple[str, str, str], str]]:
        if self.aggregates.empty:
            return [(("", "", ""), "No slices")]
        frame = self.aggregates[["timeframe", "start", "end"]].drop_duplicates().reset_index(drop=True)
        slices: list[tuple[tuple[str, str, str], str]] = []
        for _, row in frame.iterrows():
            key = (str(row["timeframe"]), str(row["start"] or ""), str(row["end"] or ""))
            start = PortfolioReportDialog._fmt_timestamp(row["start"])
            end = PortfolioReportDialog._fmt_timestamp(row["end"])
            label = f"{key[0]} | {start} -> {end}"
            slices.append((key, label))
        return slices or [(("", "", ""), "No slices")]

    def _current_slice(self) -> tuple[str, str, str]:
        data = self.slice_combo.currentData()
        if isinstance(data, tuple) and len(data) == 3:
            return tuple(str(item) for item in data)
        return ("", "", "")

    def _current_metric(self) -> str:
        data = self.metric_combo.currentData()
        return str(data) if data else "robust_score"

    def _current_axis_params(self) -> tuple[str | None, str | None]:
        row_param = self.row_param_combo.currentData()
        col_param = self.col_param_combo.currentData()
        return (str(row_param) if row_param else None, str(col_param) if col_param else None)

    def _slice_base_aggregates(self) -> pd.DataFrame:
        timeframe, start, end = self._current_slice()
        frame = self.aggregates.copy()
        if frame.empty:
            return frame
        return frame.loc[
            (frame["timeframe"].fillna("") == timeframe)
            & (frame["start"].fillna("") == start)
            & (frame["end"].fillna("") == end)
        ].reset_index(drop=True)

    def _refresh_filter_controls(self, frame: pd.DataFrame) -> None:
        row_param, col_param = self._current_axis_params()
        for param_name, (label, combo) in self.param_filter_controls.items():
            visible = param_name not in {row_param, col_param}
            label.setVisible(visible)
            combo.setVisible(visible)
            if not visible:
                continue
            current_value = combo.currentData()
            distinct_values = []
            if frame is not None and not frame.empty and param_name in frame.columns:
                distinct_values = sorted(
                    [self._normalize_param_value(value) for value in frame[param_name].drop_duplicates().tolist()],
                    key=lambda value: str(value),
                )
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("All", None)
            for value in distinct_values:
                combo.addItem(self._format_param_value(value), value)
            if current_value in distinct_values:
                combo.setCurrentIndex(combo.findData(current_value))
            combo.blockSignals(False)

    def _slice_aggregates(self) -> pd.DataFrame:
        frame = self._slice_base_aggregates()
        if frame.empty:
            return frame
        row_param, col_param = self._current_axis_params()
        for param_name, (_label, combo) in self.param_filter_controls.items():
            if param_name in {row_param, col_param}:
                continue
            selected_value = combo.currentData()
            if selected_value is None or param_name not in frame.columns:
                continue
            frame = frame.loc[
                frame[param_name].apply(self._normalize_param_value) == self._normalize_param_value(selected_value)
            ]
        return frame.reset_index(drop=True)

    def _refresh_slice_view(self) -> None:
        base_frame = self._slice_base_aggregates()
        self._refresh_filter_controls(base_frame)
        frame = self._slice_aggregates()
        self._refresh_candidate_view()
        if frame.empty:
            self.slice_summary.setText("No aggregates are available for this study slice yet.")
            self.top_table.setRowCount(0)
            self.asset_table.setRowCount(0)
            self.figure.clear()
            self.canvas.draw_idle()
            self.asset_figure.clear()
            self.asset_canvas.draw_idle()
            return
        best = frame.iloc[0]
        metric_key = self._current_metric()
        metric_label = self._METRIC_LABELS.get(metric_key, metric_key)
        self.slice_summary.setText(
            f"Candidates: {len(frame)} | Best {metric_label}: {float(best[metric_key]):.4f} | "
            f"Best robust score: {float(best['robust_score']):.4f} | "
            f"Median Sharpe: {float(best['median_sharpe']):.4f} | "
            f"Median Return: {float(best['median_total_return']):.4f} | "
            f"Worst DD: {float(best['worst_max_drawdown']):.4f}"
        )
        self._populate_top_table(frame)
        self._draw_heatmap(frame)
        if self.top_table.rowCount() > 0:
            self.top_table.selectRow(0)
        self._refresh_asset_distribution()

    def _populate_top_table(self, frame: pd.DataFrame) -> None:
        display = frame.head(50).reset_index(drop=True)
        self.top_table.setRowCount(len(display))
        for row_idx, row in display.iterrows():
            params_json = str(row["params_json"])
            params_label = ", ".join(f"{key}={json.loads(params_json).get(key)}" for key in self.param_names)
            candidate_state = "Queued" if self._find_candidate_for_aggregate(row.to_dict()) is not None else "—"
            values = [
                str(int(row["seq"])),
                candidate_state,
                params_label,
                f"{float(row['robust_score']):.4f}",
                f"{float(row['median_sharpe']):.4f}",
                f"{float(row['median_total_return']):.4f}",
                f"{float(row['worst_max_drawdown']):.4f}",
            ]
            detail_tooltip = (
                f"Sharpe Std: {float(row['sharpe_std']):.4f}\n"
                f"Profitable %: {float(row['profitable_asset_ratio']) * 100:.1f}%\n"
                f"Assets: {int(row['dataset_count'])}\n"
                f"Runs: {int(row['run_count'])}"
            )
            for col_idx, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                if col_idx == 0:
                    item.setData(QtCore.Qt.ItemDataRole.UserRole, row.to_dict())
                item.setToolTip(detail_tooltip)
                self.top_table.setItem(row_idx, col_idx, item)

    def _selected_aggregate_row(self) -> dict | None:
        selection_model = self.top_table.selectionModel()
        if selection_model is None:
            return None
        rows = selection_model.selectedRows()
        if not rows:
            return None
        item = self.top_table.item(rows[0].row(), 0)
        if item is None:
            return None
        payload = item.data(QtCore.Qt.ItemDataRole.UserRole)
        return payload if isinstance(payload, dict) else None

    def _candidate_lookup(self) -> dict[tuple[str, str, str, str], dict]:
        if self.candidates is None or self.candidates.empty:
            return {}
        lookup: dict[tuple[str, str, str, str], dict] = {}
        for _, row in self.candidates.iterrows():
            lookup[self._candidate_key(row.to_dict())] = row.to_dict()
        return lookup

    @staticmethod
    def _candidate_key(row: dict) -> tuple[str, str, str, str]:
        return (
            str(row.get("timeframe", "") or ""),
            str(row.get("start", "") or ""),
            str(row.get("end", "") or ""),
            str(row.get("param_key", "") or ""),
        )

    def _find_candidate_for_aggregate(self, aggregate_row: dict) -> dict | None:
        return self._candidate_lookup().get(self._candidate_key(aggregate_row))

    def _refresh_candidate_view(self) -> None:
        frame = self.candidates.copy() if self.candidates is not None else pd.DataFrame()
        if frame.empty:
            self.candidate_summary.setText(
                (
                    "No promoted portfolio candidates yet. Select a strong portfolio parameter set and add it to the queue."
                    if self.is_portfolio_study
                    else "No promoted candidates yet. Select a strong parameter set and add it to the queue."
                )
            )
            self.candidate_table.setRowCount(0)
            self.candidate_notes.clear()
            if self.is_portfolio_study and hasattr(self, "review_candidate_btn"):
                self.review_candidate_btn.setEnabled(False)
            return
        metric_frame = frame["metrics_json"].apply(self._parse_json_text).apply(pd.Series)
        for col in metric_frame.columns:
            if col not in frame.columns:
                frame[col] = metric_frame[col]
        frame = self._expand_candidate_params(frame)
        frame = frame.sort_values(by=["updated_at", "created_at"], ascending=[False, False], kind="mergesort")
        self.candidate_summary.setText(
            f"{'Queued portfolio candidates' if self.is_portfolio_study else 'Queued candidates'}: {len(frame)} | "
            f"{'Portfolio studies' if self.is_portfolio_study else 'Studies'} in queue: {frame['study_id'].nunique()} | "
            f"Latest update: {PortfolioReportDialog._fmt_timestamp(frame.iloc[0].get('updated_at'))}"
        )
        self.candidate_table.setRowCount(len(frame))
        for row_idx, row in frame.reset_index(drop=True).iterrows():
            params_label = ", ".join(f"{key}={row.get(key)}" for key in self.param_names if key in row)
            values = [
                str(row.get("status", "")),
                str(row.get("promotion_reason", "") or "Manual review"),
                params_label,
                self._format_numeric_value(row.get("robust_score")),
                self._format_numeric_value(row.get("median_sharpe")),
                PortfolioReportDialog._fmt_timestamp(row.get("updated_at")),
                str(row.get("notes", "") or ""),
            ]
            for col_idx, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                if col_idx == 0:
                    item.setData(QtCore.Qt.ItemDataRole.UserRole, row.to_dict())
                self.candidate_table.setItem(row_idx, col_idx, item)
        if self.candidate_table.rowCount() > 0 and self.candidate_table.selectionModel() is not None:
            self.candidate_table.selectRow(0)
        self._refresh_candidate_details()

    def _selected_candidate_row(self) -> dict | None:
        selection_model = self.candidate_table.selectionModel()
        if selection_model is None:
            return None
        rows = selection_model.selectedRows()
        if not rows:
            return None
        item = self.candidate_table.item(rows[0].row(), 0)
        if item is None:
            return None
        payload = item.data(QtCore.Qt.ItemDataRole.UserRole)
        return payload if isinstance(payload, dict) else None

    def _refresh_candidate_details(self) -> None:
        selected = self._selected_candidate_row()
        if not selected:
            self.candidate_notes.clear()
            if self.is_portfolio_study and hasattr(self, "review_candidate_btn"):
                self.review_candidate_btn.setEnabled(False)
            return
        metrics = self._parse_json_text(selected.get("metrics_json"))
        notes_lines = [
            f"Status: {selected.get('status', '') or 'queued'}",
            f"Reason: {selected.get('promotion_reason', '') or 'Manual review'}",
            f"Timeframe: {selected.get('timeframe', '')}",
            f"Window: {PortfolioReportDialog._fmt_timestamp(selected.get('start'))} -> {PortfolioReportDialog._fmt_timestamp(selected.get('end'))}",
            f"Robust Score: {self._format_numeric_value(metrics.get('robust_score'))}",
            f"Median Sharpe: {self._format_numeric_value(metrics.get('median_sharpe'))}",
            f"Median Return: {self._format_numeric_value(metrics.get('median_total_return'))}",
        ]
        if self.is_portfolio_study:
            related_wf = self._candidate_related_walk_forward_studies(selected)
            related_mc = self._candidate_related_monte_carlo_studies(selected)
            notes_lines.extend(
                [
                    f"Linked Walk-Forward Studies: {len(related_wf)}",
                    f"Linked Monte Carlo Studies: {len(related_mc)}",
                ]
            )
            if hasattr(self, "review_candidate_btn"):
                self.review_candidate_btn.setEnabled(True)
        notes_lines.extend(["", str(selected.get("notes", "") or "No analyst notes.")])
        self.candidate_notes.setPlainText("\n".join(notes_lines))

    def _candidate_related_walk_forward_studies(self, candidate_row: dict) -> pd.DataFrame:
        param_key = str(candidate_row.get("param_key", "") or "")
        if not param_key:
            return pd.DataFrame()
        rows: list[dict] = []
        for _, study_row in self._related_walk_forward_studies().iterrows():
            wf_study_id = str(study_row.get("wf_study_id", "") or "")
            if not wf_study_id:
                continue
            folds = self.catalog.load_walk_forward_folds(wf_study_id)
            if folds.empty:
                continue
            if folds["selected_param_set_id"].fillna("").astype(str).eq(param_key).any():
                rows.append(study_row.to_dict())
        return pd.DataFrame(rows)

    def _candidate_related_monte_carlo_studies(self, candidate_row: dict) -> pd.DataFrame:
        related_wf = self._candidate_related_walk_forward_studies(candidate_row)
        if related_wf.empty:
            return pd.DataFrame()
        studies = self._related_monte_carlo_studies()
        if studies.empty:
            return studies
        wf_ids = {str(item) for item in related_wf["wf_study_id"].tolist() if str(item).strip()}
        return studies.loc[studies["source_id"].fillna("").isin(wf_ids)].reset_index(drop=True)

    def _promote_selected_candidate(self) -> None:
        selected = self._selected_aggregate_row()
        if not selected:
            QtWidgets.QMessageBox.information(self, "No candidate selected", "Select a parameter row first.")
            return
        reasons = [
            "Manual review",
            "Broad plateau",
            "Strong robust score",
            "Cross-asset consistency",
            "Low drawdown profile",
        ]
        dlg = CandidatePromotionDialog(reasons, self)
        if dlg.exec() != int(QtWidgets.QDialog.DialogCode.Accepted):
            return
        reason, notes = dlg.values()
        asset_results = build_asset_distribution_frame(
            self.asset_results,
            param_key=str(selected.get("param_key", "")),
            timeframe=str(selected.get("timeframe", "")),
            start=str(selected.get("start", "")),
            end=str(selected.get("end", "")),
        )
        artifact_refs = {
            "study_id": self.study_id,
            "slice": {
                "timeframe": str(selected.get("timeframe", "") or ""),
                "start": str(selected.get("start", "") or ""),
                "end": str(selected.get("end", "") or ""),
            },
            "heatmap_metric": self._current_metric(),
            "row_param": self._current_axis_params()[0],
            "col_param": self._current_axis_params()[1],
            "filters": {
                param_name: combo.currentData()
                for param_name, (_label, combo) in self.param_filter_controls.items()
                if combo.currentData() is not None
            },
        }
        metrics = {
            "robust_score": float(selected.get("robust_score", 0.0)),
            "median_sharpe": float(selected.get("median_sharpe", 0.0)),
            "median_total_return": float(selected.get("median_total_return", 0.0)),
            "worst_max_drawdown": float(selected.get("worst_max_drawdown", 0.0)),
            "sharpe_std": float(selected.get("sharpe_std", 0.0)),
            "profitable_asset_ratio": float(selected.get("profitable_asset_ratio", 0.0)),
            "dataset_count": int(selected.get("dataset_count", 0)),
            "run_count": int(selected.get("run_count", 0)),
        }
        self.catalog.save_optimization_candidate(
            study_id=self.study_id,
            timeframe=str(selected.get("timeframe", "") or ""),
            start=str(selected.get("start", "") or ""),
            end=str(selected.get("end", "") or ""),
            param_key=str(selected.get("param_key", "") or ""),
            params_json=str(selected.get("params_json", "") or "{}"),
            source_type="manual_selection",
            promotion_reason=str(reason or "Manual review"),
            status="queued",
            metrics=metrics,
            asset_results=asset_results,
            artifact_refs=artifact_refs,
            notes=str(notes or ""),
        )
        self.candidates = self.catalog.load_optimization_candidates(self.study_id)
        self._refresh_slice_view()

    def _remove_selected_candidate(self) -> None:
        selected = self._selected_candidate_row()
        if not selected:
            QtWidgets.QMessageBox.information(self, "No candidate selected", "Select a queued candidate first.")
            return
        confirm = QtWidgets.QMessageBox.question(
            self,
            "Remove Candidate",
            "Remove this candidate from the validation queue?",
        )
        if confirm != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self.catalog.delete_optimization_candidate(str(selected.get("candidate_id", "")))
        self.candidates = self.catalog.load_optimization_candidates(self.study_id)
        self._refresh_slice_view()

    def _refresh_asset_distribution(self) -> None:
        selected = self._selected_aggregate_row()
        if not selected:
            self.asset_table.setRowCount(0)
            self.asset_summary_label.clear()
            self.asset_figure.clear()
            self.asset_canvas.draw_idle()
            return
        frame = build_asset_distribution_frame(
            self.asset_results,
            param_key=str(selected.get("param_key", "")),
            timeframe=str(selected.get("timeframe", "")),
            start=str(selected.get("start", "")),
            end=str(selected.get("end", "")),
        ).reset_index(drop=True)
        if self.is_portfolio_study:
            self._refresh_portfolio_snapshot(selected, frame)
            return
        self.asset_table.setRowCount(len(frame))
        for row_idx, row in frame.iterrows():
            values = [
                str(row["dataset_id"]),
                f"{float(row['total_return']):.4f}",
                f"{float(row['sharpe']):.4f}",
                f"{float(row['rolling_sharpe']):.4f}" if pd.notna(row["rolling_sharpe"]) else "—",
                f"{float(row['max_drawdown']):.4f}",
                str(row["run_id"]),
            ]
            for col_idx, value in enumerate(values):
                self.asset_table.setItem(row_idx, col_idx, QtWidgets.QTableWidgetItem(value))
        self.asset_summary_label.setText(
            "Per-asset distribution for the selected parameter set across the study slice."
        )
        self._draw_asset_distribution_charts(frame)

    def _refresh_portfolio_snapshot(self, selected: dict, frame: pd.DataFrame) -> None:
        dataset_ids = list(self.portfolio_dataset_ids)
        allocation_mode = str(self.batch_params.get("_portfolio_allocation_mode", PORTFOLIO_ALLOC_EQUAL) or PORTFOLIO_ALLOC_EQUAL)
        ownership = str(
            self.batch_params.get("_portfolio_allocation_ownership", ALLOCATION_OWNERSHIP_STRATEGY) or ALLOCATION_OWNERSHIP_STRATEGY
        )
        ranking_mode = str(self.batch_params.get("_portfolio_ranking_mode", RANKING_MODE_NONE) or RANKING_MODE_NONE)
        rebalance_mode = str(self.batch_params.get("_portfolio_rebalance_mode", REBALANCE_MODE_ON_CHANGE) or REBALANCE_MODE_ON_CHANGE)
        target_weights = {
            str(dataset_id): float(weight)
            for dataset_id, weight in dict(self.batch_params.get("_portfolio_target_weights", {}) or {}).items()
            if str(dataset_id).strip()
        }
        if not target_weights and dataset_ids and allocation_mode == PORTFOLIO_ALLOC_EQUAL:
            equal_weight = 1.0 / max(1, len(dataset_ids))
            target_weights = {dataset_id: equal_weight for dataset_id in dataset_ids}
        if not dataset_ids and not frame.empty:
            dataset_ids = [str(frame.iloc[0].get("dataset_id", "")) or "Portfolio Aggregate"]

        self.asset_table.setColumnCount(6)
        self.asset_table.setHorizontalHeaderLabels(self._asset_table_headers())
        self.asset_table.setRowCount(len(dataset_ids))
        for row_idx, dataset_id in enumerate(dataset_ids):
            values = [
                str(dataset_id),
                self._format_numeric_value(target_weights.get(dataset_id)),
                allocation_mode.replace("_", " ").title(),
                ownership.replace("_", " ").title(),
                ranking_mode.replace("_", " ").title(),
                rebalance_mode.replace("_", " ").title(),
            ]
            for col_idx, value in enumerate(values):
                self.asset_table.setItem(row_idx, col_idx, QtWidgets.QTableWidgetItem(value))
        self.asset_summary_label.setText(
            "Shared-strategy portfolio optimization stores portfolio-level candidate metrics, not per-asset run results. "
            "The charts below show the selected portfolio candidate plus the configured underlying asset universe."
        )
        self._draw_portfolio_snapshot_charts(selected, dataset_ids, target_weights)

    def _draw_portfolio_snapshot_charts(
        self,
        selected: dict,
        dataset_ids: Sequence[str],
        target_weights: dict[str, float],
    ) -> None:
        self.asset_figure.clear()
        self.asset_figure.set_facecolor(PALETTE["panel"])
        ax_left = self.asset_figure.add_subplot(121)
        ax_right = self.asset_figure.add_subplot(122)
        chart_bg = "#16233b"
        for ax in (ax_left, ax_right):
            ax.set_facecolor(chart_bg)
            ax.grid(alpha=0.14, color=PALETTE["grid"])
            ax.tick_params(axis="x", colors=PALETTE["text"], labelsize=11)
            ax.tick_params(axis="y", colors=PALETTE["text"], labelsize=10)
            for spine in ax.spines.values():
                spine.set_color(PALETTE["border"])
                spine.set_alpha(0.18)

        candidate_metrics = [
            ("Robust", float(selected.get("robust_score", 0.0) or 0.0)),
            ("Sharpe", float(selected.get("median_sharpe", 0.0) or 0.0)),
            ("Return", float(selected.get("median_total_return", 0.0) or 0.0)),
            ("Max DD", abs(float(selected.get("worst_max_drawdown", 0.0) or 0.0))),
        ]
        ax_left.bar(
            [label for label, _ in candidate_metrics],
            [value for _, value in candidate_metrics],
            color=[PALETTE["amber"], PALETTE["blue"], PALETTE["green"], PALETTE["red"]],
            alpha=0.88,
        )
        ax_left.set_title("Selected Portfolio Candidate Metrics", color=PALETTE["text"], fontsize=15, pad=8)
        for label in ax_left.get_xticklabels():
            label.set_color(PALETTE["text"])
        for label in ax_left.get_yticklabels():
            label.set_color(PALETTE["text"])

        ordered_dataset_ids = list(dataset_ids)
        ordered_weights = np.asarray(
            [float(target_weights.get(dataset_id, 0.0) or 0.0) for dataset_id in ordered_dataset_ids],
            dtype=float,
        )
        if not ordered_dataset_ids:
            ax_right.text(0.5, 0.5, "No underlying assets were stored for this portfolio study.", ha="center", va="center", color=PALETTE["muted"])
            ax_right.set_xticks([])
            ax_right.set_yticks([])
        else:
            ax_right.barh(ordered_dataset_ids, ordered_weights, color=PALETTE["blue"], alpha=0.82)
            ax_right.set_title("Configured Underlying Asset Weights", color=PALETTE["text"], fontsize=15, pad=8)
            ax_right.axvline(0.0, color=PALETTE["muted"], linewidth=1.0, alpha=0.75)
            for label in ax_right.get_yticklabels():
                label.set_color(PALETTE["text"])
            for label in ax_right.get_xticklabels():
                label.set_color(PALETTE["text"])
        self.asset_figure.tight_layout()
        self.asset_canvas.draw_idle()

    def _draw_asset_distribution_charts(self, frame: pd.DataFrame) -> None:
        self.asset_figure.clear()
        self.asset_figure.set_facecolor(PALETTE["panel"])
        ax_bar = self.asset_figure.add_subplot(121)
        ax_box = self.asset_figure.add_subplot(122)
        chart_bg = "#16233b"
        for ax in (ax_bar, ax_box):
            ax.set_facecolor(chart_bg)
            ax.grid(alpha=0.14, color=PALETTE["grid"])
            ax.tick_params(axis="x", colors=PALETTE["text"], labelsize=11)
            ax.tick_params(axis="y", colors=PALETTE["text"], labelsize=10)
            for spine in ax.spines.values():
                spine.set_color(PALETTE["border"])
                spine.set_alpha(0.18)
        if frame.empty:
            ax_bar.text(
                0.5,
                0.5,
                "No asset results available for the selected candidate.",
                ha="center",
                va="center",
                color=PALETTE["muted"],
                fontsize=10,
            )
            ax_bar.set_xticks([])
            ax_bar.set_yticks([])
            ax_box.set_xticks([])
            ax_box.set_yticks([])
            self.asset_canvas.draw_idle()
            return
        metric_key = str(self.asset_metric_combo.currentData() or "sharpe")
        metric_label = {
            "sharpe": "Sharpe",
            "total_return": "Total Return",
            "max_drawdown": "Max Drawdown",
        }.get(metric_key, metric_key)
        plot_frame = frame.sort_values(by=[metric_key, "dataset_id"], ascending=[metric_key == "max_drawdown", True])
        values = pd.to_numeric(plot_frame[metric_key], errors="coerce").fillna(0.0)
        ax_bar.barh(plot_frame["dataset_id"], values.to_numpy(dtype=float), color=PALETTE["blue"], alpha=0.85)
        ax_bar.set_title(f"{metric_label} By Asset", color=PALETTE["text"], fontsize=15, pad=8)
        ax_bar.axvline(0.0, color=PALETTE["muted"], linewidth=1.0, alpha=0.8)
        ax_bar.tick_params(axis="y", pad=6)
        for label in ax_bar.get_yticklabels():
            label.set_color(PALETTE["text"])
        for label in ax_bar.get_xticklabels():
            label.set_color(PALETTE["text"])
        box_data = [
            pd.to_numeric(frame["total_return"], errors="coerce").dropna().to_numpy(dtype=float),
            pd.to_numeric(frame["sharpe"], errors="coerce").dropna().to_numpy(dtype=float),
            pd.to_numeric(frame["max_drawdown"], errors="coerce").dropna().to_numpy(dtype=float),
        ]
        ax_box.boxplot(
            box_data,
            labels=["Return", "Sharpe", "Max DD"],
            patch_artist=True,
            boxprops={"facecolor": "#203253", "edgecolor": PALETTE["border"]},
            medianprops={"color": PALETTE["amber"], "linewidth": 1.4},
            whiskerprops={"color": PALETTE["border"]},
            capprops={"color": PALETTE["border"]},
        )
        ax_box.set_title("Asset Distribution Snapshot", color=PALETTE["text"], fontsize=15, pad=8)
        for label in ax_box.get_xticklabels():
            label.set_color(PALETTE["text"])
        for label in ax_box.get_yticklabels():
            label.set_color(PALETTE["text"])
        self.asset_figure.tight_layout()
        self.asset_canvas.draw_idle()

    def _draw_heatmap(self, frame: pd.DataFrame) -> None:
        self.figure.clear()
        self.figure.set_facecolor(PALETTE["panel"])
        ax = self.figure.add_subplot(111)
        ax.set_facecolor(PALETTE["bg"])
        row_key, col_key = self._current_axis_params()
        metric_key = self._current_metric()
        metric_label = self._METRIC_LABELS.get(metric_key, metric_key)
        if len(self.param_names) >= 2 and row_key and col_key and row_key != col_key:
            heatmap_df = frame.copy()
            pivot = (
                heatmap_df.pivot_table(
                    index=row_key,
                    columns=col_key,
                    values=metric_key,
                    aggfunc="median",
                )
                .sort_index(axis=0)
                .sort_index(axis=1)
            )
            if pivot.empty:
                ax.text(
                    0.5,
                    0.5,
                    "No heatmap values are available for the current slice.",
                    ha="center",
                    va="center",
                    color=PALETTE["muted"],
                    fontsize=11,
                )
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                values = pivot.to_numpy(dtype=float)
                finite_values = values[np.isfinite(values)]
                if finite_values.size == 0:
                    ax.text(
                        0.5,
                        0.5,
                        "No finite heatmap values are available for the current slice.",
                        ha="center",
                        va="center",
                        color=PALETTE["muted"],
                        fontsize=11,
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])
                else:
                    vmin = float(finite_values.min())
                    vmax = float(finite_values.max())
                    if np.isclose(vmin, vmax):
                        pad = max(1e-6, abs(vmin) * 0.05 if vmin != 0 else 0.05)
                        vmin -= pad
                        vmax += pad
                    image = ax.imshow(values, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
                    ax.set_title(f"{metric_label} Heatmap", color=PALETTE["text"], fontsize=14, pad=10)
                    ax.set_xlabel(col_key, color=PALETTE["muted"])
                    ax.set_ylabel(row_key, color=PALETTE["muted"])
                    col_labels = [self._format_param_value(value) for value in list(pivot.columns)]
                    row_labels = [self._format_param_value(value) for value in list(pivot.index)]
                    ax.set_xticks(np.arange(len(col_labels)))
                    ax.set_yticks(np.arange(len(row_labels)))
                    ax.set_xticklabels(col_labels, rotation=35, ha="right", color=PALETTE["text"])
                    ax.set_yticklabels(row_labels, color=PALETTE["text"])
                    ax.tick_params(colors=PALETTE["text"])
                    for spine in ax.spines.values():
                        spine.set_color(PALETTE["border"])
                        spine.set_alpha(0.25)
                    for row_idx in range(values.shape[0]):
                        for col_idx in range(values.shape[1]):
                            value = values[row_idx, col_idx]
                            if not np.isfinite(value):
                                continue
                            normalized = 0.5 if np.isclose(vmax, vmin) else (value - vmin) / (vmax - vmin)
                            text_color = PALETTE["bg"] if normalized > 0.62 else PALETTE["text"]
                            ax.text(
                                col_idx,
                                row_idx,
                                f"{value:.3f}",
                                ha="center",
                                va="center",
                                color=text_color,
                                fontsize=9,
                                fontweight="semibold",
                            )
                    colorbar = self.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
                    colorbar.ax.yaxis.set_tick_params(color=PALETTE["text"])
                    for tick in colorbar.ax.get_yticklabels():
                        tick.set_color(PALETTE["text"])
                    colorbar.outline.set_edgecolor(PALETTE["border"])
                    colorbar.outline.set_alpha(0.25)
        else:
            ax.text(
                0.5,
                0.5,
                "Heatmap requires two different strategy parameters.\nUse the filter controls to fix other parameters and inspect the remaining surface.",
                ha="center",
                va="center",
                color=PALETTE["muted"],
                fontsize=11,
            )
            ax.set_xticks([])
            ax.set_yticks([])
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _export_csv(self) -> None:
        if self.aggregates.empty:
            QtWidgets.QMessageBox.information(self, "No data", "There is nothing to export yet.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Optimization Study CSV",
            f"optimization_{self.study_id}_aggregates.csv",
            "CSV Files (*.csv)",
        )
        if not path:
            return
        try:
            base = Path(path)
            self.aggregates.to_csv(base, index=False)
            self.asset_results.to_csv(base.with_name(f"{base.stem}_asset_results.csv"), index=False)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Export error", str(exc))

    @staticmethod
    def _normalize_param_value(value):
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            numeric = float(value)
            if numeric.is_integer():
                return int(numeric)
            return numeric
        return value

    @staticmethod
    def _format_param_value(value) -> str:
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    @staticmethod
    def _parse_json_text(value) -> dict:
        if isinstance(value, dict):
            return value
        if isinstance(value, str) and value:
            try:
                payload = json.loads(value)
                return payload if isinstance(payload, dict) else {}
            except Exception:
                return {}
        return {}

    def _expand_candidate_params(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty or "params_json" not in frame.columns:
            return frame
        expanded = frame.copy()
        params_frame = expanded["params_json"].apply(self._parse_json_text).apply(pd.Series)
        for col in params_frame.columns:
            if col not in expanded.columns:
                expanded[col] = params_frame[col]
        return expanded

    @staticmethod
    def _format_numeric_value(value) -> str:
        try:
            if value is None or pd.isna(value):
                return "—"
            return f"{float(value):.4f}"
        except Exception:
            return "—"


class PortfolioReportDialog(DashboardDialog):
    def __init__(self, run: RunRow, report, parent=None) -> None:
        super().__init__(parent)
        self.run = run
        self.report = report
        self.setWindowTitle(f"Portfolio Report | {run.run_id[:10]}")
        self.resize(1460, 920)
        self.setMinimumSize(1240, 760)
        self.setObjectName("Panel")
        self.setStyleSheet(
            f"""
            QDialog#Panel {{
                background: {PALETTE['panel']};
                color: {PALETTE['text']};
            }}
            QLabel {{
                color: {PALETTE['text']};
            }}
            QLabel#Sub {{
                color: {PALETTE['muted']};
            }}
            QPlainTextEdit#Panel, QTableWidget#Panel {{
                background: {PALETTE['panel2']};
                color: {PALETTE['text']};
                border: 1px solid rgba(231, 238, 252, 0.55);
                border-radius: 10px;
            }}
            QTabWidget::pane {{
                border: 1px solid rgba(231, 238, 252, 0.35);
                border-radius: 10px;
                top: -1px;
                background: {PALETTE['panel']};
            }}
            QTabBar::tab {{
                background: rgba(255, 255, 255, 0.05);
                color: {PALETTE['muted']};
                border: 1px solid rgba(231, 238, 252, 0.35);
                border-bottom: none;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
                padding: 8px 16px;
                min-width: 96px;
            }}
            QTabBar::tab:selected {{
                background: {PALETTE['panel2']};
                color: {PALETTE['text']};
                font-weight: 700;
            }}
            QHeaderView::section {{
                background: rgba(255, 255, 255, 0.04);
                color: {PALETTE['muted']};
                border: none;
                border-right: 1px solid rgba(231, 238, 252, 0.18);
                border-bottom: 1px solid rgba(231, 238, 252, 0.18);
                padding: 7px 10px;
                font-weight: 700;
            }}
            QPushButton {{
                background: rgba(255,255,255,.08);
                color: {PALETTE['text']};
                border: 1px solid rgba(231, 238, 252, 0.55);
                border-radius: 8px;
                padding: 6px 12px;
            }}
            QPushButton:hover {{
                border-color: {PALETTE['blue']};
                background: rgba(77,163,255,.15);
            }}
            QPushButton:disabled {{
                color: rgba(231, 238, 252, 0.45);
                border-color: rgba(231, 238, 252, 0.25);
                background: rgba(255,255,255,.03);
            }}
            """
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        summary = QtWidgets.QPlainTextEdit()
        summary.setObjectName("Panel")
        summary.setReadOnly(True)
        summary.setMaximumHeight(110)
        summary.setPlainText("\n".join(self._summary_lines(report)))
        layout.addWidget(summary)

        tabs = QtWidgets.QTabWidget()
        layout.addWidget(tabs, 1)

        self.metrics_frame = self._overview_frame(report)
        tabs.addTab(self._build_overview_tab(), "Overview")

        self.asset_frame = portfolio_report_frame(report).sort_values(
            by=["total_return_contribution", "avg_target_weight", "dataset_id"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        self.strategy_frame = portfolio_strategy_report_frame(report).sort_values(
            by=["total_return_contribution", "avg_target_weight", "strategy_name"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        tabs.addTab(self._build_strategies_tab(), "Strategies")
        tabs.addTab(self._build_assets_tab(), "Assets")

        self.drawdown_frame = portfolio_drawdown_frame(report)
        tabs.addTab(self._build_drawdown_tab(), "Drawdowns")

        buttons = QtWidgets.QHBoxLayout()
        export_btn = QtWidgets.QPushButton("Export CSV")
        export_btn.clicked.connect(self._export_all_csv)
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        buttons.addWidget(export_btn)
        buttons.addStretch(1)
        buttons.addWidget(close_btn)
        layout.addLayout(buttons)

    def _build_overview_tab(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        self.metrics_table = QtWidgets.QTableWidget(len(self.metrics_frame), 2)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.metrics_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.metrics_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.metrics_table.setAlternatingRowColors(True)
        self.metrics_table.verticalHeader().setVisible(False)
        self.metrics_table.setObjectName("Panel")
        for row_idx, row in self.metrics_frame.iterrows():
            self.metrics_table.setItem(row_idx, 0, QtWidgets.QTableWidgetItem(str(row["metric"])))
            self.metrics_table.setItem(row_idx, 1, QtWidgets.QTableWidgetItem(str(row["value"])))
        self._configure_report_table(self.metrics_table, stretch_last=True)
        self.metrics_table.horizontalHeader().resizeSection(0, 240)
        layout.addWidget(self.metrics_table)
        return panel

    def _build_assets_tab(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        self.table = QtWidgets.QTableWidget(len(self.asset_frame), len(self.asset_frame.columns))
        self.table.setHorizontalHeaderLabels(
            [
                "Asset",
                "Avg Weight",
                "Avg Long",
                "Avg Short",
                "Avg Target",
                "Avg |Track Err|",
                "Avg |Weight Chg|",
                "Final Weight",
                "Min Weight",
                "Peak Weight",
                "Peak Short",
                "Active %",
                "Trades",
                "Realized PnL",
                "Unrealized PnL",
                "Turnover $",
                "Turnover x",
                "Avg Return Contrib",
                "Total Return Contrib",
                "Contribution Share",
            ]
        )
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setObjectName("Panel")
        for row_idx, row in self.asset_frame.iterrows():
            values = [
                str(row["dataset_id"]),
                self._fmt(row["avg_weight"]),
                self._fmt(row["avg_long_weight"]),
                self._fmt_pct(row["avg_short_weight"]),
                self._fmt(row["avg_target_weight"]),
                self._fmt(row["avg_abs_tracking_error"]),
                self._fmt(row["avg_abs_weight_change"]),
                self._fmt(row["final_weight"]),
                self._fmt(row["min_weight"]),
                self._fmt(row["peak_weight"]),
                self._fmt_pct(row["peak_short_weight"]),
                self._fmt_pct(row["active_bar_fraction"]),
                str(int(row["trade_count"])),
                self._fmt_money(row["realized_pnl"]),
                self._fmt_money(row["unrealized_pnl"]),
                self._fmt_money(row["turnover_notional"]),
                self._fmt(row["turnover_ratio"], precision=3),
                self._fmt_pct(row["avg_return_contribution"]),
                self._fmt_pct(row["total_return_contribution"]),
                self._fmt_pct(row["contribution_share"]),
            ]
            for col_idx, value in enumerate(values):
                self.table.setItem(row_idx, col_idx, QtWidgets.QTableWidgetItem(value))
        self._configure_report_table(self.table)
        self.table.horizontalHeader().resizeSection(0, 220)
        layout.addWidget(self.table)
        return panel

    def _build_strategies_tab(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        self.strategy_table = QtWidgets.QTableWidget(len(self.strategy_frame), len(self.strategy_frame.columns))
        self.strategy_table.setHorizontalHeaderLabels(
            [
                "Strategy Block",
                "Strategy",
                "Budget",
                "Assets",
                "Avg Weight",
                "Avg Long",
                "Avg Short",
                "Avg Target",
                "Avg |Track Err|",
                "Avg |Weight Chg|",
                "Final Weight",
                "Min Weight",
                "Peak Weight",
                "Peak Short",
                "Active %",
                "Trades",
                "Realized PnL",
                "Turnover $",
                "Turnover x",
                "Avg Return Contrib",
                "Total Return Contrib",
                "Contribution Share",
            ]
        )
        self.strategy_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.strategy_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.strategy_table.setAlternatingRowColors(True)
        self.strategy_table.verticalHeader().setVisible(False)
        self.strategy_table.setObjectName("Panel")
        for row_idx, row in self.strategy_frame.iterrows():
            values = [
                str(row["strategy_block_id"]),
                str(row["strategy_name"]),
                self._fmt(row["budget_weight"]),
                str(int(row["asset_count"])),
                self._fmt(row["avg_weight"]),
                self._fmt(row["avg_long_weight"]),
                self._fmt_pct(row["avg_short_weight"]),
                self._fmt(row["avg_target_weight"]),
                self._fmt(row["avg_abs_tracking_error"]),
                self._fmt(row["avg_abs_weight_change"]),
                self._fmt(row["final_weight"]),
                self._fmt(row["min_weight"]),
                self._fmt(row["peak_weight"]),
                self._fmt_pct(row["peak_short_weight"]),
                self._fmt_pct(row["active_bar_fraction"]),
                str(int(row["trade_count"])),
                self._fmt_money(row["realized_pnl"]),
                self._fmt_money(row["turnover_notional"]),
                self._fmt(row["turnover_ratio"], precision=3),
                self._fmt_pct(row["avg_return_contribution"]),
                self._fmt_pct(row["total_return_contribution"]),
                self._fmt_pct(row["contribution_share"]),
            ]
            for col_idx, value in enumerate(values):
                self.strategy_table.setItem(row_idx, col_idx, QtWidgets.QTableWidgetItem(value))
        self._configure_report_table(self.strategy_table)
        self.strategy_table.horizontalHeader().resizeSection(0, 180)
        self.strategy_table.horizontalHeader().resizeSection(1, 180)
        layout.addWidget(self.strategy_table)
        return panel

    def _build_drawdown_tab(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        self.drawdown_table = QtWidgets.QTableWidget(len(self.drawdown_frame), len(self.drawdown_frame.columns))
        self.drawdown_table.setHorizontalHeaderLabels(
            [
                "Rank",
                "Peak",
                "Trough",
                "Recovery",
                "Depth",
                "Duration Bars",
                "Recovery Bars",
            ]
        )
        self.drawdown_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.drawdown_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.drawdown_table.setAlternatingRowColors(True)
        self.drawdown_table.verticalHeader().setVisible(False)
        self.drawdown_table.setObjectName("Panel")
        for row_idx, row in self.drawdown_frame.iterrows():
            values = [
                str(int(row["rank"])),
                self._fmt_timestamp(row["peak_time"]),
                self._fmt_timestamp(row["trough_time"]),
                self._fmt_timestamp(row["recovery_time"]),
                self._fmt_pct(row["depth"]),
                str(int(row["duration_bars"])),
                "—" if pd.isna(row["recovery_bars"]) else str(int(row["recovery_bars"])),
            ]
            for col_idx, value in enumerate(values):
                self.drawdown_table.setItem(row_idx, col_idx, QtWidgets.QTableWidgetItem(value))
        self._configure_report_table(self.drawdown_table)
        layout.addWidget(self.drawdown_table)
        return panel

    def _overview_frame(self, report) -> pd.DataFrame:
        rows = [
            ("Starting equity", self._fmt_money(report.starting_equity)),
            ("Ending equity", self._fmt_money(report.ending_equity)),
            ("Total return", self._fmt_pct(report.total_return)),
            ("CAGR", self._fmt_pct(report.cagr)),
            ("Annualized vol", self._fmt_pct(report.annualized_volatility)),
            ("Downside deviation", self._fmt_pct(report.downside_deviation)),
            ("Sharpe", self._fmt(report.sharpe)),
            ("Rolling Sharpe", self._fmt(report.rolling_sharpe)),
            ("Sortino", self._fmt(report.sortino)),
            ("Calmar", self._fmt(report.calmar)),
            ("Max drawdown", self._fmt_pct(report.max_drawdown)),
            ("Max DD duration bars", str(int(report.max_drawdown_duration_bars))),
            ("Underwater fraction", self._fmt_pct(report.underwater_fraction)),
            ("Best period return", self._fmt_pct(report.best_period_return)),
            ("Worst period return", self._fmt_pct(report.worst_period_return)),
            ("Avg cash weight", self._fmt_pct(report.avg_cash_weight)),
            ("Avg net exposure", self._fmt_pct(report.avg_net_exposure)),
            ("Avg long exposure", self._fmt_pct(report.avg_long_exposure)),
            ("Avg short exposure", self._fmt_pct(report.avg_short_exposure)),
            ("Avg gross exposure", self._fmt_pct(report.avg_gross_exposure)),
            ("Peak gross exposure", self._fmt_pct(report.peak_gross_exposure)),
            ("Peak short exposure", self._fmt_pct(report.peak_short_exposure)),
            ("Avg target gross exposure", self._fmt_pct(report.avg_target_gross_exposure)),
            ("Peak target gross exposure", self._fmt_pct(report.peak_target_gross_exposure)),
            ("Avg active assets", self._fmt(report.avg_active_assets, precision=2)),
            ("Peak active assets", str(int(report.peak_active_assets))),
            ("Avg concentration HHI", self._fmt(report.avg_concentration_hhi, precision=4)),
            ("Peak concentration HHI", self._fmt(report.peak_concentration_hhi, precision=4)),
            ("Peak single-name weight", self._fmt_pct(report.peak_single_name_weight)),
            ("Trades", str(int(report.trade_count))),
            ("Total turnover", f"{self._fmt_money(report.total_turnover_notional)} ({report.total_turnover_ratio:.3f}x)"),
        ]
        return pd.DataFrame(rows, columns=["metric", "value"])

    @staticmethod
    def _configure_report_table(table: QtWidgets.QTableWidget, *, stretch_last: bool = False) -> None:
        header = table.horizontalHeader()
        header.setStretchLastSection(stretch_last)
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)
        table.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        table.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollMode.ScrollPerPixel)
        table.setWordWrap(False)
        table.setTextElideMode(QtCore.Qt.TextElideMode.ElideRight)
        table.resizeColumnsToContents()
        for col in range(table.columnCount()):
            current = header.sectionSize(col)
            header.resizeSection(col, max(110, min(current, 220)))

    @staticmethod
    def _fmt(value: float | int, precision: int = 4) -> str:
        try:
            numeric = float(value)
        except Exception:
            return "—"
        if np.isnan(numeric):
            return "—"
        return f"{numeric:.{precision}f}"

    @staticmethod
    def _fmt_pct(value: float | int) -> str:
        try:
            numeric = float(value)
        except Exception:
            return "—"
        if np.isnan(numeric):
            return "—"
        return f"{numeric * 100:.2f}%"

    @staticmethod
    def _fmt_money(value: float | int) -> str:
        try:
            numeric = float(value)
        except Exception:
            return "—"
        if np.isnan(numeric):
            return "—"
        return f"${numeric:,.2f}"

    @classmethod
    def _summary_lines(cls, report) -> list[str]:
        return [
            f"Starting equity: {cls._fmt_money(report.starting_equity)}",
            f"Ending equity: {cls._fmt_money(report.ending_equity)}",
            f"Total return: {cls._fmt_pct(report.total_return)}",
            f"CAGR: {cls._fmt_pct(report.cagr)}",
            f"Max drawdown: {cls._fmt_pct(report.max_drawdown)}",
            f"Sharpe: {report.sharpe:.4f}",
            f"Sortino: {report.sortino:.4f}",
            f"Annualized vol: {cls._fmt_pct(report.annualized_volatility)}",
            f"Avg cash weight: {cls._fmt_pct(report.avg_cash_weight)}",
            f"Avg net exposure: {cls._fmt_pct(report.avg_net_exposure)}",
            f"Avg long exposure: {cls._fmt_pct(report.avg_long_exposure)}",
            f"Avg short exposure: {cls._fmt_pct(report.avg_short_exposure)}",
            f"Avg gross exposure: {cls._fmt_pct(report.avg_gross_exposure)}",
            f"Peak gross exposure: {cls._fmt_pct(report.peak_gross_exposure)}",
            f"Peak short exposure: {cls._fmt_pct(report.peak_short_exposure)}",
            f"Peak single-name weight: {cls._fmt_pct(report.peak_single_name_weight)}",
            f"Trades: {report.trade_count}",
            f"Total turnover: {cls._fmt_money(report.total_turnover_notional)} ({report.total_turnover_ratio:.3f}x)",
        ]

    @classmethod
    def _summary_text(cls, report) -> str:
        return " | ".join(cls._summary_lines(report))
        return " | ".join(lines)

    @staticmethod
    def _fmt_timestamp(value) -> str:
        ts = pd.to_datetime(value, errors="coerce")
        if pd.isna(ts):
            return "—"
        if getattr(ts, "tzinfo", None) is not None:
            ts = ts.tz_convert("America/New_York").tz_localize(None)
        return ts.strftime("%Y-%m-%d %H:%M")

    def _export_all_csv(self) -> None:
        if self.asset_frame.empty and self.strategy_frame.empty and self.drawdown_frame.empty and self.metrics_frame.empty:
            QtWidgets.QMessageBox.information(self, "No data", "There is nothing to export yet.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Portfolio Report CSV",
            f"portfolio_report_{self.run.run_id[:10]}_assets.csv",
            "CSV Files (*.csv)",
        )
        if not path:
            return
        try:
            base = Path(path)
            self.asset_frame.to_csv(base, index=False)
            self.strategy_frame.to_csv(base.with_name(f"{base.stem}_strategies.csv"), index=False)
            self.drawdown_frame.to_csv(base.with_name(f"{base.stem}_drawdowns.csv"), index=False)
            self.metrics_frame.to_csv(base.with_name(f"{base.stem}_overview.csv"), index=False)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Export error", str(exc))

    def _export_csv(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            QtWidgets.QMessageBox.information(self, "No data", "There is nothing to export yet.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Portfolio Report CSV",
            f"portfolio_report_{self.run.run_id[:10]}.csv",
            "CSV Files (*.csv)",
        )
        if not path:
            return
        try:
            frame.to_csv(path, index=False)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Export error", str(exc))


class PortfolioRunChartDialog(DashboardDialog):
    _LINE_COLORS = ("#4da3ff", "#27d07d", "#ffcc66", "#ff6b6b", "#8b7bff", "#59c3c3")

    def __init__(self, run: RunRow, portfolio_result, report, parent=None) -> None:
        super().__init__(parent)
        self.run = run
        self.portfolio_result = portfolio_result
        self.report = report
        self.chart_data = build_portfolio_chart_data(portfolio_result, max_assets=4)
        self.setWindowTitle(f"Portfolio Chart | {run.run_id[:10]}")
        self.resize(1400, 940)

        layout = QtWidgets.QVBoxLayout(self)
        summary = QtWidgets.QLabel(
            f"Strategy: {run.strategy} | Timeframe: {run.timeframe} | Dataset: {run.dataset_id}\n"
            f"Requested: {(run.requested_execution_mode or '—').title()} | "
            f"Resolved: {(run.resolved_execution_mode or '—').title()} | "
            f"Engine: {run.engine_impl or '—'} | Duration: {_format_duration(run.run_started_at, run.run_finished_at)}\n"
            f"{PortfolioReportDialog._summary_text(report)}"
        )
        summary.setObjectName("Sub")
        summary.setWordWrap(True)
        summary.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(summary)

        note = QtWidgets.QLabel(
            "Built-in portfolio fallback viewer. Magellan still provides the richer multi-pane portfolio chart when available."
        )
        note.setObjectName("Sub")
        note.setWordWrap(True)
        layout.addWidget(note)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        layout.addWidget(splitter, 1)

        chart_panel = QtWidgets.QWidget()
        chart_layout = QtWidgets.QVBoxLayout(chart_panel)
        chart_layout.setContentsMargins(0, 0, 0, 0)
        self.figure = Figure(figsize=(14, 9), facecolor=PALETTE["panel"])
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.toolbar.setMovable(False)
        chart_layout.addWidget(self.toolbar)
        chart_layout.addWidget(self.canvas, 1)
        splitter.addWidget(chart_panel)

        table_panel = QtWidgets.QWidget()
        table_layout = QtWidgets.QVBoxLayout(table_panel)
        table_layout.setContentsMargins(0, 0, 0, 0)
        attribution = QtWidgets.QLabel("Asset Attribution")
        attribution.setObjectName("Sub")
        table_layout.addWidget(attribution)
        self.report_frame = portfolio_report_frame(report).sort_values(
            by=["avg_target_weight", "avg_weight", "dataset_id"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        self.table = QtWidgets.QTableWidget(len(self.report_frame), len(self.report_frame.columns))
        self.table.setHorizontalHeaderLabels(
            [
                "Asset",
                "Avg Weight",
                "Avg Target",
                "Avg |Track Err|",
                "Final Weight",
                "Peak Weight",
                "Active %",
                "Trades",
                "Realized PnL",
                "Turnover $",
                "Turnover x",
            ]
        )
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setObjectName("Panel")
        for row_idx, row in self.report_frame.iterrows():
            values = [
                str(row["dataset_id"]),
                PortfolioReportDialog._fmt(row["avg_weight"]),
                PortfolioReportDialog._fmt(row["avg_target_weight"]),
                PortfolioReportDialog._fmt(row["avg_abs_tracking_error"]),
                PortfolioReportDialog._fmt(row["final_weight"]),
                PortfolioReportDialog._fmt(row["peak_weight"]),
                PortfolioReportDialog._fmt_pct(row["active_bar_fraction"]),
                str(int(row["trade_count"])),
                PortfolioReportDialog._fmt_money(row["realized_pnl"]),
                PortfolioReportDialog._fmt_money(row["turnover_notional"]),
                PortfolioReportDialog._fmt(row["turnover_ratio"], precision=3),
            ]
            for col_idx, value in enumerate(values):
                self.table.setItem(row_idx, col_idx, QtWidgets.QTableWidgetItem(value))
        table_layout.addWidget(self.table, 1)
        splitter.addWidget(table_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        buttons = QtWidgets.QHBoxLayout()
        export_btn = QtWidgets.QPushButton("Export CSV")
        export_btn.clicked.connect(self._export_csv)
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        buttons.addWidget(export_btn)
        buttons.addStretch(1)
        buttons.addWidget(close_btn)
        layout.addLayout(buttons)

        self._draw_charts()

    @staticmethod
    def _display_index(index_like) -> pd.DatetimeIndex:
        idx = pd.DatetimeIndex(pd.to_datetime(index_like, utc=True, errors="coerce"))
        return idx.tz_convert("America/New_York").tz_localize(None)

    def _draw_charts(self) -> None:
        display_index = self._display_index(self.chart_data.equity_curve.index)
        gs = self.figure.add_gridspec(3, 1, height_ratios=[2.1, 1.0, 1.4], hspace=0.06)
        ax_equity = self.figure.add_subplot(gs[0, 0])
        ax_exposure = self.figure.add_subplot(gs[1, 0], sharex=ax_equity)
        ax_weights = self.figure.add_subplot(gs[2, 0], sharex=ax_equity)
        for ax in (ax_equity, ax_exposure, ax_weights):
            ax.set_facecolor(PALETTE["bg"])
            ax.grid(alpha=0.12, color="0.3")

        equity_values = self.chart_data.equity_curve.to_numpy(dtype=float)
        ax_equity.plot(display_index, equity_values, color=PALETTE["blue"], linewidth=1.6, label="Equity")
        ax_equity.axhline(
            float(self.report.starting_equity),
            color=PALETTE["muted"],
            linewidth=1.0,
            linestyle="--",
            alpha=0.8,
            label="Starting Equity",
        )
        if not self.chart_data.trades.empty:
            trades = self.chart_data.trades.copy()
            display_trade_index = self._display_index(trades["timestamp"])
            buys = trades[trades["side"].str.lower() == "buy"]
            sells = trades[trades["side"].str.lower() == "sell"]
            if not buys.empty:
                buy_pos = display_trade_index[buys.index]
                ax_equity.scatter(
                    buy_pos,
                    buys["equity_after"].to_numpy(dtype=float),
                    marker="^",
                    color=PALETTE["green"],
                    s=28,
                    alpha=0.8,
                    label="Buy Fill",
                    zorder=5,
                )
            if not sells.empty:
                sell_pos = display_trade_index[sells.index]
                ax_equity.scatter(
                    sell_pos,
                    sells["equity_after"].to_numpy(dtype=float),
                    marker="v",
                    color="#b38cff",
                    s=28,
                    alpha=0.8,
                    label="Sell Fill",
                    zorder=5,
                )
        ax_equity.set_title("Portfolio Equity")
        ax_equity.legend(loc="upper left", ncol=3)

        ax_exposure.plot(
            display_index,
            self.chart_data.cash_weight.to_numpy(dtype=float),
            color=PALETTE["muted"],
            linewidth=1.2,
            label="Cash Weight",
        )
        ax_exposure.plot(
            display_index,
            self.chart_data.gross_exposure.to_numpy(dtype=float),
            color=PALETTE["green"],
            linewidth=1.4,
            label="Gross Exposure",
        )
        ax_exposure.plot(
            display_index,
            self.chart_data.target_gross_exposure.to_numpy(dtype=float),
            color=PALETTE["amber"],
            linewidth=1.1,
            linestyle="--",
            label="Target Gross Exposure",
        )
        ax_exposure.set_title("Cash And Exposure")
        ax_exposure.legend(loc="upper left", ncol=3)

        if self.chart_data.top_assets:
            for idx, dataset_id in enumerate(self.chart_data.top_assets):
                color = self._LINE_COLORS[idx % len(self._LINE_COLORS)]
                actual = pd.to_numeric(
                    self.chart_data.asset_weights.get(dataset_id, pd.Series(0.0, index=self.chart_data.equity_curve.index)),
                    errors="coerce",
                ).fillna(0.0)
                target = pd.to_numeric(
                    self.chart_data.target_weights.get(dataset_id, pd.Series(0.0, index=self.chart_data.equity_curve.index)),
                    errors="coerce",
                ).fillna(0.0)
                ax_weights.plot(
                    display_index,
                    actual.to_numpy(dtype=float),
                    color=color,
                    linewidth=1.5,
                    label=f"{dataset_id} actual",
                )
                ax_weights.plot(
                    display_index,
                    target.to_numpy(dtype=float),
                    color=color,
                    linewidth=1.0,
                    linestyle="--",
                    alpha=0.9,
                    label=f"{dataset_id} target",
                )
            ax_weights.legend(loc="upper left", ncol=2)
        else:
            ax_weights.text(
                0.01,
                0.85,
                "No asset weight history is available for this portfolio run.",
                transform=ax_weights.transAxes,
                color=PALETTE["muted"],
                fontsize=10,
            )
        ax_weights.set_title(f"Asset Weights (Top {max(1, len(self.chart_data.top_assets))})")

        self.figure.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.07)
        plt = __import__("matplotlib.pyplot", fromlist=["setp"])
        plt.setp(ax_equity.get_xticklabels(), visible=False)
        plt.setp(ax_exposure.get_xticklabels(), visible=False)
        self.figure.autofmt_xdate(rotation=20)
        self.canvas.draw_idle()

    def _export_csv(self) -> None:
        if self.report_frame.empty:
            QtWidgets.QMessageBox.information(self, "No data", "There is nothing to export yet.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Portfolio Attribution CSV",
            f"portfolio_chart_{self.run.run_id[:10]}.csv",
            "CSV Files (*.csv)",
        )
        if not path:
            return
        try:
            self.report_frame.to_csv(path, index=False)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Export error", str(exc))


class TradesLogDialog(DashboardDialog):
    def __init__(self, run_id: str, trades_df: pd.DataFrame, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self.setWindowTitle(f"Trades Log {run_id[:12]}…")
        self.resize(900, 600)
        self.setStyleSheet(
            f"""
            QDialog#Panel {{
                background: {PALETTE['panel']};
                color: {PALETTE['text']};
            }}
            QLabel#Sub {{
                color: {PALETTE['muted']};
            }}
            QTableView#Panel {{
                background: {PALETTE['panel2']};
                color: {PALETTE['text']};
                border: 1px solid rgba(231, 238, 252, 0.55);
                border-radius: 10px;
                gridline-color: rgba(154, 176, 208, 0.22);
                alternate-background-color: rgba(255,255,255,.04);
                selection-background-color: rgba(77,163,255,.28);
                selection-color: {PALETTE['text']};
            }}
            QHeaderView::section {{
                background: rgba(16, 26, 46, 0.98);
                color: {PALETTE['muted']};
                font-size: 11px;
                font-weight: 700;
                border: none;
                border-right: 1px solid rgba(154, 176, 208, 0.18);
                border-bottom: 1px solid rgba(154, 176, 208, 0.28);
                padding: 6px 8px;
            }}
            QPushButton {{
                background: rgba(255,255,255,.08);
                color: {PALETTE['text']};
                border: 1px solid rgba(231, 238, 252, 0.55);
                border-radius: 8px;
                padding: 6px 10px;
            }}
            QPushButton:hover {{
                border-color: {PALETTE['blue']};
                background: rgba(77,163,255,.15);
            }}
            """
        )
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)
        if trades_df is None or trades_df.empty:
            msg = QtWidgets.QLabel("No trades available for this run.")
            msg.setObjectName("Sub")
            layout.addWidget(msg)
            close_btn = QtWidgets.QPushButton("Close")
            close_btn.clicked.connect(self.accept)
            layout.addWidget(close_btn, 0, QtCore.Qt.AlignmentFlag.AlignRight)
            return
        title = QtWidgets.QLabel(f"Trades Log {run_id[:12]}…")
        title.setObjectName("Title")
        layout.addWidget(title)
        model = TradesTableModel(trades_df)
        table = QtWidgets.QTableView()
        table.setModel(model)
        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        table.horizontalHeader().setStretchLastSection(True)
        table.verticalHeader().setVisible(False)
        table.setObjectName("Panel")
        layout.addWidget(table)
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn, 0, QtCore.Qt.AlignmentFlag.AlignRight)


class TradesTableModel(QtCore.QAbstractTableModel):
    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__()
        self._df = df.copy()
        self._headers = list(df.columns)

    def rowCount(self, parent=None):
        return len(self._df)

    def columnCount(self, parent=None):
        return len(self._headers)

    def data(self, index, role=None):
        if not index.isValid():
            return None
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            value = self._df.iat[index.row(), index.column()]
            if isinstance(value, float):
                return f"{value:.6f}"
            return str(value)
        return None

    def headerData(self, section, orientation, role=None):
        if role != QtCore.Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == QtCore.Qt.Orientation.Horizontal:
            return self._headers[section]
        return str(section + 1)


class TickerPickerDialog(DashboardDialog):
    def __init__(self, symbols: list[str], preselected: set[str], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Choose Tickers")
        self.resize(520, 640)
        self.selected: list[str] = []
        self.setObjectName("Panel")
        self.setStyleSheet(
            f"""
            QDialog#Panel {{
                background: {PALETTE['panel']};
                color: {PALETTE['text']};
            }}
            QLabel {{
                color: {PALETTE['text']};
            }}
            QListWidget {{
                background: {PALETTE['panel2']};
                border: 1px solid {PALETTE['border']};
                border-radius: 8px;
                padding: 6px;
            }}
            QListWidget::item {{
                padding: 6px 8px;
                color: {PALETTE['text']};
            }}
            QListWidget::item:selected {{
                background: rgba(77, 163, 255, 0.2);
            }}
            QPushButton {{
                background: {PALETTE['panel2']};
                color: {PALETTE['text']};
                border: 1px solid {PALETTE['border']};
                border-radius: 8px;
                padding: 6px 10px;
            }}
            QPushButton:hover {{
                border-color: {PALETTE['blue']};
            }}
            """
        )

        layout = QtWidgets.QVBoxLayout(self)
        info = QtWidgets.QLabel("Select tickers to schedule for data acquisition.")
        info.setObjectName("Sub")
        layout.addWidget(info)

        self.search_edit = QtWidgets.QLineEdit()
        self.search_edit.setPlaceholderText("Filter symbols...")
        self.search_edit.textChanged.connect(self._apply_filter)
        layout.addWidget(self.search_edit)

        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        layout.addWidget(self.list_widget, 1)

        btn_row = QtWidgets.QHBoxLayout()
        select_all = QtWidgets.QPushButton("Select All")
        clear_all = QtWidgets.QPushButton("Clear")
        ok_btn = QtWidgets.QPushButton("OK")
        cancel_btn = QtWidgets.QPushButton("Cancel")
        select_all.clicked.connect(self._select_all)
        clear_all.clicked.connect(self._clear_all)
        ok_btn.clicked.connect(self._accept)
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(select_all)
        btn_row.addWidget(clear_all)
        btn_row.addStretch(1)
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        self._all_symbols = symbols
        self._populate(symbols, preselected)

    def _populate(self, symbols: list[str], preselected: set[str]) -> None:
        self.list_widget.clear()
        for sym in symbols:
            item = QtWidgets.QListWidgetItem(sym)
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.CheckState.Checked if sym in preselected else QtCore.Qt.CheckState.Unchecked)
            self.list_widget.addItem(item)

    def _apply_filter(self, text: str) -> None:
        needle = text.strip().upper()
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            item.setHidden(bool(needle) and needle not in item.text())

    def _select_all(self) -> None:
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(QtCore.Qt.CheckState.Checked)

    def _clear_all(self) -> None:
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(QtCore.Qt.CheckState.Unchecked)

    def _accept(self) -> None:
        selected = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == QtCore.Qt.CheckState.Checked:
                selected.append(item.text())
        self.selected = selected
        self.accept()


class RunChartDialog(DashboardDialog):
    def __init__(self, run: RunRow, catalog_path: Path, bt_settings: Dict[str, float | bool | dict | str], parent=None) -> None:
        super().__init__(parent)
        self.catalog_path = Path(catalog_path)
        self.bt_settings = bt_settings
        self.setWindowTitle(f"Run {run.run_id[:12]}… | {run.strategy}")
        self.resize(1400, 900)

        layout = QtWidgets.QVBoxLayout(self)
        summary = QtWidgets.QLabel(
            f"Strategy: {run.strategy} | Timeframe: {run.timeframe} | Dataset: {run.dataset_id}\n"
            f"Requested: {(run.requested_execution_mode or '—').title()} | "
            f"Resolved: {(run.resolved_execution_mode or '—').title()} | "
            f"Engine: {run.engine_impl or '—'} | Duration: {_format_duration(run.run_started_at, run.run_finished_at)}\n"
            f"Params: {run.params}"
        )
        summary.setObjectName("Sub")
        summary.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(summary)

        try:
            fig = Figure(figsize=(14, 9), facecolor=PALETTE["panel"])
            canvas = FigureCanvasQTAgg(fig)
            toolbar = NavigationToolbar2QT(canvas, self)
            toolbar.setMovable(False)
            layout.addWidget(toolbar)
            gs = fig.add_gridspec(3, 1, height_ratios=[3, 0.8, 1], hspace=0.04)
            ax_price = fig.add_subplot(gs[0, 0])
            ax_atr = fig.add_subplot(gs[1, 0], sharex=ax_price)
            ax_equity = fig.add_subplot(gs[2, 0], sharex=ax_price)
            for ax in (ax_price, ax_atr, ax_equity):
                ax.set_facecolor(PALETTE["bg"])

            bars = self._load_bars(run)
            if bars is None or bars.empty:
                raise ValueError(self._missing_run_data_message_static(run))

            # Compute indicators for plotting.
            indicators = self._compute_indicators(run, bars)

            # Load stored trades; do not rerun.
            rc = ResultCatalog(self.catalog_path)
            trades = rc.load_trades(run.run_id) or []
            trades_df = pd.DataFrame(trades)
            if not trades_df.empty:
                ts = pd.to_datetime(trades_df["timestamp"], utc=True, errors="coerce")
                ts = ts.dt.tz_convert("America/New_York")
                trades_df["ts"] = ts.dt.tz_localize(None)
                trades_df["price"] = trades_df["price"].astype(float)
                trades_df["seq"] = trades_df.get("seq", trades_df.index + 1)
                trades_df = trades_df[(trades_df["ts"] >= bars.index.min()) & (trades_df["ts"] <= bars.index.max())]
            equity = None

            atr_series = indicators.pop("ATR", None) if indicators else None

            ohlc = bars[["open", "high", "low", "close"]].copy()
            x = np.arange(len(ohlc), dtype=float)
            ohlc_vals = np.column_stack(
                [x, ohlc["open"].to_numpy(), ohlc["high"].to_numpy(), ohlc["low"].to_numpy(), ohlc["close"].to_numpy()]
            )
            candlestick_ohlc(ax_price, ohlc_vals, width=0.6, colorup="#27d07d", colordown="#b38cff", alpha=0.7)
            for name, series in indicators.items():
                aligned = series.reindex(bars.index)
                ax_price.plot(x, aligned.to_numpy(), label=name, linewidth=1.0, zorder=1)

            pos_map = pd.Series(np.arange(len(bars), dtype=float), index=bars.index)
            if not trades_df.empty:
                trade_pos = pos_map.reindex(trades_df["ts"], method="nearest")
                trades_df = trades_df.assign(x=trade_pos.values, _valid=trade_pos.notna().to_numpy())
                trades_df = trades_df.loc[trades_df["_valid"]].drop(columns=["_valid"])
            buys_df = trades_df[trades_df["side"] == "buy"] if not trades_df.empty else pd.DataFrame()
            sells_df = trades_df[trades_df["side"] == "sell"] if not trades_df.empty else pd.DataFrame()
            if not buys_df.empty:
                ax_price.scatter(
                    buys_df["x"],
                    buys_df["price"],
                    marker="^",
                    color="#27d07d",
                    label="Buy",
                    zorder=6,
                    s=32,
                )
            if not sells_df.empty:
                ax_price.scatter(
                    sells_df["x"],
                    sells_df["price"],
                    marker="v",
                    color="#b38cff",
                    label="Sell",
                    zorder=6,
                    s=32,
                )
            # Label trades for clarity
            if not trades_df.empty:
                for _, row in trades_df.iterrows():
                    ax_price.annotate(
                        str(row.get("seq", 0)),
                        (row["x"], row["price"]),
                        xytext=(0, 6),
                        textcoords="offset points",
                        ha="center",
                        fontsize=7,
                        color=PALETTE["text"],
                        bbox=dict(boxstyle="round,pad=0.2", fc=PALETTE["panel2"], ec=PALETTE["border"], alpha=0.7),
                    )
            ax_price.legend(loc="upper left", ncol=3)
            ax_price.set_title("Price & Indicators (pan/zoom enabled)")
            ax_price.grid(alpha=0.12, color="0.3")

            if atr_series is not None:
                aligned_atr = atr_series.reindex(bars.index)
                ax_atr.plot(x, aligned_atr.to_numpy(), color="#a28bff", label="ATR", linewidth=1.0)
                ax_atr.legend(loc="upper left")
            ax_atr.set_title("ATR")
            ax_atr.grid(alpha=0.12, color="0.3")

            if equity is not None:
                ax_equity.plot(x, equity.values, color="#4da3ff", label="Equity", linewidth=1.3)
                ax_equity.legend(loc="upper left")
            ax_equity.set_title("Equity Curve")
            ax_equity.grid(alpha=0.12, color="0.3")
            ax_equity.set_xlim(-1, len(bars))
            ax_equity.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
            ax_equity.xaxis.set_major_formatter(
                FuncFormatter(
                    lambda val, _pos: bars.index[int(round(val))].strftime("%Y-%m-%d %H:%M")
                    if 0 <= int(round(val)) < len(bars)
                    else ""
                )
            )
            plt = __import__("matplotlib.pyplot", fromlist=["setp"])
            plt.setp(ax_price.get_xticklabels(), visible=False)
            plt.setp(ax_atr.get_xticklabels(), visible=False)
            fig.subplots_adjust(left=0.05, right=0.98, top=0.94, bottom=0.07)
            canvas.mpl_connect("scroll_event", lambda evt: self._on_scroll(evt, ax_price, ax_atr, ax_equity, canvas))

            layout.addWidget(canvas)
        except Exception as exc:
            print("Matplotlib chart error:", exc)
            msg = QtWidgets.QLabel(f"Failed to render chart: {exc}")
            msg.setWordWrap(True)
            msg.setStyleSheet("color: red;")
            layout.addWidget(msg)

    def _strategy_class(self, name: str):
        return self._strategy_class_static(name)

    @staticmethod
    def _strategy_class_static(name: str):
        mapping = {
            "SMACrossStrategy": SMACrossStrategy,
            "InverseTurtleStrategy": InverseTurtleStrategy,
            "ZScoreMeanReversionStrategy": ZScoreMeanReversionStrategy,
        }
        return mapping.get(name, SMACrossStrategy)

    def _load_bars(self, run: RunRow) -> pd.DataFrame | None:
        return self._load_bars_static(run)

    @staticmethod
    def _missing_run_data_message_static(run: RunRow) -> str:
        store = DuckDBStore()
        dataset_path = store.dataset_path(run.dataset_id)
        dataset_text = str(dataset_path.resolve()) if dataset_path.exists() else str(dataset_path)
        if not dataset_path.exists():
            detail = (
                "The parquet dataset file for this run is missing from the current project data store."
            )
        else:
            detail = (
                "The dataset file exists, but no bars could be reloaded for this run's stored time window."
            )
        return (
            "This saved run cannot be charted from the current data store.\n\n"
            f"{detail}\n\n"
            f"Dataset: {run.dataset_id}\n"
            f"Timeframe: {run.timeframe}\n"
            f"Start: {run.start}\n"
            f"End: {run.end}\n"
            f"Expected parquet file: {dataset_text}\n\n"
            "This often happens when the historical data files were not copied over from another machine."
        )

    @staticmethod
    def _load_bars_utc_static(run: RunRow) -> pd.DataFrame | None:
        try:
            duck = DuckDBStore()
            start_ts = pd.to_datetime(run.start, utc=True, errors="coerce")
            end_ts = pd.to_datetime(run.end, utc=True, errors="coerce")
            if start_ts is pd.NaT or end_ts is pd.NaT:
                return None
            tf = run.timeframe
            norm_tf = BacktestEngine._normalize_freq(tf)
            if norm_tf in ("1T", "1min"):
                bars = duck.load_range(run.dataset_id, start_ts, end_ts)
            else:
                try:
                    offset = pd.tseries.frequencies.to_offset(norm_tf)
                    start_buffer = start_ts - offset
                except Exception:
                    start_buffer = start_ts
                base = duck.load_range(run.dataset_id, start_buffer, end_ts)
                if base.empty:
                    return None
                bars = (
                    base.resample(norm_tf, label="right", closed="right")
                    .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
                    .dropna()
                )
                bars = bars[(bars.index >= start_ts) & (bars.index <= end_ts)]
            if bars.index.tz is None:
                bars.index = bars.index.tz_localize("UTC")
            else:
                bars.index = bars.index.tz_convert("UTC")
            return bars
        except Exception:
            return None

    @staticmethod
    def _load_bars_static(run: RunRow) -> pd.DataFrame | None:
        bars = RunChartDialog._load_bars_utc_static(run)
        if bars is None or bars.empty:
            return bars
        bars = bars.copy()
        bars.index = bars.index.tz_convert("America/New_York").tz_localize(None)
        return bars

    def _rerun(self, run: RunRow, bars: pd.DataFrame):
        return self._rerun_static(run, bars, self.bt_settings)

    @staticmethod
    def _rerun_static(run: RunRow, bars: pd.DataFrame, bt_settings: Dict[str, float | bool | dict | str]):
        params = json.loads(run.params) if isinstance(run.params, str) else run.params
        cls = RunChartDialog._strategy_class_static(run.strategy)
        config = BacktestConfig(
            timeframe=run.timeframe,
            starting_cash=float(bt_settings.get("starting_cash", 100_000)),
            fee_rate=float(bt_settings.get("fee_rate", 0.0002)),
            fee_schedule=bt_settings.get("fee_schedule", {"buy": 0.0003, "sell": 0.0005}),
            slippage=float(bt_settings.get("slippage", 0.0002)),
            slippage_schedule=bt_settings.get("slippage_schedule", {"buy": 0.0003, "sell": 0.0001}),
            borrow_rate=float(bt_settings.get("borrow_rate", 0.0)),
            fill_ratio=float(bt_settings.get("fill_ratio", 1.0)),
            fill_on_close=bool(bt_settings.get("fill_on_close", False)),
            recalc_on_fill=bool(bt_settings.get("recalc_on_fill", True)),
            allow_short=bool(bt_settings.get("allow_short", True)),
            use_cache=bool(bt_settings.get("use_cache", False)),
            intrabar_sim=False,
            prevent_scale_in=bool(bt_settings.get("prevent_scale_in", True)),
            one_order_per_signal=bool(bt_settings.get("one_order_per_signal", True)),
            base_execution=True if run.timeframe != "1 minutes" else False,
            base_timeframe="1 minutes",
        )
        orchestrator = ExecutionOrchestrator()
        return orchestrator.execute(
            ExecutionRequest(
                data=bars,
                dataset_id=run.dataset_id,
                strategy_cls=cls,
                strategy_params=params,
                catalog=None,
                config=config,
                base_data=bars if run.timeframe == "1 minutes" else None,
                requested_execution_mode=str(bt_settings.get("execution_mode", ExecutionMode.REFERENCE.value)),
            )
        )

    def _compute_indicators(self, run: RunRow, bars: pd.DataFrame) -> Dict[str, pd.Series]:
        return self._compute_indicators_static(run, bars)

    @staticmethod
    def _compute_indicators_static(run: RunRow, bars: pd.DataFrame) -> Dict[str, pd.Series]:
        params = json.loads(run.params) if isinstance(run.params, str) else run.params
        name = run.strategy
        out: Dict[str, pd.Series] = {}
        if name == "SMACrossStrategy":
            fast = int(params.get("fast", 10))
            slow = int(params.get("slow", 30))
            out["SMA Fast"] = bars["close"].rolling(fast).mean()
            out["SMA Slow"] = bars["close"].rolling(slow).mean()
        elif name == "ZScoreMeanReversionStrategy":
            features = compute_zscore_mean_reversion_features(bars, params)
            out["Half-Life Mean"] = features["half_life_mean"]
            out["Z-Score"] = features["z_score"]
        elif name == "InverseTurtleStrategy":
            entry_len = int(params.get("entry_len", 20))
            exit_len = int(params.get("exit_len", 10))
            atr_len = int(params.get("atr_len", 14))
            use_prev = bool(params.get("use_prev_channels", True))
            upper = bars["high"].rolling(entry_len).max()
            lower = bars["low"].rolling(entry_len).min()
            exit_upper = bars["high"].rolling(exit_len).max()
            exit_lower = bars["low"].rolling(exit_len).min()
            tr = pd.concat(
                [
                    (bars["high"] - bars["low"]).abs(),
                    (bars["high"] - bars["close"].shift(1)).abs(),
                    (bars["low"] - bars["close"].shift(1)).abs(),
                ],
                axis=1,
            ).max(axis=1)
            atr = tr.rolling(atr_len).mean()
            if use_prev:
                upper = upper.shift(1)
                lower = lower.shift(1)
                exit_upper = exit_upper.shift(1)
                exit_lower = exit_lower.shift(1)
                atr = atr.shift(1)
            out["Upper"] = upper
            out["Lower"] = lower
            out["Exit Upper"] = exit_upper
            out["Exit Lower"] = exit_lower
            out["ATR"] = atr
        return out

    @staticmethod
    def _build_snapshot_series_static(run: RunRow, bars: pd.DataFrame) -> tuple[Dict[str, pd.Series], Dict[str, pd.Series], Dict[str, dict]]:
        color_map = {
            "SMA Fast": "#4da3ff",
            "SMA Slow": "#ffd166",
            "Half-Life Mean": "#4da3ff",
            "Z-Score": "#ffcc66",
            "Upper": "#27d07d",
            "Lower": "#ff6b6b",
            "Exit Upper": "#7ee787",
            "Exit Lower": "#a28bff",
            "ATR": "#a28bff",
        }
        indicators = RunChartDialog._compute_indicators_static(run, bars)
        overlays: Dict[str, pd.Series] = {}
        panes: Dict[str, pd.Series] = {}
        styles: Dict[str, dict] = {}
        for name, series in indicators.items():
            aligned = pd.to_numeric(series.reindex(bars.index), errors="coerce")
            if name in {"ATR", "Z-Score"}:
                panes[name] = aligned
            else:
                overlays[name] = aligned
            styles[name] = {"color": color_map.get(name, "#4da3ff"), "line_width": 1.0}
        return overlays, panes, styles

    def _on_scroll(self, event, ax_price, ax_atr, ax_equity, canvas) -> None:
        # Zoom on scroll around mouse x (time) and y (price) for price axis.
        if event.inaxes is None:
            return
        scale = 0.8 if event.button == "up" else 1.25
        def _zoom_axis(ax):
            xlim = ax.get_xlim()
            x_range = (xlim[1] - xlim[0]) * scale
            x_center = event.xdata if event.xdata is not None else (xlim[0] + xlim[1]) / 2
            ax.set_xlim(x_center - x_range / 2, x_center + x_range / 2)
            if event.ydata is not None and ax is ax_price:
                ylim = ax.get_ylim()
                y_range = (ylim[1] - ylim[0]) * scale
                y_center = event.ydata
                ax.set_ylim(y_center - y_range / 2, y_center + y_range / 2)
        _zoom_axis(ax_price)
        _zoom_axis(ax_atr)
        _zoom_axis(ax_equity)
        canvas.draw_idle()


def main():
    import sys

    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(load_stylesheet())
    icon_path = Path("assets/app_icon.png")
    if icon_path.exists():
        app.setWindowIcon(QtGui.QIcon(str(icon_path)))
    db_path = Path("backtests.sqlite")
    win = DashboardWindow(db_path)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
import subprocess
