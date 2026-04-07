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
    PortfolioExecutionAsset,
    PortfolioExecutionRequest,
    PortfolioConstructionConfig,
    PortfolioAssetTarget,
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
    build_horizons,
    build_portfolio_chart_data,
    load_csv_prices,
    run_independent_asset_grid_search,
    run_vectorized_portfolio_grid_search,
    portfolio_report_frame,
    summarize_portfolio_result,
    summarize_engine_batch,
    summarize_engine_runs,
)
from backtest_engine.chart_snapshot import ChartSnapshotExporter
from backtest_engine.magellan import MagellanClient, MagellanError
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
    QPushButton {{
        background: rgba(0,0,0,.2);
        border: 1px solid {PALETTE['border']};
        border-radius: 8px;
        padding: 6px 10px;
    }}
    QPushButton:hover {{
        border-color: {PALETTE['blue']};
    }}
    QTableView {{
        gridline-color: {PALETTE['border']};
        alternate-background-color: rgba(255,255,255,.03);
        selection-background-color: rgba(77,163,255,.25);
        selection-color: {PALETTE['text']};
        background: transparent;
    }}
    QHeaderView::section {{
        background: rgba(0,0,0,.08);
        color: {PALETTE['muted']};
        font-size: 11px;
        font-weight: 700;
        border: none;
        padding: 6px 8px;
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


class DatasetSelectionDialog(QtWidgets.QDialog):
    def __init__(self, datasets: Sequence[str], selected: Sequence[str], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Choose Study Datasets")
        self.resize(420, 520)

        layout = QtWidgets.QVBoxLayout(self)
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
        self.scheduled_tasks: list[dict] = []
        self.magellan = MagellanClient(self)
        self.snapshot_exporter = ChartSnapshotExporter()
        self._closing = False
        self._current_grid_started_at: float | None = None
        self._batch_benchmark_cache: dict[str, tuple[BatchExecutionBenchmark, ...]] = {}
        self.study_dataset_ids: list[str] = []
        self.portfolio_target_weights: dict[str, float] = {}
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
        self.tabs.addTab(self._build_heatmap_tab(), "Heatmaps")
        self.tabs.addTab(self._build_control_panel(), "Orchestrate")
        self.tabs.addTab(self._build_automate_tab(), "Automate")

        layout.addWidget(self.tabs)

        self.setCentralWidget(central)
        self.refresh_timer = QtCore.QTimer(self)
        self.refresh_timer.setInterval(2000)
        self.refresh_timer.timeout.connect(self._refresh_batches_live)
        self.refresh_timer.start()
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

        main_tab = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout(main_tab)
        main_layout.setSpacing(10)

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
            "Portfolio mode is currently same-timeframe only, long-only, and requires a vectorized-supported strategy."
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
        main_layout.addWidget(QtWidgets.QLabel("Strategy"))
        main_layout.addWidget(self.strategy_combo)
        main_layout.addWidget(QtWidgets.QLabel("Execution Mode"))
        main_layout.addWidget(self.execution_mode_combo)
        main_layout.addWidget(QtWidgets.QLabel("Study Mode"))
        main_layout.addWidget(self.study_mode_combo)

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
        self.dataset_combo = QtWidgets.QComboBox()
        self.dataset_combo.setEditable(True)
        self.dataset_combo.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.NoInsert)
        self.dataset_combo.setMinimumContentsLength(12)
        self.dataset_combo.currentTextChanged.connect(lambda _: self._update_study_dataset_summary())
        main_layout.addWidget(self.dataset_combo)
        main_layout.addWidget(QtWidgets.QLabel("Study Datasets"))
        dataset_row = QtWidgets.QHBoxLayout()
        self.study_dataset_summary = QtWidgets.QLineEdit()
        self.study_dataset_summary.setReadOnly(True)
        self.study_dataset_summary.setPlaceholderText("Current dataset only")
        choose_datasets_btn = QtWidgets.QPushButton("Choose Datasets")
        choose_datasets_btn.clicked.connect(self._choose_study_datasets)
        current_only_btn = QtWidgets.QPushButton("Current Only")
        current_only_btn.clicked.connect(self._reset_study_datasets_to_current)
        dataset_row.addWidget(self.study_dataset_summary, 3)
        dataset_row.addWidget(choose_datasets_btn, 1)
        dataset_row.addWidget(current_only_btn, 1)
        main_layout.addLayout(dataset_row)
        self.study_mode_note = QtWidgets.QLabel()
        self.study_mode_note.setObjectName("Sub")
        self.study_mode_note.setWordWrap(True)
        main_layout.addWidget(self.study_mode_note)
        self.study_mode_combo.currentTextChanged.connect(lambda _: self._update_study_mode_note())
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
        left_stack.addWidget(main_tab)
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

    def _build_heatmap_tab(self) -> QtWidgets.QWidget:
        return self._build_heatmap_panel()

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
        subtitle = QtWidgets.QLabel("Schedule data acquisition using Massive (1m, 2 years).")
        subtitle.setObjectName("Sub")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        ticker_row = QtWidgets.QHBoxLayout()
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
        ticker_row.addWidget(self.ticker_summary, 3)
        ticker_row.addWidget(select_all_btn, 1)
        ticker_row.addWidget(choose_btn, 1)
        ticker_row.addWidget(refresh_btn, 1)
        layout.addWidget(QtWidgets.QLabel("Tickers (NASDAQ + Other Listed)"))
        layout.addLayout(ticker_row)

        schedule_row = QtWidgets.QHBoxLayout()
        self.schedule_combo = QtWidgets.QComboBox()
        self.schedule_combo.addItems(["Nightly", "Weekly", "Monthly"])
        schedule_row.addWidget(self.schedule_combo, 1)
        schedule_row.addStretch(2)
        layout.addWidget(QtWidgets.QLabel("Download Frequency"))
        layout.addLayout(schedule_row)

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

        layout.addLayout(schedule_grid)

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
        layout.addLayout(controls_row)

        self.download_status = QtWidgets.QLabel("Idle")
        self.download_status.setObjectName("Sub")
        layout.addWidget(self.download_status)
        self.download_progress = QtWidgets.QProgressBar()
        self.download_progress.setRange(0, 0)
        self.download_progress.setVisible(False)
        layout.addWidget(self.download_progress)

        concurrency_row = QtWidgets.QHBoxLayout()
        self.concurrency_spin = QtWidgets.QSpinBox()
        self.concurrency_spin.setRange(1, 1)
        self.concurrency_spin.setValue(1)
        self.concurrency_spin.setEnabled(False)
        concurrency_row.addWidget(QtWidgets.QLabel("Concurrent Downloads"))
        concurrency_row.addWidget(self.concurrency_spin)
        concurrency_row.addStretch(1)
        layout.addLayout(concurrency_row)

        self.resume_chk = QtWidgets.QCheckBox("Resume if previously interrupted")
        self.resume_chk.setChecked(True)
        layout.addWidget(self.resume_chk)

        autostart_row = QtWidgets.QHBoxLayout()
        self.autostart_chk = QtWidgets.QCheckBox("Auto-start scheduler on login (macOS/Windows/Linux)")
        self.autostart_chk.stateChanged.connect(self._toggle_autostart)
        self.autostart_status = QtWidgets.QLabel("Status: unknown")
        self.autostart_status.setObjectName("Sub")
        autostart_row.addWidget(self.autostart_chk)
        autostart_row.addStretch(1)
        autostart_row.addWidget(self.autostart_status)
        layout.addLayout(autostart_row)
        self._refresh_autostart_status()

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
        layout.addWidget(self.progress_table)

        self.schedule_btn = QtWidgets.QPushButton("Schedule Task")
        self.schedule_btn.clicked.connect(self._schedule_task)
        layout.addWidget(self.schedule_btn, 0, QtCore.Qt.AlignmentFlag.AlignLeft)

        self.tasks_table = QtWidgets.QTableWidget(0, 10)
        self.tasks_table.setHorizontalHeaderLabels(
            ["Created", "Frequency", "Tickers", "Schedule", "Last Run", "Next Run", "Countdown", "Status", "Log", ""]
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
        layout.addWidget(self.tasks_table)

        layout.addStretch(1)
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
        self.select_all_tickers = True
        self.selected_tickers = []
        self._update_ticker_summary()

    def _open_ticker_picker(self) -> None:
        if not self.nasdaq_symbols:
            QtWidgets.QMessageBox.warning(self, "Symbols missing", f"Symbols file not found: {NASDAQ_SYMBOLS_PATH}")
            return
        dlg = TickerPickerDialog(self.nasdaq_symbols, set(self.selected_tickers), self)
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.select_all_tickers = False
            self.selected_tickers = dlg.selected
            self._update_ticker_summary()

    def _schedule_task(self) -> None:
        if self.select_all_tickers:
            symbols = self.nasdaq_symbols
        else:
            symbols = self.selected_tickers
        if not symbols:
            QtWidgets.QMessageBox.warning(self, "No tickers", "Select at least one ticker.")
            return
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
            "source": "massive",
            "resolution": "1m",
            "history": "2y",
            "schedule": schedule,
        }
        rc = ResultCatalog(self.catalog.db_path)
        rc.upsert_task(task_id, payload, schedule, status="active")
        self._load_tasks()
        self._refresh_tasks_table()
        QtWidgets.QMessageBox.information(self, "Scheduled", f"Scheduled {len(symbols)} tickers ({frequency}).")

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
            schedule_desc = self._format_schedule(schedule)
            last_run = self._format_timestamp(task.get("last_run_at"))
            next_run = self._parse_timestamp(task.get("next_run_at")) or self._compute_next_run(schedule, now)
            countdown = self._format_countdown(next_run, now) if next_run else "—"
            status = task.get("last_run_status") or task.get("status") or "—"
            message = task.get("last_run_message") or ""
            if message:
                status = f"{status}: {message}"
            self.tasks_table.setItem(row, 0, QtWidgets.QTableWidgetItem(created))
            self.tasks_table.setItem(row, 1, QtWidgets.QTableWidgetItem(frequency))
            self.tasks_table.setItem(row, 2, QtWidgets.QTableWidgetItem(str(len(symbols))))
            self.tasks_table.setItem(row, 3, QtWidgets.QTableWidgetItem(schedule_desc))
            self.tasks_table.setItem(row, 4, QtWidgets.QTableWidgetItem(last_run))
            self.tasks_table.setItem(row, 5, QtWidgets.QTableWidgetItem(self._format_timestamp(next_run)))
            self.tasks_table.setItem(row, 6, QtWidgets.QTableWidgetItem(countdown))
            self.tasks_table.setItem(row, 7, QtWidgets.QTableWidgetItem(status))
            log_btn = QtWidgets.QPushButton("Log")
            log_btn.clicked.connect(lambda _, i=idx: self._open_task_log(i))
            self.tasks_table.setCellWidget(row, 8, log_btn)
            remove_btn = QtWidgets.QPushButton("Unschedule")
            remove_btn.clicked.connect(lambda _, i=idx: self._unschedule_task(i))
            self.tasks_table.setCellWidget(row, 9, remove_btn)

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
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(f"Download Log {ticker}")
        dlg.resize(900, 600)
        layout = QtWidgets.QVBoxLayout(dlg)
        subtitle = QtWidgets.QLabel(str(path))
        subtitle.setObjectName("Sub")
        subtitle.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(subtitle)
        text = QtWidgets.QPlainTextEdit()
        text.setReadOnly(True)
        text.setPlainText(content)
        layout.addWidget(text)
        dlg.exec()

    def _start_download(self) -> None:
        if self.download_procs:
            QtWidgets.QMessageBox.information(self, "Download running", "A download process is already running.")
            return
        symbols = self.nasdaq_symbols if self.select_all_tickers else self.selected_tickers
        if not symbols:
            QtWidgets.QMessageBox.warning(self, "No tickers", "Select at least one ticker.")
            return
        self.download_queue = list(symbols)
        self.download_progress_rows = {}
        self.download_proc_meta = {}
        self.progress_table.setRowCount(0)
        self.download_paused = False
        self.download_progress.setVisible(True)
        self.download_progress.setRange(0, 0)
        self.download_status.setText("Starting downloads…")
        self.download_procs = []
        for _ in range(min(self.concurrency_spin.value(), len(self.download_queue))):
            self._start_next_download()

    def _start_next_download(self) -> None:
        if not self.download_queue:
            self.download_status.setText("Downloads complete.")
            self.download_progress.setVisible(False)
            self.download_active_ticker = None
            return
        ticker = self.download_queue.pop(0)
        self.download_active_ticker = ticker
        end_dt = pd.Timestamp.utcnow().date()
        start_dt = end_dt - pd.Timedelta(days=365 * 2)
        out_path = Path("data") / f"{ticker}_massive_{start_dt}_{end_dt}_1m.csv"
        log_path = self._create_download_log_path(ticker)
        self._ensure_progress_row(ticker, log_path)
        proc = QtCore.QProcess(self)
        proc.setProgram(sys.executable)
        args = [
            str(Path("scripts") / "fetch_massive.py"),
            ticker,
            "--out",
            str(out_path),
            "--progress",
            "--pace",
            "12.5",
        ]
        if self.resume_chk.isChecked():
            args.append("--resume")
        proc.setArguments(
            [
                *args,
            ]
        )
        proc.setProcessChannelMode(QtCore.QProcess.ProcessChannelMode.MergedChannels)
        proc.readyReadStandardOutput.connect(self._handle_download_output)
        proc.finished.connect(self._download_finished)
        proc.errorOccurred.connect(self._download_process_error)
        self.download_proc_meta[id(proc)] = {
            "ticker": ticker,
            "log_path": log_path,
            "buffer": "",
            "last_error": "",
            "out_path": str(out_path),
        }
        launch_text = (
            f"[{pd.Timestamp.now('UTC').isoformat()}] Launching download for {ticker}\n"
            f"Command: {sys.executable} {' '.join(args)}\n\n"
        )
        self._append_download_log(log_path, launch_text)
        proc.start()
        self.download_proc = proc
        self.download_procs.append(proc)
        self.download_status.setText(f"Downloading {ticker}…")
        self._update_progress_row(ticker, status="running", tooltip=f"Log: {log_path}")

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
        if meta:
            remainder = str(meta.get("buffer", ""))
            if remainder.strip():
                self._append_download_log(Path(meta["log_path"]), remainder + "\n")
                self._process_download_output_line(meta, remainder.strip())
                meta["buffer"] = ""
            log_path = Path(meta["log_path"])
            summary = str(meta.get("last_error") or "")
            if (exit_code != 0 or exit_status != QtCore.QProcess.ExitStatus.NormalExit) and not summary:
                try:
                    summary = self._summarize_error(log_path.read_text(encoding="utf-8"))
                except Exception:
                    summary = f"Process exited with code {exit_code}"
            if exit_code != 0 or exit_status != QtCore.QProcess.ExitStatus.NormalExit:
                if not summary:
                    summary = f"Process exited with code {exit_code}"
                self.download_status.setText(f"{ticker} failed: {summary}")
                self._update_progress_row(ticker, status="error", tooltip=summary)
            elif ticker:
                self._update_progress_row(ticker, tooltip=f"Completed. Log: {log_path}")
        if isinstance(proc, QtCore.QProcess) and proc in self.download_procs:
            self.download_procs.remove(proc)
        if isinstance(proc, QtCore.QProcess):
            self.download_proc_meta.pop(id(proc), None)
        self.download_proc = self.download_procs[0] if self.download_procs else None
        if self.download_paused:
            return
        if self.download_queue:
            self._start_next_download()
        elif not self.download_procs:
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
        runs = self.catalog.load_runs()
        batches = self.catalog.load_batches()
        self._render_batches(batches)
        self._update_metrics(runs)
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
            loaded = load_csv_prices(csv_path)
            duck = DuckDBStore()
            duck.write_parquet(dataset_id, loaded.data.reset_index())
            self._refresh_dataset_options(select_id=dataset_id)
            self.status_label.setText(f"Added {csv_path.name} → {dataset_id}")
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
        if bool(bt_settings.get("allow_short", True)):
            issues.append("Portfolio vectorization v1 is long-only, so 'Allow Short' must be turned off.")
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
        self.allow_short_chk.setChecked(False)
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
    ) -> tuple[list[str], Dict[str, float | bool | dict | str]] | None:
        issues, can_apply_defaults = self._portfolio_vectorized_support_issues(strategy_factory, timeframes, bt_settings)
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
        dataset_ids = self._selected_study_dataset_ids()
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
            compatible = self._ensure_portfolio_vectorized_compatible(strategy_factory, timeframes, bt_settings)
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
            self.study_mode_note.setText(
                "Portfolio mode uses shared cash across the selected datasets with the vectorized portfolio backend. "
                "Current scope is same-timeframe only, long-only, and one strategy applied across the selected assets."
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
            self.dataset_combo.blockSignals(True)
            self.dataset_combo.clear()
            self.dataset_combo.addItems(opts)
            self.dataset_combo.blockSignals(False)
            # Preserve current text or select provided.
            if select_id:
                self.dataset_combo.setCurrentText(select_id)
            elif opts:
                self.dataset_combo.setCurrentIndex(0)
            self.study_dataset_ids = [dataset_id for dataset_id in self.study_dataset_ids if dataset_id in opts]
            self.portfolio_target_weights = {
                dataset_id: weight
                for dataset_id, weight in self.portfolio_target_weights.items()
                if dataset_id in opts
            }
            self._update_study_dataset_summary()
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

    def _selected_study_dataset_ids(self) -> list[str]:
        explicit = [dataset_id for dataset_id in self.study_dataset_ids if dataset_id]
        if explicit:
            return explicit
        current = self.dataset_combo.currentText().strip()
        return [current] if current else []

    def _update_study_dataset_summary(self) -> None:
        if not hasattr(self, "study_dataset_summary"):
            return
        dataset_ids = self._selected_study_dataset_ids()
        if not dataset_ids:
            self.study_dataset_summary.setText("")
            self.study_dataset_summary.setToolTip("")
            self._update_portfolio_allocation_summary()
            return
        if len(dataset_ids) == 1 and not self.study_dataset_ids:
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
        if len(selected) == 1 and selected[0] == current:
            self.study_dataset_ids = []
        else:
            self.study_dataset_ids = selected
        self._update_study_dataset_summary()

    def _reset_study_datasets_to_current(self) -> None:
        self.study_dataset_ids = []
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

    def _rebuild_portfolio_result_for_run(self, run: "RunRow"):
        params = json.loads(run.params) if isinstance(run.params, str) else (run.params or {})
        assets_payload = params.get("assets") or []
        if not assets_payload:
            raise ValueError(
                "This portfolio run does not contain asset definitions needed to rebuild a chart snapshot."
            )

        execution_cfg = dict(params.get("execution_config") or {})
        construction_cfg = dict(params.get("construction_config") or {})
        timeframe = str(run.timeframe or "1 minutes")
        duck = DuckDBStore()
        assets: list[PortfolioExecutionAsset] = []
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
                )
            )

        config = BacktestConfig(
            timeframe=timeframe,
            starting_cash=(
                float(run.starting_cash)
                if getattr(run, "starting_cash", None) is not None
                else float(execution_cfg.get("starting_cash", 100_000.0))
            ),
            fee_rate=float(execution_cfg.get("fee_rate", self.bt_settings.get("fee_rate", 0.0002))),
            fee_schedule=execution_cfg.get(
                "fee_schedule",
                self.bt_settings.get("fee_schedule", {"buy": 0.0003, "sell": 0.0005}),
            ),
            slippage=float(execution_cfg.get("slippage", self.bt_settings.get("slippage", 0.0002))),
            slippage_schedule=execution_cfg.get(
                "slippage_schedule",
                self.bt_settings.get("slippage_schedule", {"buy": 0.0003, "sell": 0.0001}),
            ),
            fill_ratio=float(execution_cfg.get("fill_ratio", self.bt_settings.get("fill_ratio", 1.0))),
            fill_on_close=bool(execution_cfg.get("fill_on_close", self.bt_settings.get("fill_on_close", False))),
            allow_short=False,
            use_cache=False,
            base_execution=False,
            time_horizon_start=str(run.start) if getattr(run, "start", None) else None,
            time_horizon_end=str(run.end) if getattr(run, "end", None) else None,
            prevent_scale_in=bool(
                execution_cfg.get("prevent_scale_in", self.bt_settings.get("prevent_scale_in", True))
            ),
            one_order_per_signal=bool(
                execution_cfg.get("one_order_per_signal", self.bt_settings.get("one_order_per_signal", True))
            ),
            risk_free_rate=float(execution_cfg.get("risk_free_rate", self.bt_settings.get("risk_free_rate", 0.0))),
        )
        orchestrator = ExecutionOrchestrator()
        portfolio_result = orchestrator.execute_portfolio(
            PortfolioExecutionRequest(
                assets=assets,
                config=config,
                catalog=None,
                requested_execution_mode=ExecutionMode.VECTORIZED,
                normalize_weights=bool(params.get("normalize_weights", True)),
                portfolio_dataset_id=str(run.dataset_id),
                construction_config=PortfolioConstructionConfig(**construction_cfg),
                logical_run_id=str(getattr(run, "logical_run_id", "") or ""),
            )
        )
        return portfolio_result.result

    def _ensure_portfolio_snapshot(self, run: "RunRow") -> Path:
        existing = self._existing_snapshot_root_for_run(run)
        if existing is not None:
            return existing
        portfolio_result = self._rebuild_portfolio_result_for_run(run)
        artifact = self.snapshot_exporter.export_portfolio_snapshot(
            run=run,
            portfolio_result=portfolio_result,
            overwrite=True,
        )
        return artifact.snapshot_root

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
                snapshot_root = self._ensure_portfolio_snapshot(run)
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


class BatchDetailDialog(QtWidgets.QDialog):
    def __init__(
        self,
        batch: BatchRow,
        runs: List[RunRow],
        catalog_path: Path,
        parent=None,
        batch_benchmarks: Sequence[BatchExecutionBenchmark] = (),
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Batch {batch.batch_id}")
        self.resize(900, 600)
        self.catalog_path = catalog_path
        self.runs = list(runs)
        self.batch_benchmarks = tuple(batch_benchmarks)

        layout = QtWidgets.QVBoxLayout(self)
        header = QtWidgets.QLabel(
            f"Strategy: {batch.strategy} | Dataset: {batch.dataset_id}\n"
            f"Timeframes: {batch.timeframes} | Horizons: {batch.horizons}\n"
            f"Params: {batch.params}"
        )
        header.setObjectName("Sub")
        layout.addWidget(header)
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
        self.portfolio_report_btn = QtWidgets.QPushButton("Portfolio Report")
        self.portfolio_report_btn.clicked.connect(self._open_selected_portfolio_report)
        self.benchmark_btn = QtWidgets.QPushButton("Benchmarks")
        self.benchmark_btn.clicked.connect(self._show_batch_benchmarks)
        actions.addWidget(self.open_chart_btn)
        actions.addWidget(self.compare_btn)
        actions.addWidget(self.compare_batch_btn)
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
        parent = self.parent()
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
        parent = self.parent()
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
        parent = self.parent()
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


class BatchBenchmarkDialog(QtWidgets.QDialog):
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


class EngineComparisonDialog(QtWidgets.QDialog):
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
        parent = self.parent()
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


class BatchEngineComparisonDialog(QtWidgets.QDialog):
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


class PortfolioReportDialog(QtWidgets.QDialog):
    def __init__(self, run: RunRow, report, parent=None) -> None:
        super().__init__(parent)
        self.run = run
        self.report = report
        self.setWindowTitle(f"Portfolio Report | {run.run_id[:10]}")
        self.resize(1100, 700)

        layout = QtWidgets.QVBoxLayout(self)
        summary = QtWidgets.QLabel(self._summary_text(report))
        summary.setObjectName("Sub")
        summary.setWordWrap(True)
        summary.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(summary)

        frame = portfolio_report_frame(report)
        self.table = QtWidgets.QTableWidget(len(frame), len(frame.columns))
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
        for row_idx, row in frame.iterrows():
            values = [
                str(row["dataset_id"]),
                self._fmt(row["avg_weight"]),
                self._fmt(row["avg_target_weight"]),
                self._fmt(row["avg_abs_tracking_error"]),
                self._fmt(row["final_weight"]),
                self._fmt(row["peak_weight"]),
                self._fmt_pct(row["active_bar_fraction"]),
                str(int(row["trade_count"])),
                self._fmt_money(row["realized_pnl"]),
                self._fmt_money(row["turnover_notional"]),
                self._fmt(row["turnover_ratio"], precision=3),
            ]
            for col_idx, value in enumerate(values):
                self.table.setItem(row_idx, col_idx, QtWidgets.QTableWidgetItem(value))
        layout.addWidget(self.table)

        buttons = QtWidgets.QHBoxLayout()
        export_btn = QtWidgets.QPushButton("Export CSV")
        export_btn.clicked.connect(lambda: self._export_csv(frame))
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        buttons.addWidget(export_btn)
        buttons.addStretch(1)
        buttons.addWidget(close_btn)
        layout.addLayout(buttons)

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
    def _summary_text(cls, report) -> str:
        lines = [
            f"Starting equity: {cls._fmt_money(report.starting_equity)}",
            f"Ending equity: {cls._fmt_money(report.ending_equity)}",
            f"Total return: {report.total_return:.4f}",
            f"CAGR: {report.cagr:.4f}",
            f"Max drawdown: {report.max_drawdown:.4f}",
            f"Sharpe: {report.sharpe:.4f}",
            f"Rolling Sharpe: {report.rolling_sharpe:.4f}",
            f"Avg cash weight: {report.avg_cash_weight:.4f}",
            f"Avg gross exposure: {report.avg_gross_exposure:.4f}",
            f"Peak gross exposure: {report.peak_gross_exposure:.4f}",
            f"Avg target gross exposure: {report.avg_target_gross_exposure:.4f}",
            f"Trades: {report.trade_count}",
            f"Total turnover: {cls._fmt_money(report.total_turnover_notional)} ({report.total_turnover_ratio:.3f}x)",
        ]
        return " | ".join(lines)

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


class PortfolioRunChartDialog(QtWidgets.QDialog):
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


class TradesLogDialog(QtWidgets.QDialog):
    def __init__(self, run_id: str, trades_df: pd.DataFrame, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Trades Log {run_id[:12]}…")
        self.resize(900, 600)
        layout = QtWidgets.QVBoxLayout(self)
        if trades_df is None or trades_df.empty:
            msg = QtWidgets.QLabel("No trades available for this run.")
            msg.setObjectName("Sub")
            layout.addWidget(msg)
            return
        model = TradesTableModel(trades_df)
        table = QtWidgets.QTableView()
        table.setModel(model)
        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        table.horizontalHeader().setStretchLastSection(True)
        table.verticalHeader().setVisible(False)
        table.setObjectName("Panel")
        layout.addWidget(table)


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


class TickerPickerDialog(QtWidgets.QDialog):
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


class RunChartDialog(QtWidgets.QDialog):
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
