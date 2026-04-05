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
    BacktestConfig,
    BacktestEngine,
    DuckDBStore,
    GridSearch,
    GridSpec,
    ResultCatalog,
    SMACrossStrategy,
    InverseTurtleStrategy,
    build_horizons,
    load_csv_prices,
)
from backtest_engine.magellan import MagellanClient, MagellanError
from backtest_engine.reporting import plot_param_heatmap


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
                    SELECT run_id,batch_id,strategy,params,timeframe,start,end,dataset_id,starting_cash,metrics,run_started_at,run_finished_at,status
                    FROM runs WHERE batch_id=? ORDER BY created_at DESC
                    """,
                    (batch_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT run_id,batch_id,strategy,params,timeframe,start,end,dataset_id,starting_cash,metrics,run_started_at,run_finished_at,status
                    FROM runs ORDER BY created_at DESC
                    """
                ).fetchall()
        result = []
        for r in rows:
            result.append(
                RunRow(
                    run_id=r[0],
                    batch_id=r[1],
                    strategy=r[2],
                    params=r[3],
                    timeframe=r[4],
                    start=r[5],
                    end=r[6],
                    dataset_id=r[7],
                    starting_cash=r[8],
                    metrics=json.loads(r[9]) if r[9] else {},
                    run_started_at=r[10],
                    run_finished_at=r[11],
                    status=r[12] or "finished",
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
class RunsTableModel(QtCore.QAbstractTableModel):
    def __init__(self, runs: Sequence[RunRow]) -> None:
        super().__init__()
        self._runs = list(runs)
        self._headers = [
            "Status",
            "Run ID",
            "Strategy",
            "Params",
            "Timeframe",
            "Data Start",
            "Data End",
            "Run Started",
            "Run Finished",
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
                return run.params
            if col == 4:
                return run.timeframe
            if col == 5:
                return run.start
            if col == 6:
                return run.end
            if col == 7:
                return run.run_started_at or ""
            if col == 8:
                return run.run_finished_at or ""
            if col == 9:
                return run.dataset_id
            if col == 10:
                return "—" if not metrics else f"{metrics.get('total_return', 0):.4f}"
            if col == 11:
                return "—" if not metrics else f"{metrics.get('sharpe', 0):.3f}"
            if col == 12:
                if not metrics:
                    return "—"
                roll = metrics.get("rolling_sharpe")
                if roll is None or (isinstance(roll, float) and roll != roll):
                    return "—"
                return f"{roll:.3f}"
            if col == 13:
                return "—" if not metrics else f"{metrics.get('max_drawdown', 0):.4f}"
            if col == 14:
                start_cash = run.starting_cash if run.starting_cash is not None else 100_000
                final_equity = start_cash * (1 + metrics.get("total_return", 0)) if metrics else start_cash
                return f"{final_equity:,.0f}"
            if col == 15:
                return ""
            if col == 16:
                return ""
        if role == QtCore.Qt.ItemDataRole.ForegroundRole and col == 0:
            color = PALETTE["green"] if run.status == "finished" else PALETTE["red"]
            return QtGui.QColor(color)
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
                total = batch.run_total or batch.run_count
                return f"{batch.finished_count}/{total}" if total else "0"
            if col == 10:
                return "—" if not metrics else f"{metrics.get('total_return', 0):.4f}"
            if col == 11:
                return "—" if not metrics else f"{metrics.get('sharpe', 0):.3f}"
            if col == 12:
                return "—" if not metrics else f"{metrics.get('max_drawdown', 0):.4f}"
            if col == 13:
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
        timeframes: list[str],
        horizons: list[str],
        catalog_path: Path,
        strategy_factory: Callable,
        strategy_params: Dict[str, float],
        blas_threads: int,
        intrabar_sim: bool,
        sharpe_debug: bool,
        risk_free_rate: float,
        bt_settings: Dict[str, float | bool | dict],
    ) -> None:
        super().__init__()
        self.csv_path = csv_path
        self.dataset_id = dataset_id
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

    def run(self) -> None:
        try:
            # Allow worker to tune BLAS threads per run.
            os.environ["OPENBLAS_NUM_THREADS"] = str(self.blas_threads)
            os.environ["OMP_NUM_THREADS"] = str(self.blas_threads)
            os.environ["VECLIB_MAXIMUM_THREADS"] = str(self.blas_threads)
            os.environ["NUMEXPR_MAX_THREADS"] = str(self.blas_threads)

            duck = DuckDBStore()
            try:
                raw = duck.load(self.dataset_id)
            except Exception as exc:
                raise RuntimeError(f"Dataset '{self.dataset_id}' not found in DuckDB/parquet store. Add it first.") from exc
            end_ts = raw.index[-1]

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

            def loader(tf: str):
                return duck.resample(self.dataset_id, tf)

            grid = GridSearch(
                dataset_id=self.dataset_id,
                data_loader=loader,
                strategy_cls=self.strategy_factory,
                base_config=base_config,
                catalog=catalog,
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
                description=f"Grid for {self.dataset_id}",
                batch_id=self.batch_id,
            )
            catalog.save_batch(
                batch_id=self.batch_id,
                strategy=self.strategy_factory.__name__,
                dataset_id=self.dataset_id,
                params=self.strategy_params,
                timeframes=self.timeframes,
                horizons=[str(h) for h in self.horizons],
                run_total=run_total,
                status="running",
                started_at=started_at,
                finished_at=None,
            )
            df = grid.run(
                spec,
                make_heatmap=False,  # avoid matplotlib in worker thread to prevent crashes
                stop_cb=lambda: self._stop_requested,
                progress_cb=lambda d, t: self.progress_signal.emit(d, t),
            )
            message = "Grid stopped." if self._stop_requested else "Grid completed."
            catalog.save_batch(
                batch_id=self.batch_id,
                strategy=self.strategy_factory.__name__,
                dataset_id=self.dataset_id,
                params=self.strategy_params,
                timeframes=self.timeframes,
                horizons=[str(h) for h in self.horizons],
                run_total=run_total,
                status="finished" if not self._stop_requested else "stopped",
                started_at=started_at,
                finished_at=pd.Timestamp.utcnow().isoformat(),
            )
            self.finished_signal.emit({"df": df, "spec": spec, "message": message})
        except Exception as exc:
            tb = traceback.format_exc()
            print("GridWorker error:\n", tb)
            err_text = tb if tb else str(exc)
            self.error_signal.emit(err_text)

    def request_stop(self) -> None:
        self._stop_requested = True


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
        self.scheduled_tasks: list[dict] = []
        self.magellan = MagellanClient(self)

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
        QtCore.QTimer.singleShot(0, self._warm_magellan)

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
        main_layout.addWidget(QtWidgets.QLabel("Strategy"))
        main_layout.addWidget(self.strategy_combo)

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
        main_layout.addWidget(self.dataset_combo)

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

        self.progress_table = QtWidgets.QTableWidget(0, 5)
        self.progress_table.setHorizontalHeaderLabels(["Ticker", "Status", "Pages", "Rows", "Progress"])
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
        self._ensure_progress_row(ticker)
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
        proc.start()
        self.download_proc = proc
        self.download_procs.append(proc)
        self.download_status.setText(f"Downloading {ticker}…")
        self._update_progress_row(ticker, status="running")

    def _handle_download_output(self) -> None:
        proc = self.sender()
        if not isinstance(proc, QtCore.QProcess):
            return
        data = bytes(proc.readAllStandardOutput()).decode("utf-8", errors="ignore")
        for line in data.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
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

    def _download_finished(self) -> None:
        proc = self.sender()
        if isinstance(proc, QtCore.QProcess) and proc in self.download_procs:
            self.download_procs.remove(proc)
        self.download_proc = self.download_procs[0] if self.download_procs else None
        if self.download_paused:
            return
        if self.download_queue:
            self._start_next_download()
        elif not self.download_procs:
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

    def _ensure_progress_row(self, ticker: str) -> None:
        if ticker in self.download_progress_rows:
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
        self.download_progress_rows[ticker] = {"row": row, "bar": bar}

    def _update_progress_row(
        self,
        ticker: str | None,
        status: str | None = None,
        pages: int | None = None,
        rows: int | None = None,
        done: bool = False,
    ) -> None:
        if not ticker or ticker not in self.download_progress_rows:
            return
        row = self.download_progress_rows[ticker]["row"]
        bar: QtWidgets.QProgressBar = self.download_progress_rows[ticker]["bar"]
        if status:
            item = self.progress_table.item(row, 1)
            if item:
                item.setText(status)
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

    def _collect_backtest_settings(self) -> Dict[str, float | bool | dict]:
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
        }

    def _run_grid_clicked(self) -> None:
        csv_path = Path(self.csv_path_edit.text().strip())
        dataset_id = self.dataset_combo.currentText().strip() or "dataset"
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
        # Ensure dataset already imported.
        duck = DuckDBStore()
        try:
            duck.load(dataset_id)
        except Exception:
            QtWidgets.QMessageBox.warning(self, "Dataset missing", f"Dataset '{dataset_id}' not found. Click 'Add CSV to Database' first.")
            return
        if not timeframes:
            QtWidgets.QMessageBox.warning(self, "Timeframes", "Provide at least one timeframe.")
            return

        strategy_factory, strat_params, param_errors = self._collect_strategy_params()
        if param_errors:
            QtWidgets.QMessageBox.warning(self, "Parameter warnings", "Some values were invalid and replaced with defaults:\n" + "\n".join(param_errors))

        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress.setValue(0)
        self.status_label.setText("Running grid…")

        self.worker = GridWorker(
            csv_path=csv_path,
            dataset_id=dataset_id,
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
        if isinstance(payload, dict):
            message = payload.get("message", message)
            df = payload.get("df")
            spec = payload.get("spec")
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
        self.status_label.setText(f"Running grid… {done}/{total}")

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
            fig = plot_param_heatmap(df, value_col=metric, row=row, col=col, title=f"{metric} heatmap")
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

    def _refresh_dataset_options(self, select_id: str | None = None) -> None:
        try:
            store = DuckDBStore()
            opts = []
            for path in store.data_dir.glob("*.parquet"):
                if path.is_file():
                    opts.append(path.stem)
            opts = sorted(set(opts))
            self.dataset_combo.blockSignals(True)
            self.dataset_combo.clear()
            self.dataset_combo.addItems(opts)
            self.dataset_combo.blockSignals(False)
            # Preserve current text or select provided.
            if select_id:
                self.dataset_combo.setCurrentText(select_id)
            elif opts:
                self.dataset_combo.setCurrentIndex(0)
        except Exception:
            # Best-effort; leave combo as-is if listing fails.
            pass

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.quit()
            self.worker.wait(2000)
        self.magellan.shutdown()
        event.accept()

    def _warm_magellan(self) -> None:
        try:
            self.magellan.ensure_running(timeout_ms=2500)
        except Exception:
            # Keep warmup best-effort so the dashboard still opens even if Magellan is unavailable.
            pass

    def _open_run_chart_in_magellan(self, run: "RunRow") -> bool:
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            bars = RunChartDialog._load_bars_utc_static(run)
            if bars is None or bars.empty:
                raise MagellanError("No bar data is available for this run.")

            overlay_defs, pane_defs = RunChartDialog._build_magellan_indicator_defs_static(run, bars)
            trade_markers, equity_points = RunChartDialog._build_magellan_trade_payload_static(
                run,
                self.catalog.db_path,
                bars,
            )

            if not self.magellan.ensure_running(timeout_ms=5000):
                raise MagellanError(self.magellan.last_error or "Magellan is unavailable.")

            session_id = f"backtest-run-{run.run_id}"
            title = run.dataset_id
            subtitle = f"{run.strategy} | {run.timeframe} | {run.run_id[:12]}..."
            self.magellan.open_live_session(
                session_id,
                title=title,
                subtitle=subtitle,
                status_text="Loading historical run...",
            )

            timestamps_ns = bars.index.view("int64")
            opens = bars["open"].to_numpy(dtype=float)
            highs = bars["high"].to_numpy(dtype=float)
            lows = bars["low"].to_numpy(dtype=float)
            closes = bars["close"].to_numpy(dtype=float)
            volumes = bars["volume"].to_numpy(dtype=float) if "volume" in bars.columns else np.zeros(len(bars), dtype=float)
            chunk_size = 1500
            equity_chunks: dict[int, list[dict]] = {}
            marker_chunks: dict[int, list[dict]] = {}
            for point in equity_points:
                equity_chunks.setdefault(int(point["bar_index"]) // chunk_size, []).append(point)
            for marker in trade_markers:
                marker_chunks.setdefault(int(marker["bar_index"]) // chunk_size, []).append(marker)

            def build_series_chunk(series_defs: list[tuple[str, str, np.ndarray]], start: int, stop: int) -> list[dict]:
                series_payload: list[dict] = []
                ts_chunk = timestamps_ns[start:stop]
                for name, color, values in series_defs:
                    points = [
                        {
                            "timestamp_utc_ns": str(int(ts_ns)),
                            "value": float(value),
                            "bar_index": start + offset,
                        }
                        for offset, (ts_ns, value) in enumerate(zip(ts_chunk, values[start:stop]))
                        if np.isfinite(value)
                    ]
                    if points:
                        series_payload.append({"name": name, "color": color, "points": points})
                return series_payload

            for chunk_idx, start in enumerate(range(0, len(bars), chunk_size), start=1):
                stop = min(start + chunk_size, len(bars))
                bar_payload = [
                    {
                        "timestamp_utc_ns": str(int(timestamps_ns[row])),
                        "open": float(opens[row]),
                        "high": float(highs[row]),
                        "low": float(lows[row]),
                        "close": float(closes[row]),
                        "volume": float(volumes[row]),
                        "bar_index": row,
                    }
                    for row in range(start, stop)
                ]
                overlay_payload = build_series_chunk(overlay_defs, start, stop)
                pane_payload = build_series_chunk(pane_defs, start, stop)
                chunk_key = start // chunk_size
                chunk_equity_points = equity_chunks.get(chunk_key, [])
                equity_payload = [{"name": "Equity", "color": "#27d07d", "points": chunk_equity_points}] if chunk_equity_points else []
                marker_payload = marker_chunks.get(chunk_key, [])
                is_last_chunk = stop >= len(bars)
                status_text = (
                    "Historical run ready."
                    if is_last_chunk
                    else f"Loading historical run... {stop:,}/{len(bars):,} bars"
                )
                self.magellan.send_live_update(
                    session_id,
                    title=title if is_last_chunk else "",
                    subtitle=subtitle if is_last_chunk else "",
                    status_text=status_text,
                    bars=bar_payload,
                    overlay_series=overlay_payload,
                    pane_series=pane_payload,
                    equity_series=equity_payload,
                    trade_markers=marker_payload,
                    timeout_ms=1500,
                )
                if chunk_idx % 8 == 0:
                    QtCore.QCoreApplication.processEvents()
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
        dlg = BatchDetailDialog(batch, runs, self.catalog.db_path, self)
        dlg.exec()

    # -- strategies ----------------------------------------------------------
    def _init_strategy_selector(self) -> None:
        self.strategy_specs: Dict[str, Dict[str, Tuple[type, type, float]]] = {
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
            edit.setToolTip("Enter comma-separated values or range notation start:end:step for step sizes.")
            self.strategy_params_box.addRow(QtWidgets.QLabel(param), edit)
            self.param_inputs[param] = edit

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
                        if ":" in token:
                            start_s, end_s, step_s = token.split(":")
                            start_v = ptype(start_s)
                            end_v = ptype(end_s)
                            step_v = ptype(step_s)
                            if step_v == 0:
                                raise ValueError("step cannot be 0")
                            current = start_v
                            while (current <= end_v) if step_v > 0 else (current >= end_v):
                                vals.append(current)
                                current = ptype(current + step_v)
                        else:
                            vals.append(ptype(token))
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
    def __init__(self, batch: BatchRow, runs: List[RunRow], catalog_path: Path, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Batch {batch.batch_id}")
        self.resize(900, 600)
        self.catalog_path = catalog_path

        layout = QtWidgets.QVBoxLayout(self)
        header = QtWidgets.QLabel(
            f"Strategy: {batch.strategy} | Dataset: {batch.dataset_id}\n"
            f"Timeframes: {batch.timeframes} | Horizons: {batch.horizons}\n"
            f"Params: {batch.params}"
        )
        header.setObjectName("Sub")
        layout.addWidget(header)

        model = RunsTableModel(runs)
        table = QtWidgets.QTableView()
        table.setModel(model)
        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        table.horizontalHeader().setStretchLastSection(True)
        table.verticalHeader().setVisible(False)
        table.setObjectName("Panel")
        table.doubleClicked.connect(lambda idx: self._open_run_chart(model, idx))
        # Add log/download buttons per row for trades.
        for r_idx, run in enumerate(runs):
            log_btn = QtWidgets.QPushButton("Log")
            log_btn.clicked.connect(lambda _, rr=run: self._open_trades_log(rr))
            table.setIndexWidget(model.index(r_idx, model.columnCount() - 2), log_btn)
            btn = QtWidgets.QPushButton("Download")
            btn.clicked.connect(lambda _, rr=run: self._download_trades(rr))
            table.setIndexWidget(model.index(r_idx, model.columnCount() - 1), btn)
        layout.addWidget(table)

    def _collect_backtest_settings(self) -> Dict[str, float | bool | dict]:
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
        }

    def _open_run_chart(self, model: RunsTableModel, index: QtCore.QModelIndex) -> None:
        if not index.isValid():
            return
        row = index.row()
        if row < 0 or row >= model.rowCount():
            return
        run = model._runs[row]
        parent = self.parent()
        if parent and hasattr(parent, "_open_run_chart_in_magellan"):
            try:
                if parent._open_run_chart_in_magellan(run):  # type: ignore[attr-defined]
                    return
                magellan = getattr(parent, "magellan", None)
                magellan_error = getattr(magellan, "last_error", "")
                if magellan_error:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Magellan unavailable",
                        f"{magellan_error}\n\nFalling back to the built-in chart viewer.",
                    )
            except Exception as exc:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Magellan unavailable",
                    f"{exc}\n\nFalling back to the built-in chart viewer.",
                )
        bt_settings = self._collect_backtest_settings()
        dlg = RunChartDialog(run, self.catalog_path, bt_settings, self)
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
    def __init__(self, run: RunRow, catalog_path: Path, bt_settings: Dict[str, float | bool | dict], parent=None) -> None:
        super().__init__(parent)
        self.catalog_path = Path(catalog_path)
        self.bt_settings = bt_settings
        self.setWindowTitle(f"Run {run.run_id[:12]}… | {run.strategy}")
        self.resize(1400, 900)

        layout = QtWidgets.QVBoxLayout(self)
        summary = QtWidgets.QLabel(
            f"Strategy: {run.strategy} | Timeframe: {run.timeframe} | Dataset: {run.dataset_id}\n"
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
                raise ValueError("No bar data available for this run.")

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
        }
        return mapping.get(name, SMACrossStrategy)

    def _load_bars(self, run: RunRow) -> pd.DataFrame | None:
        return self._load_bars_static(run)

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
    def _rerun_static(run: RunRow, bars: pd.DataFrame, bt_settings: Dict[str, float | bool | dict]):
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
        engine = BacktestEngine(
            data=bars,
            dataset_id=run.dataset_id,
            strategy_cls=cls,
            catalog=None,
            config=config,
            base_data=bars if run.timeframe == "1 minutes" else None,
        )
        return engine.run(params)

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
    def _build_magellan_indicator_defs_static(run: RunRow, bars: pd.DataFrame) -> tuple[list[tuple[str, str, np.ndarray]], list[tuple[str, str, np.ndarray]]]:
        color_map = {
            "SMA Fast": "#4da3ff",
            "SMA Slow": "#ffcc66",
            "Upper": "#4da3ff",
            "Lower": "#ff4d6d",
            "Exit Upper": "#7ee787",
            "Exit Lower": "#a28bff",
            "ATR": "#a28bff",
        }
        indicators = RunChartDialog._compute_indicators_static(run, bars)
        overlay_defs: list[tuple[str, str, np.ndarray]] = []
        pane_defs: list[tuple[str, str, np.ndarray]] = []
        for name, series in indicators.items():
            values = pd.to_numeric(series.reindex(bars.index), errors="coerce").to_numpy(dtype=float)
            target = pane_defs if name == "ATR" else overlay_defs
            target.append((name, color_map.get(name, "#4da3ff"), values))
        return overlay_defs, pane_defs

    @staticmethod
    def _build_magellan_trade_payload_static(
        run: RunRow,
        catalog_path: Path,
        bars: pd.DataFrame,
    ) -> tuple[list[dict], list[dict]]:
        if bars.empty:
            return [], []

        catalog = ResultCatalog(catalog_path)
        trades = catalog.load_trades(run.run_id) or []
        bar_timestamps_ns = bars.index.view("int64")
        first_ts = bars.index[0]
        last_ts = bars.index[-1]
        starting_cash = float(run.starting_cash) if run.starting_cash is not None else 100_000.0
        trade_markers: list[dict] = []
        equity_points_by_bar: dict[int, dict] = {
            0: {
                "timestamp_utc_ns": str(int(bar_timestamps_ns[0])),
                "value": starting_cash,
                "bar_index": 0,
            }
        }
        position = 0.0

        for seq, trade in enumerate(trades, start=1):
            trade_ts = pd.to_datetime(trade.get("timestamp"), utc=True, errors="coerce")
            if pd.isna(trade_ts) or trade_ts < first_ts or trade_ts > last_ts:
                continue
            bar_index = int(bars.index.get_indexer([trade_ts], method="nearest")[0])
            if bar_index < 0:
                continue

            qty = float(trade.get("qty", 0.0))
            position_before = position
            position += qty
            if position_before * position < 0:
                event = "flip"
            elif abs(position) > abs(position_before):
                event = "entry"
            elif position == position_before:
                event = "adjust"
            else:
                event = "exit"

            side = "buy" if qty > 0 else "sell"
            timestamp_ns = int(trade_ts.value)
            trade_markers.append(
                {
                    "timestamp_utc_ns": str(timestamp_ns),
                    "bar_index": bar_index,
                    "price": float(trade.get("price", 0.0)),
                    "quantity": abs(qty),
                    "side": side,
                    "event": event,
                    "label": f"{side.title()} {seq}",
                }
            )

            equity_after = trade.get("equity_after")
            if equity_after is not None:
                equity_points_by_bar[bar_index] = {
                    "timestamp_utc_ns": str(timestamp_ns),
                    "value": float(equity_after),
                    "bar_index": bar_index,
                }

        equity_points = [
            equity_points_by_bar[index]
            for index in sorted(equity_points_by_bar)
        ]
        if len(equity_points) == 1 and len(bars) > 1:
            equity_points.append(
                {
                    "timestamp_utc_ns": str(int(bar_timestamps_ns[-1])),
                    "value": equity_points[0]["value"],
                    "bar_index": len(bars) - 1,
                }
            )
        return trade_markers, equity_points

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
