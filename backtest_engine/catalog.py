from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from .metrics import PerformanceMetrics


@dataclass
class CachedRun:
    run_id: str
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
                    status TEXT DEFAULT 'finished'
                )
                """
            )
            # Backfill new columns if upgrading an existing DB.
            self._ensure_column(conn, "runs", "batch_id", "TEXT")
            self._ensure_column(conn, "runs", "run_started_at", "TEXT")
            self._ensure_column(conn, "runs", "run_finished_at", "TEXT")
            self._ensure_column(conn, "runs", "status", "TEXT", default="'finished'")
            self._ensure_column(conn, "runs", "starting_cash", "REAL")
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
                SELECT run_id,batch_id,strategy,params,timeframe,start,end,dataset_id,starting_cash,metrics,run_started_at,run_finished_at,status
                FROM runs WHERE run_id=?
                """,
                (run_id,),
            ).fetchone()
        if not row:
            return None
        metrics_json = json.loads(row[9]) if row[9] else {}
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
            batch_id=row[1],
            strategy=row[2],
            params=row[3],
            timeframe=row[4],
            start=row[5],
            end=row[6],
            dataset_id=row[7],
            starting_cash=row[8],
            metrics=metrics,
            run_started_at=row[10],
            run_finished_at=row[11],
            status=row[12] or "finished",
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
    ) -> None:
        encoded_params = json.dumps(params, sort_keys=True)
        metrics_json = metrics.to_json() if metrics else "{}"
        run_started_at = run_started_at or None
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs
                (run_id, batch_id, strategy, params, timeframe, start, end, dataset_id, starting_cash, metrics, run_started_at, run_finished_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
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
