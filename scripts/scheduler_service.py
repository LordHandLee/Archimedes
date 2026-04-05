from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, time as dtime, timezone
from pathlib import Path
from typing import Dict, List, Optional

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore[assignment]


DB_PATH = Path("backtests.sqlite")
LOG_DIR = Path("data") / "scheduler_logs"
PACE_SECONDS = "12.5"


@dataclass
class ScheduledTask:
    task_id: str
    symbols: Dict
    schedule: Dict
    status: str
    last_run_at: str | None
    last_run_status: str | None
    last_run_message: str | None
    next_run_at: str | None


def _db() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def load_tasks() -> List[ScheduledTask]:
    if not DB_PATH.exists():
        return []
    with _db() as conn:
        rows = conn.execute(
            """
            SELECT task_id, symbols, schedule, status, last_run_at, last_run_status, last_run_message, next_run_at
            FROM scheduled_tasks WHERE status='active'
            """
        ).fetchall()
    tasks = []
    for r in rows:
        tasks.append(
            ScheduledTask(
                task_id=r[0],
                symbols=json.loads(r[1]) if r[1] else {},
                schedule=json.loads(r[2]) if r[2] else {},
                status=r[3],
                last_run_at=r[4],
                last_run_status=r[5],
                last_run_message=r[6],
                next_run_at=r[7],
            )
        )
    return tasks


def update_task_run_info(
    task_id: str,
    last_run_at: str | None,
    last_run_status: str | None,
    last_run_message: str | None,
    next_run_at: str | None,
) -> None:
    with _db() as conn:
        conn.execute(
            """
            UPDATE scheduled_tasks
            SET last_run_at=?, last_run_status=?, last_run_message=?, next_run_at=?, updated_at=CURRENT_TIMESTAMP
            WHERE task_id=?
            """,
            (last_run_at, last_run_status, last_run_message, next_run_at, task_id),
        )


def add_task_run(run_id: str, task_id: str, started_at: str, status: str, ticker_count: int, log_path: str) -> None:
    with _db() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO task_runs
            (run_id, task_id, started_at, finished_at, status, ticker_count, log_path, error_message)
            VALUES (?, ?, ?, NULL, ?, ?, ?, NULL)
            """,
            (run_id, task_id, started_at, status, ticker_count, log_path),
        )


def finish_task_run(run_id: str, finished_at: str, status: str, error_message: str | None) -> None:
    with _db() as conn:
        conn.execute(
            """
            UPDATE task_runs
            SET finished_at=?, status=?, error_message=?
            WHERE run_id=?
            """,
            (finished_at, status, error_message, run_id),
        )


def compute_next_run(schedule: Dict, now_utc: datetime) -> Optional[datetime]:
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
    valid_weeks: set[int | str] = set()
    for w in weeks or ["1"]:
        if w == "Last":
            valid_weeks.add("Last")
        else:
            try:
                valid_weeks.add(int(w))
            except Exception:
                continue

    tz = ZoneInfo("America/New_York") if ZoneInfo else timezone.utc
    now_et = now_utc.astimezone(tz)
    start = now_et + timedelta(minutes=1)
    for i in range(0, 365 * 2):
        day = (start.date() + timedelta(days=i))
        candidate = datetime.combine(day, dtime(hour=hour, minute=minute), tzinfo=tz)
        if candidate <= now_et:
            continue
        if candidate.weekday() not in valid_weekdays:
            continue
        if candidate.month not in valid_months:
            continue
        if freq == "monthly":
            week_number = ((candidate.day - 1) // 7) + 1
            last_day = (candidate.replace(day=1) + timedelta(days=32)).replace(day=1) - timedelta(days=1)
            is_last = candidate.day + 7 > last_day.day
            if "Last" in valid_weeks and is_last:
                return candidate.astimezone(timezone.utc)
            if any(isinstance(w, int) and w == week_number for w in valid_weeks):
                return candidate.astimezone(timezone.utc)
            continue
        if freq in ("weekly", "nightly"):
            return candidate.astimezone(timezone.utc)
    return None


def run_task(task: ScheduledTask) -> None:
    symbols = task.symbols.get("symbols", []) if isinstance(task.symbols, dict) else []
    if not symbols:
        return
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    started_at = datetime.now(timezone.utc).isoformat()
    log_dir = LOG_DIR / task.task_id
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_id}.log"
    add_task_run(run_id, task.task_id, started_at, "running", len(symbols), str(log_path))
    update_task_run_info(task.task_id, started_at, "running", None, None)

    error_message = None
    status = "success"
    with log_path.open("a", encoding="utf-8") as log_file:
        for ticker in symbols:
            out_path = Path("data") / f"{ticker}_2y_1m.csv"
            cmd = [
                sys.executable,
                str(Path("scripts") / "fetch_massive.py"),
                ticker,
                "--out",
                str(out_path),
                "--progress",
                "--resume",
                "--pace",
                PACE_SECONDS,
            ]
            log_file.write(f"[{datetime.utcnow().isoformat()}] START {ticker}\n")
            log_file.flush()
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
                if proc.stdout:
                    log_file.write(proc.stdout + "\n")
                if proc.stderr:
                    log_file.write(proc.stderr + "\n")
            except subprocess.CalledProcessError as exc:
                status = "failed"
                error_message = f"{ticker} failed: {exc}"
                if exc.stdout:
                    log_file.write(exc.stdout + "\n")
                if exc.stderr:
                    log_file.write(exc.stderr + "\n")
                log_file.write(f"[{datetime.utcnow().isoformat()}] ERROR {ticker}\n")
                log_file.flush()
                break
            log_file.write(f"[{datetime.utcnow().isoformat()}] DONE {ticker}\n")
            log_file.flush()

    finished_at = datetime.now(timezone.utc).isoformat()
    finish_task_run(run_id, finished_at, status, error_message)
    next_run = compute_next_run(task.schedule, datetime.now(timezone.utc))
    update_task_run_info(
        task.task_id,
        finished_at,
        status,
        error_message,
        next_run.isoformat() if next_run else None,
    )


def main() -> None:
    while True:
        now = datetime.now(timezone.utc)
        tasks = load_tasks()
        for task in tasks:
            next_run = None
            if task.next_run_at:
                try:
                    next_run = datetime.fromisoformat(task.next_run_at)
                except Exception:
                    next_run = None
            if not next_run:
                next_run = compute_next_run(task.schedule, now)
                update_task_run_info(
                    task.task_id,
                    task.last_run_at,
                    task.last_run_status,
                    task.last_run_message,
                    next_run.isoformat() if next_run else None,
                )
            if next_run and next_run <= now:
                run_task(task)
        time.sleep(30)


if __name__ == "__main__":
    main()
