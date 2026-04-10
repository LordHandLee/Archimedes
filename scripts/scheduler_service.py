from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, time as dtime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from backtest_engine import (
    ACQUISITION_ACTION_GAP_FILL_SECONDARY,
    ACQUISITION_ACTION_DOWNLOAD,
    ACQUISITION_ACTION_INGEST_EXISTING,
    ACQUISITION_ACTION_SKIP_FRESH,
    DEFAULT_ACQUISITION_PROVIDER,
    ResultCatalog,
    build_download_csv_path,
    build_download_dataset_id,
    build_provider_fetch_command,
    decide_acquisition_policy,
    gap_fill_dataset_from_secondary,
    ingest_csv_to_store,
)
from backtest_engine.provider_config import build_provider_runtime_environment

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore[assignment]


DB_PATH = Path("backtests.sqlite")
LOG_DIR = Path("data") / "scheduler_logs"


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
    payload = task.symbols if isinstance(task.symbols, dict) else {}
    symbols = payload.get("symbols", []) if isinstance(payload, dict) else []
    if not symbols:
        return
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    catalog = ResultCatalog(DB_PATH)
    source = str(payload.get("source", DEFAULT_ACQUISITION_PROVIDER) or DEFAULT_ACQUISITION_PROVIDER)
    resolution = str(payload.get("resolution", "1m") or "1m")
    history_window = str(payload.get("history", "2y") or "2y")
    universe_id = str(payload.get("universe_id", "") or "") or None
    universe_name = str(payload.get("universe_name", "") or "") or None
    started_at = datetime.now(timezone.utc).isoformat()
    log_dir = LOG_DIR / task.task_id
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_id}.log"
    add_task_run(run_id, task.task_id, started_at, "running", len(symbols), str(log_path))
    update_task_run_info(task.task_id, started_at, "running", None, None)
    catalog.start_acquisition_run(
        acquisition_run_id=run_id,
        trigger_type="scheduled_task",
        source=source,
        universe_id=universe_id,
        universe_name=universe_name,
        task_id=task.task_id,
        started_at=started_at,
        status="running",
        symbol_count=len(symbols),
        log_path=str(log_path),
        notes=f"Scheduled {source} refresh.",
    )

    error_messages: list[str] = []
    status = "success"
    success_count = 0
    failed_count = 0
    ingested_count = 0
    skipped_count = 0
    provider_env_cache: dict[str, dict[str, str]] = {}
    with log_path.open("a", encoding="utf-8") as log_file:
        for seq, ticker in enumerate(symbols, start=1):
            ticker = str(ticker).strip().upper()
            dataset_id = build_download_dataset_id(
                ticker,
                source=source,
                history_window=history_window,
                resolution=resolution,
            )
            out_path = build_download_csv_path(
                ticker,
                source=source,
                history_window=history_window,
                resolution=resolution,
            )
            decision = decide_acquisition_policy(
                ticker,
                source=source,
                resolution=resolution,
                history_window=history_window,
                catalog=catalog,
                force_refresh=bool(payload.get("force_refresh", False)),
            )
            attempt_started_at = datetime.now(timezone.utc).isoformat()
            log_file.write(
                f"[{datetime.utcnow().isoformat()}] POLICY {ticker} action={decision.action} "
                f"plan={decision.plan_type} reason={decision.reason} "
                f"window={decision.request_start or 'provider_default'}->{decision.request_end or 'provider_default'} "
                f"merge={'yes' if decision.merge_with_existing else 'no'}"
                f"{f' windows={list(decision.request_windows)}' if decision.request_windows else ''}"
                f"{f' secondary_windows={list(decision.secondary_request_windows)}' if decision.secondary_request_windows else ''}"
                f"{f' secondary={decision.secondary_source}:{decision.secondary_dataset_id}' if decision.secondary_dataset_id else ''}\n"
            )
            log_file.flush()
            if decision.action == ACQUISITION_ACTION_SKIP_FRESH:
                skipped_count += 1
                catalog.record_acquisition_attempt(
                    attempt_id=f"{run_id}_{seq:04d}",
                    acquisition_run_id=run_id,
                    seq=seq,
                    source=source,
                    symbol=ticker,
                    dataset_id=dataset_id,
                    status="skipped",
                    started_at=attempt_started_at,
                    finished_at=datetime.now(timezone.utc).isoformat(),
                    csv_path=str(out_path),
                    parquet_path=decision.parquet_path,
                    coverage_start=decision.coverage_start,
                    coverage_end=decision.coverage_end,
                    bar_count=decision.bar_count,
                    ingested=decision.ingested,
                    error_message=decision.reason,
                    log_path=str(log_path),
                    task_id=task.task_id,
                    universe_id=universe_id,
                    resolution=resolution,
                    history_window=history_window,
                )
                continue
            if decision.action == ACQUISITION_ACTION_INGEST_EXISTING:
                try:
                    artifact = ingest_csv_to_store(
                        out_path,
                        dataset_id=dataset_id,
                        merge_existing=bool(decision.merge_with_existing),
                    )
                    success_count += 1
                    ingested_count += 1
                    catalog.record_acquisition_attempt(
                        attempt_id=f"{run_id}_{seq:04d}",
                        acquisition_run_id=run_id,
                        seq=seq,
                        source=source,
                        symbol=ticker,
                        dataset_id=dataset_id,
                        status="ingested",
                        started_at=attempt_started_at,
                        finished_at=datetime.now(timezone.utc).isoformat(),
                        csv_path=str(out_path),
                        parquet_path=artifact.parquet_path,
                        coverage_start=artifact.start,
                        coverage_end=artifact.end,
                        bar_count=artifact.bar_count,
                        ingested=True,
                        error_message=decision.reason,
                        log_path=str(log_path),
                        task_id=task.task_id,
                        universe_id=universe_id,
                        resolution=resolution,
                        history_window=history_window,
                    )
                except Exception as exc:
                    failed_count += 1
                    attempt_error = f"{ticker} ingest failed: {exc}"
                    error_messages.append(attempt_error)
                    log_file.write(f"[{datetime.utcnow().isoformat()}] INGEST_ERROR {ticker}: {exc}\n")
                    log_file.flush()
                    catalog.record_acquisition_attempt(
                        attempt_id=f"{run_id}_{seq:04d}",
                        acquisition_run_id=run_id,
                        seq=seq,
                        source=source,
                        symbol=ticker,
                        dataset_id=dataset_id,
                        status="ingest_error",
                        started_at=attempt_started_at,
                        finished_at=datetime.now(timezone.utc).isoformat(),
                        csv_path=str(out_path),
                        parquet_path=decision.parquet_path,
                        coverage_start=decision.coverage_start,
                        coverage_end=decision.coverage_end,
                        bar_count=decision.bar_count,
                        ingested=False,
                        error_message=attempt_error,
                        log_path=str(log_path),
                        task_id=task.task_id,
                        universe_id=universe_id,
                        resolution=resolution,
                        history_window=history_window,
                    )
                continue
            if decision.action == ACQUISITION_ACTION_GAP_FILL_SECONDARY:
                try:
                    artifact = None
                    windows = list(decision.request_windows) or [(decision.request_start, decision.request_end)]
                    for window_start, window_end in windows:
                        artifact = gap_fill_dataset_from_secondary(
                            dataset_id,
                            str(decision.secondary_dataset_id or ""),
                            start=window_start,
                            end=window_end,
                        )
                    success_count += 1
                    ingested_count += 1
                    catalog.record_acquisition_attempt(
                        attempt_id=f"{run_id}_{seq:04d}",
                        acquisition_run_id=run_id,
                        seq=seq,
                        source=source,
                        symbol=ticker,
                        dataset_id=dataset_id,
                        status="gap_filled",
                        started_at=attempt_started_at,
                        finished_at=datetime.now(timezone.utc).isoformat(),
                        csv_path=str(out_path),
                        parquet_path=artifact.parquet_path if artifact is not None else decision.parquet_path,
                        coverage_start=artifact.start if artifact is not None else decision.coverage_start,
                        coverage_end=artifact.end if artifact is not None else decision.coverage_end,
                        bar_count=artifact.bar_count if artifact is not None else decision.bar_count,
                        ingested=True,
                        error_message=(
                            f"{decision.reason} Secondary source: {decision.secondary_source or '—'} "
                            f"dataset={decision.secondary_dataset_id or '—'} "
                            f"parity={decision.parity_state} overlap={decision.parity_overlap_bars}"
                        ),
                        log_path=str(log_path),
                        task_id=task.task_id,
                        universe_id=universe_id,
                        resolution=resolution,
                        history_window=history_window,
                    )
                except Exception as exc:
                    failed_count += 1
                    attempt_error = f"{ticker} cross-source gap fill failed: {exc}"
                    error_messages.append(attempt_error)
                    log_file.write(f"[{datetime.utcnow().isoformat()}] GAP_FILL_ERROR {ticker}: {exc}\n")
                    log_file.flush()
                    catalog.record_acquisition_attempt(
                        attempt_id=f"{run_id}_{seq:04d}",
                        acquisition_run_id=run_id,
                        seq=seq,
                        source=source,
                        symbol=ticker,
                        dataset_id=dataset_id,
                        status="gap_fill_error",
                        started_at=attempt_started_at,
                        finished_at=datetime.now(timezone.utc).isoformat(),
                        csv_path=str(out_path),
                        parquet_path=decision.parquet_path,
                        coverage_start=decision.coverage_start,
                        coverage_end=decision.coverage_end,
                        bar_count=decision.bar_count,
                        ingested=False,
                        error_message=attempt_error,
                        log_path=str(log_path),
                        task_id=task.task_id,
                        universe_id=universe_id,
                        resolution=resolution,
                        history_window=history_window,
                    )
                continue
            if decision.secondary_dataset_id and decision.secondary_request_windows:
                try:
                    for window_start, window_end in list(decision.secondary_request_windows):
                        gap_fill_dataset_from_secondary(
                            dataset_id,
                            str(decision.secondary_dataset_id or ""),
                            start=window_start,
                            end=window_end,
                        )
                    log_file.write(
                        f"[{datetime.utcnow().isoformat()}] SECONDARY_GAP_FILL {ticker} "
                        f"{list(decision.secondary_request_windows)} via {decision.secondary_source}:{decision.secondary_dataset_id}\n"
                    )
                    log_file.flush()
                except Exception as exc:
                    failed_count += 1
                    attempt_error = f"{ticker} secondary gap fill failed: {exc}"
                    error_messages.append(attempt_error)
                    log_file.write(f"[{datetime.utcnow().isoformat()}] GAP_FILL_ERROR {ticker}: {exc}\n")
                    log_file.flush()
                    catalog.record_acquisition_attempt(
                        attempt_id=f"{run_id}_{seq:04d}",
                        acquisition_run_id=run_id,
                        seq=seq,
                        source=source,
                        symbol=ticker,
                        dataset_id=dataset_id,
                        status="gap_fill_error",
                        started_at=attempt_started_at,
                        finished_at=datetime.now(timezone.utc).isoformat(),
                        csv_path=str(out_path),
                        parquet_path=decision.parquet_path,
                        coverage_start=decision.coverage_start,
                        coverage_end=decision.coverage_end,
                        bar_count=decision.bar_count,
                        ingested=False,
                        error_message=attempt_error,
                        log_path=str(log_path),
                        task_id=task.task_id,
                        universe_id=universe_id,
                        resolution=resolution,
                        history_window=history_window,
                    )
                    continue
            extra_args: list[str] = []
            windows = list(decision.request_windows) or [(decision.request_start, decision.request_end)]
            artifact = None
            attempt_status = "ingested"
            attempt_error = None
            for window_idx, (window_start, window_end) in enumerate(windows, start=1):
                extra_args = []
                if window_start:
                    extra_args.extend(["--start", window_start])
                if window_end:
                    extra_args.extend(["--end", window_end])
                cmd = build_provider_fetch_command(
                    source,
                    python_executable=sys.executable,
                    ticker=ticker,
                    out_path=out_path,
                    resolution=resolution,
                    history_window=history_window,
                    progress=True,
                    resume=True,
                    extra_args=extra_args or None,
                )
                log_file.write(
                    f"[{datetime.utcnow().isoformat()}] START {ticker} window {window_idx}/{len(windows)} "
                    f"{window_start or 'provider_default'}->{window_end or 'provider_default'}\n"
                )
                log_file.flush()
                try:
                    runtime_env = provider_env_cache.setdefault(
                        source,
                        build_provider_runtime_environment(source, catalog=catalog),
                    )
                    proc = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        check=True,
                        env={**os.environ, **runtime_env},
                    )
                    if proc.stdout:
                        log_file.write(proc.stdout + "\n")
                    if proc.stderr:
                        log_file.write(proc.stderr + "\n")
                except subprocess.CalledProcessError as exc:
                    failed_count += 1
                    attempt_status = "download_error"
                    attempt_error = f"{ticker} failed: {exc}"
                    error_messages.append(attempt_error)
                    if exc.stdout:
                        log_file.write(exc.stdout + "\n")
                    if exc.stderr:
                        log_file.write(exc.stderr + "\n")
                    log_file.write(f"[{datetime.utcnow().isoformat()}] ERROR {ticker}\n")
                    log_file.flush()
                    break
                try:
                    artifact = ingest_csv_to_store(
                        out_path,
                        dataset_id=dataset_id,
                        merge_existing=bool(decision.merge_with_existing or window_idx > 1 or len(windows) > 1),
                    )
                except Exception as exc:
                    failed_count += 1
                    attempt_status = "ingest_error"
                    attempt_error = f"{ticker} ingest failed: {exc}"
                    error_messages.append(attempt_error)
                    log_file.write(f"[{datetime.utcnow().isoformat()}] INGEST_ERROR {ticker}: {exc}\n")
                    log_file.flush()
                    break
            if artifact is not None and attempt_error is None:
                success_count += 1
                ingested_count += 1
            catalog.record_acquisition_attempt(
                attempt_id=f"{run_id}_{seq:04d}",
                acquisition_run_id=run_id,
                seq=seq,
                source=source,
                symbol=ticker,
                dataset_id=dataset_id,
                status=attempt_status,
                started_at=attempt_started_at,
                finished_at=datetime.now(timezone.utc).isoformat(),
                csv_path=str(out_path),
                parquet_path=artifact.parquet_path if artifact is not None else None,
                coverage_start=artifact.start if artifact is not None else None,
                coverage_end=artifact.end if artifact is not None else None,
                bar_count=artifact.bar_count if artifact is not None else None,
                ingested=artifact is not None,
                error_message=attempt_error,
                log_path=str(log_path),
                task_id=task.task_id,
                universe_id=universe_id,
                resolution=resolution,
                history_window=history_window,
            )
            log_file.write(f"[{datetime.utcnow().isoformat()}] DONE {ticker}\n")
            log_file.flush()

    finished_at = datetime.now(timezone.utc).isoformat()
    if skipped_count and not failed_count and not success_count:
        status = "skipped"
    elif failed_count and success_count:
        status = "partial"
    elif failed_count and not success_count:
        status = "failed"
    elif error_messages:
        status = "partial"
    last_message = "; ".join(error_messages[:3]) if error_messages else (
        f"{skipped_count} symbol(s) skipped as fresh." if skipped_count else None
    )
    finish_task_run(run_id, finished_at, status, last_message)
    catalog.finish_acquisition_run(
        run_id,
        finished_at=finished_at,
        status=status,
        success_count=success_count,
        failed_count=failed_count,
        ingested_count=ingested_count,
        notes=last_message or f"Completed {success_count} scheduled download(s), ingested {ingested_count}, skipped {skipped_count}.",
    )
    next_run = compute_next_run(task.schedule, datetime.now(timezone.utc))
    update_task_run_info(
        task.task_id,
        finished_at,
        status,
        last_message,
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
