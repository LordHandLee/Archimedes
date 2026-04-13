"""
Fetch historical bars from Interactive Brokers TWS / IB Gateway and save to CSV.

Usage:
    export IB_HOST=127.0.0.1
    export IB_PORT=7497
    export IB_CLIENT_ID=9301
    python scripts/fetch_interactive_brokers.py AAPL --out data/AAPL_interactive_brokers_10y_1m.csv

Official Interactive Brokers TWS API notes that matter here:
- the historical-data request API is `reqHistoricalData`
- the old hard historical limits for `1 min` and larger bars have been lifted
- large requests are still subject to soft throttling
- `reqHeadTimestamp` can be used to discover the earliest available history per contract

This script intentionally uses chunked requests for minute bars. That chunking is a
client-side safety choice for stability and throughput, not a confirmed IB hard limit.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Callable, Optional

import pandas as pd

try:  # pragma: no cover - optional dependency
    from ibapi.client import EClient
    from ibapi.contract import Contract
    from ibapi.wrapper import EWrapper
except Exception as exc:  # pragma: no cover - exercised through runtime error path
    class _MissingEClient:  # pragma: no cover
        pass

    class _MissingEWrapper:  # pragma: no cover
        pass

    class _MissingContract:  # pragma: no cover
        pass

    EClient = _MissingEClient  # type: ignore[assignment]
    EWrapper = _MissingEWrapper  # type: ignore[assignment]
    Contract = _MissingContract  # type: ignore[assignment]
    _IBAPI_IMPORT_ERROR = exc
else:
    _IBAPI_IMPORT_ERROR = None


_BAR_SIZE_MAP = {
    "1m": "1 min",
    "2m": "2 mins",
    "3m": "3 mins",
    "5m": "5 mins",
    "10m": "10 mins",
    "15m": "15 mins",
    "20m": "20 mins",
    "30m": "30 mins",
    "1h": "1 hour",
    "2h": "2 hours",
    "3h": "3 hours",
    "4h": "4 hours",
    "8h": "8 hours",
    "1d": "1 day",
    "1w": "1 week",
    "1mo": "1 month",
}

_RETRYABLE_HISTORICAL_ERROR_SNIPPETS = (
    "hmds server disconnect occurred",
    "attempting reconnection",
    "hmds data farm connection is broken",
    "data farm connection is broken",
    "pacing violation",
)


def _normalize_bar_size(value: str) -> str:
    normalized = " ".join(str(value or "").strip().lower().split())
    normalized = (
        normalized.replace("minutes", "mins")
        .replace("minute", "min")
        .replace("hours", "hour")
        .replace("days", "day")
        .replace("weeks", "week")
        .replace("months", "month")
    )
    if normalized in _BAR_SIZE_MAP:
        return _BAR_SIZE_MAP[normalized]
    native_sizes = {native.lower(): native for native in _BAR_SIZE_MAP.values()}
    if normalized in native_sizes:
        return native_sizes[normalized]
    valid = ", ".join(sorted(_BAR_SIZE_MAP))
    raise ValueError(f"Unsupported Interactive Brokers bar size '{value}'. Valid values: {valid}")


def _history_to_offset(value: str) -> pd.DateOffset:
    text = str(value or "").strip().lower()
    if not text:
        raise ValueError("History window is required.")
    digits = "".join(ch for ch in text if ch.isdigit())
    unit = "".join(ch for ch in text if ch.isalpha())
    if not digits or not unit:
        raise ValueError(f"Invalid history window '{value}'. Expected formats like 10y, 6m, 30d.")
    amount = int(digits)
    if amount <= 0:
        raise ValueError("History window must be positive.")
    if unit == "y":
        return pd.DateOffset(years=amount)
    if unit == "m":
        return pd.DateOffset(months=amount)
    if unit == "w":
        return pd.DateOffset(weeks=amount)
    if unit == "d":
        return pd.DateOffset(days=amount)
    raise ValueError(f"Unsupported history unit '{unit}' in '{value}'.")


def _duration_string_for_window(start: pd.Timestamp, end: pd.Timestamp) -> str:
    delta = max(end - start, pd.Timedelta(seconds=1))
    total_days = max(delta.total_seconds() / 86400.0, 1.0)
    if total_days >= 365:
        return f"{max(1, math.ceil(total_days / 365.0))} Y"
    if total_days >= 30:
        return f"{max(1, math.ceil(total_days / 30.0))} M"
    if total_days >= 7:
        return f"{max(1, math.ceil(total_days / 7.0))} W"
    return f"{max(1, math.ceil(total_days))} D"


def _default_chunk_offset(bar_size: str) -> pd.DateOffset:
    normalized = _normalize_bar_size(bar_size)
    if normalized == "1 min":
        # Live gateway benchmarking showed a sharp slowdown once 1-minute requests
        # grew beyond roughly one week, so default to weekly windows here.
        return pd.DateOffset(weeks=1)
    if "min" in normalized:
        return pd.DateOffset(months=6)
    if "hour" in normalized:
        return pd.DateOffset(years=1)
    if normalized in {"1 day", "1 week", "1 month"}:
        return pd.DateOffset(years=5)
    return pd.DateOffset(years=1)


def _chunk_windows(
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    chunk_offset: pd.DateOffset,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if end <= start:
        return [(start, end)]
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = start
    while cursor < end:
        next_cursor = min(cursor + chunk_offset, end)
        windows.append((cursor, next_cursor))
        cursor = next_cursor
    return windows


def _parse_chunk_duration(value: str | None, bar_size: str) -> pd.DateOffset:
    if not value:
        return _default_chunk_offset(bar_size)
    normalized = str(value).strip().replace(" ", "").lower()
    return _history_to_offset(normalized)


def _minimum_retry_window(bar_size: str) -> pd.Timedelta:
    normalized = _normalize_bar_size(bar_size)
    if normalized == "1 min":
        return pd.Timedelta(days=7)
    if "min" in normalized:
        return pd.Timedelta(days=14)
    if "hour" in normalized:
        return pd.Timedelta(days=30)
    return pd.Timedelta(days=180)


def _split_window_midpoint(start: pd.Timestamp, end: pd.Timestamp) -> pd.Timestamp:
    midpoint = start + ((end - start) / 2)
    return _ensure_utc_timestamp(midpoint)


def _is_benign_historical_error(error_code: int, error_text: str) -> bool:
    normalized = str(error_text or "").lower()
    return (int(error_code) == 162 and "query cancelled" in normalized) or (
        int(error_code) == 366 and "no historical data query found" in normalized
    )


def _is_retryable_historical_exception(exc: BaseException) -> bool:
    normalized = str(exc or "").lower()
    return any(snippet in normalized for snippet in _RETRYABLE_HISTORICAL_ERROR_SNIPPETS)


def _retry_backoff_seconds(attempt: int) -> float:
    return min(30.0, 3.0 * (2**attempt))


def _request_window_with_retries(
    request_window: Callable[[pd.Timestamp, pd.Timestamp], _HistoricalResult],
    *,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    max_attempts: int = 3,
) -> _HistoricalResult:
    last_exc: BaseException | None = None
    for attempt in range(max(1, int(max_attempts))):
        try:
            return request_window(window_start, window_end)
        except TimeoutError as exc:
            last_exc = exc
        except RuntimeError as exc:
            if not _is_retryable_historical_exception(exc):
                raise
            last_exc = exc
        if attempt + 1 < max_attempts:
            time.sleep(_retry_backoff_seconds(attempt))
    if last_exc is None:
        raise RuntimeError("Historical data request failed without a captured exception.")
    raise last_exc


def _fetch_window_with_adaptive_retry(
    request_window: Callable[[pd.Timestamp, pd.Timestamp], _HistoricalResult],
    *,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    bar_size: str,
) -> list[_HistoricalResult]:
    try:
        return [
            _request_window_with_retries(
                request_window,
                window_start=window_start,
                window_end=window_end,
            )
        ]
    except TimeoutError as exc:
        span = window_end - window_start
        min_span = _minimum_retry_window(bar_size)
        if span <= min_span:
            raise TimeoutError(
                f"{exc} Request window {window_start.isoformat()} -> {window_end.isoformat()} "
                f"timed out even at the minimum retry span of {min_span}."
            ) from exc
        midpoint = _split_window_midpoint(window_start, window_end)
        if midpoint <= window_start or midpoint >= window_end:
            raise TimeoutError(
                f"{exc} Request window {window_start.isoformat()} -> {window_end.isoformat()} "
                "could not be split any smaller."
            ) from exc
        left_results = _fetch_window_with_adaptive_retry(
            request_window,
            window_start=window_start,
            window_end=midpoint,
            bar_size=bar_size,
        )
        right_results = _fetch_window_with_adaptive_retry(
            request_window,
            window_start=midpoint,
            window_end=window_end,
            bar_size=bar_size,
        )
        return left_results + right_results


def _ib_end_datetime(value: pd.Timestamp) -> str:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp.strftime("%Y%m%d %H:%M:%S UTC")


def _ensure_utc_timestamp(value: dt.datetime | pd.Timestamp | str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        return stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC")


def _parse_ib_bar_time(value: str) -> pd.Timestamp:
    text = str(value or "").strip()
    if not text:
        raise ValueError("Interactive Brokers returned an empty bar timestamp.")
    if text.isdigit():
        if len(text) == 8:
            return pd.to_datetime(text, format="%Y%m%d", utc=True)
        return pd.to_datetime(int(text), unit="s", utc=True)
    parsed = pd.to_datetime(text, utc=True, errors="raise")
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    return parsed.tz_convert("UTC")


def _load_checkpoint_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    if frame.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"])
    frame = frame.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")
    frame = frame.set_index("timestamp")
    return frame


def _append_checkpoint_rows(path: Path, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = frame.reset_index().rename(columns={"index": "timestamp"})
    write_header = not path.exists() or path.stat().st_size == 0
    ordered.to_csv(path, mode="a", index=False, header=write_header)


def _advance_resume_start(last_timestamp: pd.Timestamp) -> pd.Timestamp:
    return _ensure_utc_timestamp(last_timestamp) + pd.Timedelta(seconds=1)


def _build_contract(
    ticker: str,
    *,
    sec_type: str,
    exchange: str,
    currency: str,
    primary_exchange: str | None,
) -> Contract:
    contract = Contract()
    contract.symbol = str(ticker).strip().upper()
    contract.secType = str(sec_type).strip().upper()
    contract.exchange = str(exchange).strip().upper()
    contract.currency = str(currency).strip().upper()
    if primary_exchange:
        contract.primaryExchange = str(primary_exchange).strip().upper()
    return contract


@dataclass
class _HistoricalResult:
    rows: list[dict]
    start: str
    end: str


class _IBHistoricalApp(EWrapper, EClient):
    def __init__(self) -> None:
        EWrapper.__init__(self)
        EClient.__init__(self, self)
        self._connected = threading.Event()
        self._responses: dict[int, Queue] = {}
        self._req_id = 9000

    def nextValidId(self, orderId: int) -> None:  # noqa: N802
        self._connected.set()

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):  # noqa: N802
        queue = self._responses.get(int(reqId))
        if _is_benign_historical_error(int(errorCode), str(errorString or "")):
            return
        message = RuntimeError(f"Interactive Brokers error {errorCode} for request {reqId}: {errorString}")
        if queue is not None:
            queue.put(("error", message))
        elif int(errorCode) not in {2104, 2105, 2106, 2158}:  # common connection/status noise
            sys.stderr.write(f"{message}\n")

    def historicalData(self, reqId, bar):  # noqa: N802
        queue = self._responses.get(int(reqId))
        if queue is not None:
            queue.put(("bar", bar))

    def historicalDataEnd(self, reqId, start, end):  # noqa: N802
        queue = self._responses.get(int(reqId))
        if queue is not None:
            queue.put(("end", (start, end)))

    def headTimestamp(self, reqId, headTimestamp):  # noqa: N802
        queue = self._responses.get(int(reqId))
        if queue is not None:
            queue.put(("head", headTimestamp))

    def connect_and_start(self, *, host: str, port: int, client_id: int, timeout_seconds: float) -> threading.Thread:
        self.connect(host, port, client_id)
        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()
        if not self._connected.wait(timeout_seconds):
            raise TimeoutError("Timed out waiting for Interactive Brokers connection.")
        return thread

    def _next_req_id(self) -> int:
        self._req_id += 1
        return self._req_id

    def request_head_timestamp(
        self,
        *,
        contract: Contract,
        what_to_show: str,
        use_rth: bool,
        timeout_seconds: float,
    ) -> str:
        req_id = self._next_req_id()
        queue: Queue = Queue()
        self._responses[req_id] = queue
        self.reqHeadTimeStamp(req_id, contract, what_to_show, int(use_rth), 2)
        try:
            while True:
                kind, payload = queue.get(timeout=timeout_seconds)
                if kind == "error":
                    raise payload
                if kind == "head":
                    return str(payload)
        except Empty as exc:
            raise TimeoutError("Timed out waiting for Interactive Brokers head timestamp.") from exc
        finally:
            self.cancelHeadTimeStamp(req_id)
            self._responses.pop(req_id, None)

    def request_historical_window(
        self,
        *,
        contract: Contract,
        end: pd.Timestamp,
        duration: str,
        bar_size: str,
        what_to_show: str,
        use_rth: bool,
        timeout_seconds: float,
    ) -> _HistoricalResult:
        req_id = self._next_req_id()
        queue: Queue = Queue()
        self._responses[req_id] = queue
        rows: list[dict] = []
        received_start = ""
        received_end = ""
        completed = False
        self.reqHistoricalData(
            req_id,
            contract,
            _ib_end_datetime(end),
            duration,
            bar_size,
            what_to_show,
            int(use_rth),
            2,
            False,
            [],
        )
        try:
            while True:
                kind, payload = queue.get(timeout=timeout_seconds)
                if kind == "error":
                    raise payload
                if kind == "bar":
                    rows.append(
                        {
                            "timestamp": _parse_ib_bar_time(payload.date),
                            "open": float(payload.open),
                            "high": float(payload.high),
                            "low": float(payload.low),
                            "close": float(payload.close),
                            "volume": float(payload.volume),
                        }
                    )
                    continue
                if kind == "end":
                    received_start, received_end = payload
                    completed = True
                    break
        except Empty as exc:
            raise TimeoutError("Timed out waiting for Interactive Brokers historical data.") from exc
        finally:
            if not completed:
                self.cancelHistoricalData(req_id)
            self._responses.pop(req_id, None)
        return _HistoricalResult(rows=rows, start=received_start, end=received_end)


def fetch_bars(
    ticker: str,
    *,
    start: dt.datetime | None,
    end: dt.datetime | None,
    history_window: str,
    resolution: str,
    host: str,
    port: int,
    client_id: int,
    sec_type: str,
    exchange: str,
    currency: str,
    primary_exchange: str | None,
    what_to_show: str,
    use_rth: bool,
    chunk_duration: str | None,
    pace_seconds: float,
    timeout_seconds: float,
    discover_head_timestamp: bool = False,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    checkpoint_path: Path | None = None,
    resume: bool = False,
) -> pd.DataFrame:
    if _IBAPI_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Interactive Brokers provider requires the official 'ibapi' Python package and a running TWS/IB Gateway "
            "session with API access enabled."
        ) from _IBAPI_IMPORT_ERROR

    bar_size = _normalize_bar_size(resolution)
    end_ts = _ensure_utc_timestamp(end or dt.datetime.now(dt.timezone.utc))
    if start is None:
        start_ts = end_ts - _history_to_offset(history_window)
    else:
        start_ts = _ensure_utc_timestamp(start)
    requested_start_ts = start_ts
    if start_ts >= end_ts:
        raise ValueError("Start time must be before end time.")

    checkpoint_frame = pd.DataFrame()
    last_checkpoint_ts: pd.Timestamp | None = None
    checkpoint_row_count = 0
    if checkpoint_path is not None and checkpoint_path.exists() and not resume:
        checkpoint_path.unlink()
    if checkpoint_path is not None and resume and checkpoint_path.exists():
        checkpoint_frame = _load_checkpoint_frame(checkpoint_path)
        if not checkpoint_frame.empty:
            last_checkpoint_ts = _ensure_utc_timestamp(checkpoint_frame.index.max())
            checkpoint_row_count = int(len(checkpoint_frame))
            start_ts = max(start_ts, _advance_resume_start(last_checkpoint_ts))
            if progress_cb:
                progress_cb(0, checkpoint_row_count)
    if start_ts >= end_ts:
        if checkpoint_path is not None and checkpoint_path.exists():
            frame = _load_checkpoint_frame(checkpoint_path)
            return frame.loc[(frame.index >= requested_start_ts) & (frame.index <= end_ts)]
        if not checkpoint_frame.empty:
            return checkpoint_frame.loc[(checkpoint_frame.index >= requested_start_ts) & (checkpoint_frame.index <= end_ts)]
        raise RuntimeError("Requested range is already fully covered by the existing checkpoint.")

    contract = _build_contract(
        ticker,
        sec_type=sec_type,
        exchange=exchange,
        currency=currency,
        primary_exchange=primary_exchange,
    )
    app = _IBHistoricalApp()
    thread = app.connect_and_start(host=host, port=port, client_id=client_id, timeout_seconds=timeout_seconds)
    try:
        if discover_head_timestamp:
            try:
                head_value = app.request_head_timestamp(
                    contract=contract,
                    what_to_show=what_to_show,
                    use_rth=use_rth,
                    timeout_seconds=timeout_seconds,
                )
                head_ts = _parse_ib_bar_time(head_value)
                if head_ts > start_ts:
                    start_ts = head_ts
            except Exception:
                # Keep the requested start if head-timestamp discovery fails; the main historical requests may still work.
                pass

        windows = _chunk_windows(
            start=start_ts,
            end=end_ts,
            chunk_offset=_parse_chunk_duration(chunk_duration, bar_size),
        )
        all_rows: list[dict] = []
        completed_requests = 0

        def _request_window(window_start: pd.Timestamp, window_end: pd.Timestamp) -> _HistoricalResult:
            duration = _duration_string_for_window(window_start, window_end)
            return app.request_historical_window(
                contract=contract,
                end=window_end,
                duration=duration,
                bar_size=bar_size,
                what_to_show=what_to_show,
                use_rth=use_rth,
                timeout_seconds=timeout_seconds,
            )

        for index, (window_start, window_end) in enumerate(windows, start=1):
            window_results = _fetch_window_with_adaptive_retry(
                _request_window,
                window_start=window_start,
                window_end=window_end,
                bar_size=bar_size,
            )
            for result in window_results:
                completed_requests += 1
                result_rows: list[dict] = []
                for row in result.rows:
                    stamp = row["timestamp"]
                    if stamp < window_start or stamp > window_end:
                        continue
                    if last_checkpoint_ts is not None and stamp <= last_checkpoint_ts:
                        continue
                    result_rows.append(row)
                if checkpoint_path is not None:
                    if result_rows:
                        checkpoint_chunk = pd.DataFrame(result_rows)
                        checkpoint_chunk["timestamp"] = pd.to_datetime(
                            checkpoint_chunk["timestamp"], utc=True, errors="coerce"
                        )
                        checkpoint_chunk = checkpoint_chunk.dropna(subset=["timestamp"])
                        checkpoint_chunk = checkpoint_chunk.drop_duplicates(subset=["timestamp"], keep="last")
                        checkpoint_chunk = checkpoint_chunk.sort_values("timestamp").set_index("timestamp")
                        _append_checkpoint_rows(checkpoint_path, checkpoint_chunk)
                        last_checkpoint_ts = _ensure_utc_timestamp(checkpoint_chunk.index.max())
                        checkpoint_row_count += int(len(checkpoint_chunk))
                else:
                    all_rows.extend(result_rows)
                if progress_cb:
                    progress_rows = len(all_rows)
                    if checkpoint_path is not None:
                        progress_rows = checkpoint_row_count
                    progress_cb(completed_requests, progress_rows)
            if pace_seconds > 0 and index < len(windows):
                time.sleep(pace_seconds)
    finally:
        try:
            app.disconnect()
        finally:
            thread.join(timeout=1.0)

    if checkpoint_path is not None and checkpoint_path.exists():
        frame = _load_checkpoint_frame(checkpoint_path)
        frame = frame.loc[(frame.index >= requested_start_ts) & (frame.index <= end_ts)]
        frame.reset_index().to_csv(checkpoint_path, index=False)
    else:
        if not all_rows:
            raise RuntimeError("Interactive Brokers returned no historical bars for this request.")
        frame = pd.DataFrame(all_rows)
        frame = frame.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")
        frame = frame.set_index("timestamp")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Download historical bars from Interactive Brokers TWS / IB Gateway.")
    parser.add_argument("ticker", help="Ticker symbol, e.g., AAPL")
    parser.add_argument("--out", type=Path, default=Path("data") / "prices_ibkr.csv")
    parser.add_argument("--start", type=str, help="Optional start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, help="Optional end date YYYY-MM-DD")
    parser.add_argument("--resolution", type=str, default="1m", help="Bar size alias such as 1m, 5m, 1h, 1d.")
    parser.add_argument("--history-window", type=str, default="10y", help="Fallback history window if --start is omitted.")
    parser.add_argument("--host", type=str, default=os.getenv("IB_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("IB_PORT", "7497")))
    parser.add_argument("--client-id", type=int, default=int(os.getenv("IB_CLIENT_ID", "9301")))
    parser.add_argument("--sec-type", type=str, default="STK")
    parser.add_argument("--exchange", type=str, default="SMART")
    parser.add_argument("--primary-exchange", type=str, default=os.getenv("IB_PRIMARY_EXCHANGE", ""))
    parser.add_argument("--currency", type=str, default="USD")
    parser.add_argument("--what-to-show", type=str, default="TRADES")
    parser.add_argument("--use-rth", action="store_true", help="Request only regular trading hours.")
    parser.add_argument("--chunk-duration", type=str, default=None, help="Optional client-side chunk duration like 1y or 6m.")
    parser.add_argument("--discover-head-timestamp", action="store_true", help="Ask IB for the earliest available bar timestamp before downloading.")
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--progress", action="store_true", help="Emit JSON progress to stdout.")
    parser.add_argument("--resume", action="store_true", help="Accepted for compatibility; currently ignored.")
    parser.add_argument("--pace", type=float, default=1.0, help="Seconds to pause between chunked IB requests.")
    args = parser.parse_args()

    start_dt = dt.datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc) if args.start else None
    end_dt = dt.datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc) if args.end else None
    if start_dt and end_dt and start_dt > end_dt:
        raise SystemExit("Start date must be before end date.")

    def emit_progress(page_count: int, row_count: int) -> None:
        if not args.progress:
            return
        print(
            json.dumps(
                {
                    "type": "progress",
                    "ticker": args.ticker.upper(),
                    "pages": page_count,
                    "rows": row_count,
                }
            ),
            flush=True,
        )

    try:
        if args.progress:
            print(
                json.dumps(
                    {
                        "type": "start",
                        "ticker": args.ticker.upper(),
                        "start": args.start or f"{args.history_window} back",
                        "end": args.end or "now",
                        "resolution": args.resolution,
                        "provider": "interactive_brokers",
                    }
                ),
                flush=True,
            )
        frame = fetch_bars(
            args.ticker.upper(),
            start=start_dt,
            end=end_dt,
            history_window=args.history_window,
            resolution=args.resolution,
            host=args.host,
            port=args.port,
            client_id=args.client_id,
            sec_type=args.sec_type,
            exchange=args.exchange,
            currency=args.currency,
            primary_exchange=args.primary_exchange or None,
            what_to_show=args.what_to_show.upper(),
            use_rth=bool(args.use_rth),
            chunk_duration=args.chunk_duration,
            pace_seconds=float(args.pace),
            timeout_seconds=float(args.timeout_seconds),
            discover_head_timestamp=bool(args.discover_head_timestamp),
            progress_cb=emit_progress,
            checkpoint_path=args.out,
            resume=bool(args.resume),
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        frame.reset_index().to_csv(args.out, index=False)
        if args.progress:
            print(
                json.dumps(
                    {
                        "type": "done",
                        "ticker": args.ticker.upper(),
                        "rows": int(len(frame)),
                        "out": str(args.out),
                    }
                ),
                flush=True,
            )
        else:
            print(f"Saved {len(frame)} rows to {args.out}")
    except Exception as exc:
        details = traceback.format_exc()
        message = str(exc)
        if args.out.exists():
            try:
                checkpoint_rows = len(_load_checkpoint_frame(args.out))
            except Exception:
                checkpoint_rows = 0
            if checkpoint_rows > 0:
                checkpoint_note = f"Partial progress was saved to {args.out} ({checkpoint_rows} rows)."
                message = f"{message} [{checkpoint_note}]"
                details = f"{details.rstrip()}\n\n{checkpoint_note}\n"
        if args.progress:
            print(
                json.dumps(
                    {
                        "type": "error",
                        "ticker": args.ticker.upper(),
                        "message": message,
                        "details": details,
                    }
                ),
                flush=True,
            )
        else:  # pragma: no cover
            print(details or str(exc), file=sys.stderr, flush=True)
        raise SystemExit(1) from exc


if __name__ == "__main__":  # pragma: no cover
    main()
