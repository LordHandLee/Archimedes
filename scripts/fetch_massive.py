"""
Fetch 2 years of 1-minute bars from the Massive (Polygon) API for a given ticker and save to CSV.

Usage:
    export MASSIVE_API_KEY="your_api_key"
    python scripts/fetch_massive.py AAPL --out data/AAPL_2y_1m.csv

Notes:
- Massive (Polygon) free plan allows ~2 years of minute aggregates.
- We request adjusted data, ascending, and paginate using the `next_url` cursor.
- This script only downloads data; integration with data_loader/backtests is separate.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
try:
    import requests
except Exception as exc:  # pragma: no cover
    requests = None  # type: ignore[assignment]
    _REQUESTS_IMPORT_ERROR = exc
else:
    _REQUESTS_IMPORT_ERROR = None

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


API_HOST = "https://api.polygon.io"
API_KEY_ENV = "MASSIVE_API_KEY"
# Temporary migration note: keep the previous key commented here so it can be copied manually if needed.
# LEGACY_MASSIVE_API_KEY = "AAw8ohj8iAa7ENJ9YFpMmjMBbAZZhGVF"
DEFAULT_PACE_SECONDS = 15.0
DEFAULT_429_RETRIES = 12
DEFAULT_429_BACKOFF_SECONDS = 15.0
DEFAULT_429_MAX_SLEEP_SECONDS = 120.0
DEFAULT_JSON_RETRIES = 4
DEFAULT_JSON_BACKOFF_SECONDS = 5.0
DEFAULT_JSON_MAX_SLEEP_SECONDS = 60.0


def _iso_date(dt_obj: dt.datetime) -> str:
    return dt_obj.strftime("%Y-%m-%d")


def _retry_after_seconds(value: str | None) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        seconds = float(text)
    except Exception:
        return None
    return seconds if seconds >= 0 else None


def _fetch_page(
    url: str,
    api_key: str,
    *,
    max_429_retries: int = DEFAULT_429_RETRIES,
    rate_limit_backoff_seconds: float = DEFAULT_429_BACKOFF_SECONDS,
    rate_limit_max_sleep_seconds: float = DEFAULT_429_MAX_SLEEP_SECONDS,
    max_json_retries: int = DEFAULT_JSON_RETRIES,
    json_backoff_seconds: float = DEFAULT_JSON_BACKOFF_SECONDS,
    json_max_sleep_seconds: float = DEFAULT_JSON_MAX_SLEEP_SECONDS,
) -> Dict[str, Any]:
    if requests is None:
        raise RuntimeError(
            "Missing Python dependency: requests. Install project requirements before downloading data."
        ) from _REQUESTS_IMPORT_ERROR
    headers = {"Authorization": f"Bearer {api_key}"}
    rate_limit_attempt = 0
    json_attempt = 0
    backoff = max(1.0, float(rate_limit_backoff_seconds))
    max_sleep = max(backoff, float(rate_limit_max_sleep_seconds))
    json_backoff = max(1.0, float(json_backoff_seconds))
    json_max_sleep = max(json_backoff, float(json_max_sleep_seconds))
    while True:
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code == 429 and rate_limit_attempt < int(max_429_retries):
            retry_after = _retry_after_seconds(resp.headers.get("Retry-After"))
            sleep_for = max(backoff, retry_after or 0.0)
            time.sleep(min(max_sleep, sleep_for))
            backoff = min(max_sleep, backoff * 2.0)
            rate_limit_attempt += 1
            continue
        if resp.status_code == 429:
            raise RuntimeError(
                f"Massive/Polygon rate limit persisted after {rate_limit_attempt + 1} request attempt(s). "
                "The downloader kept the checkpoint state; resume later or increase --pace."
            )
        resp.raise_for_status()
        try:
            return resp.json()
        except ValueError as exc:
            if json_attempt < int(max_json_retries):
                time.sleep(min(json_max_sleep, json_backoff))
                json_backoff = min(json_max_sleep, json_backoff * 2.0)
                json_attempt += 1
                continue
            content_length_header = str(resp.headers.get("Content-Length") or "").strip()
            try:
                body_bytes = len(resp.content or b"")
            except Exception:
                body_bytes = -1
            length_note = f"; body_bytes={body_bytes}"
            if content_length_header:
                length_note += f"; content_length={content_length_header}"
            raise RuntimeError(
                "Massive/Polygon returned invalid JSON after "
                f"{json_attempt + 1} parse attempt(s){length_note}. "
                "This is usually a truncated provider response; the checkpoint was preserved so the same page can be retried."
            ) from exc


def fetch_minutes(
    ticker: str,
    start: dt.datetime,
    end: dt.datetime,
    api_key: str,
    delay_seconds: float = 0.0,
    limit: int = 50000,
    unadjusted: bool = False,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    resume_state: Optional[Dict[str, Any]] = None,
    state_path: Optional[Path] = None,
    max_429_retries: int = DEFAULT_429_RETRIES,
    rate_limit_backoff_seconds: float = DEFAULT_429_BACKOFF_SECONDS,
    rate_limit_max_sleep_seconds: float = DEFAULT_429_MAX_SLEEP_SECONDS,
    max_json_retries: int = DEFAULT_JSON_RETRIES,
    json_backoff_seconds: float = DEFAULT_JSON_BACKOFF_SECONDS,
    json_max_sleep_seconds: float = DEFAULT_JSON_MAX_SLEEP_SECONDS,
) -> pd.DataFrame:
    """
    Fetch 1-minute aggregates between start and end (inclusive) using Polygon's v2 aggregates.
    delay_seconds: optional fixed pause between page requests to respect rate limits.
    """
    limit = min(max(1, limit), 50000)
    adj_flag = "false" if unadjusted else "true"
    url = (
        f"{API_HOST}/v2/aggs/ticker/{ticker}/range/1/minute/"
        f"{_iso_date(start)}/{_iso_date(end)}?adjusted={adj_flag}&sort=asc&limit={limit}"
    )
    all_rows: List[Dict[str, Any]] = []
    next_url: Optional[str] = url
    pbar = tqdm(total=0, unit="page", desc="Pages") if (tqdm and progress_cb is None) else None
    pages = 0
    if resume_state:
        next_url = resume_state.get("next_url") or next_url
        pages = int(resume_state.get("pages", 0))

    while next_url:
        data = _fetch_page(
            next_url,
            api_key,
            max_429_retries=max_429_retries,
            rate_limit_backoff_seconds=rate_limit_backoff_seconds,
            rate_limit_max_sleep_seconds=rate_limit_max_sleep_seconds,
            max_json_retries=max_json_retries,
            json_backoff_seconds=json_backoff_seconds,
            json_max_sleep_seconds=json_max_sleep_seconds,
        )
        results = data.get("results", [])
        for r in results:
            all_rows.append(
                {
                    "timestamp": pd.to_datetime(r["t"], unit="ms", utc=True),
                    "open": r.get("o"),
                    "high": r.get("h"),
                    "low": r.get("l"),
                    "close": r.get("c"),
                    "volume": r.get("v"),
                }
            )
        next_url = data.get("next_url")
        if next_url:
            # Normalize next_url and attach key if missing.
            if next_url.startswith("/"):
                next_url = f"{API_HOST}{next_url}"
            if "apiKey=" not in next_url:
                sep = "&" if "?" in next_url else "?"
                next_url = f"{next_url}{sep}apiKey={api_key}"
            if delay_seconds > 0:
                time.sleep(delay_seconds)
        if pbar:
            pbar.total += 1
            pbar.update(1)
        pages += 1
        if state_path:
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_payload = {
                "ticker": ticker,
                "next_url": next_url,
                "pages": pages,
                "rows": len(all_rows),
                "updated_at": dt.datetime.now(dt.UTC).isoformat(),
            }
            state_path.write_text(json.dumps(state_payload, indent=2), encoding="utf-8")
        if progress_cb:
            progress_cb(pages, len(all_rows))

    if pbar:
        pbar.close()

    if not all_rows:
        raise RuntimeError("No data returned from Massive/Polygon API.")

    df = pd.DataFrame(all_rows)
    df = df.sort_values("timestamp").set_index("timestamp")
    return df


def merge_existing_output(out_path: Path, frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty or not out_path.exists():
        return frame
    try:
        existing = pd.read_csv(out_path)
    except Exception:
        return frame
    if existing.empty or "timestamp" not in existing.columns:
        return frame

    existing["timestamp"] = pd.to_datetime(existing["timestamp"], utc=True, errors="coerce")
    existing = existing.dropna(subset=["timestamp"])
    if existing.empty:
        return frame

    current = frame.reset_index()
    if "timestamp" not in current.columns:
        current = current.rename(columns={current.columns[0]: "timestamp"})
    current["timestamp"] = pd.to_datetime(current["timestamp"], utc=True, errors="coerce")
    current = current.dropna(subset=["timestamp"])

    columns = ["timestamp", "open", "high", "low", "close", "volume"]
    merged = pd.concat([existing[columns], current[columns]], ignore_index=True)
    merged = merged.dropna(subset=["timestamp"]).drop_duplicates(subset=["timestamp"], keep="last")
    merged = merged.sort_values("timestamp").set_index("timestamp")
    return merged[["open", "high", "low", "close", "volume"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download 2 years of 1-minute bars from Massive (Polygon).")
    parser.add_argument("ticker", help="Ticker symbol, e.g., AAPL")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data") / "prices_1m.csv",
        help="Output CSV path (leave default to auto-name).",
    )
    parser.add_argument("--start", type=str, help="Start date YYYY-MM-DD (optional, defaults to today-2y)")
    parser.add_argument("--end", type=str, help="End date YYYY-MM-DD (optional, defaults to today)")
    parser.add_argument("--pace", type=float, default=DEFAULT_PACE_SECONDS, help="Seconds to sleep between page requests to avoid 429s (default 15.0)")
    parser.add_argument("--limit", type=int, default=50000, help="Page size 1-50000 (use 50000 to minimize calls).")
    parser.add_argument("--unadjusted", action="store_true", help="Request unadjusted data (default: adjusted).")
    parser.add_argument("--api-key", type=str, default="", help=f"Optional Massive/Polygon API key. Falls back to ${API_KEY_ENV}.")
    parser.add_argument("--progress", action="store_true", help="Emit progress JSON on stdout.")
    parser.add_argument("--resume", action="store_true", help="Resume from saved state if present.")
    parser.add_argument("--merge-output", action="store_true", help="Merge downloaded rows with an existing output CSV before saving.")
    parser.add_argument("--max-429-retries", type=int, default=DEFAULT_429_RETRIES, help="Maximum retries for Massive 429 rate-limit responses.")
    parser.add_argument("--rate-limit-backoff", type=float, default=DEFAULT_429_BACKOFF_SECONDS, help="Initial seconds to wait after a Massive 429 response.")
    parser.add_argument("--rate-limit-max-sleep", type=float, default=DEFAULT_429_MAX_SLEEP_SECONDS, help="Maximum seconds to wait between Massive 429 retries.")
    parser.add_argument("--max-json-retries", type=int, default=DEFAULT_JSON_RETRIES, help="Maximum retries for malformed/truncated Massive JSON responses.")
    parser.add_argument("--json-backoff", type=float, default=DEFAULT_JSON_BACKOFF_SECONDS, help="Initial seconds to wait after a malformed Massive JSON response.")
    parser.add_argument("--json-max-sleep", type=float, default=DEFAULT_JSON_MAX_SLEEP_SECONDS, help="Maximum seconds to wait between malformed JSON retries.")
    args = parser.parse_args()

    api_key = str(args.api_key or os.getenv(API_KEY_ENV) or "").strip()
    if not api_key:
        raise SystemExit(f"Set {API_KEY_ENV} environment variable or pass --api-key.")

    today = dt.datetime.now(dt.UTC).date()
    default_end = today
    default_start = today - dt.timedelta(days=365 * 2)

    start_dt = dt.datetime.strptime(args.start, "%Y-%m-%d") if args.start else dt.datetime.combine(default_start, dt.time())
    end_dt = dt.datetime.strptime(args.end, "%Y-%m-%d") if args.end else dt.datetime.combine(default_end, dt.time())

    if start_dt > end_dt:
        raise SystemExit("Start date must be before end date.")

    def emit_progress(pages: int, rows: int) -> None:
        if not args.progress:
            return
        payload = {
            "type": "progress",
            "ticker": args.ticker.upper(),
            "pages": pages,
            "rows": rows,
        }
        print(json.dumps(payload), flush=True)

    state_path = Path("data") / "download_state" / f"{args.ticker.upper()}.json"
    resume_state = None
    if args.resume and state_path.exists():
        try:
            resume_state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            resume_state = None

    try:
        if args.progress:
            print(
                json.dumps(
                    {
                        "type": "start",
                        "ticker": args.ticker.upper(),
                        "start": str(start_dt.date()),
                        "end": str(end_dt.date()),
                    }
                ),
                flush=True,
            )
        else:
            print(f"Fetching {args.ticker} from {start_dt.date()} to {end_dt.date()}...")
        df = fetch_minutes(
            args.ticker.upper(),
            start_dt,
            end_dt,
            api_key,
            delay_seconds=args.pace,
            limit=args.limit,
            unadjusted=args.unadjusted,
            progress_cb=emit_progress,
            resume_state=resume_state,
            state_path=state_path if args.resume else None,
            max_429_retries=args.max_429_retries,
            rate_limit_backoff_seconds=args.rate_limit_backoff,
            rate_limit_max_sleep_seconds=args.rate_limit_max_sleep,
            max_json_retries=args.max_json_retries,
            json_backoff_seconds=args.json_backoff,
            json_max_sleep_seconds=args.json_max_sleep,
        )
        out_path = args.out
        if args.out.name == "prices_1m.csv":
            start_tag = start_dt.strftime("%Y-%m-%d")
            end_tag = end_dt.strftime("%Y-%m-%d")
            out_path = Path("data") / f"{args.ticker.upper()}_massive_{start_tag}_{end_tag}_1m.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if args.merge_output:
            df = merge_existing_output(out_path, df)
        df.to_csv(out_path)
        if args.progress:
            print(json.dumps({"type": "done", "ticker": args.ticker.upper(), "rows": len(df), "out": str(out_path)}), flush=True)
        else:
            print(f"Saved {len(df)} bars to {out_path}")
        if args.resume and state_path.exists():
            state_path.unlink()
    except Exception as exc:
        details = traceback.format_exc()
        if args.progress:
            print(
                json.dumps(
                    {
                        "type": "error",
                        "ticker": args.ticker.upper(),
                        "message": str(exc),
                        "error_type": type(exc).__name__,
                        "details": details.splitlines()[-1] if details else str(exc),
                    }
                ),
                flush=True,
            )
        else:
            print(details or str(exc), file=sys.stderr, flush=True)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
