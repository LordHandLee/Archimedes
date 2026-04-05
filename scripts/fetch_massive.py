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
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import requests

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


API_HOST = "https://api.polygon.io"
API_KEY_ENV = "AAw8ohj8iAa7ENJ9YFpMmjMBbAZZhGVF"


def _iso_date(dt_obj: dt.datetime) -> str:
    return dt_obj.strftime("%Y-%m-%d")


def _fetch_page(url: str, api_key: str) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}"}
    attempt = 0
    backoff = 1.0
    while True:
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code == 429 and attempt < 5:
            time.sleep(backoff)
            backoff *= 1.5
            attempt += 1
            continue
        resp.raise_for_status()
        return resp.json()


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
    pbar = tqdm(total=0, unit="page", desc="Pages") if tqdm else None
    pages = 0
    if resume_state:
        next_url = resume_state.get("next_url") or next_url
        pages = int(resume_state.get("pages", 0))

    while next_url:
        data = _fetch_page(next_url, api_key)
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
                "updated_at": dt.datetime.utcnow().isoformat(),
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
    parser.add_argument("--pace", type=float, default=12.5, help="Seconds to sleep between page requests to avoid 429s (default 12.5 ~ 5 calls/min)")
    parser.add_argument("--limit", type=int, default=50000, help="Page size 1-50000 (use 50000 to minimize calls).")
    parser.add_argument("--unadjusted", action="store_true", help="Request unadjusted data (default: adjusted).")
    parser.add_argument("--progress", action="store_true", help="Emit progress JSON on stdout.")
    parser.add_argument("--resume", action="store_true", help="Resume from saved state if present.")
    args = parser.parse_args()

    # api_key = os.getenv(API_KEY_ENV)
    # if not api_key:
    #     raise SystemExit(f"Set {API_KEY_ENV} environment variable.")
    api_key = API_KEY_ENV

    today = dt.datetime.utcnow().date()
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

    if args.progress:
        print(json.dumps({"type": "start", "ticker": args.ticker.upper(), "start": str(start_dt.date()), "end": str(end_dt.date())}), flush=True)
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
    )
    out_path = args.out
    if args.out.name == "prices_1m.csv":
        start_tag = start_dt.strftime("%Y-%m-%d")
        end_tag = end_dt.strftime("%Y-%m-%d")
        out_path = Path("data") / f"{args.ticker.upper()}_massive_{start_tag}_{end_tag}_1m.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path)
    if args.progress:
        print(json.dumps({"type": "done", "ticker": args.ticker.upper(), "rows": len(df), "out": str(out_path)}), flush=True)
    else:
        print(f"Saved {len(df)} bars to {out_path}")
    if args.resume and state_path.exists():
        state_path.unlink()


if __name__ == "__main__":
    main()
