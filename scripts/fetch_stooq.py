"""
Fetch daily bars from the Stooq CSV endpoint for a given ticker and save to CSV.

Usage:
    python scripts/fetch_stooq.py AAPL --out data/AAPL_stooq_max_1d.csv

Notes:
- Stooq is used here as a lightweight second provider to prove the provider registry model.
- This script downloads daily bars only.
- The output schema is normalized to the same CSV layout used by the canonical ingestor.
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import sys
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Optional

import pandas as pd


def _provider_symbol(ticker: str) -> str:
    normalized = str(ticker).strip().lower()
    if "." in normalized:
        return normalized
    return f"{normalized}.us"


def _fetch_csv_text(provider_symbol: str) -> str:
    url = f"https://stooq.com/q/d/l/?s={provider_symbol}&i=d"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:  # pragma: no cover
        raise RuntimeError(f"Stooq HTTP error {exc.code} for {provider_symbol}.") from exc
    except urllib.error.URLError as exc:  # pragma: no cover
        raise RuntimeError(f"Stooq request failed for {provider_symbol}: {exc.reason}") from exc


def fetch_daily(
    ticker: str,
    *,
    start: dt.datetime | None = None,
    end: dt.datetime | None = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> pd.DataFrame:
    provider_symbol = _provider_symbol(ticker)
    csv_text = _fetch_csv_text(provider_symbol)
    if not csv_text.strip():
        raise RuntimeError(f"No data returned from Stooq for {ticker}.")
    if "No data" in csv_text:
        raise RuntimeError(f"Stooq returned no data for {ticker}.")
    frame = pd.read_csv(io.StringIO(csv_text))
    required = {"Date", "Open", "High", "Low", "Close"}
    if not required.issubset(frame.columns):
        raise RuntimeError(f"Unexpected Stooq response schema for {ticker}: {sorted(frame.columns)}")
    frame["timestamp"] = pd.to_datetime(frame["Date"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).copy()
    if start is not None:
        frame = frame.loc[frame["timestamp"] >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        frame = frame.loc[frame["timestamp"] <= pd.Timestamp(end, tz="UTC")]
    frame = frame.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    if "volume" not in frame.columns:
        frame["volume"] = 0
    frame = frame[["timestamp", "open", "high", "low", "close", "volume"]].sort_values("timestamp")
    if frame.empty:
        raise RuntimeError(f"No Stooq bars remain for {ticker} after date filtering.")
    if progress_cb:
        progress_cb(1, len(frame))
    return frame.set_index("timestamp")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download daily bars from Stooq.")
    parser.add_argument("ticker", help="Ticker symbol, e.g., AAPL")
    parser.add_argument("--out", type=Path, default=Path("data") / "prices_1d.csv")
    parser.add_argument("--start", type=str, help="Optional start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, help="Optional end date YYYY-MM-DD")
    parser.add_argument("--progress", action="store_true", help="Emit JSON progress on stdout.")
    parser.add_argument("--resume", action="store_true", help="Accepted for compatibility; ignored.")
    parser.add_argument("--pace", type=float, default=0.0, help="Accepted for compatibility; ignored.")
    args = parser.parse_args()

    start_dt = dt.datetime.strptime(args.start, "%Y-%m-%d") if args.start else None
    end_dt = dt.datetime.strptime(args.end, "%Y-%m-%d") if args.end else None
    if start_dt and end_dt and start_dt > end_dt:
        raise SystemExit("Start date must be before end date.")

    def emit_progress(pages: int, rows: int) -> None:
        if not args.progress:
            return
        print(
            json.dumps(
                {
                    "type": "progress",
                    "ticker": args.ticker.upper(),
                    "pages": pages,
                    "rows": rows,
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
                        "start": args.start or "min",
                        "end": args.end or "max",
                    }
                ),
                flush=True,
            )
        df = fetch_daily(
            args.ticker.upper(),
            start=start_dt,
            end=end_dt,
            progress_cb=emit_progress,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.reset_index().to_csv(args.out, index=False)
        if args.progress:
            print(
                json.dumps(
                    {
                        "type": "done",
                        "ticker": args.ticker.upper(),
                        "rows": int(len(df)),
                        "out": str(args.out),
                    }
                ),
                flush=True,
            )
        else:
            print(f"Saved {len(df)} rows to {args.out}")
    except Exception as exc:
        if args.progress:
            print(
                json.dumps(
                    {
                        "type": "error",
                        "ticker": args.ticker.upper(),
                        "message": str(exc),
                        "details": traceback.format_exc(limit=5),
                    }
                ),
                flush=True,
            )
        else:  # pragma: no cover
            print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":  # pragma: no cover
    main()
