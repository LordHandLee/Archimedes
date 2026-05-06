from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional at runtime, validated by UI error paths
    from ibapi.client import EClient
    from ibapi.contract import Contract
    from ibapi.wrapper import EWrapper
except Exception as exc:  # pragma: no cover
    class _MissingEClient:
        pass

    class _MissingEWrapper:
        pass

    class _MissingContract:
        pass

    EClient = _MissingEClient  # type: ignore[assignment]
    EWrapper = _MissingEWrapper  # type: ignore[assignment]
    Contract = _MissingContract  # type: ignore[assignment]
    _IBAPI_IMPORT_ERROR = exc
else:
    _IBAPI_IMPORT_ERROR = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIVE_MARKET_DB_PATH = PROJECT_ROOT / "data/live_market.sqlite"
DEFAULT_WATCHLIST_SYMBOLS = ("SPY", "QQQ", "AAPL", "MSFT", "NVDA")
DEFAULT_LIVE_PROVIDER = "interactive_brokers"
LIVE_BAR_TIMEFRAME = "1m"
DEFAULT_STALE_QUOTE_SECONDS = 90
CHART_TIMEFRAME_OPTIONS: tuple[tuple[str, str], ...] = (
    ("1 Minute", "1m"),
    ("5 Minutes", "5m"),
    ("15 Minutes", "15m"),
    ("1 Hour", "1h"),
    ("4 Hours", "4h"),
    ("1 Day", "1d"),
)
MAX_WATCHLIST_STREAM_SYMBOLS = 100
LIVE_MARKET_SQLITE_BUSY_TIMEOUT_MS = 30_000
LIVE_MARKET_SQLITE_TIMEOUT_SECONDS = LIVE_MARKET_SQLITE_BUSY_TIMEOUT_MS / 1000.0
LIVE_MARKET_SQLITE_LOCK_RETRY_TIMEOUT_SECONDS = 30.0
LIVE_MARKET_SQLITE_LOCK_RETRY_INITIAL_SECONDS = 0.05
LIVE_MARKET_SQLITE_LOCK_RETRY_MAX_SECONDS = 1.0
LIVE_MARKET_SQLITE_WRITE_CHUNK_SIZE = 500
IB_PRIMARY_EXCHANGE_OVERRIDES = {
    "DIA": "ARCA",
    "IWM": "ARCA",
    "QQQ": "NASDAQ",
    "SOXL": "ARCA",
    "SOXS": "ARCA",
    "SPY": "ARCA",
    "SQQQ": "NASDAQ",
    "TQQQ": "NASDAQ",
}


@dataclass(frozen=True)
class LiveMarketBar:
    symbol: str
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    wap: float | None = None
    trade_count: int | None = None
    provider: str = DEFAULT_LIVE_PROVIDER
    timeframe: str = LIVE_BAR_TIMEFRAME
    received_at: pd.Timestamp | None = None

    def as_record(self) -> dict[str, object]:
        received_at = self.received_at or pd.Timestamp.now(tz="UTC")
        return {
            "provider": self.provider,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "ts_utc": _iso_utc(self.timestamp),
            "open": float(self.open),
            "high": float(self.high),
            "low": float(self.low),
            "close": float(self.close),
            "volume": float(self.volume),
            "wap": None if self.wap is None else float(self.wap),
            "trade_count": None if self.trade_count is None else int(self.trade_count),
            "received_at": _iso_utc(received_at),
        }


@dataclass(frozen=True)
class InteractiveBrokersRealtimeConfig:
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 9301
    sec_type: str = "STK"
    exchange: str = "SMART"
    currency: str = "USD"
    primary_exchange: str = ""
    what_to_show: str = "TRADES"
    use_rth: bool = False
    timeout_seconds: float = 15.0


def _clean_symbol(symbol: str | None) -> str:
    return str(symbol or "").strip().upper()


def _clean_provider(provider: str | None) -> str:
    return str(provider or DEFAULT_LIVE_PROVIDER).strip().lower() or DEFAULT_LIVE_PROVIDER


def _ensure_utc(value: object) -> pd.Timestamp:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid timestamp: {value!r}")
    return pd.Timestamp(ts).tz_convert("UTC")


def _iso_utc(value: object) -> str:
    return _ensure_utc(value).isoformat().replace("+00:00", "Z")


def sqlite_error_is_locked(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "database is locked" in text or "database table is locked" in text or "database is busy" in text


def _empty_ohlcv_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


def normalize_chart_timeframe(value: str | None) -> str:
    text = " ".join(str(value or "").strip().lower().split())
    if not text:
        return LIVE_BAR_TIMEFRAME
    aliases = {
        "1min": "1m",
        "1 min": "1m",
        "1 minute": "1m",
        "1 minutes": "1m",
        "5min": "5m",
        "5 min": "5m",
        "5 minute": "5m",
        "5 minutes": "5m",
        "15min": "15m",
        "15 min": "15m",
        "15 minute": "15m",
        "15 minutes": "15m",
        "1 hour": "1h",
        "1 hours": "1h",
        "4 hour": "4h",
        "4 hours": "4h",
        "1 day": "1d",
        "1 days": "1d",
    }
    compact = text.replace(" ", "")
    return aliases.get(text, aliases.get(compact, compact))


def chart_timeframe_label(value: str | None) -> str:
    normalized = normalize_chart_timeframe(value)
    for label, timeframe in CHART_TIMEFRAME_OPTIONS:
        if timeframe == normalized:
            return label
    return normalized


def chart_timeframe_to_pandas_rule(value: str | None) -> str:
    normalized = normalize_chart_timeframe(value)
    if normalized.endswith("m"):
        return f"{int(normalized[:-1] or '1')}min"
    if normalized.endswith("h"):
        return f"{int(normalized[:-1] or '1')}h"
    if normalized.endswith("d"):
        return f"{int(normalized[:-1] or '1')}D"
    return "1min"


def chart_timeframe_delta(value: str | None) -> pd.Timedelta:
    normalized = normalize_chart_timeframe(value)
    try:
        if normalized.endswith("m"):
            return pd.Timedelta(minutes=int(normalized[:-1] or "1"))
        if normalized.endswith("h"):
            return pd.Timedelta(hours=int(normalized[:-1] or "1"))
        if normalized.endswith("d"):
            return pd.Timedelta(days=int(normalized[:-1] or "1"))
    except Exception:
        pass
    return pd.Timedelta(minutes=1)


def resample_ohlcv(frame: pd.DataFrame, timeframe: str | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return _empty_ohlcv_frame()
    normalized = normalize_chart_timeframe(timeframe)
    out = frame.sort_index().copy()
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    columns = ["open", "high", "low", "close", "volume"]
    for column in columns:
        if column not in out.columns:
            out[column] = 0.0
    out = out[columns].apply(pd.to_numeric, errors="coerce")
    if normalized == LIVE_BAR_TIMEFRAME:
        return out.dropna(subset=["open", "high", "low", "close"])
    aggregated = (
        out.resample(chart_timeframe_to_pandas_rule(normalized), label="left", closed="left")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna(subset=["open", "high", "low", "close"])
    )
    return aggregated[columns].astype(float)


def market_session_open_utc(
    now: pd.Timestamp | None = None,
    *,
    timezone: str = "America/New_York",
) -> pd.Timestamp:
    local_now = pd.Timestamp.now(tz=timezone) if now is None else _ensure_utc(now).tz_convert(timezone)
    session_open = local_now.normalize() + pd.Timedelta(hours=9, minutes=30)
    return pd.Timestamp(session_open).tz_convert("UTC")


class LiveMarketDataStore:
    """Separate live-market store used by the Charts tab.

    This intentionally does not write into the historical DuckDB/Parquet store.
    Live data can be audited or discarded without changing research datasets.
    """

    def __init__(
        self,
        db_path: str | Path = DEFAULT_LIVE_MARKET_DB_PATH,
        *,
        sqlite_timeout_seconds: float = LIVE_MARKET_SQLITE_TIMEOUT_SECONDS,
        lock_retry_timeout_seconds: float = LIVE_MARKET_SQLITE_LOCK_RETRY_TIMEOUT_SECONDS,
    ) -> None:
        raw_path = Path(db_path)
        self.db_path = raw_path if raw_path.is_absolute() else PROJECT_ROOT / raw_path
        self.sqlite_timeout_seconds = max(0.01, float(sqlite_timeout_seconds))
        self.lock_retry_timeout_seconds = max(0.0, float(lock_retry_timeout_seconds))
        self.ensure_schema()

    def _connect(self, *, configure_wal: bool = False) -> sqlite3.Connection:
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.db_path, timeout=self.sqlite_timeout_seconds)
            conn.execute(f"PRAGMA busy_timeout={max(1, int(self.sqlite_timeout_seconds * 1000))}")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA temp_store=MEMORY")
            if configure_wal:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA wal_autocheckpoint=1000")
            return conn
        except (OSError, sqlite3.Error) as exc:
            raise sqlite3.OperationalError(f"unable to open live market database at {self.db_path}: {exc}") from exc

    def _run_with_lock_retry(self, operation: Callable[[], object]) -> object:
        deadline = time.monotonic() + self.lock_retry_timeout_seconds
        delay = LIVE_MARKET_SQLITE_LOCK_RETRY_INITIAL_SECONDS
        while True:
            try:
                return operation()
            except sqlite3.OperationalError as exc:
                if not sqlite_error_is_locked(exc) or time.monotonic() >= deadline:
                    raise
                time.sleep(delay)
                delay = min(LIVE_MARKET_SQLITE_LOCK_RETRY_MAX_SECONDS, delay * 1.5)

    def ensure_schema(self) -> None:
        def _operation() -> None:
            with self._connect(configure_wal=True) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS watchlist (
                        symbol TEXT PRIMARY KEY,
                        position INTEGER NOT NULL,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS live_bars (
                        provider TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        ts_utc TEXT NOT NULL,
                        open REAL NOT NULL,
                        high REAL NOT NULL,
                        low REAL NOT NULL,
                        close REAL NOT NULL,
                        volume REAL NOT NULL DEFAULT 0,
                        wap REAL,
                        trade_count INTEGER,
                        received_at TEXT NOT NULL,
                        PRIMARY KEY (provider, symbol, timeframe, ts_utc)
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS live_quotes (
                        provider TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        price REAL,
                        ts_utc TEXT,
                        status TEXT,
                        received_at TEXT NOT NULL,
                        PRIMARY KEY (provider, symbol)
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_live_bars_symbol_time ON live_bars(symbol, timeframe, ts_utc)"
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_live_quotes_symbol ON live_quotes(symbol)")
                conn.commit()

        self._run_with_lock_retry(_operation)

    def ensure_default_watchlist(self, symbols: Sequence[str] = DEFAULT_WATCHLIST_SYMBOLS) -> list[str]:
        current = self.load_watchlist()
        if current:
            return current
        self.save_watchlist(symbols)
        return self.load_watchlist()

    def load_watchlist(self) -> list[str]:
        def _operation() -> list[str]:
            with self._connect() as conn:
                rows = conn.execute("SELECT symbol FROM watchlist ORDER BY position ASC, symbol ASC").fetchall()
            return [str(row[0]) for row in rows]

        return self._run_with_lock_retry(_operation)  # type: ignore[return-value]

    def save_watchlist(self, symbols: Sequence[str]) -> None:
        cleaned: list[str] = []
        seen: set[str] = set()
        for symbol in symbols:
            normalized = _clean_symbol(symbol)
            if normalized and normalized not in seen:
                cleaned.append(normalized)
                seen.add(normalized)
        now = _iso_utc(pd.Timestamp.now(tz="UTC"))

        def _operation() -> None:
            with self._connect() as conn:
                conn.execute("DELETE FROM watchlist")
                conn.executemany(
                    """
                    INSERT INTO watchlist(symbol, position, created_at, updated_at)
                    VALUES(?, ?, ?, ?)
                    """,
                    [(symbol, idx, now, now) for idx, symbol in enumerate(cleaned)],
                )
                conn.commit()

        self._run_with_lock_retry(_operation)

    def add_watchlist_symbol(self, symbol: str) -> list[str]:
        normalized = _clean_symbol(symbol)
        if not normalized:
            return self.load_watchlist()
        symbols = self.load_watchlist()
        if normalized not in symbols:
            symbols.append(normalized)
            self.save_watchlist(symbols)
        return symbols

    def remove_watchlist_symbol(self, symbol: str) -> list[str]:
        normalized = _clean_symbol(symbol)
        symbols = [item for item in self.load_watchlist() if item != normalized]
        self.save_watchlist(symbols)
        return symbols

    def upsert_bar(self, bar: LiveMarketBar) -> None:
        self.upsert_bars([bar])

    def upsert_bars(
        self,
        bars: Sequence[LiveMarketBar],
        *,
        chunk_size: int = LIVE_MARKET_SQLITE_WRITE_CHUNK_SIZE,
    ) -> int:
        records = [bar.as_record() for bar in bars]
        if not records:
            return 0
        written = 0
        chunk_len = max(1, int(chunk_size))
        for start in range(0, len(records), chunk_len):
            chunk = records[start:start + chunk_len]

            def _operation(chunk_records: list[dict[str, object]] = chunk) -> None:
                with self._connect() as conn:
                    self._upsert_bar_records(conn, chunk_records)
                    conn.commit()

            self._run_with_lock_retry(_operation)
            written += len(chunk)
        return written

    @staticmethod
    def _upsert_bar_records(conn: sqlite3.Connection, records: Sequence[dict[str, object]]) -> None:
        conn.executemany(
            """
            INSERT INTO live_bars(
                provider, symbol, timeframe, ts_utc, open, high, low, close,
                volume, wap, trade_count, received_at
            )
            VALUES(:provider, :symbol, :timeframe, :ts_utc, :open, :high, :low, :close,
                   :volume, :wap, :trade_count, :received_at)
            ON CONFLICT(provider, symbol, timeframe, ts_utc) DO UPDATE SET
                open=excluded.open,
                high=excluded.high,
                low=excluded.low,
                close=excluded.close,
                volume=excluded.volume,
                wap=excluded.wap,
                trade_count=excluded.trade_count,
                received_at=excluded.received_at
            """,
            records,
        )
        conn.executemany(
            """
            INSERT INTO live_quotes(provider, symbol, price, ts_utc, status, received_at)
            VALUES(:provider, :symbol, :close, :ts_utc, 'ok', :received_at)
            ON CONFLICT(provider, symbol) DO UPDATE SET
                price=excluded.price,
                ts_utc=excluded.ts_utc,
                status=excluded.status,
                received_at=excluded.received_at
            """,
            records,
        )

    def update_quote(
        self,
        symbol: str,
        price: float | None,
        *,
        provider: str = DEFAULT_LIVE_PROVIDER,
        timestamp: object | None = None,
        status: str = "ok",
    ) -> None:
        normalized = _clean_symbol(symbol)
        if not normalized:
            return
        now = pd.Timestamp.now(tz="UTC")
        ts = _iso_utc(timestamp or now)
        received_at = _iso_utc(now)

        def _operation() -> None:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO live_quotes(provider, symbol, price, ts_utc, status, received_at)
                    VALUES(?, ?, ?, ?, ?, ?)
                    ON CONFLICT(provider, symbol) DO UPDATE SET
                        price=excluded.price,
                        ts_utc=excluded.ts_utc,
                        status=excluded.status,
                        received_at=excluded.received_at
                    """,
                    (_clean_provider(provider), normalized, price, ts, str(status or ""), received_at),
                )
                conn.commit()

        self._run_with_lock_retry(_operation)

    def latest_quotes(
        self,
        symbols: Sequence[str] | None = None,
        *,
        provider: str | None = None,
        stale_after_seconds: int = DEFAULT_STALE_QUOTE_SECONDS,
    ) -> list[dict[str, object]]:
        requested = [_clean_symbol(symbol) for symbol in list(symbols or self.load_watchlist())]
        requested = [symbol for symbol in requested if symbol]
        if not requested:
            return []
        provider_filter = _clean_provider(provider) if provider else ""
        rows_by_symbol: dict[str, dict[str, object]] = {}
        def _operation() -> tuple[list[tuple], dict[str, float]]:
            open_prices: dict[str, float] = {}
            with self._connect() as conn:
                if provider_filter:
                    placeholders = ",".join(["?"] * len(requested))
                    rows = conn.execute(
                        f"""
                        SELECT provider, symbol, price, ts_utc, status, received_at
                        FROM live_quotes
                        WHERE provider=? AND symbol IN ({placeholders})
                        """,
                        [provider_filter] + requested,
                    ).fetchall()
                else:
                    placeholders = ",".join(["?"] * len(requested))
                    rows = conn.execute(
                        f"""
                        SELECT provider, symbol, price, ts_utc, status, received_at
                        FROM live_quotes
                        WHERE symbol IN ({placeholders})
                        ORDER BY received_at DESC
                        """,
                        requested,
                    ).fetchall()
                if provider_filter:
                    session_open = _iso_utc(market_session_open_utc(now=pd.Timestamp.now(tz="UTC")))
                    placeholders = ",".join(["?"] * len(requested))
                    open_rows = conn.execute(
                        f"""
                        SELECT symbol, ts_utc, open
                        FROM live_bars
                        WHERE provider=? AND timeframe=? AND symbol IN ({placeholders}) AND ts_utc >= ?
                        ORDER BY symbol ASC, ts_utc ASC
                        """,
                        [provider_filter, LIVE_BAR_TIMEFRAME] + requested + [session_open],
                    ).fetchall()
                    for symbol, _ts_utc, open_price in open_rows:
                        symbol_text = str(symbol or "")
                        if symbol_text in open_prices:
                            continue
                        try:
                            open_float = float(open_price)
                        except Exception:
                            continue
                        if np.isfinite(open_float) and open_float > 0:
                            open_prices[symbol_text] = open_float
            return rows, open_prices

        rows, open_prices = self._run_with_lock_retry(_operation)  # type: ignore[assignment]
        now = pd.Timestamp.now(tz="UTC")
        for provider_id, symbol, price, ts_utc, status, received_at in rows:
            symbol_text = str(symbol)
            if symbol_text in rows_by_symbol:
                continue
            received_ts = pd.to_datetime(received_at, utc=True, errors="coerce")
            stale = True
            age_seconds: float | None = None
            if not pd.isna(received_ts):
                age_seconds = max(0.0, float((now - received_ts).total_seconds()))
                stale = age_seconds > float(stale_after_seconds)
            price_float = None if price is None else float(price)
            open_price = open_prices.get(symbol_text)
            change_percent = None
            if price_float is not None and open_price is not None and open_price > 0:
                change_percent = ((price_float - open_price) / open_price) * 100.0
            rows_by_symbol[symbol_text] = {
                "symbol": symbol_text,
                "provider": str(provider_id or ""),
                "price": price_float,
                "session_open_price": open_price,
                "change_percent": change_percent,
                "ts_utc": str(ts_utc or ""),
                "status": str(status or ""),
                "received_at": str(received_at or ""),
                "age_seconds": age_seconds,
                "stale": stale,
            }
        output: list[dict[str, object]] = []
        for symbol in requested:
            output.append(
                rows_by_symbol.get(
                    symbol,
                    {
                        "symbol": symbol,
                        "provider": provider_filter,
                        "price": None,
                        "session_open_price": None,
                        "change_percent": None,
                        "ts_utc": "",
                        "status": "missing",
                        "received_at": "",
                        "age_seconds": None,
                        "stale": True,
                    },
                )
            )
        return output

    def quote_is_stale(
        self,
        symbol: str,
        *,
        provider: str = DEFAULT_LIVE_PROVIDER,
        stale_after_seconds: int = DEFAULT_STALE_QUOTE_SECONDS,
    ) -> bool:
        quotes = self.latest_quotes([symbol], provider=provider, stale_after_seconds=stale_after_seconds)
        return True if not quotes else bool(quotes[0].get("stale", True))

    def load_recent_bars(
        self,
        symbol: str,
        *,
        provider: str = DEFAULT_LIVE_PROVIDER,
        timeframe: str = LIVE_BAR_TIMEFRAME,
        limit: int = 5000,
    ) -> pd.DataFrame:
        normalized = _clean_symbol(symbol)
        if not normalized:
            return _empty_ohlcv_frame()
        def _operation() -> list[tuple]:
            with self._connect() as conn:
                return conn.execute(
                    """
                    SELECT ts_utc, open, high, low, close, volume
                    FROM live_bars
                    WHERE provider=? AND symbol=? AND timeframe=?
                    ORDER BY ts_utc DESC
                    LIMIT ?
                    """,
                    (_clean_provider(provider), normalized, str(timeframe or LIVE_BAR_TIMEFRAME), int(limit)),
                ).fetchall()

        rows = self._run_with_lock_retry(_operation)
        if not rows:
            return _empty_ohlcv_frame()
        frame = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
        return frame.set_index("timestamp")[["open", "high", "low", "close", "volume"]].astype(float)

    def latest_bar_timestamp(
        self,
        symbol: str,
        *,
        provider: str = DEFAULT_LIVE_PROVIDER,
        timeframe: str = LIVE_BAR_TIMEFRAME,
    ) -> pd.Timestamp | None:
        normalized = _clean_symbol(symbol)
        if not normalized:
            return None
        def _operation() -> tuple | None:
            with self._connect() as conn:
                return conn.execute(
                    """
                    SELECT MAX(ts_utc)
                    FROM live_bars
                    WHERE provider=? AND symbol=? AND timeframe=?
                    """,
                    (_clean_provider(provider), normalized, str(timeframe or LIVE_BAR_TIMEFRAME)),
                ).fetchone()

        row = self._run_with_lock_retry(_operation)
        if not row or not row[0]:
            return None
        ts = pd.to_datetime(row[0], utc=True, errors="coerce")
        return None if pd.isna(ts) else pd.Timestamp(ts)


def compute_chart_indicators(
    bars: pd.DataFrame,
    indicator_ids: Sequence[str],
) -> tuple[dict[str, pd.Series], dict[str, pd.Series], dict[str, dict]]:
    if bars is None or bars.empty or "close" not in bars.columns:
        return {}, {}, {}
    frame = bars.sort_index().copy()
    close = pd.to_numeric(frame["close"], errors="coerce").astype(float)
    high = pd.to_numeric(frame.get("high", close), errors="coerce").astype(float)
    low = pd.to_numeric(frame.get("low", close), errors="coerce").astype(float)
    volume = pd.to_numeric(frame.get("volume", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0)
    overlays: dict[str, pd.Series] = {}
    panes: dict[str, pd.Series] = {}
    styles: dict[str, dict] = {}
    selected = {str(item).strip().lower() for item in list(indicator_ids or []) if str(item).strip()}

    if "sma20" in selected:
        overlays["SMA 20"] = close.rolling(20).mean()
        styles["SMA 20"] = {"color": "#4da3ff", "line_width": 1.2}
    if "sma50" in selected:
        overlays["SMA 50"] = close.rolling(50).mean()
        styles["SMA 50"] = {"color": "#ffd166", "line_width": 1.2}
    if "ema20" in selected:
        overlays["EMA 20"] = close.ewm(span=20, adjust=False).mean()
        styles["EMA 20"] = {"color": "#27d07d", "line_width": 1.2}
    if "vwap" in selected:
        typical = (high + low + close) / 3.0
        cumulative_volume = volume.replace(0.0, np.nan).cumsum()
        vwap = (typical * volume).cumsum() / cumulative_volume
        overlays["VWAP"] = vwap.ffill()
        styles["VWAP"] = {"color": "#ffcc66", "line_width": 1.2}
    if "rsi14" in selected:
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
        rs = gain / loss.replace(0.0, np.nan)
        panes["RSI 14"] = 100.0 - (100.0 / (1.0 + rs))
        styles["RSI 14"] = {"color": "#a28bff", "line_width": 1.2}

    return overlays, panes, styles


def incremental_series_payload(
    series_map: dict[str, pd.Series],
    *,
    bar_index: int,
    timestamp: pd.Timestamp,
    styles: dict[str, dict] | None = None,
    source_bar_index: int | None = None,
) -> list[dict]:
    payload: list[dict] = []
    styles = styles or {}
    value_index = int(bar_index if source_bar_index is None else source_bar_index)
    for name, series in series_map.items():
        if series.empty:
            continue
        if 0 <= value_index < len(series):
            value = pd.to_numeric(series.iloc[value_index], errors="coerce")
        else:
            value = pd.to_numeric(series.iloc[-1], errors="coerce")
        try:
            value_float = float(value)
        except Exception:
            continue
        if not np.isfinite(value_float):
            continue
        style = styles.get(name, {})
        item = {
            "name": name,
            "points": [
                {
                    "timestamp_utc_ns": str(int(timestamp.value)),
                    "bar_index": int(bar_index),
                    "value": value_float,
                }
            ],
        }
        color = str(style.get("color") or "").strip()
        if color:
            item["color"] = color
        payload.append(item)
    return payload


def latest_point_series_payload(
    series_map: dict[str, pd.Series],
    *,
    bar_index: int,
    timestamp: pd.Timestamp,
    styles: dict[str, dict] | None = None,
) -> list[dict]:
    payload: list[dict] = []
    if int(bar_index) < 0:
        return payload
    ts = _ensure_utc(timestamp)
    styles = styles or {}
    for name, series in series_map.items():
        if series is None or series.empty:
            continue
        aligned = pd.to_numeric(series, errors="coerce").dropna()
        if aligned.empty:
            continue
        if isinstance(aligned.index, pd.DatetimeIndex):
            if aligned.index.tz is None:
                aligned.index = aligned.index.tz_localize("UTC")
            else:
                aligned.index = aligned.index.tz_convert("UTC")
        aligned = aligned.sort_index()
        try:
            value_float = float(aligned.iloc[-1])
        except Exception:
            continue
        if not np.isfinite(value_float):
            continue
        style = styles.get(name, {})
        item = {
            "name": name,
            "points": [
                {
                    "timestamp_utc_ns": str(int(ts.value)),
                    "bar_index": int(bar_index),
                    "value": value_float,
                }
            ],
        }
        color = str(style.get("color") or "").strip()
        if color:
            item["color"] = color
        payload.append(item)
    return payload


def series_replacement_payload(
    series_map: dict[str, pd.Series],
    *,
    styles: dict[str, dict] | None = None,
    max_points: int | None = None,
) -> list[dict]:
    payload: list[dict] = []
    styles = styles or {}
    for name, series in series_map.items():
        if series.empty:
            continue
        aligned = pd.to_numeric(series, errors="coerce")
        indexed_values = list(enumerate(aligned.items()))
        if max_points is not None and int(max_points) > 0:
            indexed_values = indexed_values[-int(max_points):]
        points: list[dict] = []
        for bar_index, (timestamp, value) in indexed_values:
            try:
                value_float = float(value)
            except Exception:
                continue
            if not np.isfinite(value_float):
                continue
            ts = _ensure_utc(timestamp)
            points.append(
                {
                    "timestamp_utc_ns": str(int(ts.value)),
                    "bar_index": int(bar_index),
                    "value": value_float,
                }
            )
        if not points:
            continue
        item = {"name": name, "points": points}
        color = str(styles.get(name, {}).get("color") or "").strip()
        if color:
            item["color"] = color
        payload.append(item)
    return payload


class InteractiveBrokersRealtimeBarApp(EWrapper, EClient):
    def __init__(
        self,
        *,
        symbols: Sequence[str],
        config: InteractiveBrokersRealtimeConfig,
        bar_callback: Callable[[LiveMarketBar], None],
        partial_callback: Callable[[LiveMarketBar], None] | None = None,
        status_callback: Callable[[str], None] | None = None,
        error_callback: Callable[[str], None] | None = None,
    ) -> None:
        if _IBAPI_IMPORT_ERROR is not None:
            raise RuntimeError(
                "Interactive Brokers live market data requires the official 'ibapi' package and a running "
                "TWS/IB Gateway session with market data permissions."
            ) from _IBAPI_IMPORT_ERROR
        EWrapper.__init__(self)
        EClient.__init__(self, self)
        self.symbols = [_clean_symbol(symbol) for symbol in list(symbols or []) if _clean_symbol(symbol)]
        self.config = config
        self.bar_callback = bar_callback
        self.partial_callback = partial_callback
        self.status_callback = status_callback
        self.error_callback = error_callback
        self._connected = threading.Event()
        self._req_id_base = int(config.client_id) * 1000 + 7000
        self._req_symbols: dict[int, str] = {}
        self._partials: dict[int, dict[str, object]] = {}
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def _bar_from_partial(
        self,
        symbol: str,
        partial: dict[str, object],
        *,
        received_at: pd.Timestamp | None = None,
    ) -> LiveMarketBar:
        return LiveMarketBar(
            symbol=symbol,
            timestamp=pd.Timestamp(partial["timestamp"]).tz_convert("UTC"),
            open=float(partial["open"]),
            high=float(partial["high"]),
            low=float(partial["low"]),
            close=float(partial["close"]),
            volume=float(partial["volume"]),
            wap=float(partial["wap"]) if partial.get("wap") is not None else None,
            trade_count=int(partial["trade_count"]) if partial.get("trade_count") is not None else None,
            provider=DEFAULT_LIVE_PROVIDER,
            timeframe=LIVE_BAR_TIMEFRAME,
            received_at=received_at or pd.Timestamp.now(tz="UTC"),
        )

    def flush_stale_partials(self, *, force: bool = False, now: pd.Timestamp | None = None) -> None:
        now_ts = pd.Timestamp.now(tz="UTC") if now is None else _ensure_utc(now)
        bars: list[LiveMarketBar] = []
        with self._lock:
            for req_id, partial in list(self._partials.items()):
                if bool(partial.get("emitted")):
                    continue
                symbol = self._req_symbols.get(int(req_id), "")
                if not symbol:
                    continue
                bar_ts = pd.Timestamp(partial.get("timestamp")).tz_convert("UTC")
                if not force and bar_ts + pd.Timedelta(minutes=1) > now_ts:
                    continue
                partial["emitted"] = True
                bars.append(self._bar_from_partial(symbol, partial, received_at=now_ts))
        for bar in bars:
            self.bar_callback(bar)

    def nextValidId(self, orderId: int) -> None:  # noqa: N802
        self._connected.set()

    def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):  # noqa: N802
        code = int(errorCode)
        text = str(errorString or "")
        if code in {2103, 2104, 2105, 2106, 2107, 2108, 2158}:
            if self.status_callback:
                self.status_callback(text)
            return
        message = f"Interactive Brokers error {code} for request {reqId}: {text}"
        if self.error_callback:
            self.error_callback(message)

    def realtimeBar(self, reqId, time_, open_, high_, low_, close_, volume_, wap_, count_):  # noqa: N802
        symbol = self._req_symbols.get(int(reqId), "")
        if not symbol:
            return
        stamp = pd.to_datetime(int(time_), unit="s", utc=True).floor("min")
        emit_bar: LiveMarketBar | None = None
        preview_bar: LiveMarketBar | None = None
        with self._lock:
            partial = self._partials.get(int(reqId))
            if not partial or partial.get("timestamp") != stamp:
                if partial and not bool(partial.get("emitted")):
                    emit_bar = self._bar_from_partial(symbol, partial)
                partial = {
                    "timestamp": stamp,
                    "open": float(open_),
                    "high": float(high_),
                    "low": float(low_),
                    "close": float(close_),
                    "volume": float(volume_ or 0.0),
                    "wap": float(wap_ or 0.0),
                    "trade_count": int(count_ or 0),
                    "emitted": False,
                }
                self._partials[int(reqId)] = partial
            else:
                was_emitted = bool(partial.get("emitted"))
                partial["high"] = max(float(partial["high"]), float(high_))
                partial["low"] = min(float(partial["low"]), float(low_))
                partial["close"] = float(close_)
                partial["volume"] = float(partial["volume"]) + float(volume_ or 0.0)
                partial["wap"] = float(wap_ or partial.get("wap") or close_)
                partial["trade_count"] = int(partial.get("trade_count") or 0) + int(count_ or 0)
                if was_emitted and stamp + pd.Timedelta(minutes=1) <= pd.Timestamp.now(tz="UTC"):
                    emit_bar = self._bar_from_partial(symbol, partial)
                    partial["emitted"] = True
                else:
                    partial["emitted"] = False
            preview_bar = self._bar_from_partial(symbol, partial)
        if emit_bar is not None:
            self.bar_callback(emit_bar)
        if preview_bar is not None and self.partial_callback is not None:
            self.partial_callback(preview_bar)

    def connect_and_start(self) -> None:
        if not self.symbols:
            raise ValueError("At least one symbol is required for Interactive Brokers live data.")
        self.connect(self.config.host, int(self.config.port), int(self.config.client_id))
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()
        if not self._connected.wait(float(self.config.timeout_seconds)):
            raise TimeoutError("Timed out waiting for Interactive Brokers live data connection.")
        for idx, symbol in enumerate(self.symbols, start=1):
            req_id = self._req_id_base + idx
            self._req_symbols[req_id] = symbol
            self.reqRealTimeBars(
                req_id,
                _build_ib_contract(symbol, self.config),
                5,
                self.config.what_to_show,
                int(bool(self.config.use_rth)),
                [],
            )
        if self.status_callback:
            rth_text = "RTH only" if self.config.use_rth else "extended hours"
            self.status_callback(
                f"Interactive Brokers real-time bars active for {', '.join(self.symbols)} ({rth_text})."
            )

    def stop(self) -> None:
        self.flush_stale_partials(force=True)
        for req_id in list(self._req_symbols):
            try:
                self.cancelRealTimeBars(req_id)
            except Exception:
                pass
        self._req_symbols.clear()
        try:
            if self.isConnected():
                self.disconnect()
        except Exception:
            pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)


def _build_ib_contract(symbol: str, config: InteractiveBrokersRealtimeConfig) -> Contract:
    contract = Contract()
    cleaned_symbol = _clean_symbol(symbol)
    contract.symbol = cleaned_symbol
    contract.secType = str(config.sec_type or "STK").strip().upper()
    contract.exchange = str(config.exchange or "SMART").strip().upper()
    contract.currency = str(config.currency or "USD").strip().upper()
    primary = str(config.primary_exchange or IB_PRIMARY_EXCHANGE_OVERRIDES.get(cleaned_symbol, "")).strip().upper()
    if primary:
        contract.primaryExchange = primary
    return contract


def wait_for_realtime_bar_samples(
    symbols: Sequence[str],
    *,
    config: InteractiveBrokersRealtimeConfig,
    timeout_seconds: float = 20.0,
    status_callback: Callable[[str], None] | None = None,
    error_callback: Callable[[str], None] | None = None,
) -> list[LiveMarketBar]:
    received: dict[str, LiveMarketBar] = {}
    lock = threading.Lock()

    def _on_bar(bar: LiveMarketBar) -> None:
        with lock:
            received[bar.symbol] = bar

    app = InteractiveBrokersRealtimeBarApp(
        symbols=symbols,
        config=config,
        bar_callback=_on_bar,
        status_callback=status_callback,
        error_callback=error_callback,
    )
    app.connect_and_start()
    deadline = time.monotonic() + max(1.0, float(timeout_seconds))
    try:
        expected = {_clean_symbol(symbol) for symbol in list(symbols or []) if _clean_symbol(symbol)}
        while time.monotonic() < deadline:
            with lock:
                if expected and expected.issubset(received):
                    break
            time.sleep(0.05)
        with lock:
            return [received[symbol] for symbol in sorted(received)]
    finally:
        app.stop()
