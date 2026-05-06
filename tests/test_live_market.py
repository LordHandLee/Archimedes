from __future__ import annotations

import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_engine.live_market import (
    DEFAULT_LIVE_PROVIDER,
    LIVE_BAR_TIMEFRAME,
    InteractiveBrokersRealtimeBarApp,
    InteractiveBrokersRealtimeConfig,
    LiveMarketBar,
    LiveMarketDataStore,
    _IBAPI_IMPORT_ERROR,
    compute_chart_indicators,
    incremental_series_payload,
    latest_point_series_payload,
    resample_ohlcv,
    series_replacement_payload,
)


class LiveMarketStoreTests(unittest.TestCase):
    def test_watchlist_and_live_bars_are_persisted_separately(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LiveMarketDataStore(Path(tmpdir) / "live_market.sqlite")
            store.save_watchlist(["spy", "AAPL", "SPY", "msft"])
            self.assertEqual(store.load_watchlist(), ["SPY", "AAPL", "MSFT"])

            bar = LiveMarketBar(
                symbol="SPY",
                timestamp=pd.Timestamp("2026-04-17T19:59:00Z"),
                open=500.0,
                high=501.0,
                low=499.5,
                close=500.75,
                volume=1200.0,
            )
            store.upsert_bar(bar)

            quotes = store.latest_quotes(["SPY"], provider=DEFAULT_LIVE_PROVIDER, stale_after_seconds=10_000_000)
            self.assertEqual(quotes[0]["symbol"], "SPY")
            self.assertEqual(quotes[0]["price"], 500.75)
            self.assertFalse(quotes[0]["stale"])

            frame = store.load_recent_bars("SPY", provider=DEFAULT_LIVE_PROVIDER, timeframe=LIVE_BAR_TIMEFRAME)
            self.assertEqual(len(frame), 1)
            self.assertEqual(float(frame.iloc[0]["close"]), 500.75)

    def test_latest_quotes_include_regular_session_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LiveMarketDataStore(Path(tmpdir) / "live_market.sqlite")
            session_open = pd.Timestamp.now(tz="America/New_York").normalize() + pd.Timedelta(hours=9, minutes=30)
            session_open_utc = session_open.tz_convert("UTC")
            for timestamp, close in (
                (session_open_utc, 100.0),
                (session_open_utc + pd.Timedelta(minutes=30), 103.0),
            ):
                store.upsert_bar(
                    LiveMarketBar(
                        symbol="AAPL",
                        timestamp=pd.Timestamp(timestamp),
                        open=close,
                        high=close,
                        low=close,
                        close=close,
                    )
                )

            quotes = store.latest_quotes(["AAPL"], provider=DEFAULT_LIVE_PROVIDER, stale_after_seconds=10_000_000)
            self.assertEqual(quotes[0]["session_open_price"], 100.0)
            self.assertAlmostEqual(float(quotes[0]["change_percent"]), 3.0)

    def test_missing_quote_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LiveMarketDataStore(Path(tmpdir) / "live_market.sqlite")
            self.assertTrue(store.quote_is_stale("AAPL", provider=DEFAULT_LIVE_PROVIDER))

    def test_unopenable_store_path_raises_sqlite_error_with_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            blocked_parent = Path(tmpdir) / "not_a_directory"
            blocked_parent.write_text("blocked", encoding="utf-8")

            with self.assertRaises(sqlite3.OperationalError) as caught:
                LiveMarketDataStore(blocked_parent / "live_market.sqlite")

            self.assertIn("live_market.sqlite", str(caught.exception))

    def test_upsert_bar_retries_transient_sqlite_write_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "live_market.sqlite"
            store = LiveMarketDataStore(
                db_path,
                sqlite_timeout_seconds=0.05,
                lock_retry_timeout_seconds=1.0,
            )
            blocker = sqlite3.connect(db_path, timeout=0.05, check_same_thread=False)
            blocker.execute("BEGIN IMMEDIATE")

            def release_lock() -> None:
                time.sleep(0.2)
                blocker.commit()
                blocker.close()

            release_thread = threading.Thread(target=release_lock)
            release_thread.start()
            try:
                store.upsert_bar(
                    LiveMarketBar(
                        symbol="SOXL",
                        timestamp=pd.Timestamp("2026-04-22T20:00:00Z"),
                        open=10.0,
                        high=11.0,
                        low=9.0,
                        close=10.5,
                    )
                )
            finally:
                release_thread.join(timeout=2.0)

            frame = store.load_recent_bars("SOXL")
            self.assertEqual(len(frame), 1)
            self.assertAlmostEqual(float(frame.iloc[0]["close"]), 10.5)

    def test_upsert_bars_batches_historical_sync_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LiveMarketDataStore(Path(tmpdir) / "live_market.sqlite")
            bars = [
                LiveMarketBar(
                    symbol="SOXL",
                    timestamp=pd.Timestamp("2026-04-22T20:00:00Z") + pd.Timedelta(minutes=idx),
                    open=10.0 + idx,
                    high=11.0 + idx,
                    low=9.0 + idx,
                    close=10.5 + idx,
                )
                for idx in range(3)
            ]

            self.assertEqual(store.upsert_bars(bars, chunk_size=2), 3)

            frame = store.load_recent_bars("SOXL")
            self.assertEqual(len(frame), 3)
            self.assertAlmostEqual(float(frame.iloc[-1]["close"]), 12.5)


@unittest.skipIf(_IBAPI_IMPORT_ERROR is not None, "ibapi is not installed")
class InteractiveBrokersRealtimeBarAppTests(unittest.TestCase):
    def test_preview_partial_bars_emit_without_persisting_before_closed_bar(self) -> None:
        events: list[tuple[str, pd.Timestamp]] = []
        app = InteractiveBrokersRealtimeBarApp(
            symbols=["SOXL"],
            config=InteractiveBrokersRealtimeConfig(),
            bar_callback=lambda bar: events.append(("bar", bar.timestamp)),
            partial_callback=lambda bar: events.append(("partial", bar.timestamp)),
        )
        req_id = 1
        app._req_symbols[req_id] = "SOXL"

        app.realtimeBar(req_id, int(pd.Timestamp("2026-04-23T13:00:05Z").timestamp()), 1, 2, 0.5, 1.5, 10, 1.5, 1)
        app.realtimeBar(req_id, int(pd.Timestamp("2026-04-23T13:00:10Z").timestamp()), 1.5, 2.5, 1.0, 2.0, 12, 2.0, 1)
        app.realtimeBar(req_id, int(pd.Timestamp("2026-04-23T13:01:00Z").timestamp()), 2.0, 3.0, 1.5, 2.5, 8, 2.5, 1)

        self.assertEqual(
            events,
            [
                ("partial", pd.Timestamp("2026-04-23T13:00:00Z")),
                ("partial", pd.Timestamp("2026-04-23T13:00:00Z")),
                ("bar", pd.Timestamp("2026-04-23T13:00:00Z")),
                ("partial", pd.Timestamp("2026-04-23T13:01:00Z")),
            ],
        )


class LiveMarketIndicatorTests(unittest.TestCase):
    def test_resampled_timeframes_use_bar_start_timestamps(self) -> None:
        index = pd.date_range("2026-04-21T13:30:00Z", periods=31, freq="min")
        frame = pd.DataFrame(
            {
                "open": np.arange(len(index), dtype=float),
                "high": np.arange(len(index), dtype=float) + 1.0,
                "low": np.arange(len(index), dtype=float) - 1.0,
                "close": np.arange(len(index), dtype=float) + 0.5,
                "volume": np.ones(len(index), dtype=float),
            },
            index=index,
        )

        bars = resample_ohlcv(frame, "15m")

        self.assertEqual(pd.Timestamp(bars.index[0]), pd.Timestamp("2026-04-21T13:30:00Z"))
        self.assertEqual(pd.Timestamp(bars.index[1]), pd.Timestamp("2026-04-21T13:45:00Z"))

    def test_indicator_payloads_include_overlays_and_panes(self) -> None:
        index = pd.date_range("2026-04-17T14:30:00Z", periods=80, freq="min")
        close = pd.Series(
            110.0 + np.sin(np.linspace(0.0, 8.0, len(index))) * 5.0 + np.linspace(0.0, 2.0, len(index)),
            index=index,
        )
        frame = pd.DataFrame(
            {
                "open": close.shift(1).fillna(close.iloc[0]),
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": np.arange(1, len(index) + 1, dtype=float),
            },
            index=index,
        )

        overlays, panes, styles = compute_chart_indicators(frame, ["sma20", "sma50", "ema20", "vwap", "rsi14"])
        self.assertIn("SMA 20", overlays)
        self.assertIn("SMA 50", overlays)
        self.assertIn("EMA 20", overlays)
        self.assertIn("VWAP", overlays)
        self.assertIn("RSI 14", panes)

        overlay_payload = incremental_series_payload(
            overlays,
            bar_index=len(frame) - 1,
            timestamp=frame.index[-1],
            styles=styles,
        )
        pane_payload = incremental_series_payload(
            panes,
            bar_index=len(frame) - 1,
            timestamp=frame.index[-1],
            styles=styles,
        )
        self.assertTrue(any(item["name"] == "SMA 20" for item in overlay_payload))
        self.assertTrue(any(item["name"] == "RSI 14" for item in pane_payload))

        mid_bar_index = 40
        mid_payload = incremental_series_payload(
            {"SMA 20": overlays["SMA 20"]},
            bar_index=mid_bar_index,
            timestamp=frame.index[mid_bar_index],
            styles=styles,
        )
        self.assertEqual(mid_payload[0]["points"][0]["bar_index"], mid_bar_index)
        self.assertIsInstance(mid_payload[0]["points"][0]["timestamp_utc_ns"], str)
        self.assertAlmostEqual(
            float(mid_payload[0]["points"][0]["value"]),
            float(overlays["SMA 20"].iloc[mid_bar_index]),
        )

        replacement_payload = series_replacement_payload(overlays, styles=styles)
        sma_payload = next(item for item in replacement_payload if item["name"] == "SMA 20")
        self.assertGreater(len(sma_payload["points"]), 1)
        self.assertIn("color", sma_payload)

        bounded_payload = series_replacement_payload(overlays, styles=styles, max_points=5)
        bounded_sma = next(item for item in bounded_payload if item["name"] == "SMA 20")
        self.assertLessEqual(len(bounded_sma["points"]), 5)
        self.assertGreater(bounded_sma["points"][0]["bar_index"], 0)

    def test_latest_point_payload_carries_equity_forward_to_live_bar(self) -> None:
        equity_index = pd.to_datetime(
            ["2026-04-22T12:25:00Z", "2026-04-22T12:40:00Z"],
            utc=True,
        )
        equity = pd.Series([100_000.0, 100_325.50], index=equity_index, name="equity")
        live_bar_ts = pd.Timestamp("2026-04-22T13:00:00Z")

        payload = latest_point_series_payload(
            {"equity": equity},
            bar_index=42,
            timestamp=live_bar_ts,
            styles={"equity": {"color": "#4da3ff"}},
        )

        self.assertEqual(len(payload), 1)
        self.assertEqual(payload[0]["name"], "equity")
        self.assertEqual(payload[0]["color"], "#4da3ff")
        self.assertEqual(payload[0]["points"][0]["bar_index"], 42)
        self.assertEqual(payload[0]["points"][0]["timestamp_utc_ns"], str(int(live_bar_ts.value)))
        self.assertAlmostEqual(float(payload[0]["points"][0]["value"]), 100_325.50)

    def test_resample_ohlcv_higher_timeframe(self) -> None:
        index = pd.date_range("2026-04-20T13:30:00Z", periods=11, freq="min")
        frame = pd.DataFrame(
            {
                "open": np.arange(11, dtype=float),
                "high": np.arange(11, dtype=float) + 1.0,
                "low": np.arange(11, dtype=float) - 1.0,
                "close": np.arange(11, dtype=float) + 0.5,
                "volume": np.ones(11, dtype=float),
            },
            index=index,
        )

        resampled = resample_ohlcv(frame, "5m")
        self.assertGreaterEqual(len(resampled), 3)
        self.assertEqual(pd.Timestamp(resampled.index[0]), pd.Timestamp("2026-04-20T13:30:00Z"))
        self.assertEqual(pd.Timestamp(resampled.index[1]), pd.Timestamp("2026-04-20T13:35:00Z"))
        self.assertEqual(float(resampled.iloc[-1]["close"]), 10.5)
        self.assertEqual(float(resampled.iloc[0]["volume"]), 5.0)
        self.assertEqual(float(resampled.iloc[-1]["volume"]), 1.0)


if __name__ == "__main__":
    unittest.main()
