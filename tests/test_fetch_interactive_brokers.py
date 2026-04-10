from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "fetch_interactive_brokers.py"
_SPEC = importlib.util.spec_from_file_location("fetch_interactive_brokers_test_module", _SCRIPT_PATH)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


class InteractiveBrokersFetchHelperTests(unittest.TestCase):
    def test_normalize_bar_size_accepts_common_aliases(self) -> None:
        self.assertEqual(_MODULE._normalize_bar_size("1m"), "1 min")
        self.assertEqual(_MODULE._normalize_bar_size("1d"), "1 day")

    def test_normalize_bar_size_accepts_native_interactive_brokers_values(self) -> None:
        self.assertEqual(_MODULE._normalize_bar_size("1 min"), "1 min")
        self.assertEqual(_MODULE._normalize_bar_size("5 mins"), "5 mins")
        self.assertEqual(_MODULE._normalize_bar_size("1 hour"), "1 hour")

    def test_ensure_utc_timestamp_localizes_naive_values(self) -> None:
        stamp = _MODULE._ensure_utc_timestamp("2026-01-01 12:34:56")
        self.assertEqual(stamp.tz.zone if hasattr(stamp.tz, "zone") else str(stamp.tz), "UTC")
        self.assertEqual(stamp, pd.Timestamp("2026-01-01 12:34:56", tz="UTC"))

    def test_ensure_utc_timestamp_converts_aware_values(self) -> None:
        stamp = _MODULE._ensure_utc_timestamp(pd.Timestamp("2026-01-01 12:34:56", tz="America/New_York"))
        self.assertEqual(stamp, pd.Timestamp("2026-01-01 17:34:56", tz="UTC"))

    def test_chunk_windows_split_large_minute_ranges(self) -> None:
        start = pd.Timestamp("2020-01-01", tz="UTC")
        end = pd.Timestamp("2022-06-01", tz="UTC")
        windows = _MODULE._chunk_windows(
            start=start,
            end=end,
            chunk_offset=pd.DateOffset(years=1),
        )
        self.assertGreaterEqual(len(windows), 3)
        self.assertEqual(windows[0][0], start)
        self.assertEqual(windows[-1][1], end)

    def test_duration_string_for_short_window_uses_days(self) -> None:
        start = pd.Timestamp("2026-01-01", tz="UTC")
        end = pd.Timestamp("2026-01-10", tz="UTC")
        self.assertEqual(_MODULE._duration_string_for_window(start, end), "2 W")

    def test_default_chunk_offset_for_one_minute_is_more_conservative(self) -> None:
        offset = _MODULE._default_chunk_offset("1 min")
        self.assertEqual(getattr(offset, "weeks", None), 1)

    def test_adaptive_retry_splits_timed_out_window(self) -> None:
        calls: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        start = pd.Timestamp("2026-01-01", tz="UTC")
        end = pd.Timestamp("2026-02-01", tz="UTC")

        def _request(window_start: pd.Timestamp, window_end: pd.Timestamp):
            calls.append((window_start, window_end))
            if (window_end - window_start) > pd.Timedelta(days=20):
                raise TimeoutError("Timed out waiting for Interactive Brokers historical data.")
            return _MODULE._HistoricalResult(rows=[], start="", end="")

        results = _MODULE._fetch_window_with_adaptive_retry(
            _request,
            window_start=start,
            window_end=end,
            bar_size="1 min",
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(calls[0], (start, end))
        self.assertEqual(calls[1][0], start)
        self.assertEqual(calls[-1][1], end)

    def test_benign_historical_errors_are_detected(self) -> None:
        self.assertTrue(_MODULE._is_benign_historical_error(162, "API historical data query cancelled: 9001"))
        self.assertTrue(_MODULE._is_benign_historical_error(366, "No historical data query found for ticker id:9002"))
        self.assertFalse(_MODULE._is_benign_historical_error(162, "HMDS query returned no data"))


if __name__ == "__main__":
    unittest.main()
