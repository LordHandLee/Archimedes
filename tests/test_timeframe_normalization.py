import unittest
import tempfile
from pathlib import Path

import pandas as pd

from backtest_engine.duckdb_store import DuckDBStore
from backtest_engine.engine import BacktestEngine
from backtest_engine.metrics import _normalize_freq as normalize_metric_freq


class TimeframeNormalizationTests(unittest.TestCase):
    def test_plural_hours_normalize_for_pandas(self) -> None:
        self.assertEqual(BacktestEngine._normalize_freq("4 hours"), "4h")
        self.assertEqual(normalize_metric_freq("4 hours"), "4h")
        offset = pd.tseries.frequencies.to_offset(BacktestEngine._normalize_freq("4 hours"))
        self.assertEqual(offset.nanos, pd.Timedelta(hours=4).value)

    def test_singular_hours_and_minutes_still_normalize(self) -> None:
        self.assertEqual(BacktestEngine._normalize_freq("1 hour"), "1h")
        self.assertEqual(BacktestEngine._normalize_freq("1 hours"), "1h")
        self.assertEqual(BacktestEngine._normalize_freq("4 hrs"), "4h")
        self.assertEqual(BacktestEngine._normalize_freq("15 minutes"), "15min")
        self.assertEqual(normalize_metric_freq("1 hour"), "1h")
        self.assertEqual(normalize_metric_freq("1 hours"), "1h")
        self.assertEqual(normalize_metric_freq("4 hrs"), "4h")
        self.assertEqual(normalize_metric_freq("15 minutes"), "15min")

    def test_duckdb_resample_matches_reference_right_closed_bars(self) -> None:
        index = pd.date_range("2024-01-02 08:00", periods=12, freq="1h", tz="UTC")
        frame = pd.DataFrame(
            {
                "open": range(100, 112),
                "high": range(101, 113),
                "low": range(99, 111),
                "close": range(1000, 1012),
                "volume": [10.0] * 12,
            },
            index=index,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = DuckDBStore(db_path=root / "history.duckdb", data_dir=root / "parquet")
            store.write_parquet("sample", frame)

            actual = store.resample("sample", "4 hours")

        expected = frame.resample("4h", label="right", closed="right").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()
        expected.index.name = "timestamp"
        pd.testing.assert_frame_equal(actual, expected, check_freq=False)


if __name__ == "__main__":
    unittest.main()
