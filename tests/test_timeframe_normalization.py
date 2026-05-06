import unittest

import pandas as pd

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


if __name__ == "__main__":
    unittest.main()
