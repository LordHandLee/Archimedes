from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from backtest_engine import run_ids


class BacktestSourceGuardTest(unittest.TestCase):
    def test_raises_when_backtest_source_changed_after_import(self) -> None:
        with (
            patch.object(run_ids, "IMPORTED_BACKTEST_SOURCE_FINGERPRINT", "loaded"),
            patch.object(run_ids, "current_backtest_source_fingerprint", return_value="current"),
            patch.dict(os.environ, {}, clear=True),
        ):
            with self.assertRaisesRegex(RuntimeError, "Restart the dashboard"):
                run_ids.assert_backtest_source_current()

    def test_environment_override_allows_stale_source(self) -> None:
        with (
            patch.object(run_ids, "IMPORTED_BACKTEST_SOURCE_FINGERPRINT", "loaded"),
            patch.object(run_ids, "current_backtest_source_fingerprint", return_value="current"),
            patch.dict(os.environ, {"BACKTEST_ALLOW_STALE_SOURCE": "1"}, clear=True),
        ):
            run_ids.assert_backtest_source_current()


if __name__ == "__main__":
    unittest.main()
