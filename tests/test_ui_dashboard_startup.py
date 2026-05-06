from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtWidgets

import ui_qt_dashboard as ui
from backtest_engine.catalog import ResultCatalog


class DashboardStartupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_asset_tabs_refresh_on_demand_instead_of_constructor(self) -> None:
        calls = {"asset_information": 0, "asset_screener": 0}
        original_refresh = ui.DashboardWindow.refresh
        original_asset_information_refresh = ui.DashboardWindow._refresh_asset_information_tab
        original_asset_screener_refresh = ui.DashboardWindow._refresh_asset_screener_tab

        def _refresh_stub(self, refresh_heatmap: bool = True) -> None:
            return None

        def _asset_information_stub(self, *args, **kwargs) -> None:
            calls["asset_information"] += 1

        def _asset_screener_stub(self, *args, **kwargs) -> None:
            calls["asset_screener"] += 1

        try:
            ui.DashboardWindow.refresh = _refresh_stub
            ui.DashboardWindow._refresh_asset_information_tab = _asset_information_stub
            ui.DashboardWindow._refresh_asset_screener_tab = _asset_screener_stub
            with tempfile.TemporaryDirectory() as tmpdir:
                window = ui.DashboardWindow(Path(tmpdir) / "catalog.sqlite")
                self.assertEqual(calls["asset_information"], 0)
                self.assertEqual(calls["asset_screener"], 0)

                window.tabs.setCurrentIndex(window.tabs.indexOf(window.asset_information_tab))
                self._app.processEvents()
                self.assertEqual(calls["asset_information"], 1)

                window.tabs.setCurrentIndex(window.tabs.indexOf(window.asset_screener_tab))
                self._app.processEvents()
                self.assertEqual(calls["asset_screener"], 1)
                window.close()
        finally:
            ui.DashboardWindow.refresh = original_refresh
            ui.DashboardWindow._refresh_asset_information_tab = original_asset_information_refresh
            ui.DashboardWindow._refresh_asset_screener_tab = original_asset_screener_refresh

    def test_constructor_leaves_universe_editor_blank_until_user_selects_one(self) -> None:
        original_refresh = ui.DashboardWindow.refresh

        def _refresh_stub(self, refresh_heatmap: bool = True) -> None:
            return None

        try:
            ui.DashboardWindow.refresh = _refresh_stub
            with tempfile.TemporaryDirectory() as tmpdir:
                db_path = Path(tmpdir) / "catalog.sqlite"
                catalog = ResultCatalog(db_path)
                catalog.save_universe(
                    universe_id="universe_all_us",
                    name="All US Equities",
                    symbols=["AAPL", "MSFT", "SPY"],
                    dataset_ids=[],
                )

                window = ui.DashboardWindow(db_path)
                selection_model = window.universe_table.selectionModel()
                self.assertIsNotNone(selection_model)
                self.assertEqual(window._editing_universe_id, "")
                self.assertEqual(selection_model.selectedRows(), [])
                self.assertEqual(window.universe_symbols_edit.toPlainText(), "")
                window.close()
        finally:
            ui.DashboardWindow.refresh = original_refresh

    def test_refresh_reuses_loaded_runs_for_batches(self) -> None:
        class FakeCatalog:
            def __init__(self) -> None:
                self.db_path = Path("catalog.sqlite")
                self.runs = [{"metrics": {}}]
                self.load_runs_calls = 0
                self.load_batches_args: list[object] = []

            def load_runs(self):
                self.load_runs_calls += 1
                return self.runs

            def load_batches(self, runs=None):
                self.load_batches_args.append(runs)
                return ["batch-1"]

        class FakeLabel:
            def __init__(self) -> None:
                self.text = ""

            def setText(self, text: str) -> None:
                self.text = text

        class FakeWindow:
            pass

        window = FakeWindow()
        window.catalog = FakeCatalog()
        window.status_label = FakeLabel()
        window._load_tasks = lambda: None
        window._render_batches = lambda batches: None
        window._update_metrics = lambda runs: None
        window._refresh_visible_dashboard_panels = lambda refresh_heatmap=True: None

        ui.DashboardWindow.refresh(window)

        self.assertEqual(window.catalog.load_runs_calls, 1)
        self.assertEqual(window.catalog.load_batches_args, [window.catalog.runs])
        self.assertIn("1 runs", window.status_label.text)
        self.assertIn("1 batches", window.status_label.text)

    def test_charts_watchlist_streams_start_only_when_charts_tab_is_visible(self) -> None:
        started_symbols: list[str] = []
        original_start = ui.ChartLiveStreamWorker.start
        original_live_market_db_path = ui.DEFAULT_LIVE_MARKET_DB_PATH

        def _start_stub(self, *args, **kwargs) -> None:
            started_symbols.append(str(getattr(self, "symbol", "") or ""))

        try:
            ui.ChartLiveStreamWorker.start = _start_stub
            with tempfile.TemporaryDirectory() as tmpdir:
                ui.DEFAULT_LIVE_MARKET_DB_PATH = Path(tmpdir) / "live_market.sqlite"
                window = ui.DashboardWindow(Path(tmpdir) / "catalog.sqlite")

                self.assertEqual(started_symbols, [])
                self.assertEqual(window.charts_watchlist_stream_workers, {})

                charts_idx = window.tabs.indexOf(window.charts_tab)
                window.tabs.setCurrentIndex(charts_idx)
                self._app.processEvents()

                self.assertCountEqual(started_symbols, ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"])
                self.assertCountEqual(window.charts_watchlist_stream_workers.keys(), ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"])

                window.tabs.setCurrentIndex(0)
                self._app.processEvents()

                self.assertEqual(window.charts_watchlist_stream_workers, {})
                window.close()
        finally:
            ui.ChartLiveStreamWorker.start = original_start
            ui.DEFAULT_LIVE_MARKET_DB_PATH = original_live_market_db_path


if __name__ == "__main__":
    unittest.main()
