from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd
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

    def test_run_chart_loader_uses_duckdb_resample_for_non_base_timeframes(self) -> None:
        calls: list[tuple[str, str, object, object]] = []
        index = pd.date_range("2026-01-29 08:00", periods=3, freq="4h", tz="UTC")
        resampled = pd.DataFrame(
            {
                "open": [697.0, 696.74, 686.07],
                "high": [697.75, 697.76, 692.23],
                "low": [695.95, 685.19, 684.83],
                "close": [696.71, 686.05, 691.13],
                "volume": [3717.0, 752564.0, 893635.0],
            },
            index=index,
        )

        class FakeDuck:
            def load_range(self, dataset_id, start, end):
                calls.append(("load_range", dataset_id, start, end))
                return pd.DataFrame()

            def resample(self, dataset_id, timeframe):
                calls.append(("resample", dataset_id, timeframe, None))
                return resampled

        original_duck = ui.DuckDBStore
        try:
            ui.DuckDBStore = FakeDuck
            run = SimpleNamespace(
                dataset_id="SPY_interactive_brokers_10y_1m",
                timeframe="4 hours",
                start="2026-01-29 08:00:00+00:00",
                end="2026-01-29 16:00:00+00:00",
            )
            bars = ui.RunChartDialog._load_bars_utc_static(run)
        finally:
            ui.DuckDBStore = original_duck

        self.assertIsNotNone(bars)
        self.assertEqual(calls, [("resample", "SPY_interactive_brokers_10y_1m", "4 hours", None)])
        self.assertEqual(list(bars.index), list(index))
        self.assertAlmostEqual(float(bars.loc[index[1], "low"]), 685.19)

    def test_describe_dataset_uses_parquet_metadata_without_full_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ui.DuckDBStore(
                db_path=Path(tmpdir) / "history.duckdb",
                data_dir=Path(tmpdir) / "parquet",
            )
            frame = pd.DataFrame(
                {
                    "timestamp": pd.date_range("2026-01-02 14:30", periods=4, freq="1min", tz="UTC"),
                    "open": [100.0, 101.0, 102.0, 103.0],
                    "high": [101.0, 102.0, 103.0, 104.0],
                    "low": [99.0, 100.0, 101.0, 102.0],
                    "close": [100.5, 101.5, 102.5, 103.5],
                    "volume": [1000.0, 1100.0, 1200.0, 1300.0],
                }
            )
            store.write_parquet("fast_meta", frame)
            original_load = ui.DuckDBStore.load

            def _load_raises(*_args, **_kwargs):
                raise AssertionError("describe_dataset should not load the full parquet file")

            try:
                ui.DuckDBStore.load = _load_raises
                meta = store.describe_dataset("fast_meta")
            finally:
                ui.DuckDBStore.load = original_load
                store.close()

        self.assertEqual(meta["bar_count"], 4)
        self.assertEqual(meta["coverage_start"], "2026-01-02T14:30:00+00:00")
        self.assertEqual(meta["coverage_end"], "2026-01-02T14:33:00+00:00")

    def test_run_chart_loader_matches_reference_resample_for_reference_runs(self) -> None:
        calls: list[tuple[str, str, object, object]] = []
        index = pd.to_datetime(
            [
                "2026-01-29 04:01:00+00:00",
                "2026-01-29 08:00:00+00:00",
                "2026-01-29 08:01:00+00:00",
                "2026-01-29 12:00:00+00:00",
                "2026-01-29 12:01:00+00:00",
                "2026-01-29 16:00:00+00:00",
            ],
            utc=True,
        )
        base = pd.DataFrame(
            {
                "open": [700.0, 699.0, 698.0, 690.0, 689.0, 692.0],
                "high": [701.0, 700.0, 699.0, 691.0, 690.0, 693.0],
                "low": [699.0, 698.0, 697.0, 689.0, 688.0, 691.0],
                "close": [700.5, 698.5, 698.5, 690.5, 689.5, 692.5],
                "volume": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            },
            index=index,
        )

        class FakeDuck:
            def load_range(self, dataset_id, start, end):
                calls.append(("load_range", dataset_id, start, end))
                return base

            def resample(self, dataset_id, timeframe):
                calls.append(("resample", dataset_id, timeframe, None))
                return pd.DataFrame()

        original_duck = ui.DuckDBStore
        try:
            ui.DuckDBStore = FakeDuck
            run = SimpleNamespace(
                dataset_id="SPY_interactive_brokers_10y_1m",
                timeframe="4 hours",
                start="2026-01-29 08:00:00+00:00",
                end="2026-01-29 16:00:00+00:00",
                engine_impl="reference",
            )
            bars = ui.RunChartDialog._load_bars_utc_static(run)
        finally:
            ui.DuckDBStore = original_duck

        self.assertIsNotNone(bars)
        self.assertEqual(calls[0][0], "load_range")
        self.assertEqual(calls[0][2], pd.Timestamp("2026-01-29 04:00:00+00:00"))
        self.assertEqual(list(bars.index), list(pd.date_range("2026-01-29 08:00", periods=3, freq="4h", tz="UTC")))
        self.assertAlmostEqual(float(bars.iloc[0]["open"]), 700.0)
        self.assertAlmostEqual(float(bars.iloc[1]["low"]), 689.0)
        self.assertAlmostEqual(float(bars.iloc[2]["volume"]), 110.0)

    def test_regime_timeframe_param_uses_dropdown_options(self) -> None:
        original_refresh = ui.DashboardWindow.refresh

        def _refresh_stub(self, refresh_heatmap: bool = True) -> None:
            return None

        try:
            ui.DashboardWindow.refresh = _refresh_stub
            with tempfile.TemporaryDirectory() as tmpdir:
                window = ui.DashboardWindow(Path(tmpdir) / "catalog.sqlite")
                window.strategy_combo.setCurrentText("Regime Filtered OU Kalman Z-Score")
                widget = window.param_inputs["regime_timeframe"]
                self.assertIsInstance(widget, QtWidgets.QComboBox)
                self.assertEqual([widget.itemText(i) for i in range(widget.count())], [
                    "Same as chart",
                    "5 min",
                    "15 min",
                    "1 hour",
                    "4 hour",
                    "1 day",
                ])

                widget.setCurrentIndex(2)
                strategy_cls, params, errors = window._collect_strategy_params()
                self.assertEqual(strategy_cls.__name__, "RegimeFilteredAdaptiveOUKalmanZScoreStrategy")
                self.assertEqual(errors, [])
                self.assertEqual(params["regime_timeframe"], ["15 minutes"])
                window.close()
        finally:
            ui.DashboardWindow.refresh = original_refresh

    def test_known_choice_strategy_params_use_dropdowns(self) -> None:
        original_refresh = ui.DashboardWindow.refresh

        def _refresh_stub(self, refresh_heatmap: bool = True) -> None:
            return None

        try:
            ui.DashboardWindow.refresh = _refresh_stub
            with tempfile.TemporaryDirectory() as tmpdir:
                window = ui.DashboardWindow(Path(tmpdir) / "catalog.sqlite")
                for strategy_name, spec in window.strategy_specs.items():
                    window.strategy_combo.setCurrentText(strategy_name)
                    for param_name, (ptype, _default) in dict(spec.get("params") or {}).items():
                        widget = window.param_inputs[param_name]
                        if ptype in {bool, str}:
                            self.assertIsInstance(widget, QtWidgets.QComboBox, f"{strategy_name}.{param_name}")
                            self.assertGreater(widget.count(), 1, f"{strategy_name}.{param_name}")
                        else:
                            self.assertIsInstance(widget, QtWidgets.QLineEdit, f"{strategy_name}.{param_name}")

                window.strategy_combo.setCurrentText("SMA Trend Regime")
                false_bool_widget = window.param_inputs["enter_on_initial_state"]
                self.assertIsInstance(false_bool_widget, QtWidgets.QComboBox)
                self.assertEqual(false_bool_widget.currentData(), "False")

                mode_widget = window.param_inputs["regime_mode"]
                self.assertIsInstance(mode_widget, QtWidgets.QComboBox)
                ui._set_combo_to_data(mode_widget, "stacked")
                strategy_cls, params, errors = window._collect_strategy_params()
                self.assertEqual(strategy_cls.__name__, "SMATrendRegimeStrategy")
                self.assertEqual(errors, [])
                self.assertEqual(params["regime_mode"], ["stacked"])
                self.assertEqual(params["enter_on_initial_state"], [False])
                window.close()
        finally:
            ui.DashboardWindow.refresh = original_refresh

    def test_regime_filtered_kalman_is_allowed_by_vectorized_support_gate(self) -> None:
        fake_window = SimpleNamespace(intrabar_chk=SimpleNamespace(isChecked=lambda: False))

        issues, can_apply_defaults = ui.DashboardWindow._vectorized_support_issues(
            fake_window,
            ui.RegimeFilteredAdaptiveOUKalmanZScoreStrategy,
            ["4 hours"],
            {"execution_mode": ui.ExecutionMode.AUTO.value},
        )

        self.assertEqual(issues, [])
        self.assertTrue(can_apply_defaults)


if __name__ == "__main__":
    unittest.main()
