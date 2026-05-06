import os
import tempfile
import time
import unittest

os.environ.setdefault("MPLCONFIGDIR", "/tmp/quant-backtest-engine-matplotlib")

import numpy as np
import pandas as pd

from backtest_engine.chart_snapshot import ChartSnapshotExporter
from backtest_engine.live_market import InteractiveBrokersRealtimeConfig
from ui_qt_dashboard import (
    IB_CLIENT_OFFSET_LIVE_DEPLOYMENT,
    IB_CLIENT_OFFSET_LIVE_MONITOR,
    IB_CLIENT_OFFSET_LIVE_MONITOR_SYNC,
    ChartHistoricalSyncWorker,
    DashboardWindow,
    _market_symbol_from_dataset_id,
)


class ChartSnapshotTimestampTests(unittest.TestCase):
    def test_snapshot_timestamp_columns_are_epoch_nanoseconds(self) -> None:
        index = pd.DatetimeIndex(
            np.array(["2026-04-30T12:15:00", "2026-04-30T12:30:00"], dtype="datetime64[us]")
        ).tz_localize("UTC")
        bars = pd.DataFrame(
            {
                "open": [1.0, 2.0],
                "high": [1.5, 2.5],
                "low": [0.5, 1.5],
                "close": [1.2, 2.2],
                "volume": [10.0, 20.0],
            },
            index=index,
        )

        price_bars = ChartSnapshotExporter._build_price_bars_dataframe(bars)
        overlays, _order = ChartSnapshotExporter._build_series_dataframe(
            bars,
            {"SMA": pd.Series([1.1, 2.1], index=index)},
        )
        equity = ChartSnapshotExporter._build_equity_dataframe(
            bars,
            pd.Series([100.0, 101.0], index=index),
        )

        expected = pd.Timestamp("2026-04-30T12:15:00Z").value
        self.assertEqual(int(price_bars.loc[0, "ts_utc_ns"]), expected)
        self.assertEqual(int(overlays.loc[0, "ts_utc_ns"]), expected)
        self.assertEqual(int(equity.loc[0, "ts_utc_ns"]), expected)
        self.assertGreater(expected, 1_000_000_000_000_000_000)


class MarketSymbolNormalizationTests(unittest.TestCase):
    def test_interactive_brokers_dataset_id_normalizes_to_ticker(self) -> None:
        self.assertEqual(_market_symbol_from_dataset_id("TQQQ_INTERACTIVE_BROKERS_10Y_1M"), "TQQQ")

    def test_massive_dataset_id_normalizes_to_ticker(self) -> None:
        self.assertEqual(_market_symbol_from_dataset_id("SOXL_massive_2y_1m"), "SOXL")

    def test_prefixed_provider_dataset_id_normalizes_to_ticker(self) -> None:
        self.assertEqual(_market_symbol_from_dataset_id("MASSIVE_SOXL_2024_04_22_1m"), "SOXL")

    def test_plain_symbol_is_preserved(self) -> None:
        self.assertEqual(_market_symbol_from_dataset_id("BRK.B"), "BRK.B")

    def test_dotted_symbol_dataset_id_preserves_symbol(self) -> None:
        self.assertEqual(_market_symbol_from_dataset_id("BRK.B_interactive_brokers_10y_1m"), "BRK.B")

    def test_historical_sync_worker_never_keeps_dataset_id_as_symbol(self) -> None:
        worker = ChartHistoricalSyncWorker(
            symbol="TQQQ_INTERACTIVE_BROKERS_10Y_1M",
            start=pd.Timestamp("2026-04-22 09:30", tz="UTC"),
            end=pd.Timestamp("2026-04-22 09:31", tz="UTC"),
            store_path="/tmp/live-market-test.sqlite",
            config=InteractiveBrokersRealtimeConfig(),
        )
        self.assertEqual(worker.symbol, "TQQQ")


class MagellanLiveUpdateIndexTests(unittest.TestCase):
    def test_new_live_bar_uses_contiguous_session_index(self) -> None:
        bar_index = DashboardWindow._magellan_live_update_bar_index(
            bar_ts_ns=200,
            fallback_index=50,
            last_sent_bar_ts_ns=100,
            last_sent_bar_index=10,
        )

        self.assertEqual(bar_index, 11)

    def test_same_live_bar_reuses_session_index(self) -> None:
        bar_index = DashboardWindow._magellan_live_update_bar_index(
            bar_ts_ns=100,
            fallback_index=50,
            last_sent_bar_ts_ns=100,
            last_sent_bar_index=10,
        )

        self.assertEqual(bar_index, 10)


class DeploymentStatusSummaryTests(unittest.TestCase):
    def test_armed_deployments_are_counted(self) -> None:
        frame = pd.DataFrame(
            [
                {"status": "armed"},
                {"status": "armed"},
                {"status": "draft"},
                {"status": "paused"},
                {"status": ""},
            ]
        )

        counts = DashboardWindow._deployment_status_counts(frame)

        self.assertEqual(counts["armed"], 2)
        self.assertEqual(counts["draft"], 2)
        self.assertEqual(counts["paused"], 1)


class LiveDeploymentRunnerTests(unittest.TestCase):
    def test_secret_loader_prefers_saved_target_secret(self) -> None:
        secret = DashboardWindow._deployment_secret_value(
            DashboardWindow.__new__(DashboardWindow),
            {"secret_value": "saved-secret", "secret_ref": "LIVE_WEBHOOK_SECRET"},
        )

        self.assertEqual(secret, "saved-secret")

    def test_completed_bar_index_waits_for_start_labeled_bar_close(self) -> None:
        bars = pd.DataFrame(
            {
                "open": [1.0, 2.0, 3.0],
                "high": [1.0, 2.0, 3.0],
                "low": [1.0, 2.0, 3.0],
                "close": [1.0, 2.0, 3.0],
                "volume": [10.0, 10.0, 10.0],
            },
            index=pd.to_datetime(
                [
                    "2026-04-22 10:45:00Z",
                    "2026-04-22 10:50:00Z",
                    "2026-04-22 10:55:00Z",
                ]
            ),
        )

        index = DashboardWindow._live_deployment_completed_bar_index(
            bars,
            pd.Timestamp("2026-04-22 10:55:00Z"),
            "5m",
        )

        self.assertEqual(index, 1)

        index = DashboardWindow._live_deployment_completed_bar_index(
            bars,
            pd.Timestamp("2026-04-22 11:00:00Z"),
            "5m",
        )

        self.assertEqual(index, 2)

    def test_completed_bar_index_uses_received_timestamp_for_stale_flush(self) -> None:
        bars = pd.DataFrame(
            {
                "open": [1.0],
                "high": [2.0],
                "low": [0.5],
                "close": [1.5],
                "volume": [10.0],
            },
            index=pd.to_datetime(["2026-04-22 23:45:00Z"]),
        )
        evaluation_ts = DashboardWindow._live_record_evaluation_timestamp(
            {
                "ts_utc": "2026-04-22T23:59:00Z",
                "received_at": "2026-04-23T00:00:05Z",
            }
        )

        index = DashboardWindow._live_deployment_completed_bar_index(bars, evaluation_ts, "15m")

        self.assertEqual(index, 0)

    def test_completed_bars_for_chart_timeframe_keeps_only_closed_start_labeled_bars(self) -> None:
        bars = pd.DataFrame(
            {
                "open": [1.0, 2.0, 3.0],
                "high": [1.0, 2.0, 3.0],
                "low": [1.0, 2.0, 3.0],
                "close": [1.0, 2.0, 3.0],
                "volume": [10.0, 10.0, 10.0],
            },
            index=pd.to_datetime(
                [
                    "2026-04-22 10:45:00Z",
                    "2026-04-22 10:50:00Z",
                    "2026-04-22 10:55:00Z",
                ]
            ),
        )

        completed = DashboardWindow._completed_bars_for_chart_timeframe(
            bars,
            pd.Timestamp("2026-04-22 10:52:00Z"),
            "5m",
        )

        self.assertEqual(len(completed), 1)
        self.assertEqual(pd.Timestamp(completed.index[-1]), pd.Timestamp("2026-04-22 10:45:00Z"))

    def test_live_monitor_bucket_timestamp_uses_bar_start(self) -> None:
        bucket = DashboardWindow._live_monitor_record_bucket_timestamp(
            pd.Timestamp("2026-04-22 16:26:00Z"),
            "5m",
        )

        self.assertEqual(bucket, pd.Timestamp("2026-04-22 16:25:00Z"))

    def test_signal_plan_enters_exits_and_flips(self) -> None:
        self.assertEqual(DashboardWindow._live_deployment_signal_plan(0.0, 1.0), [("ENTRY", "LONG", 1.0)])
        self.assertEqual(DashboardWindow._live_deployment_signal_plan(1.0, 1.0), [])
        self.assertEqual(DashboardWindow._live_deployment_signal_plan(1.0, 0.0), [("EXIT", "LONG", 0.0)])
        self.assertEqual(
            DashboardWindow._live_deployment_signal_plan(1.0, -1.0),
            [("EXIT", "LONG", 0.0), ("ENTRY", "SHORT", -1.0)],
        )

    def test_conflict_guard_blocks_same_target_symbol_contexts(self) -> None:
        window = DashboardWindow.__new__(DashboardWindow)
        window.live_deployment_execution_contexts = {}

        conflicts = DashboardWindow._live_deployment_context_conflicts(
            window,
            [
                {"target_id": "algo_engine_live", "symbol": "SOXL", "deployment_id": "a"},
                {"target_id": "algo_engine_live", "symbol": "SOXL", "deployment_id": "b"},
            ],
            "parent",
        )

        self.assertEqual(len(conflicts), 1)

    def test_payload_includes_execution_engine_sizing_override(self) -> None:
        context = {
            "secret": "test-secret",
            "symbol": "SOXL",
            "deployment_id": "child-1",
            "parent_deployment_id": "parent-1",
            "strategy_name": "SMACrossStrategy",
            "timeframe": "5m",
            "dataset_id": "SOXL",
            "sizing": {"qty_type": "fixed", "qty_value": 25},
            "params": {"fast": 2, "slow": 3},
        }

        payload = DashboardWindow._deployment_signal_payload(
            context,
            action="ENTRY",
            side="LONG",
            target_percent=1.0,
            bar_ts=pd.Timestamp("2026-04-22 10:55:00Z"),
            bar_index=42,
            price=40.0,
        )

        self.assertEqual(payload["secret"], "test-secret")
        self.assertEqual(payload["action"], "ENTRY")
        self.assertEqual(payload["side"], "LONG")
        self.assertEqual(payload["position_size_override"]["mode"], "target_qty")
        self.assertEqual(payload["position_size_override"]["target_qty"], 25.0)
        self.assertEqual(payload["position_size_override"]["deployment_id"], "child-1")
        self.assertEqual(payload["position_size_override"]["parent_deployment_id"], "parent-1")
        self.assertEqual(payload["sizing_authority"], "quant_backtest_engine")

    def test_percent_equity_payload_uses_target_notional_override(self) -> None:
        context = {
            "secret": "test-secret",
            "symbol": "SOXL",
            "deployment_id": "child-1",
            "parent_deployment_id": "parent-1",
            "strategy_name": "SMACrossStrategy",
            "timeframe": "15m",
            "dataset_id": "SOXL",
            "sizing": {
                "qty_type": "percent_equity",
                "qty_value": 5,
                "execution_config": {
                    "position_sizing_model": "none",
                    "margin_enabled": False,
                    "max_gross_leverage": 1.0,
                    "annual_vol_window": 252,
                    "annual_vol_min_periods": 20,
                    "annual_vol_floor": 0.05,
                    "max_volatility_multiplier": 2.0,
                    "min_position_shares": 1.0,
                },
                "min_shares": 1.0,
            },
            "params": {"fast": 2, "slow": 3},
            "account_snapshot": {"equity": 10_000.0},
        }

        payload = DashboardWindow._deployment_signal_payload(
            context,
            action="ENTRY",
            side="LONG",
            target_percent=1.0,
            bar_ts=pd.Timestamp("2026-04-22 10:55:00Z"),
            bar_index=42,
            price=40.0,
        )

        self.assertEqual(payload["position_size_override"]["mode"], "target_notional")
        self.assertAlmostEqual(payload["position_size_override"]["target_notional"], 500.0, places=6)
        self.assertAlmostEqual(payload["position_size_override"]["deployment_slice_equity"], 500.0, places=6)
        self.assertNotIn("annual_volatility", payload["position_size_override"])
        self.assertIsNone(payload["annual_vol"])

    def test_percent_equity_payload_applies_annual_vol_multiplier(self) -> None:
        bars = pd.DataFrame(
            {
                "open": [100.0, 100.2, 100.4],
                "high": [100.5, 100.7, 100.9],
                "low": [99.8, 100.0, 100.2],
                "close": [100.0, 100.1, 100.2],
                "volume": [10.0, 10.0, 10.0],
            },
            index=pd.to_datetime(
                [
                    "2026-04-20 00:00:00Z",
                    "2026-04-21 00:00:00Z",
                    "2026-04-22 00:00:00Z",
                ]
            ),
        )
        context = {
            "secret": "test-secret",
            "symbol": "TQQQ",
            "deployment_id": "child-1",
            "parent_deployment_id": "parent-1",
            "strategy_name": "SMACrossStrategy",
            "timeframe": "1d",
            "dataset_id": "TQQQ",
            "sizing": {
                "qty_type": "percent_equity",
                "qty_value": 50,
                "execution_config": {
                    "position_sizing_model": "annual_volatility_target",
                    "margin_enabled": True,
                    "max_gross_leverage": 2.0,
                    "annual_vol_window": 2,
                    "annual_vol_min_periods": 2,
                    "annual_vol_floor": 0.05,
                    "max_volatility_multiplier": 2.0,
                    "min_position_shares": 1.0,
                },
                "min_shares": 1.0,
            },
            "params": {"fast": 2, "slow": 3},
            "account_snapshot": {"equity": 10_000.0},
        }

        payload = DashboardWindow._deployment_signal_payload(
            context,
            action="ENTRY",
            side="LONG",
            target_percent=1.0,
            bar_ts=pd.Timestamp("2026-04-22 00:00:00Z"),
            bar_index=2,
            price=100.2,
            bars=bars,
        )

        self.assertEqual(payload["position_size_override"]["mode"], "target_notional")
        self.assertAlmostEqual(payload["position_size_override"]["base_target_notional"], 5000.0, places=6)
        self.assertAlmostEqual(payload["position_size_override"]["target_notional"], 10_000.0, places=6)
        self.assertAlmostEqual(payload["position_size_override"]["volatility_multiplier"], 2.0, places=6)
        self.assertGreater(float(payload["position_size_override"]["annual_volatility"]), 0.0)
        self.assertAlmostEqual(payload["position_size_override"]["effective_annual_volatility"], 0.5, places=6)
        self.assertAlmostEqual(float(payload["annual_vol"]), payload["position_size_override"]["annual_volatility"], places=12)

    def test_percent_equity_annual_vol_can_scale_above_slice_without_margin(self) -> None:
        bars = pd.DataFrame(
            {
                "open": [100.0, 100.2, 100.4],
                "high": [100.5, 100.7, 100.9],
                "low": [99.8, 100.0, 100.2],
                "close": [100.0, 100.1, 100.2],
                "volume": [10.0, 10.0, 10.0],
            },
            index=pd.to_datetime(
                [
                    "2026-04-20 00:00:00Z",
                    "2026-04-21 00:00:00Z",
                    "2026-04-22 00:00:00Z",
                ]
            ),
        )
        context = {
            "secret": "test-secret",
            "symbol": "TQQQ",
            "deployment_id": "child-1",
            "parent_deployment_id": "parent-1",
            "strategy_name": "SMACrossStrategy",
            "timeframe": "1d",
            "dataset_id": "TQQQ",
            "sizing": {
                "qty_type": "percent_equity",
                "qty_value": 5,
                "execution_config": {
                    "position_sizing_model": "annual_volatility_target",
                    "margin_enabled": False,
                    "max_gross_leverage": 2.0,
                    "annual_vol_window": 2,
                    "annual_vol_min_periods": 2,
                    "annual_vol_floor": 0.05,
                    "max_volatility_multiplier": 2.0,
                    "min_position_shares": 1.0,
                },
            },
            "params": {"fast": 2, "slow": 3},
            "account_snapshot": {"equity": 4_000.0},
            "sizing_config_source": "deployment",
        }

        payload = DashboardWindow._deployment_signal_payload(
            context,
            action="ENTRY",
            side="LONG",
            target_percent=1.0,
            bar_ts=pd.Timestamp("2026-04-22 00:00:00Z"),
            bar_index=2,
            price=100.2,
            bars=bars,
        )

        override = payload["position_size_override"]
        self.assertAlmostEqual(override["base_target_notional"], 200.0, places=6)
        self.assertAlmostEqual(override["target_notional"], 400.0, places=6)
        self.assertAlmostEqual(override["max_deployment_notional"], 400.0, places=6)
        self.assertAlmostEqual(override["volatility_multiplier"], 2.0, places=6)
        self.assertIn("annual_volatility", payload["sizing_trace"])
        self.assertIn("final=$400.00", DashboardWindow._deployment_signal_sizing_summary(payload))

    def test_fixed_sizing_with_annual_vol_stays_fixed_shares(self) -> None:
        bars = pd.DataFrame(
            {
                "open": [100.0, 100.2, 100.4],
                "high": [100.5, 100.7, 100.9],
                "low": [99.8, 100.0, 100.2],
                "close": [100.0, 100.1, 100.2],
                "volume": [10.0, 10.0, 10.0],
            },
            index=pd.to_datetime(
                [
                    "2026-04-20 00:00:00Z",
                    "2026-04-21 00:00:00Z",
                    "2026-04-22 00:00:00Z",
                ]
            ),
        )
        context = {
            "secret": "test-secret",
            "symbol": "SMH",
            "deployment_id": "deploy-1",
            "strategy_name": "SMACrossStrategy",
            "timeframe": "1d",
            "dataset_id": "SMH",
            "sizing": {
                "qty_type": "fixed",
                "qty_value": 5,
                "execution_config": {
                    "position_sizing_model": "annual_volatility_target",
                    "margin_enabled": True,
                    "max_gross_leverage": 2.0,
                    "annual_vol_window": 2,
                    "annual_vol_min_periods": 2,
                    "annual_vol_floor": 0.05,
                    "max_volatility_multiplier": 2.0,
                    "min_position_shares": 1.0,
                },
                "min_shares": 1.0,
            },
            "params": {"fast": 2, "slow": 3},
            "account_snapshot": {"equity": 4_000.0},
            "sizing_config_source": "deployment",
        }

        payload = DashboardWindow._deployment_signal_payload(
            context,
            action="ENTRY",
            side="LONG",
            target_percent=1.0,
            bar_ts=pd.Timestamp("2026-04-22 00:00:00Z"),
            bar_index=2,
            price=505.0,
            bars=bars,
        )

        override = payload["position_size_override"]
        self.assertEqual(override["mode"], "target_qty")
        self.assertEqual(override["target_qty"], 5.0)
        self.assertAlmostEqual(override["target_notional"], 2525.0, places=6)
        self.assertEqual(override["min_shares"], 1.0)

    def test_percent_equity_suppresses_min_shares_above_target_notional(self) -> None:
        bars = pd.DataFrame(
            {
                "open": [100.0, 100.2, 100.4],
                "high": [100.5, 100.7, 100.9],
                "low": [99.8, 100.0, 100.2],
                "close": [100.0, 100.1, 100.2],
                "volume": [10.0, 10.0, 10.0],
            },
            index=pd.to_datetime(
                [
                    "2026-04-20 00:00:00Z",
                    "2026-04-21 00:00:00Z",
                    "2026-04-22 00:00:00Z",
                ]
            ),
        )
        context = {
            "secret": "test-secret",
            "symbol": "SMH",
            "deployment_id": "deploy-1",
            "strategy_name": "SMACrossStrategy",
            "timeframe": "1d",
            "dataset_id": "SMH",
            "sizing": {
                "qty_type": "percent_equity",
                "qty_value": 5,
                "execution_config": {
                    "position_sizing_model": "annual_volatility_target",
                    "margin_enabled": True,
                    "max_gross_leverage": 2.0,
                    "annual_vol_window": 2,
                    "annual_vol_min_periods": 2,
                    "annual_vol_floor": 0.05,
                    "max_volatility_multiplier": 2.0,
                    "min_position_shares": 1.0,
                },
                "min_shares": 1.0,
            },
            "params": {"fast": 2, "slow": 3},
            "account_snapshot": {"equity": 4_000.0},
            "sizing_config_source": "deployment",
        }

        payload = DashboardWindow._deployment_signal_payload(
            context,
            action="ENTRY",
            side="LONG",
            target_percent=1.0,
            bar_ts=pd.Timestamp("2026-04-22 00:00:00Z"),
            bar_index=2,
            price=505.0,
            bars=bars,
        )

        override = payload["position_size_override"]
        self.assertEqual(override["mode"], "target_notional")
        self.assertAlmostEqual(override["base_target_notional"], 200.0, places=6)
        self.assertAlmostEqual(override["target_notional"], 400.0, places=6)
        self.assertAlmostEqual(override["max_deployment_notional"], 400.0, places=6)
        self.assertEqual(override["min_shares"], 0.0)
        self.assertEqual(override["requested_min_shares"], 1.0)
        self.assertTrue(override["min_shares_suppressed_by_target_notional"])
        self.assertIn("min_shares_suppressed=yes", DashboardWindow._deployment_signal_sizing_summary(payload))

    def test_deployment_sizing_config_summary_surfaces_saved_model(self) -> None:
        summary = DashboardWindow._deployment_sizing_config_summary(
            {
                "qty_type": "percent_equity",
                "qty_value": 5,
                "execution_config": {
                    "position_sizing_model": "none",
                    "margin_enabled": False,
                    "max_gross_leverage": 2.0,
                    "annual_vol_window": 252,
                    "annual_vol_min_periods": 20,
                    "annual_vol_floor": 0.05,
                    "max_volatility_multiplier": 2.0,
                },
            }
        )

        self.assertIn("percent_equity 5", summary)
        self.assertIn("model=none", summary)
        self.assertIn("margin=off", summary)

    def test_deployment_execution_config_summary_shows_annual_vol_model(self) -> None:
        summary = DashboardWindow._deployment_execution_config_summary(
            {
                "position_sizing_model": "annual_volatility_target",
                "margin_enabled": False,
                "max_gross_leverage": 2.0,
                "annual_vol_window": 252,
                "annual_vol_min_periods": 20,
                "annual_vol_floor": 0.05,
                "max_volatility_multiplier": 2.0,
                "min_position_shares": 1.0,
            }
        )

        self.assertIn("model=annual_volatility_target", summary)
        self.assertIn("vol_window=252", summary)
        self.assertIn("max_vol_mult=2", summary)

    def test_deployment_execution_config_summary_marks_missing_capture(self) -> None:
        summary = DashboardWindow._deployment_execution_config_summary({})

        self.assertIn("No execution sizing config", summary)

    def test_live_deployment_history_lookback_expands_for_annual_volatility(self) -> None:
        lookback_days = DashboardWindow._live_deployment_history_lookback_days(
            {
                "timeframe": "15m",
                "params": {"fast": 20, "slow": 50},
                "sizing": {
                    "execution_config": {
                        "position_sizing_model": "annual_volatility_target",
                        "annual_vol_window": 252,
                        "annual_vol_min_periods": 20,
                        "annual_vol_floor": 0.05,
                        "max_volatility_multiplier": 2.0,
                        "margin_enabled": True,
                        "max_gross_leverage": 2.0,
                        "min_position_shares": 1.0,
                    }
                },
            }
        )

        self.assertGreaterEqual(lookback_days, 410)


class MagellanWarmupTests(unittest.TestCase):
    def test_dashboard_warmup_uses_async_magellan_start(self) -> None:
        class FakeMagellan:
            def __init__(self) -> None:
                self.async_calls = 0
                self.ensure_calls = 0

            def warmup_async(self) -> bool:
                self.async_calls += 1
                return True

            def ensure_running(self, timeout_ms: int = 0) -> bool:
                self.ensure_calls += 1
                return True

        window = DashboardWindow.__new__(DashboardWindow)
        window._closing = False
        window.magellan = FakeMagellan()

        DashboardWindow._warm_magellan(window)

        self.assertEqual(window.magellan.async_calls, 1)
        self.assertEqual(window.magellan.ensure_calls, 0)


class PreviewBarRoutingTests(unittest.TestCase):
    def test_preview_bar_routes_to_live_monitor_without_runner_path(self) -> None:
        window = DashboardWindow.__new__(DashboardWindow)
        calls: list[tuple[str, dict]] = []

        class FakeLabel:
            def setText(self, _text: str) -> None:
                raise AssertionError("Preview-bar fast path should not surface an error message.")

        window.status_label = FakeLabel()
        window._send_live_monitor_magellan_update = lambda symbol, record: calls.append((symbol, dict(record)))

        DashboardWindow._on_charts_live_preview_bar(
            window,
            {
                "symbol": "SOXL",
                "is_partial": True,
                "ts_utc": "2026-04-22T11:01:00Z",
            },
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "SOXL")
        self.assertTrue(calls[0][1]["is_partial"])


class LiveMonitorMagellanUpdateTests(unittest.TestCase):
    def test_deployment_stream_start_does_not_stop_live_monitor_stream(self) -> None:
        class FakeWorker:
            def __init__(self) -> None:
                self.stopped = False

            def isRunning(self) -> bool:
                return True

            def stop(self) -> None:
                self.stopped = True

            def wait(self, _ms: int) -> None:
                return None

        monitor_worker = FakeWorker()
        window = DashboardWindow.__new__(DashboardWindow)
        window.live_monitor_chart_stream_workers = {"SOXL": monitor_worker}
        window.charts_stream_worker = None
        window._stop_charts_watchlist_stream = lambda _symbol: None
        window._stop_charts_live_stream = lambda *args, **kwargs: None

        DashboardWindow._stop_non_deployment_symbol_streams(window, "SOXL")

        self.assertIn("SOXL", window.live_monitor_chart_stream_workers)
        self.assertFalse(monitor_worker.stopped)

    def test_ib_client_offset_ranges_do_not_overlap_live_monitor_and_runner(self) -> None:
        self.assertNotEqual(IB_CLIENT_OFFSET_LIVE_MONITOR, IB_CLIENT_OFFSET_LIVE_MONITOR_SYNC)
        self.assertNotEqual(IB_CLIENT_OFFSET_LIVE_MONITOR, IB_CLIENT_OFFSET_LIVE_DEPLOYMENT)
        self.assertNotEqual(IB_CLIENT_OFFSET_LIVE_MONITOR_SYNC, IB_CLIENT_OFFSET_LIVE_DEPLOYMENT)

    def test_live_monitor_and_runner_share_symbol_stream_worker(self) -> None:
        class FakeWorker:
            def isRunning(self) -> bool:
                return True

        shared_worker = FakeWorker()
        acquired = []
        window = DashboardWindow.__new__(DashboardWindow)
        window.live_monitor_chart_stream_workers = {}
        window.live_deployment_stream_workers = {}
        window._symbol_from_dataset_id = lambda value: _market_symbol_from_dataset_id(value)
        window._stop_non_deployment_symbol_streams = lambda _symbol: None

        def acquire(symbol: str, *, client_offset: int, consumer_key: str):
            acquired.append((symbol, client_offset, consumer_key))
            return shared_worker

        window._acquire_live_symbol_stream = acquire

        DashboardWindow._start_live_monitor_chart_stream(window, "SOXL", client_offset=IB_CLIENT_OFFSET_LIVE_MONITOR + 1)
        DashboardWindow._start_live_deployment_stream(window, "SOXL", client_offset=IB_CLIENT_OFFSET_LIVE_DEPLOYMENT + 1)

        self.assertIs(window.live_monitor_chart_stream_workers["SOXL"], shared_worker)
        self.assertIs(window.live_deployment_stream_workers["SOXL"], shared_worker)
        self.assertEqual(
            [item[2] for item in acquired],
            ["live-monitor:SOXL", "live-deployment:SOXL"],
        )

    def test_closing_live_monitor_releases_only_monitor_stream_consumer(self) -> None:
        class FakeMagellan:
            def close_session(self, *args, **kwargs) -> None:
                return None

        released = []
        window = DashboardWindow.__new__(DashboardWindow)
        window.magellan = FakeMagellan()
        window.live_monitor_chart_sessions = {"live-monitor:test:SOXL": {}}
        window.live_monitor_chart_stream_workers = {"SOXL": object()}
        window.live_monitor_historical_sync_workers = {}
        window.live_monitor_historical_sync_contexts = {}
        window._symbol_from_dataset_id = lambda value: _market_symbol_from_dataset_id(value)
        window._release_live_symbol_stream = lambda symbol, *, consumer_key, wait_ms=1500: released.append(
            (symbol, consumer_key, wait_ms)
        )

        DashboardWindow._close_live_monitor_charts(window, silent=True)

        self.assertEqual(released, [("SOXL", "live-monitor:SOXL", 1500)])

    def test_live_deployment_initial_bar_stamp_does_not_load_chart_history(self) -> None:
        window = DashboardWindow.__new__(DashboardWindow)
        window._build_charts_combined_bars = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("arming a deployment should not load chart history on the GUI thread")
        )

        stamp = DashboardWindow._last_completed_live_deployment_bar_ts_ns(window, "SOXL", "15m")

        self.assertGreater(stamp, 0)

    def _window_for_live_monitor_update(self, bars: pd.DataFrame, session_info: dict):
        class FakeMagellan:
            def __init__(self) -> None:
                self.live_updates = []
                self.replace_bars_calls = 0
                self.replace_series_calls = 0

            def send_live_update(self, *args, **kwargs) -> None:
                self.live_updates.append((args, kwargs))

            def replace_bars(self, *args, **kwargs) -> None:
                self.replace_bars_calls += 1

            def replace_series(self, *args, **kwargs) -> None:
                self.replace_series_calls += 1

        window = DashboardWindow.__new__(DashboardWindow)
        window.magellan = FakeMagellan()
        session_info.setdefault("bars", bars.copy())
        window.live_monitor_chart_sessions = {"live-monitor:test:SOXL": session_info}
        window._symbol_from_dataset_id = lambda value: _market_symbol_from_dataset_id(value)
        window._build_charts_combined_bars = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("live monitor incremental updates must not rebuild storage-backed chart bars")
        )
        window._deployment_chart_indicator_series = lambda completed, *_args: (
            {
                "SMA": pd.Series(
                    range(1, len(completed) + 1),
                    index=completed.index,
                    dtype=float,
                )
            },
            {
                "Pane": pd.Series(
                    range(10, 10 + len(completed)),
                    index=completed.index,
                    dtype=float,
                )
            },
            {"SMA": {"color": "#4da3ff"}, "Pane": {"color": "#a28bff"}},
        )
        window._live_monitor_equity_curve_for_session = lambda _info: pd.Series(
            range(100, 100 + len(_info.get("bars", bars))),
            index=_info.get("bars", bars).index,
            dtype=float,
            name="equity",
        )
        return window

    def test_live_monitor_uses_incremental_bar_update_for_partial_timeframe_bar(self) -> None:
        bars = pd.DataFrame(
            {
                "open": [1.0, 2.0],
                "high": [1.5, 2.5],
                "low": [0.5, 1.5],
                "close": [1.2, 2.2],
                "volume": [10.0, 20.0],
            },
            index=pd.to_datetime(["2026-04-22 10:45:00Z", "2026-04-22 11:00:00Z"]),
        )
        seed_ts_ns = int(pd.Timestamp(bars.index[0]).value)
        live_seed_ts_ns = int(pd.Timestamp(bars.index[1]).value)
        session_info = {
            "session_id": "live-monitor:test:SOXL",
            "symbol": "SOXL",
            "timeframe": "15m",
            "lookback": "3mo",
            "indicator_ids": [],
            "strategy_contexts": [],
            "last_sent_bar_ts_ns": live_seed_ts_ns,
            "last_sent_bar_index": 1,
            "last_completed_replacement_bar_ts_ns": seed_ts_ns,
        }
        window = self._window_for_live_monitor_update(bars, session_info)

        DashboardWindow._send_live_monitor_magellan_update(
            window,
            "SOXL",
            {
                "symbol": "SOXL",
                "ts_utc": "2026-04-22T11:01:00Z",
                "received_at": "2026-04-22T11:01:05Z",
                "open": 2.2,
                "high": 2.6,
                "low": 2.1,
                "close": 2.4,
                "volume": 5.0,
                "is_partial": True,
            },
        )

        self.assertEqual(window.magellan.replace_bars_calls, 0)
        self.assertEqual(window.magellan.replace_series_calls, 0)
        self.assertEqual(len(window.magellan.live_updates), 1)
        update = window.magellan.live_updates[0][1]
        self.assertEqual(update["bars"][0]["bar_index"], 1)
        self.assertEqual(update["bars"][0]["timestamp_utc_ns"], str(int(pd.Timestamp("2026-04-22 11:00:00Z").value)))
        self.assertEqual(update["overlay_series"][0]["points"][0]["bar_index"], 1)
        self.assertEqual(update["pane_series"][0]["points"][0]["bar_index"], 1)
        self.assertEqual(update["equity_series"][0]["points"][0]["bar_index"], 1)
        self.assertEqual(session_info["bars"].index[-1], pd.Timestamp("2026-04-22 11:00:00Z"))

    def test_live_monitor_partial_throttle_skips_indicator_work_before_cache_update(self) -> None:
        bars = pd.DataFrame(
            {
                "open": [1.0, 2.0],
                "high": [1.5, 2.5],
                "low": [0.5, 1.5],
                "close": [1.2, 2.2],
                "volume": [10.0, 20.0],
            },
            index=pd.to_datetime(["2026-04-22 10:45:00Z", "2026-04-22 11:00:00Z"]),
        )
        session_info = {
            "session_id": "live-monitor:test:SOXL",
            "symbol": "SOXL",
            "timeframe": "15m",
            "lookback": "3mo",
            "indicator_ids": [],
            "strategy_contexts": [],
            "last_sent_bar_ts_ns": int(pd.Timestamp(bars.index[-1]).value),
            "last_sent_bar_index": 1,
            "last_completed_replacement_bar_ts_ns": int(pd.Timestamp(bars.index[0]).value),
            "last_preview_update_monotonic": time.monotonic(),
        }
        window = self._window_for_live_monitor_update(bars, session_info)
        window._deployment_chart_indicator_series = lambda *_args: (_ for _ in ()).throw(
            AssertionError("preview throttle should run before indicator computation")
        )

        DashboardWindow._send_live_monitor_magellan_update(
            window,
            "SOXL",
            {
                "symbol": "SOXL",
                "ts_utc": "2026-04-22T11:01:00Z",
                "received_at": "2026-04-22T11:01:05Z",
                "open": 2.2,
                "high": 2.6,
                "low": 2.1,
                "close": 2.4,
                "volume": 5.0,
                "is_partial": True,
            },
        )

        self.assertEqual(window.magellan.live_updates, [])
        pd.testing.assert_frame_equal(session_info["bars"], bars)

    def test_live_monitor_finalizes_indicator_points_when_timeframe_bar_completes(self) -> None:
        bars = pd.DataFrame(
            {
                "open": [1.0, 2.0],
                "high": [1.5, 2.5],
                "low": [0.5, 1.5],
                "close": [1.2, 2.2],
                "volume": [10.0, 20.0],
            },
            index=pd.to_datetime(["2026-04-22 10:45:00Z", "2026-04-22 11:00:00Z"]),
        )
        live_seed_ts_ns = int(pd.Timestamp(bars.index[1]).value)
        completed_ts_ns = int(pd.Timestamp("2026-04-22 11:00:00Z").value)
        session_info = {
            "session_id": "live-monitor:test:SOXL",
            "symbol": "SOXL",
            "timeframe": "15m",
            "lookback": "3mo",
            "indicator_ids": [],
            "strategy_contexts": [],
            "last_sent_bar_ts_ns": live_seed_ts_ns,
            "last_sent_bar_index": 1,
            "last_completed_replacement_bar_ts_ns": int(pd.Timestamp(bars.index[0]).value),
        }
        window = self._window_for_live_monitor_update(bars, session_info)

        DashboardWindow._send_live_monitor_magellan_update(
            window,
            "SOXL",
            {
                "symbol": "SOXL",
                "ts_utc": "2026-04-22T11:14:00Z",
                "received_at": "2026-04-22T11:15:05Z",
                "open": 2.2,
                "high": 2.8,
                "low": 2.0,
                "close": 2.6,
                "volume": 15.0,
                "is_partial": False,
            },
        )

        self.assertEqual(window.magellan.replace_bars_calls, 0)
        self.assertEqual(window.magellan.replace_series_calls, 0)
        update = window.magellan.live_updates[0][1]
        self.assertEqual(update["bars"][0]["bar_index"], 1)
        self.assertEqual(update["overlay_series"][0]["points"][0]["bar_index"], 1)
        self.assertEqual(update["pane_series"][0]["points"][0]["bar_index"], 1)
        self.assertEqual(update["equity_series"][0]["points"][0]["bar_index"], 1)
        self.assertEqual(session_info["last_completed_replacement_bar_ts_ns"], completed_ts_ns)

    def test_live_monitor_updates_completed_and_preview_indicator_points_together(self) -> None:
        bars = pd.DataFrame(
            {
                "open": [1.0, 2.0],
                "high": [1.5, 2.5],
                "low": [0.5, 1.5],
                "close": [1.2, 2.2],
                "volume": [10.0, 20.0],
            },
            index=pd.to_datetime(["2026-04-22 10:45:00Z", "2026-04-22 11:00:00Z"]),
        )
        session_info = {
            "session_id": "live-monitor:test:SOXL",
            "symbol": "SOXL",
            "timeframe": "15m",
            "lookback": "3mo",
            "indicator_ids": [],
            "strategy_contexts": [],
            "bar_index_by_ts_ns": DashboardWindow._live_monitor_bar_index_map_from_bars(bars),
            "next_live_bar_index": 2,
            "last_sent_bar_ts_ns": int(pd.Timestamp(bars.index[1]).value),
            "last_sent_bar_index": 1,
            "last_completed_replacement_bar_ts_ns": int(pd.Timestamp(bars.index[0]).value),
        }
        window = self._window_for_live_monitor_update(bars, session_info)

        DashboardWindow._send_live_monitor_magellan_update(
            window,
            "SOXL",
            {
                "symbol": "SOXL",
                "ts_utc": "2026-04-22T11:15:00Z",
                "received_at": "2026-04-22T11:16:05Z",
                "open": 2.3,
                "high": 2.9,
                "low": 2.2,
                "close": 2.7,
                "volume": 8.0,
                "is_partial": False,
            },
        )

        update = window.magellan.live_updates[0][1]
        self.assertEqual(update["bars"][0]["bar_index"], 2)
        overlay_points = [
            point
            for series in update["overlay_series"]
            for point in series["points"]
        ]
        self.assertEqual([point["bar_index"] for point in overlay_points], [1, 2])
        self.assertEqual(session_info["last_completed_replacement_bar_ts_ns"], int(pd.Timestamp(bars.index[1]).value))
        self.assertEqual(session_info["bar_index_by_ts_ns"][str(int(pd.Timestamp("2026-04-22 11:15:00Z").value))], 2)

    def test_live_deployment_signal_adds_trade_marker_to_open_monitor(self) -> None:
        bars = pd.DataFrame(
            {
                "open": [500.0, 505.0],
                "high": [501.0, 506.0],
                "low": [499.0, 504.0],
                "close": [500.5, 505.5],
                "volume": [10.0, 20.0],
            },
            index=pd.to_datetime(["2026-04-30 12:00:00Z", "2026-04-30 12:15:00Z"]),
        )
        session_info = {
            "session_id": "live-monitor:deploy-1:SMH",
            "deployment_id": "deploy-1",
            "symbol": "SMH",
            "timeframe": "15m",
            "bars": bars,
            "last_sent_bar_ts_ns": int(pd.Timestamp(bars.index[-1]).value),
            "last_sent_bar_index": 1,
            "last_completed_replacement_bar_ts_ns": int(pd.Timestamp(bars.index[-1]).value),
        }
        window = self._window_for_live_monitor_update(bars, session_info)

        DashboardWindow._on_live_deployment_signal_marker(
            window,
            {
                "symbol": "SMH",
                "deployment_id": "deploy-1",
                "action": "ENTRY",
                "side": "LONG",
                "price": 505.5,
                "bar_timestamp_utc": "2026-04-30T12:15:00Z",
                "position_size_override": {"target_notional": 400.0},
            },
        )

        self.assertEqual(len(window.magellan.live_updates), 1)
        marker = window.magellan.live_updates[0][1]["trade_markers"][0]
        self.assertEqual(marker["bar_index"], 1)
        self.assertEqual(marker["side"], "buy")
        self.assertEqual(marker["event"], "entry")
        self.assertEqual(session_info["local_trade_markers"][0]["bar_index"], 1)

    def test_live_monitor_preview_without_seed_cache_does_not_create_one_bar_chart(self) -> None:
        bars = pd.DataFrame(
            {
                "open": [1.0, 2.0],
                "high": [1.5, 2.5],
                "low": [0.5, 1.5],
                "close": [1.2, 2.2],
                "volume": [10.0, 20.0],
            },
            index=pd.to_datetime(["2026-04-22 10:45:00Z", "2026-04-22 11:00:00Z"]),
        )
        session_info = {
            "session_id": "live-monitor:test:SOXL",
            "symbol": "SOXL",
            "timeframe": "15m",
            "lookback": "3mo",
            "indicator_ids": [],
            "strategy_contexts": [],
            "bars": pd.DataFrame(),
            "last_sent_bar_ts_ns": 0,
            "last_sent_bar_index": -1,
            "last_completed_replacement_bar_ts_ns": 0,
        }
        window = self._window_for_live_monitor_update(bars, session_info)

        DashboardWindow._send_live_monitor_magellan_update(
            window,
            "SOXL",
            {
                "symbol": "SOXL",
                "ts_utc": "2026-04-22T11:01:00Z",
                "received_at": "2026-04-22T11:01:05Z",
                "open": 2.2,
                "high": 2.6,
                "low": 2.1,
                "close": 2.4,
                "volume": 5.0,
                "is_partial": True,
            },
        )

        self.assertEqual(window.magellan.live_updates, [])

    def test_live_monitor_historical_reload_uses_snapshot_seed_with_strategy_series(self) -> None:
        bars = pd.DataFrame(
            {
                "open": [1.0, 2.0, 3.0],
                "high": [1.5, 2.5, 3.5],
                "low": [0.5, 1.5, 2.5],
                "close": [1.2, 2.2, 3.2],
                "volume": [10.0, 20.0, 30.0],
            },
            index=pd.to_datetime(
                [
                    "2026-04-22 10:45:00Z",
                    "2026-04-22 10:50:00Z",
                    "2026-04-22 10:55:00Z",
                ]
            ),
        )

        class FakeMagellan:
            def __init__(self) -> None:
                self.reload_calls = []
                self.open_calls = []
                self.close_calls = []

            def reload_live_seed(self, *args, **kwargs) -> None:
                self.reload_calls.append((args, kwargs))

            def open_live_session(self, *args, **kwargs) -> None:
                self.open_calls.append((args, kwargs))

            def close_session(self, *args, **kwargs) -> None:
                self.close_calls.append((args, kwargs))

        class FakeExporter:
            def __init__(self, snapshot_root: str) -> None:
                self.snapshot_root = snapshot_root
                self.calls = []

            def export_market_snapshot(self, **kwargs):
                self.calls.append(kwargs)
                return type("Artifact", (), {"snapshot_root": self.snapshot_root})()

        window = DashboardWindow.__new__(DashboardWindow)
        window.magellan = FakeMagellan()
        window._symbol_from_dataset_id = lambda value: _market_symbol_from_dataset_id(value)
        window._build_charts_combined_bars = lambda *args, **kwargs: (bars, "SOXL", "seeded")
        window._deployment_strategy_contexts_for_symbol = lambda *_args: [("Strategy", "SMACrossStrategy", {"fast": 2, "slow": 3})]
        window._deployment_chart_indicator_series = lambda completed, *_args: (
            {
                "Strategy SMA Fast": pd.Series(range(len(completed)), index=completed.index, dtype=float),
            },
            {},
            {"Strategy SMA Fast": {"color": "#4da3ff"}},
        )
        window._live_monitor_trade_marker_frame = lambda *_args: pd.DataFrame()
        window.live_monitor_chart_sessions = {"live-monitor:deploy:SOXL": {"session_id": "live-monitor:deploy:SOXL"}}

        with tempfile.TemporaryDirectory() as tmpdir:
            window.live_chart_snapshot_exporter = FakeExporter(tmpdir)
            ok, _message = DashboardWindow._open_or_reload_live_monitor_chart(
                window,
                selected={"deployment_id": "deploy", "target_id": ""},
                snapshot={},
                deployment_id="deploy",
                symbol="SOXL",
                timeframe="5m",
                lookback="5d",
                indicator_ids=[],
                equity_curve=pd.Series(dtype=float),
                status_text="reload",
                reload_existing=True,
                client_offset=0,
            )

        self.assertTrue(ok)
        self.assertIn("Strategy SMA Fast", window.live_chart_snapshot_exporter.calls[0]["overlays"])
        self.assertEqual(len(window.magellan.reload_calls), 1)
        self.assertEqual(window.magellan.open_calls, [])
        self.assertIn("snapshot_path", window.magellan.reload_calls[0][1])


class LiveMonitorMetricAggregationTests(unittest.TestCase):
    def test_external_snapshot_aggregates_trade_metrics_when_pnl_is_available(self) -> None:
        window = DashboardWindow.__new__(DashboardWindow)
        window.deployment_child_links_frame = pd.DataFrame()
        window._deployment_symbol_scope = lambda _row: {"SOXL"}
        snapshot = {
            "snapshot_ts": "2026-04-23T13:45:00Z",
            "orders": [
                {
                    "id": 1,
                    "symbol": "SOXL",
                    "status": "filled",
                    "raw_payload": {"deployment_id": "deployment-1"},
                }
            ],
            "fills": [
                {"order_id": 1, "realized_pnl": 10.0},
                {"order_id": 1, "realized_pnl": -4.0},
            ],
            "recent_trades": [],
            "positions": [{"symbol": "SOXL", "unrealized_pnl": 2.5}],
            "account": {"equity": 4000.0},
            "equity_curve": [],
        }

        aggregate = DashboardWindow._aggregate_external_snapshot_for_deployment(
            window,
            {"deployment_id": "deployment-1"},
            snapshot,
        )

        self.assertEqual(aggregate["realized_pnl"], 6.0)
        self.assertEqual(aggregate["open_pnl"], 2.5)
        self.assertEqual(aggregate["trade_count"], 2)
        self.assertEqual(aggregate["win_count"], 1)
        self.assertEqual(aggregate["loss_count"], 1)
        self.assertEqual(aggregate["win_rate"], 0.5)
        self.assertAlmostEqual(float(aggregate["profit_factor"]), 2.5)
        self.assertIsNotNone(aggregate["sharpe"])


if __name__ == "__main__":
    unittest.main()
