from __future__ import annotations

import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from backtest_engine.catalog import ResultCatalog
import pandas as pd

from backtest_engine.live_market import InteractiveBrokersRealtimeConfig, LiveMarketBar, LiveMarketDataStore
from backtest_engine.live_deployment_runner import (
    DeploymentContext,
    DeploymentRunnerConfig,
    LiveDeploymentRunnerService,
    RunnerSymbolStream,
    SMACrossStrategy,
    STREAM_STALE_RECONNECT_SECONDS,
    _decode_json_dict,
    main as live_runner_main,
)
from backtest_engine import live_monitor_chart_service
from backtest_engine.live_monitor_chart_service import LiveChartServiceConfig, LiveMonitorChartService, main as live_chart_main


class LiveDeploymentCatalogTest(unittest.TestCase):
    def test_persists_targets_manual_definitions_and_deployments(self) -> None:
        with TemporaryDirectory() as tmpdir:
            catalog = ResultCatalog(Path(tmpdir) / "backtests.sqlite")
            catalog.save_deployment_target(
                target_id="algo_engine_live",
                name="Algo Engine Live",
                mode="live",
                broker_scope="public",
                transport_mode="co_located",
                base_url="http://127.0.0.1",
                webhook_path="/live_webhook",
                status_path="/live_status",
                dashboard_path="/live",
                logs_path="/logs_data?scope=live",
                project_root="/home/ethan/algo_trading_engine",
                db_path="/home/ethan/algo_trading_engine/live.db",
                log_db_path="/home/ethan/algo_trading_engine/engine_logs.db",
                secret_ref="LIVE_WEBHOOK_SECRET",
                secret_value="saved-secret",
            )
            targets = catalog.load_deployment_targets()
            self.assertEqual(len(targets), 1)
            self.assertEqual(targets[0].target_id, "algo_engine_live")
            self.assertEqual(targets[0].mode, "live")
            self.assertEqual(targets[0].secret_value, "saved-secret")

            manual_id = catalog.save_manual_deployment_definition(
                deployment_kind="portfolio_strategy_blocks",
                strategy="Portfolio Strategy Blocks",
                dataset_scope_json=["SPY", "QQQ"],
                timeframe="5 minutes",
                params_json={"source_kind": "portfolio_fixed_blocks"},
                structure_json={
                    "portfolio_dataset_ids": ["SPY", "QQQ"],
                    "strategy_blocks": [
                        {
                            "block_id": "trend",
                            "strategy_name": "SMACrossStrategy",
                            "strategy_params": {"fast": 10, "slow": 40},
                            "asset_dataset_ids": ["SPY", "QQQ"],
                        }
                    ],
                },
                target_id="algo_engine_live",
                mode="live",
                sizing_json={"qty_type": "cash", "qty_value": 1000.0},
                notes="Manual portfolio draft.",
            )
            manual_rows = catalog.load_manual_deployment_definitions()
            self.assertEqual(len(manual_rows), 1)
            self.assertEqual(manual_rows[0].manual_definition_id, manual_id)
            self.assertEqual(manual_rows[0].deployment_kind, "portfolio_strategy_blocks")

            parent_id = catalog.save_deployment(
                deployment_kind="portfolio_strategy_blocks",
                source_type="manual",
                source_id=manual_id,
                strategy="Portfolio Strategy Blocks",
                timeframe="5 minutes",
                params_json={"source_kind": "portfolio_fixed_blocks"},
                structure_json={"portfolio_dataset_ids": ["SPY", "QQQ"]},
                validation_refs_json={"manual_definition_id": manual_id},
                target_id="algo_engine_live",
                mode="live",
                sizing_json={"qty_type": "cash", "qty_value": 1000.0},
                status="draft",
            )
            child_id = catalog.save_deployment(
                parent_deployment_id=parent_id,
                deployment_kind="single_strategy",
                source_type="portfolio_child",
                source_id=parent_id,
                strategy="SMACrossStrategy",
                dataset_id="SPY",
                symbol="SPY",
                timeframe="5 minutes",
                params_json={"fast": 10, "slow": 40},
                structure_json={},
                validation_refs_json={"manual_definition_id": manual_id},
                target_id="algo_engine_live",
                mode="live",
                sizing_json={"qty_type": "cash", "qty_value": 1000.0},
                status="draft",
            )
            catalog.save_deployment_child_link(
                parent_deployment_id=parent_id,
                child_deployment_id=child_id,
                child_role="strategy_block_asset",
                dataset_id="SPY",
                symbol="SPY",
                strategy_block_id="trend",
            )
            catalog.update_deployment_status(parent_id, status="armed", armed_at="2026-04-12T12:00:00+00:00")
            catalog.save_deployment_metric_snapshot(
                deployment_id=parent_id,
                snapshot_ts="2026-04-12T12:01:00+00:00",
                realized_pnl=125.5,
                open_pnl=12.25,
                trade_count=4,
                win_count=3,
                loss_count=1,
                win_rate=0.75,
                profit_factor=2.4,
                sharpe=1.1,
                current_position_json={"legs": 1},
                health_json={"status": "ok"},
            )

            deployments = catalog.load_deployments()
            self.assertEqual(len(deployments), 2)
            parent = next(row for row in deployments if row.deployment_id == parent_id)
            self.assertEqual(parent.deployment_kind, "portfolio_strategy_blocks")
            self.assertEqual(parent.status, "armed")
            links = catalog.load_deployment_child_links(parent_id)
            self.assertEqual(len(links), 1)
            self.assertEqual(links[0].child_deployment_id, child_id)
            snapshots = catalog.load_latest_deployment_metric_snapshots()
            self.assertEqual(len(snapshots), 1)
            self.assertEqual(snapshots[0].deployment_id, parent_id)
            self.assertAlmostEqual(float(snapshots[0].realized_pnl or 0.0), 125.5, places=6)

            deleted = catalog.delete_deployment(parent_id)
            self.assertEqual(deleted, 2)
            self.assertEqual(catalog.load_deployments(), [])
            self.assertEqual(catalog.load_deployment_child_links(parent_id), [])
            self.assertEqual(catalog.load_latest_deployment_metric_snapshots(), [])

    def test_runner_command_and_event_journal_supports_producer_consumer_flow(self) -> None:
        with TemporaryDirectory() as tmpdir:
            catalog = ResultCatalog(Path(tmpdir) / "backtests.sqlite")
            command_id = catalog.enqueue_deployment_runner_command(
                "arm",
                deployment_id="deploy-1",
                payload_json={"deployment_id": "deploy-1"},
            )

            claimed = catalog.claim_deployment_runner_commands(runner_id="runner-1", limit=10)
            self.assertEqual([row.command_id for row in claimed], [command_id])
            self.assertEqual(claimed[0].status, "running")

            catalog.save_deployment_runner_event(
                event_type="deployment_armed",
                deployment_id="deploy-1",
                message="Deployment armed.",
                payload_json={"contexts": ["ctx-1"]},
                event_id="event-1",
            )
            catalog.finish_deployment_runner_command(command_id)

            done = catalog.load_deployment_runner_commands(status="done")
            self.assertEqual([row.command_id for row in done], [command_id])
            events = catalog.load_deployment_runner_events(after_seq=0)
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].event_type, "deployment_armed")
            self.assertEqual(events[0].deployment_id, "deploy-1")

    def test_live_chart_command_and_event_journal_supports_producer_consumer_flow(self) -> None:
        with TemporaryDirectory() as tmpdir:
            catalog = ResultCatalog(Path(tmpdir) / "backtests.sqlite")
            command_id = catalog.enqueue_live_chart_command(
                "open_chart",
                session_id="session-1",
                deployment_id="deploy-1",
                payload_json={"deployment_id": "deploy-1", "symbol": "SPY"},
            )

            claimed = catalog.claim_live_chart_commands(service_id="chart-service-1", limit=10)
            self.assertEqual([row.command_id for row in claimed], [command_id])
            self.assertEqual(claimed[0].status, "running")

            catalog.save_live_chart_event(
                event_type="chart_opened",
                session_id="session-1",
                deployment_id="deploy-1",
                symbol="SPY",
                message="Chart opened.",
                payload_json={"snapshot_path": "/tmp/chart"},
                event_id="chart-event-1",
            )
            catalog.finish_live_chart_command(command_id)

            done = catalog.load_live_chart_commands(status="done")
            self.assertEqual([row.command_id for row in done], [command_id])
            events = catalog.load_live_chart_events(after_seq=0)
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].event_type, "chart_opened")
            self.assertEqual(events[0].session_id, "session-1")

    def test_runner_once_processes_arm_command_without_gui_or_streams(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            catalog_path = root / "backtests.sqlite"
            live_store_path = root / "live_market.sqlite"
            catalog = ResultCatalog(catalog_path)
            catalog.save_deployment_target(
                target_id="target-1",
                name="Test Target",
                mode="paper",
                broker_scope="paper",
                transport_mode="remote_http",
                base_url="http://127.0.0.1",
                webhook_path="/webhook",
                secret_value="secret",
            )
            deployment_id = catalog.save_deployment(
                deployment_kind="single_strategy",
                source_type="manual",
                source_id="manual-1",
                strategy="SMACrossStrategy",
                symbol="SPY",
                timeframe="1m",
                params_json={"fast": 2, "slow": 3},
                structure_json={},
                validation_refs_json={},
                target_id="target-1",
                mode="paper",
                sizing_json={"qty_type": "cash", "qty_value": 100.0},
                status="draft",
            )
            command_id = catalog.enqueue_deployment_runner_command("arm", deployment_id=deployment_id)

            exit_code = live_runner_main(
                [
                    "--catalog",
                    str(catalog_path),
                    "--live-store",
                    str(live_store_path),
                    "--once",
                    "--no-streams",
                ]
            )

            self.assertEqual(exit_code, 0)
            commands = catalog.load_deployment_runner_commands()
            command = next(row for row in commands if row.command_id == command_id)
            self.assertEqual(command.status, "done")
            deployment = next(row for row in catalog.load_deployments() if row.deployment_id == deployment_id)
            self.assertEqual(deployment.status, "live")
            event_types = {row.event_type for row in catalog.load_deployment_runner_events(after_seq=0)}
            self.assertIn("runner_started", event_types)
            self.assertIn("deployment_armed", event_types)

    def test_runner_once_resumes_live_deployment_without_new_gui_command(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            catalog_path = root / "backtests.sqlite"
            live_store_path = root / "live_market.sqlite"
            catalog = ResultCatalog(catalog_path)
            catalog.save_deployment_target(
                target_id="target-1",
                name="Test Target",
                mode="paper",
                broker_scope="paper",
                transport_mode="remote_http",
                base_url="http://127.0.0.1",
                webhook_path="/webhook",
                secret_value="secret",
            )
            deployment_id = catalog.save_deployment(
                deployment_kind="single_strategy",
                source_type="manual",
                source_id="manual-1",
                strategy="SMACrossStrategy",
                symbol="SPY",
                timeframe="1m",
                params_json={"fast": 2, "slow": 3},
                structure_json={},
                validation_refs_json={},
                target_id="target-1",
                mode="paper",
                sizing_json={"qty_type": "cash", "qty_value": 100.0},
                status="live",
                status_reason="Last signal ENTRY LONG sent at 2026-05-08T13:15:00+00:00.",
            )

            exit_code = live_runner_main(
                [
                    "--catalog",
                    str(catalog_path),
                    "--live-store",
                    str(live_store_path),
                    "--once",
                    "--no-streams",
                ]
            )

            self.assertEqual(exit_code, 0)
            deployment = next(row for row in catalog.load_deployments() if row.deployment_id == deployment_id)
            self.assertEqual(deployment.status, "live")
            self.assertEqual(deployment.status_reason, "Last signal ENTRY LONG sent at 2026-05-08T13:15:00+00:00.")
            event_types = {row.event_type for row in catalog.load_deployment_runner_events(after_seq=0)}
            self.assertIn("runner_started", event_types)
            self.assertIn("deployment_resumed", event_types)

    def test_live_chart_once_processes_open_command_without_gui_or_magellan(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            catalog_path = root / "backtests.sqlite"
            live_store_path = root / "live_market.sqlite"
            catalog = ResultCatalog(catalog_path)
            deployment_id = catalog.save_deployment(
                deployment_kind="single_strategy",
                source_type="manual",
                source_id="manual-1",
                strategy="SMACrossStrategy",
                symbol="SPY",
                timeframe="1m",
                params_json={"fast": 2, "slow": 3},
                structure_json={},
                validation_refs_json={},
                target_id="",
                mode="paper",
                sizing_json={"qty_type": "cash", "qty_value": 100.0},
                status="live",
            )
            store = LiveMarketDataStore(live_store_path)
            first_ts = pd.Timestamp.now(tz="UTC").floor("min") - pd.Timedelta(minutes=6)
            for idx in range(5):
                price = 100.0 + idx
                store.upsert_bar(
                    LiveMarketBar(
                        symbol="SPY",
                        timestamp=first_ts + pd.Timedelta(minutes=idx),
                        open=price,
                        high=price + 0.5,
                        low=price - 0.5,
                        close=price,
                        volume=1000.0 + idx,
                    )
                )
            command_id = catalog.enqueue_live_chart_command(
                "open_chart",
                session_id=f"live-monitor:{deployment_id}:SPY",
                deployment_id=deployment_id,
                payload_json={
                    "session_id": f"live-monitor:{deployment_id}:SPY",
                    "deployment_id": deployment_id,
                    "symbol": "SPY",
                    "timeframe": "1m",
                    "lookback": "5d",
                    "indicator_ids": [],
                },
            )

            exit_code = live_chart_main(
                [
                    "--catalog",
                    str(catalog_path),
                    "--live-store",
                    str(live_store_path),
                    "--once",
                    "--no-magellan",
                ]
            )

            self.assertEqual(exit_code, 0)
            commands = catalog.load_live_chart_commands()
            command = next(row for row in commands if row.command_id == command_id)
            self.assertEqual(command.status, "done")
            event_types = {row.event_type for row in catalog.load_live_chart_events(after_seq=0)}
            self.assertIn("service_started", event_types)
            self.assertIn("chart_opening", event_types)
            self.assertIn("chart_opened", event_types)

    def test_live_chart_once_processes_warmup_command(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            catalog_path = root / "backtests.sqlite"
            live_store_path = root / "live_market.sqlite"
            catalog = ResultCatalog(catalog_path)
            command_id = catalog.enqueue_live_chart_command("warmup")

            exit_code = live_chart_main(
                [
                    "--catalog",
                    str(catalog_path),
                    "--live-store",
                    str(live_store_path),
                    "--once",
                    "--no-magellan",
                ]
            )

            self.assertEqual(exit_code, 0)
            command = next(row for row in catalog.load_live_chart_commands() if row.command_id == command_id)
            self.assertEqual(command.status, "done")
            event_types = {row.event_type for row in catalog.load_live_chart_events(after_seq=0)}
            self.assertIn("service_warmed", event_types)

    def test_live_chart_service_resolves_deployment_symbols_off_gui(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            catalog_path = root / "backtests.sqlite"
            live_store_path = root / "live_market.sqlite"
            catalog = ResultCatalog(catalog_path)
            deployment_id = catalog.save_deployment(
                deployment_kind="portfolio_strategy_blocks",
                source_type="manual",
                source_id="manual-portfolio",
                strategy="Portfolio Strategy Blocks",
                symbol="",
                timeframe="1m",
                params_json={},
                structure_json={
                    "portfolio_dataset_ids": ["SPY", "QQQ"],
                    "strategy_blocks": [],
                },
                validation_refs_json={},
                target_id="",
                mode="paper",
                sizing_json={"qty_type": "cash", "qty_value": 100.0},
                status="live",
            )
            store = LiveMarketDataStore(live_store_path)
            first_ts = pd.Timestamp.now(tz="UTC").floor("min") - pd.Timedelta(minutes=6)
            for symbol in ("SPY", "QQQ"):
                for idx in range(5):
                    price = 100.0 + idx
                    store.upsert_bar(
                        LiveMarketBar(
                            symbol=symbol,
                            timestamp=first_ts + pd.Timedelta(minutes=idx),
                            open=price,
                            high=price + 0.5,
                            low=price - 0.5,
                            close=price,
                            volume=1000.0 + idx,
                        )
                    )
            command_id = catalog.enqueue_live_chart_command(
                "open_deployment_charts",
                deployment_id=deployment_id,
                payload_json={
                    "deployment_id": deployment_id,
                    "timeframe": "1m",
                    "lookback": "5d",
                    "indicator_ids": [],
                    "replace_existing": True,
                },
            )

            exit_code = live_chart_main(
                [
                    "--catalog",
                    str(catalog_path),
                    "--live-store",
                    str(live_store_path),
                    "--once",
                    "--no-magellan",
                ]
            )

            self.assertEqual(exit_code, 0)
            command = next(row for row in catalog.load_live_chart_commands() if row.command_id == command_id)
            self.assertEqual(command.status, "done")
            events = catalog.load_live_chart_events(after_seq=0)
            opened_symbols = {row.symbol for row in events if row.event_type == "chart_opened"}
            self.assertEqual(opened_symbols, {"SPY", "QQQ"})

    def test_live_chart_service_rejects_non_live_deployment_open(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            catalog_path = root / "backtests.sqlite"
            live_store_path = root / "live_market.sqlite"
            catalog = ResultCatalog(catalog_path)
            deployment_id = catalog.save_deployment(
                deployment_kind="single_strategy",
                source_type="manual",
                source_id="manual-1",
                strategy="SMACrossStrategy",
                symbol="SPY",
                timeframe="1m",
                params_json={"fast": 2, "slow": 3},
                structure_json={},
                validation_refs_json={},
                target_id="",
                mode="paper",
                sizing_json={"qty_type": "cash", "qty_value": 100.0},
                status="stopped",
            )
            command_id = catalog.enqueue_live_chart_command(
                "open_deployment_charts",
                deployment_id=deployment_id,
                payload_json={
                    "deployment_id": deployment_id,
                    "timeframe": "1m",
                    "lookback": "5d",
                    "indicator_ids": [],
                    "replace_existing": True,
                },
            )

            exit_code = live_chart_main(
                [
                    "--catalog",
                    str(catalog_path),
                    "--live-store",
                    str(live_store_path),
                    "--once",
                    "--no-magellan",
                ]
            )

            self.assertEqual(exit_code, 0)
            command = next(row for row in catalog.load_live_chart_commands() if row.command_id == command_id)
            self.assertEqual(command.status, "error")
            self.assertIn("stopped", command.error)
            events = catalog.load_live_chart_events(after_seq=0)
            event_types = {row.event_type for row in events}
            self.assertIn("deployment_chart_open_rejected", event_types)
            self.assertIn("chart_command_error", event_types)
            self.assertNotIn("chart_opened", event_types)

    def test_live_chart_signal_exit_marker_uses_exit_color_semantics(self) -> None:
        bars = pd.DataFrame(
            {
                "open": [100.0, 101.0],
                "high": [101.0, 102.0],
                "low": [99.0, 100.0],
                "close": [100.5, 101.5],
                "volume": [1000.0, 1200.0],
            },
            index=pd.to_datetime(["2026-05-07T11:30:00Z", "2026-05-07T11:45:00Z"]),
        )

        marker = LiveMonitorChartService.signal_trade_marker(
            {
                "action": "EXIT",
                "side": "LONG",
                "price": 101.5,
                "bar_timestamp_utc": "2026-05-07T11:45:00Z",
                "position_size_override": {"target_qty": 12.0},
            },
            bars,
        )

        self.assertIsNotNone(marker)
        self.assertEqual(marker["side"], "sell")
        self.assertEqual(marker["event"], "exit")
        self.assertEqual(marker["color"], "#ef4444")
        self.assertLess(marker["quantity"], 0.0)

    def test_live_chart_signal_entry_marker_uses_entry_color_semantics(self) -> None:
        bars = pd.DataFrame(
            {
                "open": [100.0, 101.0],
                "high": [101.0, 102.0],
                "low": [99.0, 100.0],
                "close": [100.5, 101.5],
                "volume": [1000.0, 1200.0],
            },
            index=pd.to_datetime(["2026-05-07T11:30:00Z", "2026-05-07T11:45:00Z"]),
        )

        marker = LiveMonitorChartService.signal_trade_marker(
            {
                "action": "ENTRY",
                "side": "SHORT",
                "price": 101.5,
                "bar_timestamp_utc": "2026-05-07T11:45:00Z",
                "position_size_override": {"target_qty": 12.0},
            },
            bars,
        )

        self.assertIsNotNone(marker)
        self.assertEqual(marker["side"], "buy")
        self.assertEqual(marker["event"], "entry")
        self.assertEqual(marker["color"], "#22c55e")
        self.assertGreater(marker["quantity"], 0.0)

    def test_live_chart_service_syncs_ibkr_gap_into_live_store_before_open(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            service = LiveMonitorChartService(
                config=LiveChartServiceConfig(
                    catalog_path=root / "backtests.sqlite",
                    live_store_path=root / "live_market.sqlite",
                    run_once=True,
                    magellan_enabled=False,
                )
            )
            start = pd.Timestamp("2026-05-07T23:46:00Z")
            end = pd.Timestamp("2026-05-08T11:42:00Z")
            frame = pd.DataFrame(
                {
                    "open": [154.0, 155.0],
                    "high": [154.5, 155.5],
                    "low": [153.5, 154.5],
                    "close": [154.25, 155.25],
                    "volume": [1000.0, 1200.0],
                },
                index=pd.to_datetime(["2026-05-08T11:30:00Z", "2026-05-08T11:31:00Z"]),
            )

            with (
                patch.object(service, "live_chart_sync_window", return_value=(start, end)),
                patch("scripts.fetch_interactive_brokers.fetch_bars", return_value=frame) as fetch_mock,
            ):
                rows = service.sync_live_store_gap_if_needed(
                    {},
                    "SOXL",
                    "3mo",
                    session_id="live-monitor:deploy-1:SOXL",
                    deployment_id="deploy-1",
                )

            self.assertEqual(rows, 2)
            self.assertGreaterEqual(fetch_mock.call_count, 1)
            stored = service.store.load_recent_bars("SOXL")
            self.assertEqual(len(stored), 2)
            self.assertEqual(pd.Timestamp(stored.index[-1]), pd.Timestamp("2026-05-08T11:31:00Z"))
            event_types = {row.event_type for row in service.catalog.load_live_chart_events(after_seq=0)}
            self.assertIn("chart_sync_started", event_types)
            self.assertIn("chart_sync_finished", event_types)

    def test_live_chart_sync_window_detects_middle_gap_even_when_latest_bar_is_current(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            service = LiveMonitorChartService(
                config=LiveChartServiceConfig(
                    catalog_path=root / "backtests.sqlite",
                    live_store_path=root / "live_market.sqlite",
                    run_once=True,
                    magellan_enabled=False,
                )
            )
            for ts, price in (
                ("2026-05-07T23:59:00Z", 152.43),
                ("2026-05-08T12:10:00Z", 160.31),
            ):
                service.store.upsert_bar(
                    LiveMarketBar(
                        symbol="SOXL",
                        timestamp=pd.Timestamp(ts),
                        open=price,
                        high=price,
                        low=price,
                        close=price,
                        volume=100.0,
                    )
                )

            window = service.live_chart_sync_window(
                {},
                "SOXL",
                "5d",
                now=pd.Timestamp("2026-05-08T12:11:00Z"),
            )

            self.assertIsNotNone(window)
            start, end = window
            self.assertLessEqual(start, pd.Timestamp("2026-05-08T08:00:00Z"))
            self.assertEqual(end, pd.Timestamp("2026-05-08T12:10:00Z"))

    def test_live_chart_sync_window_tolerates_short_gap_inside_chart_timeframe(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            service = LiveMonitorChartService(
                config=LiveChartServiceConfig(
                    catalog_path=root / "backtests.sqlite",
                    live_store_path=root / "live_market.sqlite",
                    run_once=True,
                    magellan_enabled=False,
                )
            )
            index = pd.date_range("2026-05-08T08:00:00Z", "2026-05-08T13:40:00Z", freq="1min")
            gap_start = pd.Timestamp("2026-05-08T10:00:00Z")
            gap_end = pd.Timestamp("2026-05-08T10:20:00Z")
            index = index[(index <= gap_start) | (index >= gap_end)]
            service.store.upsert_bars(
                [
                    LiveMarketBar(
                        symbol="SOXL",
                        timestamp=ts,
                        open=100.0,
                        high=100.0,
                        low=100.0,
                        close=100.0,
                        volume=100.0,
                    )
                    for ts in index
                ]
            )
            historical = pd.DataFrame(
                {
                    "open": [99.0],
                    "high": [99.0],
                    "low": [99.0],
                    "close": [99.0],
                    "volume": [100.0],
                },
                index=pd.to_datetime(["2026-05-08T07:59:00Z"]),
            )

            with (
                patch.object(service, "historical_dataset_id_for_symbol", return_value="SOXL_interactive_brokers_10y_1m"),
                patch("backtest_engine.live_monitor_chart_service.DuckDBStore.load_range", return_value=historical),
            ):
                window = service.live_chart_sync_window(
                    {"timeframe": "15 minutes"},
                    "SOXL",
                    "5d",
                    now=pd.Timestamp("2026-05-08T13:41:00Z"),
                )

            self.assertIsNone(window)

    def test_live_chart_strategy_overlays_extend_completed_value_to_live_bar(self) -> None:
        with TemporaryDirectory() as tmpdir:
            service = LiveMonitorChartService(
                config=LiveChartServiceConfig(
                    catalog_path=Path(tmpdir) / "backtests.sqlite",
                    live_store_path=Path(tmpdir) / "live_market.sqlite",
                    run_once=True,
                    magellan_enabled=False,
                )
            )
            bars = pd.DataFrame(
                {
                    "open": [100.0, 101.0, 102.0, 103.0, 120.0],
                    "high": [101.0, 102.0, 103.0, 104.0, 121.0],
                    "low": [99.0, 100.0, 101.0, 102.0, 119.0],
                    "close": [100.0, 101.0, 102.0, 103.0, 120.0],
                    "volume": [1.0, 1.0, 1.0, 1.0, 1.0],
                },
                index=pd.to_datetime(
                    [
                        "2026-05-08T12:15:00Z",
                        "2026-05-08T12:30:00Z",
                        "2026-05-08T12:45:00Z",
                        "2026-05-08T13:00:00Z",
                        "2026-05-08T13:15:00Z",
                    ]
                ),
            )

            with patch.object(
                live_monitor_chart_service,
                "_completed_bars_for_timeframe",
                wraps=lambda frame, timeframe, now=None: live_monitor_chart_service._normalize_bars(frame).iloc[:-1],
            ):
                overlays, _panes, _styles = service.indicator_series(
                    bars,
                    [],
                    [(None, "SMACrossStrategy", {"fast": 2, "slow": 3})],
                    timeframe="15m",
                )

            self.assertIn("SMA Fast", overlays)
            live_ts = pd.Timestamp("2026-05-08T13:15:00Z")
            completed_ts = pd.Timestamp("2026-05-08T13:00:00Z")
            self.assertIn(live_ts, overlays["SMA Fast"].index)
            self.assertEqual(overlays["SMA Fast"].loc[live_ts], overlays["SMA Fast"].loc[completed_ts])
            self.assertEqual(overlays["SMA Fast"].loc[live_ts], 102.5)
            self.assertEqual(overlays["SMA Slow"].loc[live_ts], 102.0)

    def test_live_chart_poll_batches_missing_bars_with_session_stable_indexes(self) -> None:
        class FakeMagellan:
            def __init__(self) -> None:
                self.live_updates = []

            def send_live_update(self, *args, **kwargs) -> None:
                self.live_updates.append((args, kwargs))

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            service = LiveMonitorChartService(
                config=LiveChartServiceConfig(
                    catalog_path=root / "backtests.sqlite",
                    live_store_path=root / "live_market.sqlite",
                    run_once=True,
                    magellan_enabled=False,
                )
            )
            service.magellan = FakeMagellan()
            service._emit_stale_warning_if_needed = lambda *_args, **_kwargs: None
            seed_index = pd.to_datetime(
                [
                    "2026-05-13T00:00:00Z",
                    "2026-05-13T00:15:00Z",
                    "2026-05-13T00:30:00Z",
                ]
            )
            seed_bars = pd.DataFrame(
                {
                    "open": [100.0, 101.0, 102.0],
                    "high": [101.0, 102.0, 103.0],
                    "low": [99.0, 100.0, 101.0],
                    "close": [100.5, 101.5, 102.5],
                    "volume": [10.0, 11.0, 12.0],
                },
                index=seed_index,
            )
            current_index = pd.to_datetime(
                [
                    "2026-05-13T00:15:00Z",
                    "2026-05-13T00:30:00Z",
                    "2026-05-13T00:45:00Z",
                    "2026-05-13T01:00:00Z",
                ]
            )
            current_bars = pd.DataFrame(
                {
                    "open": [101.0, 102.0, 103.0, 104.0],
                    "high": [102.0, 103.0, 104.0, 105.0],
                    "low": [100.0, 101.0, 102.0, 103.0],
                    "close": [101.5, 102.5, 103.5, 104.5],
                    "volume": [11.0, 12.0, 13.0, 14.0],
                },
                index=current_index,
            )
            service.build_bars = lambda *_args, **_kwargs: (current_bars.copy(), "ok")
            service.indicator_series = lambda bars, *_args, **_kwargs: (
                {"SMA": pd.Series([10.0, 20.0, 30.0, 40.0], index=bars.index)},
                {},
                {"SMA": {"color": "#4da3ff"}},
            )
            session = live_monitor_chart_service.LiveChartSession(
                session_id="live-monitor:test:SOXL",
                deployment_id="deploy-1",
                target_id="",
                symbol="SOXL",
                timeframe="15m",
                lookback="3mo",
                indicator_ids=[],
                selected={"deployment_id": "deploy-1", "status": "live"},
                bars=seed_bars.copy(),
                equity_curve=pd.Series([1234.0], index=[seed_bars.index[-1]], name="equity"),
                strategy_contexts=[],
                local_trade_markers=[],
                last_sent_bar_ts_ns=int(pd.Timestamp(seed_bars.index[-1]).value),
                last_sent_bar_index=2,
                last_completed_replacement_bar_ts_ns=int(pd.Timestamp(seed_bars.index[-1]).value),
                bar_index_by_ts_ns={
                    int(pd.Timestamp(timestamp).value): index
                    for index, timestamp in enumerate(seed_bars.index)
                },
                next_live_bar_index=3,
            )

            service.poll_session(session)

            self.assertEqual(len(service.magellan.live_updates), 1)
            update = service.magellan.live_updates[0][1]
            self.assertEqual([bar["bar_index"] for bar in update["bars"]], [3, 4])
            self.assertEqual(
                [bar["timestamp_utc_ns"] for bar in update["bars"]],
                [
                    str(int(pd.Timestamp("2026-05-13T00:45:00Z").value)),
                    str(int(pd.Timestamp("2026-05-13T01:00:00Z").value)),
                ],
            )
            overlay_points = update["overlay_series"][0]["points"]
            self.assertEqual([point["bar_index"] for point in overlay_points], [3, 4])
            self.assertEqual([point["value"] for point in overlay_points], [30.0, 40.0])
            self.assertEqual(session.last_sent_bar_index, 4)
            self.assertEqual(session.next_live_bar_index, 5)

    def test_runner_sync_window_detects_middle_gap_even_when_latest_bar_is_current(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            service = LiveDeploymentRunnerService(
                DeploymentRunnerConfig(
                    catalog_path=root / "backtests.sqlite",
                    live_store_path=root / "live_market.sqlite",
                    streams_enabled=False,
                    run_once=True,
                )
            )
            context = DeploymentContext(
                context_id="deploy-1:deploy-1:SOXL",
                deployment_id="deploy-1",
                parent_deployment_id="",
                portfolio_id="",
                symbol="SOXL",
                dataset_id="",
                strategy_block_id="",
                strategy_name="SMACrossStrategy",
                strategy_version="",
                params={"fast": 2, "slow": 3, "target": 1.0},
                timeframe="15m",
                candidate_id="",
                source_type="manual",
                source_id="manual-1",
                target_id="target-1",
                target_name="Target",
                webhook_url="http://127.0.0.1/webhook",
                secret="secret",
                sizing={"qty_type": "cash", "qty_value": 100.0},
                sizing_config_source="deployment",
                account_snapshot={"equity": 10_000.0},
                position_qty=0.0,
                avg_price=0.0,
            )
            for ts, price in (
                ("2026-05-07T23:59:00Z", 152.43),
                ("2026-05-08T13:31:00Z", 163.26),
            ):
                service.store.upsert_bar(
                    LiveMarketBar(
                        symbol="SOXL",
                        timestamp=pd.Timestamp(ts),
                        open=price,
                        high=price,
                        low=price,
                        close=price,
                        volume=100.0,
                    )
                )

            window = service.context_live_sync_window(context, pd.Timestamp("2026-05-08T13:31:00Z"))

            self.assertIsNotNone(window)
            start, end = window
            self.assertLessEqual(start, pd.Timestamp("2026-05-08T08:00:00Z"))
            self.assertEqual(end, pd.Timestamp("2026-05-08T13:31:00Z"))

    def test_runner_sync_window_tolerates_short_gap_inside_strategy_timeframe(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            service = LiveDeploymentRunnerService(
                DeploymentRunnerConfig(
                    catalog_path=root / "backtests.sqlite",
                    live_store_path=root / "live_market.sqlite",
                    streams_enabled=False,
                    run_once=True,
                )
            )
            context = DeploymentContext(
                context_id="deploy-1:deploy-1:SOXL",
                deployment_id="deploy-1",
                parent_deployment_id="",
                portfolio_id="",
                symbol="SOXL",
                dataset_id="SOXL_interactive_brokers_10y_1m",
                strategy_block_id="",
                strategy_name="SMACrossStrategy",
                strategy_version="",
                params={"fast": 2, "slow": 3, "target": 1.0},
                timeframe="15m",
                candidate_id="",
                source_type="manual",
                source_id="manual-1",
                target_id="target-1",
                target_name="Target",
                webhook_url="http://127.0.0.1/webhook",
                secret="secret",
                sizing={"qty_type": "cash", "qty_value": 100.0},
                sizing_config_source="deployment",
                account_snapshot={"equity": 10_000.0},
                position_qty=0.0,
                avg_price=0.0,
            )
            index = pd.date_range("2026-05-08T08:00:00Z", "2026-05-08T13:40:00Z", freq="1min")
            gap_start = pd.Timestamp("2026-05-08T10:00:00Z")
            gap_end = pd.Timestamp("2026-05-08T10:20:00Z")
            index = index[(index <= gap_start) | (index >= gap_end)]
            service.store.upsert_bars(
                [
                    LiveMarketBar(
                        symbol="SOXL",
                        timestamp=ts,
                        open=100.0,
                        high=100.0,
                        low=100.0,
                        close=100.0,
                        volume=100.0,
                    )
                    for ts in index
                ]
            )
            historical = pd.DataFrame(
                {
                    "open": [99.0],
                    "high": [99.0],
                    "low": [99.0],
                    "close": [99.0],
                    "volume": [100.0],
                },
                index=pd.to_datetime(["2026-05-08T07:59:00Z"]),
            )

            with patch("backtest_engine.live_deployment_runner.DuckDBStore.load_range", return_value=historical):
                window = service.context_live_sync_window(context, pd.Timestamp("2026-05-08T13:41:00Z"))

            self.assertIsNone(window)

    def test_runner_skips_evaluation_when_gap_sync_fails_without_marking_bar_processed(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            service = LiveDeploymentRunnerService(
                DeploymentRunnerConfig(
                    catalog_path=root / "backtests.sqlite",
                    live_store_path=root / "live_market.sqlite",
                    streams_enabled=False,
                    run_once=True,
                )
            )
            context = DeploymentContext(
                context_id="deploy-1:deploy-1:SOXL",
                deployment_id="deploy-1",
                parent_deployment_id="",
                portfolio_id="",
                symbol="SOXL",
                dataset_id="",
                strategy_block_id="",
                strategy_name="SMACrossStrategy",
                strategy_version="",
                params={"fast": 2, "slow": 3, "target": 1.0},
                timeframe="15m",
                candidate_id="",
                source_type="manual",
                source_id="manual-1",
                target_id="target-1",
                target_name="Target",
                webhook_url="http://127.0.0.1/webhook",
                secret="secret",
                sizing={"qty_type": "cash", "qty_value": 100.0},
                sizing_config_source="deployment",
                account_snapshot={"equity": 10_000.0},
                position_qty=0.0,
                avg_price=0.0,
            )
            index = pd.date_range("2026-05-08T12:00:00Z", periods=6, freq="15min")
            context.cached_bars = pd.DataFrame(
                {
                    "open": [1.0, 1.0, 2.0, 3.0, 5.0, 8.0],
                    "high": [1.0, 1.0, 2.0, 3.0, 5.0, 8.0],
                    "low": [1.0, 1.0, 2.0, 3.0, 5.0, 8.0],
                    "close": [1.0, 1.0, 2.0, 3.0, 5.0, 8.0],
                    "volume": [1.0] * 6,
                },
                index=index,
            )
            record = {
                "symbol": "SOXL",
                "ts_utc": "2026-05-08T13:15:00Z",
                "received_at": "2026-05-08T13:31:00Z",
                "open": 8.0,
                "high": 8.0,
                "low": 8.0,
                "close": 8.0,
                "volume": 1.0,
            }

            with patch.object(service, "sync_context_live_gap_if_needed", return_value=False):
                service.process_context_bar(context, record, pd.Timestamp("2026-05-08T13:31:00Z"), "SOXL")

            self.assertEqual(context.last_processed_bar_ts_ns, 0)

    def test_runner_evaluates_first_bar_after_expected_overnight_session_gap(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            service = LiveDeploymentRunnerService(
                DeploymentRunnerConfig(
                    catalog_path=root / "backtests.sqlite",
                    live_store_path=root / "live_market.sqlite",
                    streams_enabled=False,
                    run_once=True,
                )
            )
            context = DeploymentContext(
                context_id="deploy-1:deploy-1:SOXL",
                deployment_id="deploy-1",
                parent_deployment_id="",
                portfolio_id="",
                symbol="SOXL",
                dataset_id="",
                strategy_block_id="",
                strategy_name="SMACrossStrategy",
                strategy_version="",
                params={"fast": 2, "slow": 3, "target": 1.0},
                timeframe="15m",
                candidate_id="",
                source_type="manual",
                source_id="manual-1",
                target_id="target-1",
                target_name="Target",
                webhook_url="http://127.0.0.1/webhook",
                secret="secret",
                sizing={"qty_type": "cash", "qty_value": 100.0},
                sizing_config_source="deployment",
                account_snapshot={"equity": 10_000.0},
                position_qty=0.0,
                avg_price=0.0,
            )
            index = pd.to_datetime(
                [
                    "2026-05-12T23:30:00Z",
                    "2026-05-12T23:45:00Z",
                    "2026-05-13T08:00:00Z",
                    "2026-05-13T08:15:00Z",
                ],
                utc=True,
            )
            context.cached_bars = pd.DataFrame(
                {
                    "open": [100.0, 100.0, 100.0, 100.0],
                    "high": [100.0, 100.0, 100.0, 100.0],
                    "low": [100.0, 100.0, 100.0, 100.0],
                    "close": [100.0, 100.0, 100.0, 100.0],
                    "volume": [1.0] * 4,
                },
                index=index,
            )

            service.process_context_bar(context, None, pd.Timestamp("2026-05-13T08:15:00Z"), "SOXL")

            self.assertEqual(context.last_processed_bar_ts_ns, int(pd.Timestamp("2026-05-13T08:00:00Z").value))
            events = service.catalog.load_deployment_runner_events(after_seq=0)
            self.assertFalse(any(row.event_type == "data_gap_skipped" for row in events))

    def test_runner_replays_missed_completed_bars_after_resume_and_sends_exit(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            service = LiveDeploymentRunnerService(
                DeploymentRunnerConfig(
                    catalog_path=root / "backtests.sqlite",
                    live_store_path=root / "live_market.sqlite",
                    streams_enabled=False,
                    run_once=True,
                )
            )
            context = DeploymentContext(
                context_id="deploy-1:deploy-1:SOXL",
                deployment_id="deploy-1",
                parent_deployment_id="",
                portfolio_id="",
                symbol="SOXL",
                dataset_id="",
                strategy_block_id="",
                strategy_name="SMACrossStrategy",
                strategy_version="",
                params={"fast": 2, "slow": 3, "target": 1.0},
                timeframe="15m",
                candidate_id="",
                source_type="manual",
                source_id="manual-1",
                target_id="target-1",
                target_name="Target",
                webhook_url="http://127.0.0.1/webhook",
                secret="secret",
                sizing={"qty_type": "cash", "qty_value": 100.0},
                sizing_config_source="deployment",
                account_snapshot={"equity": 10_000.0},
                position_qty=1.0,
                avg_price=10.0,
            )
            index = pd.date_range("2026-05-12T13:00:00Z", periods=6, freq="15min")
            context.cached_bars = pd.DataFrame(
                {
                    "open": [10.0, 10.0, 10.0, 13.0, 11.0, 8.0],
                    "high": [10.0, 10.0, 10.0, 13.0, 11.0, 8.0],
                    "low": [10.0, 10.0, 10.0, 13.0, 11.0, 8.0],
                    "close": [10.0, 10.0, 10.0, 13.0, 11.0, 8.0],
                    "volume": [1.0] * 6,
                },
                index=index,
            )
            service.catalog.save_deployment_runner_event(
                event_type="signal_evaluated",
                deployment_id="deploy-1",
                context_id=context.context_id,
                symbol="SOXL",
                message="SOXL evaluated 2026-05-12T14:00:00+00:00: no strategy signal.",
                payload_json={"bar_timestamp_utc": "2026-05-12T14:00:00+00:00"},
            )

            with (
                patch.object(service, "sync_context_live_gap_if_needed", return_value=True),
                patch("backtest_engine.live_deployment_runner.post_deployment_webhook_payload", return_value={"ok": True}),
            ):
                service.replay_context_missed_bars(context, reason="resume", now=pd.Timestamp("2026-05-12T14:30:00Z"))

            self.assertEqual(context.position_qty, 0.0)
            events = service.catalog.load_deployment_runner_events(after_seq=0)
            signal_rows = [row for row in events if str(row.event_type or "") == "signal_sent"]
            self.assertEqual(len(signal_rows), 1)
            payload = _decode_json_dict(signal_rows[0].payload_json)
            self.assertEqual(payload["action"], "EXIT")
            self.assertEqual(payload["side"], "LONG")
            self.assertEqual(payload["bar_timestamp_utc"], "2026-05-12T14:15:00+00:00")

    def test_runner_symbol_stream_reconnects_stale_ibkr_subscription_during_total_session(self) -> None:
        class FakeApp:
            def __init__(self) -> None:
                self.stopped = False

            def flush_stale_partials(self) -> None:
                pass

            def connection_lost(self) -> bool:
                return False

            def stop(self) -> None:
                self.stopped = True

        statuses: list[str] = []
        errors: list[str] = []
        fake_app = FakeApp()
        stream = RunnerSymbolStream(
            symbol="SOXL",
            store=object(),
            config=InteractiveBrokersRealtimeConfig(timeout_seconds=1.0),
            bar_callback=lambda _symbol, _record: None,
            status_callback=lambda _symbol, message: statuses.append(message),
            error_callback=lambda _symbol, message: errors.append(message),
        )
        stream.app = fake_app
        stream.last_connect_monotonic = time.monotonic() - STREAM_STALE_RECONNECT_SECONDS - 5.0
        stream.last_bar_monotonic = time.monotonic() - STREAM_STALE_RECONNECT_SECONDS - 5.0
        reconnects: list[str] = []

        def reconnect_stub() -> None:
            reconnects.append("connect")
            stream.app = FakeApp()
            stream.last_connect_monotonic = time.monotonic()

        stream._connect = reconnect_stub

        with patch("backtest_engine.live_deployment_runner.is_us_equity_total_session_open", return_value=True):
            stream.poll()

        self.assertTrue(fake_app.stopped)
        self.assertEqual(reconnects, ["connect"])
        self.assertFalse(errors)
        self.assertIn("reconnecting", statuses[-1])

    def test_runner_symbol_stream_does_not_reconnect_stale_ibkr_subscription_outside_total_session(self) -> None:
        class FakeApp:
            def __init__(self) -> None:
                self.stopped = False

            def flush_stale_partials(self) -> None:
                pass

            def connection_lost(self) -> bool:
                return False

            def stop(self) -> None:
                self.stopped = True

        fake_app = FakeApp()
        stream = RunnerSymbolStream(
            symbol="SOXL",
            store=object(),
            config=InteractiveBrokersRealtimeConfig(timeout_seconds=1.0),
            bar_callback=lambda _symbol, _record: None,
            status_callback=lambda _symbol, _message: None,
            error_callback=lambda _symbol, _message: None,
        )
        stream.app = fake_app
        stream.last_connect_monotonic = time.monotonic() - STREAM_STALE_RECONNECT_SECONDS - 5.0
        stream.last_bar_monotonic = time.monotonic() - STREAM_STALE_RECONNECT_SECONDS - 5.0
        reconnects: list[str] = []
        stream._connect = lambda: reconnects.append("connect")

        with patch("backtest_engine.live_deployment_runner.is_us_equity_total_session_open", return_value=False):
            stream.poll()

        self.assertFalse(fake_app.stopped)
        self.assertEqual(reconnects, [])

    def test_runner_reloads_stale_cache_before_evaluating_live_bar(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            service = LiveDeploymentRunnerService(
                DeploymentRunnerConfig(
                    catalog_path=root / "backtests.sqlite",
                    live_store_path=root / "live_market.sqlite",
                    streams_enabled=False,
                    run_once=True,
                )
            )
            stale = pd.DataFrame(
                {
                    "open": [10.0, 10.0, 9.0, 8.0],
                    "high": [10.0, 10.0, 9.0, 8.0],
                    "low": [10.0, 10.0, 9.0, 8.0],
                    "close": [10.0, 10.0, 9.0, 8.0],
                    "volume": [1.0, 1.0, 1.0, 1.0],
                },
                index=pd.to_datetime(
                    [
                        "2026-05-05 10:00:00Z",
                        "2026-05-05 10:15:00Z",
                        "2026-05-05 10:30:00Z",
                        "2026-05-05 10:45:00Z",
                    ]
                ),
            )
            live = pd.DataFrame(
                {
                    "open": [10.0, 12.0, 14.0, 16.0],
                    "high": [10.0, 12.0, 14.0, 16.0],
                    "low": [10.0, 12.0, 14.0, 16.0],
                    "close": [10.0, 12.0, 14.0, 16.0],
                    "volume": [1.0, 1.0, 1.0, 1.0],
                },
                index=pd.to_datetime(
                    [
                        "2026-05-06 18:30:00Z",
                        "2026-05-06 18:45:00Z",
                        "2026-05-06 19:00:00Z",
                        "2026-05-06 19:15:00Z",
                    ]
                ),
            )
            service.store.load_recent_bars = lambda *args, **kwargs: live.copy()
            context = DeploymentContext(
                context_id="deploy-1:deploy-1:SOXL",
                deployment_id="deploy-1",
                parent_deployment_id="",
                portfolio_id="",
                symbol="SOXL",
                dataset_id="",
                strategy_block_id="",
                strategy_name="SMACrossStrategy",
                strategy_version="",
                params={"fast": 2, "slow": 4, "target": 1.0},
                timeframe="15m",
                candidate_id="",
                source_type="manual",
                source_id="manual-1",
                target_id="target-1",
                target_name="Target",
                webhook_url="http://127.0.0.1/webhook",
                secret="secret",
                sizing={
                    "qty_type": "cash",
                    "qty_value": 100.0,
                    "execution_config": {"position_sizing_model": "annual_volatility_target", "annual_vol_window": 20},
                },
                sizing_config_source="deployment",
                account_snapshot={"equity": 10_000.0},
                position_qty=0.0,
                avg_price=0.0,
                cached_bars=stale,
            )
            record = {
                "symbol": "SOXL",
                "ts_utc": "2026-05-06T19:30:00Z",
                "received_at": "2026-05-06T19:45:00Z",
                "open": 17.0,
                "high": 17.0,
                "low": 17.0,
                "close": 17.0,
                "volume": 1.0,
            }

            bars = service.load_context_bars(context, record=record)
            strategy = SMACrossStrategy(**context.params)
            strategy.initialize(bars)
            trace = service.strategy_decision_trace(strategy, pd.Timestamp("2026-05-06T19:30:00Z"), bars)

            self.assertNotIn(pd.Timestamp("2026-05-05T10:45:00Z"), bars.index)
            self.assertIn(pd.Timestamp("2026-05-06T19:15:00Z"), bars.index)
            self.assertIn(pd.Timestamp("2026-05-06T19:30:00Z"), bars.index)
            self.assertFalse(trace["crossed_above"])


if __name__ == "__main__":
    unittest.main()
