from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from backtest_engine.catalog import ResultCatalog
import pandas as pd

from backtest_engine.live_market import LiveMarketBar, LiveMarketDataStore
from backtest_engine.live_deployment_runner import (
    DeploymentContext,
    DeploymentRunnerConfig,
    LiveDeploymentRunnerService,
    SMACrossStrategy,
    main as live_runner_main,
)
from backtest_engine.live_monitor_chart_service import main as live_chart_main


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
                sizing={"qty_type": "cash", "qty_value": 100.0},
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
