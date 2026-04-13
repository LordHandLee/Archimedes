from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from backtest_engine.catalog import ResultCatalog


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
            )
            targets = catalog.load_deployment_targets()
            self.assertEqual(len(targets), 1)
            self.assertEqual(targets[0].target_id, "algo_engine_live")
            self.assertEqual(targets[0].mode, "live")

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


if __name__ == "__main__":
    unittest.main()
