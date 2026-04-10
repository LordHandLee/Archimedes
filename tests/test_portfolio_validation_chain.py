from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd
from PyQt6 import QtWidgets

from backtest_engine.catalog import ResultCatalog
from ui_qt_dashboard import CatalogReader
from ui_qt_dashboard import PortfolioValidationChainDialog


class PortfolioValidationChainDialogTest(unittest.TestCase):
    def test_dialog_loads_portfolio_candidates_walk_forward_and_monte_carlo(self) -> None:
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "portfolio_validation.sqlite"
            catalog = ResultCatalog(db_path)

            param_key = '{"fast": 5, "slow": 20, "target": 1.0}'
            catalog.save_optimization_candidate(
                study_id="portfolio_opt_demo",
                timeframe="15 minutes",
                start="2024-01-01T14:30:00+00:00",
                end="2024-06-01T20:00:00+00:00",
                param_key=param_key,
                params_json=param_key,
                source_type="manual_selection",
                promotion_reason="Broad plateau",
                status="queued",
                metrics={
                    "robust_score": 1.25,
                    "median_sharpe": 1.10,
                    "median_total_return": 0.24,
                    "worst_max_drawdown": -0.11,
                },
                notes="Looks stable across the shared portfolio universe.",
            )

            folds = pd.DataFrame(
                [
                    {
                        "wf_study_id": "wf_portfolio_demo",
                        "fold_index": 1,
                        "train_study_id": "wf_portfolio_demo_train_fold_001",
                        "timeframe": "15 minutes",
                        "train_start": "2024-01-01T14:30:00+00:00",
                        "train_end": "2024-05-01T20:00:00+00:00",
                        "test_start": "2024-05-02T14:30:00+00:00",
                        "test_end": "2024-06-01T20:00:00+00:00",
                        "selected_param_set_id": param_key,
                        "selected_params_json": param_key,
                        "train_rank": 1,
                        "train_robust_score": 1.25,
                        "test_run_id": "portfolio_run_demo",
                        "status": "finished",
                    }
                ]
            )
            fold_metrics = pd.DataFrame(
                [
                    {
                        "wf_study_id": "wf_portfolio_demo",
                        "fold_index": 1,
                        "train_metrics_json": '{"robust_score": 1.25}',
                        "test_metrics_json": '{"total_return": 0.21, "sharpe": 1.02, "max_drawdown": -0.12}',
                        "degradation_json": '{"total_return_delta": -0.03, "sharpe_delta": -0.08, "max_drawdown_delta": 0.01}',
                        "param_drift_json": '{"switch_count": 0, "params": {}}',
                    }
                ]
            )
            catalog.save_walk_forward_study(
                wf_study_id="wf_portfolio_demo",
                batch_id="wf_portfolio_demo",
                strategy="PortfolioExecution",
                dataset_id="portfolio_batch_demo",
                timeframe="15 minutes",
                candidate_source_mode="reduced_candidate_set",
                param_names=["fast", "slow", "target"],
                schedule_json={
                    "mode": "anchored",
                    "portfolio_mode": "shared_strategy",
                    "dataset_ids": ["asset_a", "asset_b"],
                },
                selection_rule="highest_robust_score",
                params_json={
                    "source_kind": "portfolio",
                    "source_study_id": "portfolio_opt_demo",
                    "source_batch_id": "portfolio_batch_demo",
                    "portfolio_assets": [
                        {"dataset_id": "asset_a", "target_weight": 0.5},
                        {"dataset_id": "asset_b", "target_weight": 0.5},
                    ],
                    "construction_config": {
                        "allocation_ownership": "strategy_owned",
                        "ranking_mode": "none",
                        "rebalance_mode": "on_change",
                    },
                },
                status="finished",
                description="portfolio wf",
                folds=folds,
                fold_metrics=fold_metrics,
                stitched_metrics={"total_return": 0.21, "sharpe": 1.02, "max_drawdown": -0.12},
                stitched_equity=pd.Series(
                    [100_000.0, 121_000.0],
                    index=pd.to_datetime(
                        ["2024-05-02T14:30:00+00:00", "2024-06-01T20:00:00+00:00"],
                        utc=True,
                    ),
                ),
            )

            catalog.save_monte_carlo_study(
                mc_study_id="mc_portfolio_demo",
                source_type="walk_forward",
                source_id="wf_portfolio_demo",
                resampling_mode="trade_bootstrap",
                simulation_count=128,
                seed=7,
                cost_stress_json={"bps": 1.0},
                status="finished",
                description="portfolio mc",
                source_trade_count=42,
                starting_equity=100_000.0,
                summary_json={
                    "terminal_return_p50": 0.18,
                    "max_drawdown_p95": -0.19,
                    "loss_probability": 0.11,
                    "source_is_portfolio": True,
                    "source_portfolio_mode": "shared_strategy",
                    "source_portfolio_assets": ["asset_a", "asset_b"],
                    "source_study_id": "portfolio_opt_demo",
                    "source_batch_id": "portfolio_batch_demo",
                },
                fan_quantiles_json={"p50": [100_000.0, 118_000.0]},
                terminal_returns_json=[0.18, 0.20],
                max_drawdowns_json=[-0.10, -0.19],
                terminal_equities_json=[118_000.0, 120_000.0],
                original_path_json=[100_000.0, 121_000.0],
                representative_paths=[
                    {
                        "path_id": "median",
                        "path_type": "median_path",
                        "path": [100_000.0, 118_000.0],
                        "summary": {"terminal_return": 0.18, "max_drawdown": -0.10},
                    }
                ],
            )

            reader = CatalogReader(db_path)
            dlg = PortfolioValidationChainDialog(
                title_context="portfolio_opt_demo",
                catalog=reader,
                portfolio_mode="shared_strategy",
                dataset_ids=["asset_a", "asset_b"],
                batch_params={
                    "_portfolio_dataset_ids": ["asset_a", "asset_b"],
                    "_portfolio_target_weights": {"asset_a": 0.5, "asset_b": 0.5},
                    "_portfolio_allocation_mode": "equal_weight",
                    "_portfolio_allocation_ownership": "strategy_owned",
                    "_portfolio_ranking_mode": "none",
                    "_portfolio_rebalance_mode": "on_change",
                },
                source_study_id="portfolio_opt_demo",
                source_batch_id="portfolio_batch_demo",
                initial_tab="candidates",
            )

            self.assertEqual(dlg.candidate_table.rowCount(), 1)
            self.assertEqual(dlg.wf_table.rowCount(), 1)
            self.assertEqual(dlg.mc_table.rowCount(), 1)
            self.assertIn("Linked Walk-Forward Studies: 1", dlg.candidate_notes.toPlainText())
            self.assertIn("Linked Monte Carlo Studies: 1", dlg.candidate_notes.toPlainText())
            self.assertEqual(dlg.structure_table.rowCount(), 2)
            dlg.close()
            app.processEvents()


if __name__ == "__main__":
    unittest.main()
