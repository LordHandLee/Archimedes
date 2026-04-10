from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd
from PyQt6 import QtWidgets

from backtest_engine.catalog import ResultCatalog
from backtest_engine.metrics import PerformanceMetrics
from backtest_engine.monte_carlo import (
    MONTE_CARLO_MODE_BOOTSTRAP,
    MONTE_CARLO_MODE_RESHUFFLE,
    MONTE_CARLO_SOURCE_WALK_FORWARD,
    extract_walk_forward_trade_returns,
    run_monte_carlo_study,
)
from ui_qt_dashboard import MonteCarloStudyDialog


def _trade(timestamp: str, side: str, qty: float, equity_after: float, realized_pnl: float) -> SimpleNamespace:
    return SimpleNamespace(
        timestamp=pd.Timestamp(timestamp, tz="UTC"),
        side=side,
        qty=qty,
        price=100.0,
        fee=10.0,
        realized_pnl=realized_pnl,
        equity_after=equity_after,
    )


class MonteCarloStudyTest(unittest.TestCase):
    def _seed_walk_forward_source(self, catalog: ResultCatalog, *, portfolio: bool = False, portfolio_mode: str = "shared_strategy") -> None:
        metrics = PerformanceMetrics(0.0, 0.0, -0.05, 0.0, 0.0)
        for run_id in ("wf_run_1", "wf_run_2"):
            catalog.save(
                run_id=run_id,
                batch_id="wf_batch",
                strategy="PortfolioExecution" if portfolio else "SMACrossStrategy",
                params={"fast": 5, "slow": 20, "target": 1.0},
                timeframe="1 minutes",
                start="2024-01-02T14:30:00+00:00",
                end="2024-01-02T20:00:00+00:00",
                dataset_id="wf_portfolio" if portfolio else "wf_asset",
                starting_cash=100_000.0,
                metrics=metrics,
                status="finished",
            )
        catalog.save_trades(
            "wf_run_1",
            [
                _trade("2024-01-02T14:31:00+00:00", "buy", 1.0, 99_990.0, 0.0),
                _trade("2024-01-02T15:00:00+00:00", "sell", -1.0, 100_180.0, 200.0),
                _trade("2024-01-02T15:05:00+00:00", "buy", 1.0, 100_170.0, 200.0),
                _trade("2024-01-02T15:40:00+00:00", "sell", -1.0, 100_070.0, 100.0),
            ],
        )
        catalog.save_trades(
            "wf_run_2",
            [
                _trade("2024-01-03T14:31:00+00:00", "buy", 1.0, 99_990.0, 0.0),
                _trade("2024-01-03T15:10:00+00:00", "sell", -1.0, 100_290.0, 310.0),
                _trade("2024-01-03T15:14:00+00:00", "buy", 1.0, 100_280.0, 310.0),
                _trade("2024-01-03T15:48:00+00:00", "sell", -1.0, 100_330.0, 370.0),
            ],
        )
        folds = pd.DataFrame(
            [
                {
                    "wf_study_id": "wf_demo",
                    "fold_index": 1,
                    "train_study_id": "wf_demo_train_fold_001",
                    "timeframe": "1 minutes",
                    "train_start": "2024-01-01T14:30:00+00:00",
                    "train_end": "2024-01-02T14:30:00+00:00",
                    "test_start": "2024-01-02T14:31:00+00:00",
                    "test_end": "2024-01-02T20:00:00+00:00",
                    "selected_param_set_id": "p1",
                    "selected_params_json": "{\"fast\":5,\"slow\":20,\"target\":1.0}",
                    "train_rank": 1,
                    "train_robust_score": 1.2,
                    "test_run_id": "wf_run_1",
                    "status": "finished",
                },
                {
                    "wf_study_id": "wf_demo",
                    "fold_index": 2,
                    "train_study_id": "wf_demo_train_fold_002",
                    "timeframe": "1 minutes",
                    "train_start": "2024-01-01T14:30:00+00:00",
                    "train_end": "2024-01-03T14:30:00+00:00",
                    "test_start": "2024-01-03T14:31:00+00:00",
                    "test_end": "2024-01-03T20:00:00+00:00",
                    "selected_param_set_id": "p1",
                    "selected_params_json": "{\"fast\":5,\"slow\":20,\"target\":1.0}",
                    "train_rank": 1,
                    "train_robust_score": 1.1,
                    "test_run_id": "wf_run_2",
                    "status": "finished",
                },
            ]
        )
        fold_metrics = pd.DataFrame(
            [
                {
                    "wf_study_id": "wf_demo",
                    "fold_index": 1,
                    "train_metrics_json": "{}",
                    "test_metrics_json": "{}",
                    "degradation_json": "{}",
                    "param_drift_json": "{\"switch_count\":0,\"params\":{}}",
                },
                {
                    "wf_study_id": "wf_demo",
                    "fold_index": 2,
                    "train_metrics_json": "{}",
                    "test_metrics_json": "{}",
                    "degradation_json": "{}",
                    "param_drift_json": "{\"switch_count\":0,\"params\":{}}",
                },
            ]
        )
        catalog.save_walk_forward_study(
            wf_study_id="wf_demo",
            batch_id="wf_demo",
            strategy="PortfolioExecution" if portfolio else "SMACrossStrategy",
            dataset_id="wf_portfolio" if portfolio else "wf_asset",
            timeframe="1 minutes",
            candidate_source_mode="reduced_candidates",
            param_names=["fast", "slow", "target"],
            schedule_json=(
                {
                    "mode": "anchored",
                    "portfolio_mode": portfolio_mode,
                    "dataset_ids": ["wf_asset_a", "wf_asset_b"],
                }
                if portfolio
                else {"mode": "anchored"}
            ),
            selection_rule="highest_robust_score",
            params_json=(
                {
                    "source_kind": "portfolio",
                    "source_study_id": "portfolio_opt_demo",
                    "source_batch_id": "portfolio_batch_demo",
                }
                if portfolio
                else {}
            ),
            status="finished",
            description="demo",
            folds=folds,
            fold_metrics=fold_metrics,
            stitched_metrics={"total_return": 0.01, "sharpe": 1.0, "max_drawdown": -0.05},
            stitched_equity=pd.Series(
                [100_000.0, 100_150.0, 100_330.0],
                index=pd.to_datetime(
                    ["2024-01-02T14:31:00+00:00", "2024-01-02T20:00:00+00:00", "2024-01-03T20:00:00+00:00"],
                    utc=True,
                ),
            ),
        )

    def test_extract_walk_forward_trade_returns_uses_completed_trade_cycles(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            catalog = ResultCatalog(Path(tmpdir) / "mc.sqlite")
            self._seed_walk_forward_source(catalog)
            returns, meta = extract_walk_forward_trade_returns(catalog, "wf_demo")
            self.assertEqual(len(returns), 4)
            self.assertEqual(meta["unit_mode"], "trade_cycle")
            self.assertEqual(meta["run_ids"], ["wf_run_1", "wf_run_2"])
            self.assertGreater(float(returns[0]), 0.0)
            self.assertLess(float(returns[1]), 0.0)

    def test_extract_walk_forward_trade_returns_includes_portfolio_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            catalog = ResultCatalog(Path(tmpdir) / "mc.sqlite")
            self._seed_walk_forward_source(catalog, portfolio=True, portfolio_mode="strategy_blocks")
            returns, meta = extract_walk_forward_trade_returns(catalog, "wf_demo")
            self.assertEqual(len(returns), 4)
            self.assertTrue(bool(meta["is_portfolio"]))
            self.assertEqual(meta["portfolio_mode"], "strategy_blocks")
            self.assertEqual(meta["portfolio_dataset_ids"], ["wf_asset_a", "wf_asset_b"])
            self.assertEqual(meta["source_study_id"], "portfolio_opt_demo")
            self.assertEqual(meta["source_batch_id"], "portfolio_batch_demo")

    def test_run_monte_carlo_bootstrap_persists_study_and_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            catalog = ResultCatalog(Path(tmpdir) / "mc.sqlite")
            self._seed_walk_forward_source(catalog)
            artifacts = run_monte_carlo_study(
                mc_study_id="mc_demo",
                source_type=MONTE_CARLO_SOURCE_WALK_FORWARD,
                source_id="wf_demo",
                catalog=catalog,
                resampling_mode=MONTE_CARLO_MODE_BOOTSTRAP,
                simulation_count=64,
                seed=7,
                cost_stress_bps=2.5,
                description="demo bootstrap",
            )
            self.assertEqual(artifacts.simulation_count, 64)
            self.assertEqual(artifacts.source_trade_count, 4)
            self.assertIn("loss_probability", artifacts.summary)
            self.assertEqual(len(artifacts.original_path), 5)
            self.assertGreaterEqual(len(artifacts.representative_paths), 3)

            studies = catalog.load_monte_carlo_studies()
            self.assertEqual(len(studies), 1)
            self.assertEqual(studies[0].mc_study_id, "mc_demo")
            paths = catalog.load_monte_carlo_paths("mc_demo")
            self.assertTrue(any(path.path_type == "median_path" for path in paths))

    def test_run_monte_carlo_persists_portfolio_source_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            catalog = ResultCatalog(Path(tmpdir) / "mc.sqlite")
            self._seed_walk_forward_source(catalog, portfolio=True, portfolio_mode="shared_strategy")
            artifacts = run_monte_carlo_study(
                mc_study_id="mc_portfolio",
                source_type=MONTE_CARLO_SOURCE_WALK_FORWARD,
                source_id="wf_demo",
                catalog=catalog,
                resampling_mode=MONTE_CARLO_MODE_BOOTSTRAP,
                simulation_count=32,
                seed=11,
            )
            self.assertTrue(bool(artifacts.summary["source_is_portfolio"]))
            self.assertEqual(artifacts.summary["source_portfolio_mode"], "shared_strategy")
            self.assertEqual(artifacts.summary["source_portfolio_asset_count"], 2)
            self.assertEqual(artifacts.summary["source_study_id"], "portfolio_opt_demo")
            self.assertEqual(artifacts.summary["source_batch_id"], "portfolio_batch_demo")

    def test_run_monte_carlo_reshuffle_keeps_same_trade_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            catalog = ResultCatalog(Path(tmpdir) / "mc.sqlite")
            self._seed_walk_forward_source(catalog)
            artifacts = run_monte_carlo_study(
                mc_study_id="mc_demo_reshuffle",
                source_type=MONTE_CARLO_SOURCE_WALK_FORWARD,
                source_id="wf_demo",
                catalog=catalog,
                resampling_mode=MONTE_CARLO_MODE_RESHUFFLE,
                simulation_count=32,
                seed=3,
            )
            self.assertEqual(len(artifacts.terminal_returns), 32)
            self.assertEqual(len(artifacts.original_path), artifacts.source_trade_count + 1)

    def test_study_dialog_handles_constant_histogram_data(self) -> None:
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

        class _FakeCatalog:
            @staticmethod
            def load_monte_carlo_paths(_mc_study_id: str) -> pd.DataFrame:
                return pd.DataFrame(
                    [
                        {
                            "path_type": "median_path",
                            "path_id": "p50",
                            "summary_json": "{\"terminal_return\":0.0,\"max_drawdown\":-0.1}",
                        }
                    ]
                )

        dlg = MonteCarloStudyDialog(
            {
                "mc_study_id": "mc_constant",
                "source_type": "walk_forward",
                "source_id": "wf_demo",
                "resampling_mode": MONTE_CARLO_MODE_BOOTSTRAP,
                "simulation_count": 16,
                "seed": 7,
                "summary_json": (
                    "{\"terminal_return_p50\":0.0,\"terminal_return_p05\":0.0,"
                    "\"terminal_return_p95\":0.0,\"max_drawdown_p50\":-0.1,"
                    "\"max_drawdown_p95\":-0.1,\"loss_probability\":0.0}"
                ),
                "fan_quantiles_json": "{\"p05\":[100000,100000],\"p25\":[100000,100000],"
                "\"p50\":[100000,100000],\"p75\":[100000,100000],\"p95\":[100000,100000]}",
                "original_path_json": "[100000,100000]",
                "terminal_returns_json": "[0.0,0.0,0.0,0.0]",
                "max_drawdowns_json": "[-0.1,-0.1,-0.1,-0.1]",
            },
            _FakeCatalog(),
        )
        self.assertEqual(len(dlg.hist_figure.axes), 2)
        self.assertGreater(sum(len(ax.patches) for ax in dlg.hist_figure.axes), 0)
        dlg.close()
        app.processEvents()


if __name__ == "__main__":
    unittest.main()
