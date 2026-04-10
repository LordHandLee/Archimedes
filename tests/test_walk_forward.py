from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from backtest_engine.catalog import ResultCatalog
from backtest_engine.engine import BacktestConfig
from backtest_engine.execution import ExecutionMode
from backtest_engine.sample_strategies import SMACrossStrategy
from backtest_engine.walk_forward import (
    WALK_FORWARD_SOURCE_FIXED_PORTFOLIO,
    WALK_FORWARD_SOURCE_FULL_GRID,
    WALK_FORWARD_SOURCE_REDUCED_CANDIDATES,
    WalkForwardPortfolioAssetDefinition,
    WalkForwardPortfolioStrategyBlockAssetDefinition,
    WalkForwardPortfolioStrategyBlockDefinition,
    build_anchored_walk_forward_schedule,
    candidate_param_sets_from_records,
    run_walk_forward_portfolio_study,
    run_walk_forward_study,
)


def _make_bars(periods: int = 180, price_offset: float = 0.0, wave_phase: float = 0.0) -> pd.DataFrame:
    index = pd.date_range("2024-01-02 14:30", periods=periods, freq="1min", tz="UTC")
    first_leg = np.linspace(105.0, 92.0, 50)
    second_leg = np.linspace(92.0, 130.0, periods - 50)
    close = (
        np.concatenate([first_leg, second_leg])
        + price_offset
        + (np.sin((np.arange(periods) / 6.0) + wave_phase) * 0.55)
    )
    open_ = np.roll(close, 1)
    open_[0] = close[0] + 0.1
    bars = pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + 0.30,
            "low": np.minimum(open_, close) - 0.30,
            "close": close,
            "volume": 1000,
        },
        index=index,
    )
    return bars


class WalkForwardStudyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.bars = _make_bars()
        self.portfolio_bars = {
            "wf_asset_a": _make_bars(price_offset=0.0, wave_phase=0.0),
            "wf_asset_b": _make_bars(price_offset=3.5, wave_phase=0.65),
        }
        self.base_config = BacktestConfig(
            timeframe="1 minutes",
            use_cache=False,
            base_execution=False,
            allow_short=False,
            fee_rate=0.0,
            slippage=0.0,
            sharpe_basis="period",
        )

    def test_portfolio_walk_forward_worker_preserves_short_support_in_base_config(self) -> None:
        from ui_qt_dashboard import PortfolioWalkForwardWorker

        config = PortfolioWalkForwardWorker._base_config_from_settings(
            timeframe="15 minutes",
            bt_settings={
                "starting_cash": 250_000,
                "fee_rate": 0.001,
                "slippage": 0.002,
                "allow_short": True,
                "fill_on_close": True,
            },
        )

        self.assertEqual(config.timeframe, "15 minutes")
        self.assertEqual(config.starting_cash, 250_000)
        self.assertAlmostEqual(config.fee_rate, 0.001)
        self.assertAlmostEqual(config.slippage, 0.002)
        self.assertTrue(config.fill_on_close)
        self.assertTrue(config.allow_short)

    def test_build_anchored_schedule_uses_expanding_train_windows(self) -> None:
        schedule = build_anchored_walk_forward_schedule(
            index=self.bars.index,
            first_test_start=self.bars.index[60],
            test_window_bars=20,
            num_folds=3,
            min_train_bars=30,
        )
        self.assertEqual(len(schedule), 3)
        self.assertEqual(schedule[0].train_start, self.bars.index[0].isoformat())
        self.assertEqual(schedule[0].train_end, self.bars.index[59].isoformat())
        self.assertEqual(schedule[0].test_start, self.bars.index[60].isoformat())
        self.assertEqual(schedule[0].test_end, self.bars.index[79].isoformat())
        self.assertEqual(schedule[1].train_end, self.bars.index[79].isoformat())
        self.assertEqual(schedule[1].test_start, self.bars.index[80].isoformat())
        self.assertEqual(schedule[2].test_end, self.bars.index[119].isoformat())

    def test_run_walk_forward_full_grid_persists_study_and_train_fold_surfaces(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            catalog = ResultCatalog(Path(tmpdir) / "walk_forward.sqlite")
            artifacts = run_walk_forward_study(
                wf_study_id="wf_full_grid",
                dataset_id="wf_asset",
                data_loader=lambda _: self.bars,
                strategy_cls=SMACrossStrategy,
                base_config=self.base_config,
                timeframe="1 minutes",
                first_test_start=self.bars.index[80],
                test_window_bars=30,
                num_folds=2,
                param_grid={"fast": [3], "slow": [8], "target": [0.0, 1.0]},
                candidate_source_mode=WALK_FORWARD_SOURCE_FULL_GRID,
                catalog=catalog,
                requested_execution_mode=ExecutionMode.AUTO,
                source_study_id="opt_demo",
                source_batch_id="batch_demo",
            )

            self.assertEqual(len(artifacts.folds), 2)
            self.assertEqual(len(artifacts.fold_metrics), 2)
            selected_targets = [
                float(json.loads(text)["target"])
                for text in artifacts.folds["selected_params_json"].tolist()
            ]
            self.assertEqual(selected_targets, [1.0, 1.0])
            self.assertFalse(artifacts.stitched_oos_equity.empty)
            self.assertGreater(float(artifacts.stitched_oos_metrics["total_return"]), 0.0)

            studies = catalog.load_walk_forward_studies()
            self.assertEqual(len(studies), 1)
            self.assertEqual(studies[0].wf_study_id, "wf_full_grid")
            self.assertEqual(studies[0].fold_count, 2)
            self.assertIsNotNone(studies[0].stitched_equity_json)
            stored_params = json.loads(studies[0].params_json)
            self.assertEqual(stored_params["source_study_id"], "opt_demo")
            self.assertEqual(stored_params["source_batch_id"], "batch_demo")

            folds = catalog.load_walk_forward_folds("wf_full_grid")
            self.assertEqual(len(folds), 2)
            self.assertTrue(all(row.train_study_id for row in folds))
            fold_metrics = catalog.load_walk_forward_fold_metrics("wf_full_grid")
            self.assertEqual(len(fold_metrics), 2)

            optimization_studies = catalog.load_optimization_studies()
            train_study_ids = {row.study_id for row in optimization_studies}
            self.assertIn("wf_full_grid_train_fold_001", train_study_ids)
            self.assertIn("wf_full_grid_train_fold_002", train_study_ids)

    def test_run_walk_forward_portfolio_shared_strategy_persists_study(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            catalog = ResultCatalog(Path(tmpdir) / "walk_forward.sqlite")
            artifacts = run_walk_forward_portfolio_study(
                wf_study_id="wf_portfolio_shared",
                portfolio_dataset_id="portfolio_wf_asset",
                data_loader=lambda dataset_id, timeframe: self.portfolio_bars[str(dataset_id)],
                base_config=self.base_config,
                timeframe="1 minutes",
                first_test_start=self.bars.index[80],
                test_window_bars=30,
                num_folds=2,
                shared_strategy_cls=SMACrossStrategy,
                portfolio_assets=[
                    WalkForwardPortfolioAssetDefinition(dataset_id="wf_asset_a", target_weight=0.5),
                    WalkForwardPortfolioAssetDefinition(dataset_id="wf_asset_b", target_weight=0.5),
                ],
                param_grid={"fast": [3], "slow": [8], "target": [0.0, 1.0]},
                candidate_source_mode=WALK_FORWARD_SOURCE_FULL_GRID,
                catalog=catalog,
                requested_execution_mode=ExecutionMode.VECTORIZED,
                source_study_id="portfolio_opt_demo",
                source_batch_id="portfolio_batch_demo",
            )

            self.assertEqual(len(artifacts.folds), 2)
            self.assertFalse(artifacts.stitched_oos_equity.empty)
            selected_targets = [
                float(json.loads(text)["target"])
                for text in artifacts.folds["selected_params_json"].tolist()
            ]
            self.assertEqual(selected_targets, [1.0, 1.0])

            studies = catalog.load_walk_forward_studies()
            self.assertEqual(len(studies), 1)
            self.assertEqual(studies[0].wf_study_id, "wf_portfolio_shared")
            self.assertEqual(studies[0].candidate_source_mode, WALK_FORWARD_SOURCE_FULL_GRID)
            stored_params = json.loads(studies[0].params_json)
            self.assertEqual(stored_params["source_study_id"], "portfolio_opt_demo")
            self.assertEqual(stored_params["source_batch_id"], "portfolio_batch_demo")

            folds = catalog.load_walk_forward_folds("wf_portfolio_shared")
            self.assertEqual(len(folds), 2)
            self.assertTrue(all(row.train_study_id for row in folds))

    def test_run_walk_forward_portfolio_fixed_strategy_blocks_persists_study(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            catalog = ResultCatalog(Path(tmpdir) / "walk_forward.sqlite")
            artifacts = run_walk_forward_portfolio_study(
                wf_study_id="wf_portfolio_blocks",
                portfolio_dataset_id="portfolio_wf_blocks",
                data_loader=lambda dataset_id, timeframe: self.portfolio_bars[str(dataset_id)],
                base_config=self.base_config,
                timeframe="1 minutes",
                first_test_start=self.bars.index[80],
                test_window_bars=30,
                num_folds=2,
                strategy_blocks=[
                    WalkForwardPortfolioStrategyBlockDefinition(
                        block_id="trend_a",
                        strategy_cls=SMACrossStrategy,
                        strategy_params={"fast": 3, "slow": 8, "target": 1.0},
                        assets=[
                            WalkForwardPortfolioStrategyBlockAssetDefinition(
                                dataset_id="wf_asset_a",
                                target_weight=1.0,
                            )
                        ],
                        budget_weight=0.5,
                        display_name="Trend A",
                    ),
                    WalkForwardPortfolioStrategyBlockDefinition(
                        block_id="trend_b",
                        strategy_cls=SMACrossStrategy,
                        strategy_params={"fast": 4, "slow": 10, "target": 1.0},
                        assets=[
                            WalkForwardPortfolioStrategyBlockAssetDefinition(
                                dataset_id="wf_asset_b",
                                target_weight=1.0,
                            )
                        ],
                        budget_weight=0.5,
                        display_name="Trend B",
                    ),
                ],
                strategy_label="Portfolio Blocks [2]",
                candidate_source_mode=WALK_FORWARD_SOURCE_FIXED_PORTFOLIO,
                catalog=catalog,
                requested_execution_mode=ExecutionMode.VECTORIZED,
            )

            self.assertEqual(len(artifacts.folds), 2)
            self.assertEqual(artifacts.candidate_source_mode, WALK_FORWARD_SOURCE_FIXED_PORTFOLIO)
            self.assertFalse(artifacts.stitched_oos_equity.empty)

            folds = catalog.load_walk_forward_folds("wf_portfolio_blocks")
            self.assertEqual(len(folds), 2)
            self.assertTrue(all((row.selected_param_set_id or "") == WALK_FORWARD_SOURCE_FIXED_PORTFOLIO for row in folds))
            self.assertTrue(all(not row.train_study_id for row in folds))
            self.assertFalse(artifacts.stitched_oos_equity.empty)
            self.assertGreater(float(artifacts.stitched_oos_metrics["total_return"]), 0.0)

            studies = catalog.load_walk_forward_studies()
            self.assertEqual(len(studies), 1)
            self.assertEqual(studies[0].wf_study_id, "wf_portfolio_blocks")
            self.assertEqual(studies[0].fold_count, 2)
            self.assertIsNotNone(studies[0].stitched_equity_json)

            folds = catalog.load_walk_forward_folds("wf_portfolio_blocks")
            self.assertEqual(len(folds), 2)
            self.assertTrue(all(not row.train_study_id for row in folds))
            fold_metrics = catalog.load_walk_forward_fold_metrics("wf_portfolio_blocks")
            self.assertEqual(len(fold_metrics), 2)

            optimization_studies = catalog.load_optimization_studies()
            train_study_ids = {row.study_id for row in optimization_studies}
            self.assertNotIn("wf_portfolio_blocks_train_fold_001", train_study_ids)
            self.assertNotIn("wf_portfolio_blocks_train_fold_002", train_study_ids)

    def test_reduced_candidate_mode_accepts_promoted_candidate_param_sets(self) -> None:
        candidate_records = [
            {"params_json": json.dumps({"fast": 3, "slow": 8, "target": 0.0})},
            {"params_json": json.dumps({"fast": 3, "slow": 8, "target": 1.0})},
        ]
        candidate_params = candidate_param_sets_from_records(candidate_records)
        with tempfile.TemporaryDirectory() as tmpdir:
            catalog = ResultCatalog(Path(tmpdir) / "walk_forward.sqlite")
            artifacts = run_walk_forward_study(
                wf_study_id="wf_candidates",
                dataset_id="wf_asset",
                data_loader=lambda _: self.bars,
                strategy_cls=SMACrossStrategy,
                base_config=self.base_config,
                timeframe="1 minutes",
                first_test_start=self.bars.index[70],
                test_window_bars=25,
                num_folds=2,
                candidate_params=candidate_params,
                candidate_source_mode=WALK_FORWARD_SOURCE_REDUCED_CANDIDATES,
                catalog=catalog,
                requested_execution_mode=ExecutionMode.AUTO,
            )

            self.assertEqual(len(artifacts.folds), 2)
            self.assertEqual(artifacts.candidate_source_mode, WALK_FORWARD_SOURCE_REDUCED_CANDIDATES)
            selected_targets = [
                float(json.loads(text)["target"])
                for text in artifacts.folds["selected_params_json"].tolist()
            ]
            self.assertEqual(selected_targets, [1.0, 1.0])

    def test_portfolio_reduced_candidate_mode_accepts_promoted_candidate_param_sets(self) -> None:
        candidate_params = candidate_param_sets_from_records(
            [
                {"params_json": json.dumps({"fast": 3, "slow": 8, "target": 0.0})},
                {"params_json": json.dumps({"fast": 3, "slow": 8, "target": 1.0})},
            ]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            catalog = ResultCatalog(Path(tmpdir) / "walk_forward.sqlite")
            artifacts = run_walk_forward_portfolio_study(
                wf_study_id="wf_portfolio_candidates",
                portfolio_dataset_id="portfolio_wf_candidates",
                data_loader=lambda dataset_id, timeframe: self.portfolio_bars[str(dataset_id)],
                base_config=self.base_config,
                timeframe="1 minutes",
                first_test_start=self.bars.index[70],
                test_window_bars=25,
                num_folds=2,
                shared_strategy_cls=SMACrossStrategy,
                portfolio_assets=[
                    WalkForwardPortfolioAssetDefinition(dataset_id="wf_asset_a", target_weight=0.5),
                    WalkForwardPortfolioAssetDefinition(dataset_id="wf_asset_b", target_weight=0.5),
                ],
                candidate_params=candidate_params,
                candidate_source_mode=WALK_FORWARD_SOURCE_REDUCED_CANDIDATES,
                catalog=catalog,
                requested_execution_mode=ExecutionMode.VECTORIZED,
            )

            self.assertEqual(len(artifacts.folds), 2)
            self.assertEqual(artifacts.candidate_source_mode, WALK_FORWARD_SOURCE_REDUCED_CANDIDATES)
            selected_targets = [
                float(json.loads(text)["target"])
                for text in artifacts.folds["selected_params_json"].tolist()
            ]
            self.assertEqual(selected_targets, [1.0, 1.0])

    def test_strategy_block_portfolio_reduced_candidates_select_best_promoted_definition(self) -> None:
        candidate_definitions = [
            [
                WalkForwardPortfolioStrategyBlockDefinition(
                    block_id="trend_a",
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 3, "slow": 8, "target": 0.0},
                    assets=[
                        WalkForwardPortfolioStrategyBlockAssetDefinition(
                            dataset_id="wf_asset_a",
                            target_weight=1.0,
                        )
                    ],
                    budget_weight=0.5,
                    display_name="Trend A Off",
                ),
                WalkForwardPortfolioStrategyBlockDefinition(
                    block_id="trend_b",
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 4, "slow": 10, "target": 0.0},
                    assets=[
                        WalkForwardPortfolioStrategyBlockAssetDefinition(
                            dataset_id="wf_asset_b",
                            target_weight=1.0,
                        )
                    ],
                    budget_weight=0.5,
                    display_name="Trend B Off",
                ),
            ],
            [
                WalkForwardPortfolioStrategyBlockDefinition(
                    block_id="trend_a",
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 3, "slow": 8, "target": 1.0},
                    assets=[
                        WalkForwardPortfolioStrategyBlockAssetDefinition(
                            dataset_id="wf_asset_a",
                            target_weight=1.0,
                        )
                    ],
                    budget_weight=0.5,
                    display_name="Trend A On",
                ),
                WalkForwardPortfolioStrategyBlockDefinition(
                    block_id="trend_b",
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 4, "slow": 10, "target": 1.0},
                    assets=[
                        WalkForwardPortfolioStrategyBlockAssetDefinition(
                            dataset_id="wf_asset_b",
                            target_weight=1.0,
                        )
                    ],
                    budget_weight=0.5,
                    display_name="Trend B On",
                ),
            ],
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            catalog = ResultCatalog(Path(tmpdir) / "walk_forward.sqlite")
            artifacts = run_walk_forward_portfolio_study(
                wf_study_id="wf_portfolio_blocks_candidates",
                portfolio_dataset_id="portfolio_wf_blocks_candidates",
                data_loader=lambda dataset_id, timeframe: self.portfolio_bars[str(dataset_id)],
                base_config=self.base_config,
                timeframe="1 minutes",
                first_test_start=self.bars.index[70],
                test_window_bars=25,
                num_folds=2,
                strategy_blocks=candidate_definitions[0],
                strategy_block_candidates=candidate_definitions,
                strategy_label="Portfolio Blocks [2]",
                candidate_source_mode=WALK_FORWARD_SOURCE_REDUCED_CANDIDATES,
                catalog=catalog,
                requested_execution_mode=ExecutionMode.VECTORIZED,
            )

            self.assertEqual(len(artifacts.folds), 2)
            self.assertEqual(artifacts.candidate_source_mode, WALK_FORWARD_SOURCE_REDUCED_CANDIDATES)
            selected_payloads = [json.loads(text) for text in artifacts.folds["selected_params_json"].tolist()]
            selected_targets = [
                float(payload["strategy_blocks"][0]["params"]["target"])
                for payload in selected_payloads
            ]
            self.assertEqual(selected_targets, [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
