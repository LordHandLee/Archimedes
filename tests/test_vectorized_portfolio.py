from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pandas as pd

from backtest_engine.catalog import ResultCatalog
from backtest_engine.chart_snapshot import ChartSnapshotExporter
from backtest_engine.engine import BacktestConfig
from backtest_engine.execution import (
    ExecutionMode,
    ExecutionOrchestrator,
    ExecutionRequest,
    PortfolioExecutionAsset,
    PortfolioExecutionRequest,
    UnsupportedExecutionModeError,
)
from backtest_engine.grid_search import GridSpec, PortfolioAssetTarget, run_vectorized_portfolio_grid_search
from backtest_engine.sample_strategies import SMACrossStrategy, ZScoreMeanReversionStrategy
from backtest_engine.vectorized_portfolio import (
    ALLOCATION_OWNERSHIP_HYBRID,
    ALLOCATION_OWNERSHIP_PORTFOLIO,
    ALLOCATION_OWNERSHIP_STRATEGY,
    PortfolioAssetSpec,
    PortfolioConstructionConfig,
    RANKING_MODE_SCORE_THRESHOLD,
    RANKING_MODE_TOP_N,
    RANKING_MODE_TOP_N_OVER_SCORE_THRESHOLD,
    REBALANCE_MODE_ON_CHANGE,
    REBALANCE_MODE_ON_CHANGE_OR_DRIFT,
    REBALANCE_MODE_ON_CHANGE_OR_PERIODIC,
    REBALANCE_MODE_ON_CHANGE_OR_PERIODIC_OR_DRIFT,
    VectorizedPortfolioEngine,
    WEIGHTING_MODE_EQUAL_SELECTED,
    WEIGHTING_MODE_SCORE_PROPORTIONAL,
)


def _make_bars(periods: int = 240, *, drift: float = 12.0, phase: float = 0.0) -> pd.DataFrame:
    index = pd.date_range("2024-01-02 14:30", periods=periods, freq="1min", tz="UTC")
    base = 100.0 + np.linspace(0, drift, periods)
    wave = np.sin((np.arange(periods) / 6.0) + phase) * 1.5
    close = base + wave
    open_ = np.roll(close, 1)
    open_[0] = close[0] - 0.25
    open_ += np.cos((np.arange(periods) / 9.0) + phase) * 0.15
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + 0.4,
            "low": np.minimum(open_, close) - 0.4,
            "close": close,
            "volume": 1000,
        },
        index=index,
    )


def _make_trend_bars(periods: int = 240, *, slope: float = 0.1, start: float = 100.0) -> pd.DataFrame:
    index = pd.date_range("2024-01-02 14:30", periods=periods, freq="1min", tz="UTC")
    close = start + (np.arange(periods, dtype=float) * slope)
    open_ = np.roll(close, 1)
    open_[0] = close[0] - slope
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + 0.25,
            "low": np.minimum(open_, close) - 0.25,
            "close": close,
            "volume": 1000,
        },
        index=index,
    )


class VectorizedPortfolioTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = BacktestConfig(
            timeframe="1 minutes",
            use_cache=False,
            base_execution=False,
            allow_short=False,
            fee_rate=0.0002,
            slippage=0.0001,
        )
        self.bars_a = _make_bars()
        self.bars_b = _make_bars(drift=10.0, phase=0.7)

    def test_single_asset_portfolio_matches_vectorized_single_run(self) -> None:
        params = {"fast": 5, "slow": 20, "target": 1.0}
        orchestrator = ExecutionOrchestrator()
        single = orchestrator.execute(
            ExecutionRequest(
                data=self.bars_a,
                dataset_id="asset_a",
                strategy_cls=SMACrossStrategy,
                strategy_params=params,
                config=self.config,
                requested_execution_mode=ExecutionMode.VECTORIZED,
            )
        )
        portfolio = VectorizedPortfolioEngine().run(
            [
                PortfolioAssetSpec(
                    dataset_id="asset_a",
                    data=self.bars_a,
                    strategy_cls=SMACrossStrategy,
                    strategy_params=params,
                    target_weight=1.0,
                )
            ],
            self.config,
            normalize_weights=False,
        )

        self.assertTrue(
            np.allclose(
                single.equity_curve.to_numpy(),
                portfolio.portfolio_equity_curve.to_numpy(),
                atol=1e-8,
            )
        )
        self.assertAlmostEqual(single.metrics.total_return, portfolio.metrics.total_return, places=10)
        self.assertAlmostEqual(single.metrics.max_drawdown, portfolio.metrics.max_drawdown, places=10)
        self.assertEqual(len(portfolio.trades), len(single.trades))

    def test_multi_asset_portfolio_equal_weights_share_cash(self) -> None:
        engine = VectorizedPortfolioEngine()
        result = engine.run(
            [
                PortfolioAssetSpec(
                    dataset_id="asset_a",
                    data=self.bars_a,
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 5, "slow": 20, "target": 1.0},
                ),
                PortfolioAssetSpec(
                    dataset_id="asset_b",
                    data=self.bars_b,
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 5, "slow": 20, "target": 1.0},
                ),
            ],
            self.config,
            normalize_weights=True,
        )

        self.assertEqual(result.asset_weights.columns.tolist(), ["asset_a", "asset_b"])
        self.assertEqual(len(result.portfolio_equity_curve), len(self.bars_a))
        self.assertLessEqual(float(result.target_weights.sum(axis=1).max()), 1.000001)
        both_active = result.target_weights[(result.target_weights["asset_a"] > 0) & (result.target_weights["asset_b"] > 0)]
        self.assertFalse(both_active.empty)
        self.assertTrue(np.allclose(both_active["asset_a"].to_numpy(), 0.5, atol=1e-6))
        self.assertTrue(np.allclose(both_active["asset_b"].to_numpy(), 0.5, atol=1e-6))
        trade_assets = {trade.dataset_id for trade in result.trades}
        self.assertIn("asset_a", trade_assets)
        self.assertIn("asset_b", trade_assets)

    def test_multi_asset_portfolio_caps_gross_exposure_when_weights_sum_above_one(self) -> None:
        engine = VectorizedPortfolioEngine(max_gross_exposure=1.0)
        result = engine.run(
            [
                PortfolioAssetSpec(
                    dataset_id="asset_a",
                    data=self.bars_a,
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 5, "slow": 20, "target": 1.0},
                    target_weight=0.8,
                ),
                PortfolioAssetSpec(
                    dataset_id="asset_b",
                    data=self.bars_b,
                    strategy_cls=ZScoreMeanReversionStrategy,
                    strategy_params={
                        "half_life_lookback": 30,
                        "std_len": 15,
                        "long_entry_z": -0.8,
                        "long_exit_z": 0.0,
                        "target": 1.0,
                    },
                    target_weight=0.8,
                ),
            ],
            replace(self.config, fill_on_close=True),
            normalize_weights=False,
        )

        self.assertLessEqual(float(result.target_weights.sum(axis=1).max()), 1.000001)

    def test_support_rejects_shorts_and_base_execution(self) -> None:
        engine = VectorizedPortfolioEngine()
        support = engine.supports(
            [
                PortfolioAssetSpec(
                    dataset_id="asset_a",
                    data=self.bars_a,
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 5, "slow": 20, "target": 1.0},
                )
            ],
            replace(self.config, allow_short=True),
        )
        self.assertFalse(support.supported)
        self.assertIn("long-only", support.reason or "")

        support = engine.supports(
            [
                PortfolioAssetSpec(
                    dataset_id="asset_a",
                    data=self.bars_a,
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 5, "slow": 20, "target": 1.0},
                )
            ],
            replace(self.config, base_execution=True),
        )
        self.assertFalse(support.supported)
        self.assertIn("base_execution", support.reason or "")

    def test_support_rejects_ranking_in_strategy_owned_mode(self) -> None:
        engine = VectorizedPortfolioEngine()
        support = engine.supports(
            [
                PortfolioAssetSpec(
                    dataset_id="asset_a",
                    data=self.bars_a,
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 5, "slow": 20, "target": 1.0},
                ),
                PortfolioAssetSpec(
                    dataset_id="asset_b",
                    data=self.bars_b,
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 5, "slow": 20, "target": 1.0},
                ),
            ],
            self.config,
            PortfolioConstructionConfig(
                allocation_ownership=ALLOCATION_OWNERSHIP_STRATEGY,
                ranking_mode=RANKING_MODE_TOP_N,
                max_ranked_assets=1,
            ),
        )
        self.assertFalse(support.supported)
        self.assertIn("Strategy-owned", support.reason or "")

    def test_support_rejects_weighting_override_outside_portfolio_owned(self) -> None:
        engine = VectorizedPortfolioEngine()
        support = engine.supports(
            [
                PortfolioAssetSpec(
                    dataset_id="asset_a",
                    data=self.bars_a,
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 5, "slow": 20, "target": 1.0},
                ),
                PortfolioAssetSpec(
                    dataset_id="asset_b",
                    data=self.bars_b,
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 5, "slow": 20, "target": 1.0},
                ),
            ],
            self.config,
            PortfolioConstructionConfig(
                allocation_ownership=ALLOCATION_OWNERSHIP_HYBRID,
                weighting_mode=WEIGHTING_MODE_EQUAL_SELECTED,
            ),
        )
        self.assertFalse(support.supported)
        self.assertIn("Portfolio-Owned", support.reason or "")

    def test_support_rejects_max_asset_weight_below_min_active_weight(self) -> None:
        engine = VectorizedPortfolioEngine()
        support = engine.supports(
            [
                PortfolioAssetSpec(
                    dataset_id="asset_a",
                    data=self.bars_a,
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 5, "slow": 20, "target": 1.0},
                ),
                PortfolioAssetSpec(
                    dataset_id="asset_b",
                    data=self.bars_b,
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 5, "slow": 20, "target": 1.0},
                ),
            ],
            self.config,
            PortfolioConstructionConfig(
                allocation_ownership=ALLOCATION_OWNERSHIP_PORTFOLIO,
                min_active_weight=0.25,
                max_asset_weight=0.2,
            ),
        )
        self.assertFalse(support.supported)
        self.assertIn("max_asset_weight", support.reason or "")

    def test_portfolio_owned_top_n_ranking_selects_single_asset(self) -> None:
        engine = VectorizedPortfolioEngine()
        result = engine.run(
            [
                PortfolioAssetSpec(
                    dataset_id="asset_a",
                    data=self.bars_a,
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 5, "slow": 20, "target": 1.0},
                ),
                PortfolioAssetSpec(
                    dataset_id="asset_b",
                    data=self.bars_b,
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 5, "slow": 20, "target": 1.0},
                ),
            ],
            self.config,
            normalize_weights=True,
            construction_config=PortfolioConstructionConfig(
                allocation_ownership=ALLOCATION_OWNERSHIP_PORTFOLIO,
                ranking_mode=RANKING_MODE_TOP_N,
                max_ranked_assets=1,
            ),
        )

        positive_counts = (result.target_weights > 1e-9).sum(axis=1)
        self.assertLessEqual(int(positive_counts.max()), 1)
        self.assertGreater(int((positive_counts == 1).sum()), 0)

    def test_support_rejects_score_threshold_without_min_score(self) -> None:
        engine = VectorizedPortfolioEngine()
        support = engine.supports(
            [
                PortfolioAssetSpec(
                    dataset_id="asset_a",
                    data=self.bars_a,
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 5, "slow": 20, "target": 1.0},
                ),
                PortfolioAssetSpec(
                    dataset_id="asset_b",
                    data=self.bars_b,
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 5, "slow": 20, "target": 1.0},
                ),
            ],
            self.config,
            PortfolioConstructionConfig(
                allocation_ownership=ALLOCATION_OWNERSHIP_PORTFOLIO,
                ranking_mode=RANKING_MODE_SCORE_THRESHOLD,
                min_rank_score=None,
            ),
        )
        self.assertFalse(support.supported)
        self.assertIn("min_rank_score", support.reason or "")

    def test_score_threshold_ranking_filters_low_score_assets(self) -> None:
        engine = VectorizedPortfolioEngine()
        strong_bars = _make_trend_bars(periods=240, slope=0.45)
        weak_bars = _make_trend_bars(periods=240, slope=0.0)
        result = engine.run(
            [
                PortfolioAssetSpec(
                    dataset_id="strong_asset",
                    data=strong_bars,
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 5, "slow": 20, "target": 1.0},
                ),
                PortfolioAssetSpec(
                    dataset_id="weak_asset",
                    data=weak_bars,
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 5, "slow": 20, "target": 1.0},
                ),
            ],
            self.config,
            normalize_weights=True,
            construction_config=PortfolioConstructionConfig(
                allocation_ownership=ALLOCATION_OWNERSHIP_PORTFOLIO,
                ranking_mode=RANKING_MODE_SCORE_THRESHOLD,
                min_rank_score=0.001,
            ),
        )

        positive_counts = (result.target_weights > 1e-9).sum(axis=1)
        self.assertGreater(int((positive_counts == 1).sum()), 0)
        self.assertGreater(float(result.target_weights["strong_asset"].sum()), 0.0)
        self.assertEqual(float(result.target_weights["weak_asset"].max()), 0.0)

    def test_support_rejects_top_n_over_threshold_without_required_fields(self) -> None:
        engine = VectorizedPortfolioEngine()
        support = engine.supports(
            [
                PortfolioAssetSpec(
                    dataset_id="asset_a",
                    data=self.bars_a,
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 5, "slow": 20, "target": 1.0},
                ),
                PortfolioAssetSpec(
                    dataset_id="asset_b",
                    data=self.bars_b,
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 5, "slow": 20, "target": 1.0},
                ),
            ],
            self.config,
            PortfolioConstructionConfig(
                allocation_ownership=ALLOCATION_OWNERSHIP_PORTFOLIO,
                ranking_mode=RANKING_MODE_TOP_N_OVER_SCORE_THRESHOLD,
                max_ranked_assets=1,
                min_rank_score=None,
            ),
        )
        self.assertFalse(support.supported)
        self.assertIn("min_rank_score", support.reason or "")

    def test_top_n_over_threshold_ranking_filters_then_caps(self) -> None:
        engine = VectorizedPortfolioEngine()
        result = engine.run(
            [
                PortfolioAssetSpec(
                    dataset_id="strong_asset",
                    data=_make_trend_bars(periods=240, slope=0.45),
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 5, "slow": 20, "target": 1.0},
                ),
                PortfolioAssetSpec(
                    dataset_id="medium_asset",
                    data=_make_trend_bars(periods=240, slope=0.22),
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 5, "slow": 20, "target": 1.0},
                ),
                PortfolioAssetSpec(
                    dataset_id="weak_asset",
                    data=_make_trend_bars(periods=240, slope=0.0),
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 5, "slow": 20, "target": 1.0},
                ),
            ],
            self.config,
            normalize_weights=True,
            construction_config=PortfolioConstructionConfig(
                allocation_ownership=ALLOCATION_OWNERSHIP_PORTFOLIO,
                ranking_mode=RANKING_MODE_TOP_N_OVER_SCORE_THRESHOLD,
                max_ranked_assets=1,
                min_rank_score=0.001,
            ),
        )

        positive_counts = (result.target_weights > 1e-9).sum(axis=1)
        self.assertLessEqual(int(positive_counts.max()), 1)
        self.assertGreater(int((positive_counts == 1).sum()), 0)
        self.assertGreater(float(result.target_weights["strong_asset"].sum()), 0.0)
        self.assertEqual(float(result.target_weights["weak_asset"].max()), 0.0)

    def test_equal_selected_weighting_ignores_base_target_weight_skew(self) -> None:
        engine = VectorizedPortfolioEngine()
        result = engine.run(
            [
                PortfolioAssetSpec(
                    dataset_id="asset_a",
                    data=_make_trend_bars(periods=240, slope=0.2),
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 5, "slow": 20, "target": 1.0},
                    target_weight=0.8,
                ),
                PortfolioAssetSpec(
                    dataset_id="asset_b",
                    data=_make_trend_bars(periods=240, slope=0.18),
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 5, "slow": 20, "target": 1.0},
                    target_weight=0.2,
                ),
            ],
            self.config,
            normalize_weights=True,
            construction_config=PortfolioConstructionConfig(
                allocation_ownership=ALLOCATION_OWNERSHIP_PORTFOLIO,
                weighting_mode=WEIGHTING_MODE_EQUAL_SELECTED,
            ),
        )

        both_active = result.target_weights[(result.target_weights["asset_a"] > 0) & (result.target_weights["asset_b"] > 0)]
        self.assertFalse(both_active.empty)
        self.assertTrue(np.allclose(both_active["asset_a"].to_numpy(), 0.5, atol=1e-6))
        self.assertTrue(np.allclose(both_active["asset_b"].to_numpy(), 0.5, atol=1e-6))

    def test_max_asset_weight_caps_concentration(self) -> None:
        engine = VectorizedPortfolioEngine()
        result = engine.run(
            [
                PortfolioAssetSpec(
                    dataset_id="asset_a",
                    data=_make_trend_bars(periods=240, slope=0.35),
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 5, "slow": 20, "target": 1.0},
                ),
                PortfolioAssetSpec(
                    dataset_id="asset_b",
                    data=_make_trend_bars(periods=240, slope=0.32),
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 5, "slow": 20, "target": 1.0},
                ),
            ],
            self.config,
            normalize_weights=True,
            construction_config=PortfolioConstructionConfig(
                allocation_ownership=ALLOCATION_OWNERSHIP_PORTFOLIO,
                weighting_mode=WEIGHTING_MODE_EQUAL_SELECTED,
                max_asset_weight=0.4,
            ),
        )

        self.assertLessEqual(float(result.target_weights.max().max()), 0.400001)

    def test_cash_reserve_keeps_portion_unallocated(self) -> None:
        engine = VectorizedPortfolioEngine()
        result = engine.run(
            [
                PortfolioAssetSpec(
                    dataset_id="asset_a",
                    data=_make_trend_bars(periods=240, slope=0.35),
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 5, "slow": 20, "target": 1.0},
                ),
                PortfolioAssetSpec(
                    dataset_id="asset_b",
                    data=_make_trend_bars(periods=240, slope=0.32),
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 5, "slow": 20, "target": 1.0},
                ),
            ],
            self.config,
            normalize_weights=True,
            construction_config=PortfolioConstructionConfig(
                allocation_ownership=ALLOCATION_OWNERSHIP_PORTFOLIO,
                weighting_mode=WEIGHTING_MODE_EQUAL_SELECTED,
                cash_reserve_weight=0.2,
            ),
        )

        self.assertLessEqual(float(result.target_weights.sum(axis=1).max()), 0.800001)

    def test_portfolio_snapshot_export_creates_artifact(self) -> None:
        engine = VectorizedPortfolioEngine()
        result = engine.run(
            [
                PortfolioAssetSpec(
                    dataset_id="asset_a",
                    data=_make_trend_bars(periods=120, slope=0.2),
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 5, "slow": 20, "target": 1.0},
                ),
                PortfolioAssetSpec(
                    dataset_id="asset_b",
                    data=_make_trend_bars(periods=120, slope=0.18),
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 5, "slow": 20, "target": 1.0},
                ),
            ],
            self.config,
            normalize_weights=True,
            construction_config=PortfolioConstructionConfig(
                allocation_ownership=ALLOCATION_OWNERSHIP_PORTFOLIO,
                weighting_mode=WEIGHTING_MODE_EQUAL_SELECTED,
                cash_reserve_weight=0.1,
            ),
        )
        run = SimpleNamespace(
            run_id="portfolio_snapshot_run",
            dataset_id="Portfolio | asset_a, asset_b",
            strategy="PortfolioExecution",
            params={"example": True},
            timeframe="1 minutes",
            start=str(result.portfolio_equity_curve.index[0]),
            end=str(result.portfolio_equity_curve.index[-1]),
            starting_cash=self.config.starting_cash,
            metrics=result.metrics.as_dict(),
        )
        with TemporaryDirectory() as tmpdir:
            exporter = ChartSnapshotExporter(tmpdir)
            artifact = exporter.export_portfolio_snapshot(run=run, portfolio_result=result)

            manifest_path = artifact.snapshot_root / "manifest.json"
            self.assertTrue(manifest_path.exists())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(str(manifest["dataset_id"]), "Portfolio | asset_a, asset_b")
            self.assertEqual(int(manifest["counts"]["bars"]), len(result.portfolio_equity_curve))
            self.assertGreaterEqual(int(manifest["counts"]["pane_series"]), 3)
            self.assertTrue((artifact.snapshot_root / "price_bars.feather").exists())
            self.assertTrue((artifact.snapshot_root / "equity.feather").exists())

    def test_score_proportional_weighting_favors_stronger_scores(self) -> None:
        engine = VectorizedPortfolioEngine()
        result = engine.run(
            [
                PortfolioAssetSpec(
                    dataset_id="strong_asset",
                    data=_make_trend_bars(periods=240, slope=0.45),
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 5, "slow": 20, "target": 1.0},
                ),
                PortfolioAssetSpec(
                    dataset_id="weak_asset",
                    data=_make_trend_bars(periods=240, slope=0.08),
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 5, "slow": 20, "target": 1.0},
                ),
            ],
            self.config,
            normalize_weights=True,
            construction_config=PortfolioConstructionConfig(
                allocation_ownership=ALLOCATION_OWNERSHIP_PORTFOLIO,
                weighting_mode=WEIGHTING_MODE_SCORE_PROPORTIONAL,
            ),
        )

        both_active = result.target_weights[(result.target_weights["strong_asset"] > 0) & (result.target_weights["weak_asset"] > 0)]
        self.assertFalse(both_active.empty)
        self.assertGreater(
            float((both_active["strong_asset"] - both_active["weak_asset"]).mean()),
            0.0,
        )

    def test_min_active_weight_prunes_tiny_allocations(self) -> None:
        engine = VectorizedPortfolioEngine()
        result = engine.run(
            [
                PortfolioAssetSpec(
                    dataset_id="strong_asset",
                    data=_make_trend_bars(periods=240, slope=0.45),
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 5, "slow": 20, "target": 1.0},
                ),
                PortfolioAssetSpec(
                    dataset_id="medium_asset",
                    data=_make_trend_bars(periods=240, slope=0.18),
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 5, "slow": 20, "target": 1.0},
                ),
                PortfolioAssetSpec(
                    dataset_id="weak_asset",
                    data=_make_trend_bars(periods=240, slope=0.01),
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 5, "slow": 20, "target": 1.0},
                ),
            ],
            self.config,
            normalize_weights=True,
            construction_config=PortfolioConstructionConfig(
                allocation_ownership=ALLOCATION_OWNERSHIP_PORTFOLIO,
                weighting_mode=WEIGHTING_MODE_SCORE_PROPORTIONAL,
                min_active_weight=0.2,
            ),
        )

        positive = result.target_weights.to_numpy()
        active = positive[positive > 1e-9]
        self.assertTrue(active.size > 0)
        self.assertGreaterEqual(float(active.min()), 0.2 - 1e-6)

    def test_periodic_rebalance_increases_trade_count(self) -> None:
        engine = VectorizedPortfolioEngine()
        on_change = engine.run(
            [
                PortfolioAssetSpec(
                    dataset_id="asset_a",
                    data=self.bars_a,
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 5, "slow": 20, "target": 1.0},
                ),
                PortfolioAssetSpec(
                    dataset_id="asset_b",
                    data=self.bars_b,
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 5, "slow": 20, "target": 1.0},
                ),
            ],
            replace(self.config, fill_on_close=True),
            normalize_weights=True,
            construction_config=PortfolioConstructionConfig(
                allocation_ownership=ALLOCATION_OWNERSHIP_PORTFOLIO,
                rebalance_mode=REBALANCE_MODE_ON_CHANGE,
            ),
        )
        periodic = engine.run(
            [
                PortfolioAssetSpec(
                    dataset_id="asset_a",
                    data=self.bars_a,
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 5, "slow": 20, "target": 1.0},
                ),
                PortfolioAssetSpec(
                    dataset_id="asset_b",
                    data=self.bars_b,
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 5, "slow": 20, "target": 1.0},
                ),
            ],
            replace(self.config, fill_on_close=True),
            normalize_weights=True,
            construction_config=PortfolioConstructionConfig(
                allocation_ownership=ALLOCATION_OWNERSHIP_PORTFOLIO,
                rebalance_mode=REBALANCE_MODE_ON_CHANGE_OR_PERIODIC,
                rebalance_every_n_bars=20,
            ),
        )
        self.assertGreater(len(periodic.trades), len(on_change.trades))

    def test_support_rejects_drift_rebalance_without_positive_threshold(self) -> None:
        engine = VectorizedPortfolioEngine()
        support = engine.supports(
            [
                PortfolioAssetSpec(
                    dataset_id="asset_a",
                    data=self.bars_a,
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 5, "slow": 20, "target": 1.0},
                )
            ],
            self.config,
            PortfolioConstructionConfig(
                allocation_ownership=ALLOCATION_OWNERSHIP_PORTFOLIO,
                rebalance_mode=REBALANCE_MODE_ON_CHANGE_OR_DRIFT,
                rebalance_weight_drift_threshold=0.0,
            ),
        )
        self.assertFalse(support.supported)
        self.assertIn("rebalance_weight_drift_threshold", support.reason or "")

    def test_drift_rebalance_increases_trade_count(self) -> None:
        engine = VectorizedPortfolioEngine()
        fast_trend = _make_trend_bars(periods=260, slope=0.45)
        slow_trend = _make_trend_bars(periods=260, slope=0.05)
        on_change = engine.run(
            [
                PortfolioAssetSpec(
                    dataset_id="fast_asset",
                    data=fast_trend,
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 3, "slow": 12, "target": 1.0},
                ),
                PortfolioAssetSpec(
                    dataset_id="slow_asset",
                    data=slow_trend,
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 3, "slow": 12, "target": 1.0},
                ),
            ],
            replace(self.config, fill_on_close=True),
            normalize_weights=True,
            construction_config=PortfolioConstructionConfig(
                allocation_ownership=ALLOCATION_OWNERSHIP_PORTFOLIO,
                rebalance_mode=REBALANCE_MODE_ON_CHANGE,
            ),
        )
        drift = engine.run(
            [
                PortfolioAssetSpec(
                    dataset_id="fast_asset",
                    data=fast_trend,
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 3, "slow": 12, "target": 1.0},
                ),
                PortfolioAssetSpec(
                    dataset_id="slow_asset",
                    data=slow_trend,
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 3, "slow": 12, "target": 1.0},
                ),
            ],
            replace(self.config, fill_on_close=True),
            normalize_weights=True,
            construction_config=PortfolioConstructionConfig(
                allocation_ownership=ALLOCATION_OWNERSHIP_PORTFOLIO,
                rebalance_mode=REBALANCE_MODE_ON_CHANGE_OR_DRIFT,
                rebalance_weight_drift_threshold=0.015,
            ),
        )

        self.assertGreater(len(drift.trades), len(on_change.trades))

    def test_periodic_and_drift_rebalance_increases_trade_count(self) -> None:
        engine = VectorizedPortfolioEngine()
        fast_trend = _make_trend_bars(periods=260, slope=0.45)
        slow_trend = _make_trend_bars(periods=260, slope=0.05)
        on_change = engine.run(
            [
                PortfolioAssetSpec(
                    dataset_id="fast_asset",
                    data=fast_trend,
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 3, "slow": 12, "target": 1.0},
                ),
                PortfolioAssetSpec(
                    dataset_id="slow_asset",
                    data=slow_trend,
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 3, "slow": 12, "target": 1.0},
                ),
            ],
            replace(self.config, fill_on_close=True),
            normalize_weights=True,
            construction_config=PortfolioConstructionConfig(
                allocation_ownership=ALLOCATION_OWNERSHIP_PORTFOLIO,
                rebalance_mode=REBALANCE_MODE_ON_CHANGE,
            ),
        )
        combined = engine.run(
            [
                PortfolioAssetSpec(
                    dataset_id="fast_asset",
                    data=fast_trend,
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 3, "slow": 12, "target": 1.0},
                ),
                PortfolioAssetSpec(
                    dataset_id="slow_asset",
                    data=slow_trend,
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 3, "slow": 12, "target": 1.0},
                ),
            ],
            replace(self.config, fill_on_close=True),
            normalize_weights=True,
            construction_config=PortfolioConstructionConfig(
                allocation_ownership=ALLOCATION_OWNERSHIP_PORTFOLIO,
                rebalance_mode=REBALANCE_MODE_ON_CHANGE_OR_PERIODIC_OR_DRIFT,
                rebalance_every_n_bars=20,
                rebalance_weight_drift_threshold=0.015,
            ),
        )

        self.assertGreater(len(combined.trades), len(on_change.trades))

    def test_portfolio_grid_search_saves_catalog_runs_and_benchmarks(self) -> None:
        grid = GridSpec(
            params={"fast": [5, 8], "slow": [20], "target": [1.0]},
            timeframes=["1 minutes"],
            horizons=[(None, None)],
            execution_mode=ExecutionMode.AUTO,
            batch_id="portfolio_batch",
        )
        with TemporaryDirectory() as tmpdir:
            catalog = ResultCatalog(Path(tmpdir) / "portfolio.sqlite")
            frame = run_vectorized_portfolio_grid_search(
                targets=[
                    PortfolioAssetTarget(dataset_id="asset_a", data_loader=lambda _tf: self.bars_a),
                    PortfolioAssetTarget(dataset_id="asset_b", data_loader=lambda _tf: self.bars_b),
                ],
                strategy_cls=SMACrossStrategy,
                base_config=self.config,
                grid=grid,
                catalog=catalog,
            )

            self.assertEqual(len(frame), 2)
            self.assertTrue((frame["engine_impl"] == "vectorized_portfolio").all())
            self.assertEqual(frame["resolved_execution_mode"].unique().tolist(), [ExecutionMode.VECTORIZED.value])
            benchmarks = frame.attrs.get("batch_benchmarks") or ()
            self.assertEqual(len(benchmarks), 1)
            self.assertEqual(benchmarks[0].engine_impl, "vectorized_portfolio")

            runs = catalog.load_runs_for_logical_run_id(frame.iloc[0]["logical_run_id"])
            self.assertTrue(runs)
            self.assertEqual(runs[0].engine_impl, "vectorized_portfolio")
            self.assertTrue(str(runs[0].dataset_id).startswith("Portfolio | "))

    def test_orchestrator_executes_portfolio_request(self) -> None:
        orchestrator = ExecutionOrchestrator()
        result = orchestrator.execute_portfolio(
            PortfolioExecutionRequest(
                assets=[
                    PortfolioExecutionAsset(
                        dataset_id="asset_a",
                        data=self.bars_a,
                        strategy_cls=SMACrossStrategy,
                        strategy_params={"fast": 5, "slow": 20, "target": 1.0},
                    ),
                    PortfolioExecutionAsset(
                        dataset_id="asset_b",
                        data=self.bars_b,
                        strategy_cls=SMACrossStrategy,
                        strategy_params={"fast": 5, "slow": 20, "target": 1.0},
                    ),
                ],
                config=self.config,
                requested_execution_mode=ExecutionMode.AUTO,
                normalize_weights=True,
                construction_config=PortfolioConstructionConfig(
                    allocation_ownership=ALLOCATION_OWNERSHIP_PORTFOLIO,
                    ranking_mode=RANKING_MODE_TOP_N,
                    max_ranked_assets=1,
                ),
            )
        )

        self.assertEqual(result.engine_impl, "vectorized_portfolio")
        self.assertEqual(result.resolved_execution_mode, ExecutionMode.VECTORIZED)
        self.assertTrue(str(result.dataset_id).startswith("Portfolio | "))
        self.assertEqual(result.result.asset_weights.columns.tolist(), ["asset_a", "asset_b"])

    def test_orchestrator_rejects_reference_portfolio_request(self) -> None:
        orchestrator = ExecutionOrchestrator()
        with self.assertRaises(UnsupportedExecutionModeError):
            orchestrator.execute_portfolio(
                PortfolioExecutionRequest(
                    assets=[
                        PortfolioExecutionAsset(
                            dataset_id="asset_a",
                            data=self.bars_a,
                            strategy_cls=SMACrossStrategy,
                            strategy_params={"fast": 5, "slow": 20, "target": 1.0},
                        )
                    ],
                    config=self.config,
                    requested_execution_mode=ExecutionMode.REFERENCE,
                )
            )

    def test_portfolio_grid_search_supports_fixed_weights(self) -> None:
        grid = GridSpec(
            params={"fast": [5], "slow": [20], "target": [1.0]},
            timeframes=["1 minutes"],
            horizons=[(None, None)],
            execution_mode=ExecutionMode.VECTORIZED,
            batch_id="portfolio_fixed",
        )
        config = replace(self.config, fill_on_close=True)
        expected = VectorizedPortfolioEngine().run(
            [
                PortfolioAssetSpec(
                    dataset_id="asset_a",
                    data=self.bars_a,
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 5, "slow": 20, "target": 1.0},
                    target_weight=0.8,
                ),
                PortfolioAssetSpec(
                    dataset_id="asset_b",
                    data=self.bars_b,
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 5, "slow": 20, "target": 1.0},
                    target_weight=0.2,
                ),
            ],
            config,
            normalize_weights=False,
        )
        frame = run_vectorized_portfolio_grid_search(
            targets=[
                PortfolioAssetTarget(dataset_id="asset_a", data_loader=lambda _tf: self.bars_a, target_weight=0.8),
                PortfolioAssetTarget(dataset_id="asset_b", data_loader=lambda _tf: self.bars_b, target_weight=0.2),
            ],
            strategy_cls=SMACrossStrategy,
            base_config=config,
            grid=grid,
            catalog=None,
            normalize_weights=False,
        )

        self.assertEqual(len(frame), 1)
        self.assertAlmostEqual(frame.iloc[0]["total_return"], expected.metrics.total_return, places=10)
        benchmarks = frame.attrs.get("batch_benchmarks") or ()
        self.assertEqual(len(benchmarks), 1)
        self.assertEqual(benchmarks[0].engine_impl, "vectorized_portfolio")


if __name__ == "__main__":
    unittest.main()
