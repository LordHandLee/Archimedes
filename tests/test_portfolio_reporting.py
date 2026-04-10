from __future__ import annotations

import unittest
from types import SimpleNamespace

import pandas as pd

from backtest_engine.engine import BacktestConfig
from backtest_engine.portfolio_reporting import (
    build_portfolio_chart_data,
    build_portfolio_trades_log_frame,
    portfolio_drawdown_frame,
    portfolio_report_frame,
    portfolio_strategy_report_frame,
    summarize_portfolio_result,
)
from backtest_engine.sample_strategies import SMACrossStrategy
from backtest_engine.vectorized_portfolio import (
    ALLOCATION_OWNERSHIP_PORTFOLIO,
    PortfolioAssetSpec,
    PortfolioConstructionConfig,
    PortfolioStrategyBlockAssetSpec,
    PortfolioStrategyBlockSpec,
    VectorizedPortfolioEngine,
    WEIGHTING_MODE_EQUAL_SELECTED,
)
from tests.test_vectorized_portfolio import _make_bars, _make_trend_bars


class PortfolioReportingTest(unittest.TestCase):
    def test_summarize_portfolio_result_builds_asset_attribution(self) -> None:
        config = BacktestConfig(
            timeframe="1 minutes",
            use_cache=False,
            base_execution=False,
            allow_short=False,
            fee_rate=0.0002,
            slippage=0.0001,
            starting_cash=100_000.0,
        )
        result = VectorizedPortfolioEngine().run(
            [
                PortfolioAssetSpec(
                    dataset_id="asset_a",
                    data=_make_trend_bars(periods=180, slope=0.35),
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 5, "slow": 20, "target": 1.0},
                ),
                PortfolioAssetSpec(
                    dataset_id="asset_b",
                    data=_make_trend_bars(periods=180, slope=0.22),
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 5, "slow": 20, "target": 1.0},
                ),
            ],
            config,
            normalize_weights=True,
            construction_config=PortfolioConstructionConfig(
                allocation_ownership=ALLOCATION_OWNERSHIP_PORTFOLIO,
                weighting_mode=WEIGHTING_MODE_EQUAL_SELECTED,
                cash_reserve_weight=0.1,
            ),
        )

        report = summarize_portfolio_result(result, starting_cash=config.starting_cash)
        self.assertEqual(len(report.asset_rows), 2)
        self.assertGreater(report.ending_equity, 0.0)
        self.assertGreaterEqual(report.avg_cash_weight, 0.0)
        self.assertLessEqual(report.avg_target_gross_exposure, 0.900001)
        self.assertGreater(report.total_turnover_notional, 0.0)
        self.assertGreaterEqual(report.annualized_volatility, 0.0)
        self.assertGreaterEqual(report.avg_active_assets, 0.0)
        self.assertGreaterEqual(report.peak_active_assets, 0)
        self.assertGreaterEqual(report.peak_single_name_weight, 0.0)
        self.assertEqual(len(report.strategy_rows), 1)

        frame = portfolio_report_frame(report)
        self.assertEqual(frame.shape[0], 2)
        self.assertIn("dataset_id", frame.columns)
        self.assertIn("avg_weight", frame.columns)
        self.assertIn("turnover_ratio", frame.columns)
        self.assertIn("total_return_contribution", frame.columns)
        self.assertIn("contribution_share", frame.columns)
        strategy_frame = portfolio_strategy_report_frame(report)
        self.assertEqual(strategy_frame.shape[0], 1)
        self.assertIn("strategy_block_id", strategy_frame.columns)
        self.assertIn("budget_weight", strategy_frame.columns)

        drawdowns = portfolio_drawdown_frame(report)
        self.assertIn("depth", drawdowns.columns)
        if not drawdowns.empty:
            self.assertLessEqual(float(drawdowns["depth"].min()), 0.0)

    def test_build_portfolio_chart_data_selects_top_assets_and_trade_frame(self) -> None:
        config = BacktestConfig(
            timeframe="1 minutes",
            use_cache=False,
            base_execution=False,
            allow_short=False,
            fee_rate=0.0002,
            slippage=0.0001,
            starting_cash=100_000.0,
        )
        result = VectorizedPortfolioEngine().run(
            [
                PortfolioAssetSpec(
                    dataset_id="asset_a",
                    data=_make_trend_bars(periods=180, slope=0.45),
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 5, "slow": 20, "target": 1.0},
                ),
                PortfolioAssetSpec(
                    dataset_id="asset_b",
                    data=_make_trend_bars(periods=180, slope=0.28),
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 5, "slow": 20, "target": 1.0},
                ),
            ],
            config,
            normalize_weights=True,
            construction_config=PortfolioConstructionConfig(
                allocation_ownership=ALLOCATION_OWNERSHIP_PORTFOLIO,
                weighting_mode=WEIGHTING_MODE_EQUAL_SELECTED,
            ),
        )

        chart_data = build_portfolio_chart_data(result, max_assets=1)
        self.assertEqual(len(chart_data.top_assets), 1)
        self.assertGreater(len(chart_data.equity_curve), 0)
        self.assertEqual(chart_data.asset_weights.shape[1], 2)
        self.assertEqual(chart_data.target_weights.shape[1], 2)
        self.assertIsInstance(chart_data.trades, pd.DataFrame)
        self.assertIn("timestamp", chart_data.trades.columns)
        self.assertIn("equity_after", chart_data.trades.columns)
        self.assertIn("source_dataset_id", chart_data.trades.columns)
        self.assertIn("strategy_block_id", chart_data.trades.columns)

    def test_asset_contribution_share_is_bounded(self) -> None:
        config = BacktestConfig(
            timeframe="1 minutes",
            use_cache=False,
            base_execution=False,
            allow_short=False,
            fee_rate=0.0002,
            slippage=0.0001,
            starting_cash=100_000.0,
        )
        result = VectorizedPortfolioEngine().run(
            [
                PortfolioAssetSpec(
                    dataset_id="asset_a",
                    data=_make_trend_bars(periods=140, slope=0.40),
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 5, "slow": 20, "target": 1.0},
                ),
                PortfolioAssetSpec(
                    dataset_id="asset_b",
                    data=_make_trend_bars(periods=140, slope=0.12),
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 5, "slow": 20, "target": 1.0},
                ),
            ],
            config,
            normalize_weights=True,
            construction_config=PortfolioConstructionConfig(
                allocation_ownership=ALLOCATION_OWNERSHIP_PORTFOLIO,
                weighting_mode=WEIGHTING_MODE_EQUAL_SELECTED,
            ),
        )

        report = summarize_portfolio_result(result, starting_cash=config.starting_cash)
        frame = portfolio_report_frame(report)
        self.assertTrue(((frame["contribution_share"] >= -1.0) & (frame["contribution_share"] <= 1.0)).all())
        self.assertTrue((frame["avg_abs_weight_change"] >= 0.0).all())

    def test_drawdown_frame_populates_when_portfolio_has_underwater_periods(self) -> None:
        config = BacktestConfig(
            timeframe="1 minutes",
            use_cache=False,
            base_execution=False,
            allow_short=False,
            fee_rate=0.0002,
            slippage=0.0001,
            starting_cash=100_000.0,
        )
        result = VectorizedPortfolioEngine().run(
            [
                PortfolioAssetSpec(
                    dataset_id="asset_a",
                    data=_make_bars(periods=220, drift=6.0, phase=0.0),
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 5, "slow": 20, "target": 1.0},
                ),
                PortfolioAssetSpec(
                    dataset_id="asset_b",
                    data=_make_bars(periods=220, drift=3.0, phase=1.7),
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 5, "slow": 20, "target": 1.0},
                ),
            ],
            config,
            normalize_weights=True,
            construction_config=PortfolioConstructionConfig(
                allocation_ownership=ALLOCATION_OWNERSHIP_PORTFOLIO,
                weighting_mode=WEIGHTING_MODE_EQUAL_SELECTED,
            ),
        )
        report = summarize_portfolio_result(result, starting_cash=config.starting_cash)
        drawdowns = portfolio_drawdown_frame(report)
        self.assertIn("depth", drawdowns.columns)
        self.assertFalse(drawdowns.empty)
        self.assertLess(float(drawdowns.iloc[0]["depth"]), 0.0)

    def test_strategy_report_frame_builds_multiple_strategy_rows(self) -> None:
        config = BacktestConfig(
            timeframe="1 minutes",
            use_cache=False,
            base_execution=False,
            allow_short=False,
            fee_rate=0.0002,
            slippage=0.0001,
            starting_cash=100_000.0,
        )
        result = VectorizedPortfolioEngine().run_strategy_blocks(
            [
                PortfolioStrategyBlockSpec(
                    block_id="trend",
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 5, "slow": 20, "target": 1.0},
                    assets=[
                        PortfolioStrategyBlockAssetSpec(dataset_id="asset_a", data=_make_trend_bars(periods=150, slope=0.30)),
                    ],
                    budget_weight=0.55,
                    display_name="Trend",
                ),
                PortfolioStrategyBlockSpec(
                    block_id="trend_slow",
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 8, "slow": 30, "target": 1.0},
                    assets=[
                        PortfolioStrategyBlockAssetSpec(dataset_id="asset_b", data=_make_trend_bars(periods=150, slope=0.18)),
                    ],
                    budget_weight=0.25,
                    display_name="Trend Slow",
                ),
            ],
            config,
            normalize_weights=False,
        )

        report = summarize_portfolio_result(result, starting_cash=config.starting_cash)
        strategy_frame = portfolio_strategy_report_frame(report)
        self.assertEqual(strategy_frame.shape[0], 2)
        self.assertIn("strategy_name", strategy_frame.columns)
        self.assertIn("asset_count", strategy_frame.columns)
        self.assertTrue((strategy_frame["budget_weight"] > 0.0).all())

    def test_summarize_portfolio_result_tracks_short_exposure_metrics(self) -> None:
        config = BacktestConfig(
            timeframe="1 minutes",
            use_cache=False,
            base_execution=False,
            allow_short=True,
            fee_rate=0.0002,
            slippage=0.0001,
            starting_cash=100_000.0,
        )
        downtrend = _make_trend_bars(periods=220, slope=-0.22, start=140.0)
        result = VectorizedPortfolioEngine().run(
            [
                PortfolioAssetSpec(
                    dataset_id="short_asset",
                    data=downtrend,
                    strategy_cls=SMACrossStrategy,
                    strategy_params={"fast": 5, "slow": 20, "target": 1.0},
                ),
            ],
            config,
            normalize_weights=True,
        )

        report = summarize_portfolio_result(result, starting_cash=config.starting_cash)
        self.assertGreater(report.avg_short_exposure, 0.0)
        self.assertGreater(report.peak_short_exposure, 0.0)

        asset_frame = portfolio_report_frame(report)
        self.assertIn("avg_short_weight", asset_frame.columns)
        self.assertIn("peak_short_weight", asset_frame.columns)
        self.assertGreater(float(asset_frame.iloc[0]["avg_short_weight"]), 0.0)

        strategy_frame = portfolio_strategy_report_frame(report)
        self.assertIn("avg_short_weight", strategy_frame.columns)
        self.assertIn("peak_short_weight", strategy_frame.columns)

    def test_summarize_portfolio_result_reconstructs_realized_and_unrealized_pnl(self) -> None:
        index = pd.date_range("2024-01-02 14:30", periods=3, freq="1D", tz="UTC")
        equity = pd.Series([999.0, 1018.0, 1036.0], index=index, name="portfolio_equity")
        cash = pd.Series([899.0, 946.0, 946.0], index=index, name="cash")
        asset_market_values = pd.DataFrame({"asset_a": [100.0, 72.0, 90.0]}, index=index)
        positions = pd.DataFrame({"asset_a": [10.0, 6.0, 6.0]}, index=index)
        asset_weights = asset_market_values.div(equity, axis=0)
        strategy_weights = pd.DataFrame({"default": asset_weights["asset_a"]}, index=index)
        target_weights = asset_weights.copy()
        strategy_target_weights = strategy_weights.copy()

        result = SimpleNamespace(
            portfolio_equity_curve=equity,
            cash_curve=cash,
            asset_market_values=asset_market_values,
            asset_weights=asset_weights,
            target_weights=target_weights,
            strategy_market_values=pd.DataFrame({"default": asset_market_values["asset_a"]}, index=index),
            strategy_weights=strategy_weights,
            strategy_target_weights=strategy_target_weights,
            positions=positions,
            trades=[
                SimpleNamespace(
                    dataset_id="asset_a",
                    timestamp=index[0],
                    side="buy",
                    qty=10.0,
                    price=10.0,
                    fee=1.0,
                    realized_pnl=-999.0,
                ),
                SimpleNamespace(
                    dataset_id="asset_a",
                    timestamp=index[1],
                    side="sell",
                    qty=-4.0,
                    price=12.0,
                    fee=1.0,
                    realized_pnl=-5000.0,
                ),
            ],
            metrics=None,
            asset_to_strategy_block={"asset_a": "default"},
            asset_source_dataset_ids={"asset_a": "asset_a"},
            strategy_display_names={"default": "Default"},
            strategy_budget_weights={"default": 1.0},
        )

        report = summarize_portfolio_result(result, starting_cash=1000.0)
        asset_row = report.asset_rows[0]
        strategy_row = report.strategy_rows[0]

        self.assertAlmostEqual(asset_row.realized_pnl, 6.0, places=8)
        self.assertAlmostEqual(asset_row.unrealized_pnl, 30.0, places=8)
        self.assertAlmostEqual(strategy_row.realized_pnl, 6.0, places=8)
        self.assertAlmostEqual(report.ending_equity - report.starting_equity, 36.0, places=8)
        self.assertAlmostEqual(asset_row.realized_pnl + asset_row.unrealized_pnl, 36.0, places=8)

    def test_build_portfolio_trades_log_frame_keeps_asset_identity_and_asset_level_pnl(self) -> None:
        trades = [
            SimpleNamespace(
                dataset_id="Block A | asset_a",
                source_dataset_id="asset_a",
                strategy_block_id="block_a",
                timestamp="2024-01-02T14:30:00+00:00",
                side="buy",
                qty=10.0,
                price=10.0,
                realized_pnl=0.0,
            ),
            SimpleNamespace(
                dataset_id="Block B | asset_b",
                source_dataset_id="asset_b",
                strategy_block_id="block_b",
                timestamp="2024-01-02T14:31:00+00:00",
                side="buy",
                qty=5.0,
                price=20.0,
                realized_pnl=0.0,
            ),
            SimpleNamespace(
                dataset_id="Block A | asset_a",
                source_dataset_id="asset_a",
                strategy_block_id="block_a",
                timestamp="2024-01-02T14:32:00+00:00",
                side="sell",
                qty=-10.0,
                price=12.0,
                realized_pnl=20.0,
            ),
            SimpleNamespace(
                dataset_id="Block B | asset_b",
                source_dataset_id="asset_b",
                strategy_block_id="block_b",
                timestamp="2024-01-02T14:33:00+00:00",
                side="sell",
                qty=-5.0,
                price=19.0,
                realized_pnl=-5.0,
            ),
        ]

        frame = build_portfolio_trades_log_frame(SimpleNamespace(trades=trades))

        self.assertIn("asset", frame.columns)
        self.assertIn("strategy_block", frame.columns)
        self.assertEqual(frame.iloc[0]["asset"], "asset_a")
        self.assertEqual(frame.iloc[1]["asset"], "asset_b")
        self.assertAlmostEqual(float(frame.iloc[2]["net_pnl"]), 20.0, places=8)
        self.assertAlmostEqual(float(frame.iloc[3]["net_pnl"]), -5.0, places=8)


if __name__ == "__main__":
    unittest.main()
