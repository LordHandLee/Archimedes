from __future__ import annotations

import unittest

import pandas as pd

from backtest_engine.engine import BacktestConfig
from backtest_engine.portfolio_reporting import (
    build_portfolio_chart_data,
    portfolio_report_frame,
    summarize_portfolio_result,
)
from backtest_engine.sample_strategies import SMACrossStrategy
from backtest_engine.vectorized_portfolio import (
    ALLOCATION_OWNERSHIP_PORTFOLIO,
    PortfolioAssetSpec,
    PortfolioConstructionConfig,
    VectorizedPortfolioEngine,
    WEIGHTING_MODE_EQUAL_SELECTED,
)
from tests.test_vectorized_portfolio import _make_trend_bars


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

        frame = portfolio_report_frame(report)
        self.assertEqual(frame.shape[0], 2)
        self.assertIn("dataset_id", frame.columns)
        self.assertIn("avg_weight", frame.columns)
        self.assertIn("turnover_ratio", frame.columns)

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


if __name__ == "__main__":
    unittest.main()
