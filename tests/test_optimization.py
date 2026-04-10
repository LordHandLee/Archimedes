from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from backtest_engine.catalog import ResultCatalog
from backtest_engine.optimization import (
    ROBUST_SCORE_VERSION,
    build_asset_distribution_frame,
    build_optimization_study_artifacts,
    compute_robust_score,
)


class OptimizationStudyTest(unittest.TestCase):
    def _sample_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "dataset_id": "asset_a",
                    "timeframe": "5 minutes",
                    "start": "2024-01-01T00:00:00+00:00",
                    "end": "2024-01-31T00:00:00+00:00",
                    "fast": 5,
                    "slow": 20,
                    "total_return": 0.12,
                    "cagr": 0.12,
                    "max_drawdown": -0.09,
                    "sharpe": 1.30,
                    "rolling_sharpe": 1.12,
                    "run_id": "run_a_1",
                },
                {
                    "dataset_id": "asset_b",
                    "timeframe": "5 minutes",
                    "start": "2024-01-01T00:00:00+00:00",
                    "end": "2024-01-31T00:00:00+00:00",
                    "fast": 5,
                    "slow": 20,
                    "total_return": 0.08,
                    "cagr": 0.08,
                    "max_drawdown": -0.07,
                    "sharpe": 1.05,
                    "rolling_sharpe": 0.98,
                    "run_id": "run_b_1",
                },
                {
                    "dataset_id": "asset_a",
                    "timeframe": "5 minutes",
                    "start": "2024-01-01T00:00:00+00:00",
                    "end": "2024-01-31T00:00:00+00:00",
                    "fast": 10,
                    "slow": 30,
                    "total_return": 0.03,
                    "cagr": 0.03,
                    "max_drawdown": -0.14,
                    "sharpe": 0.55,
                    "rolling_sharpe": 0.42,
                    "run_id": "run_a_2",
                },
                {
                    "dataset_id": "asset_b",
                    "timeframe": "5 minutes",
                    "start": "2024-01-01T00:00:00+00:00",
                    "end": "2024-01-31T00:00:00+00:00",
                    "fast": 10,
                    "slow": 30,
                    "total_return": -0.01,
                    "cagr": -0.01,
                    "max_drawdown": -0.16,
                    "sharpe": 0.20,
                    "rolling_sharpe": 0.15,
                    "run_id": "run_b_2",
                },
            ]
        )

    def test_build_optimization_study_artifacts_ranks_by_robust_score(self) -> None:
        frame = self._sample_frame()
        artifacts = build_optimization_study_artifacts(
            df=frame,
            study_id="study_1",
            batch_id="batch_1",
            strategy="SMACrossStrategy",
            dataset_scope=["asset_a", "asset_b"],
            param_names=["fast", "slow"],
            timeframes=["5 minutes"],
            horizons=["30d"],
        )

        self.assertEqual(artifacts.score_version, ROBUST_SCORE_VERSION)
        self.assertEqual(len(artifacts.aggregates), 2)
        best = artifacts.aggregates.iloc[0]
        self.assertEqual(int(best["fast"]), 5)
        self.assertEqual(int(best["slow"]), 20)
        expected = compute_robust_score(
            median_sharpe=1.175,
            sharpe_std=0.125,
            worst_max_drawdown=0.09,
            profitable_asset_ratio=1.0,
        )
        self.assertAlmostEqual(float(best["robust_score"]), expected, places=9)

    def test_asset_distribution_frame_filters_selected_candidate(self) -> None:
        artifacts = build_optimization_study_artifacts(
            df=self._sample_frame(),
            study_id="study_1",
            batch_id="batch_1",
            strategy="SMACrossStrategy",
            dataset_scope=["asset_a", "asset_b"],
            param_names=["fast", "slow"],
            timeframes=["5 minutes"],
            horizons=["30d"],
        )
        selected = artifacts.aggregates.iloc[0]
        dist = build_asset_distribution_frame(
            artifacts.asset_results,
            param_key=str(selected["param_key"]),
            timeframe=str(selected["timeframe"]),
            start=str(selected["start"]),
            end=str(selected["end"]),
        )
        self.assertEqual(list(dist["dataset_id"]), ["asset_a", "asset_b"])
        self.assertIn("run_id", dist.columns)

    def test_catalog_persists_optimization_study(self) -> None:
        artifacts = build_optimization_study_artifacts(
            df=self._sample_frame(),
            study_id="study_1",
            batch_id="batch_1",
            strategy="SMACrossStrategy",
            dataset_scope=["asset_a", "asset_b"],
            param_names=["fast", "slow"],
            timeframes=["5 minutes"],
            horizons=["30d"],
        )
        with TemporaryDirectory() as tmpdir:
            catalog = ResultCatalog(Path(tmpdir) / "backtests.sqlite")
            catalog.save_optimization_study(
                study_id=artifacts.study_id,
                batch_id=artifacts.batch_id,
                strategy=artifacts.strategy,
                dataset_scope=artifacts.dataset_scope,
                param_names=artifacts.param_names,
                timeframes=artifacts.timeframes,
                horizons=artifacts.horizons,
                score_version=artifacts.score_version,
                aggregates=artifacts.aggregates,
                asset_results=artifacts.asset_results,
            )
            studies = catalog.load_optimization_studies()
            self.assertEqual(len(studies), 1)
            self.assertEqual(studies[0].study_id, "study_1")
            aggregates = catalog.load_optimization_aggregates("study_1")
            assets = catalog.load_optimization_asset_results("study_1")
            self.assertEqual(len(aggregates), 2)
            self.assertEqual(len(assets), 4)

    def test_catalog_persists_optimization_candidates(self) -> None:
        artifacts = build_optimization_study_artifacts(
            df=self._sample_frame(),
            study_id="study_1",
            batch_id="batch_1",
            strategy="SMACrossStrategy",
            dataset_scope=["asset_a", "asset_b"],
            param_names=["fast", "slow"],
            timeframes=["5 minutes"],
            horizons=["30d"],
        )
        selected = artifacts.aggregates.iloc[0]
        dist = build_asset_distribution_frame(
            artifacts.asset_results,
            param_key=str(selected["param_key"]),
            timeframe=str(selected["timeframe"]),
            start=str(selected["start"]),
            end=str(selected["end"]),
        )
        with TemporaryDirectory() as tmpdir:
            catalog = ResultCatalog(Path(tmpdir) / "backtests.sqlite")
            catalog.save_optimization_candidate(
                study_id="study_1",
                timeframe=str(selected["timeframe"]),
                start=str(selected["start"]),
                end=str(selected["end"]),
                param_key=str(selected["param_key"]),
                params_json=str(selected["params_json"]),
                source_type="manual_selection",
                promotion_reason="Broad plateau",
                status="queued",
                metrics={
                    "robust_score": float(selected["robust_score"]),
                    "median_sharpe": float(selected["median_sharpe"]),
                },
                asset_results=dist,
                artifact_refs={"heatmap_metric": "robust_score"},
                notes="Looks stable across both assets.",
            )
            rows = catalog.load_optimization_candidates("study_1")
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row.study_id, "study_1")
            self.assertEqual(row.status, "queued")
            self.assertEqual(row.promotion_reason, "Broad plateau")
            self.assertIn("robust_score", row.metrics_json)
            self.assertIn("asset_a", row.asset_results_json or "")


if __name__ == "__main__":
    unittest.main()
