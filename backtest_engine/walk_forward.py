from __future__ import annotations

import json
import itertools
from dataclasses import dataclass
from dataclasses import replace
from typing import Any
from typing import Callable
from typing import Dict
from typing import Sequence
from typing import Type

import numpy as np
import pandas as pd

from .catalog import OptimizationCandidateRecord
from .catalog import ResultCatalog
from .engine import BacktestConfig
from .execution import ExecutionMode
from .execution import ExecutionOrchestrator
from .execution import PortfolioExecutionAsset
from .execution import PortfolioExecutionRequest
from .execution import PortfolioExecutionStrategyBlock
from .execution import PortfolioExecutionStrategyBlockAsset
from .execution import ExecutionRequest
from .execution import WorkloadType
from .metrics import compute_metrics
from .optimization import ROBUST_SCORE_VERSION
from .optimization import build_optimization_study_artifacts
from .optimization import compute_robust_score
from .strategy import Strategy
from .vectorized_portfolio import PortfolioConstructionConfig

WALK_FORWARD_SOURCE_FULL_GRID = "full_grid"
WALK_FORWARD_SOURCE_REDUCED_CANDIDATES = "reduced_candidate_set"
WALK_FORWARD_SOURCE_FIXED_PORTFOLIO = "fixed_portfolio_definition"
WALK_FORWARD_SELECTION_HIGHEST_ROBUST_SCORE = "highest_robust_score"
WALK_FORWARD_SELECTION_FIXED_PORTFOLIO = "fixed_portfolio_definition"


@dataclass(frozen=True)
class WalkForwardFoldWindow:
    fold_index: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str


@dataclass(frozen=True)
class WalkForwardStudyArtifacts:
    wf_study_id: str
    strategy: str
    dataset_id: str
    timeframe: str
    candidate_source_mode: str
    param_names: tuple[str, ...]
    schedule_json: dict[str, Any]
    folds: pd.DataFrame
    fold_metrics: pd.DataFrame
    stitched_oos_equity: pd.Series
    stitched_oos_metrics: dict[str, float]


@dataclass(frozen=True)
class WalkForwardPortfolioAssetDefinition:
    dataset_id: str
    target_weight: float | None = None
    display_name: str | None = None


@dataclass(frozen=True)
class WalkForwardPortfolioStrategyBlockAssetDefinition:
    dataset_id: str
    target_weight: float | None = None
    display_name: str | None = None


@dataclass(frozen=True)
class WalkForwardPortfolioStrategyBlockDefinition:
    block_id: str
    strategy_cls: Type[Strategy]
    strategy_params: Dict[str, Any]
    assets: Sequence[WalkForwardPortfolioStrategyBlockAssetDefinition]
    budget_weight: float | None = None
    display_name: str | None = None


def build_anchored_walk_forward_schedule(
    *,
    index: pd.DatetimeIndex,
    first_test_start: str | pd.Timestamp,
    test_window_bars: int,
    num_folds: int,
    min_train_bars: int = 1,
) -> tuple[WalkForwardFoldWindow, ...]:
    if test_window_bars <= 0:
        raise ValueError("test_window_bars must be positive.")
    if num_folds <= 0:
        raise ValueError("num_folds must be positive.")
    if min_train_bars <= 0:
        raise ValueError("min_train_bars must be positive.")
    normalized_index = _normalize_index(index)
    if normalized_index.empty:
        raise ValueError("Walk-forward schedule requires a non-empty datetime index.")
    first_test_ts = _normalize_timestamp(first_test_start)
    first_test_pos = int(normalized_index.searchsorted(first_test_ts, side="left"))
    if first_test_pos >= len(normalized_index):
        raise ValueError("first_test_start falls after the available data range.")
    if first_test_pos < min_train_bars:
        raise ValueError("Not enough training bars before first_test_start.")

    folds: list[WalkForwardFoldWindow] = []
    for fold_idx in range(1, num_folds + 1):
        test_start_pos = first_test_pos + ((fold_idx - 1) * int(test_window_bars))
        test_end_pos = test_start_pos + int(test_window_bars) - 1
        if test_end_pos >= len(normalized_index):
            break
        train_end_pos = test_start_pos - 1
        if train_end_pos + 1 < min_train_bars:
            break
        folds.append(
            WalkForwardFoldWindow(
                fold_index=fold_idx,
                train_start=normalized_index[0].isoformat(),
                train_end=normalized_index[train_end_pos].isoformat(),
                test_start=normalized_index[test_start_pos].isoformat(),
                test_end=normalized_index[test_end_pos].isoformat(),
            )
        )
    if not folds:
        raise ValueError("Walk-forward schedule could not form any complete folds from the available data.")
    return tuple(folds)


def candidate_param_sets_from_records(
    records: Sequence[OptimizationCandidateRecord | dict[str, Any]],
) -> list[dict[str, Any]]:
    param_sets: list[dict[str, Any]] = []
    for record in records:
        params_json = getattr(record, "params_json", None)
        if params_json is None and isinstance(record, dict):
            params_json = record.get("params_json")
        if not params_json:
            continue
        decoded = json.loads(str(params_json))
        if isinstance(decoded, dict) and decoded:
            param_sets.append(decoded)
    if not param_sets:
        raise ValueError("No candidate parameter sets were provided.")
    return param_sets


def run_walk_forward_study(
    *,
    wf_study_id: str,
    dataset_id: str,
    data_loader: Callable[[str], pd.DataFrame],
    strategy_cls: Type[Strategy],
    base_config: BacktestConfig,
    timeframe: str,
    first_test_start: str | pd.Timestamp,
    test_window_bars: int,
    num_folds: int,
    param_grid: dict[str, Sequence[Any]] | None = None,
    candidate_params: Sequence[dict[str, Any]] | None = None,
    candidate_source_mode: str = WALK_FORWARD_SOURCE_FULL_GRID,
    catalog: ResultCatalog | None = None,
    requested_execution_mode: ExecutionMode | str = ExecutionMode.AUTO,
    min_train_bars: int = 1,
    selection_rule: str = WALK_FORWARD_SELECTION_HIGHEST_ROBUST_SCORE,
    description: str = "",
    source_study_id: str | None = None,
    source_batch_id: str | None = None,
) -> WalkForwardStudyArtifacts:
    source_mode = str(candidate_source_mode or WALK_FORWARD_SOURCE_FULL_GRID)
    if source_mode not in {WALK_FORWARD_SOURCE_FULL_GRID, WALK_FORWARD_SOURCE_REDUCED_CANDIDATES}:
        raise ValueError(f"Unsupported walk-forward candidate source mode: {source_mode}")
    if selection_rule != WALK_FORWARD_SELECTION_HIGHEST_ROBUST_SCORE:
        raise ValueError(f"Unsupported walk-forward selection rule: {selection_rule}")

    bars = _normalize_bars(data_loader(timeframe))
    schedule = build_anchored_walk_forward_schedule(
        index=bars.index,
        first_test_start=first_test_start,
        test_window_bars=test_window_bars,
        num_folds=num_folds,
        min_train_bars=min_train_bars,
    )
    if source_mode == WALK_FORWARD_SOURCE_FULL_GRID:
        param_sets = _param_sets_from_grid(param_grid or {})
    else:
        param_sets = [dict(item) for item in list(candidate_params or ())]
    if not param_sets:
        raise ValueError("Walk-forward study requires a non-empty parameter universe.")
    param_names = _infer_param_names(param_sets)

    folds_rows: list[dict[str, Any]] = []
    fold_metrics_rows: list[dict[str, Any]] = []
    stitched_segments: list[pd.Series] = []
    previous_params: dict[str, Any] | None = None
    orchestrator = ExecutionOrchestrator()

    for fold in schedule:
        train_study_id = f"{wf_study_id}_train_fold_{fold.fold_index:03d}"
        train_batch_id = f"{wf_study_id}_train_{fold.fold_index:03d}"
        train_df = _evaluate_param_sets_for_window(
            orchestrator=orchestrator,
            data_loader=data_loader,
            bars=bars,
            dataset_id=dataset_id,
            strategy_cls=strategy_cls,
            param_sets=param_sets,
            base_config=base_config,
            timeframe=timeframe,
            start=fold.train_start,
            end=fold.train_end,
            batch_id=train_batch_id,
            catalog=catalog,
            requested_execution_mode=requested_execution_mode,
        )
        train_artifacts = build_optimization_study_artifacts(
            df=train_df,
            study_id=train_study_id,
            batch_id=train_batch_id,
            strategy=strategy_cls.__name__,
            dataset_scope=[dataset_id],
            param_names=param_names,
            timeframes=[timeframe],
            horizons=[f"{fold.train_start}->{fold.train_end}"],
            score_version=ROBUST_SCORE_VERSION,
        )
        if catalog is not None:
            catalog.save_optimization_study(
                study_id=train_artifacts.study_id,
                batch_id=train_artifacts.batch_id,
                strategy=train_artifacts.strategy,
                dataset_scope=train_artifacts.dataset_scope,
                param_names=train_artifacts.param_names,
                timeframes=train_artifacts.timeframes,
                horizons=train_artifacts.horizons,
                score_version=train_artifacts.score_version,
                aggregates=train_artifacts.aggregates,
                asset_results=train_artifacts.asset_results,
            )
        selected = train_artifacts.aggregates.iloc[0]
        selected_params = json.loads(str(selected["params_json"]))
        test_result = _execute_fold_test(
            orchestrator=orchestrator,
            data_loader=data_loader,
            bars=bars,
            dataset_id=dataset_id,
            strategy_cls=strategy_cls,
            strategy_params=selected_params,
            base_config=base_config,
            timeframe=timeframe,
            start=fold.test_start,
            end=fold.test_end,
            batch_id=f"{wf_study_id}_test_{fold.fold_index:03d}",
            catalog=catalog,
            requested_execution_mode=requested_execution_mode,
        )
        stitched_segments.append(test_result.equity_curve)
        train_metrics = {
            "robust_score": float(selected["robust_score"]),
            "median_total_return": float(selected["median_total_return"]),
            "median_cagr": float(selected["median_cagr"]) if pd.notna(selected.get("median_cagr")) else None,
            "median_sharpe": float(selected["median_sharpe"]),
            "median_rolling_sharpe": (
                float(selected["median_rolling_sharpe"])
                if pd.notna(selected.get("median_rolling_sharpe"))
                else None
            ),
            "worst_max_drawdown": float(selected["worst_max_drawdown"]),
            "sharpe_std": float(selected["sharpe_std"]),
            "profitable_asset_ratio": float(selected["profitable_asset_ratio"]),
            "dataset_count": int(selected["dataset_count"]),
            "run_count": int(selected["run_count"]),
            "train_study_id": train_study_id,
        }
        test_metrics = dict(test_result.metrics.as_dict())
        test_metrics["trade_count"] = int(len(test_result.trades))
        degradation = {
            "total_return_delta": float(test_metrics["total_return"]) - float(train_metrics["median_total_return"]),
            "sharpe_delta": float(test_metrics["sharpe"]) - float(train_metrics["median_sharpe"]),
            "max_drawdown_delta": abs(float(test_metrics["max_drawdown"])) - abs(float(train_metrics["worst_max_drawdown"])),
        }
        param_drift = _compute_param_drift(previous_params, selected_params)
        previous_params = dict(selected_params)
        folds_rows.append(
            {
                "wf_study_id": wf_study_id,
                "fold_index": int(fold.fold_index),
                "train_study_id": train_study_id,
                "timeframe": timeframe,
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                "selected_param_set_id": str(selected["param_key"]),
                "selected_params_json": json.dumps(selected_params, sort_keys=True),
                "train_rank": int(selected["seq"]),
                "train_robust_score": float(selected["robust_score"]),
                "test_run_id": str(test_result.run_id),
                "status": "finished",
            }
        )
        fold_metrics_rows.append(
            {
                "wf_study_id": wf_study_id,
                "fold_index": int(fold.fold_index),
                "train_metrics_json": json.dumps(train_metrics, sort_keys=True),
                "test_metrics_json": json.dumps(test_metrics, sort_keys=True),
                "degradation_json": json.dumps(degradation, sort_keys=True),
                "param_drift_json": json.dumps(param_drift, sort_keys=True),
            }
        )

    folds_df = pd.DataFrame(folds_rows)
    fold_metrics_df = pd.DataFrame(fold_metrics_rows)
    stitched_equity = stitch_oos_equity_curves(stitched_segments)
    stitched_metrics = compute_metrics(
        stitched_equity,
        risk_free_rate=base_config.risk_free_rate,
        timeframe=timeframe,
        annualization=base_config.sharpe_annualization,
        session_seconds_per_day=base_config.sharpe_session_seconds_per_day,
        sharpe_basis=base_config.sharpe_basis,
    ).as_dict()
    schedule_json = {
        "mode": "anchored",
        "first_test_start": _normalize_timestamp(first_test_start).isoformat(),
        "test_window_bars": int(test_window_bars),
        "num_requested_folds": int(num_folds),
        "num_actual_folds": int(len(schedule)),
        "min_train_bars": int(min_train_bars),
        "folds": [fold.__dict__ for fold in schedule],
    }
    if catalog is not None:
        catalog.save_walk_forward_study(
            wf_study_id=wf_study_id,
            batch_id=wf_study_id,
            strategy=strategy_cls.__name__,
            dataset_id=dataset_id,
            timeframe=timeframe,
            candidate_source_mode=source_mode,
            param_names=param_names,
            schedule_json=schedule_json,
            selection_rule=selection_rule,
            params_json={
                "param_grid": param_grid or {},
                "candidate_params": list(candidate_params or ()),
                "source_study_id": str(source_study_id or ""),
                "source_batch_id": str(source_batch_id or ""),
                "source_kind": "single_strategy",
            },
            status="finished",
            description=description,
            folds=folds_df,
            fold_metrics=fold_metrics_df,
            stitched_metrics=stitched_metrics,
            stitched_equity=stitched_equity,
        )
    return WalkForwardStudyArtifacts(
        wf_study_id=wf_study_id,
        strategy=strategy_cls.__name__,
        dataset_id=dataset_id,
        timeframe=timeframe,
        candidate_source_mode=source_mode,
        param_names=tuple(param_names),
        schedule_json=schedule_json,
        folds=folds_df,
        fold_metrics=fold_metrics_df,
        stitched_oos_equity=stitched_equity,
        stitched_oos_metrics=stitched_metrics,
    )


def run_walk_forward_portfolio_study(
    *,
    wf_study_id: str,
    portfolio_dataset_id: str,
    data_loader: Callable[[str, str], pd.DataFrame],
    base_config: BacktestConfig,
    timeframe: str,
    first_test_start: str | pd.Timestamp,
    test_window_bars: int,
    num_folds: int,
    shared_strategy_cls: Type[Strategy] | None = None,
    portfolio_assets: Sequence[WalkForwardPortfolioAssetDefinition] | None = None,
    param_grid: dict[str, Sequence[Any]] | None = None,
    candidate_params: Sequence[dict[str, Any]] | None = None,
    strategy_blocks: Sequence[WalkForwardPortfolioStrategyBlockDefinition] | None = None,
    strategy_block_candidates: Sequence[Sequence[WalkForwardPortfolioStrategyBlockDefinition]] | None = None,
    strategy_label: str = "PortfolioExecution",
    construction_config: PortfolioConstructionConfig | None = None,
    normalize_weights: bool = True,
    candidate_source_mode: str = WALK_FORWARD_SOURCE_FULL_GRID,
    catalog: ResultCatalog | None = None,
    requested_execution_mode: ExecutionMode | str = ExecutionMode.AUTO,
    min_train_bars: int = 1,
    selection_rule: str | None = None,
    description: str = "",
    source_study_id: str | None = None,
    source_batch_id: str | None = None,
) -> WalkForwardStudyArtifacts:
    has_strategy_blocks = bool(strategy_blocks)
    if has_strategy_blocks:
        source_mode = str(candidate_source_mode or WALK_FORWARD_SOURCE_FIXED_PORTFOLIO)
        effective_selection_rule = (
            WALK_FORWARD_SELECTION_FIXED_PORTFOLIO
            if source_mode == WALK_FORWARD_SOURCE_FIXED_PORTFOLIO
            else str(selection_rule or WALK_FORWARD_SELECTION_HIGHEST_ROBUST_SCORE)
        )
    else:
        source_mode = str(candidate_source_mode or WALK_FORWARD_SOURCE_FULL_GRID)
        effective_selection_rule = str(selection_rule or WALK_FORWARD_SELECTION_HIGHEST_ROBUST_SCORE)
    if source_mode not in {
        WALK_FORWARD_SOURCE_FULL_GRID,
        WALK_FORWARD_SOURCE_REDUCED_CANDIDATES,
        WALK_FORWARD_SOURCE_FIXED_PORTFOLIO,
    }:
        raise ValueError(f"Unsupported portfolio walk-forward candidate source mode: {source_mode}")
    if effective_selection_rule not in {
        WALK_FORWARD_SELECTION_HIGHEST_ROBUST_SCORE,
        WALK_FORWARD_SELECTION_FIXED_PORTFOLIO,
    }:
        raise ValueError(f"Unsupported portfolio walk-forward selection rule: {effective_selection_rule}")
    if has_strategy_blocks:
        if shared_strategy_cls is not None or portfolio_assets:
            raise ValueError("Fixed strategy-block portfolio walk-forward cannot also receive shared strategy assets.")
        if source_mode == WALK_FORWARD_SOURCE_REDUCED_CANDIDATES and not strategy_block_candidates:
            raise ValueError("Reduced-candidate strategy-block walk-forward requires strategy_block_candidates.")
    else:
        if shared_strategy_cls is None:
            raise ValueError("Shared-strategy portfolio walk-forward requires shared_strategy_cls.")
        if not portfolio_assets:
            raise ValueError("Shared-strategy portfolio walk-forward requires portfolio_assets.")

    dataset_ids = sorted(
        {
            str(asset.dataset_id)
            for asset in list(portfolio_assets or ())
        }.union(
            {
                str(asset.dataset_id)
                for block in list(strategy_blocks or ())
                for asset in list(block.assets or ())
            }
        )
    )
    bars_by_dataset = _load_portfolio_bars_by_dataset(
        data_loader=data_loader,
        dataset_ids=dataset_ids,
        timeframe=timeframe,
    )
    common_index = _common_portfolio_index(tuple(bars_by_dataset.values()))
    schedule = build_anchored_walk_forward_schedule(
        index=common_index,
        first_test_start=first_test_start,
        test_window_bars=test_window_bars,
        num_folds=num_folds,
        min_train_bars=min_train_bars,
    )

    if has_strategy_blocks:
        candidate_block_sets = (
            [tuple(blocks) for blocks in list(strategy_block_candidates or ()) if blocks]
            if source_mode == WALK_FORWARD_SOURCE_REDUCED_CANDIDATES
            else [tuple(strategy_blocks or ())]
        )
        param_sets = [_serialize_portfolio_strategy_blocks(blocks) for blocks in candidate_block_sets]
        param_names: list[str] = []
    elif source_mode == WALK_FORWARD_SOURCE_FULL_GRID:
        param_sets = _param_sets_from_grid(param_grid or {})
        param_names = _infer_param_names(param_sets)
    else:
        param_sets = [dict(item) for item in list(candidate_params or ())]
        param_names = _infer_param_names(param_sets)
    if not param_sets:
        raise ValueError("Portfolio walk-forward study requires a non-empty parameter universe.")

    folds_rows: list[dict[str, Any]] = []
    fold_metrics_rows: list[dict[str, Any]] = []
    stitched_segments: list[pd.Series] = []
    previous_params: dict[str, Any] | None = None
    orchestrator = ExecutionOrchestrator()

    for fold in schedule:
        train_study_id = f"{wf_study_id}_train_fold_{fold.fold_index:03d}"
        train_batch_id = f"{wf_study_id}_train_{fold.fold_index:03d}"
        if has_strategy_blocks:
            if source_mode == WALK_FORWARD_SOURCE_REDUCED_CANDIDATES:
                train_df = _evaluate_strategy_block_portfolio_candidates_for_window(
                    orchestrator=orchestrator,
                    bars_by_dataset=bars_by_dataset,
                    portfolio_dataset_id=portfolio_dataset_id,
                    strategy_block_candidates=candidate_block_sets,
                    base_config=base_config,
                    timeframe=timeframe,
                    start=fold.train_start,
                    end=fold.train_end,
                    batch_id=train_batch_id,
                    catalog=catalog,
                    requested_execution_mode=requested_execution_mode,
                    construction_config=construction_config,
                    normalize_weights=normalize_weights,
                )
                selected = train_df.iloc[0]
                selected_candidate_index = int(selected["candidate_index"])
                selected_blocks = candidate_block_sets[selected_candidate_index]
                selected_params_json = str(selected["params_json"])
                selected_params = json.loads(selected_params_json)
                selected_param_set_id = str(selected["param_key"])
                train_metrics = {
                    "robust_score": float(selected["robust_score"]),
                    "total_return": float(selected["total_return"]),
                    "sharpe": float(selected["sharpe"]),
                    "max_drawdown": float(selected["max_drawdown"]),
                    "trade_count": int(selected["trade_count"]),
                    "candidate_count": int(len(candidate_block_sets)),
                    "train_study_id": None,
                }
                train_row_rank = int(selected["seq"])
                train_robust_score = float(selected["robust_score"])
            else:
                selected_blocks = tuple(strategy_blocks or ())
                selected_params = _serialize_portfolio_strategy_blocks(selected_blocks)
                train_result = _execute_strategy_block_portfolio_fold(
                    orchestrator=orchestrator,
                    bars_by_dataset=bars_by_dataset,
                    portfolio_dataset_id=portfolio_dataset_id,
                    strategy_blocks=selected_blocks,
                    base_config=base_config,
                    timeframe=timeframe,
                    start=fold.train_start,
                    end=fold.train_end,
                    batch_id=train_batch_id,
                    catalog=catalog,
                    requested_execution_mode=requested_execution_mode,
                    construction_config=construction_config,
                    normalize_weights=normalize_weights,
                )
                train_metrics = dict(train_result.metrics.as_dict())
                train_metrics["trade_count"] = int(len(train_result.trades))
                train_metrics["train_study_id"] = None
                train_row_rank = None
                train_robust_score = None
                selected_param_set_id = WALK_FORWARD_SOURCE_FIXED_PORTFOLIO
                selected_params_json = json.dumps(selected_params, sort_keys=True)
            test_result = _execute_strategy_block_portfolio_fold(
                orchestrator=orchestrator,
                bars_by_dataset=bars_by_dataset,
                portfolio_dataset_id=portfolio_dataset_id,
                strategy_blocks=selected_blocks,
                base_config=base_config,
                timeframe=timeframe,
                start=fold.test_start,
                end=fold.test_end,
                batch_id=f"{wf_study_id}_test_{fold.fold_index:03d}",
                catalog=catalog,
                requested_execution_mode=requested_execution_mode,
                construction_config=construction_config,
                normalize_weights=normalize_weights,
            )
        else:
            train_df = _evaluate_shared_strategy_portfolio_param_sets_for_window(
                orchestrator=orchestrator,
                bars_by_dataset=bars_by_dataset,
                portfolio_dataset_id=portfolio_dataset_id,
                portfolio_assets=portfolio_assets or (),
                shared_strategy_cls=shared_strategy_cls,
                param_sets=param_sets,
                base_config=base_config,
                timeframe=timeframe,
                start=fold.train_start,
                end=fold.train_end,
                batch_id=train_batch_id,
                catalog=catalog,
                requested_execution_mode=requested_execution_mode,
                construction_config=construction_config,
                normalize_weights=normalize_weights,
            )
            train_artifacts = build_optimization_study_artifacts(
                df=train_df,
                study_id=train_study_id,
                batch_id=train_batch_id,
                strategy=strategy_label,
                dataset_scope=[portfolio_dataset_id],
                param_names=param_names,
                timeframes=[timeframe],
                horizons=[f"{fold.train_start}->{fold.train_end}"],
                score_version=ROBUST_SCORE_VERSION,
            )
            if catalog is not None:
                catalog.save_optimization_study(
                    study_id=train_artifacts.study_id,
                    batch_id=train_artifacts.batch_id,
                    strategy=train_artifacts.strategy,
                    dataset_scope=train_artifacts.dataset_scope,
                    param_names=train_artifacts.param_names,
                    timeframes=train_artifacts.timeframes,
                    horizons=train_artifacts.horizons,
                    score_version=train_artifacts.score_version,
                    aggregates=train_artifacts.aggregates,
                    asset_results=train_artifacts.asset_results,
                )
            selected = train_artifacts.aggregates.iloc[0]
            selected_params = json.loads(str(selected["params_json"]))
            test_result = _execute_shared_strategy_portfolio_fold_test(
                orchestrator=orchestrator,
                bars_by_dataset=bars_by_dataset,
                portfolio_dataset_id=portfolio_dataset_id,
                portfolio_assets=portfolio_assets or (),
                shared_strategy_cls=shared_strategy_cls,
                strategy_params=selected_params,
                base_config=base_config,
                timeframe=timeframe,
                start=fold.test_start,
                end=fold.test_end,
                batch_id=f"{wf_study_id}_test_{fold.fold_index:03d}",
                catalog=catalog,
                requested_execution_mode=requested_execution_mode,
                construction_config=construction_config,
                normalize_weights=normalize_weights,
            )
            train_metrics = {
                "robust_score": float(selected["robust_score"]),
                "median_total_return": float(selected["median_total_return"]),
                "median_cagr": float(selected["median_cagr"]) if pd.notna(selected.get("median_cagr")) else None,
                "median_sharpe": float(selected["median_sharpe"]),
                "median_rolling_sharpe": (
                    float(selected["median_rolling_sharpe"])
                    if pd.notna(selected.get("median_rolling_sharpe"))
                    else None
                ),
                "worst_max_drawdown": float(selected["worst_max_drawdown"]),
                "sharpe_std": float(selected["sharpe_std"]),
                "profitable_asset_ratio": float(selected["profitable_asset_ratio"]),
                "dataset_count": int(selected["dataset_count"]),
                "run_count": int(selected["run_count"]),
                "train_study_id": train_study_id,
            }
            train_row_rank = int(selected["seq"])
            train_robust_score = float(selected["robust_score"])
            selected_param_set_id = str(selected["param_key"])
            selected_params_json = json.dumps(selected_params, sort_keys=True)

        stitched_segments.append(test_result.equity_curve)
        test_metrics = dict(test_result.metrics.as_dict())
        test_metrics["trade_count"] = int(len(test_result.trades))
        degradation = {
            "total_return_delta": float(test_metrics["total_return"]) - float(train_metrics.get("median_total_return", train_metrics.get("total_return", 0.0))),
            "sharpe_delta": float(test_metrics["sharpe"]) - float(train_metrics.get("median_sharpe", train_metrics.get("sharpe", 0.0))),
            "max_drawdown_delta": abs(float(test_metrics["max_drawdown"])) - abs(float(train_metrics.get("worst_max_drawdown", train_metrics.get("max_drawdown", 0.0)))),
        }
        param_drift = _compute_param_drift(previous_params, selected_params)
        previous_params = dict(selected_params)
        folds_rows.append(
            {
                "wf_study_id": wf_study_id,
                "fold_index": int(fold.fold_index),
                "train_study_id": "" if has_strategy_blocks else train_study_id,
                "timeframe": timeframe,
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                "selected_param_set_id": selected_param_set_id,
                "selected_params_json": selected_params_json,
                "train_rank": train_row_rank,
                "train_robust_score": train_robust_score,
                "test_run_id": str(test_result.run_id),
                "status": "finished",
            }
        )
        fold_metrics_rows.append(
            {
                "wf_study_id": wf_study_id,
                "fold_index": int(fold.fold_index),
                "train_metrics_json": json.dumps(train_metrics, sort_keys=True),
                "test_metrics_json": json.dumps(test_metrics, sort_keys=True),
                "degradation_json": json.dumps(degradation, sort_keys=True),
                "param_drift_json": json.dumps(param_drift, sort_keys=True),
            }
        )

    folds_df = pd.DataFrame(folds_rows)
    fold_metrics_df = pd.DataFrame(fold_metrics_rows)
    stitched_equity = stitch_oos_equity_curves(stitched_segments)
    stitched_metrics = compute_metrics(
        stitched_equity,
        risk_free_rate=base_config.risk_free_rate,
        timeframe=timeframe,
        annualization=base_config.sharpe_annualization,
        session_seconds_per_day=base_config.sharpe_session_seconds_per_day,
        sharpe_basis=base_config.sharpe_basis,
    ).as_dict()
    schedule_json = {
        "mode": "anchored",
        "portfolio_mode": "strategy_blocks" if has_strategy_blocks else "shared_strategy",
        "first_test_start": _normalize_timestamp(first_test_start).isoformat(),
        "test_window_bars": int(test_window_bars),
        "num_requested_folds": int(num_folds),
        "num_actual_folds": int(len(schedule)),
        "min_train_bars": int(min_train_bars),
        "dataset_ids": dataset_ids,
        "folds": [fold.__dict__ for fold in schedule],
    }
    stored_params = {
        "shared_strategy_cls": shared_strategy_cls.__name__ if shared_strategy_cls is not None else None,
        "portfolio_assets": [
            {
                "dataset_id": asset.dataset_id,
                "target_weight": asset.target_weight,
                "display_name": asset.display_name,
            }
            for asset in list(portfolio_assets or ())
        ],
        "strategy_blocks": _serialize_strategy_block_definitions(strategy_blocks or ()),
        "strategy_block_candidates": [
            _serialize_strategy_block_definitions(blocks)
            for blocks in list(strategy_block_candidates or ())
            if blocks
        ],
        "param_grid": param_grid or {},
        "candidate_params": list(candidate_params or ()),
        "construction_config": _serialize_construction_config(construction_config),
        "normalize_weights": bool(normalize_weights),
        "source_kind": "portfolio",
        "source_study_id": str(source_study_id or ""),
        "source_batch_id": str(source_batch_id or ""),
    }
    if catalog is not None:
        catalog.save_walk_forward_study(
            wf_study_id=wf_study_id,
            batch_id=wf_study_id,
            strategy=str(strategy_label),
            dataset_id=portfolio_dataset_id,
            timeframe=timeframe,
            candidate_source_mode=source_mode,
            param_names=param_names,
            schedule_json=schedule_json,
            selection_rule=effective_selection_rule,
            params_json=stored_params,
            status="finished",
            description=description,
            folds=folds_df,
            fold_metrics=fold_metrics_df,
            stitched_metrics=stitched_metrics,
            stitched_equity=stitched_equity,
        )
    return WalkForwardStudyArtifacts(
        wf_study_id=wf_study_id,
        strategy=str(strategy_label),
        dataset_id=portfolio_dataset_id,
        timeframe=timeframe,
        candidate_source_mode=source_mode,
        param_names=tuple(param_names),
        schedule_json=schedule_json,
        folds=folds_df,
        fold_metrics=fold_metrics_df,
        stitched_oos_equity=stitched_equity,
        stitched_oos_metrics=stitched_metrics,
    )


def stitch_oos_equity_curves(curves: Sequence[pd.Series]) -> pd.Series:
    stitched: list[pd.Series] = []
    running_end: float | None = None
    last_ts = None
    for curve in curves:
        series = curve.copy()
        if series is None or series.empty:
            continue
        if running_end is None:
            scaled = series.astype(float)
        else:
            first_value = float(series.iloc[0])
            scale = 1.0 if first_value == 0.0 else float(running_end) / first_value
            scaled = series.astype(float) * scale
        if last_ts is not None:
            scaled = scaled.loc[scaled.index > last_ts]
        if scaled.empty:
            continue
        stitched.append(scaled)
        running_end = float(scaled.iloc[-1])
        last_ts = scaled.index[-1]
    if not stitched:
        raise ValueError("Cannot stitch an empty set of walk-forward OOS equity curves.")
    result = pd.concat(stitched).sort_index(kind="mergesort")
    result.name = "stitched_oos_equity"
    return result


def _evaluate_param_sets_for_window(
    *,
    orchestrator: ExecutionOrchestrator,
    data_loader: Callable[[str], pd.DataFrame],
    bars: pd.DataFrame,
    dataset_id: str,
    strategy_cls: Type[Strategy],
    param_sets: Sequence[dict[str, Any]],
    base_config: BacktestConfig,
    timeframe: str,
    start: str,
    end: str,
    batch_id: str,
    catalog: ResultCatalog | None,
    requested_execution_mode: ExecutionMode | str,
) -> pd.DataFrame:
    requested_mode = ExecutionMode.from_value(requested_execution_mode)
    initial_base_execution = timeframe != "1 minutes" and requested_mode != ExecutionMode.VECTORIZED
    config = replace(
        base_config,
        batch_id=batch_id,
        timeframe=timeframe,
        time_horizon_start=_normalize_timestamp(start),
        time_horizon_end=_normalize_timestamp(end),
        base_execution=initial_base_execution,
    )
    resolution = orchestrator.resolve_param_grid(
        data=bars,
        dataset_id=dataset_id,
        strategy_cls=strategy_cls,
        param_grid=param_sets,
        config=config,
        requested_execution_mode=requested_mode,
        base_data=None,
    )
    effective_config = resolution.effective_config or config
    base_bars = data_loader("1 minutes") if effective_config.base_execution else None
    results = orchestrator.execute_param_grid(
        data=bars,
        base_data=base_bars,
        dataset_id=dataset_id,
        strategy_cls=strategy_cls,
        param_grid=param_sets,
        catalog=catalog,
        config=effective_config,
        requested_execution_mode=requested_mode,
        workload_type=WorkloadType.WALK_FORWARD_FOLD,
    )
    records: list[dict[str, Any]] = []
    for params, result in zip(param_sets, results):
        metrics = result.metrics.as_dict()
        records.append(
            {
                **dict(params),
                "dataset_id": dataset_id,
                "timeframe": timeframe,
                "start": start,
                "end": end,
                "total_return": metrics.get("total_return"),
                "cagr": metrics.get("cagr"),
                "max_drawdown": metrics.get("max_drawdown"),
                "sharpe": metrics.get("sharpe"),
                "rolling_sharpe": metrics.get("rolling_sharpe"),
                "run_id": result.run_id,
                "logical_run_id": result.logical_run_id,
                "requested_execution_mode": result.requested_execution_mode.value,
                "resolved_execution_mode": result.resolved_execution_mode.value,
                "engine_impl": result.engine_impl,
                "engine_version": result.engine_version,
            }
        )
    return pd.DataFrame(records)


def _execute_fold_test(
    *,
    orchestrator: ExecutionOrchestrator,
    data_loader: Callable[[str], pd.DataFrame],
    bars: pd.DataFrame,
    dataset_id: str,
    strategy_cls: Type[Strategy],
    strategy_params: dict[str, Any],
    base_config: BacktestConfig,
    timeframe: str,
    start: str,
    end: str,
    batch_id: str,
    catalog: ResultCatalog | None,
    requested_execution_mode: ExecutionMode | str,
):
    requested_mode = ExecutionMode.from_value(requested_execution_mode)
    initial_base_execution = timeframe != "1 minutes" and requested_mode != ExecutionMode.VECTORIZED
    config = replace(
        base_config,
        batch_id=batch_id,
        timeframe=timeframe,
        time_horizon_start=_normalize_timestamp(start),
        time_horizon_end=_normalize_timestamp(end),
        base_execution=initial_base_execution,
        use_cache=False,
    )
    resolution = orchestrator.resolve(
        ExecutionRequest(
            data=bars,
            dataset_id=dataset_id,
            strategy_cls=strategy_cls,
            strategy_params=strategy_params,
            config=config,
            catalog=catalog,
            base_data=None,
            requested_execution_mode=requested_mode,
            workload_type=WorkloadType.WALK_FORWARD_FOLD,
        )
    )
    effective_config = resolution.effective_config or config
    effective_base = data_loader("1 minutes") if effective_config.base_execution else None
    return orchestrator.execute(
        ExecutionRequest(
            data=bars,
            dataset_id=dataset_id,
            strategy_cls=strategy_cls,
            strategy_params=strategy_params,
            config=effective_config,
            catalog=catalog,
            base_data=effective_base,
            requested_execution_mode=requested_mode,
            workload_type=WorkloadType.WALK_FORWARD_FOLD,
        )
    )


def _evaluate_shared_strategy_portfolio_param_sets_for_window(
    *,
    orchestrator: ExecutionOrchestrator,
    bars_by_dataset: dict[str, pd.DataFrame],
    portfolio_dataset_id: str,
    portfolio_assets: Sequence[WalkForwardPortfolioAssetDefinition],
    shared_strategy_cls: Type[Strategy],
    param_sets: Sequence[dict[str, Any]],
    base_config: BacktestConfig,
    timeframe: str,
    start: str,
    end: str,
    batch_id: str,
    catalog: ResultCatalog | None,
    requested_execution_mode: ExecutionMode | str,
    construction_config: PortfolioConstructionConfig | None,
    normalize_weights: bool,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for params in param_sets:
        result = _execute_shared_strategy_portfolio_fold_test(
            orchestrator=orchestrator,
            bars_by_dataset=bars_by_dataset,
            portfolio_dataset_id=portfolio_dataset_id,
            portfolio_assets=portfolio_assets,
            shared_strategy_cls=shared_strategy_cls,
            strategy_params=params,
            base_config=base_config,
            timeframe=timeframe,
            start=start,
            end=end,
            batch_id=batch_id,
            catalog=catalog,
            requested_execution_mode=requested_execution_mode,
            construction_config=construction_config,
            normalize_weights=normalize_weights,
        )
        metrics = result.metrics.as_dict()
        records.append(
            {
                **dict(params),
                "dataset_id": portfolio_dataset_id,
                "timeframe": timeframe,
                "start": start,
                "end": end,
                "total_return": metrics.get("total_return"),
                "cagr": metrics.get("cagr"),
                "max_drawdown": metrics.get("max_drawdown"),
                "sharpe": metrics.get("sharpe"),
                "rolling_sharpe": metrics.get("rolling_sharpe"),
                "run_id": result.run_id,
                "logical_run_id": result.logical_run_id,
                "requested_execution_mode": result.requested_execution_mode.value,
                "resolved_execution_mode": result.resolved_execution_mode.value,
                "engine_impl": result.engine_impl,
                "engine_version": result.engine_version,
            }
        )
    return pd.DataFrame(records)


def _execute_shared_strategy_portfolio_fold_test(
    *,
    orchestrator: ExecutionOrchestrator,
    bars_by_dataset: dict[str, pd.DataFrame],
    portfolio_dataset_id: str,
    portfolio_assets: Sequence[WalkForwardPortfolioAssetDefinition],
    shared_strategy_cls: Type[Strategy],
    strategy_params: dict[str, Any],
    base_config: BacktestConfig,
    timeframe: str,
    start: str,
    end: str,
    batch_id: str,
    catalog: ResultCatalog | None,
    requested_execution_mode: ExecutionMode | str,
    construction_config: PortfolioConstructionConfig | None,
    normalize_weights: bool,
):
    config = replace(
        base_config,
        batch_id=batch_id,
        timeframe=timeframe,
        time_horizon_start=_normalize_timestamp(start),
        time_horizon_end=_normalize_timestamp(end),
        base_execution=False,
        use_cache=False,
    )
    request = PortfolioExecutionRequest(
        assets=[
            PortfolioExecutionAsset(
                dataset_id=asset.dataset_id,
                data=bars_by_dataset[str(asset.dataset_id)],
                strategy_cls=shared_strategy_cls,
                strategy_params=dict(strategy_params),
                target_weight=asset.target_weight,
                display_name=asset.display_name,
            )
            for asset in portfolio_assets
        ],
        config=config,
        catalog=catalog,
        requested_execution_mode=requested_execution_mode,
        normalize_weights=normalize_weights,
        portfolio_dataset_id=portfolio_dataset_id,
        construction_config=construction_config,
    )
    return orchestrator.execute_portfolio(request)


def _execute_strategy_block_portfolio_fold(
    *,
    orchestrator: ExecutionOrchestrator,
    bars_by_dataset: dict[str, pd.DataFrame],
    portfolio_dataset_id: str,
    strategy_blocks: Sequence[WalkForwardPortfolioStrategyBlockDefinition],
    base_config: BacktestConfig,
    timeframe: str,
    start: str,
    end: str,
    batch_id: str,
    catalog: ResultCatalog | None,
    requested_execution_mode: ExecutionMode | str,
    construction_config: PortfolioConstructionConfig | None,
    normalize_weights: bool,
):
    config = replace(
        base_config,
        batch_id=batch_id,
        timeframe=timeframe,
        time_horizon_start=_normalize_timestamp(start),
        time_horizon_end=_normalize_timestamp(end),
        base_execution=False,
        use_cache=False,
    )
    request = PortfolioExecutionRequest(
        assets=[],
        strategy_blocks=[
            PortfolioExecutionStrategyBlock(
                block_id=block.block_id,
                strategy_cls=block.strategy_cls,
                strategy_params=dict(block.strategy_params),
                assets=[
                    PortfolioExecutionStrategyBlockAsset(
                        dataset_id=asset.dataset_id,
                        data=bars_by_dataset[str(asset.dataset_id)],
                        target_weight=asset.target_weight,
                        display_name=asset.display_name,
                    )
                    for asset in list(block.assets or ())
                ],
                budget_weight=block.budget_weight,
                display_name=block.display_name,
            )
            for block in strategy_blocks
        ],
        config=config,
        catalog=catalog,
        requested_execution_mode=requested_execution_mode,
        normalize_weights=normalize_weights,
        portfolio_dataset_id=portfolio_dataset_id,
        construction_config=construction_config,
    )
    return orchestrator.execute_portfolio(request)


def _evaluate_strategy_block_portfolio_candidates_for_window(
    *,
    orchestrator: ExecutionOrchestrator,
    bars_by_dataset: dict[str, pd.DataFrame],
    portfolio_dataset_id: str,
    strategy_block_candidates: Sequence[Sequence[WalkForwardPortfolioStrategyBlockDefinition]],
    base_config: BacktestConfig,
    timeframe: str,
    start: str,
    end: str,
    batch_id: str,
    catalog: ResultCatalog | None,
    requested_execution_mode: ExecutionMode | str,
    construction_config: PortfolioConstructionConfig | None,
    normalize_weights: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx, candidate_blocks in enumerate(list(strategy_block_candidates or ())):
        if not candidate_blocks:
            continue
        result = _execute_strategy_block_portfolio_fold(
            orchestrator=orchestrator,
            bars_by_dataset=bars_by_dataset,
            portfolio_dataset_id=portfolio_dataset_id,
            strategy_blocks=candidate_blocks,
            base_config=base_config,
            timeframe=timeframe,
            start=start,
            end=end,
            batch_id=f"{batch_id}_candidate_{idx + 1:03d}",
            catalog=catalog,
            requested_execution_mode=requested_execution_mode,
            construction_config=construction_config,
            normalize_weights=normalize_weights,
        )
        metrics = dict(result.metrics.as_dict())
        total_return = float(metrics.get("total_return", 0.0) or 0.0)
        sharpe = float(metrics.get("sharpe", 0.0) or 0.0)
        max_drawdown = float(metrics.get("max_drawdown", 0.0) or 0.0)
        robust_score = compute_robust_score(
            median_sharpe=sharpe,
            sharpe_std=0.0,
            worst_max_drawdown=abs(max_drawdown),
            profitable_asset_ratio=(1.0 if total_return > 0.0 else 0.0),
        )
        params_json = json.dumps(_serialize_portfolio_strategy_blocks(candidate_blocks), sort_keys=True)
        rows.append(
            {
                "candidate_index": int(idx),
                "param_key": params_json,
                "params_json": params_json,
                "total_return": total_return,
                "sharpe": sharpe,
                "max_drawdown": max_drawdown,
                "trade_count": int(len(result.trades)),
                "run_id": str(result.run_id),
                "robust_score": float(robust_score),
            }
        )
    if not rows:
        raise ValueError("No strategy-block candidates were available for train-fold evaluation.")
    frame = pd.DataFrame(rows)
    frame = frame.sort_values(
        by=["robust_score", "sharpe", "total_return"],
        ascending=[False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    frame.insert(0, "seq", np.arange(1, len(frame) + 1, dtype=int))
    return frame


def _load_portfolio_bars_by_dataset(
    *,
    data_loader: Callable[[str, str], pd.DataFrame],
    dataset_ids: Sequence[str],
    timeframe: str,
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for dataset_id in dataset_ids:
        bars = _normalize_bars(data_loader(str(dataset_id), timeframe))
        frames[str(dataset_id)] = bars
    if not frames:
        raise ValueError("Portfolio walk-forward requires at least one dataset.")
    return frames


def _common_portfolio_index(frames: Sequence[pd.DataFrame]) -> pd.DatetimeIndex:
    if not frames:
        raise ValueError("Portfolio walk-forward requires at least one loaded dataset.")
    common_index = _normalize_index(frames[0].index)
    for frame in frames[1:]:
        common_index = common_index.intersection(_normalize_index(frame.index))
    if common_index.empty:
        raise ValueError("Portfolio walk-forward could not align a non-empty common timestamp index.")
    return common_index


def _serialize_portfolio_strategy_blocks(
    strategy_blocks: Sequence[WalkForwardPortfolioStrategyBlockDefinition],
) -> dict[str, Any]:
    return {
        "strategy_blocks": _serialize_strategy_block_definitions(strategy_blocks),
    }


def _serialize_strategy_block_definitions(
    strategy_blocks: Sequence[WalkForwardPortfolioStrategyBlockDefinition],
) -> list[dict[str, Any]]:
    return [
        {
            "block_id": block.block_id,
            "display_name": block.display_name,
            "budget_weight": block.budget_weight,
            "strategy": block.strategy_cls.__name__,
            "params": dict(block.strategy_params),
            "assets": [
                {
                    "dataset_id": asset.dataset_id,
                    "target_weight": asset.target_weight,
                    "display_name": asset.display_name,
                }
                for asset in list(block.assets or ())
            ],
        }
        for block in list(strategy_blocks or ())
    ]


def _serialize_construction_config(construction_config: PortfolioConstructionConfig | None) -> dict[str, Any]:
    if construction_config is None:
        return {}
    return {
        "allocation_ownership": construction_config.allocation_ownership,
        "ranking_mode": construction_config.ranking_mode,
        "max_ranked_assets": construction_config.max_ranked_assets,
        "min_rank_score": construction_config.min_rank_score,
        "weighting_mode": construction_config.weighting_mode,
        "min_active_weight": construction_config.min_active_weight,
        "max_asset_weight": construction_config.max_asset_weight,
        "cash_reserve_weight": construction_config.cash_reserve_weight,
        "rebalance_mode": construction_config.rebalance_mode,
        "rebalance_every_n_bars": construction_config.rebalance_every_n_bars,
        "rebalance_weight_drift_threshold": construction_config.rebalance_weight_drift_threshold,
    }


def _compute_param_drift(previous_params: dict[str, Any] | None, current_params: dict[str, Any]) -> dict[str, Any]:
    if not previous_params:
        return {
            "has_previous": False,
            "switch_count": 0,
            "params": {},
        }
    details: dict[str, Any] = {}
    switch_count = 0
    keys = sorted(set(previous_params.keys()).union(current_params.keys()))
    for key in keys:
        previous = previous_params.get(key)
        current = current_params.get(key)
        changed = previous != current
        if changed:
            switch_count += 1
        abs_change = None
        try:
            abs_change = abs(float(current) - float(previous))
        except Exception:
            abs_change = None
        details[str(key)] = {
            "previous": previous,
            "current": current,
            "changed": bool(changed),
            "absolute_change": abs_change,
        }
    return {
        "has_previous": True,
        "switch_count": int(switch_count),
        "params": details,
    }


def _param_sets_from_grid(param_grid: dict[str, Sequence[Any]]) -> list[dict[str, Any]]:
    if not param_grid:
        raise ValueError("Full-grid walk-forward studies require a non-empty param_grid.")
    keys = list(param_grid.keys())
    values = [list(param_grid[key]) for key in keys]
    if any(len(items) == 0 for items in values):
        raise ValueError("Full-grid walk-forward studies require every parameter axis to contain values.")
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _infer_param_names(param_sets: Sequence[dict[str, Any]]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for params in param_sets:
        for key in params.keys():
            name = str(key)
            if name not in seen:
                ordered.append(name)
                seen.add(name)
    return ordered


def _normalize_bars(bars: pd.DataFrame) -> pd.DataFrame:
    if bars is None or bars.empty:
        raise ValueError("Walk-forward requires non-empty bar data.")
    frame = bars.copy().sort_index()
    frame.index = _normalize_index(frame.index)
    return frame


def _normalize_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("Walk-forward requires a DatetimeIndex.")
    normalized = index
    if normalized.tz is None:
        normalized = normalized.tz_localize("UTC")
    else:
        normalized = normalized.tz_convert("UTC")
    return normalized.sort_values()


def _normalize_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")
