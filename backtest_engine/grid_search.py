from __future__ import annotations

import hashlib
import itertools
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Type

import pandas as pd
import matplotlib.pyplot as plt

from .engine import BacktestConfig
from .execution import (
    BatchExecutionBenchmark,
    ExecutionMode,
    ExecutionOrchestrator,
    PortfolioExecutionAsset,
    PortfolioExecutionRequest,
    PortfolioExecutionStrategyBlock,
    PortfolioExecutionStrategyBlockAsset,
    UnsupportedExecutionModeError,
    WorkloadType,
)
from .strategy import Strategy
from .catalog import ResultCatalog
from .reporting import plot_param_heatmap
from .vectorized_engine import VectorizedEngine
from .vectorized_portfolio import (
    PortfolioAssetSpec,
    PortfolioConstructionConfig,
    PortfolioStrategyBlockAssetSpec,
    PortfolioStrategyBlockSpec,
    VectorizedPortfolioEngine,
)


def _hash_heatmap(payload: Dict) -> str:
    return hashlib.sha256(repr(sorted(payload.items())).encode()).hexdigest()


@dataclass
class GridSpec:
    params: Dict[str, Iterable]
    timeframes: Iterable[str]
    horizons: Iterable[tuple[pd.Timestamp | None, pd.Timestamp | None]]
    metric: str = "total_return"
    heatmap_rows: str = "param1"
    heatmap_cols: str = "param2"
    description: str = ""
    batch_id: str | None = None
    execution_mode: ExecutionMode | str = ExecutionMode.AUTO


@dataclass(frozen=True)
class IndependentAssetTarget:
    dataset_id: str
    data_loader: Callable[[str], pd.DataFrame]


@dataclass(frozen=True)
class PortfolioAssetTarget:
    dataset_id: str
    data_loader: Callable[[str], pd.DataFrame]
    target_weight: float | None = None


@dataclass(frozen=True)
class PortfolioStrategyBlockAssetTarget:
    dataset_id: str
    data_loader: Callable[[str], pd.DataFrame]
    target_weight: float | None = None
    display_name: str | None = None


@dataclass(frozen=True)
class PortfolioStrategyBlockTarget:
    block_id: str
    strategy_cls: Type[Strategy]
    strategy_params: Dict
    assets: Sequence[PortfolioStrategyBlockAssetTarget]
    budget_weight: float | None = None
    display_name: str | None = None


@dataclass(frozen=True)
class IndependentAssetStudyBenchmark:
    study_id: str
    asset_count: int
    total_runs: int
    total_duration_seconds: float
    vectorized_batches: int
    reference_batches: int
    total_chunk_count: int
    prepared_context_reused_batches: int
    per_batch: tuple[BatchExecutionBenchmark, ...] = ()


def build_horizons(end: pd.Timestamp, windows: Iterable[pd.Timedelta]) -> List[tuple[pd.Timestamp, pd.Timestamp]]:
    return [(end - w, end) for w in windows]


def _estimate_grid_total(grid: GridSpec) -> int:
    tf_list = list(grid.timeframes)
    horizon_list = list(grid.horizons)
    param_lists = list(grid.params.values()) if grid.params else []
    total = len(tf_list) * len(horizon_list)
    for lst in param_lists:
        total *= len(list(lst))
    return total


class GridSearch:
    """
    Orchestrates parameter/timeframe/horizon sweeps optionally producing heatmaps.
    """

    def __init__(
        self,
        dataset_id: str,
        data_loader,  # callable that returns bars for a timeframe
        strategy_cls: Type[Strategy],
        base_config: BacktestConfig,
        catalog: Optional[ResultCatalog],
    ) -> None:
        self.dataset_id = dataset_id
        self.data_loader = data_loader
        self.strategy_cls = strategy_cls
        self.base_config = base_config
        self.catalog = catalog
        self.last_batch_benchmarks: List[BatchExecutionBenchmark] = []

    def run(
        self,
        grid: GridSpec,
        make_heatmap: bool = True,
        stop_cb: Optional[Callable[[], bool]] = None,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> pd.DataFrame:
        records: List[Dict] = []
        self.last_batch_benchmarks = []
        orchestrator = ExecutionOrchestrator()
        tf_list = list(grid.timeframes)
        horizon_list = list(grid.horizons)
        param_lists = list(grid.params.values()) if grid.params else []
        total = len(tf_list) * len(horizon_list)
        for lst in param_lists:
            total *= len(lst)
        batch_id = grid.batch_id or f"batch_{uuid.uuid4().hex[:8]}"
        requested_mode = ExecutionMode.from_value(grid.execution_mode)
        if progress_cb:
            progress_cb(0, total)
        done = 0
        for timeframe in tf_list:
            bars = self.data_loader(timeframe)
            for (start, end) in horizon_list:
                param_grid = [dict(zip(grid.params.keys(), combo)) for combo in itertools.product(*grid.params.values())]
                initial_base_execution = timeframe != "1 minutes" and requested_mode != ExecutionMode.VECTORIZED
                base_config = replace(
                    self.base_config,
                    batch_id=batch_id,
                    timeframe=timeframe,
                    time_horizon_start=start,
                    time_horizon_end=end,
                    base_execution=initial_base_execution,
                )
                resolution = orchestrator.resolve_param_grid(
                    data=bars,
                    dataset_id=self.dataset_id,
                    strategy_cls=self.strategy_cls,
                    param_grid=param_grid,
                    config=base_config,
                    requested_execution_mode=requested_mode,
                    base_data=None,
                )
                config = resolution.effective_config or base_config
                base_bars = None
                if config.base_execution:
                    base_bars = self.data_loader("1 minutes")
                if stop_cb and stop_cb():
                    return pd.DataFrame(records)
                batch_results = orchestrator.execute_param_grid(
                    data=bars,
                    base_data=base_bars,
                    dataset_id=self.dataset_id,
                    strategy_cls=self.strategy_cls,
                    param_grid=param_grid,
                    catalog=self.catalog,
                    config=config,
                    requested_execution_mode=requested_mode,
                    workload_type=WorkloadType.OPTIMIZATION_BATCH,
                )
                if orchestrator.last_batch_benchmark is not None:
                    self.last_batch_benchmarks.append(orchestrator.last_batch_benchmark)
                for params, result in zip(param_grid, batch_results):
                    metrics = result.metrics.as_dict()
                    records.append(
                        {
                            **params,
                            "dataset_id": self.dataset_id,
                            "timeframe": timeframe,
                            "start": start,
                            "end": end,
                            "total_return": metrics.get("total_return"),
                            "cagr": metrics.get("cagr"),
                            "max_drawdown": metrics.get("max_drawdown"),
                            "sharpe": metrics.get("sharpe"),
                            "rolling_sharpe": metrics.get("rolling_sharpe"),
                            grid.metric: metrics[grid.metric],
                            "run_id": result.run_id,
                            "logical_run_id": result.logical_run_id,
                            "requested_execution_mode": result.requested_execution_mode.value,
                            "resolved_execution_mode": result.resolved_execution_mode.value,
                            "engine_impl": result.engine_impl,
                            "engine_version": result.engine_version,
                        }
                    )
                    done += 1
                    if progress_cb:
                        progress_cb(done, total)

        df = pd.DataFrame(records)
        df.attrs["batch_benchmarks"] = tuple(self.last_batch_benchmarks)
        if make_heatmap and len(grid.params) >= 2:
            p1 = grid.heatmap_cols or list(grid.params.keys())[0]
            p2 = grid.heatmap_rows or list(grid.params.keys())[1]
            heatmap_id = _hash_heatmap(
                {"params": grid.params, "timeframes": list(grid.timeframes), "horizons": list(grid.horizons), "metric": grid.metric}
            )
            fig = plot_param_heatmap(df, value_col=grid.metric, row=p2, col=p1, title=f"{grid.metric} heatmap")
            heatmap_dir = "heatmaps"
            Path(heatmap_dir).mkdir(parents=True, exist_ok=True)
            file_path = str(Path(heatmap_dir) / f"heatmap_{heatmap_id[:8]}.png")
            fig.savefig(file_path)
            plt_close = getattr(__import__("matplotlib.pyplot"), "close", None)
            if plt_close:
                plt_close(fig)
            if self.catalog:
                self.catalog.save_heatmap(heatmap_id, {"params": grid.params, "metric": grid.metric}, file_path, grid.description)
        return df


def run_independent_asset_grid_search(
    *,
    targets: Sequence[IndependentAssetTarget],
    strategy_cls: Type[Strategy],
    base_config: BacktestConfig,
    grid: GridSpec,
    catalog: Optional[ResultCatalog],
    make_heatmap: bool = False,
    stop_cb: Optional[Callable[[], bool]] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    share_batch_id: bool = False,
) -> pd.DataFrame:
    study_id = grid.batch_id or f"study_{uuid.uuid4().hex[:8]}"
    frames: List[pd.DataFrame] = []
    per_batch: List[BatchExecutionBenchmark] = []
    per_asset_total = _estimate_grid_total(grid)
    overall_total = per_asset_total * len(targets)
    completed = 0
    if progress_cb:
        progress_cb(0, overall_total)

    for asset_index, target in enumerate(targets):
        asset_batch_id = study_id if share_batch_id else f"{study_id}_{asset_index + 1:03d}"
        asset_grid = replace(grid, batch_id=asset_batch_id)
        search = GridSearch(
            dataset_id=target.dataset_id,
            data_loader=target.data_loader,
            strategy_cls=strategy_cls,
            base_config=base_config,
            catalog=catalog,
        )
        asset_done = 0

        def _asset_progress(done: int, total: int) -> None:
            nonlocal asset_done
            asset_done = done
            if progress_cb:
                progress_cb(completed + done, overall_total)

        frame = search.run(
            asset_grid,
            make_heatmap=make_heatmap,
            stop_cb=stop_cb,
            progress_cb=_asset_progress if progress_cb else None,
        )
        frames.append(frame)
        per_batch.extend(search.last_batch_benchmarks)
        completed += asset_done or per_asset_total
        if stop_cb and stop_cb():
            break

    if frames:
        combined = pd.concat(frames, ignore_index=True)
    else:
        combined = pd.DataFrame()

    vectorized_batches = sum(1 for batch in per_batch if batch.resolved_execution_mode == ExecutionMode.VECTORIZED)
    reference_batches = sum(1 for batch in per_batch if batch.resolved_execution_mode == ExecutionMode.REFERENCE)
    study_benchmark = IndependentAssetStudyBenchmark(
        study_id=study_id,
        asset_count=len(frames),
        total_runs=len(combined),
        total_duration_seconds=sum(batch.duration_seconds for batch in per_batch),
        vectorized_batches=vectorized_batches,
        reference_batches=reference_batches,
        total_chunk_count=sum(batch.chunk_count for batch in per_batch),
        prepared_context_reused_batches=sum(1 for batch in per_batch if batch.prepared_context_reused),
        per_batch=tuple(per_batch),
    )
    combined.attrs["batch_benchmarks"] = tuple(per_batch)
    combined.attrs["study_benchmark"] = study_benchmark
    return combined


def run_vectorized_portfolio_grid_search(
    *,
    targets: Sequence[PortfolioAssetTarget],
    strategy_cls: Type[Strategy],
    base_config: BacktestConfig,
    grid: GridSpec,
    catalog: Optional[ResultCatalog],
    stop_cb: Optional[Callable[[], bool]] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    normalize_weights: bool = True,
    construction_config: PortfolioConstructionConfig | None = None,
) -> pd.DataFrame:
    if not targets:
        return pd.DataFrame()

    requested_mode = ExecutionMode.from_value(grid.execution_mode)
    if requested_mode == ExecutionMode.REFERENCE:
        raise UnsupportedExecutionModeError(
            "Portfolio study mode currently requires Auto or Vectorized. Reference portfolio execution is not implemented."
        )

    engine = VectorizedPortfolioEngine()
    orchestrator = ExecutionOrchestrator()
    construction = construction_config or PortfolioConstructionConfig()
    records: List[Dict] = []
    per_batch: List[BatchExecutionBenchmark] = []
    tf_list = list(grid.timeframes)
    horizon_list = list(grid.horizons)
    total = _estimate_grid_total(grid)
    if progress_cb:
        progress_cb(0, total)
    done = 0
    dataset_ids = [target.dataset_id for target in targets]
    dataset_label = "Portfolio | " + ", ".join(dataset_ids)
    batch_id = grid.batch_id or f"portfolio_{uuid.uuid4().hex[:8]}"

    helper_engine = VectorizedEngine()

    for timeframe in tf_list:
        loaded_frames = {
            target.dataset_id: target.data_loader(timeframe)
            for target in targets
        }
        param_grid = [dict(zip(grid.params.keys(), combo)) for combo in itertools.product(*grid.params.values())]
        for (start, end) in horizon_list:
            config = replace(
                base_config,
                batch_id=batch_id,
                timeframe=timeframe,
                time_horizon_start=start,
                time_horizon_end=end,
                base_execution=False,
            )
            sliced_frames = []
            for target in targets:
                normalized = helper_engine._normalize_data(loaded_frames[target.dataset_id])
                sliced = helper_engine._slice_data(normalized, config)
                sliced_frames.append((target, sliced))
            if not sliced_frames:
                continue
            common_index = sliced_frames[0][1].index
            for _, frame in sliced_frames[1:]:
                common_index = common_index.intersection(frame.index)
            if common_index.empty:
                raise ValueError("Portfolio study could not align a non-empty common timestamp index.")
            common_start = str(common_index[0])
            common_end = str(common_index[-1])
            bars = len(common_index)

            support = engine.supports(
                [
                    PortfolioAssetSpec(
                        dataset_id=target.dataset_id,
                        data=loaded_frames[target.dataset_id],
                        strategy_cls=strategy_cls,
                        strategy_params=param_grid[0] if param_grid else {},
                        target_weight=target.target_weight,
                    )
                    for target in targets
                ],
                config,
                construction,
            )
            if not support.supported:
                raise UnsupportedExecutionModeError(
                    support.reason or "Portfolio vectorization is not available for this study."
                )

            started = perf_counter()
            cached_runs = 0
            uncached_runs = 0

            for params in param_grid:
                if stop_cb and stop_cb():
                    frame = pd.DataFrame(records)
                    frame.attrs["batch_benchmarks"] = tuple(per_batch)
                    return frame
                portfolio_result = orchestrator.execute_portfolio(
                    PortfolioExecutionRequest(
                        assets=[
                            PortfolioExecutionAsset(
                                dataset_id=target.dataset_id,
                                data=loaded_frames[target.dataset_id],
                                strategy_cls=strategy_cls,
                                strategy_params=params,
                                target_weight=target.target_weight,
                            )
                            for target in targets
                        ],
                        config=config,
                        catalog=catalog,
                        requested_execution_mode=requested_mode,
                        normalize_weights=normalize_weights,
                        portfolio_dataset_id=dataset_label,
                        construction_config=construction,
                    )
                )
                if portfolio_result.cached:
                    cached_runs += 1
                else:
                    uncached_runs += 1
                metrics = portfolio_result.metrics.as_dict()

                records.append(
                    {
                        **params,
                        "dataset_id": dataset_label,
                        "timeframe": timeframe,
                        "start": start,
                        "end": end,
                        "total_return": metrics.get("total_return"),
                        "cagr": metrics.get("cagr"),
                        "max_drawdown": metrics.get("max_drawdown"),
                        "sharpe": metrics.get("sharpe"),
                        "rolling_sharpe": metrics.get("rolling_sharpe"),
                        grid.metric: metrics[grid.metric],
                        "run_id": portfolio_result.run_id,
                        "logical_run_id": portfolio_result.logical_run_id,
                        "requested_execution_mode": portfolio_result.requested_execution_mode.value,
                        "resolved_execution_mode": portfolio_result.resolved_execution_mode.value,
                        "engine_impl": portfolio_result.engine_impl,
                        "engine_version": portfolio_result.engine_version,
                    }
                )
                done += 1
                if progress_cb:
                    progress_cb(done, total)

            per_batch.append(
                BatchExecutionBenchmark(
                    dataset_id=dataset_label,
                    strategy=strategy_cls.__name__,
                    timeframe=timeframe,
                    requested_execution_mode=requested_mode,
                    resolved_execution_mode=ExecutionMode.VECTORIZED,
                    engine_impl=engine.engine_impl,
                    engine_version=engine.engine_version,
                    bars=bars,
                    total_params=len(param_grid),
                    cached_runs=cached_runs,
                    uncached_runs=uncached_runs,
                    duration_seconds=perf_counter() - started,
                    chunk_count=1 if uncached_runs else 0,
                    chunk_sizes=(uncached_runs,) if uncached_runs else (),
                    effective_param_batch_size=uncached_runs if uncached_runs else None,
                    prepared_context_reused=False,
                )
            )

    frame = pd.DataFrame(records)
    frame.attrs["batch_benchmarks"] = tuple(per_batch)
    return frame


def run_vectorized_strategy_block_portfolio_search(
    *,
    strategy_blocks: Sequence[PortfolioStrategyBlockTarget],
    base_config: BacktestConfig,
    grid: GridSpec,
    catalog: Optional[ResultCatalog],
    stop_cb: Optional[Callable[[], bool]] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    normalize_weights: bool = True,
    construction_config: PortfolioConstructionConfig | None = None,
) -> pd.DataFrame:
    if not strategy_blocks:
        return pd.DataFrame()

    requested_mode = ExecutionMode.from_value(grid.execution_mode)
    if requested_mode == ExecutionMode.REFERENCE:
        raise UnsupportedExecutionModeError(
            "Portfolio strategy-block studies currently require Auto or Vectorized. Reference portfolio execution is not implemented."
        )

    engine = VectorizedPortfolioEngine()
    orchestrator = ExecutionOrchestrator()
    construction = construction_config or PortfolioConstructionConfig()
    records: List[Dict] = []
    per_batch: List[BatchExecutionBenchmark] = []
    tf_list = list(grid.timeframes)
    horizon_list = list(grid.horizons)
    total = max(1, len(tf_list) * len(horizon_list))
    if progress_cb:
        progress_cb(0, total)
    done = 0
    dataset_ids = list(
        dict.fromkeys(
            asset.dataset_id
            for block in strategy_blocks
            for asset in block.assets
            if asset.dataset_id
        )
    )
    block_names = [str(block.display_name or block.block_id) for block in strategy_blocks]
    dataset_label = "Portfolio | " + ", ".join(dataset_ids)
    batch_id = grid.batch_id or f"portfolio_blocks_{uuid.uuid4().hex[:8]}"
    helper_engine = VectorizedEngine()
    placeholder_params = {"portfolio_blocks": len(strategy_blocks)}

    for timeframe in tf_list:
        loaded_frames = {
            dataset_id: next(
                asset.data_loader(timeframe)
                for block in strategy_blocks
                for asset in block.assets
                if asset.dataset_id == dataset_id
            )
            for dataset_id in dataset_ids
        }
        for (start, end) in horizon_list:
            config = replace(
                base_config,
                batch_id=batch_id,
                timeframe=timeframe,
                time_horizon_start=start,
                time_horizon_end=end,
                base_execution=False,
            )
            sliced_frames = []
            for block in strategy_blocks:
                for asset in block.assets:
                    normalized = helper_engine._normalize_data(loaded_frames[asset.dataset_id])
                    sliced = helper_engine._slice_data(normalized, config)
                    sliced_frames.append((block, asset, sliced))
            if not sliced_frames:
                continue
            common_index = sliced_frames[0][2].index
            for _, _, frame in sliced_frames[1:]:
                common_index = common_index.intersection(frame.index)
            if common_index.empty:
                raise ValueError("Portfolio strategy-block study could not align a non-empty common timestamp index.")
            bars = len(common_index)
            support = engine.supports_strategy_blocks(
                [
                    PortfolioStrategyBlockSpec(
                        block_id=block.block_id,
                        strategy_cls=block.strategy_cls,
                        strategy_params=block.strategy_params,
                        assets=[
                            PortfolioStrategyBlockAssetSpec(
                                dataset_id=asset.dataset_id,
                                data=loaded_frames[asset.dataset_id],
                                target_weight=asset.target_weight,
                                display_name=asset.display_name,
                            )
                            for asset in block.assets
                        ],
                        budget_weight=block.budget_weight,
                        display_name=block.display_name,
                    )
                    for block in strategy_blocks
                ],
                config,
                construction,
            )
            if not support.supported:
                raise UnsupportedExecutionModeError(
                    support.reason or "Portfolio strategy-block vectorization is not available for this study."
                )

            if stop_cb and stop_cb():
                frame = pd.DataFrame(records)
                frame.attrs["batch_benchmarks"] = tuple(per_batch)
                return frame

            started = perf_counter()
            portfolio_result = orchestrator.execute_portfolio(
                PortfolioExecutionRequest(
                    assets=[],
                    strategy_blocks=[
                        PortfolioExecutionStrategyBlock(
                            block_id=block.block_id,
                            strategy_cls=block.strategy_cls,
                            strategy_params=block.strategy_params,
                            assets=[
                                PortfolioExecutionStrategyBlockAsset(
                                    dataset_id=asset.dataset_id,
                                    data=loaded_frames[asset.dataset_id],
                                    target_weight=asset.target_weight,
                                    display_name=asset.display_name,
                                )
                                for asset in block.assets
                            ],
                            budget_weight=block.budget_weight,
                            display_name=block.display_name,
                        )
                        for block in strategy_blocks
                    ],
                    config=config,
                    catalog=catalog,
                    requested_execution_mode=requested_mode,
                    normalize_weights=normalize_weights,
                    portfolio_dataset_id=dataset_label,
                    construction_config=construction,
                )
            )
            metrics = portfolio_result.metrics.as_dict()
            records.append(
                {
                    **placeholder_params,
                    "dataset_id": dataset_label,
                    "timeframe": timeframe,
                    "start": start,
                    "end": end,
                    "total_return": metrics.get("total_return"),
                    "cagr": metrics.get("cagr"),
                    "max_drawdown": metrics.get("max_drawdown"),
                    "sharpe": metrics.get("sharpe"),
                    "rolling_sharpe": metrics.get("rolling_sharpe"),
                    grid.metric: metrics[grid.metric],
                    "run_id": portfolio_result.run_id,
                    "logical_run_id": portfolio_result.logical_run_id,
                    "requested_execution_mode": portfolio_result.requested_execution_mode.value,
                    "resolved_execution_mode": portfolio_result.resolved_execution_mode.value,
                    "engine_impl": portfolio_result.engine_impl,
                    "engine_version": portfolio_result.engine_version,
                }
            )
            done += 1
            if progress_cb:
                progress_cb(done, total)
            uncached_runs = 0 if portfolio_result.cached else 1
            per_batch.append(
                BatchExecutionBenchmark(
                    dataset_id=dataset_label,
                    strategy="Portfolio Blocks | " + ", ".join(block_names),
                    timeframe=timeframe,
                    requested_execution_mode=requested_mode,
                    resolved_execution_mode=ExecutionMode.VECTORIZED,
                    engine_impl=engine.engine_impl,
                    engine_version=engine.engine_version,
                    bars=bars,
                    total_params=1,
                    cached_runs=1 if portfolio_result.cached else 0,
                    uncached_runs=uncached_runs,
                    duration_seconds=perf_counter() - started,
                    chunk_count=1 if uncached_runs else 0,
                    chunk_sizes=(uncached_runs,) if uncached_runs else (),
                    effective_param_batch_size=uncached_runs if uncached_runs else None,
                    prepared_context_reused=False,
                )
            )

    frame = pd.DataFrame(records)
    frame.attrs["batch_benchmarks"] = tuple(per_batch)
    return frame
