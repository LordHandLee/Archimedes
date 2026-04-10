from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence, Type

import pandas as pd

from .catalog import ResultCatalog
from .engine import BacktestConfig, BacktestResult
from .run_ids import compute_engine_run_id, compute_portfolio_logical_run_id
from .strategy import Strategy


class ExecutionMode(str, Enum):
    AUTO = "auto"
    REFERENCE = "reference"
    VECTORIZED = "vectorized"

    @classmethod
    def from_value(cls, value: "ExecutionMode | str") -> "ExecutionMode":
        if isinstance(value, cls):
            return value
        return cls(str(value).lower())


class WorkloadType(str, Enum):
    SINGLE_RUN = "single_run"
    OPTIMIZATION_BATCH = "optimization_batch"
    WALK_FORWARD_FOLD = "walk_forward_fold"

    @classmethod
    def from_value(cls, value: "WorkloadType | str") -> "WorkloadType":
        if isinstance(value, cls):
            return value
        return cls(str(value).lower())


class UnsupportedExecutionModeError(RuntimeError):
    """Raised when the requested execution mode is not available."""


@dataclass
class ExecutionRequest:
    data: pd.DataFrame
    dataset_id: str
    strategy_cls: Type[Strategy]
    strategy_params: Dict
    config: BacktestConfig
    catalog: Optional[ResultCatalog] = None
    base_data: Optional[pd.DataFrame] = None
    requested_execution_mode: ExecutionMode | str = ExecutionMode.AUTO
    workload_type: WorkloadType | str = WorkloadType.SINGLE_RUN
    logical_run_id: str | None = None


@dataclass(frozen=True)
class PortfolioExecutionAsset:
    dataset_id: str
    data: pd.DataFrame
    strategy_cls: Type[Strategy]
    strategy_params: Dict[str, Any]
    target_weight: float | None = None
    display_name: str | None = None


@dataclass(frozen=True)
class PortfolioExecutionStrategyBlockAsset:
    dataset_id: str
    data: pd.DataFrame
    target_weight: float | None = None
    display_name: str | None = None


@dataclass(frozen=True)
class PortfolioExecutionStrategyBlock:
    block_id: str
    strategy_cls: Type[Strategy]
    strategy_params: Dict[str, Any]
    assets: Sequence[PortfolioExecutionStrategyBlockAsset]
    budget_weight: float | None = None
    display_name: str | None = None


@dataclass
class PortfolioExecutionRequest:
    assets: Sequence[PortfolioExecutionAsset]
    config: BacktestConfig
    strategy_blocks: Sequence[PortfolioExecutionStrategyBlock] = field(default_factory=tuple)
    catalog: Optional[ResultCatalog] = None
    requested_execution_mode: ExecutionMode | str = ExecutionMode.AUTO
    logical_run_id: str | None = None
    normalize_weights: bool = True
    portfolio_dataset_id: str | None = None
    construction_config: Any = None


@dataclass
class ExecutionResult:
    result: BacktestResult
    logical_run_id: str
    requested_execution_mode: ExecutionMode
    resolved_execution_mode: ExecutionMode
    engine_impl: str
    engine_version: str
    fallback_reason: str | None = None

    @property
    def run_id(self) -> str:
        return self.result.run_id

    @property
    def equity_curve(self):
        return self.result.equity_curve

    @property
    def trades(self):
        return self.result.trades

    @property
    def metrics(self):
        return self.result.metrics

    @property
    def cached(self) -> bool:
        return self.result.cached

    def to_dict(self) -> Dict:
        payload = self.result.to_dict()
        payload.update(
            {
                "logical_run_id": self.logical_run_id,
                "requested_execution_mode": self.requested_execution_mode.value,
                "resolved_execution_mode": self.resolved_execution_mode.value,
                "engine_impl": self.engine_impl,
                "engine_version": self.engine_version,
                "fallback_reason": self.fallback_reason,
            }
        )
        return payload


@dataclass
class PortfolioExecutionResult:
    run_id: str
    result: object
    logical_run_id: str
    dataset_id: str
    requested_execution_mode: ExecutionMode
    resolved_execution_mode: ExecutionMode
    engine_impl: str
    engine_version: str
    fallback_reason: str | None = None
    cached: bool = False

    @property
    def equity_curve(self):
        return self.result.portfolio_equity_curve

    @property
    def trades(self):
        return self.result.trades

    @property
    def metrics(self):
        return self.result.metrics

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "logical_run_id": self.logical_run_id,
            "dataset_id": self.dataset_id,
            "metrics": self.result.metrics.as_dict(),
            "equity_curve_len": len(self.result.portfolio_equity_curve),
            "trade_count": len(self.result.trades),
            "cached": self.cached,
            "requested_execution_mode": self.requested_execution_mode.value,
            "resolved_execution_mode": self.resolved_execution_mode.value,
            "engine_impl": self.engine_impl,
            "engine_version": self.engine_version,
            "fallback_reason": self.fallback_reason,
        }


@dataclass
class ExecutionResolution:
    requested_mode: ExecutionMode
    resolved_mode: ExecutionMode
    fallback_reason: str | None = None
    effective_config: BacktestConfig | None = None
    effective_base_data: pd.DataFrame | None = None


@dataclass(frozen=True)
class BatchExecutionBenchmark:
    dataset_id: str
    strategy: str
    timeframe: str
    requested_execution_mode: ExecutionMode
    resolved_execution_mode: ExecutionMode
    engine_impl: str
    engine_version: str
    bars: int
    total_params: int
    cached_runs: int
    uncached_runs: int
    duration_seconds: float
    chunk_count: int = 0
    chunk_sizes: tuple[int, ...] = ()
    effective_param_batch_size: int | None = None
    prepared_context_reused: bool = False


@dataclass
class ExecutionOrchestrator:
    """
    Stable execution entry point for higher-level workflows.

    Phase 1 only wires in the reference engine. The vectorized path will plug
    into the same contract later.
    """

    _reference_engine: object | None = field(default=None, init=False, repr=False)
    _vectorized_engine: object | None = field(default=None, init=False, repr=False)
    _vectorized_portfolio_engine: object | None = field(default=None, init=False, repr=False)
    _last_batch_benchmark: BatchExecutionBenchmark | None = field(default=None, init=False, repr=False)

    def resolve(self, request: ExecutionRequest) -> ExecutionResolution:
        requested_mode = ExecutionMode.from_value(request.requested_execution_mode)
        effective_config = request.config
        effective_base_data = request.base_data
        vectorized_support = self._get_vectorized_engine().supports(request)
        if requested_mode == ExecutionMode.VECTORIZED:
            if not vectorized_support.supported:
                raise UnsupportedExecutionModeError(
                    vectorized_support.reason or "Vectorized execution is not available for this request."
                )
            return ExecutionResolution(
                requested_mode=requested_mode,
                resolved_mode=ExecutionMode.VECTORIZED,
                fallback_reason=None,
                effective_config=effective_config,
                effective_base_data=effective_base_data,
            )
        if requested_mode == ExecutionMode.AUTO:
            if vectorized_support.supported:
                return ExecutionResolution(
                    requested_mode=requested_mode,
                    resolved_mode=ExecutionMode.VECTORIZED,
                    fallback_reason=None,
                    effective_config=effective_config,
                    effective_base_data=effective_base_data,
                )
            auto_candidate = self._auto_same_timeframe_candidate(request)
            if auto_candidate is not None:
                auto_support = self._get_vectorized_engine().supports(auto_candidate)
                if auto_support.supported:
                    return ExecutionResolution(
                        requested_mode=requested_mode,
                        resolved_mode=ExecutionMode.VECTORIZED,
                        fallback_reason=None,
                        effective_config=auto_candidate.config,
                        effective_base_data=auto_candidate.base_data,
                    )
            return ExecutionResolution(
                requested_mode=requested_mode,
                resolved_mode=ExecutionMode.REFERENCE,
                fallback_reason=vectorized_support.reason or "Vectorized execution is not available for this request.",
                effective_config=effective_config,
                effective_base_data=effective_base_data,
            )
        return ExecutionResolution(
            requested_mode=requested_mode,
            resolved_mode=ExecutionMode.REFERENCE,
            fallback_reason=None,
            effective_config=effective_config,
            effective_base_data=effective_base_data,
        )

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        resolution = self.resolve(request)
        effective_request = self._apply_resolution(request, resolution)
        if resolution.resolved_mode == ExecutionMode.VECTORIZED:
            return self._get_vectorized_engine().execute(
                effective_request,
                requested_mode=resolution.requested_mode,
                resolved_mode=resolution.resolved_mode,
                fallback_reason=resolution.fallback_reason,
            )
        if resolution.resolved_mode == ExecutionMode.REFERENCE:
            return self._get_reference_engine().execute(
                effective_request,
                requested_mode=resolution.requested_mode,
                resolved_mode=resolution.resolved_mode,
                fallback_reason=resolution.fallback_reason,
            )
        raise UnsupportedExecutionModeError(
            f"Unsupported resolved execution mode: {resolution.resolved_mode.value}"
        )

    def resolve_portfolio(self, request: PortfolioExecutionRequest) -> ExecutionResolution:
        requested_mode = ExecutionMode.from_value(request.requested_execution_mode)
        construction_config = self._normalize_portfolio_construction_config(request.construction_config)
        strategy_blocks = self._normalize_portfolio_strategy_blocks(request)
        support = self._get_vectorized_portfolio_engine().supports_strategy_blocks(
            [self._to_portfolio_strategy_block_spec(block) for block in strategy_blocks],
            request.config,
            construction_config,
        )
        if requested_mode == ExecutionMode.REFERENCE:
            raise UnsupportedExecutionModeError(
                "Reference portfolio execution is not implemented. Portfolio runs currently require Auto or Vectorized."
            )
        if not support.supported:
            if requested_mode == ExecutionMode.AUTO:
                raise UnsupportedExecutionModeError(
                    support.reason or "Portfolio vectorized execution is not available for this request."
                )
            raise UnsupportedExecutionModeError(
                support.reason or "Portfolio vectorized execution is not available for this request."
            )
        return ExecutionResolution(
            requested_mode=requested_mode,
            resolved_mode=ExecutionMode.VECTORIZED,
            fallback_reason=None,
            effective_config=request.config,
            effective_base_data=None,
        )

    def execute_portfolio(self, request: PortfolioExecutionRequest) -> PortfolioExecutionResult:
        resolution = self.resolve_portfolio(request)
        engine = self._get_vectorized_portfolio_engine()
        construction_config = self._normalize_portfolio_construction_config(request.construction_config)
        strategy_blocks = self._normalize_portfolio_strategy_blocks(request)
        flattened_assets = self._flatten_portfolio_strategy_blocks(strategy_blocks)
        dataset_label = request.portfolio_dataset_id or self._portfolio_dataset_label(flattened_assets)
        start_bound, end_bound = self._portfolio_index_bounds(flattened_assets, request.config)
        identity_payload = {
            "assets": [
                {
                    "dataset_id": asset.dataset_id,
                    "strategy": asset.strategy_cls.__name__,
                    "params": asset.strategy_params,
                    "target_weight": asset.target_weight,
                    "display_name": asset.display_name,
                }
                for asset in flattened_assets
            ],
            "strategy_blocks": [
                {
                    "block_id": block.block_id,
                    "display_name": block.display_name,
                    "budget_weight": block.budget_weight,
                    "strategy": block.strategy_cls.__name__,
                    "params": block.strategy_params,
                    "assets": [
                        {
                            "dataset_id": asset.dataset_id,
                            "target_weight": asset.target_weight,
                            "display_name": asset.display_name,
                        }
                        for asset in block.assets
                    ],
                }
                for block in strategy_blocks
            ],
            "normalize_weights": request.normalize_weights,
            "construction_config": {
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
            },
        }
        stored_params = {
            **identity_payload,
            "execution_config": {
                "starting_cash": request.config.starting_cash,
                "fee_rate": request.config.fee_rate,
                "fee_schedule": request.config.fee_schedule,
                "slippage": request.config.slippage,
                "slippage_schedule": request.config.slippage_schedule,
                "borrow_rate": request.config.borrow_rate,
                "fill_ratio": request.config.fill_ratio,
                "fill_on_close": request.config.fill_on_close,
                "recalc_on_fill": request.config.recalc_on_fill,
                "allow_short": request.config.allow_short,
                "prevent_scale_in": request.config.prevent_scale_in,
                "one_order_per_signal": request.config.one_order_per_signal,
                "risk_free_rate": request.config.risk_free_rate,
            },
        }
        logical_run_id = request.logical_run_id or compute_portfolio_logical_run_id(
            dataset_ids=[asset.dataset_id for asset in flattened_assets],
            strategy="PortfolioExecution",
            params=identity_payload,
            config=request.config,
            start=str(start_bound),
            end=str(end_bound),
            normalize_weights=request.normalize_weights,
            max_gross_exposure=engine.max_gross_exposure,
        )
        run_id = compute_engine_run_id(logical_run_id, engine.engine_impl, engine.engine_version)

        cached = request.catalog.fetch(run_id) if request.config.use_cache and request.catalog else None
        if cached:
            empty_index = pd.DatetimeIndex([], tz="UTC")
            empty_series = pd.Series(dtype=float, index=empty_index, name="portfolio_equity")
            empty_frame = pd.DataFrame(index=empty_index)
            from .vectorized_portfolio import VectorizedPortfolioResult

            result = VectorizedPortfolioResult(
                portfolio_equity_curve=empty_series,
                asset_market_values=empty_frame,
                asset_weights=empty_frame.copy(),
                target_weights=empty_frame.copy(),
                strategy_market_values=empty_frame.copy(),
                strategy_weights=empty_frame.copy(),
                strategy_target_weights=empty_frame.copy(),
                positions=empty_frame.copy(),
                cash_curve=pd.Series(dtype=float, index=empty_index, name="cash"),
                trades=[],
                metrics=cached.metrics,
                asset_to_strategy_block={},
                asset_source_dataset_ids={},
                strategy_display_names={},
                strategy_budget_weights={},
            )
            return PortfolioExecutionResult(
                run_id=run_id,
                result=result,
                logical_run_id=logical_run_id,
                dataset_id=dataset_label,
                requested_execution_mode=resolution.requested_mode,
                resolved_execution_mode=resolution.resolved_mode,
                engine_impl=engine.engine_impl,
                engine_version=engine.engine_version,
                fallback_reason=resolution.fallback_reason,
                cached=True,
            )

        from .vectorized_portfolio import PortfolioStrategyBlockSpec

        run_started_at = pd.Timestamp.now("UTC").isoformat()
        result = engine.run_strategy_blocks(
            [
                PortfolioStrategyBlockSpec(
                    block_id=block.block_id,
                    strategy_cls=block.strategy_cls,
                    strategy_params=block.strategy_params,
                    assets=[
                        self._to_portfolio_strategy_block_asset_spec(asset)
                        for asset in block.assets
                    ],
                    budget_weight=block.budget_weight,
                    display_name=block.display_name,
                )
                for block in strategy_blocks
            ],
            request.config,
            normalize_weights=request.normalize_weights,
            construction_config=construction_config,
        )
        run_finished_at = pd.Timestamp.now("UTC").isoformat()
        if request.catalog:
            request.catalog.save(
                run_id=run_id,
                batch_id=request.config.batch_id,
                strategy="PortfolioExecution",
                params=stored_params,
                timeframe=request.config.timeframe,
                start=str(result.portfolio_equity_curve.index[0]),
                end=str(result.portfolio_equity_curve.index[-1]),
                dataset_id=dataset_label,
                starting_cash=request.config.starting_cash,
                metrics=result.metrics,
                run_started_at=run_started_at,
                run_finished_at=run_finished_at,
                status="finished",
                logical_run_id=logical_run_id,
                requested_execution_mode=resolution.requested_mode.value,
                resolved_execution_mode=resolution.resolved_mode.value,
                engine_impl=engine.engine_impl,
                engine_version=engine.engine_version,
                fallback_reason=resolution.fallback_reason,
            )
            request.catalog.save_trades(run_id, result.trades)
        return PortfolioExecutionResult(
            run_id=run_id,
            result=result,
            logical_run_id=logical_run_id,
            dataset_id=dataset_label,
            requested_execution_mode=resolution.requested_mode,
            resolved_execution_mode=resolution.resolved_mode,
            engine_impl=engine.engine_impl,
            engine_version=engine.engine_version,
            fallback_reason=resolution.fallback_reason,
            cached=False,
        )

    def execute_param_grid(
        self,
        *,
        data: pd.DataFrame,
        dataset_id: str,
        strategy_cls: Type[Strategy],
        param_grid: Sequence[Dict],
        config: BacktestConfig,
        catalog: Optional[ResultCatalog] = None,
        base_data: Optional[pd.DataFrame] = None,
        requested_execution_mode: ExecutionMode | str = ExecutionMode.AUTO,
        workload_type: WorkloadType | str = WorkloadType.OPTIMIZATION_BATCH,
    ) -> List[ExecutionResult]:
        started = perf_counter()
        requested_mode = ExecutionMode.from_value(requested_execution_mode)
        resolution = self.resolve_param_grid(
            data=data,
            dataset_id=dataset_id,
            strategy_cls=strategy_cls,
            param_grid=param_grid,
            config=config,
            requested_execution_mode=requested_mode,
            base_data=base_data,
        )
        effective_config = resolution.effective_config or config
        effective_base_data = resolution.effective_base_data
        if resolution.resolved_mode == ExecutionMode.VECTORIZED:
            results = self._get_vectorized_engine().execute_param_grid(
                data=data,
                dataset_id=dataset_id,
                strategy_cls=strategy_cls,
                param_grid=param_grid,
                catalog=catalog,
                config=effective_config,
                base_data=effective_base_data,
                requested_mode=requested_mode,
                resolved_mode=ExecutionMode.VECTORIZED,
                fallback_reason=resolution.fallback_reason,
            )
            self._last_batch_benchmark = self._get_vectorized_engine().last_batch_benchmark
            return results
        results: List[ExecutionResult] = []
        workload = WorkloadType.from_value(workload_type)
        for params in param_grid:
            results.append(
                self._get_reference_engine().execute(
                    ExecutionRequest(
                        data=data,
                        dataset_id=dataset_id,
                        strategy_cls=strategy_cls,
                        strategy_params=params,
                        config=effective_config,
                        catalog=catalog,
                        base_data=effective_base_data,
                        requested_execution_mode=requested_mode,
                        workload_type=workload,
                    ),
                    requested_mode=requested_mode,
                    resolved_mode=ExecutionMode.REFERENCE,
                    fallback_reason=resolution.fallback_reason,
                )
            )
        duration_seconds = perf_counter() - started
        cached_runs = sum(1 for result in results if result.cached)
        self._last_batch_benchmark = BatchExecutionBenchmark(
            dataset_id=dataset_id,
            strategy=strategy_cls.__name__,
            timeframe=effective_config.timeframe,
            requested_execution_mode=requested_mode,
            resolved_execution_mode=ExecutionMode.REFERENCE,
            engine_impl="reference",
            engine_version="1",
            bars=len(data),
            total_params=len(param_grid),
            cached_runs=cached_runs,
            uncached_runs=len(param_grid) - cached_runs,
            duration_seconds=duration_seconds,
        )
        return results

    @property
    def last_batch_benchmark(self) -> BatchExecutionBenchmark | None:
        return self._last_batch_benchmark

    def resolve_param_grid(
        self,
        *,
        data: pd.DataFrame,
        dataset_id: str,
        strategy_cls: Type[Strategy],
        param_grid: Sequence[Dict],
        config: BacktestConfig,
        requested_execution_mode: ExecutionMode | str = ExecutionMode.AUTO,
        base_data: Optional[pd.DataFrame] = None,
    ) -> ExecutionResolution:
        requested_mode = ExecutionMode.from_value(requested_execution_mode)
        support = self._get_vectorized_engine().supports_param_grid(
            data=data,
            dataset_id=dataset_id,
            strategy_cls=strategy_cls,
            param_grid=param_grid,
            config=config,
            base_data=base_data,
        )
        if requested_mode == ExecutionMode.VECTORIZED:
            if not support.supported:
                raise UnsupportedExecutionModeError(
                    support.reason or "Vectorized execution is not available for this parameter grid."
                )
            return ExecutionResolution(
                requested_mode=requested_mode,
                resolved_mode=ExecutionMode.VECTORIZED,
                fallback_reason=None,
                effective_config=config,
                effective_base_data=base_data,
            )
        if requested_mode == ExecutionMode.AUTO:
            if support.supported:
                return ExecutionResolution(
                    requested_mode=requested_mode,
                    resolved_mode=ExecutionMode.VECTORIZED,
                    fallback_reason=None,
                    effective_config=config,
                    effective_base_data=base_data,
                )
            auto_candidate = self._auto_same_timeframe_grid_candidate(
                data=data,
                dataset_id=dataset_id,
                strategy_cls=strategy_cls,
                param_grid=param_grid,
                config=config,
            )
            if auto_candidate is not None:
                return ExecutionResolution(
                    requested_mode=requested_mode,
                    resolved_mode=ExecutionMode.VECTORIZED,
                    fallback_reason=None,
                    effective_config=auto_candidate,
                    effective_base_data=None,
                )
            return ExecutionResolution(
                requested_mode=requested_mode,
                resolved_mode=ExecutionMode.REFERENCE,
                fallback_reason=support.reason or "Vectorized execution is not available for this parameter grid.",
                effective_config=config,
                effective_base_data=base_data,
            )
        return ExecutionResolution(
            requested_mode=requested_mode,
            resolved_mode=ExecutionMode.REFERENCE,
            fallback_reason=None,
            effective_config=config,
            effective_base_data=base_data,
        )

    def _get_reference_engine(self):
        if self._reference_engine is None:
            from .reference_engine import ReferenceEngine

            self._reference_engine = ReferenceEngine()
        return self._reference_engine

    def _get_vectorized_engine(self):
        if self._vectorized_engine is None:
            from .vectorized_engine import VectorizedEngine

            self._vectorized_engine = VectorizedEngine()
        return self._vectorized_engine

    def _get_vectorized_portfolio_engine(self):
        if self._vectorized_portfolio_engine is None:
            from .vectorized_portfolio import VectorizedPortfolioEngine

            self._vectorized_portfolio_engine = VectorizedPortfolioEngine()
        return self._vectorized_portfolio_engine

    def _apply_resolution(self, request: ExecutionRequest, resolution: ExecutionResolution) -> ExecutionRequest:
        config = resolution.effective_config or request.config
        base_data = resolution.effective_base_data if resolution.effective_config is not None else request.base_data
        if config is request.config and base_data is request.base_data:
            return request
        return replace(request, config=config, base_data=base_data)

    def _auto_same_timeframe_candidate(self, request: ExecutionRequest) -> ExecutionRequest | None:
        if not request.config.base_execution:
            return None
        candidate_config = replace(request.config, base_execution=False)
        return replace(request, config=candidate_config, base_data=None)

    def _auto_same_timeframe_grid_candidate(
        self,
        *,
        data: pd.DataFrame,
        dataset_id: str,
        strategy_cls: Type[Strategy],
        param_grid: Sequence[Dict],
        config: BacktestConfig,
    ) -> BacktestConfig | None:
        if not config.base_execution:
            return None
        candidate_config = replace(config, base_execution=False)
        support = self._get_vectorized_engine().supports_param_grid(
            data=data,
            dataset_id=dataset_id,
            strategy_cls=strategy_cls,
            param_grid=param_grid,
            config=candidate_config,
            base_data=None,
        )
        if not support.supported:
            return None
        return candidate_config

    @staticmethod
    def _portfolio_dataset_label(assets: Sequence[PortfolioExecutionAsset]) -> str:
        dataset_ids = [asset.dataset_id for asset in assets if asset.dataset_id]
        return "Portfolio | " + ", ".join(dataset_ids)

    def _portfolio_index_bounds(
        self,
        assets: Sequence[PortfolioExecutionAsset],
        config: BacktestConfig,
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        helper = self._get_vectorized_engine()
        common_index: pd.DatetimeIndex | None = None
        for asset in assets:
            normalized = helper._normalize_data(asset.data)
            sliced = helper._slice_data(normalized, config)
            if common_index is None:
                common_index = sliced.index
            else:
                common_index = common_index.intersection(sliced.index)
        if common_index is None or common_index.empty:
            raise ValueError("Portfolio execution could not align a non-empty common timestamp index.")
        return common_index[0], common_index[-1]

    @staticmethod
    def _to_portfolio_asset_spec(asset: PortfolioExecutionAsset):
        from .vectorized_portfolio import PortfolioAssetSpec

        return PortfolioAssetSpec(
            dataset_id=asset.dataset_id,
            data=asset.data,
            strategy_cls=asset.strategy_cls,
            strategy_params=asset.strategy_params,
            target_weight=asset.target_weight,
            display_name=asset.display_name,
        )

    @staticmethod
    def _to_portfolio_strategy_block_asset_spec(asset: PortfolioExecutionStrategyBlockAsset):
        from .vectorized_portfolio import PortfolioStrategyBlockAssetSpec

        return PortfolioStrategyBlockAssetSpec(
            dataset_id=asset.dataset_id,
            data=asset.data,
            target_weight=asset.target_weight,
            display_name=asset.display_name,
        )

    @staticmethod
    def _to_portfolio_strategy_block_spec(block: PortfolioExecutionStrategyBlock):
        from .vectorized_portfolio import PortfolioStrategyBlockSpec

        return PortfolioStrategyBlockSpec(
            block_id=block.block_id,
            strategy_cls=block.strategy_cls,
            strategy_params=block.strategy_params,
            assets=[
                ExecutionOrchestrator._to_portfolio_strategy_block_asset_spec(asset)
                for asset in block.assets
            ],
            budget_weight=block.budget_weight,
            display_name=block.display_name,
        )

    def _normalize_portfolio_strategy_blocks(
        self,
        request: PortfolioExecutionRequest,
    ) -> list[PortfolioExecutionStrategyBlock]:
        if request.strategy_blocks:
            return list(request.strategy_blocks)
        if not request.assets:
            raise UnsupportedExecutionModeError(
                "Portfolio execution requires assets or strategy_blocks to be defined."
            )
        blocks: list[PortfolioExecutionStrategyBlock] = []
        block_lookup: dict[tuple[str, tuple[tuple[str, str], ...]], PortfolioExecutionStrategyBlock] = {}
        block_counts: dict[str, int] = {}
        for asset in request.assets:
            signature = (
                asset.strategy_cls.__name__,
                tuple(sorted((str(key), repr(value)) for key, value in dict(asset.strategy_params).items())),
            )
            existing = block_lookup.get(signature)
            if existing is None:
                base = asset.strategy_cls.__name__
                block_counts[base] = block_counts.get(base, 0) + 1
                block_id = base if block_counts[base] == 1 else f"{base}_{block_counts[base]}"
                existing = PortfolioExecutionStrategyBlock(
                    block_id=block_id,
                    strategy_cls=asset.strategy_cls,
                    strategy_params=dict(asset.strategy_params),
                    assets=tuple(),
                    budget_weight=None,
                    display_name=base,
                )
                block_lookup[signature] = existing
                blocks.append(existing)
            updated_assets = list(existing.assets)
            updated_assets.append(
                PortfolioExecutionStrategyBlockAsset(
                    dataset_id=asset.dataset_id,
                    data=asset.data,
                    target_weight=asset.target_weight,
                    display_name=asset.display_name,
                )
            )
            updated_block = replace(existing, assets=tuple(updated_assets))
            block_lookup[signature] = updated_block
            blocks[blocks.index(existing)] = updated_block
        return blocks

    @staticmethod
    def _flatten_portfolio_strategy_blocks(
        strategy_blocks: Sequence[PortfolioExecutionStrategyBlock],
    ) -> list[PortfolioExecutionAsset]:
        assets: list[PortfolioExecutionAsset] = []
        for block in strategy_blocks:
            for asset in block.assets:
                assets.append(
                    PortfolioExecutionAsset(
                        dataset_id=asset.dataset_id,
                        data=asset.data,
                        strategy_cls=block.strategy_cls,
                        strategy_params=dict(block.strategy_params),
                        target_weight=asset.target_weight,
                        display_name=asset.display_name,
                    )
                )
        return assets

    @staticmethod
    def _normalize_portfolio_construction_config(construction_config: Any):
        from .vectorized_portfolio import PortfolioConstructionConfig

        if isinstance(construction_config, PortfolioConstructionConfig):
            return construction_config
        if construction_config is None:
            return PortfolioConstructionConfig()
        if isinstance(construction_config, dict):
            return PortfolioConstructionConfig(**construction_config)
        raise TypeError("construction_config must be a PortfolioConstructionConfig, dict, or None.")
