from __future__ import annotations

from dataclasses import dataclass

from .engine import BacktestEngine
from .execution import ExecutionMode, ExecutionRequest, ExecutionResult


@dataclass(frozen=True)
class ReferenceEngine:
    """
    Thin adapter around the existing event-driven BacktestEngine.
    """

    engine_impl: str = "reference"
    engine_version: str = "1"

    def execute(
        self,
        request: ExecutionRequest,
        *,
        requested_mode: ExecutionMode,
        resolved_mode: ExecutionMode,
        fallback_reason: str | None = None,
    ) -> ExecutionResult:
        engine = BacktestEngine(
            data=request.data,
            dataset_id=request.dataset_id,
            strategy_cls=request.strategy_cls,
            catalog=request.catalog,
            config=request.config,
            base_data=request.base_data,
            execution_metadata={
                "logical_run_id": request.logical_run_id,
                "requested_execution_mode": requested_mode.value,
                "resolved_execution_mode": resolved_mode.value,
                "engine_impl": self.engine_impl,
                "engine_version": self.engine_version,
                "fallback_reason": fallback_reason,
            },
        )
        result = engine.run(request.strategy_params)
        logical_run_id = request.logical_run_id or result.run_id
        return ExecutionResult(
            result=result,
            logical_run_id=logical_run_id,
            requested_execution_mode=requested_mode,
            resolved_execution_mode=resolved_mode,
            engine_impl=self.engine_impl,
            engine_version=self.engine_version,
            fallback_reason=fallback_reason,
        )
