from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Type

import numpy as np
import pandas as pd

from .engine import BacktestConfig
from .sample_strategies import compute_zscore_mean_reversion_features
from .strategy import Strategy

ORDER_TYPE_MARKET = 0
ORDER_TYPE_LIMIT = 1
ORDER_TYPE_STOP = 2


@dataclass(frozen=True)
class VectorizedSupport:
    supported: bool
    reason: str | None = None


@dataclass(frozen=True)
class VectorizedOrderPlan:
    enter_long: np.ndarray
    exit_long: np.ndarray
    enter_short: np.ndarray
    exit_short: np.ndarray
    cancel_pending: np.ndarray
    enter_long_order_type: np.ndarray
    exit_long_order_type: np.ndarray
    enter_short_order_type: np.ndarray
    exit_short_order_type: np.ndarray
    enter_long_price: np.ndarray
    exit_long_price: np.ndarray
    enter_short_price: np.ndarray
    exit_short_price: np.ndarray
    long_targets: np.ndarray
    short_targets: np.ndarray


@dataclass(frozen=True)
class VectorizedPendingOrderUpdate:
    cancel_pending: np.ndarray
    submit_order: np.ndarray
    order_qty: np.ndarray
    order_type: np.ndarray
    order_price: np.ndarray


class VectorizedStrategyAdapter:
    strategy_name: str = ""

    def supports(self, config: BacktestConfig, strategy_cls: Type[Strategy]) -> VectorizedSupport:
        raise NotImplementedError

    def build_order_plan(self, data: pd.DataFrame, param_grid: List[Dict], config: BacktestConfig) -> VectorizedOrderPlan:
        raise NotImplementedError

    def prepare_order_plan_context(
        self,
        data: pd.DataFrame,
        param_grid: List[Dict],
        config: BacktestConfig,
    ) -> Any:
        return None

    def build_order_plan_from_context(
        self,
        data: pd.DataFrame,
        param_grid: List[Dict],
        config: BacktestConfig,
        prepared_context: Any = None,
    ) -> VectorizedOrderPlan:
        return self.build_order_plan(data, param_grid, config)

    def supports_after_fill(self) -> bool:
        return False

    def build_after_fill_update(
        self,
        *,
        data: pd.DataFrame,
        bar_idx: int,
        param_grid: List[Dict],
        config: BacktestConfig,
        filled_mask: np.ndarray,
        cash: np.ndarray,
        position: np.ndarray,
        avg_price: np.ndarray,
        pending_qty: np.ndarray,
        pending_order_type: np.ndarray,
        pending_order_price: np.ndarray,
        pending_order_active: np.ndarray,
    ) -> VectorizedPendingOrderUpdate | None:
        return None


@dataclass(frozen=True)
class SMACrossVectorizedAdapter(VectorizedStrategyAdapter):
    strategy_name: str = "SMACrossStrategy"

    def supports(self, config: BacktestConfig, strategy_cls: Type[Strategy]) -> VectorizedSupport:
        if strategy_cls.__name__ != self.strategy_name:
            return VectorizedSupport(False, f"Strategy {strategy_cls.__name__} is not handled by the SMA adapter.")
        if config.base_execution:
            return VectorizedSupport(False, "Vectorized v1 does not support separate signal/base execution timeframes.")
        if config.intrabar_sim:
            return VectorizedSupport(False, "Vectorized SMA v1 does not support intrabar simulation.")
        return VectorizedSupport(True)

    def prepare_order_plan_context(
        self,
        data: pd.DataFrame,
        param_grid: List[Dict],
        config: BacktestConfig,
    ) -> Any:
        close = data["close"]
        unique_fast = sorted({int(params.get("fast", 10)) for params in param_grid})
        unique_slow = sorted({int(params.get("slow", 30)) for params in param_grid})
        return {
            "rolling_fast": {window: close.rolling(window).mean().to_numpy(dtype=float) for window in unique_fast},
            "rolling_slow": {window: close.rolling(window).mean().to_numpy(dtype=float) for window in unique_slow},
        }

    def build_order_plan(self, data: pd.DataFrame, param_grid: List[Dict], config: BacktestConfig) -> VectorizedOrderPlan:
        return self.build_order_plan_from_context(data, param_grid, config, prepared_context=None)

    def build_order_plan_from_context(
        self,
        data: pd.DataFrame,
        param_grid: List[Dict],
        config: BacktestConfig,
        prepared_context: Any = None,
    ) -> VectorizedOrderPlan:
        if not param_grid:
            empty_bool = np.empty((len(data), 0), dtype=bool)
            empty_float = np.empty(0, dtype=float)
            empty_types = np.empty((len(data), 0), dtype=np.int8)
            empty_prices = np.empty((len(data), 0), dtype=float)
            return VectorizedOrderPlan(
                empty_bool,
                empty_bool,
                empty_bool,
                empty_bool,
                empty_bool,
                empty_types,
                empty_types,
                empty_types,
                empty_types,
                empty_prices,
                empty_prices,
                empty_prices,
                empty_prices,
                empty_float,
                empty_float,
            )
        if prepared_context is None:
            prepared_context = self.prepare_order_plan_context(data, param_grid, config)
        rolling_fast = prepared_context["rolling_fast"]
        rolling_slow = prepared_context["rolling_slow"]
        enter_long_cols = []
        exit_long_cols = []
        long_targets = []
        for params in param_grid:
            fast = int(params.get("fast", 10))
            slow = int(params.get("slow", 30))
            target = float(params.get("target", 1.0))
            if fast >= slow:
                raise ValueError("SMACrossStrategy requires fast < slow.")
            if target < 0:
                raise ValueError("Current vectorized SMA adapter expects target >= 0.")
            fast_arr = rolling_fast[fast]
            slow_arr = rolling_slow[slow]
            valid = ~np.isnan(fast_arr) & ~np.isnan(slow_arr)
            bullish = (fast_arr > slow_arr) & valid
            bearish = (fast_arr < slow_arr) & valid
            enter_long_cols.append(bullish)
            exit_long_cols.append(bearish)
            long_targets.append(target)
        enter_long = np.column_stack(enter_long_cols).astype(bool, copy=False)
        exit_long = np.column_stack(exit_long_cols).astype(bool, copy=False)
        empty_short = np.zeros_like(enter_long, dtype=bool)
        market_types = np.full_like(enter_long, ORDER_TYPE_MARKET, dtype=np.int8)
        nan_prices = np.full(enter_long.shape, np.nan, dtype=float)
        return VectorizedOrderPlan(
            enter_long=enter_long,
            exit_long=exit_long,
            enter_short=empty_short,
            exit_short=empty_short.copy(),
            cancel_pending=np.zeros_like(enter_long, dtype=bool),
            enter_long_order_type=market_types,
            exit_long_order_type=market_types.copy(),
            enter_short_order_type=market_types.copy(),
            exit_short_order_type=market_types.copy(),
            enter_long_price=nan_prices,
            exit_long_price=nan_prices.copy(),
            enter_short_price=nan_prices.copy(),
            exit_short_price=nan_prices.copy(),
            long_targets=np.array(long_targets, dtype=float),
            short_targets=np.zeros(len(param_grid), dtype=float),
        )


def _supports_long_only_v1(config: BacktestConfig, strategy_name: str) -> VectorizedSupport:
    if config.base_execution:
        return VectorizedSupport(False, "Vectorized v1 does not support separate signal/base execution timeframes.")
    if config.intrabar_sim:
        return VectorizedSupport(False, f"Vectorized {strategy_name} v1 does not support intrabar simulation.")
    return VectorizedSupport(True)


@dataclass(frozen=True)
class ZScoreMeanReversionVectorizedAdapter(VectorizedStrategyAdapter):
    strategy_name: str = "ZScoreMeanReversionStrategy"

    def supports(self, config: BacktestConfig, strategy_cls: Type[Strategy]) -> VectorizedSupport:
        if strategy_cls.__name__ != self.strategy_name:
            return VectorizedSupport(False, f"Strategy {strategy_cls.__name__} is not handled by the z-score adapter.")
        return _supports_long_only_v1(config, "z-score mean reversion")

    def prepare_order_plan_context(
        self,
        data: pd.DataFrame,
        param_grid: List[Dict],
        config: BacktestConfig,
    ) -> Any:
        context: Dict[tuple, pd.DataFrame] = {}
        for params in param_grid:
            key = tuple(sorted(params.items()))
            if key not in context:
                context[key] = compute_zscore_mean_reversion_features(data, params)
        return context

    def build_order_plan(self, data: pd.DataFrame, param_grid: List[Dict], config: BacktestConfig) -> VectorizedOrderPlan:
        return self.build_order_plan_from_context(data, param_grid, config, prepared_context=None)

    def build_order_plan_from_context(
        self,
        data: pd.DataFrame,
        param_grid: List[Dict],
        config: BacktestConfig,
        prepared_context: Any = None,
    ) -> VectorizedOrderPlan:
        if not param_grid:
            empty_bool = np.empty((len(data), 0), dtype=bool)
            empty_float = np.empty(0, dtype=float)
            empty_types = np.empty((len(data), 0), dtype=np.int8)
            empty_prices = np.empty((len(data), 0), dtype=float)
            return VectorizedOrderPlan(
                empty_bool,
                empty_bool,
                empty_bool,
                empty_bool,
                empty_bool,
                empty_types,
                empty_types,
                empty_types,
                empty_types,
                empty_prices,
                empty_prices,
                empty_prices,
                empty_prices,
                empty_float,
                empty_float,
            )
        if prepared_context is None:
            prepared_context = self.prepare_order_plan_context(data, param_grid, config)
        enter_long_cols = []
        exit_long_cols = []
        long_targets = []
        for params in param_grid:
            target = float(params.get("target", 1.0))
            if target < 0:
                raise ValueError("ZScoreMeanReversionStrategy requires target >= 0 in vectorized v1.")
            features = prepared_context[tuple(sorted(params.items()))]
            enter_long_cols.append(features["long_entry_signal"].to_numpy(dtype=bool))
            exit_long_cols.append(features["long_exit_signal"].to_numpy(dtype=bool))
            long_targets.append(target)
        enter_long = np.column_stack(enter_long_cols).astype(bool, copy=False)
        exit_long = np.column_stack(exit_long_cols).astype(bool, copy=False)
        empty_short = np.zeros_like(enter_long, dtype=bool)
        market_types = np.full_like(enter_long, ORDER_TYPE_MARKET, dtype=np.int8)
        nan_prices = np.full(enter_long.shape, np.nan, dtype=float)
        return VectorizedOrderPlan(
            enter_long=enter_long,
            exit_long=exit_long,
            enter_short=empty_short,
            exit_short=empty_short.copy(),
            cancel_pending=np.zeros_like(enter_long, dtype=bool),
            enter_long_order_type=market_types,
            exit_long_order_type=market_types.copy(),
            enter_short_order_type=market_types.copy(),
            exit_short_order_type=market_types.copy(),
            enter_long_price=nan_prices,
            exit_long_price=nan_prices.copy(),
            enter_short_price=nan_prices.copy(),
            exit_short_price=nan_prices.copy(),
            long_targets=np.array(long_targets, dtype=float),
            short_targets=np.zeros(len(param_grid), dtype=float),
        )


_ADAPTERS: Dict[str, VectorizedStrategyAdapter] = {
    "SMACrossStrategy": SMACrossVectorizedAdapter(),
    "ZScoreMeanReversionStrategy": ZScoreMeanReversionVectorizedAdapter(),
}


def get_vectorized_adapter(strategy_cls: Type[Strategy]) -> VectorizedStrategyAdapter | None:
    return _ADAPTERS.get(strategy_cls.__name__)
