from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Type

import numpy as np
import pandas as pd

from .engine import BacktestConfig
from .sample_strategies import (
    compute_adaptive_ou_kalman_zscore_features,
    compute_regime_filtered_adaptive_ou_kalman_zscore_features,
    compute_sma_regime_filtered_zscore_mean_reversion_features,
    compute_sma_trend_regime_features,
    compute_trend_following_ribbon_features,
    compute_zscore_mean_reversion_features,
)
from .sizing import POSITION_SIZING_NONE, normalize_position_sizing_model
from .strategy import Strategy

ORDER_TYPE_MARKET = 0
ORDER_TYPE_LIMIT = 1
ORDER_TYPE_STOP = 2


def _coerce_bool_value(value, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _mode_token_value(value: object) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


_STOP_REENTRY_IMMEDIATE_MODES = {"immediate", "nextbar", "next_bar", "next_bar_immediate", "same_regime"}


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
    context: Any = None


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

    def supports_params(self, param_grid: List[Dict]) -> VectorizedSupport:
        return VectorizedSupport(True)

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

    def build_before_fill_update(
        self,
        *,
        order_plan: VectorizedOrderPlan | None = None,
        data: pd.DataFrame,
        bar_idx: int,
        param_grid: List[Dict],
        config: BacktestConfig,
        cash: np.ndarray,
        position: np.ndarray,
        avg_price: np.ndarray,
        pending_qty: np.ndarray,
        pending_order_type: np.ndarray,
        pending_order_price: np.ndarray,
        pending_order_active: np.ndarray,
    ) -> VectorizedPendingOrderUpdate | None:
        return None

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

    def build_after_fill_update_from_fills(self, **kwargs) -> VectorizedPendingOrderUpdate | None:
        legacy_keys = {
            "data",
            "bar_idx",
            "param_grid",
            "config",
            "filled_mask",
            "cash",
            "position",
            "avg_price",
            "pending_qty",
            "pending_order_type",
            "pending_order_price",
            "pending_order_active",
        }
        return self.build_after_fill_update(**{key: kwargs[key] for key in legacy_keys})

    def build_bar_close_update(
        self,
        *,
        order_plan: VectorizedOrderPlan | None = None,
        data: pd.DataFrame,
        bar_idx: int,
        param_grid: List[Dict],
        config: BacktestConfig,
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
        enter_short_cols = []
        exit_short_cols = []
        long_targets = []
        short_targets = []
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
            prev_fast = np.roll(fast_arr, 1)
            prev_slow = np.roll(slow_arr, 1)
            prev_fast[0] = np.nan
            prev_slow[0] = np.nan
            prev_valid = ~np.isnan(prev_fast) & ~np.isnan(prev_slow)
            crossed_above = (fast_arr > slow_arr) & (prev_fast <= prev_slow) & valid & prev_valid
            crossed_below = (fast_arr < slow_arr) & (prev_fast >= prev_slow) & valid & prev_valid
            enter_long_cols.append(crossed_above)
            if config.allow_short:
                exit_long_cols.append(np.zeros_like(crossed_below, dtype=bool))
                enter_short_cols.append(crossed_below)
                exit_short_cols.append(np.zeros_like(crossed_above, dtype=bool))
            else:
                exit_long_cols.append(crossed_below)
                enter_short_cols.append(np.zeros_like(crossed_below, dtype=bool))
                exit_short_cols.append(np.zeros_like(crossed_above, dtype=bool))
            long_targets.append(target)
            short_targets.append(target)
        enter_long = np.column_stack(enter_long_cols).astype(bool, copy=False)
        exit_long = np.column_stack(exit_long_cols).astype(bool, copy=False)
        enter_short = np.column_stack(enter_short_cols).astype(bool, copy=False)
        exit_short = np.column_stack(exit_short_cols).astype(bool, copy=False)
        market_types = np.full_like(enter_long, ORDER_TYPE_MARKET, dtype=np.int8)
        nan_prices = np.full(enter_long.shape, np.nan, dtype=float)
        return VectorizedOrderPlan(
            enter_long=enter_long,
            exit_long=exit_long,
            enter_short=enter_short,
            exit_short=exit_short,
            cancel_pending=enter_long | exit_long | enter_short | exit_short,
            enter_long_order_type=market_types,
            exit_long_order_type=market_types.copy(),
            enter_short_order_type=market_types.copy(),
            exit_short_order_type=market_types.copy(),
            enter_long_price=nan_prices,
            exit_long_price=nan_prices.copy(),
            enter_short_price=nan_prices.copy(),
            exit_short_price=nan_prices.copy(),
            long_targets=np.array(long_targets, dtype=float),
            short_targets=np.array(short_targets, dtype=float),
        )


@dataclass(frozen=True)
class SMATrendRegimeVectorizedAdapter(VectorizedStrategyAdapter):
    strategy_name: str = "SMATrendRegimeStrategy"

    def supports(self, config: BacktestConfig, strategy_cls: Type[Strategy]) -> VectorizedSupport:
        if strategy_cls.__name__ != self.strategy_name:
            return VectorizedSupport(False, f"Strategy {strategy_cls.__name__} is not handled by the SMA trend regime adapter.")
        if config.base_execution:
            return VectorizedSupport(False, "Vectorized SMA trend regime v1 does not support separate signal/base execution timeframes.")
        if config.intrabar_sim:
            return VectorizedSupport(False, "Vectorized SMA trend regime v1 does not support intrabar simulation.")
        return VectorizedSupport(True)

    def supports_after_fill(self) -> bool:
        return True

    def supports_params(self, param_grid: List[Dict]) -> VectorizedSupport:
        return VectorizedSupport(True)

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
                context[key] = compute_sma_trend_regime_features(data, params)
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
        enter_short_cols = []
        exit_short_cols = []
        cancel_cols = []
        long_targets = []
        short_targets = []
        for params in param_grid:
            target = float(params.get("target", 1.0))
            if target < 0:
                raise ValueError("SMATrendRegimeStrategy requires target >= 0 in vectorized v1.")
            features = prepared_context[tuple(sorted(params.items()))]
            enter_long = features["long_entry_signal"].to_numpy(dtype=bool)
            enter_short = features["short_entry_signal"].to_numpy(dtype=bool)
            exit_long = features["long_exit_signal"].to_numpy(dtype=bool)
            exit_short = features["short_exit_signal"].to_numpy(dtype=bool)
            enter_long_cols.append(enter_long)
            exit_long_cols.append(exit_long)
            enter_short_cols.append(enter_short)
            exit_short_cols.append(exit_short)
            cancel_cols.append(enter_long | enter_short | exit_long | exit_short)
            long_targets.append(target)
            short_targets.append(target)
        enter_long = np.column_stack(enter_long_cols).astype(bool, copy=False)
        exit_long = np.column_stack(exit_long_cols).astype(bool, copy=False)
        enter_short = np.column_stack(enter_short_cols).astype(bool, copy=False)
        exit_short = np.column_stack(exit_short_cols).astype(bool, copy=False)
        market_types = np.full_like(enter_long, ORDER_TYPE_MARKET, dtype=np.int8)
        nan_prices = np.full(enter_long.shape, np.nan, dtype=float)
        return VectorizedOrderPlan(
            enter_long=enter_long,
            exit_long=exit_long,
            enter_short=enter_short,
            exit_short=exit_short,
            cancel_pending=np.column_stack(cancel_cols).astype(bool, copy=False),
            enter_long_order_type=market_types,
            exit_long_order_type=market_types.copy(),
            enter_short_order_type=market_types.copy(),
            exit_short_order_type=market_types.copy(),
            enter_long_price=nan_prices,
            exit_long_price=nan_prices.copy(),
            enter_short_price=nan_prices.copy(),
            exit_short_price=nan_prices.copy(),
            long_targets=np.array(long_targets, dtype=float),
            short_targets=np.array(short_targets, dtype=float),
            context=prepared_context,
        )

    def build_after_fill_update_from_fills(self, **kwargs) -> VectorizedPendingOrderUpdate | None:
        order_plan = kwargs.get("order_plan")
        context = order_plan.context if order_plan is not None else None
        param_grid = list(kwargs["param_grid"])
        runtime = self._runtime_state(context, len(param_grid))
        filled_slot_mask = np.asarray(kwargs.get("filled_slot_mask"), dtype=bool)
        filled_order_type = np.asarray(kwargs.get("filled_order_type"))
        filled_order_qty = np.asarray(kwargs.get("filled_order_qty"), dtype=float)
        filled_order_price = np.asarray(kwargs.get("filled_order_price"), dtype=float)
        position = np.asarray(kwargs["position"], dtype=float)

        if filled_slot_mask.shape == filled_order_type.shape:
            for col, params in enumerate(param_grid):
                if not _coerce_bool_value(params.get("reenter_after_stop", False), False):
                    continue
                filled_stops = filled_slot_mask[col, :] & (filled_order_type[col, :] == ORDER_TYPE_STOP)
                if not np.any(filled_stops):
                    if abs(float(position[col])) > 1e-12:
                        runtime["pending_reentry_side"][col] = 0
                        runtime["pending_reentry_stop"][col] = np.nan
                    continue
                if abs(float(position[col])) > 1e-12:
                    runtime["pending_reentry_side"][col] = 0
                    runtime["pending_reentry_stop"][col] = np.nan
                    continue
                first_stop_slot = int(np.flatnonzero(filled_stops)[0])
                stop_qty = float(filled_order_qty[col, first_stop_slot])
                runtime["pending_reentry_side"][col] = 1 if stop_qty < 0.0 else -1
                runtime["pending_reentry_stop"][col] = float(filled_order_price[col, first_stop_slot])

        return self._build_atr_stop_update(
            context=context,
            data=kwargs["data"],
            bar_idx=kwargs["bar_idx"],
            param_grid=param_grid,
            config=kwargs["config"],
            cash=kwargs["cash"],
            position=kwargs["position"],
            avg_price=kwargs["avg_price"],
            pending_qty=kwargs["pending_qty"],
            pending_order_type=kwargs["pending_order_type"],
            pending_order_price=kwargs["pending_order_price"],
            pending_order_active=kwargs["pending_order_active"],
            refresh_mask=np.asarray(kwargs["filled_mask"], dtype=bool),
            from_fill=True,
        )

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
        return self._build_atr_stop_update(
            context=None,
            data=data,
            bar_idx=bar_idx,
            param_grid=param_grid,
            config=config,
            cash=cash,
            position=position,
            avg_price=avg_price,
            pending_qty=pending_qty,
            pending_order_type=pending_order_type,
            pending_order_price=pending_order_price,
            pending_order_active=pending_order_active,
            refresh_mask=np.asarray(filled_mask, dtype=bool),
            from_fill=True,
        )

    def build_bar_close_update(
        self,
        *,
        order_plan: VectorizedOrderPlan | None = None,
        data: pd.DataFrame,
        bar_idx: int,
        param_grid: List[Dict],
        config: BacktestConfig,
        cash: np.ndarray,
        position: np.ndarray,
        avg_price: np.ndarray,
        pending_qty: np.ndarray,
        pending_order_type: np.ndarray,
        pending_order_price: np.ndarray,
        pending_order_active: np.ndarray,
    ) -> VectorizedPendingOrderUpdate | None:
        return self._build_atr_stop_update(
            context=order_plan.context if order_plan is not None else None,
            data=data,
            bar_idx=bar_idx,
            param_grid=param_grid,
            config=config,
            cash=cash,
            position=position,
            avg_price=avg_price,
            pending_qty=pending_qty,
            pending_order_type=pending_order_type,
            pending_order_price=pending_order_price,
            pending_order_active=pending_order_active,
            refresh_mask=np.abs(position) > 1e-12,
            from_fill=False,
        )

    def _runtime_state(self, context: Any, n_params: int) -> Dict[str, np.ndarray]:
        if not isinstance(context, dict):
            return {
                "pending_reentry_side": np.zeros(n_params, dtype=np.int8),
                "pending_reentry_stop": np.full(n_params, np.nan, dtype=float),
                "high_watermark": np.full(n_params, np.nan, dtype=float),
                "low_watermark": np.full(n_params, np.nan, dtype=float),
            }
        runtime = context.get("__sma_trend_runtime__")
        if not isinstance(runtime, dict) or len(runtime.get("pending_reentry_side", [])) != n_params:
            runtime = {
                "pending_reentry_side": np.zeros(n_params, dtype=np.int8),
                "pending_reentry_stop": np.full(n_params, np.nan, dtype=float),
                "high_watermark": np.full(n_params, np.nan, dtype=float),
                "low_watermark": np.full(n_params, np.nan, dtype=float),
            }
            context["__sma_trend_runtime__"] = runtime
        return runtime

    def _reentry_order_qty(
        self,
        *,
        side: int,
        target: float,
        cash_value: float,
        close_px: float,
        config: BacktestConfig,
    ) -> float:
        if close_px <= 0.0:
            return 0.0
        signed_target = float(target) if side > 0 else -float(target)
        desired_qty = signed_target * float(cash_value) / float(close_px)
        if normalize_position_sizing_model(config.position_sizing_model) != POSITION_SIZING_NONE:
            min_shares = max(float(config.min_position_shares), 0.0)
            if min_shares > 0.0 and 0.0 < abs(desired_qty) < min_shares:
                desired_qty = min_shares if desired_qty > 0.0 else -min_shares
        return desired_qty

    def _build_atr_stop_update(
        self,
        *,
        context: Any,
        data: pd.DataFrame,
        bar_idx: int,
        param_grid: List[Dict],
        config: BacktestConfig,
        cash: np.ndarray,
        position: np.ndarray,
        avg_price: np.ndarray,
        pending_qty: np.ndarray,
        pending_order_type: np.ndarray,
        pending_order_price: np.ndarray,
        pending_order_active: np.ndarray,
        refresh_mask: np.ndarray,
        from_fill: bool,
    ) -> VectorizedPendingOrderUpdate | None:
        n_params = len(param_grid)
        cancel_pending = np.zeros_like(pending_order_active, dtype=bool)
        submit_order = np.zeros(n_params, dtype=bool)
        order_qty = np.zeros(n_params, dtype=float)
        order_type = np.zeros(n_params, dtype=np.int8)
        order_price = np.full(n_params, np.nan, dtype=float)
        changed = False
        high_px = float(data["high"].iloc[bar_idx])
        low_px = float(data["low"].iloc[bar_idx])
        close_px = float(data["close"].iloc[bar_idx])
        runtime = self._runtime_state(context, n_params)
        for col, params in enumerate(param_grid):
            pos = float(position[col])
            reenter_after_stop = _coerce_bool_value(params.get("reenter_after_stop", False), False)
            active_exit_slots = pending_order_active[col, :] & (np.sign(pending_qty[col, :]) == -np.sign(pos))
            active_stop_slots = active_exit_slots & (pending_order_type[col, :] == ORDER_TYPE_STOP)
            active_non_stop_exits = active_exit_slots & (pending_order_type[col, :] != ORDER_TYPE_STOP)
            if abs(pos) <= 1e-12:
                stale_stop_slots = pending_order_active[col, :] & (pending_order_type[col, :] == ORDER_TYPE_STOP)
                if np.any(stale_stop_slots):
                    cancel_pending[col, stale_stop_slots] = True
                    changed = True
                runtime["high_watermark"][col] = np.nan
                runtime["low_watermark"][col] = np.nan
                if not from_fill and reenter_after_stop:
                    pending_side = int(runtime["pending_reentry_side"][col])
                    if pending_side != 0:
                        features = self._features_for_params(data, params, context)
                        if bar_idx < len(features):
                            stop_price = float(runtime["pending_reentry_stop"][col])
                            mode = _mode_token_value(params.get("stop_reentry_mode", params.get("reentry_mode", "Immediate")))
                            trend_ok = (
                                bool(features["trend_is_bullish"].iloc[bar_idx])
                                if pending_side > 0
                                else bool(features["trend_is_bearish"].iloc[bar_idx])
                            )
                            if not trend_ok:
                                runtime["pending_reentry_side"][col] = 0
                                runtime["pending_reentry_stop"][col] = np.nan
                            else:
                                recaptured = (
                                    pending_side > 0 and close_px > stop_price
                                ) or (
                                    pending_side < 0 and close_px < stop_price
                                )
                                if mode in _STOP_REENTRY_IMMEDIATE_MODES or recaptured:
                                    target = float(params.get("target", 1.0))
                                    order_qty[col] = self._reentry_order_qty(
                                        side=pending_side,
                                        target=target,
                                        cash_value=float(cash[col]),
                                        close_px=close_px,
                                        config=config,
                                    )
                                    if abs(float(order_qty[col])) > 1e-12:
                                        submit_order[col] = True
                                        order_type[col] = ORDER_TYPE_MARKET
                                        order_price[col] = np.nan
                                        changed = True
                                    runtime["pending_reentry_side"][col] = 0
                                    runtime["pending_reentry_stop"][col] = np.nan
                continue
            if not bool(refresh_mask[col]):
                continue
            if np.any(active_non_stop_exits):
                continue
            if not _coerce_bool_value(params.get("use_atr_stop", True), True):
                continue
            atr_stop_mult = max(float(params.get("atr_stop_mult", params.get("atr_mult", 3.0))), 0.0)
            if atr_stop_mult <= 0.0:
                continue
            features = self._features_for_params(data, params, context)
            if bar_idx >= len(features):
                continue
            atr_distance = float(features["atr_stop_distance"].iloc[bar_idx])
            if not np.isfinite(atr_distance) or atr_distance <= 0.0:
                continue
            update_mode = str(params.get("atr_stop_update_mode", "Trailing")).strip().lower()
            basis_mode = str(params.get("atr_stop_basis", "HighLow")).strip().lower()
            if position[col] > 1e-12:
                existing_prices = pending_order_price[col, active_stop_slots]
                finite_existing = existing_prices[np.isfinite(existing_prices)]
                existing_stop = float(finite_existing.max()) if finite_existing.size else np.nan
                if reenter_after_stop:
                    if from_fill or not np.isfinite(runtime["high_watermark"][col]):
                        runtime["high_watermark"][col] = float(avg_price[col])
                    if not from_fill:
                        basis_value = close_px if basis_mode in {"close", "closing"} else high_px
                        runtime["high_watermark"][col] = max(float(runtime["high_watermark"][col]), basis_value)
                    basis = float(runtime["high_watermark"][col])
                else:
                    basis = float(avg_price[col]) if from_fill else (close_px if basis_mode in {"close", "closing"} else high_px)
                candidate_stop = basis - atr_distance * atr_stop_mult
                if not from_fill and not reenter_after_stop:
                    feature_stop = float(features["long_atr_stop_price"].iloc[bar_idx])
                    avg_stop = float(avg_price[col]) - atr_distance * atr_stop_mult
                    if np.isfinite(feature_stop):
                        candidate_stop = max(feature_stop, avg_stop)
                if np.isfinite(existing_stop):
                    stop_price = candidate_stop if update_mode in {"rolling", "dynamic"} else max(existing_stop, candidate_stop)
                else:
                    stop_price = candidate_stop
                order_qty[col] = -float(position[col])
            else:
                existing_prices = pending_order_price[col, active_stop_slots]
                finite_existing = existing_prices[np.isfinite(existing_prices)]
                existing_stop = float(finite_existing.min()) if finite_existing.size else np.nan
                if reenter_after_stop:
                    if from_fill or not np.isfinite(runtime["low_watermark"][col]):
                        runtime["low_watermark"][col] = float(avg_price[col])
                    if not from_fill:
                        basis_value = close_px if basis_mode in {"close", "closing"} else low_px
                        runtime["low_watermark"][col] = min(float(runtime["low_watermark"][col]), basis_value)
                    basis = float(runtime["low_watermark"][col])
                else:
                    basis = float(avg_price[col]) if from_fill else (close_px if basis_mode in {"close", "closing"} else low_px)
                candidate_stop = basis + atr_distance * atr_stop_mult
                if not from_fill and not reenter_after_stop:
                    feature_stop = float(features["short_atr_stop_price"].iloc[bar_idx])
                    avg_stop = float(avg_price[col]) + atr_distance * atr_stop_mult
                    if np.isfinite(feature_stop):
                        candidate_stop = min(feature_stop, avg_stop)
                if np.isfinite(existing_stop):
                    stop_price = candidate_stop if update_mode in {"rolling", "dynamic"} else min(existing_stop, candidate_stop)
                else:
                    stop_price = candidate_stop
                order_qty[col] = -float(position[col])
            if np.any(active_stop_slots):
                cancel_pending[col, active_stop_slots] = True
            submit_order[col] = True
            order_type[col] = ORDER_TYPE_STOP
            order_price[col] = stop_price
            changed = True
        if not changed:
            return None
        return VectorizedPendingOrderUpdate(
            cancel_pending=cancel_pending,
            submit_order=submit_order,
            order_qty=order_qty,
            order_type=order_type,
            order_price=order_price,
        )

    def _features_for_params(self, data: pd.DataFrame, params: Dict, context: Any) -> pd.DataFrame:
        key = tuple(sorted(params.items()))
        if isinstance(context, dict) and key in context:
            return context[key]
        return compute_sma_trend_regime_features(data, params)


@dataclass(frozen=True)
class TrendFollowingRibbonVectorizedAdapter(VectorizedStrategyAdapter):
    strategy_name: str = "TrendFollowingRibbonStrategy"

    def supports(self, config: BacktestConfig, strategy_cls: Type[Strategy]) -> VectorizedSupport:
        if strategy_cls.__name__ != self.strategy_name:
            return VectorizedSupport(False, f"Strategy {strategy_cls.__name__} is not handled by the ribbon adapter.")
        if config.base_execution:
            return VectorizedSupport(False, "Vectorized trend-following ribbon v1 does not support separate signal/base execution timeframes.")
        if config.intrabar_sim:
            return VectorizedSupport(False, "Vectorized trend-following ribbon v1 does not support intrabar simulation.")
        return VectorizedSupport(True)

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
                context[key] = compute_trend_following_ribbon_features(data, params)
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
        enter_short_cols = []
        exit_short_cols = []
        cancel_cols = []
        long_targets = []
        short_targets = []
        for params in param_grid:
            target = float(params.get("target", 1.0))
            if target < 0:
                raise ValueError("TrendFollowingRibbonStrategy requires target >= 0 in vectorized v1.")
            features = prepared_context[tuple(sorted(params.items()))]
            enter_long = features["long_entry_signal"].to_numpy(dtype=bool)
            raw_enter_short = features["short_entry_signal"].to_numpy(dtype=bool)
            exit_long = features["long_exit_signal"].to_numpy(dtype=bool) & ~raw_enter_short
            exit_short = features["short_exit_signal"].to_numpy(dtype=bool) & ~enter_long
            if config.allow_short:
                enter_short = raw_enter_short
            else:
                exit_long = exit_long | raw_enter_short
                enter_short = np.zeros_like(raw_enter_short, dtype=bool)
                exit_short = np.zeros_like(exit_short, dtype=bool)
            enter_long_cols.append(enter_long)
            exit_long_cols.append(exit_long)
            enter_short_cols.append(enter_short)
            exit_short_cols.append(exit_short)
            cancel_cols.append(enter_long | enter_short | exit_long | exit_short)
            long_targets.append(target)
            short_targets.append(target)
        enter_long = np.column_stack(enter_long_cols).astype(bool, copy=False)
        exit_long = np.column_stack(exit_long_cols).astype(bool, copy=False)
        enter_short = np.column_stack(enter_short_cols).astype(bool, copy=False)
        exit_short = np.column_stack(exit_short_cols).astype(bool, copy=False)
        market_types = np.full_like(enter_long, ORDER_TYPE_MARKET, dtype=np.int8)
        nan_prices = np.full(enter_long.shape, np.nan, dtype=float)
        return VectorizedOrderPlan(
            enter_long=enter_long,
            exit_long=exit_long,
            enter_short=enter_short,
            exit_short=exit_short,
            cancel_pending=np.column_stack(cancel_cols).astype(bool, copy=False),
            enter_long_order_type=market_types,
            exit_long_order_type=market_types.copy(),
            enter_short_order_type=market_types.copy(),
            exit_short_order_type=market_types.copy(),
            enter_long_price=nan_prices,
            exit_long_price=nan_prices.copy(),
            enter_short_price=nan_prices.copy(),
            exit_short_price=nan_prices.copy(),
            long_targets=np.array(long_targets, dtype=float),
            short_targets=np.array(short_targets, dtype=float),
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


@dataclass(frozen=True)
class SMARegimeFilteredZScoreMeanReversionVectorizedAdapter(VectorizedStrategyAdapter):
    strategy_name: str = "SMARegimeFilteredZScoreMeanReversionStrategy"

    def supports(self, config: BacktestConfig, strategy_cls: Type[Strategy]) -> VectorizedSupport:
        if strategy_cls.__name__ != self.strategy_name:
            return VectorizedSupport(
                False,
                f"Strategy {strategy_cls.__name__} is not handled by the SMA-regime z-score adapter.",
            )
        if config.base_execution:
            return VectorizedSupport(False, "Vectorized SMA-regime z-score v1 does not support separate signal/base execution timeframes.")
        if config.intrabar_sim:
            return VectorizedSupport(False, "Vectorized SMA-regime z-score v1 does not support intrabar simulation.")
        return VectorizedSupport(True)

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
                context[key] = compute_sma_regime_filtered_zscore_mean_reversion_features(data, params)
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
        enter_short_cols = []
        exit_short_cols = []
        cancel_cols = []
        long_targets = []
        short_targets = []
        for params in param_grid:
            target = float(params.get("target", 1.0))
            if target < 0:
                raise ValueError("SMARegimeFilteredZScoreMeanReversionStrategy requires target >= 0 in vectorized v1.")
            features = prepared_context[tuple(sorted(params.items()))]
            enter_long = features["long_entry_signal"].to_numpy(dtype=bool)
            exit_long = features["long_exit_signal"].to_numpy(dtype=bool)
            raw_enter_short = features["short_entry_signal"].to_numpy(dtype=bool)
            raw_exit_short = features["short_exit_signal"].to_numpy(dtype=bool)
            if config.allow_short:
                enter_short = raw_enter_short
                exit_short = raw_exit_short
            else:
                exit_long = exit_long | raw_enter_short
                enter_short = np.zeros_like(raw_enter_short, dtype=bool)
                exit_short = np.zeros_like(raw_exit_short, dtype=bool)
            enter_long_cols.append(enter_long)
            exit_long_cols.append(exit_long)
            enter_short_cols.append(enter_short)
            exit_short_cols.append(exit_short)
            cancel_cols.append(enter_long | exit_long | enter_short | exit_short)
            long_targets.append(target)
            short_targets.append(target)
        enter_long = np.column_stack(enter_long_cols).astype(bool, copy=False)
        exit_long = np.column_stack(exit_long_cols).astype(bool, copy=False)
        enter_short = np.column_stack(enter_short_cols).astype(bool, copy=False)
        exit_short = np.column_stack(exit_short_cols).astype(bool, copy=False)
        market_types = np.full_like(enter_long, ORDER_TYPE_MARKET, dtype=np.int8)
        nan_prices = np.full(enter_long.shape, np.nan, dtype=float)
        return VectorizedOrderPlan(
            enter_long=enter_long,
            exit_long=exit_long,
            enter_short=enter_short,
            exit_short=exit_short,
            cancel_pending=np.column_stack(cancel_cols).astype(bool, copy=False),
            enter_long_order_type=market_types,
            exit_long_order_type=market_types.copy(),
            enter_short_order_type=market_types.copy(),
            exit_short_order_type=market_types.copy(),
            enter_long_price=nan_prices,
            exit_long_price=nan_prices.copy(),
            enter_short_price=nan_prices.copy(),
            exit_short_price=nan_prices.copy(),
            long_targets=np.array(long_targets, dtype=float),
            short_targets=np.array(short_targets, dtype=float),
        )


@dataclass(frozen=True)
class AdaptiveOUKalmanZScoreVectorizedAdapter(VectorizedStrategyAdapter):
    strategy_name: str = "AdaptiveOUKalmanZScoreStrategy"

    def supports(self, config: BacktestConfig, strategy_cls: Type[Strategy]) -> VectorizedSupport:
        if strategy_cls.__name__ != self.strategy_name:
            return VectorizedSupport(
                False,
                f"Strategy {strategy_cls.__name__} is not handled by the adaptive OU Kalman z-score adapter.",
            )
        return _supports_long_only_v1(config, "adaptive OU Kalman z-score mean reversion")

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
                context[key] = compute_adaptive_ou_kalman_zscore_features(data, params)
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
                raise ValueError("AdaptiveOUKalmanZScoreStrategy requires target >= 0 in vectorized v1.")
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


@dataclass(frozen=True)
class RegimeFilteredAdaptiveOUKalmanZScoreVectorizedAdapter(VectorizedStrategyAdapter):
    strategy_name: str = "RegimeFilteredAdaptiveOUKalmanZScoreStrategy"

    def supports(self, config: BacktestConfig, strategy_cls: Type[Strategy]) -> VectorizedSupport:
        if strategy_cls.__name__ != self.strategy_name:
            return VectorizedSupport(
                False,
                f"Strategy {strategy_cls.__name__} is not handled by the regime-filtered adaptive OU Kalman adapter.",
            )
        return _supports_long_only_v1(config, "regime-filtered adaptive OU Kalman z-score mean reversion")

    def supports_after_fill(self) -> bool:
        return True

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
                context[key] = compute_regime_filtered_adaptive_ou_kalman_zscore_features(data, params)
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
        cancel_cols = []
        long_targets = []
        for params in param_grid:
            target = float(params.get("target", 1.0))
            if target < 0:
                raise ValueError("RegimeFilteredAdaptiveOUKalmanZScoreStrategy requires target >= 0 in vectorized v1.")
            features = prepared_context[tuple(sorted(params.items()))]
            long_exit = features["long_exit_signal"].to_numpy(dtype=bool)
            enter_long_cols.append(features["long_entry_signal"].to_numpy(dtype=bool))
            exit_long_cols.append(long_exit)
            cancel_cols.append(long_exit)
            long_targets.append(target)
        enter_long = np.column_stack(enter_long_cols).astype(bool, copy=False)
        exit_long = np.column_stack(exit_long_cols).astype(bool, copy=False)
        cancel_pending = np.column_stack(cancel_cols).astype(bool, copy=False)
        empty_short = np.zeros_like(enter_long, dtype=bool)
        market_types = np.full_like(enter_long, ORDER_TYPE_MARKET, dtype=np.int8)
        nan_prices = np.full(enter_long.shape, np.nan, dtype=float)
        return VectorizedOrderPlan(
            enter_long=enter_long,
            exit_long=exit_long,
            enter_short=empty_short,
            exit_short=empty_short.copy(),
            cancel_pending=cancel_pending,
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
            context=prepared_context,
        )

    def build_before_fill_update(
        self,
        *,
        order_plan: VectorizedOrderPlan | None = None,
        data: pd.DataFrame,
        bar_idx: int,
        param_grid: List[Dict],
        config: BacktestConfig,
        cash: np.ndarray,
        position: np.ndarray,
        avg_price: np.ndarray,
        pending_qty: np.ndarray,
        pending_order_type: np.ndarray,
        pending_order_price: np.ndarray,
        pending_order_active: np.ndarray,
    ) -> VectorizedPendingOrderUpdate | None:
        cancel_pending = np.zeros_like(pending_order_active, dtype=bool)
        context = order_plan.context if order_plan is not None else None
        for col, params in enumerate(param_grid):
            if not _coerce_bool_value(params.get("require_regime_on_fill", True), True):
                continue
            features = self._features_for_params(data, params, context)
            if bar_idx >= len(features) or bool(features["regime_is_choppy"].iloc[bar_idx]):
                continue
            cancel_pending[col, :] = pending_order_active[col, :] & (pending_qty[col, :] > 1e-12)
        if not np.any(cancel_pending):
            return None
        return VectorizedPendingOrderUpdate(
            cancel_pending=cancel_pending,
            submit_order=np.zeros_like(pending_order_active, dtype=bool),
            order_qty=np.zeros_like(pending_qty, dtype=float),
            order_type=np.zeros_like(pending_order_type, dtype=np.int8),
            order_price=np.full_like(pending_order_price, np.nan, dtype=float),
        )

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
        return self._build_atr_stop_update(
            context=None,
            data=data,
            bar_idx=bar_idx,
            param_grid=param_grid,
            position=position,
            avg_price=avg_price,
            pending_qty=pending_qty,
            pending_order_type=pending_order_type,
            pending_order_price=pending_order_price,
            pending_order_active=pending_order_active,
            refresh_mask=np.asarray(filled_mask, dtype=bool),
        )

    def build_bar_close_update(
        self,
        *,
        order_plan: VectorizedOrderPlan | None = None,
        data: pd.DataFrame,
        bar_idx: int,
        param_grid: List[Dict],
        config: BacktestConfig,
        cash: np.ndarray,
        position: np.ndarray,
        avg_price: np.ndarray,
        pending_qty: np.ndarray,
        pending_order_type: np.ndarray,
        pending_order_price: np.ndarray,
        pending_order_active: np.ndarray,
    ) -> VectorizedPendingOrderUpdate | None:
        return self._build_atr_stop_update(
            context=order_plan.context if order_plan is not None else None,
            data=data,
            bar_idx=bar_idx,
            param_grid=param_grid,
            position=position,
            avg_price=avg_price,
            pending_qty=pending_qty,
            pending_order_type=pending_order_type,
            pending_order_price=pending_order_price,
            pending_order_active=pending_order_active,
            refresh_mask=position > 1e-12,
        )

    def _build_atr_stop_update(
        self,
        *,
        context: Any,
        data: pd.DataFrame,
        bar_idx: int,
        param_grid: List[Dict],
        position: np.ndarray,
        avg_price: np.ndarray,
        pending_qty: np.ndarray,
        pending_order_type: np.ndarray,
        pending_order_price: np.ndarray,
        pending_order_active: np.ndarray,
        refresh_mask: np.ndarray,
    ) -> VectorizedPendingOrderUpdate | None:
        n_params = len(param_grid)
        cancel_pending = np.zeros_like(pending_order_active, dtype=bool)
        submit_order = np.zeros(n_params, dtype=bool)
        order_qty = np.zeros(n_params, dtype=float)
        order_type = np.zeros(n_params, dtype=np.int8)
        order_price = np.full(n_params, np.nan, dtype=float)
        changed = False
        for col, params in enumerate(param_grid):
            active_exit_slots = pending_order_active[col, :] & (pending_qty[col, :] < -1e-12)
            active_stop_slots = active_exit_slots & (pending_order_type[col, :] == ORDER_TYPE_STOP)
            if position[col] <= 1e-12:
                if np.any(active_stop_slots):
                    cancel_pending[col, active_stop_slots] = True
                    changed = True
                continue
            if not bool(refresh_mask[col]):
                continue
            if np.any(active_exit_slots & (pending_order_type[col, :] != ORDER_TYPE_STOP)):
                continue
            if not _coerce_bool_value(params.get("use_atr_stop", True), True):
                continue
            atr_stop_mult = max(float(params.get("atr_stop_mult", params.get("atr_mult", 2.0))), 0.0)
            if atr_stop_mult <= 0.0:
                continue
            features = self._features_for_params(data, params, context)
            if bar_idx >= len(features):
                continue
            atr_distance = float(features["atr_stop_distance"].iloc[bar_idx])
            existing_prices = pending_order_price[col, active_stop_slots]
            finite_existing = existing_prices[np.isfinite(existing_prices)]
            existing_stop = float(finite_existing.max()) if finite_existing.size else np.nan
            if not np.isfinite(atr_distance) or atr_distance <= 0.0:
                continue
            candidate_stop = float(avg_price[col]) - atr_distance * atr_stop_mult
            update_mode = str(params.get("atr_stop_update_mode", "Trailing")).strip().lower()
            if np.isfinite(existing_stop):
                if update_mode in {"rolling", "dynamic"}:
                    stop_price = candidate_stop
                elif update_mode in {"trailing", "trail"}:
                    stop_price = max(existing_stop, candidate_stop)
                else:
                    stop_price = existing_stop
            else:
                stop_price = candidate_stop
            if np.any(active_stop_slots):
                cancel_pending[col, active_stop_slots] = True
            submit_order[col] = True
            order_qty[col] = -float(position[col])
            order_type[col] = ORDER_TYPE_STOP
            order_price[col] = stop_price
            changed = True
        if not changed:
            return None
        return VectorizedPendingOrderUpdate(
            cancel_pending=cancel_pending,
            submit_order=submit_order,
            order_qty=order_qty,
            order_type=order_type,
            order_price=order_price,
        )

    def _features_for_params(self, data: pd.DataFrame, params: Dict, context: Any) -> pd.DataFrame:
        key = tuple(sorted(params.items()))
        if isinstance(context, dict) and key in context:
            return context[key]
        return compute_regime_filtered_adaptive_ou_kalman_zscore_features(data, params)


_ADAPTERS: Dict[str, VectorizedStrategyAdapter] = {
    "SMACrossStrategy": SMACrossVectorizedAdapter(),
    "SMATrendRegimeStrategy": SMATrendRegimeVectorizedAdapter(),
    "TrendFollowingRibbonStrategy": TrendFollowingRibbonVectorizedAdapter(),
    "ZScoreMeanReversionStrategy": ZScoreMeanReversionVectorizedAdapter(),
    "SMARegimeFilteredZScoreMeanReversionStrategy": SMARegimeFilteredZScoreMeanReversionVectorizedAdapter(),
    "AdaptiveOUKalmanZScoreStrategy": AdaptiveOUKalmanZScoreVectorizedAdapter(),
    "RegimeFilteredAdaptiveOUKalmanZScoreStrategy": RegimeFilteredAdaptiveOUKalmanZScoreVectorizedAdapter(),
}


def get_vectorized_adapter(strategy_cls: Type[Strategy]) -> VectorizedStrategyAdapter | None:
    return _ADAPTERS.get(strategy_cls.__name__)
