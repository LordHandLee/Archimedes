from __future__ import annotations

import unittest
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest_engine.engine import BacktestConfig
from backtest_engine.execution import ExecutionMode, ExecutionOrchestrator, ExecutionRequest
from backtest_engine.strategy import Strategy
from backtest_engine.vectorized_strategies import (
    ORDER_TYPE_LIMIT,
    ORDER_TYPE_MARKET,
    ORDER_TYPE_STOP,
    VectorizedOrderPlan,
    VectorizedPendingOrderUpdate,
    VectorizedStrategyAdapter,
    VectorizedSupport,
)
import backtest_engine.vectorized_strategies as vectorized_module


def _make_bars(periods: int = 120) -> pd.DataFrame:
    index = pd.date_range("2024-01-04 14:30", periods=periods, freq="1min", tz="UTC")
    close = 100.0 + np.linspace(0, 8, periods) + np.sin(np.arange(periods) / 7.0)
    open_ = np.roll(close, 1)
    open_[0] = close[0] - 0.1
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + 0.2,
            "low": np.minimum(open_, close) - 0.2,
            "close": close,
            "volume": 1000,
        },
        index=index,
    )


def _make_reversal_bars(periods: int = 80) -> pd.DataFrame:
    index = pd.date_range("2024-01-04 14:30", periods=periods, freq="1min", tz="UTC")
    close = np.linspace(100.0, 104.0, periods)
    close[18:] = np.linspace(close[17], 99.0, periods - 18)
    open_ = np.roll(close, 1)
    open_[0] = close[0] - 0.1
    high = np.maximum(open_, close) + 0.15
    low = np.minimum(open_, close) - 0.15
    low[22] = min(low[22], close[11] - 0.7)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 1000,
        },
        index=index,
    )


def _empty_plan(n_bars: int, n_params: int) -> VectorizedOrderPlan:
    empty_bool = np.zeros((n_bars, n_params), dtype=bool)
    market_types = np.full((n_bars, n_params), ORDER_TYPE_MARKET, dtype=np.int8)
    nan_prices = np.full((n_bars, n_params), np.nan, dtype=float)
    empty_targets = np.zeros(n_params, dtype=float)
    return VectorizedOrderPlan(
        enter_long=empty_bool.copy(),
        exit_long=empty_bool.copy(),
        enter_short=empty_bool.copy(),
        exit_short=empty_bool.copy(),
        cancel_pending=empty_bool.copy(),
        enter_long_order_type=market_types.copy(),
        exit_long_order_type=market_types.copy(),
        enter_short_order_type=market_types.copy(),
        exit_short_order_type=market_types.copy(),
        enter_long_price=nan_prices.copy(),
        exit_long_price=nan_prices.copy(),
        enter_short_price=nan_prices.copy(),
        exit_short_price=nan_prices.copy(),
        long_targets=empty_targets.copy(),
        short_targets=empty_targets.copy(),
    )


class ScheduledShortStrategy(Strategy):
    def initialize(self, data: pd.DataFrame) -> None:
        self.target = abs(float(self.params.get("target", 1.0)))
        self.enter_index = int(self.params.get("enter_index", 10))
        self.exit_index = int(self.params.get("exit_index", 60))
        self.enter_ts = data.index[self.enter_index]
        self.exit_ts = data.index[self.exit_index]

    def on_bar(self, timestamp: pd.Timestamp, bar: pd.Series, broker) -> None:
        if timestamp == self.enter_ts and broker.position_qty >= 0:
            broker.target_percent(-self.target, bar["close"])
        elif timestamp == self.exit_ts and broker.position_qty < 0:
            broker.target_percent(0.0, bar["close"])


class ScheduledStopEntryStrategy(Strategy):
    def initialize(self, data: pd.DataFrame) -> None:
        self.target = abs(float(self.params.get("target", 1.0)))
        self.submit_index = int(self.params.get("submit_index", 8))
        self.cancel_index = int(self.params.get("cancel_index", 16))
        self.replace_index = int(self.params.get("replace_index", 18))
        self.exit_index = int(self.params.get("exit_index", 90))
        self.initial_stop_offset = float(self.params.get("initial_stop_offset", 1.5))
        self.replacement_stop_offset = float(self.params.get("replacement_stop_offset", 0.2))
        self.data = data
        self.submit_ts = data.index[self.submit_index]
        self.cancel_ts = data.index[self.cancel_index]
        self.replace_ts = data.index[self.replace_index]
        self.exit_ts = data.index[self.exit_index]

    def on_bar(self, timestamp: pd.Timestamp, bar: pd.Series, broker) -> None:
        if timestamp == self.submit_ts and broker.position_qty <= 0:
            broker.buy_stop(1_000_000, float(bar["close"]) + self.initial_stop_offset, tag="entry_stop")
        elif timestamp == self.cancel_ts:
            broker.cancel_orders(tag="entry_stop")
        elif timestamp == self.replace_ts and broker.position_qty <= 0:
            broker.buy_stop(1_000_000, float(bar["close"]) + self.replacement_stop_offset, tag="entry_stop")
        elif timestamp == self.exit_ts and broker.position_qty > 0:
            broker.target_percent(0.0, bar["close"])


class ScheduledLimitEntryStrategy(Strategy):
    def initialize(self, data: pd.DataFrame) -> None:
        self.target = abs(float(self.params.get("target", 1.0)))
        self.submit_index = int(self.params.get("submit_index", 8))
        self.cancel_index = int(self.params.get("cancel_index", 16))
        self.replace_index = int(self.params.get("replace_index", 18))
        self.exit_index = int(self.params.get("exit_index", 90))
        self.initial_limit_offset = float(self.params.get("initial_limit_offset", 0.4))
        self.replacement_limit_offset = float(self.params.get("replacement_limit_offset", 0.2))
        self.submit_ts = data.index[self.submit_index]
        self.cancel_ts = data.index[self.cancel_index]
        self.replace_ts = data.index[self.replace_index]
        self.exit_ts = data.index[self.exit_index]

    def on_bar(self, timestamp: pd.Timestamp, bar: pd.Series, broker) -> None:
        if timestamp == self.submit_ts and broker.position_qty <= 0:
            broker.target_percent(
                self.target,
                float(bar["close"]),
                order_type="limit",
                limit_price=float(bar["close"]) - self.initial_limit_offset,
                tag="entry_limit",
            )
        elif timestamp == self.cancel_ts:
            broker.cancel_orders(tag="entry_limit")
        elif timestamp == self.replace_ts and broker.position_qty <= 0:
            broker.target_percent(
                self.target,
                float(bar["close"]),
                order_type="limit",
                limit_price=float(bar["close"]) - self.replacement_limit_offset,
                tag="entry_limit",
            )
        elif timestamp == self.exit_ts and broker.position_qty > 0:
            broker.target_percent(0.0, bar["close"])


class AfterFillProtectiveStopStrategy(Strategy):
    def initialize(self, data: pd.DataFrame) -> None:
        self.target = abs(float(self.params.get("target", 1.0)))
        self.enter_index = int(self.params.get("enter_index", 10))
        self.stop_offset = float(self.params.get("stop_offset", 0.4))
        self.enter_ts = data.index[self.enter_index]

    def on_bar(self, timestamp: pd.Timestamp, bar: pd.Series, broker) -> None:
        if timestamp == self.enter_ts and broker.position_qty <= 0:
            broker.target_percent(self.target, float(bar["close"]))

    def on_after_fill(self, timestamp: pd.Timestamp, bar: pd.Series, broker) -> None:
        if broker.position_qty > 0:
            broker.cancel_orders(tag="protective_stop")
            broker.sell_stop(abs(broker.position_qty), broker.avg_price - self.stop_offset, tag="protective_stop")


class AfterFillSameBarTakeProfitStrategy(Strategy):
    def initialize(self, data: pd.DataFrame) -> None:
        self.target = abs(float(self.params.get("target", 1.0)))
        self.enter_index = int(self.params.get("enter_index", 10))
        self.limit_offset = float(self.params.get("limit_offset", 0.1))
        self.enter_ts = data.index[self.enter_index]

    def on_bar(self, timestamp: pd.Timestamp, bar: pd.Series, broker) -> None:
        if timestamp == self.enter_ts and broker.position_qty <= 0:
            broker.target_percent(self.target, float(bar["close"]))

    def on_after_fill(self, timestamp: pd.Timestamp, bar: pd.Series, broker) -> None:
        if broker.position_qty > 0 and not broker.pending_orders:
            broker.sell_limit(abs(broker.position_qty), broker.avg_price + self.limit_offset, tag="same_bar_tp")


class AfterFillBracketStrategy(Strategy):
    def initialize(self, data: pd.DataFrame) -> None:
        self.target = abs(float(self.params.get("target", 1.0)))
        self.enter_index = int(self.params.get("enter_index", 10))
        self.stop_offset = float(self.params.get("stop_offset", 0.4))
        self.limit_offset = float(self.params.get("limit_offset", 0.8))
        self.enter_ts = data.index[self.enter_index]

    def on_bar(self, timestamp: pd.Timestamp, bar: pd.Series, broker) -> None:
        if timestamp == self.enter_ts and broker.position_qty <= 0:
            broker.target_percent(self.target, float(bar["close"]))

    def on_after_fill(self, timestamp: pd.Timestamp, bar: pd.Series, broker) -> None:
        broker.cancel_orders(tag="protective_stop")
        broker.cancel_orders(tag="take_profit")
        if broker.position_qty > 0:
            broker.sell_stop(abs(broker.position_qty), broker.avg_price - self.stop_offset, tag="protective_stop")
            broker.sell_limit(abs(broker.position_qty), broker.avg_price + self.limit_offset, tag="take_profit")


class AfterFillLadderedExitStrategy(Strategy):
    def initialize(self, data: pd.DataFrame) -> None:
        self.target = abs(float(self.params.get("target", 1.0)))
        self.enter_index = int(self.params.get("enter_index", 10))
        self.stop_offset = float(self.params.get("stop_offset", 0.2))
        self.tp1_offset = float(self.params.get("tp1_offset", 0.3))
        self.tp2_offset = float(self.params.get("tp2_offset", 0.7))
        self.enter_ts = data.index[self.enter_index]

    def on_bar(self, timestamp: pd.Timestamp, bar: pd.Series, broker) -> None:
        if timestamp == self.enter_ts and broker.position_qty <= 0:
            broker.target_percent(self.target, float(bar["close"]))

    def on_after_fill(self, timestamp: pd.Timestamp, bar: pd.Series, broker) -> None:
        has_live_orders = bool(broker.pending_orders)
        if broker.position_qty <= 0:
            broker.cancel_orders(tag="protective_stop")
            broker.cancel_orders(tag="tp1")
            broker.cancel_orders(tag="tp2")
            return
        if not has_live_orders:
            broker.sell_stop(abs(broker.position_qty), broker.avg_price - self.stop_offset, tag="protective_stop")
            tp1_qty = abs(broker.position_qty) / 2.0
            tp2_qty = abs(broker.position_qty) - tp1_qty
            broker.sell_limit(tp1_qty, broker.avg_price + self.tp1_offset, tag="tp1")
            broker.sell_limit(tp2_qty, broker.avg_price + self.tp2_offset, tag="tp2")
            return
        broker.cancel_orders(tag="protective_stop")
        broker.sell_stop(abs(broker.position_qty), broker.avg_price - self.stop_offset, tag="protective_stop")


@dataclass(frozen=True)
class ScheduledShortVectorizedAdapter(VectorizedStrategyAdapter):
    strategy_name: str = "ScheduledShortStrategy"

    def supports(self, config: BacktestConfig, strategy_cls) -> VectorizedSupport:
        if strategy_cls.__name__ != self.strategy_name:
            return VectorizedSupport(False, f"Strategy {strategy_cls.__name__} is not handled by the scheduled short adapter.")
        if config.base_execution:
            return VectorizedSupport(False, "Vectorized test adapter does not support base_execution.")
        if config.intrabar_sim:
            return VectorizedSupport(False, "Vectorized test adapter does not support intrabar simulation.")
        if not config.allow_short:
            return VectorizedSupport(False, "Scheduled short strategy requires allow_short=True.")
        return VectorizedSupport(True)

    def build_order_plan(self, data: pd.DataFrame, param_grid, config: BacktestConfig) -> VectorizedOrderPlan:
        n_bars = len(data)
        n_params = len(param_grid)
        plan = _empty_plan(n_bars, n_params)
        enter_short = np.zeros((n_bars, n_params), dtype=bool)
        exit_short = np.zeros((n_bars, n_params), dtype=bool)
        short_targets = np.zeros(n_params, dtype=float)
        for col, params in enumerate(param_grid):
            enter_idx = int(params.get("enter_index", 10))
            exit_idx = int(params.get("exit_index", 60))
            short_targets[col] = abs(float(params.get("target", 1.0)))
            enter_short[enter_idx, col] = True
            exit_short[exit_idx, col] = True
        plan.enter_short[:, :] = enter_short
        plan.exit_short[:, :] = exit_short
        plan.short_targets[:] = short_targets
        return plan


@dataclass(frozen=True)
class ScheduledStopEntryVectorizedAdapter(VectorizedStrategyAdapter):
    strategy_name: str = "ScheduledStopEntryStrategy"

    def supports(self, config: BacktestConfig, strategy_cls) -> VectorizedSupport:
        if strategy_cls.__name__ != self.strategy_name:
            return VectorizedSupport(False, f"Strategy {strategy_cls.__name__} is not handled by the scheduled stop adapter.")
        if config.base_execution:
            return VectorizedSupport(False, "Vectorized test adapter does not support base_execution.")
        if config.intrabar_sim:
            return VectorizedSupport(False, "Vectorized test adapter does not support intrabar simulation.")
        return VectorizedSupport(True)

    def build_order_plan(self, data: pd.DataFrame, param_grid, config: BacktestConfig) -> VectorizedOrderPlan:
        n_bars = len(data)
        n_params = len(param_grid)
        plan = _empty_plan(n_bars, n_params)
        close = data["close"].to_numpy(dtype=float)
        for col, params in enumerate(param_grid):
            submit_idx = int(params.get("submit_index", 8))
            cancel_idx = int(params.get("cancel_index", 16))
            replace_idx = int(params.get("replace_index", 18))
            exit_idx = int(params.get("exit_index", 90))
            target = abs(float(params.get("target", 1.0)))
            initial_stop_offset = float(params.get("initial_stop_offset", 1.5))
            replacement_stop_offset = float(params.get("replacement_stop_offset", 0.2))

            plan.enter_long[submit_idx, col] = True
            plan.enter_long_order_type[submit_idx, col] = ORDER_TYPE_STOP
            plan.enter_long_price[submit_idx, col] = close[submit_idx] + initial_stop_offset

            plan.cancel_pending[cancel_idx, col] = True

            plan.enter_long[replace_idx, col] = True
            plan.enter_long_order_type[replace_idx, col] = ORDER_TYPE_STOP
            plan.enter_long_price[replace_idx, col] = close[replace_idx] + replacement_stop_offset

            plan.exit_long[exit_idx, col] = True
            plan.long_targets[col] = target
        return plan


@dataclass(frozen=True)
class ScheduledLimitEntryVectorizedAdapter(VectorizedStrategyAdapter):
    strategy_name: str = "ScheduledLimitEntryStrategy"

    def supports(self, config: BacktestConfig, strategy_cls) -> VectorizedSupport:
        if strategy_cls.__name__ != self.strategy_name:
            return VectorizedSupport(
                False, f"Strategy {strategy_cls.__name__} is not handled by the scheduled limit adapter."
            )
        if config.base_execution:
            return VectorizedSupport(False, "Vectorized test adapter does not support base_execution.")
        if config.intrabar_sim:
            return VectorizedSupport(False, "Vectorized test adapter does not support intrabar simulation.")
        return VectorizedSupport(True)

    def build_order_plan(self, data: pd.DataFrame, param_grid, config: BacktestConfig) -> VectorizedOrderPlan:
        n_bars = len(data)
        n_params = len(param_grid)
        plan = _empty_plan(n_bars, n_params)
        close = data["close"].to_numpy(dtype=float)
        for col, params in enumerate(param_grid):
            submit_idx = int(params.get("submit_index", 8))
            cancel_idx = int(params.get("cancel_index", 16))
            replace_idx = int(params.get("replace_index", 18))
            exit_idx = int(params.get("exit_index", 90))
            target = abs(float(params.get("target", 1.0)))
            initial_limit_offset = float(params.get("initial_limit_offset", 0.4))
            replacement_limit_offset = float(params.get("replacement_limit_offset", 0.2))

            plan.enter_long[submit_idx, col] = True
            plan.enter_long_order_type[submit_idx, col] = ORDER_TYPE_LIMIT
            plan.enter_long_price[submit_idx, col] = close[submit_idx] - initial_limit_offset

            plan.cancel_pending[cancel_idx, col] = True

            plan.enter_long[replace_idx, col] = True
            plan.enter_long_order_type[replace_idx, col] = ORDER_TYPE_LIMIT
            plan.enter_long_price[replace_idx, col] = close[replace_idx] - replacement_limit_offset

            plan.exit_long[exit_idx, col] = True
            plan.long_targets[col] = target
        return plan


@dataclass(frozen=True)
class AfterFillProtectiveStopVectorizedAdapter(VectorizedStrategyAdapter):
    strategy_name: str = "AfterFillProtectiveStopStrategy"

    def supports(self, config: BacktestConfig, strategy_cls) -> VectorizedSupport:
        if strategy_cls.__name__ != self.strategy_name:
            return VectorizedSupport(
                False, f"Strategy {strategy_cls.__name__} is not handled by the after-fill protective stop adapter."
            )
        if config.base_execution:
            return VectorizedSupport(False, "Vectorized test adapter does not support base_execution.")
        if config.intrabar_sim:
            return VectorizedSupport(False, "Vectorized test adapter does not support intrabar simulation.")
        return VectorizedSupport(True)

    def supports_after_fill(self) -> bool:
        return True

    def build_order_plan(self, data: pd.DataFrame, param_grid, config: BacktestConfig) -> VectorizedOrderPlan:
        n_bars = len(data)
        n_params = len(param_grid)
        plan = _empty_plan(n_bars, n_params)
        for col, params in enumerate(param_grid):
            enter_idx = int(params.get("enter_index", 10))
            target = abs(float(params.get("target", 1.0)))
            plan.enter_long[enter_idx, col] = True
            plan.long_targets[col] = target
        return plan

    def build_after_fill_update(
        self,
        *,
        data: pd.DataFrame,
        bar_idx: int,
        param_grid,
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
        n_params = len(param_grid)
        cancel_pending = np.zeros(n_params, dtype=bool)
        submit_order = np.zeros(n_params, dtype=bool)
        order_qty = np.zeros(n_params, dtype=float)
        order_type = np.zeros(n_params, dtype=np.int8)
        order_price = np.full(n_params, np.nan, dtype=float)
        for col, params in enumerate(param_grid):
            if not bool(filled_mask[col]) or position[col] <= 1e-12:
                continue
            stop_offset = float(params.get("stop_offset", 0.4))
            cancel_pending[col] = True
            submit_order[col] = True
            order_qty[col] = -float(position[col])
            order_type[col] = ORDER_TYPE_STOP
            order_price[col] = float(avg_price[col]) - stop_offset
        return VectorizedPendingOrderUpdate(
            cancel_pending=cancel_pending,
            submit_order=submit_order,
            order_qty=order_qty,
            order_type=order_type,
            order_price=order_price,
        )


@dataclass(frozen=True)
class AfterFillSameBarTakeProfitVectorizedAdapter(VectorizedStrategyAdapter):
    strategy_name: str = "AfterFillSameBarTakeProfitStrategy"

    def supports(self, config: BacktestConfig, strategy_cls) -> VectorizedSupport:
        if strategy_cls.__name__ != self.strategy_name:
            return VectorizedSupport(
                False, f"Strategy {strategy_cls.__name__} is not handled by the after-fill same-bar take-profit adapter."
            )
        if config.base_execution:
            return VectorizedSupport(False, "Vectorized test adapter does not support base_execution.")
        if config.intrabar_sim:
            return VectorizedSupport(False, "Vectorized test adapter does not support intrabar simulation.")
        return VectorizedSupport(True)

    def supports_after_fill(self) -> bool:
        return True

    def build_order_plan(self, data: pd.DataFrame, param_grid, config: BacktestConfig) -> VectorizedOrderPlan:
        n_bars = len(data)
        n_params = len(param_grid)
        plan = _empty_plan(n_bars, n_params)
        for col, params in enumerate(param_grid):
            enter_idx = int(params.get("enter_index", 10))
            target = abs(float(params.get("target", 1.0)))
            plan.enter_long[enter_idx, col] = True
            plan.long_targets[col] = target
        return plan

    def build_after_fill_update(
        self,
        *,
        data: pd.DataFrame,
        bar_idx: int,
        param_grid,
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
        n_params = len(param_grid)
        cancel_pending = np.zeros(n_params, dtype=bool)
        submit_order = np.zeros(n_params, dtype=bool)
        order_qty = np.zeros(n_params, dtype=float)
        order_type = np.zeros(n_params, dtype=np.int8)
        order_price = np.full(n_params, np.nan, dtype=float)
        for col, params in enumerate(param_grid):
            if not bool(filled_mask[col]) or position[col] <= 1e-12 or np.any(pending_order_active[col, :]):
                continue
            limit_offset = float(params.get("limit_offset", 0.1))
            submit_order[col] = True
            order_qty[col] = -float(position[col])
            order_type[col] = ORDER_TYPE_LIMIT
            order_price[col] = float(avg_price[col]) + limit_offset
        return VectorizedPendingOrderUpdate(
            cancel_pending=cancel_pending,
            submit_order=submit_order,
            order_qty=order_qty,
            order_type=order_type,
            order_price=order_price,
        )


@dataclass(frozen=True)
class AfterFillBracketVectorizedAdapter(VectorizedStrategyAdapter):
    strategy_name: str = "AfterFillBracketStrategy"

    def supports(self, config: BacktestConfig, strategy_cls) -> VectorizedSupport:
        if strategy_cls.__name__ != self.strategy_name:
            return VectorizedSupport(
                False, f"Strategy {strategy_cls.__name__} is not handled by the after-fill bracket adapter."
            )
        if config.base_execution:
            return VectorizedSupport(False, "Vectorized test adapter does not support base_execution.")
        if config.intrabar_sim:
            return VectorizedSupport(False, "Vectorized test adapter does not support intrabar simulation.")
        return VectorizedSupport(True)

    def supports_after_fill(self) -> bool:
        return True

    def build_order_plan(self, data: pd.DataFrame, param_grid, config: BacktestConfig) -> VectorizedOrderPlan:
        n_bars = len(data)
        n_params = len(param_grid)
        plan = _empty_plan(n_bars, n_params)
        for col, params in enumerate(param_grid):
            enter_idx = int(params.get("enter_index", 10))
            target = abs(float(params.get("target", 1.0)))
            plan.enter_long[enter_idx, col] = True
            plan.long_targets[col] = target
        return plan

    def build_after_fill_update(
        self,
        *,
        data: pd.DataFrame,
        bar_idx: int,
        param_grid,
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
        n_params = len(param_grid)
        cancel_pending = np.zeros(n_params, dtype=bool)
        submit_order = np.zeros((n_params, 2), dtype=bool)
        order_qty = np.zeros((n_params, 2), dtype=float)
        order_type = np.zeros((n_params, 2), dtype=np.int8)
        order_price = np.full((n_params, 2), np.nan, dtype=float)
        for col, params in enumerate(param_grid):
            if not bool(filled_mask[col]):
                continue
            cancel_pending[col] = True
            if position[col] <= 1e-12:
                continue
            stop_offset = float(params.get("stop_offset", 0.4))
            limit_offset = float(params.get("limit_offset", 0.8))
            qty = -float(position[col])
            submit_order[col, 0] = True
            order_qty[col, 0] = qty
            order_type[col, 0] = ORDER_TYPE_STOP
            order_price[col, 0] = float(avg_price[col]) - stop_offset

            submit_order[col, 1] = True
            order_qty[col, 1] = qty
            order_type[col, 1] = ORDER_TYPE_LIMIT
            order_price[col, 1] = float(avg_price[col]) + limit_offset
        return VectorizedPendingOrderUpdate(
            cancel_pending=cancel_pending,
            submit_order=submit_order,
            order_qty=order_qty,
            order_type=order_type,
            order_price=order_price,
        )


@dataclass(frozen=True)
class AfterFillLadderedExitVectorizedAdapter(VectorizedStrategyAdapter):
    strategy_name: str = "AfterFillLadderedExitStrategy"

    def supports(self, config: BacktestConfig, strategy_cls) -> VectorizedSupport:
        if strategy_cls.__name__ != self.strategy_name:
            return VectorizedSupport(
                False, f"Strategy {strategy_cls.__name__} is not handled by the after-fill laddered-exit adapter."
            )
        if config.base_execution:
            return VectorizedSupport(False, "Vectorized test adapter does not support base_execution.")
        if config.intrabar_sim:
            return VectorizedSupport(False, "Vectorized test adapter does not support intrabar simulation.")
        return VectorizedSupport(True)

    def supports_after_fill(self) -> bool:
        return True

    def build_order_plan(self, data: pd.DataFrame, param_grid, config: BacktestConfig) -> VectorizedOrderPlan:
        n_bars = len(data)
        n_params = len(param_grid)
        plan = _empty_plan(n_bars, n_params)
        for col, params in enumerate(param_grid):
            enter_idx = int(params.get("enter_index", 10))
            target = abs(float(params.get("target", 1.0)))
            plan.enter_long[enter_idx, col] = True
            plan.long_targets[col] = target
        return plan

    def build_after_fill_update(
        self,
        *,
        data: pd.DataFrame,
        bar_idx: int,
        param_grid,
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
        n_params = len(param_grid)
        cancel_pending = np.zeros((n_params, 3), dtype=bool)
        submit_order = np.zeros((n_params, 3), dtype=bool)
        order_qty = np.zeros((n_params, 3), dtype=float)
        order_type = np.zeros((n_params, 3), dtype=np.int8)
        order_price = np.full((n_params, 3), np.nan, dtype=float)
        for col, params in enumerate(param_grid):
            if not bool(filled_mask[col]):
                continue
            stop_offset = float(params.get("stop_offset", 0.2))
            tp1_offset = float(params.get("tp1_offset", 0.3))
            tp2_offset = float(params.get("tp2_offset", 0.7))
            if position[col] <= 1e-12:
                cancel_pending[col, :] = True
                continue
            has_live_orders = bool(np.any(pending_order_active[col, :]))
            if not has_live_orders:
                qty = float(position[col])
                tp1_qty = qty / 2.0
                tp2_qty = qty - tp1_qty
                submit_order[col, 0] = True
                order_qty[col, 0] = -qty
                order_type[col, 0] = ORDER_TYPE_STOP
                order_price[col, 0] = float(avg_price[col]) - stop_offset

                submit_order[col, 1] = True
                order_qty[col, 1] = -tp1_qty
                order_type[col, 1] = ORDER_TYPE_LIMIT
                order_price[col, 1] = float(avg_price[col]) + tp1_offset

                submit_order[col, 2] = True
                order_qty[col, 2] = -tp2_qty
                order_type[col, 2] = ORDER_TYPE_LIMIT
                order_price[col, 2] = float(avg_price[col]) + tp2_offset
                continue

            cancel_pending[col, 0] = True
            submit_order[col, 0] = True
            order_qty[col, 0] = -float(position[col])
            order_type[col, 0] = ORDER_TYPE_STOP
            order_price[col, 0] = float(avg_price[col]) - stop_offset
        return VectorizedPendingOrderUpdate(
            cancel_pending=cancel_pending,
            submit_order=submit_order,
            order_qty=order_qty,
            order_type=order_type,
            order_price=order_price,
        )


class VectorizedBrokerSemanticsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._previous_adapter = vectorized_module._ADAPTERS.get("ScheduledShortStrategy")
        vectorized_module._ADAPTERS["ScheduledShortStrategy"] = ScheduledShortVectorizedAdapter()
        cls._previous_stop_adapter = vectorized_module._ADAPTERS.get("ScheduledStopEntryStrategy")
        vectorized_module._ADAPTERS["ScheduledStopEntryStrategy"] = ScheduledStopEntryVectorizedAdapter()
        cls._previous_limit_adapter = vectorized_module._ADAPTERS.get("ScheduledLimitEntryStrategy")
        vectorized_module._ADAPTERS["ScheduledLimitEntryStrategy"] = ScheduledLimitEntryVectorizedAdapter()
        cls._previous_after_fill_adapter = vectorized_module._ADAPTERS.get("AfterFillProtectiveStopStrategy")
        vectorized_module._ADAPTERS["AfterFillProtectiveStopStrategy"] = AfterFillProtectiveStopVectorizedAdapter()
        cls._previous_same_bar_tp_adapter = vectorized_module._ADAPTERS.get("AfterFillSameBarTakeProfitStrategy")
        vectorized_module._ADAPTERS["AfterFillSameBarTakeProfitStrategy"] = AfterFillSameBarTakeProfitVectorizedAdapter()
        cls._previous_bracket_adapter = vectorized_module._ADAPTERS.get("AfterFillBracketStrategy")
        vectorized_module._ADAPTERS["AfterFillBracketStrategy"] = AfterFillBracketVectorizedAdapter()
        cls._previous_laddered_adapter = vectorized_module._ADAPTERS.get("AfterFillLadderedExitStrategy")
        vectorized_module._ADAPTERS["AfterFillLadderedExitStrategy"] = AfterFillLadderedExitVectorizedAdapter()

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._previous_adapter is None:
            vectorized_module._ADAPTERS.pop("ScheduledShortStrategy", None)
        else:
            vectorized_module._ADAPTERS["ScheduledShortStrategy"] = cls._previous_adapter
        if cls._previous_stop_adapter is None:
            vectorized_module._ADAPTERS.pop("ScheduledStopEntryStrategy", None)
        else:
            vectorized_module._ADAPTERS["ScheduledStopEntryStrategy"] = cls._previous_stop_adapter
        if cls._previous_limit_adapter is None:
            vectorized_module._ADAPTERS.pop("ScheduledLimitEntryStrategy", None)
        else:
            vectorized_module._ADAPTERS["ScheduledLimitEntryStrategy"] = cls._previous_limit_adapter
        if cls._previous_after_fill_adapter is None:
            vectorized_module._ADAPTERS.pop("AfterFillProtectiveStopStrategy", None)
        else:
            vectorized_module._ADAPTERS["AfterFillProtectiveStopStrategy"] = cls._previous_after_fill_adapter
        if cls._previous_same_bar_tp_adapter is None:
            vectorized_module._ADAPTERS.pop("AfterFillSameBarTakeProfitStrategy", None)
        else:
            vectorized_module._ADAPTERS["AfterFillSameBarTakeProfitStrategy"] = cls._previous_same_bar_tp_adapter
        if cls._previous_bracket_adapter is None:
            vectorized_module._ADAPTERS.pop("AfterFillBracketStrategy", None)
        else:
            vectorized_module._ADAPTERS["AfterFillBracketStrategy"] = cls._previous_bracket_adapter
        if cls._previous_laddered_adapter is None:
            vectorized_module._ADAPTERS.pop("AfterFillLadderedExitStrategy", None)
        else:
            vectorized_module._ADAPTERS["AfterFillLadderedExitStrategy"] = cls._previous_laddered_adapter

    def test_vectorized_matches_reference_for_short_borrow(self) -> None:
        bars = _make_bars()
        config = BacktestConfig(
            timeframe="1 minutes",
            use_cache=False,
            base_execution=False,
            allow_short=True,
            borrow_rate=0.35,
            fee_rate=0.0002,
            slippage=0.0001,
        )
        params = {"target": 1.0, "enter_index": 10, "exit_index": 80}
        orchestrator = ExecutionOrchestrator()
        reference = orchestrator.execute(
            ExecutionRequest(
                data=bars,
                base_data=bars,
                dataset_id="scheduled_short",
                strategy_cls=ScheduledShortStrategy,
                strategy_params=params,
                config=config,
                requested_execution_mode=ExecutionMode.REFERENCE,
            )
        )
        vectorized = orchestrator.execute(
            ExecutionRequest(
                data=bars,
                base_data=bars,
                dataset_id="scheduled_short",
                strategy_cls=ScheduledShortStrategy,
                strategy_params=params,
                config=config,
                requested_execution_mode=ExecutionMode.VECTORIZED,
            )
        )

        self.assertEqual(reference.resolved_execution_mode.value, "reference")
        self.assertEqual(vectorized.resolved_execution_mode.value, "vectorized")
        self.assertEqual(len(reference.trades), len(vectorized.trades))
        self.assertTrue(np.allclose(reference.equity_curve.to_numpy(), vectorized.equity_curve.to_numpy(), atol=1e-8))
        self.assertAlmostEqual(reference.metrics.total_return, vectorized.metrics.total_return, places=10)
        self.assertAlmostEqual(reference.metrics.max_drawdown, vectorized.metrics.max_drawdown, places=10)

    def test_vectorized_matches_reference_for_stop_order_cancel_replace(self) -> None:
        bars = _make_bars()
        config = BacktestConfig(
            timeframe="1 minutes",
            use_cache=False,
            base_execution=False,
            allow_short=False,
            fee_rate=0.0002,
            slippage=0.0001,
        )
        params = {
            "target": 1.0,
            "submit_index": 8,
            "cancel_index": 16,
            "replace_index": 18,
            "exit_index": 90,
            "initial_stop_offset": 1.5,
            "replacement_stop_offset": 0.2,
        }
        orchestrator = ExecutionOrchestrator()
        reference = orchestrator.execute(
            ExecutionRequest(
                data=bars,
                base_data=bars,
                dataset_id="scheduled_stop_entry",
                strategy_cls=ScheduledStopEntryStrategy,
                strategy_params=params,
                config=config,
                requested_execution_mode=ExecutionMode.REFERENCE,
            )
        )
        vectorized = orchestrator.execute(
            ExecutionRequest(
                data=bars,
                base_data=bars,
                dataset_id="scheduled_stop_entry",
                strategy_cls=ScheduledStopEntryStrategy,
                strategy_params=params,
                config=config,
                requested_execution_mode=ExecutionMode.VECTORIZED,
            )
        )

        self.assertEqual(len(reference.trades), len(vectorized.trades))
        self.assertTrue(np.allclose(reference.equity_curve.to_numpy(), vectorized.equity_curve.to_numpy(), atol=1e-8))
        self.assertAlmostEqual(reference.metrics.total_return, vectorized.metrics.total_return, places=10)
        self.assertAlmostEqual(reference.metrics.max_drawdown, vectorized.metrics.max_drawdown, places=10)

    def test_vectorized_matches_reference_for_limit_order_cancel_replace(self) -> None:
        bars = _make_bars()
        config = BacktestConfig(
            timeframe="1 minutes",
            use_cache=False,
            base_execution=False,
            allow_short=False,
            fee_rate=0.0002,
            slippage=0.0001,
        )
        params = {
            "target": 1.0,
            "submit_index": 8,
            "cancel_index": 16,
            "replace_index": 18,
            "exit_index": 90,
            "initial_limit_offset": 0.4,
            "replacement_limit_offset": 0.2,
        }
        orchestrator = ExecutionOrchestrator()
        reference = orchestrator.execute(
            ExecutionRequest(
                data=bars,
                base_data=bars,
                dataset_id="scheduled_limit_entry",
                strategy_cls=ScheduledLimitEntryStrategy,
                strategy_params=params,
                config=config,
                requested_execution_mode=ExecutionMode.REFERENCE,
            )
        )
        vectorized = orchestrator.execute(
            ExecutionRequest(
                data=bars,
                base_data=bars,
                dataset_id="scheduled_limit_entry",
                strategy_cls=ScheduledLimitEntryStrategy,
                strategy_params=params,
                config=config,
                requested_execution_mode=ExecutionMode.VECTORIZED,
            )
        )

        self.assertEqual(len(reference.trades), len(vectorized.trades))
        self.assertTrue(np.allclose(reference.equity_curve.to_numpy(), vectorized.equity_curve.to_numpy(), atol=1e-8))
        self.assertAlmostEqual(reference.metrics.total_return, vectorized.metrics.total_return, places=10)
        self.assertAlmostEqual(reference.metrics.max_drawdown, vectorized.metrics.max_drawdown, places=10)

    def test_vectorized_matches_reference_for_recalc_on_fill_protective_stop(self) -> None:
        bars = _make_reversal_bars()
        config = BacktestConfig(
            timeframe="1 minutes",
            use_cache=False,
            base_execution=False,
            allow_short=False,
            fee_rate=0.0002,
            slippage=0.0001,
            recalc_on_fill=True,
            one_order_per_signal=False,
        )
        params = {
            "target": 1.0,
            "enter_index": 10,
            "stop_offset": 0.4,
        }
        orchestrator = ExecutionOrchestrator()
        reference = orchestrator.execute(
            ExecutionRequest(
                data=bars,
                base_data=bars,
                dataset_id="after_fill_protective_stop",
                strategy_cls=AfterFillProtectiveStopStrategy,
                strategy_params=params,
                config=config,
                requested_execution_mode=ExecutionMode.REFERENCE,
            )
        )
        vectorized = orchestrator.execute(
            ExecutionRequest(
                data=bars,
                base_data=bars,
                dataset_id="after_fill_protective_stop",
                strategy_cls=AfterFillProtectiveStopStrategy,
                strategy_params=params,
                config=config,
                requested_execution_mode=ExecutionMode.VECTORIZED,
            )
        )

        self.assertEqual(len(reference.trades), len(vectorized.trades))
        self.assertTrue(np.allclose(reference.equity_curve.to_numpy(), vectorized.equity_curve.to_numpy(), atol=1e-8))
        self.assertAlmostEqual(reference.metrics.total_return, vectorized.metrics.total_return, places=10)
        self.assertAlmostEqual(reference.metrics.max_drawdown, vectorized.metrics.max_drawdown, places=10)

    def test_vectorized_matches_reference_for_same_bar_recursive_take_profit(self) -> None:
        bars = _make_bars()
        config = BacktestConfig(
            timeframe="1 minutes",
            use_cache=False,
            base_execution=False,
            allow_short=False,
            fee_rate=0.0002,
            slippage=0.0001,
            recalc_on_fill=True,
            max_recalc_passes=4,
        )
        params = {
            "target": 1.0,
            "enter_index": 10,
            "limit_offset": 0.1,
        }
        orchestrator = ExecutionOrchestrator()
        reference = orchestrator.execute(
            ExecutionRequest(
                data=bars,
                base_data=bars,
                dataset_id="after_fill_same_bar_tp",
                strategy_cls=AfterFillSameBarTakeProfitStrategy,
                strategy_params=params,
                config=config,
                requested_execution_mode=ExecutionMode.REFERENCE,
            )
        )
        vectorized = orchestrator.execute(
            ExecutionRequest(
                data=bars,
                base_data=bars,
                dataset_id="after_fill_same_bar_tp",
                strategy_cls=AfterFillSameBarTakeProfitStrategy,
                strategy_params=params,
                config=config,
                requested_execution_mode=ExecutionMode.VECTORIZED,
            )
        )

        self.assertEqual(len(reference.trades), 2)
        self.assertEqual(len(reference.trades), len(vectorized.trades))
        self.assertEqual(reference.trades[0].timestamp, reference.trades[1].timestamp)
        self.assertEqual(vectorized.trades[0].timestamp, vectorized.trades[1].timestamp)
        self.assertTrue(np.allclose(reference.equity_curve.to_numpy(), vectorized.equity_curve.to_numpy(), atol=1e-8))
        self.assertAlmostEqual(reference.metrics.total_return, vectorized.metrics.total_return, places=10)
        self.assertAlmostEqual(reference.metrics.max_drawdown, vectorized.metrics.max_drawdown, places=10)

    def test_vectorized_matches_reference_for_after_fill_bracket_orders(self) -> None:
        bars = _make_bars()
        config = BacktestConfig(
            timeframe="1 minutes",
            use_cache=False,
            base_execution=False,
            allow_short=False,
            fee_rate=0.0002,
            slippage=0.0001,
            recalc_on_fill=True,
        )
        params = {
            "target": 1.0,
            "enter_index": 10,
            "stop_offset": 0.4,
            "limit_offset": 0.8,
        }
        orchestrator = ExecutionOrchestrator()
        reference = orchestrator.execute(
            ExecutionRequest(
                data=bars,
                base_data=bars,
                dataset_id="after_fill_bracket",
                strategy_cls=AfterFillBracketStrategy,
                strategy_params=params,
                config=config,
                requested_execution_mode=ExecutionMode.REFERENCE,
            )
        )
        vectorized = orchestrator.execute(
            ExecutionRequest(
                data=bars,
                base_data=bars,
                dataset_id="after_fill_bracket",
                strategy_cls=AfterFillBracketStrategy,
                strategy_params=params,
                config=config,
                requested_execution_mode=ExecutionMode.VECTORIZED,
            )
        )

        self.assertEqual(len(reference.trades), len(vectorized.trades))
        self.assertTrue(np.allclose(reference.equity_curve.to_numpy(), vectorized.equity_curve.to_numpy(), atol=1e-8))
        self.assertAlmostEqual(reference.metrics.total_return, vectorized.metrics.total_return, places=10)
        self.assertAlmostEqual(reference.metrics.max_drawdown, vectorized.metrics.max_drawdown, places=10)

    def test_vectorized_matches_reference_for_laddered_exit_stop_refresh(self) -> None:
        bars = _make_reversal_bars()
        config = BacktestConfig(
            timeframe="1 minutes",
            use_cache=False,
            base_execution=False,
            allow_short=False,
            fee_rate=0.0002,
            slippage=0.0001,
            recalc_on_fill=True,
        )
        params = {
            "target": 1.0,
            "enter_index": 10,
            "stop_offset": 0.2,
            "tp1_offset": 0.3,
            "tp2_offset": 0.7,
        }
        orchestrator = ExecutionOrchestrator()
        reference = orchestrator.execute(
            ExecutionRequest(
                data=bars,
                base_data=bars,
                dataset_id="after_fill_laddered_exit",
                strategy_cls=AfterFillLadderedExitStrategy,
                strategy_params=params,
                config=config,
                requested_execution_mode=ExecutionMode.REFERENCE,
            )
        )
        vectorized = orchestrator.execute(
            ExecutionRequest(
                data=bars,
                base_data=bars,
                dataset_id="after_fill_laddered_exit",
                strategy_cls=AfterFillLadderedExitStrategy,
                strategy_params=params,
                config=config,
                requested_execution_mode=ExecutionMode.VECTORIZED,
            )
        )

        self.assertEqual(len(reference.trades), len(vectorized.trades))
        self.assertTrue(np.allclose(reference.equity_curve.to_numpy(), vectorized.equity_curve.to_numpy(), atol=1e-8))
        self.assertAlmostEqual(reference.metrics.total_return, vectorized.metrics.total_return, places=10)
        self.assertAlmostEqual(reference.metrics.max_drawdown, vectorized.metrics.max_drawdown, places=10)


if __name__ == "__main__":
    unittest.main()
