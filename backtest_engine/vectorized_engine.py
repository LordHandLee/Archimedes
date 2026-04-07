from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Dict, Iterable, List, Sequence, Type

import numpy as np
import pandas as pd

from .broker import Trade
from .catalog import ResultCatalog
from .engine import BacktestConfig, BacktestResult
from .execution import BatchExecutionBenchmark, ExecutionMode, ExecutionRequest, ExecutionResult
from .metrics import compute_metrics
from .run_ids import compute_engine_run_id, compute_logical_run_id
from .strategy import Strategy
from .vectorized_strategies import VectorizedPendingOrderUpdate, VectorizedSupport, get_vectorized_adapter


@dataclass(frozen=True)
class VectorizedEngine:
    """
    Narrow v1 vectorized backend.

    Supported semantics:
    - single asset
    - same signal/execution timeframe
    - SMACrossStrategy
    - long-only
    - bar-close signal generation
    - next-bar-open fills
    """

    engine_impl: str = "vectorized"
    engine_version: str = "1"
    pending_order_slots: int = 4
    _last_batch_benchmark: BatchExecutionBenchmark | None = field(default=None, init=False, repr=False, compare=False)

    def supports(self, request: ExecutionRequest) -> VectorizedSupport:
        return self.supports_param_grid(
            data=request.data,
            dataset_id=request.dataset_id,
            strategy_cls=request.strategy_cls,
            param_grid=[request.strategy_params],
            config=request.config,
            base_data=request.base_data,
        )

    def supports_param_grid(
        self,
        *,
        data: pd.DataFrame,
        dataset_id: str,
        strategy_cls: Type[Strategy],
        param_grid: Sequence[Dict],
        config: BacktestConfig,
        base_data: pd.DataFrame | None = None,
    ) -> VectorizedSupport:
        if base_data is not None and config.base_execution:
            return VectorizedSupport(False, "Vectorized v1 does not support base_execution.")
        adapter = get_vectorized_adapter(strategy_cls)
        if adapter is None:
            return VectorizedSupport(False, f"Strategy {strategy_cls.__name__} has no vectorized adapter.")
        support = adapter.supports(config, strategy_cls)
        if not support.supported:
            return support
        if config.recalc_on_fill and strategy_cls.on_after_fill is not Strategy.on_after_fill and not adapter.supports_after_fill():
            return VectorizedSupport(
                False,
                f"Strategy {strategy_cls.__name__} requires after-fill recalculation support that its vectorized adapter does not provide.",
            )
        if not param_grid:
            return VectorizedSupport(False, "Parameter grid is empty.")
        return VectorizedSupport(True)

    def execute(
        self,
        request: ExecutionRequest,
        *,
        requested_mode: ExecutionMode,
        resolved_mode: ExecutionMode,
        fallback_reason: str | None = None,
    ) -> ExecutionResult:
        results = self.execute_param_grid(
            data=request.data,
            dataset_id=request.dataset_id,
            strategy_cls=request.strategy_cls,
            param_grid=[request.strategy_params],
            catalog=request.catalog,
            config=request.config,
            base_data=request.base_data,
            requested_mode=requested_mode,
            resolved_mode=resolved_mode,
            fallback_reason=fallback_reason,
            logical_run_ids=[request.logical_run_id] if request.logical_run_id is not None else None,
        )
        return results[0]

    def execute_param_grid(
        self,
        *,
        data: pd.DataFrame,
        dataset_id: str,
        strategy_cls: Type[Strategy],
        param_grid: Sequence[Dict],
        catalog: ResultCatalog | None,
        config: BacktestConfig,
        requested_mode: ExecutionMode,
        resolved_mode: ExecutionMode,
        base_data: pd.DataFrame | None = None,
        fallback_reason: str | None = None,
        logical_run_ids: Sequence[str | None] | None = None,
    ) -> List[ExecutionResult]:
        started = perf_counter()
        object.__setattr__(self, "_last_batch_benchmark", None)
        support = self.supports_param_grid(
            data=data,
            dataset_id=dataset_id,
            strategy_cls=strategy_cls,
            param_grid=param_grid,
            config=config,
            base_data=base_data,
        )
        if not support.supported:
            raise RuntimeError(support.reason or "Vectorized execution is not supported for this workload.")

        sliced_data = self._slice_data(self._normalize_data(data), config)
        adapter = get_vectorized_adapter(strategy_cls)
        assert adapter is not None

        param_list = list(param_grid)
        requested_mode = ExecutionMode.from_value(requested_mode)
        resolved_mode = ExecutionMode.from_value(resolved_mode)

        if logical_run_ids is None:
            logical_ids: List[str | None] = [None] * len(param_list)
        else:
            logical_ids = list(logical_run_ids)
            if len(logical_ids) != len(param_list):
                raise ValueError("logical_run_ids length must match the parameter grid length.")

        run_keys: List[tuple[str, str]] = []
        for i, params in enumerate(param_list):
            logical_run_id = logical_ids[i] or compute_logical_run_id(
                dataset_id=dataset_id,
                strategy=strategy_cls.__name__,
                params=params,
                config=config,
                data=sliced_data,
            )
            run_id = compute_engine_run_id(logical_run_id, self.engine_impl, self.engine_version)
            logical_ids[i] = logical_run_id
            run_keys.append((logical_run_id, run_id))

        results: List[ExecutionResult | None] = [None] * len(param_list)
        uncached_indices: List[int] = []
        uncached_params: List[Dict] = []
        chunk_sizes: List[int] = []

        for i, params in enumerate(param_list):
            logical_run_id, run_id = run_keys[i]
            cached = catalog.fetch(run_id) if config.use_cache and catalog else None
            if cached:
                dummy_equity = pd.Series([], dtype=float)
                result = BacktestResult(
                    run_id=run_id,
                    equity_curve=dummy_equity,
                    trades=[],
                    metrics=cached.metrics,
                    cached=True,
                )
                results[i] = ExecutionResult(
                    result=result,
                    logical_run_id=logical_run_id,
                    requested_execution_mode=requested_mode,
                    resolved_execution_mode=resolved_mode,
                    engine_impl=self.engine_impl,
                    engine_version=self.engine_version,
                    fallback_reason=fallback_reason,
                )
                continue
            uncached_indices.append(i)
            uncached_params.append(params)

        if uncached_indices:
            run_started_at = pd.Timestamp.now("UTC").isoformat()
            prepared_context = adapter.prepare_order_plan_context(sliced_data, uncached_params, config)
            for chunk_start, chunk_stop in self._iter_param_batches(
                total=len(uncached_indices),
                n_bars=len(sliced_data),
                config=config,
            ):
                chunk_indices = uncached_indices[chunk_start:chunk_stop]
                chunk_params = uncached_params[chunk_start:chunk_stop]
                chunk_sizes.append(len(chunk_params))
                order_plan = adapter.build_order_plan_from_context(
                    sliced_data,
                    chunk_params,
                    config,
                    prepared_context=prepared_context,
                )
                simulations = self._simulate_batch(
                    data=sliced_data,
                    adapter=adapter,
                    order_plan=order_plan,
                    param_grid=chunk_params,
                    config=config,
                )
                run_finished_at = pd.Timestamp.now("UTC").isoformat()
                for local_idx, sim in enumerate(simulations):
                    outer_idx = chunk_indices[local_idx]
                    logical_run_id, run_id = run_keys[outer_idx]
                    if catalog:
                        catalog.save(
                            run_id=run_id,
                            batch_id=config.batch_id,
                            strategy=strategy_cls.__name__,
                            params=param_list[outer_idx],
                            timeframe=config.timeframe,
                            start=str(sliced_data.index[0]),
                            end=str(sliced_data.index[-1]),
                            dataset_id=dataset_id,
                            starting_cash=config.starting_cash,
                            metrics=sim.metrics,
                            run_started_at=run_started_at,
                            run_finished_at=run_finished_at,
                            status="finished",
                            logical_run_id=logical_run_id,
                            requested_execution_mode=requested_mode.value,
                            resolved_execution_mode=resolved_mode.value,
                            engine_impl=self.engine_impl,
                            engine_version=self.engine_version,
                            fallback_reason=fallback_reason,
                        )
                        catalog.save_trades(run_id, sim.trades)
                    result = BacktestResult(
                        run_id=run_id,
                        equity_curve=sim.equity_curve,
                        trades=sim.trades,
                        metrics=sim.metrics,
                        cached=False,
                    )
                    results[outer_idx] = ExecutionResult(
                        result=result,
                        logical_run_id=logical_run_id,
                        requested_execution_mode=requested_mode,
                        resolved_execution_mode=resolved_mode,
                        engine_impl=self.engine_impl,
                        engine_version=self.engine_version,
                        fallback_reason=fallback_reason,
                    )

        output = [result for result in results if result is not None]
        duration_seconds = perf_counter() - started
        chunk_sizes_tuple = tuple(chunk_sizes) if uncached_indices else ()
        object.__setattr__(
            self,
            "_last_batch_benchmark",
            BatchExecutionBenchmark(
                dataset_id=dataset_id,
                strategy=strategy_cls.__name__,
                timeframe=config.timeframe,
                requested_execution_mode=requested_mode,
                resolved_execution_mode=resolved_mode,
                engine_impl=self.engine_impl,
                engine_version=self.engine_version,
                bars=len(sliced_data),
                total_params=len(param_list),
                cached_runs=len(param_list) - len(uncached_indices),
                uncached_runs=len(uncached_indices),
                duration_seconds=duration_seconds,
                chunk_count=len(chunk_sizes_tuple),
                chunk_sizes=chunk_sizes_tuple,
                effective_param_batch_size=max(chunk_sizes_tuple) if chunk_sizes_tuple else None,
                prepared_context_reused=bool(uncached_indices and len(chunk_sizes_tuple) > 1),
            ),
        )
        return output

    @property
    def last_batch_benchmark(self) -> BatchExecutionBenchmark | None:
        return self._last_batch_benchmark

    def _simulate_batch(
        self,
        *,
        data: pd.DataFrame,
        adapter,
        order_plan,
        param_grid: Sequence[Dict],
        config: BacktestConfig,
    ) -> List[BacktestResult]:
        opens = data["open"].to_numpy(dtype=float)
        highs = data["high"].to_numpy(dtype=float)
        lows = data["low"].to_numpy(dtype=float)
        closes = data["close"].to_numpy(dtype=float)
        index = data.index
        n_bars = len(index)
        n_params = len(param_grid)

        cash = np.full(n_params, float(config.starting_cash), dtype=float)
        position = np.zeros(n_params, dtype=float)
        avg_price = np.zeros(n_params, dtype=float)
        realized_pnl = np.zeros(n_params, dtype=float)
        pending_qty = np.zeros((n_params, self.pending_order_slots), dtype=float)
        pending_order_type = np.zeros((n_params, self.pending_order_slots), dtype=np.int8)
        pending_order_price = np.full((n_params, self.pending_order_slots), np.nan, dtype=float)
        pending_order_active = np.zeros((n_params, self.pending_order_slots), dtype=bool)
        equity = np.empty((n_bars, n_params), dtype=float)
        trades_by_param: List[List[Trade]] = [[] for _ in range(n_params)]
        last_timestamp: pd.Timestamp | None = None

        buy_fee = self._side_rate(config.fee_schedule, config.fee_rate, "buy")
        sell_fee = self._side_rate(config.fee_schedule, config.fee_rate, "sell")
        buy_slip = self._side_rate(config.slippage_schedule, config.slippage, "buy")
        sell_slip = self._side_rate(config.slippage_schedule, config.slippage, "sell")
        eps = 1e-12

        for bar_idx, ts in enumerate(index):
            open_px = float(opens[bar_idx])
            high_px = float(highs[bar_idx])
            low_px = float(lows[bar_idx])
            close_px = float(closes[bar_idx])
            if not config.fill_on_close:
                self._execute_with_recalc_loop(
                    adapter=adapter,
                    data=data,
                    bar_idx=bar_idx,
                    ts=ts,
                    open_px=open_px,
                    high_px=high_px,
                    low_px=low_px,
                    close_px=close_px,
                    cash=cash,
                    position=position,
                    avg_price=avg_price,
                    realized_pnl=realized_pnl,
                    pending_qty=pending_qty,
                    pending_order_type=pending_order_type,
                    pending_order_price=pending_order_price,
                    pending_order_active=pending_order_active,
                    trades_by_param=trades_by_param,
                    config=config,
                    buy_fee=buy_fee,
                    sell_fee=sell_fee,
                    buy_slip=buy_slip,
                    sell_slip=sell_slip,
                    param_grid=param_grid,
                )
                equity_close = self._record_equity_step(
                    timestamp=ts,
                    last_timestamp=last_timestamp,
                    mark_price=close_px,
                    cash=cash,
                    position=position,
                    equity_store=equity,
                    row_index=bar_idx,
                    config=config,
                )
                new_order_qty = self._compute_pending_qty(
                    order_plan=order_plan,
                    bar_idx=bar_idx,
                    equity_mark=equity_close,
                    position=position,
                    mark_price=close_px,
                )
                self._update_pending_orders(
                    order_plan=order_plan,
                    bar_idx=bar_idx,
                    new_order_qty=new_order_qty,
                    pending_qty=pending_qty,
                    pending_order_type=pending_order_type,
                    pending_order_price=pending_order_price,
                    pending_order_active=pending_order_active,
                )
                if config.one_order_per_signal:
                    self._prune_pending_orders(
                        position=position,
                        pending_qty=pending_qty,
                        pending_order_type=pending_order_type,
                        pending_order_price=pending_order_price,
                        pending_order_active=pending_order_active,
                    )
                last_timestamp = ts
                continue

            equity_signal = self._accrue_borrow_and_mark(
                timestamp=ts,
                last_timestamp=last_timestamp,
                mark_price=close_px,
                cash=cash,
                position=position,
                config=config,
            )
            new_order_qty = self._compute_pending_qty(
                order_plan=order_plan,
                bar_idx=bar_idx,
                equity_mark=equity_signal,
                position=position,
                mark_price=close_px,
            )
            self._update_pending_orders(
                order_plan=order_plan,
                bar_idx=bar_idx,
                new_order_qty=new_order_qty,
                pending_qty=pending_qty,
                pending_order_type=pending_order_type,
                pending_order_price=pending_order_price,
                pending_order_active=pending_order_active,
            )
            if config.one_order_per_signal:
                self._prune_pending_orders(
                    position=position,
                    pending_qty=pending_qty,
                    pending_order_type=pending_order_type,
                    pending_order_price=pending_order_price,
                    pending_order_active=pending_order_active,
                )
            self._execute_with_recalc_loop(
                adapter=adapter,
                data=data,
                bar_idx=bar_idx,
                ts=ts,
                open_px=close_px,
                high_px=close_px,
                low_px=close_px,
                close_px=close_px,
                cash=cash,
                position=position,
                avg_price=avg_price,
                realized_pnl=realized_pnl,
                pending_qty=pending_qty,
                pending_order_type=pending_order_type,
                pending_order_price=pending_order_price,
                pending_order_active=pending_order_active,
                trades_by_param=trades_by_param,
                config=config,
                buy_fee=buy_fee,
                sell_fee=sell_fee,
                buy_slip=buy_slip,
                sell_slip=sell_slip,
                param_grid=param_grid,
            )
            equity[bar_idx, :] = cash + position * close_px
            last_timestamp = ts

        session_seconds = config.sharpe_session_seconds_per_day
        if session_seconds is None and config.sharpe_annualization == "equities" and config.sharpe_basis != "daily":
            session_seconds = self._estimate_session_seconds_per_day(data)

        results: List[BacktestResult] = []
        for idx in range(n_params):
            equity_curve = pd.Series(equity[:, idx], index=index)
            metrics = compute_metrics(
                equity_curve,
                risk_free_rate=config.risk_free_rate,
                timeframe=config.timeframe,
                annualization=config.sharpe_annualization,
                session_seconds_per_day=session_seconds,
                sharpe_basis=config.sharpe_basis,
            )
            results.append(
                BacktestResult(
                    run_id="",
                    equity_curve=equity_curve,
                    trades=trades_by_param[idx],
                    metrics=metrics,
                    cached=False,
                )
            )
        return results

    def _prune_pending_orders(
        self,
        *,
        position: np.ndarray,
        pending_qty: np.ndarray,
        pending_order_type: np.ndarray,
        pending_order_price: np.ndarray,
        pending_order_active: np.ndarray,
    ) -> None:
        for param_idx in range(pending_order_active.shape[0]):
            keep_slot = None
            pos = float(position[param_idx])
            for slot_idx in range(pending_order_active.shape[1]):
                if not pending_order_active[param_idx, slot_idx]:
                    continue
                qty = float(pending_qty[param_idx, slot_idx])
                if pos == 0.0:
                    keep_slot = slot_idx
                    break
                if pos > 0.0 and qty < 0.0:
                    keep_slot = slot_idx
                    break
                if pos < 0.0 and qty > 0.0:
                    keep_slot = slot_idx
                    break
            if keep_slot is None:
                pending_order_active[param_idx, :] = False
                pending_qty[param_idx, :] = 0.0
                pending_order_type[param_idx, :] = 0
                pending_order_price[param_idx, :] = np.nan
                continue
            for slot_idx in range(pending_order_active.shape[1]):
                if slot_idx == keep_slot:
                    continue
                pending_order_active[param_idx, slot_idx] = False
                pending_qty[param_idx, slot_idx] = 0.0
                pending_order_type[param_idx, slot_idx] = 0
                pending_order_price[param_idx, slot_idx] = np.nan

    def _execute_with_recalc_loop(
        self,
        *,
        adapter,
        data: pd.DataFrame,
        bar_idx: int,
        ts: pd.Timestamp,
        open_px: float,
        high_px: float,
        low_px: float,
        close_px: float,
        cash: np.ndarray,
        position: np.ndarray,
        avg_price: np.ndarray,
        realized_pnl: np.ndarray,
        pending_qty: np.ndarray,
        pending_order_type: np.ndarray,
        pending_order_price: np.ndarray,
        pending_order_active: np.ndarray,
        trades_by_param: List[List[Trade]],
        config: BacktestConfig,
        buy_fee: float,
        sell_fee: float,
        buy_slip: float,
        sell_slip: float,
        param_grid: Sequence[Dict],
    ) -> None:
        filled_slot_mask = self._execute_pending_orders(
            ts=ts,
            open_px=open_px,
            high_px=high_px,
            low_px=low_px,
            close_px=close_px,
            cash=cash,
            position=position,
            avg_price=avg_price,
            realized_pnl=realized_pnl,
            pending_qty=pending_qty,
            pending_order_type=pending_order_type,
            pending_order_price=pending_order_price,
            pending_order_active=pending_order_active,
            trades_by_param=trades_by_param,
            config=config,
            buy_fee=buy_fee,
            sell_fee=sell_fee,
            buy_slip=buy_slip,
            sell_slip=sell_slip,
        )
        if not config.recalc_on_fill:
            return
        passes = 0
        while np.any(filled_slot_mask):
            filled_mask = np.any(filled_slot_mask, axis=1)
            self._apply_pending_order_update(
                update=adapter.build_after_fill_update(
                    data=data,
                    bar_idx=bar_idx,
                    param_grid=list(param_grid),
                    config=config,
                    filled_mask=filled_mask,
                    cash=cash,
                    position=position,
                    avg_price=avg_price,
                    pending_qty=pending_qty,
                    pending_order_type=pending_order_type,
                    pending_order_price=pending_order_price,
                    pending_order_active=pending_order_active,
                ),
                pending_qty=pending_qty,
                pending_order_type=pending_order_type,
                pending_order_price=pending_order_price,
                pending_order_active=pending_order_active,
            )
            if passes >= max(int(config.max_recalc_passes), 0):
                break
            filled_slot_mask = self._execute_pending_orders(
                ts=ts,
                open_px=open_px,
                high_px=high_px,
                low_px=low_px,
                close_px=close_px,
                cash=cash,
                position=position,
                avg_price=avg_price,
                realized_pnl=realized_pnl,
                pending_qty=pending_qty,
                pending_order_type=pending_order_type,
                pending_order_price=pending_order_price,
                pending_order_active=pending_order_active,
                trades_by_param=trades_by_param,
                config=config,
                buy_fee=buy_fee,
                sell_fee=sell_fee,
                buy_slip=buy_slip,
                sell_slip=sell_slip,
            )
            passes += 1

    def _compute_pending_qty(
        self,
        order_plan,
        bar_idx: int,
        equity_mark: np.ndarray,
        position: np.ndarray,
        mark_price: float,
    ) -> np.ndarray:
        if mark_price <= 0:
            return np.zeros_like(position, dtype=float)
        pending_qty = np.zeros_like(position, dtype=float)
        pos_eps = 1e-12
        order_eps = 1e-9

        exit_long_mask = order_plan.exit_long[bar_idx, :] & (position > pos_eps)
        if np.any(exit_long_mask):
            pending_qty[exit_long_mask] = -position[exit_long_mask]

        exit_short_mask = order_plan.exit_short[bar_idx, :] & (position < -pos_eps)
        if np.any(exit_short_mask):
            pending_qty[exit_short_mask] = -position[exit_short_mask]

        enter_long_mask = order_plan.enter_long[bar_idx, :] & (position <= pos_eps) & (np.abs(pending_qty) < pos_eps)
        if np.any(enter_long_mask):
            desired_long_qty = (order_plan.long_targets[enter_long_mask] * equity_mark[enter_long_mask]) / mark_price
            pending_qty[enter_long_mask] = desired_long_qty - position[enter_long_mask]

        enter_short_mask = order_plan.enter_short[bar_idx, :] & (position >= -pos_eps) & (np.abs(pending_qty) < pos_eps)
        if np.any(enter_short_mask):
            desired_short_qty = (-order_plan.short_targets[enter_short_mask] * equity_mark[enter_short_mask]) / mark_price
            pending_qty[enter_short_mask] = desired_short_qty - position[enter_short_mask]

        pending_qty[np.abs(pending_qty) < order_eps] = 0.0
        return pending_qty

    def _update_pending_orders(
        self,
        *,
        order_plan,
        bar_idx: int,
        new_order_qty: np.ndarray,
        pending_qty: np.ndarray,
        pending_order_type: np.ndarray,
        pending_order_price: np.ndarray,
        pending_order_active: np.ndarray,
    ) -> None:
        cancel_mask = order_plan.cancel_pending[bar_idx, :]
        if np.any(cancel_mask):
            self._clear_pending_params(
                param_mask=cancel_mask,
                pending_qty=pending_qty,
                pending_order_type=pending_order_type,
                pending_order_price=pending_order_price,
                pending_order_active=pending_order_active,
            )

        active_mask = np.abs(new_order_qty) > 1e-12
        if not np.any(active_mask):
            return

        exit_long_mask = order_plan.exit_long[bar_idx, :] & active_mask
        exit_short_mask = order_plan.exit_short[bar_idx, :] & active_mask & ~exit_long_mask
        enter_long_mask = order_plan.enter_long[bar_idx, :] & active_mask & ~exit_long_mask & ~exit_short_mask
        enter_short_mask = order_plan.enter_short[bar_idx, :] & active_mask & ~exit_long_mask & ~exit_short_mask & ~enter_long_mask

        if np.any(exit_long_mask):
            self._submit_single_orders(
                param_mask=exit_long_mask,
                order_qty=new_order_qty,
                order_type=order_plan.exit_long_order_type[bar_idx, :],
                order_price=order_plan.exit_long_price[bar_idx, :],
                pending_qty=pending_qty,
                pending_order_type=pending_order_type,
                pending_order_price=pending_order_price,
                pending_order_active=pending_order_active,
            )
        if np.any(exit_short_mask):
            self._submit_single_orders(
                param_mask=exit_short_mask,
                order_qty=new_order_qty,
                order_type=order_plan.exit_short_order_type[bar_idx, :],
                order_price=order_plan.exit_short_price[bar_idx, :],
                pending_qty=pending_qty,
                pending_order_type=pending_order_type,
                pending_order_price=pending_order_price,
                pending_order_active=pending_order_active,
            )
        if np.any(enter_long_mask):
            self._submit_single_orders(
                param_mask=enter_long_mask,
                order_qty=new_order_qty,
                order_type=order_plan.enter_long_order_type[bar_idx, :],
                order_price=order_plan.enter_long_price[bar_idx, :],
                pending_qty=pending_qty,
                pending_order_type=pending_order_type,
                pending_order_price=pending_order_price,
                pending_order_active=pending_order_active,
            )
        if np.any(enter_short_mask):
            self._submit_single_orders(
                param_mask=enter_short_mask,
                order_qty=new_order_qty,
                order_type=order_plan.enter_short_order_type[bar_idx, :],
                order_price=order_plan.enter_short_price[bar_idx, :],
                pending_qty=pending_qty,
                pending_order_type=pending_order_type,
                pending_order_price=pending_order_price,
                pending_order_active=pending_order_active,
            )

    def _apply_pending_order_update(
        self,
        *,
        update: VectorizedPendingOrderUpdate | None,
        pending_qty: np.ndarray,
        pending_order_type: np.ndarray,
        pending_order_price: np.ndarray,
        pending_order_active: np.ndarray,
    ) -> None:
        if update is None:
            return
        cancel_mask = np.asarray(update.cancel_pending, dtype=bool)
        if cancel_mask.ndim == 1:
            if cancel_mask.shape[0] != pending_order_active.shape[0]:
                raise ValueError("after-fill cancel_pending 1D shape must match param count")
            if np.any(cancel_mask):
                self._clear_pending_params(
                    param_mask=cancel_mask,
                    pending_qty=pending_qty,
                    pending_order_type=pending_order_type,
                    pending_order_price=pending_order_price,
                    pending_order_active=pending_order_active,
                )
        elif cancel_mask.shape == pending_order_active.shape:
            if np.any(cancel_mask):
                pending_order_active[cancel_mask] = False
                pending_qty[cancel_mask] = 0.0
                pending_order_type[cancel_mask] = 0
                pending_order_price[cancel_mask] = np.nan
        elif cancel_mask.ndim == 2 and cancel_mask.shape[0] == pending_order_active.shape[0] and cancel_mask.shape[1] <= pending_order_active.shape[1]:
            expanded_cancel_mask = np.zeros_like(pending_order_active, dtype=bool)
            expanded_cancel_mask[:, : cancel_mask.shape[1]] = cancel_mask
            if np.any(expanded_cancel_mask):
                pending_order_active[expanded_cancel_mask] = False
                pending_qty[expanded_cancel_mask] = 0.0
                pending_order_type[expanded_cancel_mask] = 0
                pending_order_price[expanded_cancel_mask] = np.nan
        else:
            raise ValueError("after-fill cancel_pending shape must match pending-order state")

        submit_mask = np.asarray(update.submit_order, dtype=bool)
        if not np.any(submit_mask):
            return
        order_qty = np.asarray(update.order_qty, dtype=float)
        order_type = np.asarray(update.order_type, dtype=np.int8)
        order_price = np.asarray(update.order_price, dtype=float)
        if submit_mask.ndim == 1:
            if submit_mask.shape[0] != pending_order_active.shape[0]:
                raise ValueError("after-fill submit_order 1D shape must match param count")
            self._submit_single_orders(
                param_mask=submit_mask,
                order_qty=order_qty,
                order_type=order_type,
                order_price=order_price,
                pending_qty=pending_qty,
                pending_order_type=pending_order_type,
                pending_order_price=pending_order_price,
                pending_order_active=pending_order_active,
            )
            return
        if submit_mask.shape == pending_order_active.shape:
            expanded_submit_mask = submit_mask
            expanded_order_qty = order_qty
            expanded_order_type = order_type
            expanded_order_price = order_price
        elif submit_mask.shape[0] == pending_order_active.shape[0] and submit_mask.shape[1] <= pending_order_active.shape[1]:
            expanded_submit_mask = np.zeros_like(pending_order_active, dtype=bool)
            expanded_submit_mask[:, : submit_mask.shape[1]] = submit_mask
            expanded_order_qty = np.zeros_like(pending_qty, dtype=float)
            expanded_order_type = np.zeros_like(pending_order_type, dtype=np.int8)
            expanded_order_price = np.full_like(pending_order_price, np.nan, dtype=float)
            if order_qty.shape != submit_mask.shape:
                raise ValueError("after-fill order_qty shape must match submit_order shape for slot-level submissions")
            if order_type.shape != submit_mask.shape:
                raise ValueError("after-fill order_type shape must match submit_order shape for slot-level submissions")
            if order_price.shape != submit_mask.shape:
                raise ValueError("after-fill order_price shape must match submit_order shape for slot-level submissions")
            expanded_order_qty[:, : submit_mask.shape[1]] = order_qty
            expanded_order_type[:, : submit_mask.shape[1]] = order_type
            expanded_order_price[:, : submit_mask.shape[1]] = order_price
        else:
            raise ValueError("after-fill submit_order shape must match pending-order state")
        pending_order_active[expanded_submit_mask] = True
        pending_qty[expanded_submit_mask] = expanded_order_qty[expanded_submit_mask]
        pending_order_type[expanded_submit_mask] = expanded_order_type[expanded_submit_mask]
        pending_order_price[expanded_submit_mask] = expanded_order_price[expanded_submit_mask]

    def _clear_pending_params(
        self,
        *,
        param_mask: np.ndarray,
        pending_qty: np.ndarray,
        pending_order_type: np.ndarray,
        pending_order_price: np.ndarray,
        pending_order_active: np.ndarray,
    ) -> None:
        if not np.any(param_mask):
            return
        pending_order_active[param_mask, :] = False
        pending_qty[param_mask, :] = 0.0
        pending_order_type[param_mask, :] = 0
        pending_order_price[param_mask, :] = np.nan

    def _submit_single_orders(
        self,
        *,
        param_mask: np.ndarray,
        order_qty: np.ndarray,
        order_type: np.ndarray,
        order_price: np.ndarray,
        pending_qty: np.ndarray,
        pending_order_type: np.ndarray,
        pending_order_price: np.ndarray,
        pending_order_active: np.ndarray,
    ) -> None:
        for param_idx in np.flatnonzero(param_mask):
            free_slots = np.flatnonzero(~pending_order_active[param_idx, :])
            slot_idx = int(free_slots[0]) if free_slots.size else 0
            pending_order_active[param_idx, slot_idx] = True
            pending_qty[param_idx, slot_idx] = float(order_qty[param_idx])
            pending_order_type[param_idx, slot_idx] = int(order_type[param_idx])
            pending_order_price[param_idx, slot_idx] = float(order_price[param_idx])

    def _record_equity_step(
        self,
        *,
        timestamp: pd.Timestamp,
        last_timestamp: pd.Timestamp | None,
        mark_price: float,
        cash: np.ndarray,
        position: np.ndarray,
        equity_store: np.ndarray,
        row_index: int,
        config: BacktestConfig,
    ) -> np.ndarray:
        equity_close = self._accrue_borrow_and_mark(
            timestamp=timestamp,
            last_timestamp=last_timestamp,
            mark_price=mark_price,
            cash=cash,
            position=position,
            config=config,
        )
        equity_store[row_index, :] = equity_close
        return equity_close

    def _accrue_borrow_and_mark(
        self,
        *,
        timestamp: pd.Timestamp,
        last_timestamp: pd.Timestamp | None,
        mark_price: float,
        cash: np.ndarray,
        position: np.ndarray,
        config: BacktestConfig,
    ) -> np.ndarray:
        if last_timestamp is not None and config.borrow_rate > 0:
            dt_years = (timestamp - last_timestamp).total_seconds() / (365.25 * 24 * 3600)
            if dt_years > 0:
                short_mask = position < -1e-12
                if np.any(short_mask):
                    short_notional = np.abs(position[short_mask]) * mark_price
                    carry = short_notional * float(config.borrow_rate) * dt_years
                    cash[short_mask] -= carry
        return cash + position * mark_price

    def _execute_pending_orders(
        self,
        *,
        ts: pd.Timestamp,
        open_px: float,
        high_px: float,
        low_px: float,
        close_px: float,
        cash: np.ndarray,
        position: np.ndarray,
        avg_price: np.ndarray,
        realized_pnl: np.ndarray,
        pending_qty: np.ndarray,
        pending_order_type: np.ndarray,
        pending_order_price: np.ndarray,
        pending_order_active: np.ndarray,
        trades_by_param: List[List[Trade]],
        config: BacktestConfig,
        buy_fee: float,
        sell_fee: float,
        buy_slip: float,
        sell_slip: float,
    ) -> np.ndarray:
        filled_mask = np.zeros_like(pending_order_active, dtype=bool)
        if pending_qty.size == 0 or pending_order_active.size == 0:
            return filled_mask
        for slot_idx in range(pending_order_active.shape[1]):
            active_params = np.flatnonzero(pending_order_active[:, slot_idx] & (np.abs(pending_qty[:, slot_idx]) > 1e-12))
            for param_idx in active_params:
                fill_price = self._fill_price_for_order(
                    order_type=int(pending_order_type[param_idx, slot_idx]),
                    qty=float(pending_qty[param_idx, slot_idx]),
                    order_price=float(pending_order_price[param_idx, slot_idx]),
                    open_px=open_px,
                    high_px=high_px,
                    low_px=low_px,
                    close_px=close_px,
                )
                if fill_price is None:
                    continue
                trade = self._execute_order(
                    qty=float(pending_qty[param_idx, slot_idx]),
                    price=fill_price,
                    timestamp=ts,
                    cash=cash,
                    position=position,
                    avg_price=avg_price,
                    realized_pnl=realized_pnl,
                    param_idx=int(param_idx),
                    config=config,
                    buy_fee=buy_fee,
                    sell_fee=sell_fee,
                    buy_slip=buy_slip,
                    sell_slip=sell_slip,
                )
                if trade is not None:
                    trades_by_param[int(param_idx)].append(trade)
                    filled_mask[param_idx, slot_idx] = True
                    pending_order_active[param_idx, slot_idx] = False
                    pending_qty[param_idx, slot_idx] = 0.0
                    pending_order_type[param_idx, slot_idx] = 0
                    pending_order_price[param_idx, slot_idx] = np.nan
        return filled_mask

    def _fill_price_for_order(
        self,
        *,
        order_type: int,
        qty: float,
        order_price: float,
        open_px: float,
        high_px: float,
        low_px: float,
        close_px: float,
    ) -> float | None:
        if order_type == 0:
            return open_px
        if order_type == 1:
            if not np.isfinite(order_price):
                return None
            if qty > 0:
                return float(order_price) if low_px <= order_price else None
            return float(order_price) if high_px >= order_price else None
        if order_type == 2:
            if not np.isfinite(order_price):
                return None
            if qty > 0:
                return max(float(order_price), open_px) if high_px >= order_price else None
            return min(float(order_price), open_px) if low_px <= order_price else None
        raise ValueError(f"Unsupported vectorized order_type: {order_type}")

    def _execute_order(
        self,
        *,
        qty: float,
        price: float,
        timestamp: pd.Timestamp,
        cash: np.ndarray,
        position: np.ndarray,
        avg_price: np.ndarray,
        realized_pnl: np.ndarray,
        param_idx: int,
        config: BacktestConfig,
        buy_fee: float,
        sell_fee: float,
        buy_slip: float,
        sell_slip: float,
    ) -> Trade | None:
        qty = qty * float(config.fill_ratio)
        if abs(qty) < 1e-12:
            return None

        side = "buy" if qty > 0 else "sell"
        prev_qty = float(position[param_idx])
        if config.prevent_scale_in:
            if side == "buy" and prev_qty > 0:
                return None
            if side == "sell" and prev_qty < 0:
                return None

        if side == "sell" and not config.allow_short:
            qty = -min(abs(qty), max(prev_qty, 0.0))
            if abs(qty) < 1e-12:
                return None

        slip = buy_slip if qty > 0 else sell_slip
        adj_price = float(price) * (1.0 + slip if qty > 0 else 1.0 - slip)
        fee_rate = buy_fee if qty > 0 else sell_fee
        buying_to_cover = qty > 0 and prev_qty < 0
        if qty > 0 and not buying_to_cover and (adj_price * (1.0 + fee_rate)) > 0:
            max_affordable_qty = max(float(cash[param_idx]) / (adj_price * (1.0 + fee_rate)), 0.0)
            if qty > max_affordable_qty:
                qty = max_affordable_qty
                if qty < 1e-12:
                    return None

        notional = qty * adj_price
        fee = abs(notional) * fee_rate
        new_qty = prev_qty + qty
        realized = 0.0

        if prev_qty != 0 and (
            (prev_qty > 0 > new_qty)
            or (prev_qty < 0 < new_qty)
            or (prev_qty > 0 and new_qty < prev_qty and new_qty >= 0)
            or (prev_qty < 0 and new_qty > prev_qty and new_qty <= 0)
        ):
            if prev_qty > 0:
                closed = min(abs(qty), prev_qty)
                realized = (adj_price - float(avg_price[param_idx])) * closed
            else:
                closed = min(abs(qty), abs(prev_qty))
                realized = (float(avg_price[param_idx]) - adj_price) * closed
            realized_pnl[param_idx] += realized

        position[param_idx] = new_qty
        if abs(new_qty) < 1e-12:
            avg_price[param_idx] = 0.0
            position[param_idx] = 0.0
        else:
            if prev_qty == 0 or (prev_qty > 0 > new_qty) or (prev_qty < 0 < new_qty):
                avg_price[param_idx] = adj_price
            else:
                prev_abs = abs(prev_qty)
                qty_abs = abs(qty)
                new_abs = abs(new_qty)
                avg_price[param_idx] = ((float(avg_price[param_idx]) * prev_abs) + (adj_price * qty_abs)) / new_abs

        cash[param_idx] -= notional
        cash[param_idx] -= fee
        equity_after = cash[param_idx] + position[param_idx] * adj_price
        return Trade(
            timestamp=timestamp,
            side=side,
            qty=float(qty),
            price=float(adj_price),
            fee=float(fee),
            realized_pnl=float(realized_pnl[param_idx]),
            equity_after=float(equity_after),
        )

    def _normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        normalized = data.copy()
        if normalized.index.tz is None:
            normalized.index = normalized.index.tz_localize("UTC")
        else:
            normalized.index = normalized.index.tz_convert("UTC")
        return normalized

    def _slice_data(self, data: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
        sliced = data.copy()
        if config.time_horizon_start is not None:
            sliced = sliced[sliced.index >= config.time_horizon_start]
        if config.time_horizon_end is not None:
            sliced = sliced[sliced.index <= config.time_horizon_end]
        if sliced.empty:
            raise ValueError("Selected time horizon produced empty signal data.")
        return sliced

    def _side_rate(self, schedule: Dict[str, float] | None, default: float, side: str) -> float:
        if schedule and side in schedule:
            return float(schedule[side])
        return float(default)

    def _iter_param_batches(self, *, total: int, n_bars: int, config: BacktestConfig) -> Iterable[tuple[int, int]]:
        batch_size = self._resolve_param_batch_size(total=total, n_bars=n_bars, config=config)
        for start in range(0, total, batch_size):
            yield start, min(start + batch_size, total)

    def _resolve_param_batch_size(self, *, total: int, n_bars: int, config: BacktestConfig) -> int:
        configured = getattr(config, "vectorized_param_batch_size", None)
        if configured is not None:
            return max(1, min(int(configured), total))
        # Keep the active signal/order matrices to a manageable size for larger studies.
        target_cells = 250_000
        inferred = max(1, target_cells // max(int(n_bars), 1))
        return max(1, min(total, inferred))

    def _estimate_session_seconds_per_day(self, data: pd.DataFrame) -> float | None:
        index = data.index
        if not isinstance(index, pd.DatetimeIndex) or index.size < 2:
            return None
        idx = index
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        try:
            ny = idx.tz_convert("America/New_York")
        except Exception:
            return None
        diffs = pd.Series(ny).sort_values().diff().dropna()
        diffs = diffs[diffs > pd.Timedelta(0)]
        if diffs.empty:
            return None
        bar_seconds = diffs.median().total_seconds()
        if bar_seconds <= 0:
            return None
        day_counts = pd.Series(1, index=ny.normalize()).groupby(level=0).sum()
        day_counts = day_counts[day_counts > 0]
        if day_counts.empty:
            return None
        return float(day_counts.median() * bar_seconds)
