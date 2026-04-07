from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Type

import pandas as pd

from .broker import Broker, Trade
from .catalog import ResultCatalog
from .metrics import PerformanceMetrics, compute_metrics, sharpe_diagnostics
from .run_ids import compute_logical_run_id
from .strategy import Strategy


@dataclass
class BacktestConfig:
    timeframe: str
    batch_id: str | None = None
    starting_cash: float = 100_000.0
    fee_rate: float = 0.0
    fee_schedule: Optional[Dict[str, float]] = None
    slippage: float = 0.0
    slippage_schedule: Optional[Dict[str, float]] = None
    borrow_rate: float = 0.0
    fill_ratio: float = 1.0
    fill_on_close: bool = False
    recalc_on_fill: bool = True
    max_recalc_passes: int = 4
    allow_short: bool = False
    time_horizon_start: Optional[pd.Timestamp] = None
    time_horizon_end: Optional[pd.Timestamp] = None
    use_cache: bool = True
    intrabar_sim: bool = False
    base_execution: bool = True
    base_timeframe: str = "1 minutes"
    prevent_scale_in: bool = True
    one_order_per_signal: bool = True
    sharpe_annualization: str = "equities"
    sharpe_session_seconds_per_day: float | None = None
    sharpe_debug: bool = False
    sharpe_basis: str = "daily"
    risk_free_rate: float = 0.0
    vectorized_param_batch_size: int | None = None


@dataclass
class BacktestResult:
    run_id: str
    equity_curve: pd.Series
    trades: list[Trade]
    metrics: PerformanceMetrics
    cached: bool

    def to_dict(self) -> Dict:
        return {
            "run_id": self.run_id,
            "metrics": self.metrics.as_dict(),
            "equity_curve_len": len(self.equity_curve),
            "trades": [t.__dict__ for t in self.trades],
            "cached": self.cached,
        }


class BacktestEngine:
    def __init__(
        self,
        data: pd.DataFrame,
        dataset_id: str,
        strategy_cls: Type[Strategy],
        catalog: Optional[ResultCatalog],
        config: BacktestConfig,
        base_data: Optional[pd.DataFrame] = None,
        execution_metadata: Optional[Dict[str, str | None]] = None,
    ) -> None:
        # Base data (1m/tick) is source of truth; fall back to provided data if base not given.
        self.base_data = base_data.copy() if base_data is not None else data.copy()
        if self.base_data.index.tz is None:
            self.base_data.index = self.base_data.index.tz_localize("UTC")
        else:
            self.base_data.index = self.base_data.index.tz_convert("UTC")
        self.dataset_id = dataset_id
        self.strategy_cls = strategy_cls
        self.catalog = catalog
        self.config = config
        self.execution_metadata = {
            "logical_run_id": None,
            "requested_execution_mode": "reference",
            "resolved_execution_mode": "reference",
            "engine_impl": "reference",
            "engine_version": "1",
            "fallback_reason": None,
            **(execution_metadata or {}),
        }
        self.signal_data = self._build_signal_data(self.config.timeframe)
        self.data = data.copy()  # kept for compatibility

    def run(self, strategy_params: Dict) -> BacktestResult:
        data = self._slice_signal_data()
        run_id = self._compute_run_id(strategy_params, data)
        execution_metadata = self._execution_metadata_for_run(run_id)
        run_started_at = pd.Timestamp.now("UTC").isoformat()

        if self.config.use_cache and self.catalog:
            cached = self.catalog.fetch(run_id)
            if cached:
                dummy_equity = pd.Series([], dtype=float)
                if self.config.batch_id and self.catalog:
                    # Attach cached run to current batch for UI visibility.
                    self.catalog.save(
                        run_id=run_id,
                        batch_id=self.config.batch_id,
                        strategy=self.strategy_cls.__name__,
                        params=strategy_params,
                        timeframe=self.config.timeframe,
                        start=cached.start,
                        end=cached.end,
                        dataset_id=self.dataset_id,
                        starting_cash=self.config.starting_cash,
                        metrics=cached.metrics,
                        run_started_at=cached.run_started_at,
                        run_finished_at=cached.run_finished_at or pd.Timestamp.now("UTC").isoformat(),
                        status=cached.status or "finished",
                        logical_run_id=execution_metadata["logical_run_id"],
                        requested_execution_mode=execution_metadata["requested_execution_mode"],
                        resolved_execution_mode=execution_metadata["resolved_execution_mode"],
                        engine_impl=execution_metadata["engine_impl"],
                        engine_version=execution_metadata["engine_version"],
                        fallback_reason=execution_metadata["fallback_reason"],
                    )
                return BacktestResult(
                    run_id=run_id,
                    equity_curve=dummy_equity,
                    trades=[],
                    metrics=cached.metrics,
                    cached=True,
                )

        # Mark run as in-progress in catalog for UI visibility.
        if self.catalog:
            self.catalog.save(
                run_id=run_id,
                batch_id=self.config.batch_id,
                strategy=self.strategy_cls.__name__,
                params=strategy_params,
                timeframe=self.config.timeframe,
                start=str(data.index[0]),
                end=str(data.index[-1]),
                dataset_id=self.dataset_id,
                starting_cash=self.config.starting_cash,
                metrics=None,
                run_started_at=run_started_at,
                run_finished_at=None,
                status="running",
                logical_run_id=execution_metadata["logical_run_id"],
                requested_execution_mode=execution_metadata["requested_execution_mode"],
                resolved_execution_mode=execution_metadata["resolved_execution_mode"],
                engine_impl=execution_metadata["engine_impl"],
                engine_version=execution_metadata["engine_version"],
                fallback_reason=execution_metadata["fallback_reason"],
            )

        strategy = self.strategy_cls(**strategy_params)
        strategy.initialize(data.copy())
        broker = Broker(
            starting_cash=self.config.starting_cash,
            fee_rate=self.config.fee_rate,
            fee_schedule=self.config.fee_schedule,
            slippage=self.config.slippage,
            slippage_schedule=self.config.slippage_schedule,
            borrow_rate=self.config.borrow_rate,
            fill_ratio=self.config.fill_ratio,
            allow_short=self.config.allow_short,
            prevent_scale_in=self.config.prevent_scale_in,
        )

        if self.config.base_execution and self.base_data is not None:
            self._run_with_base_execution(strategy, broker, data)
        else:
            # simple loop with one-bar delay fills
            timestamps = list(data.index)
            next_lookup = {timestamps[i]: timestamps[i + 1] for i in range(len(timestamps) - 1)}
            for timestamp, bar in data.iterrows():
                # fill any orders eligible now at bar open
                self._flush_with_recalc(strategy, broker, timestamp, bar)

                if self.config.intrabar_sim:
                    path = self._intrabar_path(bar)
                    steps = len(path)
                    for i, price in enumerate(path):
                        ts_step = self._step_timestamp(timestamp, i, steps)
                        step_bar = pd.Series(
                            {"open": price, "high": price, "low": price, "close": price, "volume": bar.get("volume", 0)},
                            name=timestamp,
                        )
                        strategy.on_bar(timestamp, step_bar, broker)
                        if self.config.one_order_per_signal:
                            self._prune_pending_orders(broker, broker.position_qty)
                        # schedule to same timestamp (intrabars handled inside)
                        self._flush_with_recalc(strategy, broker, ts_step, step_bar)
                        broker.record_equity(ts_step, price)
                else:
                    strategy.on_bar(timestamp, bar, broker)
                    if self.config.one_order_per_signal:
                        self._prune_pending_orders(broker, broker.position_qty)
                    if self.config.fill_on_close:
                        for o in broker.pending_orders:
                            if o.earliest_ts is None:
                                o.earliest_ts = timestamp
                        close_fill_bar = self._close_fill_bar(bar)
                        self._flush_with_recalc(strategy, broker, timestamp, close_fill_bar)
                    else:
                        # push earliest fill to next bar
                        nxt = next_lookup.get(timestamp)
                        for o in broker.pending_orders:
                            if o.earliest_ts is None:
                                o.earliest_ts = nxt if nxt is not None else timestamp
                    broker.record_equity(timestamp, bar["close"])

        equity_curve = pd.Series({ts: eq for ts, eq in broker.equity_curve})
        session_seconds = self.config.sharpe_session_seconds_per_day
        if (
            session_seconds is None
            and self.config.sharpe_annualization == "equities"
            and self.config.sharpe_basis != "daily"
        ):
            session_seconds = self._estimate_session_seconds_per_day()
        metrics = compute_metrics(
            equity_curve,
            risk_free_rate=self.config.risk_free_rate,
            timeframe=self.config.timeframe,
            annualization=self.config.sharpe_annualization,
            session_seconds_per_day=session_seconds,
            sharpe_basis=self.config.sharpe_basis,
        )
        if self.config.sharpe_debug:
            diag = sharpe_diagnostics(
                equity_curve,
                risk_free_rate=self.config.risk_free_rate,
                timeframe=self.config.timeframe,
                annualization=self.config.sharpe_annualization,
                session_seconds_per_day=session_seconds,
                sharpe_basis=self.config.sharpe_basis,
            )
            print(
                "Sharpe diagnostics:",
                f"mean={diag['mean']:.6g}",
                f"std={diag['std']:.6g}",
                f"periods={diag['periods']}",
                f"periods_per_year={diag['periods_per_year']:.3f}",
                f"seconds_per_period={diag['seconds_per_period']}",
                f"session_seconds_per_day={diag['session_seconds_per_day']}",
                f"annualization={diag['annualization']}",
                f"basis={diag['basis']}",
                f"rf_per_period={diag['rf_per_period']:.6g}",
            )

        run_finished_at = pd.Timestamp.now("UTC").isoformat()
        if self.catalog:
            self.catalog.save(
                run_id=run_id,
                batch_id=self.config.batch_id,
                strategy=self.strategy_cls.__name__,
                params=strategy_params,
                timeframe=self.config.timeframe,
                start=str(data.index[0]),
                end=str(data.index[-1]),
                dataset_id=self.dataset_id,
                starting_cash=self.config.starting_cash,
                metrics=metrics,
                run_started_at=run_started_at,
                run_finished_at=run_finished_at,
                status="finished",
                logical_run_id=execution_metadata["logical_run_id"],
                requested_execution_mode=execution_metadata["requested_execution_mode"],
                resolved_execution_mode=execution_metadata["resolved_execution_mode"],
                engine_impl=execution_metadata["engine_impl"],
                engine_version=execution_metadata["engine_version"],
                fallback_reason=execution_metadata["fallback_reason"],
            )
            self.catalog.save_trades(run_id, broker.trades)

        return BacktestResult(
            run_id=run_id,
            equity_curve=equity_curve,
            trades=broker.trades,
            metrics=metrics,
            cached=False,
        )

    def _execution_metadata_for_run(self, run_id: str) -> Dict[str, str | None]:
        metadata = dict(self.execution_metadata)
        metadata["logical_run_id"] = metadata.get("logical_run_id") or run_id
        return metadata

    def _slice_data(self) -> pd.DataFrame:
        data = self.data.copy()
        if self.config.time_horizon_start:
            data = data[data.index >= self.config.time_horizon_start]
        if self.config.time_horizon_end:
            data = data[data.index <= self.config.time_horizon_end]
        if data.empty:
            raise ValueError("Selected time horizon produced empty data.")
        return data

    def _slice_signal_data(self) -> pd.DataFrame:
        data = self.signal_data.copy()
        if self.config.time_horizon_start:
            data = data[data.index >= self.config.time_horizon_start]
        if self.config.time_horizon_end:
            data = data[data.index <= self.config.time_horizon_end]
        if data.empty:
            raise ValueError("Selected time horizon produced empty signal data.")
        return data

    def _slice_base_data(self) -> pd.DataFrame:
        base = self.base_data.copy() if self.base_data is not None else pd.DataFrame()
        if self.config.time_horizon_start is not None:
            base = base[base.index >= self.config.time_horizon_start]
        if self.config.time_horizon_end is not None:
            base = base[base.index <= self.config.time_horizon_end]
        return base

    def _build_signal_data(self, tf: str) -> pd.DataFrame:
        norm_tf = self._normalize_freq(tf)
        base_tf = self._normalize_freq(self.config.base_timeframe)
        if norm_tf == base_tf:
            return self.base_data.copy()
        agg = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
        sig = self.base_data.resample(norm_tf, label="right", closed="right").agg(agg).dropna()
        return sig

    @staticmethod
    def _normalize_freq(tf: str) -> str:
        """Convert user-friendly timeframe strings like '5 minutes' to pandas offsets like '5T'."""
        s = tf.strip().lower()
        s = s.replace("minute", "min").replace("mins", "min")
        s = s.replace("hour", "h")
        # extract leading integer
        tokens = s.replace(" ", "")
        num = ""
        unit = ""
        for ch in tokens:
            if ch.isdigit():
                num += ch
            else:
                unit += ch
        if not num:
            num = "1"
        if unit in ("min", "m"):
            return f"{num}min"
        if unit in ("h", "hr"):
            return f"{num}h"
        return tf  # fallback to original

    @staticmethod
    def _prune_pending_orders(broker: Broker, pos: float) -> None:
        """Keep only one actionable order per signal based on current position."""
        keep = None
        for ord in broker.pending_orders:
            qty = ord.qty
            if pos == 0:
                keep = ord
                break
            if pos > 0 and qty < 0:
                keep = ord
                break
            if pos < 0 and qty > 0:
                keep = ord
                break
        broker.pending_orders.clear()
        if keep:
            broker.pending_orders.append(keep)

    def _flush_with_recalc(self, strategy: Strategy, broker: Broker, timestamp: pd.Timestamp, bar: pd.Series) -> list[Trade]:
        fills = broker.flush_orders(bar=bar, timestamp=timestamp)
        all_fills = list(fills)
        if not self.config.recalc_on_fill:
            return all_fills
        passes = 0
        while fills:
            strategy.on_after_fill(timestamp, bar, broker)
            if passes >= max(int(self.config.max_recalc_passes), 0):
                break
            fills = broker.flush_orders(bar=bar, timestamp=timestamp)
            all_fills.extend(fills)
            passes += 1
        return all_fills

    def _intrabar_path(self, bar: pd.Series) -> list[float]:
        o, h, l, c = bar["open"], bar["high"], bar["low"], bar["close"]
        if c >= o:
            return [o, l, h, c]
        return [o, h, l, c]

    def _step_timestamp(self, base_ts: pd.Timestamp, idx: int, total: int) -> pd.Timestamp:
        # distribute microsecond offsets within the same bar
        return base_ts + pd.Timedelta(microseconds=idx + 1)

    def _close_fill_bar(self, bar: pd.Series) -> pd.Series:
        close_px = float(bar.get("close", bar.get("open", 0.0)))
        close_bar = bar.copy()
        close_bar["open"] = close_px
        close_bar["high"] = close_px
        close_bar["low"] = close_px
        close_bar["close"] = close_px
        return close_bar

    def _run_with_base_execution(self, strategy: Strategy, broker: Broker, signal_data: pd.DataFrame) -> None:
        """
        Evaluate signals only on signal bar closes; fill on next base bar open.
        """
        base = self._slice_base_data()
        if base is None or base.empty:
            raise ValueError("Base execution requested but base_data is empty.")

        base_times = list(base.index)
        next_lookup = {base_times[i]: base_times[i + 1] for i in range(len(base_times) - 1)}
        sig_iter = iter(signal_data.iterrows())
        next_sig = next(sig_iter, None)

        for ts, base_bar in base.iterrows():
            # 1) fill eligible orders at base open
            self._flush_with_recalc(strategy, broker, ts, base_bar)

            # 2) evaluate any signal bars that close at or before this base timestamp
            while next_sig and next_sig[0] < ts:
                sig_ts, sig_bar = next_sig
                # print(sig_ts, ts, sig_ts == ts)
                strategy.on_bar(sig_ts, sig_bar, broker)
                if self.config.one_order_per_signal:
                    self._prune_pending_orders(broker, broker.position_qty)

                # 3) set earliest_ts to next base bar for newly queued orders
                nxt = next_lookup.get(ts)
                for o in broker.pending_orders:
                    if o.earliest_ts is None:
                        o.earliest_ts = nxt if nxt is not None else ts

                next_sig = next(sig_iter, None)

            # 4) mark equity at base close
            broker.record_equity(ts, base_bar["close"])

    def _compute_run_id(self, strategy_params: Dict, data: pd.DataFrame) -> str:
        return compute_logical_run_id(
            dataset_id=self.dataset_id,
            strategy=self.strategy_cls.__name__,
            params=strategy_params,
            config=self.config,
            data=data,
        )

    def _estimate_session_seconds_per_day(self) -> float | None:
        data = self.base_data if self.base_data is not None else self.signal_data
        if data is None or data.empty:
            return None
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
        # Estimate bar duration from median diff.
        diffs = pd.Series(ny).sort_values().diff().dropna()
        diffs = diffs[diffs > pd.Timedelta(0)]
        if diffs.empty:
            return None
        bar_seconds = diffs.median().total_seconds()
        if bar_seconds <= 0:
            return None
        # Count bars per NY date; use median to reduce impact of partial days.
        day_counts = pd.Series(1, index=ny.normalize()).groupby(level=0).sum()
        day_counts = day_counts[day_counts > 0]
        if day_counts.empty:
            return None
        return float(day_counts.median() * bar_seconds)
