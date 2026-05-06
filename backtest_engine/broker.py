from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

from .sizing import POSITION_SIZING_NONE, effective_max_gross_leverage, normalize_position_sizing_model


@dataclass
class Trade:
    timestamp: pd.Timestamp
    side: str  # 'buy' or 'sell'
    qty: float
    price: float
    fee: float
    realized_pnl: float
    equity_after: float


@dataclass
class OrderRequest:
    qty: float  # positive buys, negative sells
    earliest_ts: pd.Timestamp | None = None
    order_type: str = "market"  # market, limit, stop
    limit_price: float | None = None
    stop_price: float | None = None
    tag: str | None = None


class Broker:
    """Simple single-asset broker that tracks cash, position and trades."""

    def __init__(
        self,
        starting_cash: float = 100_000.0,
        fee_rate: float = 0.0,
        fee_schedule: dict | None = None,
        slippage: float = 0.0,
        slippage_schedule: dict | None = None,
        borrow_rate: float = 0.0,
        fill_ratio: float = 1.0,
        allow_short: bool = False,
        prevent_scale_in: bool = False,
        margin_enabled: bool = False,
        max_gross_leverage: float = 1.0,
        position_sizing_model: str = POSITION_SIZING_NONE,
        min_position_shares: float = 1.0,
        sizing_multiplier_by_ts: pd.Series | None = None,
    ) -> None:
        self.starting_cash = float(starting_cash)
        self.cash = float(starting_cash)
        self.fee_rate = float(fee_rate)
        self.fee_schedule = fee_schedule or {}
        self.slippage = float(slippage)
        self.slippage_schedule = slippage_schedule or {}
        self.borrow_rate = float(borrow_rate)  # annualized cost on short notional
        self.fill_ratio = max(0.0, min(1.0, float(fill_ratio)))  # partial fills per bar
        self.allow_short = allow_short
        self.prevent_scale_in = prevent_scale_in
        self.margin_enabled = bool(margin_enabled)
        self.max_gross_leverage = max(1.0, float(max_gross_leverage or 1.0))
        self.position_sizing_model = normalize_position_sizing_model(position_sizing_model)
        self.min_position_shares = max(0.0, float(min_position_shares or 0.0))
        self.sizing_multiplier_by_ts = sizing_multiplier_by_ts
        self.current_timestamp: pd.Timestamp | None = None

        self.position_qty: float = 0.0
        self.avg_price: float = 0.0
        self.realized_pnl: float = 0.0
        self.borrow_cost_paid: float = 0.0
        self.trades: List[Trade] = []
        self.pending_orders: List[OrderRequest] = []
        self.equity_curve: List[tuple[pd.Timestamp, float]] = []
        self._last_timestamp: pd.Timestamp | None = None

    # --- Order entry helpers -------------------------------------------------
    def buy(self, qty: float, earliest_ts: pd.Timestamp | None = None, tag: str | None = None) -> None:
        if qty <= 0:
            raise ValueError("Buy quantity must be positive.")
        self.pending_orders.append(OrderRequest(qty=qty, earliest_ts=earliest_ts, order_type="market", tag=tag))

    def sell(self, qty: float, earliest_ts: pd.Timestamp | None = None, tag: str | None = None) -> None:
        if qty <= 0:
            raise ValueError("Sell quantity must be positive.")
        self.pending_orders.append(OrderRequest(qty=-qty, earliest_ts=earliest_ts, order_type="market", tag=tag))

    def buy_limit(
        self, qty: float, limit_price: float, earliest_ts: pd.Timestamp | None = None, tag: str | None = None
    ) -> None:
        if qty <= 0:
            raise ValueError("Buy quantity must be positive.")
        self.pending_orders.append(
            OrderRequest(qty=qty, earliest_ts=earliest_ts, order_type="limit", limit_price=limit_price, tag=tag)
        )

    def sell_limit(
        self, qty: float, limit_price: float, earliest_ts: pd.Timestamp | None = None, tag: str | None = None
    ) -> None:
        if qty <= 0:
            raise ValueError("Sell quantity must be positive.")
        self.pending_orders.append(
            OrderRequest(qty=-qty, earliest_ts=earliest_ts, order_type="limit", limit_price=limit_price, tag=tag)
        )

    def buy_stop(
        self, qty: float, stop_price: float, earliest_ts: pd.Timestamp | None = None, tag: str | None = None
    ) -> None:
        if qty <= 0:
            raise ValueError("Buy quantity must be positive.")
        self.pending_orders.append(
            OrderRequest(qty=qty, earliest_ts=earliest_ts, order_type="stop", stop_price=stop_price, tag=tag)
        )

    def sell_stop(
        self, qty: float, stop_price: float, earliest_ts: pd.Timestamp | None = None, tag: str | None = None
    ) -> None:
        if qty <= 0:
            raise ValueError("Sell quantity must be positive.")
        self.pending_orders.append(
            OrderRequest(qty=-qty, earliest_ts=earliest_ts, order_type="stop", stop_price=stop_price, tag=tag)
        )

    def cancel_orders(self, tag: str | None = None) -> None:
        if tag is None:
            self.pending_orders.clear()
            return
        self.pending_orders = [o for o in self.pending_orders if o.tag != tag]

    def target_percent(
        self,
        target: float,
        mark_price: float,
        earliest_ts: pd.Timestamp | None = None,
        order_type: str = "market",
        limit_price: float | None = None,
        stop_price: float | None = None,
        tag: str | None = None,
    ) -> None:
        """
        Move position to a target fraction of equity (e.g., 1.0 = 100% long).
        """
        target = self._apply_position_sizing_target(float(target), earliest_ts=earliest_ts)
        equity = self._mark_to_market(mark_price)
        desired_notional = target * equity
        desired_qty = desired_notional / mark_price
        if self.position_sizing_model != POSITION_SIZING_NONE and target != 0.0:
            min_shares = float(self.min_position_shares)
            if min_shares > 0.0 and 0.0 < abs(desired_qty) < min_shares:
                desired_qty = min_shares if desired_qty > 0.0 else -min_shares
        delta = desired_qty - self.position_qty
        if abs(delta) > 1e-9:
            self.pending_orders.append(
                OrderRequest(
                    qty=delta,
                    earliest_ts=earliest_ts,
                    order_type=order_type,
                    limit_price=limit_price,
                    stop_price=stop_price,
                    tag=tag,
                )
            )

    # --- Execution -----------------------------------------------------------
    def flush_orders(self, bar: pd.Series, timestamp: pd.Timestamp) -> List[Trade]:
        self.current_timestamp = timestamp
        fills: List[Trade] = []
        if not self.pending_orders:
            return fills

        remaining: List[OrderRequest] = []
        for order in self.pending_orders:
            if order.earliest_ts is not None and timestamp < order.earliest_ts:
                remaining.append(order)
                continue
            fill_price = self._fill_price_for_order(order, bar)
            if fill_price is None:
                remaining.append(order)
                continue
            trade = self._execute_order(order, fill_price, timestamp)
            if trade:
                fills.append(trade)

        self.pending_orders = remaining
        return fills

    def _execute_order(self, order: OrderRequest, price: float, timestamp: pd.Timestamp) -> Trade | None:
        qty = order.qty * self.fill_ratio
        if abs(qty) < 1e-12:
            return None

        side = "buy" if qty > 0 else "sell"

        if self.prevent_scale_in:
            if side == "buy" and self.position_qty > 0:
                return None
            if side == "sell" and self.position_qty < 0:
                return None

        if side == "sell" and not self.allow_short:
            qty = -min(abs(qty), self.position_qty)
            if abs(qty) < 1e-12:
                return None

        slip = self._slippage_for_side(side)
        adj_price = price * (1 + slip if qty > 0 else 1 - slip)
        fee_rate = self._fee_for_side(side)
        max_affordable_qty = float("inf")
        buying_to_cover = qty > 0 and self.position_qty < 0
        if qty > 0 and not buying_to_cover and (adj_price * (1 + fee_rate)) > 0:
            max_affordable_qty = self._max_affordable_buy_qty(adj_price, fee_rate)
            if qty > max_affordable_qty:
                qty = max_affordable_qty
                if qty < 1e-12:
                    return None

        if side == "sell" and self.allow_short and qty < 0:
            qty = self._cap_short_qty(qty, adj_price)
            if abs(qty) < 1e-12:
                return None

        notional = qty * adj_price
        fee = abs(notional) * fee_rate

        prev_qty = self.position_qty
        new_qty = prev_qty + qty
        realized = 0.0

        # Closing component: if trade moves position toward zero, compute realized on closed portion.
        if prev_qty != 0 and ((prev_qty > 0 > new_qty) or (prev_qty < 0 < new_qty) or (prev_qty > 0 and new_qty < prev_qty and new_qty >= 0) or (prev_qty < 0 and new_qty > prev_qty and new_qty <= 0)):
            if prev_qty > 0:
                closed = min(abs(qty), prev_qty)
                realized = (adj_price - self.avg_price) * closed
            else:
                closed = min(abs(qty), abs(prev_qty))
                realized = (self.avg_price - adj_price) * closed
            self.realized_pnl += realized

        # Update average price for remaining open size
        self.position_qty = new_qty
        if self.position_qty == 0:
            self.avg_price = 0.0
        else:
            if prev_qty == 0 or (prev_qty > 0 > self.position_qty) or (prev_qty < 0 < self.position_qty):
                self.avg_price = adj_price
            else:
                prev_abs = abs(prev_qty)
                qty_abs = abs(qty)
                new_abs = abs(self.position_qty)
                self.avg_price = (self.avg_price * prev_abs + adj_price * qty_abs) / new_abs

        # Cash movement (after validation)
        self.cash -= notional
        self.cash -= fee

        equity_after = self._mark_to_market(adj_price)
        trade = Trade(
            timestamp=timestamp,
            side=side,
            qty=qty,
            price=adj_price,
            fee=fee,
            realized_pnl=self.realized_pnl,
            equity_after=equity_after,
        )
        self.trades.append(trade)
        return trade

    def _fill_price_for_order(self, order: OrderRequest, bar: pd.Series) -> float | None:
        try:
            o = float(bar["open"])
            h = float(bar["high"])
            l = float(bar["low"])
        except Exception:
            o = float(bar.get("close", bar.iloc[0]))
            h = o
            l = o

        order_type = (order.order_type or "market").lower()
        if order_type == "market":
            return o
        if order_type == "limit":
            limit = order.limit_price
            if limit is None:
                return None
            if order.qty > 0:
                return float(limit) if l <= limit else None
            return float(limit) if h >= limit else None
        if order_type == "stop":
            stop = order.stop_price
            if stop is None:
                return None
            if order.qty > 0:
                return max(float(stop), o) if h >= stop else None
            return min(float(stop), o) if l <= stop else None
        raise ValueError(f"Unsupported order_type: {order.order_type}")

    # --- Accounting ----------------------------------------------------------
    def record_equity(self, timestamp: pd.Timestamp, mark_price: float) -> None:
        # Borrow/financing accrual on shorts between timestamps.
        if self._last_timestamp is not None and self.position_qty < -1e-12 and self.borrow_rate > 0:
            dt_years = (timestamp - self._last_timestamp).total_seconds() / (365.25 * 24 * 3600)
            short_notional = abs(self.position_qty) * mark_price
            carry = short_notional * self.borrow_rate * dt_years
            self.cash -= carry
            self.borrow_cost_paid += carry

        self.equity_curve.append((timestamp, self._mark_to_market(mark_price)))
        self._last_timestamp = timestamp

    def _mark_to_market(self, price: float) -> float:
        return self.cash + self.position_qty * price

    def current_equity(self, price: float) -> float:
        return self._mark_to_market(price)

    def _apply_position_sizing_target(self, target: float, *, earliest_ts: pd.Timestamp | None = None) -> float:
        if self.position_sizing_model == POSITION_SIZING_NONE:
            return float(target)
        series = self.sizing_multiplier_by_ts
        if series is None or series.empty:
            return float(target)
        timestamp = earliest_ts or self.current_timestamp
        if timestamp is None:
            return float(target)
        ts = pd.Timestamp(timestamp)
        ts = ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
        aligned = series
        if aligned.index.tz is None:
            aligned = aligned.copy()
            aligned.index = aligned.index.tz_localize("UTC")
        else:
            aligned = aligned.tz_convert("UTC")
        candidates = aligned.loc[aligned.index <= ts]
        if candidates.empty:
            return float(target)
        try:
            multiplier = float(candidates.iloc[-1])
        except Exception:
            return float(target)
        if not pd.notna(multiplier):
            return float(target)
        return float(target) * max(0.0, multiplier)

    def _max_affordable_buy_qty(self, adj_price: float, fee_rate: float) -> float:
        denominator = adj_price * (1.0 + fee_rate)
        if denominator <= 0:
            return 0.0
        if not self.margin_enabled:
            return max(self.cash / denominator, 0.0)
        equity = max(self._mark_to_market(adj_price), 0.0)
        max_gross = equity * effective_max_gross_leverage(self)
        current_gross = abs(self.position_qty) * adj_price
        available_notional = max(max_gross - current_gross, 0.0)
        return available_notional / denominator

    def _cap_short_qty(self, qty: float, adj_price: float) -> float:
        if qty >= 0 or adj_price <= 0:
            return qty
        prev_short_qty = abs(min(self.position_qty, 0.0))
        requested_short_qty = abs(min(self.position_qty + qty, 0.0))
        if requested_short_qty <= prev_short_qty:
            return qty
        equity = max(self._mark_to_market(adj_price), 0.0)
        max_gross = equity * effective_max_gross_leverage(self)
        current_long_notional = max(self.position_qty, 0.0) * adj_price
        allowed_short_qty = max(max_gross - current_long_notional, 0.0) / adj_price
        if requested_short_qty <= allowed_short_qty:
            return qty
        target_new_qty = -allowed_short_qty
        return target_new_qty - self.position_qty

    def _fee_for_side(self, side: str) -> float:
        return float(self.fee_schedule.get(side, self.fee_rate))

    def _slippage_for_side(self, side: str) -> float:
        return float(self.slippage_schedule.get(side, self.slippage))
