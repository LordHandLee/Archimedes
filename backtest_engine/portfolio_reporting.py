from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PortfolioAssetAttribution:
    dataset_id: str
    avg_weight: float
    min_weight: float
    avg_long_weight: float
    avg_short_weight: float
    peak_short_weight: float
    avg_target_weight: float
    avg_abs_tracking_error: float
    avg_abs_weight_change: float
    final_weight: float
    peak_weight: float
    active_bar_fraction: float
    trade_count: int
    realized_pnl: float
    unrealized_pnl: float
    turnover_notional: float
    turnover_ratio: float
    avg_return_contribution: float
    total_return_contribution: float
    contribution_share: float


@dataclass(frozen=True)
class PortfolioStrategyAttribution:
    strategy_block_id: str
    strategy_name: str
    budget_weight: float
    asset_count: int
    avg_weight: float
    min_weight: float
    avg_long_weight: float
    avg_short_weight: float
    peak_short_weight: float
    avg_target_weight: float
    avg_abs_tracking_error: float
    avg_abs_weight_change: float
    final_weight: float
    peak_weight: float
    active_bar_fraction: float
    trade_count: int
    realized_pnl: float
    turnover_notional: float
    turnover_ratio: float
    avg_return_contribution: float
    total_return_contribution: float
    contribution_share: float


@dataclass(frozen=True)
class PortfolioDrawdownEpisode:
    rank: int
    peak_time: pd.Timestamp | None
    trough_time: pd.Timestamp | None
    recovery_time: pd.Timestamp | None
    depth: float
    duration_bars: int
    recovery_bars: int | None


@dataclass(frozen=True)
class PortfolioReport:
    starting_equity: float
    ending_equity: float
    total_return: float
    cagr: float
    max_drawdown: float
    sharpe: float
    rolling_sharpe: float
    annualized_volatility: float
    downside_deviation: float
    sortino: float
    calmar: float
    best_period_return: float
    worst_period_return: float
    underwater_fraction: float
    max_drawdown_duration_bars: int
    avg_cash_weight: float
    min_cash_weight: float
    max_cash_weight: float
    avg_net_exposure: float
    avg_long_exposure: float
    avg_short_exposure: float
    avg_gross_exposure: float
    peak_gross_exposure: float
    peak_short_exposure: float
    avg_target_gross_exposure: float
    peak_target_gross_exposure: float
    avg_active_assets: float
    peak_active_assets: int
    avg_concentration_hhi: float
    peak_concentration_hhi: float
    peak_single_name_weight: float
    trade_count: int
    total_turnover_notional: float
    total_turnover_ratio: float
    strategy_rows: tuple[PortfolioStrategyAttribution, ...]
    asset_rows: tuple[PortfolioAssetAttribution, ...]
    drawdown_episodes: tuple[PortfolioDrawdownEpisode, ...]


@dataclass(frozen=True)
class PortfolioChartData:
    equity_curve: pd.Series
    cash_weight: pd.Series
    gross_exposure: pd.Series
    target_gross_exposure: pd.Series
    asset_weights: pd.DataFrame
    target_weights: pd.DataFrame
    top_assets: tuple[str, ...]
    trades: pd.DataFrame


def build_portfolio_trades_log_frame(portfolio_result) -> pd.DataFrame:
    trades = list(getattr(portfolio_result, "trades", []) or [])
    if not trades:
        return pd.DataFrame()
    rows = []
    position_by_asset: dict[str, float] = {}
    prev_realized_by_asset: dict[str, float] = {}
    for i, trade in enumerate(trades, start=1):
        asset_label = str(getattr(trade, "dataset_id", "") or "")
        source_dataset_id = str(getattr(trade, "source_dataset_id", asset_label) or asset_label)
        strategy_block_id = str(getattr(trade, "strategy_block_id", "") or "")
        qty = float(getattr(trade, "qty", 0.0))
        realized_cum = float(getattr(trade, "realized_pnl", 0.0))
        position_before = float(position_by_asset.get(asset_label, 0.0))
        position_after = position_before + qty
        position_by_asset[asset_label] = position_after
        prev_realized = float(prev_realized_by_asset.get(asset_label, 0.0))
        net_pnl = realized_cum - prev_realized
        prev_realized_by_asset[asset_label] = realized_cum
        if position_before * position_after < 0:
            trade_type = "flip"
        elif abs(position_after) > abs(position_before):
            trade_type = "entry"
        elif position_after == position_before:
            trade_type = "adjust"
        else:
            trade_type = "exit"
        side = str(getattr(trade, "side", "") or ("buy" if qty > 0 else "sell"))
        if trade_type == "entry":
            signal = "long" if side == "buy" else "short"
        elif trade_type == "flip":
            signal = "flip"
        else:
            signal = "open"
        rows.append(
            {
                "trade_number": i,
                "asset": source_dataset_id,
                "asset_label": asset_label if asset_label != source_dataset_id else "",
                "strategy_block": strategy_block_id,
                "type": trade_type,
                "timestamp": str(getattr(trade, "timestamp", "")),
                "signal": signal,
                "side": side,
                "price": float(getattr(trade, "price", 0.0)),
                "qty": qty,
                "position_after": position_after,
                "net_pnl": net_pnl,
                "realized_pnl_cum": realized_cum,
            }
        )
    frame = pd.DataFrame(rows)
    if "asset_label" in frame.columns and (frame["asset_label"] == "").all():
        frame = frame.drop(columns=["asset_label"])
    if "strategy_block" in frame.columns and (frame["strategy_block"] == "").all():
        frame = frame.drop(columns=["strategy_block"])
    return frame


def summarize_portfolio_result(portfolio_result, *, starting_cash: float | None = None) -> PortfolioReport:
    equity = pd.to_numeric(getattr(portfolio_result, "portfolio_equity_curve", pd.Series(dtype=float)), errors="coerce")
    if equity.empty:
        raise ValueError("Portfolio result has no equity curve to summarize.")
    equity = equity.astype(float)
    start_equity = float(starting_cash if starting_cash is not None else equity.iloc[0])
    if not np.isfinite(start_equity) or start_equity <= 0.0:
        start_equity = float(equity.iloc[0])
    end_equity = float(equity.iloc[-1])

    metrics = getattr(portfolio_result, "metrics", None)
    total_return = float(getattr(metrics, "total_return", (end_equity / start_equity) - 1.0))
    cagr = float(getattr(metrics, "cagr", total_return))
    max_drawdown = float(getattr(metrics, "max_drawdown", 0.0))
    sharpe = float(getattr(metrics, "sharpe", 0.0))
    rolling_sharpe = float(getattr(metrics, "rolling_sharpe", 0.0))

    cash_curve = pd.to_numeric(getattr(portfolio_result, "cash_curve", pd.Series(dtype=float)), errors="coerce").reindex(equity.index)
    cash_weight = (cash_curve / equity.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    asset_weights = getattr(portfolio_result, "asset_weights", pd.DataFrame(index=equity.index)).reindex(equity.index).fillna(0.0)
    target_weights = getattr(portfolio_result, "target_weights", pd.DataFrame(index=equity.index)).reindex(equity.index).fillna(0.0)
    strategy_weights = getattr(portfolio_result, "strategy_weights", pd.DataFrame(index=equity.index)).reindex(equity.index).fillna(0.0)
    strategy_target_weights = (
        getattr(portfolio_result, "strategy_target_weights", pd.DataFrame(index=equity.index)).reindex(equity.index).fillna(0.0)
    )
    strategy_display_names = dict(getattr(portfolio_result, "strategy_display_names", {}) or {})
    strategy_budget_weights = dict(getattr(portfolio_result, "strategy_budget_weights", {}) or {})
    asset_to_strategy_block = dict(getattr(portfolio_result, "asset_to_strategy_block", {}) or {})
    long_exposure = asset_weights.clip(lower=0.0).sum(axis=1) if not asset_weights.empty else pd.Series(0.0, index=equity.index)
    short_exposure = (-asset_weights.clip(upper=0.0)).sum(axis=1) if not asset_weights.empty else pd.Series(0.0, index=equity.index)
    net_exposure = asset_weights.sum(axis=1) if not asset_weights.empty else pd.Series(0.0, index=equity.index)
    gross_exposure = asset_weights.abs().sum(axis=1) if not asset_weights.empty else pd.Series(0.0, index=equity.index)
    target_gross_exposure = (
        target_weights.abs().sum(axis=1) if not target_weights.empty else pd.Series(0.0, index=equity.index)
    )
    active_assets = (target_weights.abs() > 1e-9).sum(axis=1) if not target_weights.empty else pd.Series(0, index=equity.index)
    concentration_hhi = (asset_weights.clip(lower=0.0) ** 2).sum(axis=1) if not asset_weights.empty else pd.Series(0.0, index=equity.index)
    peak_single_name_weight = (
        float(asset_weights.clip(lower=0.0).max(axis=1).max()) if not asset_weights.empty else 0.0
    )

    returns = equity.pct_change().fillna(0.0)
    annualized_volatility = _annualized_volatility(returns)
    downside_deviation = _downside_deviation(returns)
    sortino = _sortino_ratio(returns)
    calmar = (cagr / abs(max_drawdown)) if max_drawdown < 0 else float("nan")
    best_period_return = float(returns.max()) if not returns.empty else 0.0
    worst_period_return = float(returns.min()) if not returns.empty else 0.0
    drawdown = (equity / equity.cummax()) - 1.0
    underwater_fraction = float((drawdown < -1e-12).mean()) if not drawdown.empty else 0.0
    max_drawdown_duration_bars = _max_drawdown_duration_bars(drawdown)
    drawdown_episodes = tuple(_top_drawdown_episodes(drawdown, top_n=5))

    trades = list(getattr(portfolio_result, "trades", []) or [])
    turnover_notional_by_asset: dict[str, float] = {}
    trade_count_by_asset: dict[str, int] = {}
    for trade in trades:
        dataset_id = str(getattr(trade, "dataset_id", "asset"))
        turnover_notional_by_asset[dataset_id] = turnover_notional_by_asset.get(dataset_id, 0.0) + abs(
            float(getattr(trade, "qty", 0.0)) * float(getattr(trade, "price", 0.0))
        )
        trade_count_by_asset[dataset_id] = trade_count_by_asset.get(dataset_id, 0) + 1

    asset_ids: list[str] = []
    seen: set[str] = set()
    for source in (asset_weights.columns, target_weights.columns):
        for dataset_id in source:
            if dataset_id not in seen:
                asset_ids.append(str(dataset_id))
                seen.add(str(dataset_id))
    for dataset_id in turnover_notional_by_asset:
        if dataset_id not in seen:
            asset_ids.append(dataset_id)
            seen.add(dataset_id)

    asset_market_values = getattr(portfolio_result, "asset_market_values", pd.DataFrame(index=equity.index)).reindex(equity.index).fillna(0.0)
    positions = getattr(portfolio_result, "positions", pd.DataFrame(index=equity.index)).reindex(equity.index).fillna(0.0)
    lagged_weights = asset_weights.shift(1).fillna(0.0)
    asset_prices = _infer_asset_prices(
        market_values=asset_market_values,
        positions=positions,
    )
    pnl_by_asset = _reconstruct_asset_pnl(
        trades=trades,
        asset_ids=asset_ids,
        final_positions=positions.iloc[-1] if not positions.empty else pd.Series(dtype=float),
        final_prices=asset_prices.iloc[-1] if not asset_prices.empty else pd.Series(dtype=float),
    )
    asset_returns = asset_prices.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    contribution_frame = lagged_weights.reindex(asset_returns.index).fillna(0.0) * asset_returns
    total_contribution_sum = float(contribution_frame.sum().sum()) if not contribution_frame.empty else 0.0

    asset_rows: list[PortfolioAssetAttribution] = []
    strategy_turnover_notional: dict[str, float] = {}
    strategy_trade_count: dict[str, int] = {}
    strategy_realized_pnl: dict[str, float] = {}
    for dataset_id in asset_ids:
        actual = pd.to_numeric(asset_weights.get(dataset_id, pd.Series(0.0, index=equity.index)), errors="coerce").fillna(0.0)
        target = pd.to_numeric(target_weights.get(dataset_id, pd.Series(0.0, index=equity.index)), errors="coerce").fillna(0.0)
        market_value = pd.to_numeric(asset_market_values.get(dataset_id, pd.Series(0.0, index=equity.index)), errors="coerce").fillna(0.0)
        turnover_notional = float(turnover_notional_by_asset.get(dataset_id, 0.0))
        asset_pnl = pnl_by_asset.get(dataset_id, {"realized_pnl": 0.0, "unrealized_pnl": 0.0})
        total_return_contribution = (
            float(contribution_frame.get(dataset_id, pd.Series(dtype=float)).sum())
            if not contribution_frame.empty and dataset_id in contribution_frame.columns
            else 0.0
        )
        asset_rows.append(
            PortfolioAssetAttribution(
                dataset_id=dataset_id,
                avg_weight=float(actual.mean()) if not actual.empty else 0.0,
                min_weight=float(actual.min()) if not actual.empty else 0.0,
                avg_long_weight=float(actual.clip(lower=0.0).mean()) if not actual.empty else 0.0,
                avg_short_weight=float((-actual.clip(upper=0.0)).mean()) if not actual.empty else 0.0,
                peak_short_weight=float((-actual.clip(upper=0.0)).max()) if not actual.empty else 0.0,
                avg_target_weight=float(target.mean()) if not target.empty else 0.0,
                avg_abs_tracking_error=float((target - actual).abs().mean()) if not target.empty else 0.0,
                avg_abs_weight_change=float(actual.diff().abs().fillna(0.0).mean()) if not actual.empty else 0.0,
                final_weight=float(actual.iloc[-1]) if not actual.empty else 0.0,
                peak_weight=float(actual.max()) if not actual.empty else 0.0,
                active_bar_fraction=float((target.abs() > 1e-9).mean()) if not target.empty else 0.0,
                trade_count=int(trade_count_by_asset.get(dataset_id, 0)),
                realized_pnl=float(asset_pnl["realized_pnl"]),
                unrealized_pnl=float(asset_pnl["unrealized_pnl"]),
                turnover_notional=turnover_notional,
                turnover_ratio=(turnover_notional / start_equity) if start_equity > 0 else 0.0,
                avg_return_contribution=(
                    float(contribution_frame.get(dataset_id, pd.Series(dtype=float)).mean())
                    if not contribution_frame.empty and dataset_id in contribution_frame.columns
                    else 0.0
                ),
                total_return_contribution=total_return_contribution,
                contribution_share=(total_return_contribution / total_contribution_sum) if abs(total_contribution_sum) > 1e-12 else 0.0,
            )
        )
        strategy_block_id = str(asset_to_strategy_block.get(dataset_id, "default"))
        strategy_turnover_notional[strategy_block_id] = strategy_turnover_notional.get(strategy_block_id, 0.0) + turnover_notional
        strategy_trade_count[strategy_block_id] = strategy_trade_count.get(strategy_block_id, 0) + int(trade_count_by_asset.get(dataset_id, 0))
        strategy_realized_pnl[strategy_block_id] = strategy_realized_pnl.get(strategy_block_id, 0.0) + float(
            asset_pnl["realized_pnl"]
        )

    strategy_rows: list[PortfolioStrategyAttribution] = []
    strategy_ids: list[str] = list(strategy_weights.columns)
    for block_id in strategy_target_weights.columns:
        if block_id not in strategy_ids:
            strategy_ids.append(str(block_id))
    if not strategy_ids and asset_to_strategy_block:
        strategy_ids = list(dict.fromkeys(asset_to_strategy_block.values()))
    for block_id in strategy_ids:
        actual = pd.to_numeric(strategy_weights.get(block_id, pd.Series(0.0, index=equity.index)), errors="coerce").fillna(0.0)
        target = pd.to_numeric(strategy_target_weights.get(block_id, pd.Series(0.0, index=equity.index)), errors="coerce").fillna(0.0)
        block_assets = [asset_id for asset_id, mapped_block_id in asset_to_strategy_block.items() if mapped_block_id == block_id]
        block_contribution = 0.0
        if not contribution_frame.empty and block_assets:
            existing_assets = [asset_id for asset_id in block_assets if asset_id in contribution_frame.columns]
            if existing_assets:
                block_contribution = float(contribution_frame[existing_assets].sum().sum())
        strategy_rows.append(
            PortfolioStrategyAttribution(
                strategy_block_id=str(block_id),
                strategy_name=str(strategy_display_names.get(block_id, block_id)),
                budget_weight=float(strategy_budget_weights.get(block_id, 1.0)),
                asset_count=len(block_assets),
                avg_weight=float(actual.mean()) if not actual.empty else 0.0,
                min_weight=float(actual.min()) if not actual.empty else 0.0,
                avg_long_weight=float(actual.clip(lower=0.0).mean()) if not actual.empty else 0.0,
                avg_short_weight=float((-actual.clip(upper=0.0)).mean()) if not actual.empty else 0.0,
                peak_short_weight=float((-actual.clip(upper=0.0)).max()) if not actual.empty else 0.0,
                avg_target_weight=float(target.mean()) if not target.empty else 0.0,
                avg_abs_tracking_error=float((target - actual).abs().mean()) if not target.empty else 0.0,
                avg_abs_weight_change=float(actual.diff().abs().fillna(0.0).mean()) if not actual.empty else 0.0,
                final_weight=float(actual.iloc[-1]) if not actual.empty else 0.0,
                peak_weight=float(actual.max()) if not actual.empty else 0.0,
                active_bar_fraction=float((target.abs() > 1e-9).mean()) if not target.empty else 0.0,
                trade_count=int(strategy_trade_count.get(str(block_id), 0)),
                realized_pnl=float(strategy_realized_pnl.get(str(block_id), 0.0)),
                turnover_notional=float(strategy_turnover_notional.get(str(block_id), 0.0)),
                turnover_ratio=(float(strategy_turnover_notional.get(str(block_id), 0.0)) / start_equity) if start_equity > 0 else 0.0,
                avg_return_contribution=(block_contribution / len(contribution_frame.index)) if not contribution_frame.empty else 0.0,
                total_return_contribution=block_contribution,
                contribution_share=(block_contribution / total_contribution_sum) if abs(total_contribution_sum) > 1e-12 else 0.0,
            )
        )

    total_turnover_notional = float(sum(turnover_notional_by_asset.values()))
    return PortfolioReport(
        starting_equity=start_equity,
        ending_equity=end_equity,
        total_return=total_return,
        cagr=cagr,
        max_drawdown=max_drawdown,
        sharpe=sharpe,
        rolling_sharpe=rolling_sharpe,
        annualized_volatility=annualized_volatility,
        downside_deviation=downside_deviation,
        sortino=sortino,
        calmar=calmar,
        best_period_return=best_period_return,
        worst_period_return=worst_period_return,
        underwater_fraction=underwater_fraction,
        max_drawdown_duration_bars=max_drawdown_duration_bars,
        avg_cash_weight=float(cash_weight.mean()),
        min_cash_weight=float(cash_weight.min()),
        max_cash_weight=float(cash_weight.max()),
        avg_net_exposure=float(net_exposure.mean()),
        avg_long_exposure=float(long_exposure.mean()),
        avg_short_exposure=float(short_exposure.mean()),
        avg_gross_exposure=float(gross_exposure.mean()),
        peak_gross_exposure=float(gross_exposure.max()),
        peak_short_exposure=float(short_exposure.max()),
        avg_target_gross_exposure=float(target_gross_exposure.mean()),
        peak_target_gross_exposure=float(target_gross_exposure.max()),
        avg_active_assets=float(active_assets.mean()) if not active_assets.empty else 0.0,
        peak_active_assets=int(active_assets.max()) if not active_assets.empty else 0,
        avg_concentration_hhi=float(concentration_hhi.mean()) if not concentration_hhi.empty else 0.0,
        peak_concentration_hhi=float(concentration_hhi.max()) if not concentration_hhi.empty else 0.0,
        peak_single_name_weight=peak_single_name_weight,
        trade_count=len(trades),
        total_turnover_notional=total_turnover_notional,
        total_turnover_ratio=(total_turnover_notional / start_equity) if start_equity > 0 else 0.0,
        strategy_rows=tuple(strategy_rows),
        asset_rows=tuple(asset_rows),
        drawdown_episodes=drawdown_episodes,
    )


def build_portfolio_chart_data(portfolio_result, *, max_assets: int = 4) -> PortfolioChartData:
    if int(max_assets) <= 0:
        raise ValueError("max_assets must be > 0.")

    equity = pd.to_numeric(getattr(portfolio_result, "portfolio_equity_curve", pd.Series(dtype=float)), errors="coerce")
    if equity.empty:
        raise ValueError("Portfolio result has no equity curve to chart.")
    equity = equity.astype(float)

    cash_curve = pd.to_numeric(getattr(portfolio_result, "cash_curve", pd.Series(dtype=float)), errors="coerce").reindex(equity.index)
    cash_weight = (cash_curve / equity.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    asset_weights = getattr(portfolio_result, "asset_weights", pd.DataFrame(index=equity.index)).reindex(equity.index)
    target_weights = getattr(portfolio_result, "target_weights", pd.DataFrame(index=equity.index)).reindex(equity.index)
    asset_weights = asset_weights.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    target_weights = target_weights.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    gross_exposure = asset_weights.abs().sum(axis=1) if not asset_weights.empty else pd.Series(0.0, index=equity.index)
    target_gross_exposure = (
        target_weights.abs().sum(axis=1) if not target_weights.empty else pd.Series(0.0, index=equity.index)
    )

    asset_ids: list[str] = []
    seen: set[str] = set()
    for source in (asset_weights.columns, target_weights.columns):
        for raw_dataset_id in source:
            dataset_id = str(raw_dataset_id)
            if dataset_id not in seen:
                asset_ids.append(dataset_id)
                seen.add(dataset_id)

    top_assets = tuple(
        dataset_id
        for dataset_id, _ in sorted(
            (
                (
                    dataset_id,
                    max(
                        float(asset_weights.get(dataset_id, pd.Series(0.0, index=equity.index)).abs().max()),
                        float(target_weights.get(dataset_id, pd.Series(0.0, index=equity.index)).abs().max()),
                    ),
                )
                for dataset_id in asset_ids
            ),
            key=lambda item: item[1],
            reverse=True,
        )[: int(max_assets)]
    )

    trade_rows = []
    for trade in list(getattr(portfolio_result, "trades", []) or []):
        timestamp = pd.to_datetime(getattr(trade, "timestamp", None), utc=True, errors="coerce")
        if pd.isna(timestamp):
            continue
        trade_rows.append(
            {
                "dataset_id": str(getattr(trade, "dataset_id", "")),
                "source_dataset_id": str(getattr(trade, "source_dataset_id", "")),
                "strategy_block_id": str(getattr(trade, "strategy_block_id", "")),
                "timestamp": timestamp,
                "side": str(getattr(trade, "side", "")),
                "qty": float(getattr(trade, "qty", 0.0)),
                "price": float(getattr(trade, "price", 0.0)),
                "fee": float(getattr(trade, "fee", 0.0)),
                "realized_pnl": float(getattr(trade, "realized_pnl", 0.0)),
                "equity_after": float(getattr(trade, "equity_after", np.nan)),
            }
        )
    trades = pd.DataFrame(
        trade_rows,
        columns=[
            "dataset_id",
            "source_dataset_id",
            "strategy_block_id",
            "timestamp",
            "side",
            "qty",
            "price",
            "fee",
            "realized_pnl",
            "equity_after",
        ],
    )
    if not trades.empty:
        trades = trades.sort_values("timestamp").reset_index(drop=True)

    return PortfolioChartData(
        equity_curve=equity,
        cash_weight=cash_weight,
        gross_exposure=gross_exposure,
        target_gross_exposure=target_gross_exposure,
        asset_weights=asset_weights,
        target_weights=target_weights,
        top_assets=top_assets,
        trades=trades,
    )


def portfolio_report_frame(report: PortfolioReport) -> pd.DataFrame:
    rows = []
    for asset in report.asset_rows:
        rows.append(
            {
                "dataset_id": asset.dataset_id,
                "avg_weight": asset.avg_weight,
                "min_weight": asset.min_weight,
                "avg_long_weight": asset.avg_long_weight,
                "avg_short_weight": asset.avg_short_weight,
                "peak_short_weight": asset.peak_short_weight,
                "avg_target_weight": asset.avg_target_weight,
                "avg_abs_tracking_error": asset.avg_abs_tracking_error,
                "avg_abs_weight_change": asset.avg_abs_weight_change,
                "final_weight": asset.final_weight,
                "peak_weight": asset.peak_weight,
                "active_bar_fraction": asset.active_bar_fraction,
                "trade_count": asset.trade_count,
                "realized_pnl": asset.realized_pnl,
                "unrealized_pnl": asset.unrealized_pnl,
                "turnover_notional": asset.turnover_notional,
                "turnover_ratio": asset.turnover_ratio,
                "avg_return_contribution": asset.avg_return_contribution,
                "total_return_contribution": asset.total_return_contribution,
                "contribution_share": asset.contribution_share,
            }
        )
    return pd.DataFrame(rows)


def portfolio_strategy_report_frame(report: PortfolioReport) -> pd.DataFrame:
    rows = []
    for strategy in report.strategy_rows:
        rows.append(
            {
                "strategy_block_id": strategy.strategy_block_id,
                "strategy_name": strategy.strategy_name,
                "budget_weight": strategy.budget_weight,
                "asset_count": strategy.asset_count,
                "avg_weight": strategy.avg_weight,
                "min_weight": strategy.min_weight,
                "avg_long_weight": strategy.avg_long_weight,
                "avg_short_weight": strategy.avg_short_weight,
                "peak_short_weight": strategy.peak_short_weight,
                "avg_target_weight": strategy.avg_target_weight,
                "avg_abs_tracking_error": strategy.avg_abs_tracking_error,
                "avg_abs_weight_change": strategy.avg_abs_weight_change,
                "final_weight": strategy.final_weight,
                "peak_weight": strategy.peak_weight,
                "active_bar_fraction": strategy.active_bar_fraction,
                "trade_count": strategy.trade_count,
                "realized_pnl": strategy.realized_pnl,
                "turnover_notional": strategy.turnover_notional,
                "turnover_ratio": strategy.turnover_ratio,
                "avg_return_contribution": strategy.avg_return_contribution,
                "total_return_contribution": strategy.total_return_contribution,
                "contribution_share": strategy.contribution_share,
            }
        )
    return pd.DataFrame(rows)


def portfolio_drawdown_frame(report: PortfolioReport) -> pd.DataFrame:
    rows = []
    for episode in report.drawdown_episodes:
        rows.append(
            {
                "rank": episode.rank,
                "peak_time": episode.peak_time,
                "trough_time": episode.trough_time,
                "recovery_time": episode.recovery_time,
                "depth": episode.depth,
                "duration_bars": episode.duration_bars,
                "recovery_bars": episode.recovery_bars,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "rank",
            "peak_time",
            "trough_time",
            "recovery_time",
            "depth",
            "duration_bars",
            "recovery_bars",
        ],
    )


def _reconstruct_asset_pnl(
    *,
    trades: Sequence[object],
    asset_ids: Sequence[str],
    final_positions: pd.Series,
    final_prices: pd.Series,
) -> dict[str, dict[str, float]]:
    states: dict[str, dict[str, float]] = {
        str(dataset_id): {"position": 0.0, "avg_price": 0.0, "realized_pnl": 0.0}
        for dataset_id in asset_ids
    }

    ordered_trades = sorted(
        trades,
        key=lambda trade: pd.to_datetime(getattr(trade, "timestamp", None), utc=True, errors="coerce"),
    )
    for trade in ordered_trades:
        dataset_id = str(getattr(trade, "dataset_id", "asset"))
        state = states.setdefault(dataset_id, {"position": 0.0, "avg_price": 0.0, "realized_pnl": 0.0})
        qty = float(getattr(trade, "qty", 0.0))
        if abs(qty) < 1e-12:
            continue
        price = float(getattr(trade, "price", 0.0))
        fee = float(getattr(trade, "fee", 0.0))
        prev_position = float(state["position"])
        prev_avg_price = float(state["avg_price"])
        new_position = prev_position + qty

        if prev_position != 0.0 and (
            (prev_position > 0 > new_position)
            or (prev_position < 0 < new_position)
            or (prev_position > 0 and new_position < prev_position and new_position >= 0)
            or (prev_position < 0 and new_position > prev_position and new_position <= 0)
        ):
            if prev_position > 0:
                closed = min(abs(qty), prev_position)
                state["realized_pnl"] += (price - prev_avg_price) * closed
            else:
                closed = min(abs(qty), abs(prev_position))
                state["realized_pnl"] += (prev_avg_price - price) * closed

        state["realized_pnl"] -= fee

        if abs(new_position) < 1e-12:
            state["position"] = 0.0
            state["avg_price"] = 0.0
            continue

        if prev_position == 0.0 or (prev_position > 0 > new_position) or (prev_position < 0 < new_position):
            state["position"] = new_position
            state["avg_price"] = price
            continue

        prev_abs = abs(prev_position)
        new_abs = abs(new_position)
        if new_abs < prev_abs:
            state["position"] = new_position
            state["avg_price"] = prev_avg_price
            continue

        state["position"] = new_position
        state["avg_price"] = ((prev_avg_price * prev_abs) + (price * abs(qty))) / new_abs

    results: dict[str, dict[str, float]] = {}
    for dataset_id in asset_ids:
        state = states.get(str(dataset_id), {"position": 0.0, "avg_price": 0.0, "realized_pnl": 0.0})
        position = float(pd.to_numeric(pd.Series([final_positions.get(dataset_id, state["position"])]), errors="coerce").iloc[0])
        if not np.isfinite(position):
            position = float(state["position"])
        avg_price = float(state["avg_price"])
        final_price = float(pd.to_numeric(pd.Series([final_prices.get(dataset_id, np.nan)]), errors="coerce").iloc[0])
        if not np.isfinite(final_price):
            final_price = 0.0
        if position > 0.0:
            unrealized_pnl = (final_price - avg_price) * position
        elif position < 0.0:
            unrealized_pnl = (avg_price - final_price) * abs(position)
        else:
            unrealized_pnl = 0.0
        results[str(dataset_id)] = {
            "realized_pnl": float(state["realized_pnl"]),
            "unrealized_pnl": float(unrealized_pnl),
        }
    return results


def _annualized_volatility(returns: pd.Series, periods_per_year: float = 252.0) -> float:
    if returns.empty:
        return 0.0
    std = float(returns.std())
    if not np.isfinite(std):
        return 0.0
    return float(std * np.sqrt(periods_per_year))


def _downside_deviation(returns: pd.Series, periods_per_year: float = 252.0) -> float:
    if returns.empty:
        return 0.0
    downside = np.minimum(returns.to_numpy(dtype=float), 0.0)
    if downside.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(downside))) * np.sqrt(periods_per_year))


def _sortino_ratio(returns: pd.Series, periods_per_year: float = 252.0) -> float:
    downside_dev = _downside_deviation(returns, periods_per_year=periods_per_year)
    if downside_dev <= 1e-12:
        return 0.0
    return float((returns.mean() * periods_per_year) / downside_dev)


def _max_drawdown_duration_bars(drawdown: pd.Series) -> int:
    if drawdown.empty:
        return 0
    underwater = drawdown < -1e-12
    longest = 0
    current = 0
    for is_underwater in underwater.to_numpy(dtype=bool):
        if is_underwater:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def _top_drawdown_episodes(drawdown: pd.Series, *, top_n: int = 5) -> list[PortfolioDrawdownEpisode]:
    if drawdown.empty:
        return []
    episodes: list[PortfolioDrawdownEpisode] = []
    underwater = drawdown < -1e-12
    in_episode = False
    start_idx = 0
    for idx, is_underwater in enumerate(underwater.to_numpy(dtype=bool)):
        if is_underwater and not in_episode:
            start_idx = max(idx - 1, 0)
            in_episode = True
        elif not is_underwater and in_episode:
            episodes.append(_build_drawdown_episode(drawdown, start_idx=start_idx, end_idx=idx - 1, recovery_idx=idx))
            in_episode = False
    if in_episode:
        episodes.append(_build_drawdown_episode(drawdown, start_idx=start_idx, end_idx=len(drawdown) - 1, recovery_idx=None))
    episodes.sort(key=lambda episode: episode.depth)
    for rank, episode in enumerate(episodes[:top_n], start=1):
        episodes[rank - 1] = PortfolioDrawdownEpisode(
            rank=rank,
            peak_time=episode.peak_time,
            trough_time=episode.trough_time,
            recovery_time=episode.recovery_time,
            depth=episode.depth,
            duration_bars=episode.duration_bars,
            recovery_bars=episode.recovery_bars,
        )
    return episodes[:top_n]


def _build_drawdown_episode(
    drawdown: pd.Series,
    *,
    start_idx: int,
    end_idx: int,
    recovery_idx: int | None,
) -> PortfolioDrawdownEpisode:
    window = drawdown.iloc[start_idx : end_idx + 1]
    if window.empty:
        peak_time = drawdown.index[start_idx] if len(drawdown.index) > start_idx else None
        trough_time = peak_time
        depth = 0.0
    else:
        trough_loc = int(window.argmin())
        trough_time = window.index[trough_loc]
        peak_time = drawdown.index[start_idx] if len(drawdown.index) > start_idx else None
        depth = float(window.min())
    recovery_time = drawdown.index[recovery_idx] if recovery_idx is not None and recovery_idx < len(drawdown.index) else None
    recovery_bars = (recovery_idx - end_idx) if recovery_idx is not None else None
    return PortfolioDrawdownEpisode(
        rank=0,
        peak_time=peak_time,
        trough_time=trough_time,
        recovery_time=recovery_time,
        depth=depth,
        duration_bars=max(0, end_idx - start_idx + 1),
        recovery_bars=int(recovery_bars) if recovery_bars is not None else None,
    )


def _infer_asset_prices(*, market_values: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    if market_values.empty:
        return pd.DataFrame(index=market_values.index)
    prices = pd.DataFrame(index=market_values.index, columns=market_values.columns, dtype=float)
    for dataset_id in market_values.columns:
        mv = pd.to_numeric(market_values[dataset_id], errors="coerce")
        pos = pd.to_numeric(positions.get(dataset_id, pd.Series(0.0, index=market_values.index)), errors="coerce")
        inferred = mv.where(pos.abs() <= 1e-12, mv / pos.replace(0.0, np.nan))
        inferred = inferred.replace([np.inf, -np.inf], np.nan)
        inferred = inferred.where(inferred > 0.0)
        prices[dataset_id] = inferred.ffill().bfill().fillna(0.0)
    return prices
