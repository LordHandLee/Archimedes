from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PortfolioAssetAttribution:
    dataset_id: str
    avg_weight: float
    avg_target_weight: float
    avg_abs_tracking_error: float
    final_weight: float
    peak_weight: float
    active_bar_fraction: float
    trade_count: int
    realized_pnl: float
    turnover_notional: float
    turnover_ratio: float


@dataclass(frozen=True)
class PortfolioReport:
    starting_equity: float
    ending_equity: float
    total_return: float
    cagr: float
    max_drawdown: float
    sharpe: float
    rolling_sharpe: float
    avg_cash_weight: float
    min_cash_weight: float
    max_cash_weight: float
    avg_gross_exposure: float
    peak_gross_exposure: float
    avg_target_gross_exposure: float
    peak_target_gross_exposure: float
    trade_count: int
    total_turnover_notional: float
    total_turnover_ratio: float
    asset_rows: tuple[PortfolioAssetAttribution, ...]


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
    gross_exposure = asset_weights.abs().sum(axis=1) if not asset_weights.empty else pd.Series(0.0, index=equity.index)
    target_gross_exposure = (
        target_weights.abs().sum(axis=1) if not target_weights.empty else pd.Series(0.0, index=equity.index)
    )

    trades = list(getattr(portfolio_result, "trades", []) or [])
    turnover_notional_by_asset: dict[str, float] = {}
    trade_count_by_asset: dict[str, int] = {}
    realized_pnl_by_asset: dict[str, float] = {}
    for trade in trades:
        dataset_id = str(getattr(trade, "dataset_id", "asset"))
        turnover_notional_by_asset[dataset_id] = turnover_notional_by_asset.get(dataset_id, 0.0) + abs(
            float(getattr(trade, "qty", 0.0)) * float(getattr(trade, "price", 0.0))
        )
        trade_count_by_asset[dataset_id] = trade_count_by_asset.get(dataset_id, 0) + 1
        realized_pnl_by_asset[dataset_id] = float(getattr(trade, "realized_pnl", 0.0))

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

    asset_rows: list[PortfolioAssetAttribution] = []
    for dataset_id in asset_ids:
        actual = pd.to_numeric(asset_weights.get(dataset_id, pd.Series(0.0, index=equity.index)), errors="coerce").fillna(0.0)
        target = pd.to_numeric(target_weights.get(dataset_id, pd.Series(0.0, index=equity.index)), errors="coerce").fillna(0.0)
        turnover_notional = float(turnover_notional_by_asset.get(dataset_id, 0.0))
        asset_rows.append(
            PortfolioAssetAttribution(
                dataset_id=dataset_id,
                avg_weight=float(actual.mean()) if not actual.empty else 0.0,
                avg_target_weight=float(target.mean()) if not target.empty else 0.0,
                avg_abs_tracking_error=float((target - actual).abs().mean()) if not target.empty else 0.0,
                final_weight=float(actual.iloc[-1]) if not actual.empty else 0.0,
                peak_weight=float(actual.max()) if not actual.empty else 0.0,
                active_bar_fraction=float((target.abs() > 1e-9).mean()) if not target.empty else 0.0,
                trade_count=int(trade_count_by_asset.get(dataset_id, 0)),
                realized_pnl=float(realized_pnl_by_asset.get(dataset_id, 0.0)),
                turnover_notional=turnover_notional,
                turnover_ratio=(turnover_notional / start_equity) if start_equity > 0 else 0.0,
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
        avg_cash_weight=float(cash_weight.mean()),
        min_cash_weight=float(cash_weight.min()),
        max_cash_weight=float(cash_weight.max()),
        avg_gross_exposure=float(gross_exposure.mean()),
        peak_gross_exposure=float(gross_exposure.max()),
        avg_target_gross_exposure=float(target_gross_exposure.mean()),
        peak_target_gross_exposure=float(target_gross_exposure.max()),
        trade_count=len(trades),
        total_turnover_notional=total_turnover_notional,
        total_turnover_ratio=(total_turnover_notional / start_equity) if start_equity > 0 else 0.0,
        asset_rows=tuple(asset_rows),
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
                "avg_target_weight": asset.avg_target_weight,
                "avg_abs_tracking_error": asset.avg_abs_tracking_error,
                "final_weight": asset.final_weight,
                "peak_weight": asset.peak_weight,
                "active_bar_fraction": asset.active_bar_fraction,
                "trade_count": asset.trade_count,
                "realized_pnl": asset.realized_pnl,
                "turnover_notional": asset.turnover_notional,
                "turnover_ratio": asset.turnover_ratio,
            }
        )
    return pd.DataFrame(rows)
