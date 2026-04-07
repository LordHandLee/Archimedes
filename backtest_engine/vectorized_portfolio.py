from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence, Type

import numpy as np
import pandas as pd

from .engine import BacktestConfig
from .metrics import PerformanceMetrics, compute_metrics
from .sample_strategies import compute_zscore_mean_reversion_features
from .strategy import Strategy
from .vectorized_engine import VectorizedEngine
from .vectorized_strategies import VectorizedSupport, get_vectorized_adapter

ALLOCATION_OWNERSHIP_STRATEGY = "strategy_owned"
ALLOCATION_OWNERSHIP_PORTFOLIO = "portfolio_owned"
ALLOCATION_OWNERSHIP_HYBRID = "hybrid"

RANKING_MODE_NONE = "none"
RANKING_MODE_TOP_N = "top_n"
RANKING_MODE_SCORE_THRESHOLD = "score_threshold"
RANKING_MODE_TOP_N_OVER_SCORE_THRESHOLD = "top_n_over_score_threshold"

WEIGHTING_MODE_PRESERVE = "preserve"
WEIGHTING_MODE_EQUAL_SELECTED = "equal_selected"
WEIGHTING_MODE_SCORE_PROPORTIONAL = "score_proportional"

REBALANCE_MODE_ON_CHANGE = "on_change"
REBALANCE_MODE_ON_CHANGE_OR_PERIODIC = "on_change_or_periodic"
REBALANCE_MODE_ON_CHANGE_OR_DRIFT = "on_change_or_drift"
REBALANCE_MODE_ON_CHANGE_OR_PERIODIC_OR_DRIFT = "on_change_or_periodic_or_drift"


@dataclass(frozen=True)
class PortfolioAssetSpec:
    dataset_id: str
    data: pd.DataFrame
    strategy_cls: Type[Strategy]
    strategy_params: Dict[str, Any]
    target_weight: float | None = None


@dataclass(frozen=True)
class PortfolioConstructionConfig:
    allocation_ownership: str = ALLOCATION_OWNERSHIP_STRATEGY
    ranking_mode: str = RANKING_MODE_NONE
    max_ranked_assets: int | None = None
    min_rank_score: float | None = None
    weighting_mode: str = WEIGHTING_MODE_PRESERVE
    min_active_weight: float | None = None
    max_asset_weight: float | None = None
    cash_reserve_weight: float = 0.0
    rebalance_mode: str = REBALANCE_MODE_ON_CHANGE
    rebalance_every_n_bars: int | None = None
    rebalance_weight_drift_threshold: float | None = None


@dataclass(frozen=True)
class PortfolioTrade:
    dataset_id: str
    timestamp: pd.Timestamp
    side: str
    qty: float
    price: float
    fee: float
    realized_pnl: float
    equity_after: float


@dataclass(frozen=True)
class VectorizedPortfolioResult:
    portfolio_equity_curve: pd.Series
    asset_market_values: pd.DataFrame
    asset_weights: pd.DataFrame
    target_weights: pd.DataFrame
    positions: pd.DataFrame
    cash_curve: pd.Series
    trades: list[PortfolioTrade]
    metrics: PerformanceMetrics


@dataclass(frozen=True)
class _PortfolioIntent:
    strategy_weights: np.ndarray
    candidate_weights: np.ndarray
    scores: np.ndarray


@dataclass(frozen=True)
class VectorizedPortfolioEngine:
    engine_impl: str = "vectorized_portfolio"
    engine_version: str = "1"
    max_gross_exposure: float = 1.0

    def supports(
        self,
        assets: Sequence[PortfolioAssetSpec],
        config: BacktestConfig,
        construction_config: PortfolioConstructionConfig | None = None,
    ) -> VectorizedSupport:
        if not assets:
            return VectorizedSupport(False, "Portfolio vectorization requires at least one asset.")
        if config.base_execution:
            return VectorizedSupport(False, "Portfolio vectorization v1 does not support base_execution.")
        if config.intrabar_sim:
            return VectorizedSupport(False, "Portfolio vectorization v1 does not support intrabar simulation.")
        if config.allow_short:
            return VectorizedSupport(False, "Portfolio vectorization v1 is long-only.")

        construction = construction_config or PortfolioConstructionConfig()
        ownership = str(construction.allocation_ownership or ALLOCATION_OWNERSHIP_STRATEGY)
        ranking_mode = str(construction.ranking_mode or RANKING_MODE_NONE)
        weighting_mode = str(construction.weighting_mode or WEIGHTING_MODE_PRESERVE)
        rebalance_mode = str(construction.rebalance_mode or REBALANCE_MODE_ON_CHANGE)
        if ownership not in {
            ALLOCATION_OWNERSHIP_STRATEGY,
            ALLOCATION_OWNERSHIP_PORTFOLIO,
            ALLOCATION_OWNERSHIP_HYBRID,
        }:
            return VectorizedSupport(False, f"Unknown portfolio allocation ownership mode: {ownership}.")
        if ranking_mode not in {
            RANKING_MODE_NONE,
            RANKING_MODE_TOP_N,
            RANKING_MODE_SCORE_THRESHOLD,
            RANKING_MODE_TOP_N_OVER_SCORE_THRESHOLD,
        }:
            return VectorizedSupport(False, f"Unknown portfolio ranking mode: {ranking_mode}.")
        if ownership == ALLOCATION_OWNERSHIP_STRATEGY and ranking_mode != RANKING_MODE_NONE:
            return VectorizedSupport(
                False,
                "Strategy-owned allocation cannot be combined with portfolio ranking. Use Hybrid or Portfolio-Owned allocation.",
            )
        if ranking_mode == RANKING_MODE_TOP_N and (construction.max_ranked_assets is None or int(construction.max_ranked_assets) <= 0):
            return VectorizedSupport(False, "Top-N portfolio ranking requires max_ranked_assets > 0.")
        if ranking_mode == RANKING_MODE_SCORE_THRESHOLD and construction.min_rank_score is None:
            return VectorizedSupport(False, "Score-threshold ranking requires min_rank_score to be set.")
        if ranking_mode == RANKING_MODE_TOP_N_OVER_SCORE_THRESHOLD:
            if construction.min_rank_score is None:
                return VectorizedSupport(False, "Top-N-over-threshold ranking requires min_rank_score to be set.")
            if construction.max_ranked_assets is None or int(construction.max_ranked_assets) <= 0:
                return VectorizedSupport(False, "Top-N-over-threshold ranking requires max_ranked_assets > 0.")
        if weighting_mode not in {
            WEIGHTING_MODE_PRESERVE,
            WEIGHTING_MODE_EQUAL_SELECTED,
            WEIGHTING_MODE_SCORE_PROPORTIONAL,
        }:
            return VectorizedSupport(False, f"Unknown portfolio weighting mode: {weighting_mode}.")
        if ownership != ALLOCATION_OWNERSHIP_PORTFOLIO and weighting_mode != WEIGHTING_MODE_PRESERVE:
            return VectorizedSupport(
                False,
                "Portfolio weighting overrides require Portfolio-Owned allocation. Strategy-Owned and Hybrid modes must preserve strategy sizing.",
            )
        if construction.min_active_weight is not None and float(construction.min_active_weight) <= 0.0:
            return VectorizedSupport(False, "min_active_weight must be > 0 when provided.")
        if construction.max_asset_weight is not None and float(construction.max_asset_weight) <= 0.0:
            return VectorizedSupport(False, "max_asset_weight must be > 0 when provided.")
        if (
            construction.max_asset_weight is not None
            and construction.min_active_weight is not None
            and float(construction.max_asset_weight) < float(construction.min_active_weight)
        ):
            return VectorizedSupport(False, "max_asset_weight must be >= min_active_weight.")
        if not (0.0 <= float(construction.cash_reserve_weight or 0.0) < 1.0):
            return VectorizedSupport(False, "cash_reserve_weight must be between 0 and 1.")
        if rebalance_mode not in {
            REBALANCE_MODE_ON_CHANGE,
            REBALANCE_MODE_ON_CHANGE_OR_PERIODIC,
            REBALANCE_MODE_ON_CHANGE_OR_DRIFT,
            REBALANCE_MODE_ON_CHANGE_OR_PERIODIC_OR_DRIFT,
        }:
            return VectorizedSupport(False, f"Unknown portfolio rebalance mode: {rebalance_mode}.")
        if (
            rebalance_mode in {REBALANCE_MODE_ON_CHANGE_OR_PERIODIC, REBALANCE_MODE_ON_CHANGE_OR_PERIODIC_OR_DRIFT}
            and (construction.rebalance_every_n_bars is None or int(construction.rebalance_every_n_bars) <= 0)
        ):
            return VectorizedSupport(False, "Periodic portfolio rebalancing requires rebalance_every_n_bars > 0.")
        if (
            rebalance_mode in {REBALANCE_MODE_ON_CHANGE_OR_DRIFT, REBALANCE_MODE_ON_CHANGE_OR_PERIODIC_OR_DRIFT}
            and (construction.rebalance_weight_drift_threshold is None or float(construction.rebalance_weight_drift_threshold) <= 0.0)
        ):
            return VectorizedSupport(False, "Drift-threshold portfolio rebalancing requires rebalance_weight_drift_threshold > 0.")

        for asset in assets:
            if asset.target_weight is not None and float(asset.target_weight) < 0:
                return VectorizedSupport(False, f"{asset.dataset_id} has a negative target_weight, which is unsupported.")
            adapter = get_vectorized_adapter(asset.strategy_cls)
            if adapter is None:
                return VectorizedSupport(
                    False,
                    f"Strategy {asset.strategy_cls.__name__} has no vectorized adapter for portfolio execution.",
                )
            support = adapter.supports(config, asset.strategy_cls)
            if not support.supported:
                return support
            if asset.strategy_cls.__name__ not in {"SMACrossStrategy", "ZScoreMeanReversionStrategy"}:
                return VectorizedSupport(
                    False,
                    f"Strategy {asset.strategy_cls.__name__} has no portfolio intent builder yet.",
                )
        return VectorizedSupport(True)

    def run(
        self,
        assets: Sequence[PortfolioAssetSpec],
        config: BacktestConfig,
        *,
        normalize_weights: bool = True,
        construction_config: PortfolioConstructionConfig | None = None,
    ) -> VectorizedPortfolioResult:
        construction = construction_config or PortfolioConstructionConfig()
        support = self.supports(assets, config, construction)
        if not support.supported:
            raise RuntimeError(support.reason or "Portfolio vectorization is not supported for this workload.")

        helper = VectorizedEngine()
        aligned_assets = self._align_assets(assets, config, helper)
        index = aligned_assets[0][1].index
        n_bars = len(index)
        n_assets = len(aligned_assets)
        dataset_ids = [asset.dataset_id for asset, _ in aligned_assets]

        opens = np.column_stack([frame["open"].to_numpy(dtype=float) for _, frame in aligned_assets])
        closes = np.column_stack([frame["close"].to_numpy(dtype=float) for _, frame in aligned_assets])

        intents = [
            self._build_portfolio_intent(
                asset=asset,
                data=frame,
                config=config,
            )
            for asset, frame in aligned_assets
        ]
        strategy_weight_matrix = np.column_stack([intent.strategy_weights for intent in intents])
        candidate_weight_matrix = np.column_stack([intent.candidate_weights for intent in intents])
        score_matrix = np.column_stack([intent.scores for intent in intents])

        cash = float(config.starting_cash)
        positions = np.zeros(n_assets, dtype=float)
        avg_price = np.zeros(n_assets, dtype=float)
        realized_pnl = np.zeros(n_assets, dtype=float)
        target_weights = np.zeros(n_assets, dtype=float)
        target_qty = np.zeros(n_assets, dtype=float)

        buy_fee = helper._side_rate(config.fee_schedule, config.fee_rate, "buy")
        sell_fee = helper._side_rate(config.fee_schedule, config.fee_rate, "sell")
        buy_slip = helper._side_rate(config.slippage_schedule, config.slippage, "buy")
        sell_slip = helper._side_rate(config.slippage_schedule, config.slippage, "sell")

        equity_curve = np.zeros(n_bars, dtype=float)
        cash_curve = np.zeros(n_bars, dtype=float)
        market_values = np.zeros((n_bars, n_assets), dtype=float)
        actual_weights = np.zeros((n_bars, n_assets), dtype=float)
        target_weight_history = np.zeros((n_bars, n_assets), dtype=float)
        position_history = np.zeros((n_bars, n_assets), dtype=float)
        trades: list[PortfolioTrade] = []

        for bar_idx, timestamp in enumerate(index):
            execution_prices = closes[bar_idx, :] if config.fill_on_close else opens[bar_idx, :]
            mark_prices = closes[bar_idx, :]
            reference_prices = execution_prices
            if not config.fill_on_close and bar_idx > 0:
                reference_prices = closes[bar_idx - 1, :]
            reference_equity = cash + float(np.dot(positions, reference_prices))
            current_actual_weights = np.zeros(n_assets, dtype=float)
            if reference_equity > 0:
                current_actual_weights = (positions * reference_prices) / reference_equity
            desired_weights = self._construction_weights_for_bar(
                bar_idx=bar_idx,
                strategy_weight_matrix=strategy_weight_matrix,
                candidate_weight_matrix=candidate_weight_matrix,
                score_matrix=score_matrix,
                normalize_weights=normalize_weights,
                construction_config=construction,
            )
            target_weights_changed = not np.allclose(desired_weights, target_weights, atol=1e-12, rtol=0.0)
            periodic_due = self._is_periodic_rebalance_bar(bar_idx, construction)
            drift_due = self._is_drift_rebalance_due(current_actual_weights, target_weights, construction)
            if target_weights_changed or periodic_due or drift_due:
                portfolio_equity = reference_equity
                target_weights = desired_weights
                target_qty = self._weights_to_qty(target_weights, portfolio_equity, reference_prices)

            deltas = target_qty - positions
            if config.prevent_scale_in:
                scale_in_mask = (positions > 1e-12) & (deltas > 1e-12)
                deltas[scale_in_mask] = 0.0

            sell_indices = np.where(deltas < -1e-12)[0]
            buy_indices = np.where(deltas > 1e-12)[0]
            for asset_idx in sell_indices:
                trade = self._execute_order(
                    dataset_id=dataset_ids[asset_idx],
                    asset_idx=asset_idx,
                    qty=float(deltas[asset_idx]),
                    timestamp=timestamp,
                    execution_prices=execution_prices,
                    positions=positions,
                    avg_price=avg_price,
                    realized_pnl=realized_pnl,
                    cash_state={"cash": cash},
                    buy_fee=buy_fee,
                    sell_fee=sell_fee,
                    buy_slip=buy_slip,
                    sell_slip=sell_slip,
                    fill_ratio=config.fill_ratio,
                )
                cash = trade.equity_after - float(np.dot(positions, execution_prices)) if trade is not None else cash
                if trade is not None:
                    trades.append(trade)
            for asset_idx in buy_indices:
                trade = self._execute_order(
                    dataset_id=dataset_ids[asset_idx],
                    asset_idx=asset_idx,
                    qty=float(deltas[asset_idx]),
                    timestamp=timestamp,
                    execution_prices=execution_prices,
                    positions=positions,
                    avg_price=avg_price,
                    realized_pnl=realized_pnl,
                    cash_state={"cash": cash},
                    buy_fee=buy_fee,
                    sell_fee=sell_fee,
                    buy_slip=buy_slip,
                    sell_slip=sell_slip,
                    fill_ratio=config.fill_ratio,
                )
                cash = trade.equity_after - float(np.dot(positions, execution_prices)) if trade is not None else cash
                if trade is not None:
                    trades.append(trade)

            cash_curve[bar_idx] = cash
            market_values[bar_idx, :] = positions * mark_prices
            equity = cash + float(np.dot(positions, mark_prices))
            equity_curve[bar_idx] = equity
            if equity > 0:
                actual_weights[bar_idx, :] = market_values[bar_idx, :] / equity
            target_weight_history[bar_idx, :] = target_weights
            position_history[bar_idx, :] = positions

        equity_series = pd.Series(equity_curve, index=index, name="portfolio_equity")
        session_seconds = config.sharpe_session_seconds_per_day
        if session_seconds is None and config.sharpe_annualization == "equities" and config.sharpe_basis != "daily":
            session_seconds = helper._estimate_session_seconds_per_day(aligned_assets[0][1])
        metrics = compute_metrics(
            equity_series,
            risk_free_rate=config.risk_free_rate,
            timeframe=config.timeframe,
            annualization=config.sharpe_annualization,
            session_seconds_per_day=session_seconds,
            sharpe_basis=config.sharpe_basis,
        )
        return VectorizedPortfolioResult(
            portfolio_equity_curve=equity_series,
            asset_market_values=pd.DataFrame(market_values, index=index, columns=dataset_ids),
            asset_weights=pd.DataFrame(actual_weights, index=index, columns=dataset_ids),
            target_weights=pd.DataFrame(target_weight_history, index=index, columns=dataset_ids),
            positions=pd.DataFrame(position_history, index=index, columns=dataset_ids),
            cash_curve=pd.Series(cash_curve, index=index, name="cash"),
            trades=trades,
            metrics=metrics,
        )

    def _align_assets(
        self,
        assets: Sequence[PortfolioAssetSpec],
        config: BacktestConfig,
        helper: VectorizedEngine,
    ) -> list[tuple[PortfolioAssetSpec, pd.DataFrame]]:
        aligned: list[tuple[PortfolioAssetSpec, pd.DataFrame]] = []
        common_index: pd.DatetimeIndex | None = None
        for asset in assets:
            normalized = helper._normalize_data(asset.data)
            sliced = helper._slice_data(normalized, config)
            if common_index is None:
                common_index = sliced.index
            else:
                common_index = common_index.intersection(sliced.index)
            aligned.append((asset, sliced))
        if common_index is None or common_index.empty:
            raise ValueError("Portfolio vectorization could not align a non-empty common timestamp index.")
        return [(asset, frame.loc[common_index].copy()) for asset, frame in aligned]

    def _build_portfolio_intent(
        self,
        *,
        asset: PortfolioAssetSpec,
        data: pd.DataFrame,
        config: BacktestConfig,
    ) -> _PortfolioIntent:
        strategy_name = asset.strategy_cls.__name__
        if strategy_name == "SMACrossStrategy":
            return self._build_sma_intent(asset=asset, data=data, config=config)
        if strategy_name == "ZScoreMeanReversionStrategy":
            return self._build_zscore_intent(asset=asset, data=data, config=config)
        raise RuntimeError(f"Strategy {strategy_name} has no portfolio intent builder.")

    def _build_sma_intent(
        self,
        *,
        asset: PortfolioAssetSpec,
        data: pd.DataFrame,
        config: BacktestConfig,
    ) -> _PortfolioIntent:
        params = asset.strategy_params
        fast = int(params.get("fast", 10))
        slow = int(params.get("slow", 30))
        target = max(float(params.get("target", 1.0)), 0.0)
        preferred_weight = float(asset.target_weight) if asset.target_weight is not None else 1.0
        close = data["close"].astype(float)
        fast_series = close.rolling(fast).mean()
        slow_series = close.rolling(slow).mean()
        valid = fast_series.notna() & slow_series.notna()
        enter = (fast_series > slow_series) & valid
        exit_ = (fast_series < slow_series) & valid
        strategy_weights = self._state_weight_series(
            enter.to_numpy(dtype=bool),
            exit_.to_numpy(dtype=bool),
            base_target=target * preferred_weight,
            config=config,
        )
        candidate_weights = self._state_weight_series(
            enter.to_numpy(dtype=bool),
            exit_.to_numpy(dtype=bool),
            base_target=preferred_weight,
            config=config,
        )
        raw_score = ((fast_series - slow_series) / close.replace(0.0, np.nan)).fillna(0.0).to_numpy(dtype=float)
        scores = np.where(candidate_weights > 0.0, np.maximum(raw_score, 0.0), 0.0)
        return _PortfolioIntent(strategy_weights=strategy_weights, candidate_weights=candidate_weights, scores=scores)

    def _build_zscore_intent(
        self,
        *,
        asset: PortfolioAssetSpec,
        data: pd.DataFrame,
        config: BacktestConfig,
    ) -> _PortfolioIntent:
        params = asset.strategy_params
        target = max(float(params.get("target", 1.0)), 0.0)
        preferred_weight = float(asset.target_weight) if asset.target_weight is not None else 1.0
        features = compute_zscore_mean_reversion_features(data, params)
        enter = features["long_entry_signal"].to_numpy(dtype=bool)
        exit_ = features["long_exit_signal"].to_numpy(dtype=bool)
        strategy_weights = self._state_weight_series(
            enter,
            exit_,
            base_target=target * preferred_weight,
            config=config,
        )
        candidate_weights = self._state_weight_series(
            enter,
            exit_,
            base_target=preferred_weight,
            config=config,
        )
        z_score = features["z_score"].fillna(0.0).to_numpy(dtype=float)
        scores = np.where(candidate_weights > 0.0, np.maximum(-z_score, 0.0), 0.0)
        return _PortfolioIntent(strategy_weights=strategy_weights, candidate_weights=candidate_weights, scores=scores)

    @staticmethod
    def _state_weight_series(
        enter: np.ndarray,
        exit_: np.ndarray,
        *,
        base_target: float,
        config: BacktestConfig,
    ) -> np.ndarray:
        state = 0.0
        desired = np.zeros(len(enter), dtype=float)
        for bar_idx in range(len(enter)):
            if config.fill_on_close:
                if exit_[bar_idx]:
                    state = 0.0
                if enter[bar_idx]:
                    state = base_target
                desired[bar_idx] = state
                continue
            desired[bar_idx] = state
            if exit_[bar_idx]:
                state = 0.0
            if enter[bar_idx]:
                state = base_target
        return desired

    def _construction_weights_for_bar(
        self,
        *,
        bar_idx: int,
        strategy_weight_matrix: np.ndarray,
        candidate_weight_matrix: np.ndarray,
        score_matrix: np.ndarray,
        normalize_weights: bool,
        construction_config: PortfolioConstructionConfig,
    ) -> np.ndarray:
        ownership = str(construction_config.allocation_ownership or ALLOCATION_OWNERSHIP_STRATEGY)
        ranking_mode = str(construction_config.ranking_mode or RANKING_MODE_NONE)
        weighting_mode = str(construction_config.weighting_mode or WEIGHTING_MODE_PRESERVE)
        if ownership == ALLOCATION_OWNERSHIP_PORTFOLIO:
            raw_weights = candidate_weight_matrix[bar_idx, :].copy()
        else:
            raw_weights = strategy_weight_matrix[bar_idx, :].copy()
        if ranking_mode == RANKING_MODE_TOP_N:
            raw_weights = self._apply_top_n_ranking(
                raw_weights=raw_weights,
                scores=score_matrix[bar_idx, :],
                max_ranked_assets=int(construction_config.max_ranked_assets or 0),
            )
        elif ranking_mode == RANKING_MODE_SCORE_THRESHOLD:
            raw_weights = self._apply_score_threshold_ranking(
                raw_weights=raw_weights,
                scores=score_matrix[bar_idx, :],
                min_rank_score=float(construction_config.min_rank_score or 0.0),
            )
        elif ranking_mode == RANKING_MODE_TOP_N_OVER_SCORE_THRESHOLD:
            raw_weights = self._apply_top_n_over_score_threshold_ranking(
                raw_weights=raw_weights,
                scores=score_matrix[bar_idx, :],
                min_rank_score=float(construction_config.min_rank_score or 0.0),
                max_ranked_assets=int(construction_config.max_ranked_assets or 0),
            )
        weight_inputs = self._apply_weighting_mode(
            raw_weights=raw_weights,
            scores=score_matrix[bar_idx, :],
            weighting_mode=weighting_mode,
        )
        return self._resolve_desired_weights(
            raw_weights=weight_inputs,
            normalize_weights=normalize_weights,
            construction_config=construction_config,
        )

    @staticmethod
    def _apply_top_n_ranking(
        *,
        raw_weights: np.ndarray,
        scores: np.ndarray,
        max_ranked_assets: int,
    ) -> np.ndarray:
        ranked = np.zeros_like(raw_weights, dtype=float)
        if max_ranked_assets <= 0:
            return ranked
        candidate_indices = np.where(raw_weights > 1e-12)[0]
        if candidate_indices.size == 0:
            return ranked
        candidate_scores = np.nan_to_num(scores[candidate_indices], nan=-np.inf)
        order = np.lexsort((-raw_weights[candidate_indices], -candidate_scores))
        selected = candidate_indices[order[: min(max_ranked_assets, candidate_indices.size)]]
        ranked[selected] = raw_weights[selected]
        return ranked

    @staticmethod
    def _apply_score_threshold_ranking(
        *,
        raw_weights: np.ndarray,
        scores: np.ndarray,
        min_rank_score: float,
    ) -> np.ndarray:
        ranked = raw_weights.astype(float, copy=True)
        eligible = np.nan_to_num(scores, nan=-np.inf) >= float(min_rank_score)
        ranked[~eligible] = 0.0
        return ranked

    @classmethod
    def _apply_top_n_over_score_threshold_ranking(
        cls,
        *,
        raw_weights: np.ndarray,
        scores: np.ndarray,
        min_rank_score: float,
        max_ranked_assets: int,
    ) -> np.ndarray:
        thresholded = cls._apply_score_threshold_ranking(
            raw_weights=raw_weights,
            scores=scores,
            min_rank_score=min_rank_score,
        )
        return cls._apply_top_n_ranking(
            raw_weights=thresholded,
            scores=scores,
            max_ranked_assets=max_ranked_assets,
        )

    @staticmethod
    def _apply_weighting_mode(
        *,
        raw_weights: np.ndarray,
        scores: np.ndarray,
        weighting_mode: str,
    ) -> np.ndarray:
        weights = np.maximum(raw_weights.astype(float, copy=True), 0.0)
        active = weights > 1e-12
        if not np.any(active):
            return weights
        if weighting_mode == WEIGHTING_MODE_PRESERVE:
            return weights
        if weighting_mode == WEIGHTING_MODE_EQUAL_SELECTED:
            equalized = np.zeros_like(weights, dtype=float)
            equalized[active] = 1.0
            return equalized
        if weighting_mode == WEIGHTING_MODE_SCORE_PROPORTIONAL:
            score_inputs = np.maximum(np.nan_to_num(scores, nan=0.0), 0.0)
            weighted = np.zeros_like(weights, dtype=float)
            weighted[active] = weights[active] * score_inputs[active]
            if float(weighted.sum()) > 1e-12:
                return weighted
            return weights
        return weights

    @staticmethod
    def _is_periodic_rebalance_bar(bar_idx: int, construction_config: PortfolioConstructionConfig) -> bool:
        if str(construction_config.rebalance_mode or REBALANCE_MODE_ON_CHANGE) not in {
            REBALANCE_MODE_ON_CHANGE_OR_PERIODIC,
            REBALANCE_MODE_ON_CHANGE_OR_PERIODIC_OR_DRIFT,
        }:
            return False
        interval = max(int(construction_config.rebalance_every_n_bars or 0), 0)
        if interval <= 0:
            return False
        return bar_idx > 0 and (bar_idx % interval == 0)

    @staticmethod
    def _is_drift_rebalance_due(
        current_actual_weights: np.ndarray,
        target_weights: np.ndarray,
        construction_config: PortfolioConstructionConfig,
    ) -> bool:
        if str(construction_config.rebalance_mode or REBALANCE_MODE_ON_CHANGE) not in {
            REBALANCE_MODE_ON_CHANGE_OR_DRIFT,
            REBALANCE_MODE_ON_CHANGE_OR_PERIODIC_OR_DRIFT,
        }:
            return False
        threshold = float(construction_config.rebalance_weight_drift_threshold or 0.0)
        if threshold <= 0.0:
            return False
        return bool(np.max(np.abs(current_actual_weights - target_weights)) >= threshold)

    def _resolve_desired_weights(
        self,
        *,
        raw_weights: np.ndarray,
        normalize_weights: bool,
        construction_config: PortfolioConstructionConfig,
    ) -> np.ndarray:
        weights = np.maximum(raw_weights.astype(float, copy=True), 0.0)
        gross = float(weights.sum())
        if gross <= 0:
            return weights
        min_active_weight = (
            float(construction_config.min_active_weight)
            if construction_config.min_active_weight is not None
            else 0.0
        )
        max_asset_weight = (
            float(construction_config.max_asset_weight)
            if construction_config.max_asset_weight is not None
            else None
        )
        target_gross = float(self.max_gross_exposure)
        cash_reserve_weight = float(construction_config.cash_reserve_weight or 0.0)
        if cash_reserve_weight > 0.0:
            target_gross = min(target_gross, max(0.0, 1.0 - cash_reserve_weight))

        if normalize_weights:
            weights = self._normalize_with_constraints(
                weights=weights,
                target_gross=target_gross,
                min_active_weight=min_active_weight,
                max_asset_weight=max_asset_weight,
            )
            return weights

        if min_active_weight > 0.0:
            weights[weights < min_active_weight] = 0.0
        if max_asset_weight is not None:
            weights = np.minimum(weights, max_asset_weight)
        gross = float(weights.sum())
        if gross <= 0:
            return np.zeros_like(weights, dtype=float)
        if target_gross > 0 and gross > target_gross:
            weights *= target_gross / gross
        elif self.max_gross_exposure > 0 and gross > self.max_gross_exposure:
            weights *= self.max_gross_exposure / gross
        return weights

    @classmethod
    def _normalize_with_constraints(
        cls,
        *,
        weights: np.ndarray,
        target_gross: float,
        min_active_weight: float,
        max_asset_weight: float | None,
    ) -> np.ndarray:
        active_seed = np.maximum(weights.astype(float, copy=True), 0.0)
        active_seed[active_seed <= 1e-12] = 0.0
        if float(active_seed.sum()) <= 1e-12 or target_gross <= 0.0:
            return np.zeros_like(active_seed, dtype=float)

        current_seed = active_seed.copy()
        while True:
            allocated = cls._waterfill_with_cap(
                seed_weights=current_seed,
                target_gross=target_gross,
                max_asset_weight=max_asset_weight,
            )
            if min_active_weight <= 0.0:
                return allocated
            tiny_mask = (allocated > 1e-12) & (allocated < (min_active_weight - 1e-12))
            if not np.any(tiny_mask):
                return allocated
            current_seed[tiny_mask] = 0.0
            if float(current_seed.sum()) <= 1e-12:
                return np.zeros_like(active_seed, dtype=float)

    @staticmethod
    def _waterfill_with_cap(
        *,
        seed_weights: np.ndarray,
        target_gross: float,
        max_asset_weight: float | None,
    ) -> np.ndarray:
        seed = np.maximum(seed_weights.astype(float, copy=True), 0.0)
        if float(seed.sum()) <= 1e-12 or target_gross <= 0.0:
            return np.zeros_like(seed, dtype=float)

        if max_asset_weight is None:
            return (seed / float(seed.sum())) * target_gross

        result = np.zeros_like(seed, dtype=float)
        remaining_mask = seed > 1e-12
        remaining_target = float(target_gross)
        while np.any(remaining_mask) and remaining_target > 1e-12:
            remaining_seed = seed[remaining_mask]
            remaining_sum = float(remaining_seed.sum())
            if remaining_sum <= 1e-12:
                break
            provisional = (remaining_seed / remaining_sum) * remaining_target
            capped_local = provisional > (max_asset_weight + 1e-12)
            remaining_indices = np.where(remaining_mask)[0]
            if not np.any(capped_local):
                result[remaining_indices] = provisional
                break
            capped_indices = remaining_indices[capped_local]
            result[capped_indices] = max_asset_weight
            remaining_target -= max_asset_weight * len(capped_indices)
            remaining_mask[capped_indices] = False
        return result

    @staticmethod
    def _weights_to_qty(weights: np.ndarray, equity: float, prices: np.ndarray) -> np.ndarray:
        qty = np.zeros_like(weights, dtype=float)
        valid = prices > 0
        qty[valid] = (weights[valid] * equity) / prices[valid]
        return qty

    def _execute_order(
        self,
        *,
        dataset_id: str,
        asset_idx: int,
        qty: float,
        timestamp: pd.Timestamp,
        execution_prices: np.ndarray,
        positions: np.ndarray,
        avg_price: np.ndarray,
        realized_pnl: np.ndarray,
        cash_state: Dict[str, float],
        buy_fee: float,
        sell_fee: float,
        buy_slip: float,
        sell_slip: float,
        fill_ratio: float,
    ) -> PortfolioTrade | None:
        qty *= max(0.0, min(1.0, float(fill_ratio)))
        if abs(qty) < 1e-12:
            return None

        prev_qty = float(positions[asset_idx])
        if prev_qty <= 0 and qty < 0:
            return None

        side = "buy" if qty > 0 else "sell"
        price = float(execution_prices[asset_idx])
        slip = buy_slip if qty > 0 else sell_slip
        adj_price = price * (1.0 + slip if qty > 0 else 1.0 - slip)
        fee_rate = buy_fee if qty > 0 else sell_fee
        if qty > 0 and adj_price * (1.0 + fee_rate) > 0:
            max_affordable = max(float(cash_state["cash"]) / (adj_price * (1.0 + fee_rate)), 0.0)
            if qty > max_affordable:
                qty = max_affordable
                if qty < 1e-12:
                    return None

        if qty < 0:
            qty = -min(abs(qty), prev_qty)
            if abs(qty) < 1e-12:
                return None

        notional = qty * adj_price
        fee = abs(notional) * fee_rate
        new_qty = prev_qty + qty
        if qty < 0 and prev_qty > 0:
            closed = min(abs(qty), prev_qty)
            realized_pnl[asset_idx] += (adj_price - float(avg_price[asset_idx])) * closed

        positions[asset_idx] = new_qty
        if abs(new_qty) < 1e-12:
            positions[asset_idx] = 0.0
            avg_price[asset_idx] = 0.0
        elif prev_qty <= 0:
            avg_price[asset_idx] = adj_price
        elif qty > 0:
            prev_abs = abs(prev_qty)
            new_abs = abs(new_qty)
            avg_price[asset_idx] = ((float(avg_price[asset_idx]) * prev_abs) + (adj_price * abs(qty))) / new_abs

        cash_state["cash"] -= notional
        cash_state["cash"] -= fee
        equity_after = cash_state["cash"] + float(np.dot(positions, execution_prices))
        return PortfolioTrade(
            dataset_id=dataset_id,
            timestamp=timestamp,
            side=side,
            qty=float(qty),
            price=float(adj_price),
            fee=float(fee),
            realized_pnl=float(realized_pnl[asset_idx]),
            equity_after=float(equity_after),
        )
