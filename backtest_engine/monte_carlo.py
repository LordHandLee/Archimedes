from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from .catalog import ResultCatalog


MONTE_CARLO_SOURCE_WALK_FORWARD = "walk_forward"
MONTE_CARLO_MODE_BOOTSTRAP = "trade_bootstrap"
MONTE_CARLO_MODE_RESHUFFLE = "trade_reshuffle"


@dataclass(frozen=True)
class MonteCarloStudyArtifacts:
    mc_study_id: str
    source_type: str
    source_id: str
    resampling_mode: str
    simulation_count: int
    seed: int | None
    source_trade_count: int
    starting_equity: float
    summary: dict[str, Any]
    fan_quantiles: dict[str, list[float]]
    terminal_returns: np.ndarray
    max_drawdowns: np.ndarray
    terminal_equities: np.ndarray
    original_path: np.ndarray
    representative_paths: tuple[dict[str, Any], ...]


def run_monte_carlo_study(
    *,
    mc_study_id: str,
    source_type: str,
    source_id: str,
    catalog: ResultCatalog,
    resampling_mode: str = MONTE_CARLO_MODE_BOOTSTRAP,
    simulation_count: int = 500,
    seed: int | None = None,
    cost_stress_bps: float = 0.0,
    description: str = "",
) -> MonteCarloStudyArtifacts:
    if source_type != MONTE_CARLO_SOURCE_WALK_FORWARD:
        raise ValueError(f"Unsupported Monte Carlo source type: {source_type}")
    if resampling_mode not in {MONTE_CARLO_MODE_BOOTSTRAP, MONTE_CARLO_MODE_RESHUFFLE}:
        raise ValueError(f"Unsupported Monte Carlo resampling mode: {resampling_mode}")
    if int(simulation_count) <= 0:
        raise ValueError("Monte Carlo simulation_count must be positive.")

    source_returns, source_meta = extract_walk_forward_trade_returns(catalog, source_id)
    if source_returns.size == 0:
        raise ValueError("Monte Carlo source did not yield any realized trade-return observations.")

    starting_equity = float(source_meta.get("starting_equity", 100_000.0) or 100_000.0)
    cost_drag = float(cost_stress_bps or 0.0) / 10_000.0
    rng = np.random.default_rng(seed)
    n_obs = int(source_returns.size)
    sim_count = int(simulation_count)

    paths = np.empty((sim_count, n_obs + 1), dtype=float)
    terminal_returns = np.empty(sim_count, dtype=float)
    terminal_equities = np.empty(sim_count, dtype=float)
    max_drawdowns = np.empty(sim_count, dtype=float)

    original_path = _equity_path_from_returns(source_returns, starting_equity=starting_equity)

    for idx in range(sim_count):
        if resampling_mode == MONTE_CARLO_MODE_BOOTSTRAP:
            sampled = rng.choice(source_returns, size=n_obs, replace=True)
        else:
            sampled = rng.permutation(source_returns)
        adjusted = np.maximum(sampled - cost_drag, -0.999999)
        path = _equity_path_from_returns(adjusted, starting_equity=starting_equity)
        paths[idx, :] = path
        terminal_equity = float(path[-1])
        terminal_equities[idx] = terminal_equity
        terminal_returns[idx] = (terminal_equity / starting_equity) - 1.0
        max_drawdowns[idx] = _max_drawdown(path)

    fan_quantiles = {
        "p05": np.quantile(paths, 0.05, axis=0).astype(float).tolist(),
        "p25": np.quantile(paths, 0.25, axis=0).astype(float).tolist(),
        "p50": np.quantile(paths, 0.50, axis=0).astype(float).tolist(),
        "p75": np.quantile(paths, 0.75, axis=0).astype(float).tolist(),
        "p95": np.quantile(paths, 0.95, axis=0).astype(float).tolist(),
    }
    summary = _build_summary(
        terminal_returns=terminal_returns,
        terminal_equities=terminal_equities,
        max_drawdowns=max_drawdowns,
        source_trade_count=n_obs,
        cost_stress_bps=cost_stress_bps,
        source_meta=source_meta,
    )
    representative_paths = _representative_paths(
        paths=paths,
        terminal_returns=terminal_returns,
        max_drawdowns=max_drawdowns,
    )

    catalog.save_monte_carlo_study(
        mc_study_id=mc_study_id,
        source_type=source_type,
        source_id=source_id,
        resampling_mode=resampling_mode,
        simulation_count=sim_count,
        seed=seed,
        cost_stress_json={"return_drag_bps": float(cost_stress_bps or 0.0)},
        status="finished",
        description=description,
        source_trade_count=n_obs,
        starting_equity=starting_equity,
        summary_json=summary,
        fan_quantiles_json=fan_quantiles,
        terminal_returns_json=terminal_returns.astype(float).tolist(),
        max_drawdowns_json=max_drawdowns.astype(float).tolist(),
        terminal_equities_json=terminal_equities.astype(float).tolist(),
        original_path_json=original_path.astype(float).tolist(),
        representative_paths=list(representative_paths),
    )
    return MonteCarloStudyArtifacts(
        mc_study_id=mc_study_id,
        source_type=source_type,
        source_id=source_id,
        resampling_mode=resampling_mode,
        simulation_count=sim_count,
        seed=seed,
        source_trade_count=n_obs,
        starting_equity=starting_equity,
        summary=summary,
        fan_quantiles=fan_quantiles,
        terminal_returns=terminal_returns,
        max_drawdowns=max_drawdowns,
        terminal_equities=terminal_equities,
        original_path=original_path,
        representative_paths=representative_paths,
    )


def extract_walk_forward_trade_returns(catalog: ResultCatalog, wf_study_id: str) -> tuple[np.ndarray, dict[str, Any]]:
    folds = catalog.load_walk_forward_folds(str(wf_study_id))
    if not folds:
        raise ValueError(f"Walk-forward study '{wf_study_id}' does not contain any saved folds.")

    all_returns: list[float] = []
    run_ids: list[str] = []
    trade_unit_modes: list[str] = []
    starting_equity: float | None = None
    for fold in sorted(folds, key=lambda item: int(item.fold_index)):
        run_id = str(fold.test_run_id or "").strip()
        if not run_id:
            continue
        run = catalog.fetch(run_id)
        trades = catalog.load_trades(run_id)
        if run is None or not trades:
            continue
        if starting_equity is None and run.starting_cash is not None:
            starting_equity = float(run.starting_cash)
        run_returns, unit_mode = build_trade_return_sequence(
            trades=trades,
            starting_equity=float(run.starting_cash or 100_000.0),
        )
        if run_returns:
            all_returns.extend(run_returns)
            run_ids.append(run_id)
            trade_unit_modes.append(unit_mode)

    if not all_returns:
        raise ValueError(
            f"Walk-forward study '{wf_study_id}' did not yield any usable out-of-sample trade returns."
        )
    unit_mode = "trade_cycle" if all(mode == "trade_cycle" for mode in trade_unit_modes) else "trade_event_fallback"
    source_meta = {
        "wf_study_id": str(wf_study_id),
        "run_ids": run_ids,
        "starting_equity": float(starting_equity or 100_000.0),
        "unit_mode": unit_mode,
    }
    study_record = next(
        (row for row in catalog.load_walk_forward_studies() if str(row.wf_study_id) == str(wf_study_id)),
        None,
    )
    if study_record is not None:
        try:
            schedule_payload = json.loads(str(study_record.schedule_json or "{}"))
        except Exception:
            schedule_payload = {}
        try:
            params_payload = json.loads(str(study_record.params_json or "{}"))
        except Exception:
            params_payload = {}
        dataset_ids = [
            str(item)
            for item in list(schedule_payload.get("dataset_ids") or [])
            if str(item).strip()
        ]
        source_meta.update(
            {
                "strategy": str(study_record.strategy or ""),
                "dataset_id": str(study_record.dataset_id or ""),
                "timeframe": str(study_record.timeframe or ""),
                "fold_count": int(study_record.fold_count or 0),
                "candidate_source_mode": str(study_record.candidate_source_mode or ""),
                "selection_rule": str(study_record.selection_rule or ""),
                "is_portfolio": str(params_payload.get("source_kind", "")) == "portfolio",
                "portfolio_mode": str(schedule_payload.get("portfolio_mode", "") or ""),
                "portfolio_dataset_ids": dataset_ids,
                "source_study_id": str(params_payload.get("source_study_id", "") or ""),
                "source_batch_id": str(params_payload.get("source_batch_id", "") or ""),
            }
        )
    return np.asarray(all_returns, dtype=float), source_meta


def build_trade_return_sequence(*, trades: list[dict[str, Any]], starting_equity: float) -> tuple[list[float], str]:
    if not trades:
        return [], "none"
    cycle_returns: list[float] = []
    position_qty = 0.0
    cycle_open = False
    cycle_start_equity = float(starting_equity)
    previous_equity = float(starting_equity)

    for trade in trades:
        qty = _signed_trade_qty(trade)
        equity_after = _trade_equity_after(trade, fallback=previous_equity)
        prev_position = position_qty
        position_qty = prev_position + qty
        if not cycle_open and abs(prev_position) < 1e-12 and abs(position_qty) > 1e-12:
            cycle_open = True
            cycle_start_equity = previous_equity
        if cycle_open and abs(position_qty) < 1e-12:
            if cycle_start_equity != 0.0:
                cycle_returns.append((equity_after / cycle_start_equity) - 1.0)
            cycle_open = False
            cycle_start_equity = equity_after
        previous_equity = equity_after

    if cycle_returns:
        return cycle_returns, "trade_cycle"

    event_returns: list[float] = []
    previous_equity = float(starting_equity)
    for trade in trades:
        equity_after = _trade_equity_after(trade, fallback=previous_equity)
        if previous_equity != 0.0:
            event_returns.append((equity_after / previous_equity) - 1.0)
        previous_equity = equity_after
    return event_returns, "trade_event_fallback"


def _signed_trade_qty(trade: dict[str, Any]) -> float:
    try:
        qty = float(trade.get("qty", 0.0))
    except Exception:
        qty = 0.0
    side = str(trade.get("side", "") or "").strip().lower()
    if side == "sell" and qty > 0:
        return -qty
    if side == "buy" and qty < 0:
        return abs(qty)
    return qty


def _trade_equity_after(trade: dict[str, Any], *, fallback: float) -> float:
    try:
        value = float(trade.get("equity_after", fallback))
        if np.isfinite(value):
            return value
    except Exception:
        pass
    return float(fallback)


def _equity_path_from_returns(returns: np.ndarray, *, starting_equity: float) -> np.ndarray:
    path = np.empty(len(returns) + 1, dtype=float)
    path[0] = float(starting_equity)
    if len(returns) == 0:
        return path
    growth = np.cumprod(1.0 + np.asarray(returns, dtype=float))
    path[1:] = float(starting_equity) * growth
    return path


def _max_drawdown(path: np.ndarray) -> float:
    values = np.asarray(path, dtype=float)
    if values.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(values)
    drawdowns = np.divide(values - peaks, peaks, out=np.zeros_like(values), where=peaks != 0.0)
    return float(np.nanmin(drawdowns))


def _build_summary(
    *,
    terminal_returns: np.ndarray,
    terminal_equities: np.ndarray,
    max_drawdowns: np.ndarray,
    source_trade_count: int,
    cost_stress_bps: float,
    source_meta: dict[str, Any],
) -> dict[str, Any]:
    drawdown_thresholds = {}
    for threshold in (0.10, 0.20, 0.30):
        drawdown_thresholds[f"{int(threshold * 100)}pct"] = float(np.mean(np.abs(max_drawdowns) >= threshold))
    return_thresholds = {}
    for threshold in (-0.10, 0.0, 0.10):
        label = ("loss10" if threshold == -0.10 else "loss0" if threshold == 0.0 else "gain10")
        return_thresholds[label] = float(np.mean(terminal_returns <= threshold if threshold <= 0 else terminal_returns >= threshold))
    return {
        "terminal_return_p05": float(np.quantile(terminal_returns, 0.05)),
        "terminal_return_p25": float(np.quantile(terminal_returns, 0.25)),
        "terminal_return_p50": float(np.quantile(terminal_returns, 0.50)),
        "terminal_return_p75": float(np.quantile(terminal_returns, 0.75)),
        "terminal_return_p95": float(np.quantile(terminal_returns, 0.95)),
        "terminal_equity_p05": float(np.quantile(terminal_equities, 0.05)),
        "terminal_equity_p50": float(np.quantile(terminal_equities, 0.50)),
        "terminal_equity_p95": float(np.quantile(terminal_equities, 0.95)),
        "max_drawdown_p50": float(np.quantile(max_drawdowns, 0.50)),
        "max_drawdown_p95": float(np.quantile(max_drawdowns, 0.95)),
        "loss_probability": float(np.mean(terminal_returns < 0.0)),
        "drawdown_thresholds": drawdown_thresholds,
        "return_thresholds": return_thresholds,
        "source_trade_count": int(source_trade_count),
        "cost_stress_bps": float(cost_stress_bps or 0.0),
        "unit_mode": str(source_meta.get("unit_mode", "")),
        "source_strategy": str(source_meta.get("strategy", "")),
        "source_dataset_id": str(source_meta.get("dataset_id", "")),
        "source_timeframe": str(source_meta.get("timeframe", "")),
        "source_fold_count": int(source_meta.get("fold_count", 0) or 0),
        "source_candidate_mode": str(source_meta.get("candidate_source_mode", "")),
        "source_selection_rule": str(source_meta.get("selection_rule", "")),
        "source_is_portfolio": bool(source_meta.get("is_portfolio", False)),
        "source_portfolio_mode": str(source_meta.get("portfolio_mode", "")),
        "source_portfolio_asset_count": int(len(list(source_meta.get("portfolio_dataset_ids") or []))),
        "source_portfolio_assets": [str(item) for item in list(source_meta.get("portfolio_dataset_ids") or [])],
        "source_study_id": str(source_meta.get("source_study_id", "")),
        "source_batch_id": str(source_meta.get("source_batch_id", "")),
    }


def _representative_paths(
    *,
    paths: np.ndarray,
    terminal_returns: np.ndarray,
    max_drawdowns: np.ndarray,
) -> tuple[dict[str, Any], ...]:
    if paths.size == 0:
        return ()
    targets = {
        "median_path": float(np.quantile(terminal_returns, 0.50)),
        "p05_path": float(np.quantile(terminal_returns, 0.05)),
        "p95_path": float(np.quantile(terminal_returns, 0.95)),
    }
    rows: list[dict[str, Any]] = []
    for path_type, target in targets.items():
        idx = int(np.argmin(np.abs(terminal_returns - target)))
        rows.append(
            {
                "path_id": path_type,
                "path_type": path_type,
                "path": paths[idx, :].astype(float).tolist(),
                "summary": {
                    "terminal_return": float(terminal_returns[idx]),
                    "max_drawdown": float(max_drawdowns[idx]),
                    "simulation_index": idx,
                },
            }
        )
    worst_idx = int(np.argmin(max_drawdowns))
    rows.append(
        {
            "path_id": "worst_drawdown_path",
            "path_type": "worst_drawdown_path",
            "path": paths[worst_idx, :].astype(float).tolist(),
            "summary": {
                "terminal_return": float(terminal_returns[worst_idx]),
                "max_drawdown": float(max_drawdowns[worst_idx]),
                "simulation_index": worst_idx,
            },
        }
    )
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if row["path_id"] in seen:
            continue
        seen.add(row["path_id"])
        deduped.append(row)
    return tuple(deduped)
