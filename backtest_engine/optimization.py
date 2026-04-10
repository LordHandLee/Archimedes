from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


ROBUST_SCORE_VERSION = "robust_score_v1"


@dataclass(frozen=True)
class OptimizationStudyArtifacts:
    study_id: str
    batch_id: str
    strategy: str
    dataset_scope: tuple[str, ...]
    param_names: tuple[str, ...]
    timeframes: tuple[str, ...]
    horizons: tuple[str, ...]
    score_version: str
    aggregates: pd.DataFrame
    asset_results: pd.DataFrame


def compute_robust_score(
    *,
    median_sharpe: float,
    sharpe_std: float,
    worst_max_drawdown: float,
    profitable_asset_ratio: float,
) -> float:
    return float(
        float(median_sharpe)
        - (0.25 * float(sharpe_std))
        - (0.50 * abs(float(worst_max_drawdown)))
        + (0.20 * float(profitable_asset_ratio))
    )


def build_optimization_study_artifacts(
    *,
    df: pd.DataFrame,
    study_id: str,
    batch_id: str,
    strategy: str,
    dataset_scope: Sequence[str],
    param_names: Sequence[str],
    timeframes: Sequence[str],
    horizons: Sequence[str],
    score_version: str = ROBUST_SCORE_VERSION,
) -> OptimizationStudyArtifacts:
    if df is None or df.empty:
        raise ValueError("Optimization study requires a non-empty result frame.")

    params = [str(name) for name in param_names if str(name) in df.columns]
    if not params:
        raise ValueError("Optimization study requires at least one parameter column.")

    required = {"dataset_id", "timeframe", "start", "end", "total_return", "sharpe", "max_drawdown", "run_id"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Optimization study is missing required columns: {', '.join(missing)}")

    asset_results = df.copy()
    asset_results["start"] = asset_results["start"].apply(_normalize_timestamp_text)
    asset_results["end"] = asset_results["end"].apply(_normalize_timestamp_text)
    asset_results["params_json"] = asset_results.apply(
        lambda row: json.dumps({name: _json_safe_value(row[name]) for name in params}, sort_keys=True),
        axis=1,
    )
    asset_results["param_key"] = asset_results["params_json"]
    asset_results["worst_max_drawdown"] = asset_results["max_drawdown"].apply(_drawdown_magnitude)
    asset_results["profitable"] = (pd.to_numeric(asset_results["total_return"], errors="coerce").fillna(0.0) > 0.0).astype(int)

    group_cols = ["timeframe", "start", "end", "param_key", "params_json", *params]
    grouped = asset_results.groupby(group_cols, dropna=False, sort=True)
    aggregates = grouped.agg(
        dataset_count=("dataset_id", "nunique"),
        run_count=("run_id", "count"),
        median_total_return=("total_return", "median"),
        median_cagr=("cagr", "median"),
        median_sharpe=("sharpe", "median"),
        median_rolling_sharpe=("rolling_sharpe", "median"),
        worst_max_drawdown=("worst_max_drawdown", "max"),
        sharpe_std=("sharpe", lambda values: float(pd.Series(values, dtype=float).std(ddof=0) or 0.0)),
        profitable_asset_ratio=("profitable", "mean"),
    ).reset_index()
    numeric_defaults = {
        "median_total_return": 0.0,
        "median_cagr": 0.0,
        "median_sharpe": 0.0,
        "median_rolling_sharpe": 0.0,
        "worst_max_drawdown": 0.0,
        "sharpe_std": 0.0,
        "profitable_asset_ratio": 0.0,
    }
    for column, default_value in numeric_defaults.items():
        if column not in aggregates.columns:
            continue
        numeric = pd.to_numeric(aggregates[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        aggregates[column] = numeric.fillna(default_value)
    aggregates["robust_score"] = aggregates.apply(
        lambda row: compute_robust_score(
            median_sharpe=float(row["median_sharpe"]),
            sharpe_std=float(row["sharpe_std"]),
            worst_max_drawdown=float(row["worst_max_drawdown"]),
            profitable_asset_ratio=float(row["profitable_asset_ratio"]),
        ),
        axis=1,
    )
    aggregates = aggregates.sort_values(
        by=["robust_score", "median_sharpe", "median_total_return"],
        ascending=[False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    aggregates.insert(0, "seq", np.arange(1, len(aggregates) + 1, dtype=int))
    aggregates.insert(0, "study_id", str(study_id))

    asset_results.insert(0, "study_id", str(study_id))
    asset_results.insert(1, "seq", np.arange(1, len(asset_results) + 1, dtype=int))
    asset_results = asset_results.sort_values(
        by=["timeframe", "start", "end", "param_key", "dataset_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    asset_results["seq"] = np.arange(1, len(asset_results) + 1, dtype=int)

    return OptimizationStudyArtifacts(
        study_id=str(study_id),
        batch_id=str(batch_id),
        strategy=str(strategy),
        dataset_scope=tuple(str(item) for item in dataset_scope),
        param_names=tuple(params),
        timeframes=tuple(str(item) for item in timeframes),
        horizons=tuple(str(item) for item in horizons),
        score_version=str(score_version),
        aggregates=aggregates,
        asset_results=asset_results,
    )


def build_asset_distribution_frame(
    asset_results: pd.DataFrame,
    *,
    param_key: str,
    timeframe: str,
    start: str | None,
    end: str | None,
) -> pd.DataFrame:
    if asset_results is None or asset_results.empty:
        return pd.DataFrame(
            columns=[
                "dataset_id",
                "total_return",
                "sharpe",
                "rolling_sharpe",
                "max_drawdown",
                "run_id",
            ]
        )
    frame = asset_results.copy()
    frame = frame.loc[
        (frame["param_key"] == str(param_key))
        & (frame["timeframe"] == str(timeframe))
        & (frame["start"].fillna("") == str(start or ""))
        & (frame["end"].fillna("") == str(end or ""))
    ]
    return frame[
        [
            "dataset_id",
            "total_return",
            "sharpe",
            "rolling_sharpe",
            "max_drawdown",
            "run_id",
        ]
    ].sort_values(by=["sharpe", "total_return", "dataset_id"], ascending=[False, False, True], kind="mergesort")


def _drawdown_magnitude(value) -> float:
    try:
        numeric = float(value)
    except Exception:
        return 0.0
    if not np.isfinite(numeric):
        return 0.0
    return abs(numeric)


def _normalize_timestamp_text(value) -> str:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return ""
    return ts.isoformat()


def _json_safe_value(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        if numeric.is_integer():
            return int(numeric)
        return numeric
    if isinstance(value, pd.Timestamp):
        return _normalize_timestamp_text(value)
    return value
