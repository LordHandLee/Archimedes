from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pandas as pd


def parse_catalog_timestamp(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    try:
        ts = pd.to_datetime(value, utc=True, errors="coerce")
    except Exception:
        return None
    if ts is pd.NaT or pd.isna(ts):
        return None
    return ts


def duration_seconds(start: str | None, end: str | None) -> float | None:
    start_ts = parse_catalog_timestamp(start)
    end_ts = parse_catalog_timestamp(end)
    if start_ts is None or end_ts is None:
        return None
    return max(0.0, float((end_ts - start_ts).total_seconds()))


@dataclass(frozen=True)
class EngineRunSnapshot:
    run_id: str
    logical_run_id: str | None
    engine_impl: str | None
    requested_execution_mode: str | None
    resolved_execution_mode: str | None
    engine_version: str | None
    run_started_at: str | None
    run_finished_at: str | None
    status: str | None
    total_return: float | None
    sharpe: float | None
    rolling_sharpe: float | None
    max_drawdown: float | None
    fallback_reason: str | None

    @property
    def duration_seconds(self) -> float | None:
        return duration_seconds(self.run_started_at, self.run_finished_at)


@dataclass(frozen=True)
class EngineComparisonSummary:
    logical_run_id: str | None
    latest_reference: EngineRunSnapshot | None
    latest_vectorized: EngineRunSnapshot | None
    speedup_vs_reference: float | None
    total_return_delta: float | None
    sharpe_delta: float | None
    rolling_sharpe_delta: float | None
    max_drawdown_delta: float | None
    available_engines: tuple[str, ...]
    compared_run_count: int


@dataclass(frozen=True)
class EngineBatchComparisonSummary:
    groups: tuple[EngineComparisonSummary, ...]
    total_groups: int
    paired_groups: int
    reference_only_groups: int
    vectorized_only_groups: int
    median_speedup_vs_reference: float | None
    mean_speedup_vs_reference: float | None
    max_abs_total_return_delta: float | None
    max_abs_sharpe_delta: float | None
    max_abs_max_drawdown_delta: float | None


def summarize_engine_runs(runs: Sequence[Any]) -> EngineComparisonSummary:
    snapshots = [_snapshot_from_run(run) for run in runs]
    reference_runs = [snap for snap in snapshots if _engine_bucket(snap) == "reference"]
    vectorized_runs = [snap for snap in snapshots if _engine_bucket(snap) == "vectorized"]
    latest_reference = max(reference_runs, key=_snapshot_sort_key, default=None)
    latest_vectorized = max(vectorized_runs, key=_snapshot_sort_key, default=None)

    available_engines = tuple(
        sorted(
            {
                engine
                for engine in (_engine_bucket(snap) for snap in snapshots)
                if engine is not None
            }
        )
    )

    return EngineComparisonSummary(
        logical_run_id=next((snap.logical_run_id for snap in snapshots if snap.logical_run_id), None),
        latest_reference=latest_reference,
        latest_vectorized=latest_vectorized,
        speedup_vs_reference=_speedup_vs_reference(latest_reference, latest_vectorized),
        total_return_delta=_metric_delta(latest_vectorized, latest_reference, "total_return"),
        sharpe_delta=_metric_delta(latest_vectorized, latest_reference, "sharpe"),
        rolling_sharpe_delta=_metric_delta(latest_vectorized, latest_reference, "rolling_sharpe"),
        max_drawdown_delta=_metric_delta(latest_vectorized, latest_reference, "max_drawdown"),
        available_engines=available_engines,
        compared_run_count=len(snapshots),
    )


def summarize_engine_batch(runs: Sequence[Any]) -> EngineBatchComparisonSummary:
    groups: list[EngineComparisonSummary] = []
    grouped_runs: dict[str, list[Any]] = {}
    for run in runs:
        logical_run_id = getattr(run, "logical_run_id", None)
        if logical_run_id:
            key = f"logical::{logical_run_id}"
        else:
            key = f"run::{getattr(run, 'run_id', '')}"
        grouped_runs.setdefault(key, []).append(run)

    for key in sorted(grouped_runs.keys()):
        groups.append(summarize_engine_runs(grouped_runs[key]))

    paired_groups = sum(1 for group in groups if group.latest_reference is not None and group.latest_vectorized is not None)
    reference_only_groups = sum(1 for group in groups if group.latest_reference is not None and group.latest_vectorized is None)
    vectorized_only_groups = sum(1 for group in groups if group.latest_reference is None and group.latest_vectorized is not None)

    speedups = [group.speedup_vs_reference for group in groups if group.speedup_vs_reference is not None]
    total_return_deltas = [abs(group.total_return_delta) for group in groups if group.total_return_delta is not None]
    sharpe_deltas = [abs(group.sharpe_delta) for group in groups if group.sharpe_delta is not None]
    max_drawdown_deltas = [abs(group.max_drawdown_delta) for group in groups if group.max_drawdown_delta is not None]

    return EngineBatchComparisonSummary(
        groups=tuple(groups),
        total_groups=len(groups),
        paired_groups=paired_groups,
        reference_only_groups=reference_only_groups,
        vectorized_only_groups=vectorized_only_groups,
        median_speedup_vs_reference=_median(speedups),
        mean_speedup_vs_reference=_mean(speedups),
        max_abs_total_return_delta=max(total_return_deltas, default=None),
        max_abs_sharpe_delta=max(sharpe_deltas, default=None),
        max_abs_max_drawdown_delta=max(max_drawdown_deltas, default=None),
    )


def _snapshot_from_run(run: Any) -> EngineRunSnapshot:
    metrics = getattr(run, "metrics", None)
    return EngineRunSnapshot(
        run_id=str(getattr(run, "run_id", "")),
        logical_run_id=getattr(run, "logical_run_id", None),
        engine_impl=getattr(run, "engine_impl", None),
        requested_execution_mode=getattr(run, "requested_execution_mode", None),
        resolved_execution_mode=getattr(run, "resolved_execution_mode", None),
        engine_version=getattr(run, "engine_version", None),
        run_started_at=getattr(run, "run_started_at", None),
        run_finished_at=getattr(run, "run_finished_at", None),
        status=getattr(run, "status", None),
        total_return=_metric_value(metrics, "total_return"),
        sharpe=_metric_value(metrics, "sharpe"),
        rolling_sharpe=_metric_value(metrics, "rolling_sharpe"),
        max_drawdown=_metric_value(metrics, "max_drawdown"),
        fallback_reason=getattr(run, "fallback_reason", None),
    )


def _metric_value(metrics: Any, key: str) -> float | None:
    if metrics is None:
        return None
    value: Any
    if isinstance(metrics, Mapping):
        value = metrics.get(key)
    else:
        value = getattr(metrics, key, None)
    if value is None:
        return None
    try:
        numeric = float(value)
    except Exception:
        return None
    if math.isnan(numeric):
        return None
    return numeric


def _engine_bucket(snapshot: EngineRunSnapshot) -> str | None:
    for candidate in (snapshot.engine_impl, snapshot.resolved_execution_mode, snapshot.requested_execution_mode):
        if not candidate:
            continue
        name = str(candidate).strip().lower()
        if name in {"reference", "vectorized"}:
            return name
    return None


def _snapshot_sort_key(snapshot: EngineRunSnapshot) -> tuple[int, int, int, str]:
    finished = parse_catalog_timestamp(snapshot.run_finished_at)
    started = parse_catalog_timestamp(snapshot.run_started_at)
    status_score = 1 if str(snapshot.status or "").lower() == "finished" else 0
    finished_score = finished.value if finished is not None else -1
    started_score = started.value if started is not None else -1
    return (status_score, finished_score, started_score, snapshot.run_id)


def _speedup_vs_reference(
    reference: EngineRunSnapshot | None, vectorized: EngineRunSnapshot | None
) -> float | None:
    if reference is None or vectorized is None:
        return None
    ref_seconds = reference.duration_seconds
    vec_seconds = vectorized.duration_seconds
    if ref_seconds is None or vec_seconds is None or vec_seconds <= 0:
        return None
    return ref_seconds / vec_seconds


def _metric_delta(
    vectorized: EngineRunSnapshot | None, reference: EngineRunSnapshot | None, field: str
) -> float | None:
    if vectorized is None or reference is None:
        return None
    lhs = getattr(vectorized, field)
    rhs = getattr(reference, field)
    if lhs is None or rhs is None:
        return None
    return lhs - rhs


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    series = pd.Series(list(values), dtype=float)
    value = float(series.median())
    if math.isnan(value):
        return None
    return value


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    series = pd.Series(list(values), dtype=float)
    value = float(series.mean())
    if math.isnan(value):
        return None
    return value
