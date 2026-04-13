from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .acquisition import (
    build_download_csv_path,
    build_download_dataset_id,
    load_download_artifact_state,
)
from .catalog import AcquisitionDatasetRecord, ResultCatalog
from .duckdb_store import DuckDBStore


ACQUISITION_ACTION_DOWNLOAD = "download"
ACQUISITION_ACTION_INGEST_EXISTING = "ingest_existing_csv"
ACQUISITION_ACTION_SKIP_FRESH = "skip_fresh"
ACQUISITION_ACTION_GAP_FILL_SECONDARY = "gap_fill_secondary"

ACQUISITION_PLAN_INITIAL_DOWNLOAD = "initial_download"
ACQUISITION_PLAN_SKIP_FRESH = "skip_fresh"
ACQUISITION_PLAN_FORCE_REFRESH = "force_refresh"
ACQUISITION_PLAN_INGEST_RAW = "ingest_existing_csv"
ACQUISITION_PLAN_INCREMENTAL_REFRESH = "incremental_refresh"
ACQUISITION_PLAN_BACKFILL = "backfill_refresh"
ACQUISITION_PLAN_INCREMENTAL_GAP_REPAIR_REFRESH = "incremental_gap_repair_refresh"
ACQUISITION_PLAN_BACKFILL_GAP_REPAIR_REFRESH = "backfill_gap_repair_refresh"
ACQUISITION_PLAN_FULL_REBUILD = "full_rebuild"
ACQUISITION_PLAN_GAP_REPAIR_REFRESH = "gap_repair_refresh"
ACQUISITION_PLAN_MULTI_WINDOW_GAP_REPAIR_REFRESH = "multi_window_gap_repair_refresh"
ACQUISITION_PLAN_COMPOUND_MULTI_WINDOW_REFRESH = "compound_multi_window_refresh"
ACQUISITION_PLAN_CROSS_SOURCE_GAP_FILL = "cross_source_gap_fill"
ACQUISITION_PLAN_CROSS_SOURCE_GAP_FILL_PLUS_INCREMENTAL_REFRESH = "cross_source_gap_fill_plus_incremental_refresh"
ACQUISITION_PLAN_GAP_REPAIR_REBUILD = ACQUISITION_PLAN_GAP_REPAIR_REFRESH
ACQUISITION_PLAN_RESUME_INTERRUPTED_DOWNLOAD = "resume_interrupted_download"


@dataclass(frozen=True)
class AcquisitionPolicyDecision:
    dataset_id: str
    source: str
    symbol: str
    resolution: str
    history_window: str
    action: str
    reason: str
    freshness_state: str
    csv_path: str
    parquet_path: str | None
    coverage_start: str | None
    coverage_end: str | None
    bar_count: int | None
    ingested: bool
    quality_state: str
    suspicious_gap_count: int
    max_suspicious_gap: str | None
    plan_type: str
    request_start: str | None
    request_end: str | None
    merge_with_existing: bool
    request_windows: tuple[tuple[str, str], ...] = ()
    secondary_source: str | None = None
    secondary_dataset_id: str | None = None
    parity_state: str = "unknown"
    parity_overlap_bars: int = 0
    parity_close_mae: float | None = None
    parity_close_mean_abs_bps: float | None = None
    secondary_request_windows: tuple[tuple[str, str], ...] = ()


def _freshness_threshold_for_resolution(resolution: str) -> pd.Timedelta:
    normalized = str(resolution or "").strip().lower()
    if normalized.endswith("m"):
        try:
            minutes = int(normalized[:-1] or "1")
        except Exception:
            minutes = 1
        if minutes <= 60:
            return pd.Timedelta(days=3)
        return pd.Timedelta(days=7)
    if normalized.endswith("h"):
        return pd.Timedelta(days=7)
    if normalized.endswith("d"):
        return pd.Timedelta(days=7)
    if normalized.endswith("w"):
        return pd.Timedelta(days=30)
    return pd.Timedelta(days=7)


def compute_freshness_state(
    coverage_end: str | None,
    resolution: str,
    *,
    now: pd.Timestamp | None = None,
) -> str:
    if not coverage_end:
        return "missing"
    stamp = pd.to_datetime(coverage_end, utc=True, errors="coerce")
    if pd.isna(stamp):
        return "unknown"
    current = now if now is not None else pd.Timestamp.now("UTC")
    threshold = _freshness_threshold_for_resolution(resolution)
    if stamp >= current - threshold:
        return "fresh"
    return "stale"


def _load_dataset_record(catalog: ResultCatalog, dataset_id: str) -> AcquisitionDatasetRecord | None:
    for record in catalog.load_acquisition_datasets():
        if str(record.dataset_id) == dataset_id:
            return record
    return None


def _history_window_to_start_timestamp(history_window: str, now: pd.Timestamp) -> pd.Timestamp | None:
    normalized = str(history_window or "").strip().lower()
    if not normalized or normalized == "max":
        return None
    try:
        if normalized.endswith("y"):
            return now - pd.DateOffset(years=int(normalized[:-1] or "1"))
        if normalized.endswith("mo"):
            return now - pd.DateOffset(months=int(normalized[:-2] or "1"))
        if normalized.endswith("m"):
            return now - pd.DateOffset(months=int(normalized[:-1] or "1"))
        if normalized.endswith("w"):
            return now - pd.Timedelta(weeks=int(normalized[:-1] or "1"))
        if normalized.endswith("d"):
            return now - pd.Timedelta(days=int(normalized[:-1] or "1"))
    except Exception:
        return None
    return None


def _safe_date_stamp(value: str | None) -> pd.Timestamp | None:
    stamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(stamp):
        return None
    return stamp


def _normalize_request_windows(windows: list[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    normalized: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for raw_start, raw_end in windows:
        start_ts = pd.to_datetime(raw_start, utc=True, errors="coerce")
        end_ts = pd.to_datetime(raw_end, utc=True, errors="coerce")
        if pd.isna(start_ts) or pd.isna(end_ts):
            continue
        if start_ts > end_ts:
            start_ts, end_ts = end_ts, start_ts
        normalized.append((start_ts.normalize(), end_ts.normalize()))
    if not normalized:
        return ()
    normalized.sort(key=lambda item: item[0])
    merged: list[list[pd.Timestamp]] = []
    for start_ts, end_ts in normalized:
        if not merged:
            merged.append([start_ts, end_ts])
            continue
        last_start, last_end = merged[-1]
        if start_ts <= last_end:
            merged[-1][1] = max(last_end, end_ts)
        else:
            merged.append([start_ts, end_ts])
    return tuple((start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")) for start, end in merged)


def _saved_request_windows(payload: object) -> tuple[tuple[str, str], ...]:
    windows: list[tuple[str, str]] = []
    for item in list(payload or []):
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        start = str(item[0] or "").strip()
        end = str(item[1] or "").strip()
        if not start or not end:
            continue
        windows.append((start, end))
    return _normalize_request_windows(windows)


def _find_secondary_gap_fill_candidate(
    *,
    catalog: ResultCatalog,
    store: DuckDBStore,
    dataset_id: str,
    symbol: str,
    source: str,
    resolution: str,
    gap_windows: tuple[tuple[str, str], ...],
) -> dict | None:
    if not gap_windows:
        return None
    candidates: list[dict] = []
    for record in catalog.load_acquisition_datasets():
        if str(record.dataset_id) == dataset_id:
            continue
        if str(record.symbol or "").strip().upper() != str(symbol or "").strip().upper():
            continue
        if str(record.source or "").strip().lower() == str(source or "").strip().lower():
            continue
        if str(record.resolution or "").strip().lower() != str(resolution or "").strip().lower():
            continue
        if not bool(record.ingested):
            continue
        other_path = store.dataset_path(str(record.dataset_id))
        if not other_path.exists():
            continue
        record_start = _safe_date_stamp(record.coverage_start)
        record_end = _safe_date_stamp(record.coverage_end)
        if record_start is None or record_end is None:
            continue
        covers_all = True
        for window_start, window_end in gap_windows:
            start_ts = _safe_date_stamp(window_start)
            end_ts = _safe_date_stamp(window_end)
            if start_ts is None or end_ts is None:
                covers_all = False
                break
            if record_start > start_ts or record_end < end_ts:
                covers_all = False
                break
        if not covers_all:
            continue
        parity = store.compare_dataset_parity(dataset_id, str(record.dataset_id))
        if str(parity.get("parity_state") or "") != "matching":
            continue
        candidates.append(
            {
                "dataset_id": str(record.dataset_id),
                "source": str(record.source or ""),
                "parity_state": str(parity.get("parity_state") or "unknown"),
                "parity_overlap_bars": int(parity.get("overlap_bar_count") or 0),
                "parity_close_mae": parity.get("close_mae"),
                "parity_close_mean_abs_bps": parity.get("close_mean_abs_bps"),
            }
        )
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: (
            -int(item.get("parity_overlap_bars") or 0),
            float(item.get("parity_close_mean_abs_bps") or 0.0),
            str(item.get("dataset_id") or ""),
        )
    )
    return candidates[0]


def decide_acquisition_policy(
    symbol: str,
    *,
    source: str,
    resolution: str,
    history_window: str,
    catalog: ResultCatalog | None = None,
    store: DuckDBStore | None = None,
    data_dir: Path | str = "data",
    now: pd.Timestamp | None = None,
    force_refresh: bool = False,
) -> AcquisitionPolicyDecision:
    dataset_id = build_download_dataset_id(
        symbol,
        source=source,
        history_window=history_window,
        resolution=resolution,
    )
    csv_path = build_download_csv_path(
        symbol,
        source=source,
        history_window=history_window,
        resolution=resolution,
        data_dir=data_dir,
    )
    resolved_catalog = catalog or ResultCatalog()
    resolved_store = store or DuckDBStore()
    current = now if now is not None else pd.Timestamp.now("UTC")
    record = _load_dataset_record(resolved_catalog, dataset_id)
    parquet_path = Path(record.parquet_path) if record and record.parquet_path else resolved_store.dataset_path(dataset_id)
    parquet_exists = parquet_path.exists()
    csv_exists = csv_path.exists()
    download_state = load_download_artifact_state(csv_path)
    download_state_status = str(download_state.get("status") or "").strip().lower()
    coverage_start = record.coverage_start if record else None
    coverage_end = record.coverage_end if record else None
    bar_count = record.bar_count if record else None
    ingested = bool(record.ingested) if record else False
    if parquet_exists and (coverage_end is None or bar_count is None):
        try:
            described = resolved_store.describe_dataset(dataset_id)
            coverage_start = coverage_start or described.get("coverage_start")
            coverage_end = coverage_end or described.get("coverage_end")
            bar_count = bar_count or described.get("bar_count")
            ingested = True
        except Exception:
            pass
    freshness_state = compute_freshness_state(coverage_end, resolution, now=current)
    quality_state = "unknown"
    suspicious_gap_count = 0
    max_suspicious_gap: str | None = None
    suspicious_gap_ranges: list[dict] = []
    gap_repair_request_start: str | None = None
    gap_repair_request_end: str | None = None
    if record and record.quality_analyzed_at:
        quality_state = str(record.quality_state or "unknown")
        suspicious_gap_count = int(record.suspicious_gap_count or 0)
        max_suspicious_gap = record.max_suspicious_gap
        gap_repair_request_start = str(record.repair_request_start or "").strip() or None
        gap_repair_request_end = str(record.repair_request_end or "").strip() or None
        try:
            suspicious_gap_ranges = list(json.loads(record.suspicious_gap_ranges_json or "[]"))
        except Exception:
            suspicious_gap_ranges = []
    elif parquet_exists and ingested:
        try:
            quality = resolved_store.inspect_dataset_quality(dataset_id, resolution)
            quality_state = str(quality.get("quality_state") or "unknown")
            suspicious_gap_count = int(quality.get("suspicious_gap_count") or 0)
            max_suspicious_gap = quality.get("max_suspicious_gap")
            suspicious_gap_ranges = list(quality.get("suspicious_gap_ranges") or [])
            gap_repair_request_start = str(quality.get("repair_request_start") or "").strip() or None
            gap_repair_request_end = str(quality.get("repair_request_end") or "").strip() or None
        except Exception:
            quality_state = "unknown"
            suspicious_gap_count = 0
            max_suspicious_gap = None
            suspicious_gap_ranges = []
            gap_repair_request_start = None
            gap_repair_request_end = None

    coverage_start_ts = pd.to_datetime(coverage_start, utc=True, errors="coerce")
    coverage_end_ts = pd.to_datetime(coverage_end, utc=True, errors="coerce")
    gap_repair_start_ts = _safe_date_stamp(gap_repair_request_start)
    gap_repair_end_ts = _safe_date_stamp(gap_repair_request_end)
    desired_history_start = _history_window_to_start_timestamp(history_window, current)
    needs_backfill = bool(
        desired_history_start is not None
        and pd.notna(coverage_start_ts)
        and coverage_start_ts > (desired_history_start + pd.Timedelta(days=2))
    )
    needs_incremental = bool(pd.notna(coverage_end_ts) and freshness_state == "stale")
    near_left_edge_gap = bool(
        gap_repair_start_ts is not None
        and pd.notna(coverage_start_ts)
        and gap_repair_start_ts <= (coverage_start_ts.normalize() + pd.Timedelta(days=7))
    )
    near_right_edge_gap = bool(
        gap_repair_end_ts is not None
        and pd.notna(coverage_end_ts)
        and gap_repair_end_ts >= (coverage_end_ts.normalize() - pd.Timedelta(days=7))
    )
    gap_windows = _normalize_request_windows(
        [
            (
                str(item.get("request_start") or ""),
                str(item.get("request_end") or ""),
            )
            for item in suspicious_gap_ranges
            if str(item.get("request_start") or "").strip() and str(item.get("request_end") or "").strip()
        ]
    )

    if force_refresh:
        return AcquisitionPolicyDecision(
            dataset_id=dataset_id,
            source=source,
            symbol=str(symbol).strip().upper(),
            resolution=resolution,
            history_window=history_window,
            action=ACQUISITION_ACTION_DOWNLOAD,
            reason="Forced refresh requested; re-downloading from the provider.",
            freshness_state=freshness_state,
            csv_path=str(csv_path),
            parquet_path=str(parquet_path) if parquet_exists else None,
            coverage_start=coverage_start,
            coverage_end=coverage_end,
            bar_count=bar_count,
            ingested=ingested,
            quality_state=quality_state,
            suspicious_gap_count=suspicious_gap_count,
            max_suspicious_gap=max_suspicious_gap,
            plan_type=ACQUISITION_PLAN_FORCE_REFRESH,
            request_start=None,
            request_end=None,
            merge_with_existing=False,
        )
    resumable_download_statuses = {"queued", "running", "paused", "interrupted", "retry_wait", "stopped"}
    if csv_exists and download_state_status in resumable_download_statuses:
        saved_windows = _saved_request_windows(download_state.get("request_windows"))
        try:
            saved_window_index = max(0, int(download_state.get("window_index") or 0))
        except Exception:
            saved_window_index = 0
        remaining_windows = saved_windows[saved_window_index:] if saved_windows else ()
        request_start = remaining_windows[0][0] if remaining_windows else None
        request_end = remaining_windows[0][1] if remaining_windows else None
        return AcquisitionPolicyDecision(
            dataset_id=dataset_id,
            source=source,
            symbol=str(symbol).strip().upper(),
            resolution=resolution,
            history_window=history_window,
            action=ACQUISITION_ACTION_DOWNLOAD,
            reason=(
                "A resumable raw CSV checkpoint already exists for this dataset, so the provider download will resume "
                "before any ingestion step."
            ),
            freshness_state=freshness_state,
            csv_path=str(csv_path),
            parquet_path=str(parquet_path) if parquet_exists else None,
            coverage_start=coverage_start,
            coverage_end=coverage_end,
            bar_count=bar_count,
            ingested=ingested,
            quality_state=quality_state,
            suspicious_gap_count=suspicious_gap_count,
            max_suspicious_gap=max_suspicious_gap,
            plan_type=ACQUISITION_PLAN_RESUME_INTERRUPTED_DOWNLOAD,
            request_start=request_start,
            request_end=request_end,
            merge_with_existing=bool(download_state.get("merge_with_existing") or parquet_exists),
            request_windows=remaining_windows,
            secondary_source=str(download_state.get("secondary_source") or "").strip() or None,
            secondary_dataset_id=str(download_state.get("secondary_dataset_id") or "").strip() or None,
        )
    if csv_exists and (not parquet_exists or not ingested):
        return AcquisitionPolicyDecision(
            dataset_id=dataset_id,
            source=source,
            symbol=str(symbol).strip().upper(),
            resolution=resolution,
            history_window=history_window,
            action=ACQUISITION_ACTION_INGEST_EXISTING,
            reason="A raw CSV already exists locally and the canonical dataset is missing or not ingested.",
            freshness_state=freshness_state,
            csv_path=str(csv_path),
            parquet_path=str(parquet_path) if parquet_exists else None,
            coverage_start=coverage_start,
            coverage_end=coverage_end,
            bar_count=bar_count,
            ingested=ingested,
            quality_state=quality_state,
            suspicious_gap_count=suspicious_gap_count,
            max_suspicious_gap=max_suspicious_gap,
            plan_type=ACQUISITION_PLAN_INGEST_RAW,
            request_start=None,
            request_end=None,
            merge_with_existing=parquet_exists,
        )
    if parquet_exists and ingested and freshness_state == "fresh" and quality_state != "gappy":
        if needs_backfill:
            request_start = desired_history_start.strftime("%Y-%m-%d") if desired_history_start is not None else None
            request_end = coverage_start_ts.strftime("%Y-%m-%d") if pd.notna(coverage_start_ts) else None
            return AcquisitionPolicyDecision(
                dataset_id=dataset_id,
                source=source,
                symbol=str(symbol).strip().upper(),
                resolution=resolution,
                history_window=history_window,
                action=ACQUISITION_ACTION_DOWNLOAD,
                reason="The dataset is fresh at the right edge but does not cover the full requested history window.",
                freshness_state=freshness_state,
                csv_path=str(csv_path),
                parquet_path=str(parquet_path),
                coverage_start=coverage_start,
                coverage_end=coverage_end,
                bar_count=bar_count,
                ingested=ingested,
                quality_state=quality_state,
                suspicious_gap_count=suspicious_gap_count,
                max_suspicious_gap=max_suspicious_gap,
                plan_type=ACQUISITION_PLAN_BACKFILL,
                request_start=request_start,
                request_end=request_end,
                merge_with_existing=True,
            )
        return AcquisitionPolicyDecision(
            dataset_id=dataset_id,
            source=source,
            symbol=str(symbol).strip().upper(),
            resolution=resolution,
            history_window=history_window,
            action=ACQUISITION_ACTION_SKIP_FRESH,
            reason="The local canonical dataset is already ingested and considered fresh for this resolution.",
            freshness_state=freshness_state,
            csv_path=str(csv_path),
            parquet_path=str(parquet_path),
            coverage_start=coverage_start,
            coverage_end=coverage_end,
            bar_count=bar_count,
            ingested=ingested,
            quality_state=quality_state,
            suspicious_gap_count=suspicious_gap_count,
            max_suspicious_gap=max_suspicious_gap,
            plan_type=ACQUISITION_PLAN_SKIP_FRESH,
            request_start=None,
            request_end=None,
            merge_with_existing=False,
        )
    selected_action = ACQUISITION_ACTION_DOWNLOAD
    plan_type = ACQUISITION_PLAN_INITIAL_DOWNLOAD
    request_start: str | None = None
    request_end: str | None = None
    merge_with_existing = False
    request_windows: tuple[tuple[str, str], ...] = ()
    secondary_source: str | None = None
    secondary_dataset_id: str | None = None
    parity_state = "unknown"
    parity_overlap_bars = 0
    parity_close_mae: float | None = None
    parity_close_mean_abs_bps: float | None = None
    secondary_request_windows: tuple[tuple[str, str], ...] = ()
    right_edge_window = (
        (coverage_end_ts.strftime("%Y-%m-%d"), current.strftime("%Y-%m-%d"))
        if pd.notna(coverage_end_ts) and needs_incremental
        else None
    )
    left_edge_window = (
        (
            desired_history_start.strftime("%Y-%m-%d"),
            coverage_start_ts.strftime("%Y-%m-%d"),
        )
        if desired_history_start is not None and pd.notna(coverage_start_ts) and needs_backfill
        else None
    )
    compound_windows_list = list(gap_windows)
    if left_edge_window:
        compound_windows_list.append(left_edge_window)
    if right_edge_window:
        compound_windows_list.append(right_edge_window)
    compound_windows = _normalize_request_windows(compound_windows_list)
    secondary_gap_fill_candidate = None
    if parquet_exists and ingested and quality_state == "gappy" and gap_windows and not needs_incremental and not needs_backfill:
        secondary_gap_fill_candidate = _find_secondary_gap_fill_candidate(
            catalog=resolved_catalog,
            store=resolved_store,
            dataset_id=dataset_id,
            symbol=str(symbol).strip().upper(),
            source=source,
            resolution=resolution,
            gap_windows=gap_windows,
        )
        if secondary_gap_fill_candidate is not None:
            selected_action = ACQUISITION_ACTION_GAP_FILL_SECONDARY
            plan_type = ACQUISITION_PLAN_CROSS_SOURCE_GAP_FILL
            request_windows = gap_windows
            request_start = gap_windows[0][0]
            request_end = gap_windows[-1][1]
            merge_with_existing = True
            secondary_source = str(secondary_gap_fill_candidate.get("source") or "")
            secondary_dataset_id = str(secondary_gap_fill_candidate.get("dataset_id") or "")
            parity_state = str(secondary_gap_fill_candidate.get("parity_state") or "unknown")
            parity_overlap_bars = int(secondary_gap_fill_candidate.get("parity_overlap_bars") or 0)
            parity_close_mae = (
                float(secondary_gap_fill_candidate["parity_close_mae"])
                if secondary_gap_fill_candidate.get("parity_close_mae") is not None
                else None
            )
            parity_close_mean_abs_bps = (
                float(secondary_gap_fill_candidate["parity_close_mean_abs_bps"])
                if secondary_gap_fill_candidate.get("parity_close_mean_abs_bps") is not None
                else None
            )
        else:
            request_windows = gap_windows
            if len(gap_windows) == 1:
                plan_type = ACQUISITION_PLAN_GAP_REPAIR_REFRESH
                request_start = gap_windows[0][0]
                request_end = gap_windows[0][1]
            else:
                plan_type = ACQUISITION_PLAN_MULTI_WINDOW_GAP_REPAIR_REFRESH
                request_start = gap_windows[0][0]
                request_end = gap_windows[-1][1]
            merge_with_existing = True
    elif (
        parquet_exists
        and ingested
        and quality_state == "gappy"
        and gap_windows
        and needs_incremental
        and not needs_backfill
    ):
        secondary_gap_fill_candidate = _find_secondary_gap_fill_candidate(
            catalog=resolved_catalog,
            store=resolved_store,
            dataset_id=dataset_id,
            symbol=str(symbol).strip().upper(),
            source=source,
            resolution=resolution,
            gap_windows=gap_windows,
        )
        if secondary_gap_fill_candidate is not None and right_edge_window is not None:
            selected_action = ACQUISITION_ACTION_DOWNLOAD
            plan_type = ACQUISITION_PLAN_CROSS_SOURCE_GAP_FILL_PLUS_INCREMENTAL_REFRESH
            request_windows = (right_edge_window,)
            secondary_request_windows = gap_windows
            request_start = right_edge_window[0]
            request_end = right_edge_window[1]
            merge_with_existing = True
            secondary_source = str(secondary_gap_fill_candidate.get("source") or "")
            secondary_dataset_id = str(secondary_gap_fill_candidate.get("dataset_id") or "")
            parity_state = str(secondary_gap_fill_candidate.get("parity_state") or "unknown")
            parity_overlap_bars = int(secondary_gap_fill_candidate.get("parity_overlap_bars") or 0)
            parity_close_mae = (
                float(secondary_gap_fill_candidate["parity_close_mae"])
                if secondary_gap_fill_candidate.get("parity_close_mae") is not None
                else None
            )
            parity_close_mean_abs_bps = (
                float(secondary_gap_fill_candidate["parity_close_mean_abs_bps"])
                if secondary_gap_fill_candidate.get("parity_close_mean_abs_bps") is not None
                else None
            )
        else:
            request_windows = compound_windows
            if len(compound_windows) == 1:
                request_start = compound_windows[0][0]
                request_end = compound_windows[0][1]
                plan_type = ACQUISITION_PLAN_INCREMENTAL_GAP_REPAIR_REFRESH if near_right_edge_gap else ACQUISITION_PLAN_GAP_REPAIR_REFRESH
            else:
                plan_type = ACQUISITION_PLAN_COMPOUND_MULTI_WINDOW_REFRESH
                request_start = compound_windows[0][0]
                request_end = compound_windows[-1][1]
            merge_with_existing = True
    elif (
        parquet_exists
        and ingested
        and quality_state == "gappy"
        and compound_windows
    ):
        request_windows = compound_windows
        if len(compound_windows) == 1:
            request_start = compound_windows[0][0]
            request_end = compound_windows[0][1]
            if needs_incremental and not needs_backfill and near_right_edge_gap:
                plan_type = ACQUISITION_PLAN_INCREMENTAL_GAP_REPAIR_REFRESH
            elif needs_backfill and not needs_incremental and desired_history_start is not None and near_left_edge_gap:
                plan_type = ACQUISITION_PLAN_BACKFILL_GAP_REPAIR_REFRESH
            else:
                plan_type = ACQUISITION_PLAN_GAP_REPAIR_REFRESH
        else:
            plan_type = ACQUISITION_PLAN_COMPOUND_MULTI_WINDOW_REFRESH
            request_start = compound_windows[0][0]
            request_end = compound_windows[-1][1]
        merge_with_existing = True
    elif parquet_exists and ingested and quality_state == "gappy":
        plan_type = ACQUISITION_PLAN_FULL_REBUILD
    elif parquet_exists and ingested and needs_incremental and needs_backfill:
        plan_type = ACQUISITION_PLAN_FULL_REBUILD
    elif parquet_exists and ingested and needs_incremental:
        plan_type = ACQUISITION_PLAN_INCREMENTAL_REFRESH
        request_start = coverage_end_ts.strftime("%Y-%m-%d") if pd.notna(coverage_end_ts) else None
        request_end = current.strftime("%Y-%m-%d")
        merge_with_existing = True
    elif parquet_exists and ingested and needs_backfill:
        plan_type = ACQUISITION_PLAN_BACKFILL
        request_start = desired_history_start.strftime("%Y-%m-%d") if desired_history_start is not None else None
        request_end = coverage_start_ts.strftime("%Y-%m-%d") if pd.notna(coverage_start_ts) else None
        merge_with_existing = True
    elif parquet_exists:
        plan_type = ACQUISITION_PLAN_FULL_REBUILD

    if not parquet_exists:
        reason = "No canonical dataset exists yet."
    elif plan_type == ACQUISITION_PLAN_CROSS_SOURCE_GAP_FILL:
        reason = (
            f"The local dataset has {suspicious_gap_count} suspicious gap(s)"
            f"{f' (max gap {max_suspicious_gap})' if max_suspicious_gap else ''}. "
            f"A matching secondary source ({secondary_source or 'alternate provider'}) can fill "
            f"{len(request_windows) or 1} repair window(s) with parity {parity_state}"
            f"{f', {parity_overlap_bars} overlap bars' if parity_overlap_bars else ''}"
            f"{f', mean abs {parity_close_mean_abs_bps:.2f} bps' if parity_close_mean_abs_bps is not None else ''}."
        )
    elif plan_type == ACQUISITION_PLAN_CROSS_SOURCE_GAP_FILL_PLUS_INCREMENTAL_REFRESH:
        reason = (
            f"The local dataset is stale and has {suspicious_gap_count} suspicious gap(s)"
            f"{f' (max gap {max_suspicious_gap})' if max_suspicious_gap else ''}. "
            f"A matching secondary source ({secondary_source or 'alternate provider'}) will fill "
            f"{len(secondary_request_windows) or 1} internal repair window(s), then the primary source will refresh "
            f"{request_start or 'the right edge'} -> {request_end or 'today'}."
        )
    elif plan_type == ACQUISITION_PLAN_GAP_REPAIR_REFRESH:
        reason = (
            f"The local dataset looks fresh but has {suspicious_gap_count} suspicious gap(s)"
            f"{f' (max gap {max_suspicious_gap})' if max_suspicious_gap else ''}. "
            f"The repair pass will refresh {request_start or 'the affected window'}"
            f" -> {request_end or 'the affected window'} and merge it into the canonical dataset."
        )
    elif plan_type == ACQUISITION_PLAN_MULTI_WINDOW_GAP_REPAIR_REFRESH:
        windows_text = ", ".join(f"{start}->{end}" for start, end in request_windows[:4])
        if len(request_windows) > 4:
            windows_text += f", +{len(request_windows) - 4} more"
        reason = (
            f"The local dataset looks fresh but has {suspicious_gap_count} suspicious non-contiguous gap(s)"
            f"{f' (max gap {max_suspicious_gap})' if max_suspicious_gap else ''}. "
            f"The repair pass will refresh {len(request_windows)} separate window(s): {windows_text}."
        )
    elif plan_type == ACQUISITION_PLAN_COMPOUND_MULTI_WINDOW_REFRESH:
        windows_text = ", ".join(f"{start}->{end}" for start, end in request_windows[:4])
        if len(request_windows) > 4:
            windows_text += f", +{len(request_windows) - 4} more"
        reason = (
            f"The local dataset needs a broader compound refresh: {suspicious_gap_count} suspicious gap(s)"
            f"{f' (max gap {max_suspicious_gap})' if max_suspicious_gap else ''}"
            f"{', stale right edge' if needs_incremental else ''}"
            f"{', incomplete left history' if needs_backfill else ''}. "
            f"The refresh will cover {len(request_windows)} request window(s): {windows_text}."
        )
    elif plan_type == ACQUISITION_PLAN_INCREMENTAL_GAP_REPAIR_REFRESH:
        reason = (
            f"The local dataset is stale and has {suspicious_gap_count} suspicious gap(s)"
            f"{f' (max gap {max_suspicious_gap})' if max_suspicious_gap else ''} near the right edge. "
            f"The refresh will extend from {request_start or 'the latest local coverage'} -> {request_end or 'today'} "
            "and merge the repaired range into the canonical dataset."
        )
    elif plan_type == ACQUISITION_PLAN_BACKFILL_GAP_REPAIR_REFRESH:
        reason = (
            f"The local dataset does not cover the full requested history window and has {suspicious_gap_count} suspicious gap(s)"
            f"{f' (max gap {max_suspicious_gap})' if max_suspicious_gap else ''} near the left edge. "
            f"The refresh will backfill {request_start or 'the requested history window'} -> {request_end or 'the left coverage edge'} "
            "and merge the repaired range into the canonical dataset."
        )
    elif quality_state == "gappy":
        reason = (
            f"The local dataset has {suspicious_gap_count} suspicious gap(s)"
            f"{f' (max gap {max_suspicious_gap})' if max_suspicious_gap else ''} and also needs a broader refresh."
        )
    elif needs_incremental and needs_backfill:
        reason = "The local dataset is stale and also does not cover the full requested history window."
    elif needs_incremental:
        reason = "The local dataset exists but is stale for this resolution."
    else:
        reason = "The local dataset does not cover the full requested history window."

    return AcquisitionPolicyDecision(
        dataset_id=dataset_id,
        source=source,
        symbol=str(symbol).strip().upper(),
        resolution=resolution,
        history_window=history_window,
        action=selected_action,
        reason=reason,
        freshness_state=freshness_state,
        csv_path=str(csv_path),
        parquet_path=str(parquet_path) if parquet_exists else None,
        coverage_start=coverage_start,
        coverage_end=coverage_end,
        bar_count=bar_count,
        ingested=ingested,
        quality_state=quality_state,
        suspicious_gap_count=suspicious_gap_count,
        max_suspicious_gap=max_suspicious_gap,
        plan_type=plan_type,
        request_start=request_start,
        request_end=request_end,
        merge_with_existing=merge_with_existing,
        request_windows=request_windows,
        secondary_source=secondary_source,
        secondary_dataset_id=secondary_dataset_id,
        parity_state=parity_state,
        parity_overlap_bars=parity_overlap_bars,
        parity_close_mae=parity_close_mae,
        parity_close_mean_abs_bps=parity_close_mean_abs_bps,
        secondary_request_windows=secondary_request_windows,
    )
