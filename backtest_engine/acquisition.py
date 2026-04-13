from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .catalog import ResultCatalog
from .data_loader import load_csv_prices
from .duckdb_store import DuckDBStore


@dataclass
class IngestedDatasetArtifact:
    dataset_id: str
    csv_path: str
    parquet_path: str
    start: str
    end: str
    bar_count: int
    quality_snapshot: dict | None = None


DOWNLOAD_ARTIFACT_STATE_VERSION = 1


def compute_and_store_quality_snapshot(
    dataset_id: str,
    *,
    resolution: str | None,
    catalog: ResultCatalog | None = None,
    store: DuckDBStore | None = None,
) -> dict | None:
    resolved_resolution = str(resolution or "").strip()
    if not resolved_resolution:
        return None
    duck = store or DuckDBStore()
    try:
        quality_snapshot = duck.inspect_dataset_quality(str(dataset_id), resolved_resolution)
    except Exception:
        return None
    quality_snapshot["quality_analyzed_at"] = pd.Timestamp.now("UTC").isoformat()
    if catalog is not None:
        catalog.update_acquisition_dataset_quality(str(dataset_id), quality_snapshot)
    return quality_snapshot


def backfill_missing_quality_snapshots(
    *,
    catalog: ResultCatalog | None = None,
    store: DuckDBStore | None = None,
    dataset_ids: list[str] | None = None,
    limit: int | None = None,
) -> int:
    resolved_catalog = catalog or ResultCatalog()
    duck = store or DuckDBStore()
    normalized_dataset_ids = {str(item).strip() for item in list(dataset_ids or []) if str(item).strip()}
    updated = 0
    for record in resolved_catalog.load_acquisition_datasets():
        if normalized_dataset_ids and str(record.dataset_id) not in normalized_dataset_ids:
            continue
        if record.quality_analyzed_at:
            continue
        if not record.ingested or not str(record.resolution or "").strip():
            continue
        if not duck.dataset_path(str(record.dataset_id)).exists():
            continue
        quality_snapshot = compute_and_store_quality_snapshot(
            str(record.dataset_id),
            resolution=str(record.resolution or ""),
            catalog=resolved_catalog,
            store=duck,
        )
        if quality_snapshot is not None:
            updated += 1
        if limit is not None and updated >= int(limit):
            break
    return updated


def _window_bound_timestamp(value: str | None, *, is_end: bool) -> pd.Timestamp | None:
    if not value:
        return None
    stamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(stamp):
        return None
    text = str(value).strip()
    if text and "T" not in text and " " not in text:
        if is_end:
            return stamp + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        return stamp
    return stamp


def infer_dataset_id_from_csv_path(csv_path: Path | str) -> str:
    return Path(csv_path).stem


def download_artifact_state_path(csv_path: Path | str) -> Path:
    path = Path(csv_path)
    return path.with_name(f"{path.name}.download_state.json")


def load_download_artifact_state(csv_path: Path | str) -> dict:
    state_path = download_artifact_state_path(csv_path)
    if not state_path.exists():
        return {}
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    return dict(payload)


def write_download_artifact_state(csv_path: Path | str, **state: object) -> Path:
    path = Path(csv_path)
    state_path = download_artifact_state_path(path)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": DOWNLOAD_ARTIFACT_STATE_VERSION,
        "csv_path": str(path),
        **dict(state or {}),
    }
    state_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return state_path


def clear_download_artifact_state(csv_path: Path | str) -> None:
    state_path = download_artifact_state_path(csv_path)
    try:
        state_path.unlink()
    except FileNotFoundError:
        return
    except Exception:
        return


def build_download_dataset_id(
    symbol: str,
    *,
    source: str = "massive",
    history_window: str = "2y",
    resolution: str = "1m",
) -> str:
    safe_symbol = str(symbol).strip().upper()
    safe_source = str(source).strip().lower() or "source"
    safe_history = str(history_window).strip().lower() or "history"
    safe_resolution = str(resolution).strip().lower() or "resolution"
    return f"{safe_symbol}_{safe_source}_{safe_history}_{safe_resolution}"


def build_download_csv_path(
    symbol: str,
    *,
    source: str = "massive",
    history_window: str = "2y",
    resolution: str = "1m",
    data_dir: Path | str = "data",
) -> Path:
    dataset_id = build_download_dataset_id(
        symbol,
        source=source,
        history_window=history_window,
        resolution=resolution,
    )
    return Path(data_dir) / f"{dataset_id}.csv"


def ingest_csv_to_store(
    csv_path: Path | str,
    dataset_id: str | None = None,
    store: DuckDBStore | None = None,
    *,
    merge_existing: bool = False,
    resolution: str | None = None,
) -> IngestedDatasetArtifact:
    path = Path(csv_path)
    resolved_dataset_id = str(dataset_id or infer_dataset_id_from_csv_path(path)).strip()
    if not resolved_dataset_id:
        raise ValueError("Dataset id is required for ingestion.")

    loaded = load_csv_prices(path)
    duck = store or DuckDBStore()
    final_frame = loaded.data
    if merge_existing:
        existing_path = duck.dataset_path(resolved_dataset_id)
        if existing_path.exists():
            existing = duck.load(resolved_dataset_id)
            final_frame = pd.concat([existing, loaded.data], axis=0)
            final_frame = final_frame[~final_frame.index.duplicated(keep="last")]
            final_frame = final_frame.sort_index()
    parquet_path = duck.write_parquet(resolved_dataset_id, final_frame.reset_index())
    quality_snapshot = compute_and_store_quality_snapshot(
        resolved_dataset_id,
        resolution=resolution,
        store=duck,
    )
    return IngestedDatasetArtifact(
        dataset_id=resolved_dataset_id,
        csv_path=str(path),
        parquet_path=str(parquet_path),
        start=final_frame.index.min().isoformat(),
        end=final_frame.index.max().isoformat(),
        bar_count=int(len(final_frame)),
        quality_snapshot=quality_snapshot,
    )


def gap_fill_dataset_from_secondary(
    target_dataset_id: str,
    secondary_dataset_id: str,
    *,
    store: DuckDBStore | None = None,
    start: str | None = None,
    end: str | None = None,
    resolution: str | None = None,
) -> IngestedDatasetArtifact:
    duck = store or DuckDBStore()
    target = duck.load(target_dataset_id)
    if start or end:
        start_ts = _window_bound_timestamp(start, is_end=False)
        end_ts = _window_bound_timestamp(end, is_end=True)
        if start_ts is not None and end_ts is not None:
            secondary = duck.load_range(secondary_dataset_id, start_ts, end_ts)
        else:
            secondary = duck.load(secondary_dataset_id)
            if start_ts is not None:
                secondary = secondary.loc[secondary.index >= start_ts]
            if end_ts is not None:
                secondary = secondary.loc[secondary.index <= end_ts]
    else:
        secondary = duck.load(secondary_dataset_id)
    if secondary.empty:
        raise ValueError("Secondary dataset did not provide any rows for the requested gap-fill window.")
    missing_only = secondary.loc[~secondary.index.isin(target.index)]
    final_frame = pd.concat([target, missing_only], axis=0)
    final_frame = final_frame[~final_frame.index.duplicated(keep="first")]
    final_frame = final_frame.sort_index()
    parquet_path = duck.write_parquet(target_dataset_id, final_frame.reset_index())
    quality_snapshot = compute_and_store_quality_snapshot(
        target_dataset_id,
        resolution=resolution,
        store=duck,
    )
    return IngestedDatasetArtifact(
        dataset_id=str(target_dataset_id),
        csv_path=str(duck.dataset_path(secondary_dataset_id)),
        parquet_path=str(parquet_path),
        start=final_frame.index.min().isoformat(),
        end=final_frame.index.max().isoformat(),
        bar_count=int(len(final_frame)),
        quality_snapshot=quality_snapshot,
    )
