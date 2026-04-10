from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

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
    return IngestedDatasetArtifact(
        dataset_id=resolved_dataset_id,
        csv_path=str(path),
        parquet_path=str(parquet_path),
        start=final_frame.index.min().isoformat(),
        end=final_frame.index.max().isoformat(),
        bar_count=int(len(final_frame)),
    )


def gap_fill_dataset_from_secondary(
    target_dataset_id: str,
    secondary_dataset_id: str,
    *,
    store: DuckDBStore | None = None,
    start: str | None = None,
    end: str | None = None,
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
    return IngestedDatasetArtifact(
        dataset_id=str(target_dataset_id),
        csv_path=str(duck.dataset_path(secondary_dataset_id)),
        parquet_path=str(parquet_path),
        start=final_frame.index.min().isoformat(),
        end=final_frame.index.max().isoformat(),
        bar_count=int(len(final_frame)),
    )
