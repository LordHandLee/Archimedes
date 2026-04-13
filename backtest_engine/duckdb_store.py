from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import duckdb
import numpy as np
import pandas as pd

from .data_loader import REQUIRED_COLUMNS


class DuckDBStore:
    """
    Simple DuckDB-backed parquet store for OHLCV history.

    - Stores each dataset as a parquet file on disk.
    - Maintains a DuckDB database to query and resample efficiently.
    """

    def __init__(self, db_path: str | Path = "data/history.duckdb", data_dir: str | Path = "data/parquet") -> None:
        self.db_path = Path(db_path)
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[duckdb.DuckDBPyConnection] = None

    # -- basic lifecycle -----------------------------------------------------
    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            self._conn = duckdb.connect(self.db_path)
        return self._conn

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # -- data IO -------------------------------------------------------------
    def dataset_path(self, dataset_id: str) -> Path:
        safe = dataset_id.replace("/", "_").replace(" ", "_")
        return self.data_dir / f"{safe}.parquet"

    def write_parquet(self, dataset_id: str, df: pd.DataFrame) -> Path:
        missing = set(REQUIRED_COLUMNS) - set(df.reset_index().columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")
        path = self.dataset_path(dataset_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Use DuckDB to write parquet to avoid optional pandas engines (pyarrow/fastparquet) dependency.
        rel = duckdb.from_df(df.reset_index())
        rel.write_parquet(str(path))
        return path

    def load(
        self,
        dataset_id: str,
        columns: Iterable[str] = ("timestamp", "open", "high", "low", "close", "volume"),
    ) -> pd.DataFrame:
        """
        Load a dataset by id using DuckDB; returns a pandas DataFrame indexed by timestamp.
        """
        path = self.dataset_path(dataset_id)
        if not path.exists():
            raise FileNotFoundError(f"Parquet not found for dataset_id={dataset_id}")
        query = f"SELECT {', '.join(columns)} FROM parquet_scan('{path}') ORDER BY timestamp"
        df = self.conn.execute(query).df()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
        return df

    def load_range(
        self,
        dataset_id: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        columns: Iterable[str] = ("timestamp", "open", "high", "low", "close", "volume"),
    ) -> pd.DataFrame:
        """
        Load a dataset by id within a timestamp range (inclusive) using DuckDB.
        """
        path = self.dataset_path(dataset_id)
        if not path.exists():
            raise FileNotFoundError(f"Parquet not found for dataset_id={dataset_id}")
        query = f"""
        SELECT {', '.join(columns)}
        FROM parquet_scan('{path}')
        WHERE timestamp >= ? AND timestamp <= ?
        ORDER BY timestamp
        """
        df = self.conn.execute(query, [str(start), str(end)]).df()
        if df.empty:
            return df
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df.set_index("timestamp")

    def resample(self, dataset_id: str, timeframe: str) -> pd.DataFrame:
        """
        Resample using DuckDB's time_bucket for performance, returning pandas DataFrame.
        """
        path = self.dataset_path(dataset_id)
        if not path.exists():
            raise FileNotFoundError(f"Parquet not found for dataset_id={dataset_id}")

        sql = f"""
        WITH src AS (
            SELECT *
            FROM parquet_scan('{path}')
        )
        SELECT
            time_bucket(INTERVAL '{timeframe}', src.timestamp) AS timestamp,
            first(open) AS open,
            max(high) AS high,
            min(low) AS low,
            last(close) AS close,
            sum(volume) AS volume
        FROM src
        GROUP BY 1
        HAVING open IS NOT NULL AND close IS NOT NULL
        ORDER BY 1
        """
        df = self.conn.execute(sql).df()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df.set_index("timestamp")

    def describe_dataset(self, dataset_id: str) -> dict:
        path = self.dataset_path(dataset_id)
        if not path.exists():
            raise FileNotFoundError(f"Parquet not found for dataset_id={dataset_id}")
        df = self.load(dataset_id)
        start_ts = df.index.min() if not df.empty else None
        end_ts = df.index.max() if not df.empty else None
        return {
            "dataset_id": dataset_id,
            "parquet_path": str(path),
            "coverage_start": start_ts.isoformat() if start_ts is not None else None,
            "coverage_end": end_ts.isoformat() if end_ts is not None else None,
            "bar_count": int(len(df)),
        }

    @staticmethod
    def _resolution_to_timedelta(resolution: str) -> pd.Timedelta | None:
        normalized = str(resolution or "").strip().lower()
        if not normalized:
            return None
        try:
            if normalized.endswith("m"):
                return pd.Timedelta(minutes=int(normalized[:-1] or "1"))
            if normalized.endswith("h"):
                return pd.Timedelta(hours=int(normalized[:-1] or "1"))
            if normalized.endswith("d"):
                return pd.Timedelta(days=int(normalized[:-1] or "1"))
            if normalized.endswith("w"):
                return pd.Timedelta(weeks=int(normalized[:-1] or "1"))
        except Exception:
            return None
        return None

    @staticmethod
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

    @staticmethod
    def _intraday_core_session_mask(index: pd.DatetimeIndex) -> pd.Series:
        local = index.tz_convert("America/New_York")
        minutes = (local.hour * 60) + local.minute
        return pd.Series((minutes >= 570) & (minutes <= 960), index=range(len(index)))

    @staticmethod
    def _intraday_local_day_mask(index: pd.DatetimeIndex) -> pd.Series:
        local = index.tz_convert("America/New_York")
        return pd.Series(local[1:].normalize() == local[:-1].normalize())

    @staticmethod
    def _intraday_analysis_index(index: pd.DatetimeIndex) -> tuple[pd.DatetimeIndex, str]:
        if index.empty:
            return index, "unknown"
        core_mask = DuckDBStore._intraday_core_session_mask(index)
        core_ratio = float(core_mask.mean()) if len(core_mask) else 0.0
        if core_ratio >= 0.9:
            return pd.DatetimeIndex(index[core_mask.to_numpy()]), "nyse_core"
        return index, "local_intraday"

    def inspect_dataset_quality(self, dataset_id: str, resolution: str) -> dict:
        path = self.dataset_path(dataset_id)
        if not path.exists():
            raise FileNotFoundError(f"Parquet not found for dataset_id={dataset_id}")
        df = self.load(dataset_id, columns=("timestamp",))
        timestamps = pd.DatetimeIndex(df.index).sort_values().unique()
        expected_interval = self._resolution_to_timedelta(resolution)
        suspicious_gap_count = 0
        max_suspicious_gap: pd.Timedelta | None = None
        quality_state = "unknown"
        suspicious_gap_ranges: list[dict[str, str]] = []
        repair_request_start: str | None = None
        repair_request_end: str | None = None
        session_profile: str | None = None

        if expected_interval is not None and len(timestamps) >= 2:
            if expected_interval < pd.Timedelta(days=1):
                analysis_index, session_profile = self._intraday_analysis_index(timestamps)
                if len(analysis_index) >= 2:
                    diffs = analysis_index[1:] - analysis_index[:-1]
                    diff_series = pd.Series(diffs)
                    gap_after_series = pd.Series(pd.Index(analysis_index[:-1]))
                    gap_before_series = pd.Series(pd.Index(analysis_index[1:]))
                    same_day_mask = self._intraday_local_day_mask(analysis_index)
                    suspicious_mask = same_day_mask & (diff_series > (expected_interval * 3))
                else:
                    diff_series = pd.Series(dtype="timedelta64[ns]")
                    gap_after_series = pd.Series(dtype="datetime64[ns, UTC]")
                    gap_before_series = pd.Series(dtype="datetime64[ns, UTC]")
                    suspicious_mask = pd.Series(dtype=bool)
            else:
                diffs = timestamps[1:] - timestamps[:-1]
                diff_series = pd.Series(diffs)
                gap_after_series = pd.Series(pd.Index(timestamps[:-1]))
                gap_before_series = pd.Series(pd.Index(timestamps[1:]))
                suspicious_mask = diff_series > (expected_interval * 3)
            suspicious_diffs = diff_series[suspicious_mask]
            suspicious_gap_count = int(len(suspicious_diffs))
            if suspicious_gap_count:
                max_suspicious_gap = suspicious_diffs.max()
                suspicious_rows = pd.DataFrame(
                    {
                        "gap_after": gap_after_series[suspicious_mask].reset_index(drop=True),
                        "gap_before": gap_before_series[suspicious_mask].reset_index(drop=True),
                        "gap_duration": suspicious_diffs.reset_index(drop=True),
                    }
                )
                for _, gap_row in suspicious_rows.iterrows():
                    gap_after = pd.Timestamp(gap_row["gap_after"])
                    gap_before = pd.Timestamp(gap_row["gap_before"])
                    suspicious_gap_ranges.append(
                        {
                            "gap_after": gap_after.isoformat(),
                            "gap_before": gap_before.isoformat(),
                            "gap_duration": str(gap_row["gap_duration"]),
                            "request_start": gap_after.strftime("%Y-%m-%d"),
                            "request_end": gap_before.strftime("%Y-%m-%d"),
                        }
                    )
                if expected_interval < pd.Timedelta(days=1):
                    total_points = max(int(len(analysis_index)), 1)
                    gap_ratio = float(suspicious_gap_count) / float(total_points)
                    minor_intraday_limit = max(pd.Timedelta(minutes=60), expected_interval * 12)
                    if max_suspicious_gap is not None and max_suspicious_gap <= minor_intraday_limit and gap_ratio <= 0.005:
                        quality_state = "sparse_intraday"
                    else:
                        quality_state = "gappy"
                else:
                    quality_state = "gappy"
                if quality_state == "gappy":
                    repair_request_start = suspicious_gap_ranges[0]["request_start"]
                    repair_request_end = suspicious_gap_ranges[-1]["request_end"]
            else:
                quality_state = "gap_free"
        elif expected_interval is not None:
            quality_state = "gap_free"

        return {
            "dataset_id": dataset_id,
            "resolution": resolution,
            "quality_state": quality_state,
            "expected_interval": str(expected_interval) if expected_interval is not None else None,
            "suspicious_gap_count": suspicious_gap_count,
            "max_suspicious_gap": str(max_suspicious_gap) if max_suspicious_gap is not None else None,
            "suspicious_gap_ranges": suspicious_gap_ranges,
            "repair_request_start": repair_request_start,
            "repair_request_end": repair_request_end,
            "session_profile": session_profile,
        }

    def compare_dataset_parity(
        self,
        primary_dataset_id: str,
        secondary_dataset_id: str,
        *,
        start: str | None = None,
        end: str | None = None,
    ) -> dict:
        primary = self.load(primary_dataset_id, columns=("timestamp", "close"))
        secondary = self.load(secondary_dataset_id, columns=("timestamp", "close"))
        if start:
            start_ts = self._window_bound_timestamp(start, is_end=False)
            if start_ts is not None:
                primary = primary.loc[primary.index >= start_ts]
                secondary = secondary.loc[secondary.index >= start_ts]
        if end:
            end_ts = self._window_bound_timestamp(end, is_end=True)
            if end_ts is not None:
                primary = primary.loc[primary.index <= end_ts]
                secondary = secondary.loc[secondary.index <= end_ts]
        joined = primary.rename(columns={"close": "primary_close"}).join(
            secondary.rename(columns={"close": "secondary_close"}),
            how="inner",
        )
        if joined.empty:
            return {
                "primary_dataset_id": primary_dataset_id,
                "secondary_dataset_id": secondary_dataset_id,
                "parity_state": "no_overlap",
                "overlap_bar_count": 0,
                "close_mae": None,
                "close_mean_abs_bps": None,
                "close_max_abs_diff": None,
            }
        abs_diff = (joined["primary_close"] - joined["secondary_close"]).abs()
        denominator = joined["primary_close"].abs().replace(0.0, np.nan)
        abs_bps = (abs_diff / denominator) * 10000.0
        abs_bps = abs_bps.replace([np.inf, -np.inf], np.nan)
        close_mae = float(abs_diff.mean())
        close_mean_abs_bps = float(abs_bps.dropna().mean()) if abs_bps.dropna().size else 0.0
        close_max_abs_diff = float(abs_diff.max())
        parity_state = "matching" if close_mean_abs_bps <= 25.0 else "divergent"
        return {
            "primary_dataset_id": primary_dataset_id,
            "secondary_dataset_id": secondary_dataset_id,
            "parity_state": parity_state,
            "overlap_bar_count": int(len(joined)),
            "close_mae": close_mae,
            "close_mean_abs_bps": close_mean_abs_bps,
            "close_max_abs_diff": close_max_abs_diff,
        }
