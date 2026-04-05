from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import duckdb
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
