"""Lightweight, cache-aware backtesting utilities."""

from .engine import BacktestConfig, BacktestEngine, BacktestResult
from .catalog import ResultCatalog
from .data_loader import load_csv_prices, resample_bars
from .duckdb_store import DuckDBStore
from .grid_search import GridSearch, GridSpec, build_horizons
from .reporting import plot_param_heatmap
from .sample_strategies import SMACrossStrategy, InverseTurtleStrategy

__all__ = [
    "BacktestConfig",
    "BacktestEngine",
    "BacktestResult",
    "ResultCatalog",
    "DuckDBStore",
    "GridSearch",
    "GridSpec",
    "build_horizons",
    "load_csv_prices",
    "resample_bars",
    "SMACrossStrategy",
    "InverseTurtleStrategy",
    "plot_param_heatmap",
]
