# Quant Backtest Engine – System Architecture

## Core Principles
- Deterministic, reproducible runs (hash-based run_id); no double work via SQLite cache.
- Separation of concerns: data lake (DuckDB/parquet), metadata/results (SQLite), compute (engine/grid).
- Pluggable data sources (APIs, CSV), pluggable strategies (Strategy subclass), consistent OHLCV schema.
- Transparent assumptions: fill-on-close vs open, recalc after fills, slippage/fees, shorting toggle.

## Components
### Data Pipeline (ingest + clean)
- **Fetchers**: adapters for yfinance/FMP/Alpaca/Polygon/custom CSVs -> raw DataFrame.
- **Cleaner**: normalize to schema `[timestamp(UTC), open, high, low, close, volume]`.
- **Persistence**: store canonical bars per symbol/timeframe to Parquet; register via DuckDBStore.
- **Functions**: `load_csv_prices`, future API fetchers, `DuckDBStore.write_parquet` for storage.

### Historical Store (DuckDB + Parquet)
- Parquet files per dataset_id under `data/parquet/`.
- DuckDB catalog (`data/history.duckdb`) used for efficient queries/resampling via `DuckDBStore.resample`.
- Benefits: columnar compression, vectorized resampling, SQL-friendly for ETL.

### Metadata / Results (SQLite)
- `ResultCatalog` maintains `runs` table: run_id, strategy, params, timeframe, start/end, dataset_id, metrics (JSON).
- Enables dedupe, cache, and audit trail of backtests and grid searches.

### Backtest Engine
- Inputs: DataFrame of bars (from DuckDBStore), Strategy class + params, BacktestConfig.
- Config knobs: timeframe, starting_cash, fee_rate, slippage, fill_on_close, recalc_on_fill, allow_short, horizon filters, cache toggle.
- Broker: single-asset, target-percent sizing, cash/position tracking, slippage/fees, order clipping to cash/position.
- Broker extensions: optional partial fills per bar, side-specific fee/slippage schedules, and borrow cost on shorts.
- Strategy lifecycle: `initialize(data)`, `on_bar(ts, bar, broker)`, optional `on_after_fill` (used when recalc_on_fill).
- Metrics: total return, CAGR, max drawdown, Sharpe; extendable.

### Grid Search / Orchestration
- `GridSearch` module (OOP, optional) accepts `GridSpec` (params, timeframes, horizons, metric, heatmap axes, description) and orchestrates sweeps.
- Reuses cached runs by run_id hash; plugs into any data loader callable (e.g., DuckDB resamples).
- Parallelization: embarrassingly parallel at run-level; future: job queue or multiprocessing.
- Reporting: collects metrics into DataFrame; `plot_param_heatmap` visualizes top performers across params/timeframes and records heatmap metadata in SQLite.

### Result Catalog & Heatmaps
- Persist each run; grid aggregations query SQLite for runs (or in-memory DataFrame) to produce heatmaps of metric vs param/timeframe.
- Heatmaps saved to disk and registered in SQLite (`heatmaps` table) with metadata (params, description, file path).

## Typical Flow
1) Ingest: fetch API or read CSV -> normalize -> `DuckDBStore.write_parquet(dataset_id, df)`.
2) Select data: `bars = duck.resample(dataset_id, '5 minutes')` (or use raw 1m).
3) Configure: `BacktestConfig(...)`, choose Strategy subclass + params.
4) Run: `engine.run(params)`; results cached in SQLite by run_id.
5) Grid: loop params/timeframes/horizons; gather metrics; visualize with heatmap; pick best candidates.

## Extensibility Hooks
- New sources: implement fetcher returning normalized DataFrame; persist via DuckDBStore.
- New strategies: subclass `Strategy`, implement `initialize`, `on_bar`, optional `on_after_fill`.
- Risk/fees: extend Broker to support borrow costs, borrow limits, partial fills.
- Multi-asset: evolve Broker + Engine to track portfolios and routing.

## Alignment with TradingView Behaviors
- `fill_on_close=False` is the default for next-bar open fills; `recalc_on_fill=True` mirrors “recalculate after order filled”.
- Set `fill_on_close=True` only if you explicitly want bar-close fills (higher lookahead risk).

## Testing Considerations
- Unit: broker accounting, run_id hashing stability, metrics correctness.
- Integration: backtest deterministic on fixed data; caching prevents duplicate work.
- Data: spot-check DuckDB resampling vs pandas resample for parity.
