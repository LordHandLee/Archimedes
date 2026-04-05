# ChartSnapshot Schema

## Goal

Make chart rendering a pure read-only step.

The backtest engine should compute the run once, write a `ChartSnapshot` artifact keyed by `run_id`, and let any renderer (Matplotlib, Qt Quick, or a future C++ desktop app) open that artifact without rerunning the strategy.

This design is meant to fit the current engine:

- `run_id` already uniquely identifies a deterministic run.
- SQLite already stores run metadata and trades.
- DuckDB/Parquet already stores historical market data.

## Why This Exists

Decoupling chart open from backtest execution solves the first performance problem:

- no strategy rerun when opening a chart
- no recomputing indicators in the UI
- no repeated database resampling work
- identical chart output for the same `run_id`

It does **not** solve the second performance problem by itself:

- Matplotlib is still a CPU-side plotting library with expensive redraws during zoom and pan
- a TradingView-like chart needs a retained GPU-friendly renderer

So the plan is:

1. Compute once and save a snapshot.
2. Keep Matplotlib as a fallback reader.
3. Build a custom Qt Quick/C++ renderer that reads the same snapshot.

## Recommended Storage Layout

Store bulk chart data as files on disk, not inside SQLite blobs.

Recommended path:

```text
data/chart_snapshots/<run_id>/
  manifest.json
  price_bars.feather
  trades.feather
  equity.feather
  overlays.feather
  panes.feather
```

SQLite should only store metadata plus the snapshot path.

## File Format Choice

Recommended format: Apache Arrow IPC / Feather

Why:

- columnar and compact
- fast to write from pandas
- easy to read from Python now
- easy to read from C++ later via Arrow C++
- better than JSON for large intraday runs

Parquet is also fine for archival storage, but Feather is a better default for fast local open/read of one run artifact.

## Snapshot Contract

Each snapshot contains everything needed to render the chart at the run timeframe:

- price bars shown on the chart
- all trade markers
- all indicator series needed by the chart
- equity curve
- metadata needed for labels, formatting, and validation

The renderer should never need to know how to rerun the strategy.

This document defines the static snapshot artifact. Magellan process startup, background prelaunch, and live-session IPC are defined separately in [MAGELLAN_INTEGRATION.md](/home/ethan/quant_backtest_engine/MAGELLAN_INTEGRATION.md).

## Manifest Schema

`manifest.json`

```json
{
  "schema_version": 1,
  "run_id": "abc123...",
  "created_at_utc": "2026-03-25T14:00:00Z",
  "dataset_id": "AAPL_US_1M",
  "strategy": "InverseTurtleStrategy",
  "params": {
    "entry_len": 20,
    "exit_len": 10,
    "atr_len": 14,
    "target": 0.1
  },
  "timeframe": "5 minutes",
  "base_timeframe": "1 minutes",
  "timezone_display": "America/New_York",
  "start_utc": "2024-01-01T14:30:00Z",
  "end_utc": "2024-12-31T21:00:00Z",
  "starting_cash": 100000.0,
  "metrics": {
    "total_return": 0.24,
    "cagr": 0.18,
    "max_drawdown": -0.11,
    "sharpe": 1.42,
    "rolling_sharpe": 1.10
  },
  "files": {
    "price_bars": "price_bars.feather",
    "trades": "trades.feather",
    "equity": "equity.feather",
    "overlays": "overlays.feather",
    "panes": "panes.feather"
  },
  "counts": {
    "bars": 50231,
    "trades": 187,
    "overlay_series": 4,
    "pane_series": 1,
    "equity_points": 50231
  }
}
```

## price_bars.feather

One row per displayed candle/bar.

Required columns:

```text
ts_utc_ns        int64
open             float64
high             float64
low              float64
close            float64
volume           float64
bar_index        int32
```

Notes:

- `ts_utc_ns` is the canonical timestamp for storage and cross-language reads.
- `bar_index` is a stable dense integer index for fast screen-space mapping.
- Keep timestamps in UTC on disk. Convert to local display time in the renderer.

## trades.feather

One row per fill event.

Required columns:

```text
seq              int32
ts_utc_ns        int64
side             utf8      # "buy" or "sell"
qty              float64
price            float64
fee              float64
realized_pnl     float64
equity_after     float64
bar_index        int32      # nearest displayed bar index
event_type       utf8       # "entry", "exit", "flip", "adjust"
position_after   float64
```

Notes:

- `bar_index` lets the renderer place markers without recomputing timestamp alignment.
- `event_type` should be written by the engine, not reconstructed in the UI.

## equity.feather

One row per equity sample.

Required columns:

```text
ts_utc_ns        int64
equity           float64
drawdown         float64
bar_index        int32
```

Notes:

- `drawdown` is precomputed to avoid repeating chart math in every frontend.
- If equity is sampled once per displayed bar, `bar_index` lines up directly with `price_bars`.

## overlays.feather

Series drawn on the main price pane.

Recommended shape: wide table aligned to `price_bars`.

Required columns:

```text
ts_utc_ns        int64
bar_index        int32
SMA_Fast         float64?   # nullable
SMA_Slow         float64?   # nullable
Upper            float64?
Lower            float64?
Exit_Upper       float64?
Exit_Lower       float64?
```

Notes:

- One timestamp-aligned row per displayed bar.
- Nullable values are expected during warmup periods.
- Column names should be sanitized and stable because C++ code will read them by name.

## panes.feather

Series drawn in lower indicator panes.

Recommended shape: wide table aligned to `price_bars`.

Required columns:

```text
ts_utc_ns        int64
bar_index        int32
ATR              float64?
```

If you later add RSI, MACD, volume profile summaries, or custom diagnostics, they belong here or in additional pane files.

## Strategy Chart Metadata

Some rendering details should live in the manifest so the C++ viewer does not need strategy-specific hardcoding.

Recommended section:

```json
{
  "chart_layout": {
    "main_price_series": "candles",
    "overlay_order": ["SMA_Fast", "SMA_Slow", "Upper", "Lower", "Exit_Upper", "Exit_Lower"],
    "pane_order": ["ATR"],
    "pane_heights": {
      "price": 3.0,
      "ATR": 0.8,
      "equity": 1.0
    },
    "series_styles": {
      "SMA_Fast": { "color": "#4da3ff", "line_width": 1.0 },
      "SMA_Slow": { "color": "#ffd166", "line_width": 1.0 },
      "Upper": { "color": "#27d07d", "line_width": 1.0 },
      "Lower": { "color": "#ff6b6b", "line_width": 1.0 },
      "ATR": { "color": "#a28bff", "line_width": 1.0 },
      "equity": { "color": "#4da3ff", "line_width": 1.3 }
    }
  }
}
```

This lets the frontend stay generic.

## Python Object Model

Suggested engine-side model:

```python
@dataclass
class ChartSnapshot:
    run_id: str
    manifest: dict
    price_bars: pd.DataFrame
    trades: pd.DataFrame
    equity: pd.DataFrame
    overlays: pd.DataFrame
    panes: pd.DataFrame
```

## C++ Reader Model

Suggested C++ side model:

```cpp
struct PriceBar {
    int64_t tsUtcNs;
    double open;
    double high;
    double low;
    double close;
    double volume;
    int32_t barIndex;
};

struct TradeMarker {
    int32_t seq;
    int64_t tsUtcNs;
    std::string side;
    std::string eventType;
    double qty;
    double price;
    double fee;
    double realizedPnl;
    double equityAfter;
    double positionAfter;
    int32_t barIndex;
};
```

The viewer can then keep GPU vertex buffers keyed by `barIndex`.

## Integration With Current Engine

### 1. Extend `BacktestResult`

Current result object:

- `run_id`
- `equity_curve`
- `trades`
- `metrics`
- `cached`

Recommended additions:

```python
chart_snapshot_path: str | None = None
```

### 2. Add a snapshot builder

Recommended new module:

```text
backtest_engine/chart_snapshot.py
```

Responsibilities:

- build the snapshot tables
- write Feather files
- write `manifest.json`
- read snapshots back for any frontend

### 3. Move chart-series generation out of the UI

Today, indicator logic lives in `RunChartDialog._compute_indicators(...)`.

That logic should move into engine-owned code so indicators are computed once during the run artifact build, not recomputed in the Qt UI.

Better options:

- strategy method like `build_chart_series(data, params) -> dict[str, pd.Series]`
- snapshot builder with per-strategy adapters

### 4. Extend `ResultCatalog`

Do not store bulk chart arrays in SQLite.

Instead add either:

- a `chart_snapshot_path` column to `runs`, or
- a separate `chart_snapshots` table keyed by `run_id`

Suggested table:

```text
chart_snapshots
  run_id TEXT PRIMARY KEY
  schema_version INTEGER NOT NULL
  snapshot_path TEXT NOT NULL
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
```

### 5. Chart open flow

New flow:

1. User double-clicks a run.
2. UI loads snapshot metadata from SQLite.
3. UI opens the Feather files for that `run_id`.
4. Renderer paints immediately.

No strategy execution. No indicator recomputation. No resample work in the UI.

## Optional Level Of Detail Files

For very large intraday histories, add pre-aggregated display files later:

```text
lod_1.feather    # full resolution
lod_4.feather    # 4 bars merged
lod_16.feather   # 16 bars merged
lod_64.feather   # 64 bars merged
```

That is optional for v1.

For the first version, a good custom renderer can already feel much faster than Matplotlib by:

- keeping geometry on the GPU
- updating only visible ranges
- avoiding full CPU redraws

## Qt Quick Rendering Guidance

For the future C++ chart app, prefer:

- `QQuickItem` + `updatePaintNode()` for custom scene graph content, or
- `QSGRenderNode` if you need lower-level custom draw recording

Avoid for the main chart surface:

- `QQuickPaintedItem`
- embedding Matplotlib into Qt
- CPU-side redraw of every candle on every scroll tick

Qt Quick is fast when you let it keep retained geometry and batch draw calls. It becomes much less attractive when used as a wrapper around a software-style painter.

## First Implementation Scope

Recommended v1:

- write `price_bars.feather`
- write `trades.feather`
- write `equity.feather`
- write `overlays.feather`
- write `panes.feather`
- store snapshot path in SQLite
- update the Qt dashboard to read snapshot artifacts instead of computing chart data in the dialog

That gives you the "compute once, render many" contract immediately, while leaving room for the C++ renderer next.
