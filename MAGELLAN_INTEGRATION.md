# Magellan Integration Design

## Purpose

This document defines how the quant backtest engine should integrate with Magellan, the standalone C++ charting project.

It covers:

- static snapshot viewing for completed research runs
- live market sessions
- paper-engine sessions
- live-deployment sessions
- viewer startup and background prelaunch behavior
- the integration boundary between this Python project and Magellan

## Project Locations

Current Magellan project root:

- [Magellan](/home/ethan/Magellan)

Current charting engine root:

- [charting_engine](/home/ethan/Magellan/charting_engine)

Useful entry points:

- [README.md](/home/ethan/Magellan/charting_engine/README.md)
- [main.cpp](/home/ethan/Magellan/charting_engine/src/app/main.cpp)
- [viewer_command_protocol.cpp](/home/ethan/Magellan/charting_engine/src/platform/viewer_command_protocol.cpp)
- [chart_snapshot.h](/home/ethan/Magellan/charting_engine/src/domain/chart_snapshot.h)

## Integration Philosophy

This project should own:

- backtest execution
- research workflows
- walk-forward and Monte Carlo studies
- paper engine state
- live deployment state
- market data adapters
- broker adapters
- `ChartSnapshot` production
- live chart update generation

Magellan should own:

- chart rendering
- chart windows
- zoom, pan, crosshair, and visual interaction
- loading snapshot artifacts
- mutating chart state from live updates

The clean boundary is:

- this project produces chart data
- Magellan renders chart data

## Two Integration Modes

Magellan should be integrated in two primary modes.

### 1. Artifact Mode

Use this for completed or saved runs.

Examples:

- backtests
- parameter-study candidate reviews
- walk-forward study folds
- stitched walk-forward OOS reviews
- Monte Carlo representative paths
- completed paper sessions
- completed live sessions

Flow:

1. This project writes a snapshot folder.
2. The UI requests Magellan to open that snapshot.
3. Magellan opens a chart window and loads the snapshot asynchronously.

### 2. Live Session Mode

Use this for streaming or continuously updated charts.

Examples:

- ticker market view
- paper engine monitoring
- live deployment monitoring

Flow:

1. This project opens a Magellan live session by `session_id`.
2. The session may be seeded from a historical snapshot.
3. This project streams incremental updates to Magellan over local IPC.
4. Magellan mutates the chart state in memory and repaints in place.

## Existing Magellan Capabilities

Magellan already supports the core integration model we need.

Confirmed capabilities:

- long-lived single-instance viewer process
- local IPC for repeated open requests
- buffered length-prefixed IPC for large live-update payloads, with legacy newline fallback
- snapshot viewing
- seeded live sessions
- incremental live updates for bars, overlays, panes, equity, and trade markers
- native-parent embedded chart windows for dashboard tabs
- live-session series replacement for interactive indicator selection
- live-session bar replacement and seed reload for ticker/timeframe/lookback changes
- close/release commands for removing hosted chart sessions
- embedded resize IPC for hosted chart surfaces
- date/time x-axis labels on the lowest visible pane, including price-only market charts
- robust timestamp display for epoch seconds, milliseconds, microseconds, or nanoseconds
- horizontal chart pan plus vertical price-range recentering by drag
- lower-pane crosshair value tags for indicator and equity panes
- equity-pane scaling based on plotted equity/benchmark series, with `drawdown` excluded from the shared equity y-axis

These are described in [README.md](/home/ethan/Magellan/charting_engine/README.md#L7) and implemented from [main.cpp](/home/ethan/Magellan/charting_engine/src/app/main.cpp#L61).

## Snapshot Producer Requirements From Viewer Testing

Recent viewer testing against real snapshots showed two producer-side requirements.

First, every `ts_utc_ns` column should be written as UTC Unix epoch nanoseconds. Some generated snapshots currently contain microsecond values in `price_bars.feather`, `overlays.feather`, `panes.feather`, and `equity.feather`, while `trades.feather` contains nanoseconds. Magellan now infers units defensively for display, but this project should normalize all snapshot timestamp columns to nanoseconds.

Second, the equity pane can only draw benchmark or buy-and-hold lines if this project writes those series. Current snapshots provide `equity` and `drawdown`. Magellan draws `equity` and skips `drawdown` on the shared equity y-axis so the account-equity curve scales correctly. To display a green buy-and-hold line, write an aligned `buy_hold_equity` or `benchmark_equity` column to `equity.feather` and style it in `manifest.json`.

## Viewer Process Model

Magellan is already designed as a long-lived process.

Important startup behavior:

- the application uses a single local server name: `MagellanChartViewer`
- if another instance is already running, new open requests are forwarded to it
- the app sets `QuitOnLastWindowClosed(false)`
- the root QML object is a `QtObject`, not a permanently visible main window
- chart windows are created only when a snapshot or live session is opened

This behavior comes from:

- [main.cpp](/home/ethan/Magellan/charting_engine/src/app/main.cpp#L15)
- [main.cpp](/home/ethan/Magellan/charting_engine/src/app/main.cpp#L24)
- [Main.qml](/home/ethan/Magellan/charting_engine/src/ui/qml/Main.qml#L1)
- [ChartWindow.qml](/home/ethan/Magellan/charting_engine/src/ui/qml/ChartWindow.qml#L23)

### Practical Consequence

Yes, Magellan can be launched in the background when this project's UI starts.

Because no chart window is created until a request arrives, the viewer can sit resident with no visible chart and no first-open lag from process startup or Qt initialization.

That is the recommended integration model.

## Recommended Startup Strategy

When this project's UI launches:

1. Check whether the Magellan IPC server is already available.
2. If not, start `magellan_chart_viewer` in the background with no snapshot argument.
3. Let Magellan remain resident for the life of the UI session.
4. When the user opens a chart, send an IPC request instead of launching a fresh process.

Benefits:

- instant or near-instant chart opens after UI startup
- avoids repeated Qt startup overhead
- fits Magellan's intended architecture
- works for snapshot mode and live-session mode

## Process Ownership and Shutdown

For this project, Magellan should be owned by the UI process that launched it.

Recommended rule:

- if this project's UI started Magellan, this project's UI should track the Magellan PID and terminate it on clean shutdown
- if Magellan was already running before the UI launched, the UI should not assume ownership of that existing process

This gives us:

- instant chart opens during the UI session
- no stray background Magellan process after the UI closes
- predictable lifecycle behavior

### Ownership Model

When the UI starts:

1. Check whether the `MagellanChartViewer` IPC server is already available.
2. If it is already available, treat Magellan as externally owned and do not claim shutdown ownership.
3. If it is not available, launch Magellan in the background and record:
   - spawned PID
   - launch timestamp
   - ownership flag

When the UI exits cleanly:

1. If the ownership flag is set, terminate the tracked Magellan process.
2. Wait briefly for clean exit.
3. If needed, escalate to a stronger kill path.

### Important Constraint

Magellan does not currently expose a documented quit IPC command.

That means the first implementation should use:

- UI-side PID tracking
- direct process termination on clean shutdown

If we later add a quit IPC command to Magellan, the preferred shutdown flow can become:

1. send quit command
2. wait for clean exit
3. force terminate only if needed

### Failure Rules

- if the UI did not launch Magellan, it must not kill Magellan on shutdown
- if the PID is gone already, shutdown should continue without error
- if Magellan does not exit promptly, log the failure and continue shutting down the UI safely

### Recommended Fallback

If the background viewer is not running when a chart-open request happens:

- launch the viewer
- retry the IPC request
- only fall back to direct one-shot launch if the IPC path fails

## Launch and IPC Contract

### Static Snapshot Launch

Magellan already supports:

```bash
magellan_chart_viewer --snapshot /path/to/snapshot_dir
```

If the process is already running, the new request is forwarded to the existing instance in [main.cpp](/home/ethan/Magellan/charting_engine/src/app/main.cpp#L63).

### Live Session Launch

Magellan already supports:

```bash
magellan_chart_viewer \
  --live-session <session_id> \
  --snapshot /path/to/seed_snapshot \
  --live-title "Title" \
  --live-subtitle "Subtitle"
```

That behavior is implemented in [main.cpp](/home/ethan/Magellan/charting_engine/src/app/main.cpp#L64) and [main.cpp](/home/ethan/Magellan/charting_engine/src/app/main.cpp#L111).

### IPC Commands

Magellan currently supports these commands:

- `open_snapshot`
- `open_live`
- `live_update`
- `replace_series`
- `replace_bars`
- `reload_live_seed`
- `resize_embedded`
- `close_session`
- `release_session`

Defined in:

- [viewer_command_protocol.h](/home/ethan/Magellan/charting_engine/src/platform/viewer_command_protocol.h#L7)
- [viewer_command_protocol.cpp](/home/ethan/Magellan/charting_engine/src/platform/viewer_command_protocol.cpp#L161)

## Snapshot Contract Alignment

The current `ChartSnapshot` schema in this project already aligns well with what Magellan loads.

Shared core fields:

- `schema_version`
- `run_id`
- `strategy`
- `timeframe`
- `files.price_bars`
- `files.trades`
- `files.equity`
- `files.overlays`
- `files.panes`
- `counts.bars`

See:

- [CHART_SNAPSHOT_SCHEMA.md](/home/ethan/quant_backtest_engine/CHART_SNAPSHOT_SCHEMA.md#L79)
- [manifest_reader.cpp](/home/ethan/Magellan/charting_engine/src/snapshot/manifest_reader.cpp#L65)

### Recommended Manifest Additions

To align this project's schema more explicitly with Magellan, the snapshot manifest should also allow:

- `data_format`
- `preview_bars`
- `title`
- `subtitle`
- `status_text`

Why:

- `data_format` is already consumed by Magellan in [manifest_reader.cpp](/home/ethan/Magellan/charting_engine/src/snapshot/manifest_reader.cpp#L70)
- `preview_bars` gives Magellan a fallback and a faster initial load path in [manifest_reader.cpp](/home/ethan/Magellan/charting_engine/src/snapshot/manifest_reader.cpp#L82)
- explicit display strings will let this project control Magellan window labeling more cleanly in the future

### Recommended Rule

This project should treat the current snapshot schema as the canonical base contract, then extend it in Magellan-compatible ways rather than letting the two formats drift.

## Static Snapshot Responsibilities

For artifact mode, this project must provide:

- valid `manifest.json`
- `price_bars.feather`
- `trades.feather`
- `equity.feather`
- `overlays.feather`
- `panes.feather`
- optional `preview_bars`

Magellan then loads and renders those files asynchronously using:

- [snapshot_loader.cpp](/home/ethan/Magellan/charting_engine/src/snapshot/snapshot_loader.cpp#L27)

## Live Session Responsibilities

For live mode, this project must provide:

### Session Open

- `session_id`
- optional seed snapshot path
- title
- subtitle
- status text

Modeled by:

- [ChartLiveSessionRequest](/home/ethan/Magellan/charting_engine/src/domain/chart_snapshot.h#L100)

### Incremental Updates

This project must stream any combination of:

- bars
- overlay series updates
- pane series updates
- equity series updates
- trade markers

Modeled by:

- [ChartLiveUpdate](/home/ethan/Magellan/charting_engine/src/domain/chart_snapshot.h#L109)

Magellan already upserts those updates in place in:

- [chart_snapshot_store.cpp](/home/ethan/Magellan/charting_engine/src/app/chart_snapshot_store.cpp#L142)

## Recommended Session Types

This project should standardize three live-session categories.

### 1. Market Sessions

Purpose:

- ticker-first chart view with historical bars plus live updates

Characteristics:

- session id based on ticker and timeframe
- seeded from a historical snapshot or recent bar snapshot
- overlay series driven by user-selected indicators
- no strategy trades required

### 2. Paper Sessions

Purpose:

- monitor a paper-deployed strategy or portfolio

Characteristics:

- session id based on paper deployment id
- seeded from recent historical context
- live updates include bars, selected indicators, equity, and paper trade markers

### 3. Live Sessions

Purpose:

- monitor a live-deployed strategy or portfolio

Characteristics:

- session id based on live deployment id
- seeded from recent historical context
- live updates include bars, selected indicators, equity, and live trade markers

## UI Integration Map

### `Runs / Artifacts`

Action:

- open Magellan in artifact mode using a saved snapshot path

### `Backtest`

Action:

- open completed backtests in artifact mode

### `Market`

Action:

- open or attach to a market live session
- seed from historical data
- continue streaming live bars and selected indicators

### `Paper Engine`

Action:

- open or attach to a paper live session
- overlay paper fills, positions, and equity

### `Live Deployment`

Action:

- open or attach to a live session
- overlay live fills, positions, and equity

### `Charts`

Action:

- select a ticker from the Python UI watchlist
- seed a market live session from local historical bars plus separately stored live bars
- stream Interactive Brokers real-time bar updates into Magellan
- show user-selected indicator series supplied by this project
- support chart timeframes derived from the stored 1-minute live feed, including 1m, 5m, 15m, 1h, 4h, and 1d
- auto-subscribe the Python watchlist to Interactive Brokers live bars and show price plus regular-session percent change

The first Python-side implementation keeps the live data in `data/live_market.sqlite`, not in the historical DuckDB/Parquet store. Historical bars can seed a chart, but IB real-time bars are persisted separately so bad live ticks or provider outages cannot contaminate research datasets.

Interactive Brokers should be the first live provider for this tab. The live adapter should use `reqRealTimeBars` with `useRTH=false`, aggregate the 5-second IB bars into upserted 1-minute bars, and store each completed/current minute in the live store. Massive and Alpaca should be added later behind the same live-store/provider boundary.

When the user clicks a watchlist ticker with stale or missing live data, the UI should request a fresh live update first, then open or refresh the Magellan live session. If the local historical dataset is stale, the UI should start the live stream immediately and also request an Interactive Brokers historical gap fill into the separate live-market store. If live refresh or gap sync fails, the UI may still open a historical seed chart while clearly surfacing the live-data error.

## Embedded Chart Contract

Magellan now supports a native-parent embedded mode for the Python dashboard `Charts` tab.

The Python Qt UI should create a native host widget/window and pass its platform window id to Magellan:

```bash
magellan_chart_viewer \
  --embed-parent <native_window_id> \
  --embed-width <width_px> \
  --embed-height <height_px> \
  --live-session <session_id> \
  --snapshot /path/to/seed_snapshot
```

Static snapshots can also be embedded:

```bash
magellan_chart_viewer \
  --embed-parent <native_window_id> \
  --embed-width <width_px> \
  --embed-height <height_px> \
  --snapshot /path/to/snapshot_dir
```

The equivalent IPC fields are:

```json
{
  "type": "open_live",
  "session_id": "charts:AAPL:1m",
  "snapshot_path": "/path/to/seed_snapshot",
  "embed_parent_id": "12345678",
  "embed_width": 1180,
  "embed_height": 680
}
```

Ownership rules:

- Python owns the native host widget/window and decides when the tab exists.
- Magellan owns the child chart window and its renderer.
- Closing the Magellan chart releases the chart data for that surface.
- If the Python host widget is destroyed, this project should close/reopen the Magellan session or terminate the owned viewer process as part of dashboard cleanup.

Resize and focus behavior:

- Python should pass the current host size when opening the embedded surface.
- When the host widget resizes, Python should send `resize_embedded` for live sessions.
- Focus uses normal native child-window focus; Magellan requests activation after attaching.

Resize IPC example:

```json
{
  "type": "resize_embedded",
  "session_id": "charts:AAPL:1m",
  "width": 1180,
  "height": 680
}
```

Failure handling:

- If native-parent attachment fails, Magellan logs the failure and falls back to a top-level chart window instead of dropping the chart request.
- The Python UI should still show an integration warning if it expected an embedded surface but sees no child chart.

Surface model:

- A live `session_id` maps to one Magellan chart surface.
- Reusing the same `session_id` updates that surface in place.
- Opening multiple strategy/ticker charts at once should use distinct session ids.
- Switching one embedded chart in place should reuse the same session id and send `replace_bars`, `replace_series`, or `reload_live_seed`.
- If a tab/strategy chart is removed, send `close_session` or `release_session` for that `session_id`.

Seed reload example:

```json
{
  "type": "reload_live_seed",
  "session_id": "charts:AAPL:1m",
  "snapshot_path": "/path/to/new_seed_snapshot",
  "title": "AAPL 5m",
  "status_text": "Reloaded 5m live seed."
}
```

Close example:

```json
{
  "type": "close_session",
  "session_id": "charts:AAPL:1m"
}
```

Implemented Magellan lifecycle additions:

- `replace_bars` and `reload_live_seed` let the Python `Charts` tab change ticker, timeframe, or lookback inside one embedded surface without opening another child window.
- `close_session`/`release_session` lets Python explicitly tear down a previous embedded live surface when it must open a new session id.
- `open_live` still behaves as attach/create. Reusing a session id with a new seed should use `reload_live_seed` rather than overloading attach semantics.

Platform note:

- Magellan uses Qt's `QWindow::fromWinId` native-parent path.
- On Linux, X11 should be the first validation target. Wayland/compositor behavior can vary and should be tested on the deployment machine.

## Indicator Selection Contract

Magellan now supports interactive indicator replacement for an existing live session.

Use `live_update` for incremental points and new bars. Use `replace_series` when the user changes selected indicators and this project wants the chart to remove old indicators, clear lower panes, or replace a full group in place.

Replacement IPC example:

```json
{
  "type": "replace_series",
  "session_id": "charts:AAPL:1m",
  "replace_overlays": true,
  "replace_panes": true,
  "replace_equity": false,
  "overlay_series": [
    {
      "name": "SMA 20",
      "color": "#f5c542",
      "points": [
        {"timestamp_utc_ns": "1713551400000000000", "bar_index": 0, "value": 181.22}
      ]
    }
  ],
  "pane_series": []
}
```

Rules:

- `replace_overlays: true` replaces the full price-overlay series set with `overlay_series`.
- `replace_panes: true` replaces the full lower-indicator series set with `pane_series`.
- `replace_equity: true` replaces the full equity/benchmark series set with `equity_series`.
- Clearing a group means sending the replace flag with an empty list.
- Removing one selected indicator means recomputing the group and sending the full replacement list without that series.
- Series style currently supports `name` and `color`; line width/style metadata can be added later if the UI needs it.

## Bar And Seed Replacement Contract

Use `replace_bars` when only the visible bar set changes and this project wants to keep the existing live session/window.

```json
{
  "type": "replace_bars",
  "session_id": "charts:AAPL:1m",
  "replace_trade_markers": true,
  "bars": [
    {
      "timestamp_utc_ns": "1713551400000000000",
      "bar_index": 0,
      "open": 181.10,
      "high": 181.50,
      "low": 180.90,
      "close": 181.22,
      "volume": 1200
    }
  ],
  "trade_markers": []
}
```

Rules:

- `replace_bars` replaces the full in-memory price bar array for that session.
- `replace_trade_markers: true` replaces trade markers with `trade_markers`; sending an empty list clears markers.
- Existing overlays, lower-pane series, and equity series remain until this project sends `replace_series`.
- For ticker, timeframe, or seed-lookback changes where a complete seed snapshot exists, prefer `reload_live_seed` because it replaces bars, overlays, panes, equity, and markers together.

Python-side higher-timeframe live update rule:

- For all live market timeframes, including aggregated chart timeframes (`5m`, `15m`, `1h`, `4h`, `1d`) and lookbacks longer than `5D`, this project sends `live_update` with the newest displayed bar plus incremental indicator points.
- The Python dashboard keeps a per-session cursor for the last bar timestamp and Magellan bar index. If a live update belongs to the same aggregated bucket, it reuses the same `bar_index` so Magellan upserts the in-progress bar. If the bucket advances, the dashboard sends the next `bar_index`.
- The live `bar_index` cursor is session-local and contiguous. It must not jump to the current backing-store row number after historical gap sync or rolling lookback changes, because sparse jumps create disconnected indicator/equity segments in Magellan.
- `replace_bars` and `reload_live_seed` are reserved for explicit ticker, timeframe, lookback, or seed reload changes. They should not be sent for ordinary live ticks because Magellan intentionally refits the chart view after a full bar replacement.
- Magellan should preserve `live_update` upsert semantics by `bar_index`; that is what prevents duplicate higher-timeframe bars while avoiding the viewport recentering caused by full replacements.
- The Python dashboard labels aggregated live bars by bar start time, matching the 1-minute source timestamps. It should not use right-edge labels for `5m`/`15m`/`1h` bars because that makes live charts appear one bucket ahead.
- Historical gap sync should use minute-level freshness checks, not the selected display timeframe. A 15-minute chart can still be missing 1-minute source bars, and those gaps need to be backfilled before resampling.
- IPC live-update timestamps should be sent as decimal strings, not JSON numbers. Nanosecond epoch values exceed JavaScript/Qt JSON's exact integer range, and sending them as strings preserves x-axis/crosshair date labels for newly appended bars.
- Strategy indicator colors are owned by this project. Live Monitor uses the same snapshot/update style map for full seeds and incremental points; `Z-Score` is purple (`#a28bff`) and should not be recolored by Magellan during live updates.

IPC transport rule:

- Magellan's own sender now writes length-prefixed frames, so large JSON payloads are not split by `readAll()`.
- The Python client should either send the same `MAGELLAN_IPC_V1 <byte_count>\n<payload>` frame format or send compact one-line JSON terminated by `\n`.
- Avoid pretty-printed multi-line legacy JSON; use framed IPC for large or formatted payloads.
- During dashboard development, the Python client defaults to compact one-line JSON. This prevents an already-running older Magellan process from treating `MAGELLAN_IPC_V1 ...` as a snapshot path and opening bogus blank windows. Set `MAGELLAN_IPC_FRAMED=1` only after confirming the running Magellan viewer binary has been rebuilt and restarted with framed IPC support.

## Live Market Time Axis

Magellan now renders x-axis date/time labels and the crosshair time tag on the lowest visible pane. If a market live session has no lower-pane indicators and no equity series, the price pane shows the time axis directly. If indicators exist but equity does not, the indicator pane becomes the bottom time-axis pane.

If live charts still show empty "Indicator Pane" or "Equity Pane" regions when `paneSeriesCount == 0` and `equitySeriesCount == 0`, the dashboard is almost certainly connected to an older Magellan process. Fully terminate the existing `MagellanChartViewer` process and relaunch the dashboard so the rebuilt viewer binary owns the IPC server.

## IBKR Session Idle Handling

Interactive Brokers `reqRealTimeBars` can go quiet for US equities after the extended session closes, roughly after 8:00 PM America/New_York until premarket activity resumes. The dashboard should treat that as an idle/stale stream state, keep the last chart seed visible, mark quotes stale, and avoid doing SQLite writes outside guarded worker error handling. A stopped or idle IBKR stream must not crash the Qt process.

Live Monitor deployment charts should run the same historical-gap repair as the Charts tab. On chart open, start the real-time stream immediately, request the missing 1-minute IBKR historical window into the separate live-market SQLite store, and reload the existing Magellan live session after the gap sync completes. Gap checks should ignore overnight/weekend closures outside the US equity extended-hours window (`04:00-20:00 America/New_York`) so the dashboard does not repeatedly chase non-trading minutes.

Live Monitor must normalize any deployment dataset id or provider label into the tradable ticker before sending requests to Interactive Brokers. For example, `TQQQ_interactive_brokers_10y_1m` is a local dataset id, while `TQQQ` is the IB contract symbol. Sending the dataset id to IB produces error 200, "No security definition has been found for the request."

The live-market SQLite store uses WAL mode, busy timeouts, transient-lock retry, and batched historical-sync writes. A temporary SQLite writer lock should surface as a busy/retry status at most; it must not stop the Live Monitor chart stream.

## Python-Side Integration Layer

This repo should add a small Magellan client/launcher layer rather than scattering subprocess and IPC logic across the UI.

Recommended responsibilities:

- discover configured Magellan binary path
- ensure the background viewer process is running
- open snapshot charts
- open live sessions
- send live updates
- handle retry and fallback behavior

Recommended module shape:

- `backtest_engine/magellan_client.py`
  or
- `backtest_engine/charting/magellan_client.py`

Recommended functions:

- `ensure_viewer_running()`
- `open_snapshot(snapshot_path)`
- `open_embedded_snapshot(snapshot_path, parent_window_id, width, height)`
- `open_live_session(session_id, snapshot_path=None, title=None, subtitle=None, status_text=None)`
- `open_embedded_live_session(session_id, parent_window_id, width, height, snapshot_path=None, title=None, subtitle=None, status_text=None)`
- `send_live_update(session_id, bars=None, overlay_series=None, pane_series=None, equity_series=None, trade_markers=None)`
- `replace_series(session_id, overlay_series=None, pane_series=None, equity_series=None, replace_overlays=False, replace_panes=False, replace_equity=False)`
- `replace_bars(session_id, bars, trade_markers=None, replace_trade_markers=False, title=None, subtitle=None, status_text=None)`
- `reload_live_seed(session_id, snapshot_path, title=None, subtitle=None, status_text=None)`
- `resize_embedded(session_id, width, height)`
- `close_session(session_id)`

## Configuration

This project should support configurable Magellan settings:

- Magellan binary path
- IPC server name, default `MagellanChartViewer`
- whether to prelaunch viewer on UI startup
- launch timeout
- IPC retry timeout

Recommended defaults:

- prelaunch enabled for desktop UI sessions
- IPC server name matching Magellan's current default

## Error Handling Rules

### Snapshot Open Failures

- if the snapshot folder is missing or incomplete, fail in this project before calling Magellan when possible
- if Magellan reports a load problem, surface it in the UI and keep the run record intact

### Viewer Startup Failures

- if Magellan is not installed or the binary path is invalid, show a clear integration error
- do not block the rest of the research UI from loading

### Live Session Failures

- if session open fails, allow retry without restarting the whole UI
- if live updates fail, keep the paper/live engine running and surface chart-disconnect status separately

## Recommended Implementation Order

1. Add a Python-side Magellan launcher/client wrapper.
2. Add background prelaunch on UI startup.
3. Add static snapshot open from backtest and runs views.
4. Add embedded live-session open for the `Charts` tab using a native host widget id.
5. Add embedded resize handling from the host widget resize event.
6. Add `replace_series` calls when indicator checkboxes/selections change.
7. Add `replace_bars` or `reload_live_seed` for ticker/timeframe/lookback changes.
8. Add `close_session` when embedded tabs or strategy monitors are removed.
9. Add a market live-session path for ticker charts with historical seed plus live updates.
10. Add paper-session live updates.
11. Add live-deployment session updates.
12. Add richer status metadata and reconnect behavior later.

## Non-Goals for the First Integration Pass

- making Magellan responsible for strategy logic
- making Magellan fetch market data directly
- coupling Magellan to the backtest engine internals

## Summary

The right integration model is:

- this project produces snapshots and live chart updates
- Magellan runs as a long-lived background viewer
- the desktop UI prelaunches Magellan when it starts
- completed runs open through snapshot paths
- market, paper, and live monitoring use seeded live sessions plus incremental IPC updates
- the `Charts` tab can host Magellan through native-parent embedded windows
- indicator changes use `replace_series` to update a live chart in place
- ticker/timeframe/lookback changes use `replace_bars` or `reload_live_seed` to reuse an embedded surface
- dashboard cleanup uses `close_session`/`release_session`

This keeps the integration fast, clean, and aligned with how Magellan is already built.
