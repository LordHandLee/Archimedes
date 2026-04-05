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
- snapshot viewing
- seeded live sessions
- incremental live updates for bars, overlays, panes, equity, and trade markers

These are described in [README.md](/home/ethan/Magellan/charting_engine/README.md#L7) and implemented from [main.cpp](/home/ethan/Magellan/charting_engine/src/app/main.cpp#L61).

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
- `open_live_session(session_id, snapshot_path=None, title=None, subtitle=None, status_text=None)`
- `send_live_update(session_id, bars=None, overlay_series=None, pane_series=None, equity_series=None, trade_markers=None)`

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
4. Add a market live-session path for ticker charts with historical seed plus live updates.
5. Add paper-session live updates.
6. Add live-deployment session updates.
7. Add richer status metadata and reconnect behavior later.

## Non-Goals for the First Integration Pass

- embedding Magellan directly inside the Python UI
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

This keeps the integration fast, clean, and aligned with how Magellan is already built.
