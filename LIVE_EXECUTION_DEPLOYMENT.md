# Live Execution and Deployment Design

## Purpose

This document defines a practical version 1 plan for deploying strategies from the quant backtest engine into the existing live execution engine at [algo_trading_engine](/home/ethan/algo_trading_engine).

The goal is to keep the first version simple while still covering the full workflow we actually need:

- promote validated strategies from research into live deployment
- allow manual deployment of strategies or parameter sets that did not go through the full validation pipeline
- send live execution requests to the existing external engine
- monitor deployed strategies from this project's UI
- preserve lineage from research artifacts to live behavior
- avoid editing code in [algo_trading_engine](/home/ethan/algo_trading_engine) for version 1

This document extends the research pipeline described in [PARAMETER_OPTIMIZATION.md](/home/ethan/quant_backtest_engine/PARAMETER_OPTIMIZATION.md), [WALK_FORWARD_OPTIMIZATION.md](/home/ethan/quant_backtest_engine/WALK_FORWARD_OPTIMIZATION.md), [MONTE_CARLO_SIMULATION.md](/home/ethan/quant_backtest_engine/MONTE_CARLO_SIMULATION.md), and [SYSTEM_DESIGN.md](/home/ethan/quant_backtest_engine/SYSTEM_DESIGN.md).

## Core Principles

- Keep deployment manual and explicit. Passing optimization, walk-forward, and Monte Carlo should make a candidate eligible for deployment, not auto-deploy it.
- Keep version 1 simple, but include both single-strategy and portfolio deployment paths.
- Treat the external execution engine as an integration target, not as a codebase we need to modify right now.
- Separate deployment setup from live monitoring with two top-level tabs:
  - `Deployment`
  - `Live Monitor`
- Preserve validation lineage so every live deployment can be traced back to:
  - source candidate or manual definition
  - strategy identity and version
  - parameter set
  - symbol or dataset
  - timeframe
  - optimization, walk-forward, and Monte Carlo artifacts when they exist

## Version 1 Scope

Version 1 should focus on:

- selecting deployable candidates from the existing research pipeline
- storing manual deployment definitions
- supporting both single-strategy and portfolio deployments
- defining deployment targets such as Public live, Alpaca paper, or Coinbase in the external engine
- using a simple live market-data feed stack of Interactive Brokers first, Alpaca second, and Massive third for the symbols attached to active live deployments
- emitting TradingView-compatible webhook payloads to the external engine
- storing an outbound deployment signal journal locally
- monitoring deployed strategies from the external engine's existing SQLite data and HTTP endpoints
- showing live strategy metrics such as:
  - realized PnL
  - open PnL when available
  - trade count
  - wins
  - losses
  - win rate
  - simple Sharpe when enough observations exist
  - current position state
  - last signal, order, fill, and error information
- pausing, resuming, and stopping deployments from this project by controlling whether this project continues to emit live signals

Version 1 should **not** include:

- automatic live promotion after Monte Carlo
- automatic strategy decay or drift shutdown
- embedded Magellan live charts inside the monitor tab
- benchmarked live equity charts versus SPY or the S&P 500
- remote control changes inside [algo_trading_engine](/home/ethan/algo_trading_engine)

Those are all reasonable later upgrades, but they are not required for a strong first version.

## Integration Boundary

Current external engine root:

- [algo_trading_engine](/home/ethan/algo_trading_engine)

Current useful entry points there:

- live webhook route: [paper_engine.py](/home/ethan/algo_trading_engine/paper_engine.py:159)
- live status route: [paper_engine.py](/home/ethan/algo_trading_engine/paper_engine.py:269)
- live dashboard route: [paper_engine.py](/home/ethan/algo_trading_engine/paper_engine.py:306)
- log route: [paper_engine.py](/home/ethan/algo_trading_engine/paper_engine.py:439)
- shared webhook processing and dedupe: [core.py](/home/ethan/algo_trading_engine/core.py:90)
- live execution schema: [live_db.py](/home/ethan/algo_trading_engine/live_db.py:14)

Important current facts:

- the engine already accepts a TradingView-style webhook JSON payload
- the engine dedupes by `event_id`
- the engine already persists:
  - orders
  - fills
  - account snapshots
  - live positions
  - signal queue state
  - engine logs
- the engine already exposes HTTP routes that are useful for monitoring

This means version 1 does **not** need code changes in the external project to become useful.

For the quant backtest engine UI, this deployment work should now be considered alongside two adjacent operator views:

- `Asset Screener` for building better universes from the asset master plus SimFin and defeatbeta enrichment
- `Correlation Matrix` for on-demand covariance and correlation checks across assets, sectors, strategies, and portfolios

## Current Webhook Contract

The external engine already accepts the TradingView-style request shape the user provided. The current live endpoint example from the user is:

- `http://167.99.235.90/live_webhook`

That address is explicitly subject to change, so version 1 must treat it as configuration, not a hardcoded constant.

The `regime_state` field should be ignored by this project for now.

Version 1 outbound payload should use this minimum field set:

- `secret`
- `symbol`
- `action`
- `side`
- `price`
- `time`
- `bar_index`
- `sent_ts`
- `event_id`

Version 1 should also send:

- `annual_vol`

Why:

- the external engine's default sizing path uses `percent_equity`
- that sizing path requires `annual_vol`

Version 1 should optionally support:

- `qty_type`
- `qty_value`

Recommended supported sizing modes for version 1:

- `fixed`
- `cash`
- `percent_equity`

If a deployment uses `percent_equity`, `annual_vol` must be present and valid. If not, the deployment should either:

- use `fixed` or `cash` sizing instead
- or fail fast before sending the webhook

## Key Version 1 Trick

The simplest way to integrate without editing [algo_trading_engine](/home/ethan/algo_trading_engine) is this:

- this project should attach extra metadata fields to the outbound webhook payload
- the external engine will ignore fields it does not need for execution
- those extra fields will still be preserved inside the external engine's stored `raw_payload`

That lets this project recover deployment lineage later without changing the external engine schema.

Recommended extra metadata fields:

- `deployment_id`
- `source_type`
- `source_id`
- `candidate_id`
- `strategy_name`
- `strategy_version`
- `timeframe`
- `dataset_id`
- `run_id`
- `wf_study_id`
- `mc_study_id`

This is the main reason version 1 can stay simple.

## Why The Earlier Plan Was Conservative

The earlier plan was conservative for one main reason:

- the external execution engine currently exposes a broker, order, position, and account model, not a portfolio-native deployment and attribution model

That caution was about the live integration boundary, not about this project's research capabilities.

This project already supports:

- portfolio backtests
- portfolio optimization
- portfolio walk-forward studies
- portfolio Monte Carlo studies

So portfolio live deployment should absolutely be in scope.

The practical version 1 answer is:

- model portfolio deployments in this project
- let one parent portfolio deployment own a coordinated set of child live routes or symbol legs
- keep using the external engine as the actual order-routing backend for those child legs

That gives us portfolio live deployment without pretending the external engine is already a full portfolio management system.

## Deployment Unit

Version 1 should support two deployment units.

### 1. Single-Strategy Deployment

- one strategy
- one parameter set
- one symbol or dataset
- one timeframe
- one deployment target

If the user wants to deploy the same validated candidate to five symbols, version 1 should create five deployment records.

Why this is the right version 1 choice:

- easy to explain
- easy to monitor
- easy to pause or stop independently

### 2. Portfolio Deployment

Version 1 should also support portfolio deployment as a first-class deployment kind.

A portfolio deployment should represent:

- one portfolio definition or portfolio candidate
- one deployment target
- one coordinated set of child asset or strategy-block routes
- one parent portfolio deployment record for monitoring and lifecycle control

The first version does not need a new mature live portfolio allocator to do this.

Instead, version 1 can treat a live portfolio deployment as:

- one parent portfolio deployment record in this project
- one or more child deployment legs derived from the stored portfolio structure
- shared monitoring and grouped status from this project's catalog layer

That is enough to support:

- shared-strategy portfolios across multiple assets
- fixed strategy-block portfolios

while still routing actual orders through the existing external engine.

## Deployment Sources

Version 1 should support both single-strategy and portfolio deployment sources.

### 1. Validated Single-Strategy Candidates

This is the main path.

Eligible candidates come from the existing validation chain:

- parameter optimization candidate
- linked walk-forward validation
- linked Monte Carlo study

The `Deployment` tab should surface these as the preferred source list.

Design rule:

- do not create a separate third candidate system for validated strategies
- reuse the current optimization candidate and validation-chain model

### 2. Validated Portfolio Candidates

This should also be a first-class path.

Eligible sources include:

- shared-strategy portfolio optimization candidates
- fixed strategy-block portfolio candidates
- linked portfolio walk-forward studies
- linked portfolio Monte Carlo studies

The `Deployment` tab should surface these alongside single-strategy validated candidates rather than treating them as a future special case.

### 3. Manual Deployment Definitions

This is the override path.

The user explicitly wants a manual section for strategies or parameter sets that did not go through the full pipeline.

This path should still be supported, but it should require:

- explicit source classification such as `manual`
- user notes
- a warning that the deployment skipped some or all validation stages

This gives flexibility without confusing manual deployments with validated ones.

## Deployment Tab

The `Deployment` tab should have three main sections.

### 1. Validated Candidates

Purpose:

- choose both:
  - single-strategy candidates that made it through optimization, walk-forward, and Monte Carlo
  - portfolio candidates that made it through the portfolio validation chain

Recommended columns:

- source candidate ID
- deployment kind
- strategy
- symbol or dataset scope
- timeframe
- linked walk-forward study
- linked Monte Carlo study
- key validation metrics
- notes
- created or updated time

Recommended actions:

- `Create Deployment`
- `Open Validation Chain`
- `Open Source Optimization Study`
- `Open Source Walk-Forward Study`
- `Open Source Monte Carlo Study`

Creating a deployment from this section should prefill:

- strategy identity
- parameter set
- timeframe
- linked study references
- deployment notes
- deployment kind
- portfolio structure when the source is portfolio-based

The user should still choose:

- concrete symbol or dataset if the source study was broader than one symbol
- deployment target
- sizing mode
- sizing value
- live versus paper mode on the target

### 2. Manual Deployment

Purpose:

- create deployable definitions that did not go through the full validation chain

Recommended fields:

- deployment kind
- strategy
- strategy version
- symbol or dataset
- timeframe
- params JSON or form inputs
- signal source mode
- target
- sizing mode
- sizing value
- optional notes

Recommended actions:

- `Save Manual Definition`
- `Create Deployment`
- `Clone Existing Manual Definition`

Recommended UX rule:

- manual deployments should be clearly labeled `Manual Override`

### 3. Deployed / Live Strategies

Purpose:

- list deployments that are currently armed, live, paused, stopped, or in error

Recommended columns:

- deployment ID
- deployment kind
- strategy
- symbol
- timeframe
- source type
- target
- status
- last signal time
- last order status
- current position
- realized PnL
- open PnL

Portfolio deployments should also appear here as parent rows.

Recommended actions:

- `Arm`
- `Start`
- `Pause`
- `Resume`
- `Stop`
- `Open Live Monitor`
- `Open External Dashboard`

Version 1 deployment states should stay simple:

- `draft`
- `armed`
- `live`
- `paused`
- `stopped`
- `error`

## Live Monitor Tab

The `Live Monitor` tab should focus on active and recent deployments.

Recommended layout:

### 1. Summary Row

Show:

- active live deployments
- paused deployments
- deployments in error
- total open positions
- aggregate realized PnL
- aggregate open PnL

### 2. Deployment Table

One row per deployment.

Recommended columns:

- deployment ID
- deployment kind
- strategy
- symbol
- timeframe
- target
- status
- realized PnL
- open PnL
- trade count
- wins
- losses
- win rate
- Sharpe
- last signal
- last fill
- last error

Portfolio rows should show aggregate portfolio metrics plus child-leg health counts.

### 3. Deployment Detail Panel

When one deployment is selected, show:

- source lineage
- params
- sizing settings
- current position
- recent signals
- recent orders
- recent fills
- recent errors
- recent engine logs

If the selected deployment is portfolio-based, also show:

- portfolio mode such as shared-strategy or strategy-block
- configured asset universe or block structure
- child deployment legs
- child-leg statuses
- aggregate and per-leg metrics

Recommended controls:

- `Pause`
- `Resume`
- `Stop`
- `Sync Now`
- `Open Source Validation`
- `Open External Live Dashboard`

## Version 1 Metrics

Version 1 should calculate and display these per deployment:

- realized PnL
- open PnL when current position data is available
- total trade count
- win count
- loss count
- win rate
- average win
- average loss
- profit factor
- simple Sharpe ratio
- current position quantity
- current position side
- last signal timestamp
- last order timestamp
- last fill timestamp
- last error timestamp

Practical version 1 rule for Sharpe:

- compute it only when there are enough observations to make the number meaningful
- otherwise show blank or `N/A`

That is better than showing noisy nonsense early.

## Logging

Version 1 should preserve four useful log streams:

### 1. Outbound Signal Log

Stored in this project.

Should include:

- deployment ID
- event ID
- signal timestamp
- action
- side
- price
- payload
- HTTP response code
- response body
- send failure details

### 2. Order / Fill Log

Read from the external engine and normalized into deployment attribution.

### 3. Engine Error Log

Read from the external engine's log stream.

### 4. Deployment Lifecycle Log

Stored in this project.

Should include:

- armed
- started
- paused
- resumed
- stopped
- sync failures
- reconciliation notes

## Monitoring Transport

Version 1 should support two monitor paths.

### 1. Co-Located Read-Only Mode

Use this when the external engine lives on the same machine, which is the current situation.

Read-only inputs can come from:

- [live_db.py](/home/ethan/algo_trading_engine/live_db.py:14)
- the other broker-specific SQLite databases when needed
- the engine log database
- the existing HTTP status routes

This mode should be the default version 1 monitor path because it gives us:

- orders
- fills
- positions
- account snapshots
- logs

without requiring external engine code changes.

### 2. Remote HTTP Mode

Use this when the execution engine is remote and only an address is known.

This mode can still use:

- `/live_status`
- `/live_dashboard_data`
- `/logs_data`

But it will be weaker unless the external engine later exposes richer deployment-specific endpoints.

Design rule:

- version 1 target configuration should let us choose `co_located` or `remote_http`

## Proposed Data Model

Version 1 should add a small live-deployment catalog to the existing SQLite database.

### `deployment_targets`

Purpose:

- define where live or paper execution requests are sent

Suggested fields:

- `target_id`
- `name`
- `mode` such as `live` or `paper`
- `broker_scope` such as `public`, `alpaca`, or `coinbase`
- `transport_mode` such as `co_located` or `remote_http`
- `base_url`
- `webhook_path`
- `status_path`
- `dashboard_path`
- `logs_path`
- `project_root`
- `db_path`
- `log_db_path`
- `secret_ref`
- `is_active`
- `created_at`
- `updated_at`

Important rule:

- store only `secret_ref` here, not the raw secret value

### `manual_deployment_definitions`

Purpose:

- persist user-authored deployment definitions outside the research pipeline

Suggested fields:

- `manual_definition_id`
- `deployment_kind`
- `strategy`
- `strategy_version`
- `dataset_id`
- `symbol`
- `dataset_scope_json`
- `timeframe`
- `params_json`
- `structure_json`
- `target_id`
- `mode`
- `sizing_json`
- `notes`
- `created_at`
- `updated_at`

### `deployments`

Purpose:

- one row per actual live deployment definition

Suggested fields:

- `deployment_id`
- `parent_deployment_id`
- `deployment_kind`
- `source_type` such as `validated_candidate` or `manual`
- `source_id`
- `candidate_id`
- `strategy`
- `strategy_version`
- `dataset_id`
- `symbol`
- `timeframe`
- `params_json`
- `structure_json`
- `validation_refs_json`
- `target_id`
- `mode`
- `sizing_json`
- `status`
- `status_reason`
- `last_signal_at`
- `last_sync_at`
- `last_error_at`
- `notes`
- `created_at`
- `updated_at`
- `armed_at`
- `started_at`
- `stopped_at`

Recommended `deployment_kind` values:

- `single_strategy`
- `portfolio_shared_strategy`
- `portfolio_strategy_blocks`

### `deployment_events`

Purpose:

- durable audit trail for signal sending and deployment lifecycle activity

Suggested fields:

- `deployment_id`
- `seq`
- `event_id`
- `event_type`
- `action`
- `side`
- `signal_ts`
- `sent_ts`
- `price`
- `http_status`
- `payload_json`
- `response_json`
- `error_message`
- `created_at`

### `deployment_trade_ledger`

Purpose:

- normalized trade-attribution layer used for live metrics

Suggested fields:

- `deployment_id`
- `trade_id`
- `symbol`
- `side`
- `entry_ts`
- `exit_ts`
- `entry_price`
- `exit_price`
- `quantity`
- `gross_pnl`
- `net_pnl`
- `fees`
- `source_order_refs_json`
- `source_fill_refs_json`

This table is what lets the `Live Monitor` tab show wins, losses, trade count, and win rate cleanly.

### `deployment_metric_snapshots`

Purpose:

- cache the latest computed monitor metrics per deployment

Suggested fields:

- `deployment_id`
- `snapshot_ts`
- `realized_pnl`
- `open_pnl`
- `trade_count`
- `win_count`
- `loss_count`
- `win_rate`
- `profit_factor`
- `sharpe`
- `current_position_json`
- `health_json`

### `deployment_child_links`

Purpose:

- link parent portfolio deployments to their child legs

Suggested fields:

- `parent_deployment_id`
- `child_deployment_id`
- `child_role`
- `dataset_id`
- `symbol`
- `strategy_block_id`
- `created_at`

## Recommended Version 1 Data Flow

1. User selects a validated candidate or manual definition in the `Deployment` tab.
2. User chooses a target and sizing settings.
3. This project creates either:
   - one single-strategy `deployments` record
   - or one parent portfolio `deployments` record plus child deployment links
4. The live runner evaluates the strategy and emits TradingView-compatible webhook payloads to the configured target.
5. This project records each outbound signal in `deployment_events`.
6. A monitor sync worker reads the external engine state.
7. The sync worker attributes orders and fills back to `deployment_id` using stored payload metadata.
8. This project updates `deployment_trade_ledger` and `deployment_metric_snapshots`.
9. The `Live Monitor` tab renders the latest deployment state, metrics, and logs.

## Signal Source Modes

Version 1 should allow two practical source modes:

- `native_bar_close`
- `external_alert_compatible`

`native_bar_close` means this project evaluates strategy logic itself and sends the webhook.

`external_alert_compatible` means we still use the same deployment catalog and payload shape, but the event source may be external as long as it preserves the same deployment metadata contract.

This keeps version 1 flexible without forcing us to solve every live-data problem at once.

## Portfolio Deployment Shape For Version 1

Portfolio deployment does not require us to wait for a hypothetical mature live allocator.

The recommended version 1 shape is:

### Shared-Strategy Portfolio

- one parent deployment
- one child leg per asset
- shared strategy params and shared portfolio structure metadata
- parent monitor row aggregates child-leg state

### Fixed Strategy-Block Portfolio

- one parent deployment
- one child leg per block-asset route
- stored strategy-block structure from the validated portfolio definition
- parent monitor row aggregates child-leg state

This keeps portfolio live deployment aligned with the research model the platform already has.

## Important Version 1 Constraints

- The system should never hardcode the current IP address or webhook secret in source-controlled code.
- Deployment should remain a human approval step even for fully validated candidates.
- Manual deployments should always remain distinguishable from validated ones.
- Version 1 should not pretend that portfolio deployment is portfolio-attribution complete on day one.
- Version 1 should not depend on edits to [algo_trading_engine](/home/ethan/algo_trading_engine).

## Notes For `algo_trading_engine`

No edits are required there for version 1.

If we later want a cleaner version 2 integration, the most useful optional changes in that project would be:

- explicit `deployment_id` columns in orders and fills tables
- a deployment-filtered fills endpoint
- a deployment-filtered summary endpoint
- a deployment pause or disable endpoint
- richer per-deployment health and queue metrics

Those are nice-to-have improvements, not version 1 blockers.

## Version 2 Roadmap

Version 2 can build on the version 1 deployment spine with richer monitoring and control.

Recommended version 2 additions:

- strategy decay and drift detection against validation baselines
- live embedded Magellan charts for active trades
- live equity chart benchmarked against SPY or the S&P 500
- paper-to-live promotion workflow with stricter gates
- automated alerts for:
  - repeated rejected orders
  - abnormal slippage
  - missing fills
  - unusual divergence from expected signal timing
- deeper deployment analytics such as:
  - turnover drift
  - holding-period drift
  - fill-quality drift
  - live versus expected behavior deltas

## Summary

The clean version 1 plan is:

- add a `Deployment` tab for validated candidates, manual deployments, and deployed strategies
- add a `Live Monitor` tab for runtime metrics, controls, trades, and logs
- treat portfolio deployment as a supported deployment kind, not as a future-only idea
- use the existing external execution engine as-is
- send TradingView-compatible webhook payloads from this project
- attach deployment metadata to those payloads so live state can be attributed back to research lineage
- read the external engine in a read-only way for monitoring
- defer strategy decay, embedded charts, and benchmark visuals to version 2

That gives us a simple but complete first live-deployment workflow without overbuilding it.
