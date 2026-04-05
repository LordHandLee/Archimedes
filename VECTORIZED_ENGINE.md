# Vectorized Engine Design

## Purpose

This document defines the target design for a hybrid execution layer that supports both:

- a reference event-driven engine for correctness and feature completeness
- a vectorized engine for high-throughput research workloads

The goal is not to replace the current engine in one risky rewrite.

The goal is to let both engines coexist behind a stable execution contract so the platform can:

- stay correct
- gain speed where appropriate
- avoid reworking higher-level workflows later

## Why This Exists

The current engine is event-driven and stateful.

That is the right foundation for realism and correctness, but it is not the best shape for:

- large parameter sweeps
- repeated optimization runs
- batched research studies across many assets

At the same time, not every strategy or execution rule is a good fit for vectorization.

So the platform should not choose between:

- speed only
- realism only

It should support both and select the right engine for the task.

## Core Principles

- The reference engine remains the source of truth for correctness.
- The vectorized engine is an acceleration layer, not a semantic shortcut hidden from the user.
- Both engines must expose the same external execution contract.
- Unsupported vectorized runs must fail clearly or fall back clearly, never silently change semantics.
- Strategy compatibility must be explicit and testable.
- Higher-level workflows such as optimization, walk-forward, and portfolio research must depend on an execution interface, not a specific engine implementation.

## Role in the Platform

This is a core execution-layer design, not a separate research stage.

It affects:

- single backtests
- parameter optimization
- walk-forward optimization
- future portfolio research
- chart snapshot generation
- caching and result cataloging

The highest immediate value is in optimization and research throughput.

## The Two Engines

### 1. Reference Engine

This is the current event-driven engine and broker model.

Responsibilities:

- execute bar by bar
- maintain full state across time
- support richer order behavior
- remain the parity benchmark for the vectorized engine

This engine should continue handling:

- complex path-dependent strategies
- pending orders
- stop and limit logic
- `on_after_fill` behavior
- intrabar simulation
- advanced portfolio behavior until vectorized support matures

### 2. Vectorized Engine

This is a new execution backend optimized for compatible workloads.

Responsibilities:

- compute indicators and signals in bulk
- evaluate many compatible runs efficiently
- support large optimization studies
- emit results in the same format as the reference engine

This engine should initially focus on:

- simple strategies
- simpler fills
- single-asset run semantics
- batched execution across many parameter sets and optionally many independent assets

## User-Facing Execution Modes

The platform should expose three execution modes:

### `Auto`

Recommended default.

Behavior:

- inspect the strategy, config, and requested workload
- use the vectorized engine if the run is supported
- otherwise fall back to the reference engine

Requirements:

- the resolved mode must be persisted in metadata
- the reason for fallback must be available in logs or run metadata

### `Reference`

Behavior:

- always use the event-driven reference engine

Use cases:

- correctness checks
- complex strategies
- parity testing
- debugging

### `Vectorized`

Behavior:

- require the vectorized engine
- fail clearly if the run is unsupported

Use cases:

- optimization studies where speed is the goal
- explicit benchmarking
- deliberate use of compatible vectorized strategies

## Shared Execution Contract

Higher-level modules should not call the current engine directly.

They should submit an execution request through a stable orchestration layer.

### Execution Request

A request should contain at minimum:

- dataset or dataset group
- timeframe
- strategy identity
- strategy params
- backtest config
- requested execution mode
- workload type such as single run, optimization batch, or walk-forward fold

### Execution Result

Every successful execution should produce a normalized result with:

- `run_id`
- `logical_run_id`
- `requested_execution_mode`
- `resolved_execution_mode`
- `engine_impl`
- `engine_version`
- `fallback_reason` if applicable
- equity curve
- trades
- metrics
- snapshot/artifact references
- cache metadata

### Why `logical_run_id` and `run_id` Should Be Separate

The same logical run may be executed on different engines for comparison.

Recommended meaning:

- `logical_run_id`: data + strategy + params + semantic config identity
- `run_id`: `logical_run_id` plus resolved engine implementation/version

This allows:

- engine-to-engine comparisons
- clean caching
- durable audit trails

## Execution Orchestrator

The system should add an orchestration layer above the engines.

Recommended responsibilities:

- validate the execution request
- resolve execution mode
- check strategy capability
- dispatch to the appropriate backend
- normalize outputs
- persist metadata and artifacts

Conceptually, this layer chooses:

- reference engine
- vectorized engine
- fallback from `Auto`

The rest of the platform should talk to this orchestration layer, not directly to engine internals.

## Strategy Compatibility Model

Not every strategy should support vectorization.

This must be explicit.

### Why Some Strategies Do Not Support Vectorization Well

A strategy becomes difficult to vectorize when it depends heavily on evolving path state such as:

- current cash after prior fills
- current position after partial exits
- pending stop or limit orders
- intrabar order sequencing
- immediate recalculation after fills
- dynamic trailing stops based on execution price
- pyramiding or scale-in/scale-out logic
- ranking assets against each other using shared capital

These are not impossible to compute, but they are often a poor fit for a clean vectorized kernel.

When forced into vectorization too early, they tend to become:

- hard to reason about
- hard to test
- less performant than expected
- easy to get subtly wrong

### Strategy Capability Declaration

Each strategy should declare its execution compatibility explicitly.

Recommended capability metadata:

- supports reference engine: yes or no
- supports vectorized engine v1: yes or no
- supports long-only vectorized execution
- supports simple long/short vectorized execution
- requires pending orders
- requires `on_after_fill`
- requires intrabar execution
- requires portfolio-aware shared capital

This compatibility should be machine-readable so the orchestrator can resolve execution mode cleanly.

### Recommended Authoring Model

The existing event-driven strategy API should remain valid.

Strategies should be able to opt into vectorization with a separate adapter or capability definition rather than rewriting the entire base strategy model.

Recommended concept:

- event-driven strategy logic remains in the current `Strategy` API
- vectorizable strategies may provide a `VectorizedStrategyAdapter`

That adapter should define things like:

- parameter validation rules
- feature and indicator generation
- signal generation
- target position or target exposure generation
- overlay and pane series for chart snapshots if supported

This keeps vectorized logic explicit and avoids pretending every event-driven strategy is automatically vectorizable.

## Current Strategy Examples

### `SMACrossStrategy`

This is a strong candidate for vectorized support in version 1 because it is close to:

- precompute indicators
- compare arrays
- derive a target position state
- apply a simple execution rule

### `InverseTurtleStrategy`

This should likely remain reference-only in version 1 because it currently depends on:

- channel logic plus ATR logic
- dynamic stop-order maintenance
- order cancel/reissue behavior
- position-dependent stop placement based on average fill price
- short behavior and richer order state

It may become vectorizable later in a restricted form, but it should not define the first implementation boundary.

## Vectorized Engine Version 1 Scope

Vectorized version 1 should be deliberately narrow.

### Supported Run Semantics

- single-asset run semantics
- one position stream per run
- bar-based execution only
- deterministic parameter grids
- large batches of compatible runs
- shared market data arrays reused across many parameter combinations

### Supported Order / Position Model

Recommended version 1 support:

- market-style entry and exit semantics
- target state or target exposure model
- flat to long transitions
- optional simple flat/long/short transitions if borrowing assumptions remain simple
- fixed fee and slippage modeling

### Recommended Config Support in Version 1

Vectorized version 1 should support only a subset of the current config surface.

Good candidates:

- timeframe
- horizon filters
- starting cash
- fee rate
- slippage
- allow short only if the target-state model supports it cleanly
- cache usage

Reference-only for version 1:

- `intrabar_sim`
- `recalc_on_fill`
- pending stop/limit order semantics
- dynamic cancel/reissue order flows
- partial fills
- borrow costs with realistic path dependence
- highly stateful scaling rules

### Supported Workload Shapes

Vectorized version 1 should support:

- one asset, many parameter combinations
- many assets, many parameter combinations, as independent batched runs

This is important:

- batched independent multi-asset studies are not the same thing as true portfolio backtesting

Version 1 can absolutely accelerate optimization across many assets, as long as each run remains an independent single-asset calculation.

## Why Single-Asset Run Semantics Come First

True portfolio vectorization is harder because assets interact through shared state such as:

- one cash pool
- one equity curve
- allocation competition
- rebalancing
- exposure limits
- cross-asset selection rules

That interaction is real portfolio logic, not just "more arrays."

So the right phased approach is:

1. single-asset run semantics
2. batched independent runs across assets and params
3. later portfolio-aware shared-capital vectorization

This still gives large speed gains early, especially for optimization.

## Batch Execution Model

The vectorized engine should be designed to execute many compatible runs together.

Conceptually:

- time is one dimension
- parameter combinations are one dimension
- assets may be another dimension when runs are independent

Examples:

- `[time, param]` for one asset and many parameter combinations
- `[time, asset, param]` for many independent assets and parameter combinations

The engine then computes:

- features
- signals
- position states
- returns
- equity

in bulk across those dimensions.

Even when runs are computed in batch, the catalog should still persist them as atomic runs.

## Vectorized Engine Pipeline

A compatible vectorized workload should flow like this:

1. Validate requested execution mode.
2. Validate strategy and config compatibility.
3. Load and align market data arrays.
4. Expand the parameter grid into a batch dimension.
5. Compute indicators/features in bulk.
6. Compute signals in bulk.
7. Convert signals into target positions or exposure states.
8. Apply execution timing rules such as next-bar-open or supported close-based semantics.
9. Apply fee and slippage adjustments.
10. Compute returns, pnl, drawdowns, and equity curves in bulk.
11. Extract trade events from position-state changes.
12. Normalize and persist outputs as atomic run results.

## Strategy Output Model for Vectorization

Version 1 vectorized strategies should not place discrete pending orders.

Instead, they should emit a simpler representation such as:

- target position state
- target exposure
- entry/exit masks that can be translated deterministically

This is one reason vectorized execution is narrower than the reference engine.

## Supported Execution Semantics in Version 1

A supported vectorized run should be explicit about execution timing.

Recommended version 1 default:

- signals computed on bar close
- fills applied at next bar open

Optional later support:

- close-to-close semantics

Reference-only semantics in version 1:

- intrabar path simulation
- same-bar recursive fill/recalc behavior
- pending order books with stop/limit evaluation

## Trades and Artifact Generation

Both engines should feed the same downstream systems.

That means the vectorized engine must still generate:

- equity curves
- trade logs
- metrics
- chart snapshots where supported

### Trade Extraction

In vectorized mode, trades will usually be synthesized from position changes rather than emitted one order at a time during execution.

This is acceptable as long as:

- the semantics are documented
- parity stays within agreed tolerance
- unsupported behaviors remain reference-only

### Chart Snapshots

For supported strategies, vectorized runs should still be able to generate the same `ChartSnapshot` artifact shape used by the charting system.

Required snapshot components remain:

- price bars
- trades
- equity
- overlays
- panes

If a strategy cannot provide vectorized overlay/pane series cleanly, it should not claim vectorized snapshot support.

## Caching and Cataloging

The result catalog must be aware of execution mode and engine version.

Recommended run metadata additions:

- `logical_run_id`
- `requested_execution_mode`
- `resolved_execution_mode`
- `engine_impl`
- `engine_version`
- `fallback_reason`
- `capability_profile`

This prevents:

- cache collisions across engines
- ambiguous comparisons
- silent mismatches between runs

## Optimization Integration

Parameter optimization is the first major consumer of the vectorized engine.

Recommended design rules:

- optimization studies should choose an execution mode at study start
- `Auto` may resolve once for the whole study
- if the study resolves to vectorized mode, all atomic runs in the study should use the same resolved execution mode
- if unsupported, the study should either fall back cleanly or fail clearly depending on the requested mode

Version 1 optimization should benefit most from:

- reused data arrays
- batched parameter evaluation
- batched independent asset evaluation

## Walk-Forward Integration

Walk-forward should also use the shared execution contract.

Recommended design rules:

- choose one execution mode per walk-forward study
- keep the execution mode consistent across folds
- persist the resolved mode in fold metadata
- do not mix reference and vectorized semantics inside one study unless explicitly running a comparison workflow

This keeps fold-to-fold results interpretable.

## Portfolio Integration

Portfolio support should be phased carefully.

### Version 1

- no true shared-capital portfolio vectorization
- allow only independent batched runs

### Later Version

- support shared capital
- support allocation competition
- support rebalance logic
- support portfolio-level attribution in vectorized form

Until then, true portfolio backtests should remain on the reference engine.

## Failure and Fallback Rules

These rules must be explicit.

### In `Auto`

- if vectorized support is available, use it
- if not, fall back to reference
- record the fallback reason

### In `Vectorized`

- if unsupported, fail with a clear error
- explain which feature, config, or strategy capability blocked execution

### In `Reference`

- always run on the reference engine

There should never be a silent downgrade that the user cannot see later.

## Parity and Confidence Strategy

The vectorized engine should earn trust through parity testing.

### Parity Levels

Recommended expectations:

- exact or near-exact parity for compatible simple strategies
- documented tolerance bands for floating-point or trade-extraction differences
- no claim of parity for unsupported semantics

### Recommended Test Coverage

- signal timestamp parity
- trade count parity
- trade direction parity
- equity curve parity within tolerance
- metric parity within tolerance
- snapshot schema parity

### Golden Strategy Set

Start parity with a small stable set such as:

- SMA crossover
- one more simple indicator or breakout strategy with market-style execution only

Do not start parity validation with the most stateful strategy in the codebase.

## Performance Goals

The vectorized engine should be judged on research throughput, not just raw microbenchmarks.

Important measures:

- optimization study wall-clock time
- runs per second for compatible workloads
- memory usage under batched execution
- result parity versus the reference engine

The goal is not just "faster."

The goal is:

- materially faster optimization and research workflows without sacrificing trust

## Phased Implementation Plan

### Phase 1: Execution Abstraction

Build:

- shared execution request/result contract
- orchestrator
- execution mode selection
- metadata and catalog support for engine identity

This phase avoids future rework even before the vectorized kernel is complete.

### Phase 2: Vectorized Engine v1

Build:

- single-asset compatible kernel
- batched parameter evaluation
- support for a simple vectorizable strategy such as SMA crossover
- parity tests against the reference engine

### Phase 3: Optimization Integration

Build:

- optimization studies that run through the shared execution contract
- study-level mode selection
- vectorized execution for compatible studies

### Phase 4: More Strategy Coverage

Add:

- more simple vectorizable strategies
- richer but still compatible cost and short semantics
- better artifact support

### Phase 5: Portfolio-Aware Vectorization

Add later:

- shared-capital multi-asset logic
- rebalance and allocation rules
- portfolio-level vectorized attribution

## Non-Goals for Version 1

These should stay out of scope for the first implementation:

- rewriting every current strategy for vectorization
- full parity across all order types and execution semantics
- intrabar simulation in vectorized mode
- full portfolio-aware shared-cash vectorization
- silent behavior changes for unsupported strategies

## Summary

The right design is not:

- delete the reference engine
- or pretend every strategy can be vectorized immediately

The right design is:

- keep the reference engine as the correctness engine
- add a vectorized speed engine for compatible workloads
- let the user choose `Auto`, `Reference`, or `Vectorized`
- standardize contracts, metadata, and artifacts across both
- start with single-asset run semantics and batched independent runs
- expand support only after parity and trust are established

That approach gives the platform real speed gains without forcing a dangerous full-engine rewrite up front.
