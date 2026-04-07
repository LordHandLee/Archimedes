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

## Lessons from the Local `vectorbt` Reference

The local `vectorbt` checkout at `/home/ethan/vectorbt-master` is useful as an architectural reference, not as a drop-in dependency.

The most valuable lessons for this project are:

- The speed win does not come from eliminating all loops. It comes from moving loops out of Python strategy orchestration and into a small fast simulation core over broadcasted arrays.
- Inputs, indicators, signals, and parameter grids should be aligned into a common shape before execution.
- Signal-driven execution is the best first target. Order-book-style flexibility should come later.
- Indicator preparation and parameter expansion should be treated as a reusable pipeline, not rebuilt ad hoc inside each optimization loop.
- Repeated parameter studies should reuse data arrays and cached intermediate calculations whenever possible.
- Independent batched runs are not the same thing as true shared-capital portfolio simulation.

This means the right mental model for our engine is not:

- rewrite everything as NumPy in one shot

It is:

- standardize data and parameter broadcasting
- standardize vectorizable strategy adapters
- run a fast execution kernel over that prepared workload
- normalize results back into the same platform contract

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
- dashboard comparison views keyed by `logical_run_id`
- batch-level benchmark views that summarize paired reference/vectorized runs
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

## First Implementation Target

The first production target should be intentionally narrow.

Use this as the semantic baseline:

- `SMACrossStrategy`
- single asset
- long-only
- bar-close signal generation
- next-bar-open fills
- fixed fee rate
- fixed slippage
- no intrabar simulation
- no `on_after_fill`
- no pending stop/limit orders

If this slice is fast, clean, and parity-tested, it will already unlock meaningful acceleration for grid search and optimization.

If this slice is not solid, broadening scope will only multiply complexity.

The current implementation has expanded beyond this original baseline. Version 1 now also includes limited pending stop/limit support, adapter-driven cancel/replace flows, partial fills, `fill_on_close`, and short borrow accrual for supported strategies.

## Vectorized Engine Version 1 Scope

Vectorized version 1 should be deliberately narrow.

### Supported Run Semantics

- single-asset run semantics
- one position stream per run
- bar-based execution only
- deterministic parameter grids
- large batches of compatible runs
- shared market data arrays reused across many parameter combinations
- same-timeframe execution on the provided bar set, including resampled `5m` / `15m` / `1h` bars when requested explicitly

### Supported Order / Position Model

Recommended version 1 support:

- market-style entry and exit semantics
- adapter-defined order plans for entry and exit conditions
- flat to long transitions
- simple flat/long/short transitions when the adapter emits clean short entry and exit rules
- a small fixed-size pending order book per run, including simultaneous stop and limit entry/exit orders emitted by the adapter
- adapter-driven cancel/replace of one or more pending orders within that small order book
- limited adapter-driven `recalc_on_fill` support for same-timeframe runs
- bounded same-bar fill -> after-fill -> re-execute loops for same-timeframe runs, controlled by a deterministic pass cap
- parity with reference-style `one_order_per_signal` pruning when that option is enabled
- fixed fee and slippage modeling
- partial fills via `fill_ratio`
- same-bar market fills when `fill_on_close=true`
- borrow accrual on short positions

### Recommended Config Support in Version 1

Vectorized version 1 should support only a subset of the current config surface.

Good candidates:

- timeframe
- horizon filters
- starting cash
- fee rate
- slippage
- `fill_ratio`
- `fill_on_close`
- `borrow_rate`
- `max_recalc_passes`
- `allow_short` only if the adapter emits clean short semantics
- cache usage

Reference-only for version 1:

- `intrabar_sim`
- unrestricted `recalc_on_fill` for adapters that do not provide an after-fill update path
- full `base_execution` parity where signals are generated on one timeframe and fills are driven from a lower base timeframe
- large or unbounded pending-order books per run
- deeply recursive or effectively unbounded cancel/reissue flows that depend on repeated fill-driven state changes within the same bar
- borrow costs with more complex path dependence than bar-close accrual
- highly stateful scaling rules

### Supported Workload Shapes

Vectorized version 1 should support:

- one asset, many parameter combinations
- many assets, many parameter combinations, as independent batched runs
- explicit same-timeframe studies on resampled bars, even when the chosen timeframe is not `1 minutes`

This is important:

- batched independent multi-asset studies are not the same thing as true portfolio backtesting
- same-timeframe resampled execution is not the same thing as full lower-timeframe `base_execution` parity

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

## Concrete Module Layout for This Repo

The current design is correct, but the repo still needs a more explicit module split.

Recommended additions:

- `backtest_engine/execution.py`
  Shared `ExecutionRequest`, `ExecutionResult`, portfolio execution request/result types, mode resolution, and orchestrator entry point.
- `backtest_engine/reference_engine.py`
  Thin adapter around the current `BacktestEngine` so the orchestrator can treat the reference engine as one backend among two.
- `backtest_engine/vectorized_engine.py`
  Vectorized backend entry point for compatible workloads.
- `backtest_engine/vectorized_types.py`
  Batch shapes, prepared arrays, compatibility reports, and normalized vectorized result containers.
- `backtest_engine/vectorized_batch.py`
  Parameter-grid expansion, array broadcasting, horizon slicing, and workload preparation.
- `backtest_engine/vectorized_strategies.py`
  `VectorizedStrategyAdapter` implementations, starting with `SMACrossStrategy`.
- `backtest_engine/vectorized_trade_extractor.py`
  Convert target-position or exposure changes into synthetic trades and equity events.

Recommended updates to existing modules:

- `backtest_engine/strategy.py`
  Add capability metadata and optional adapter registration hooks.
- `backtest_engine/grid_search.py`
  Stop instantiating `BacktestEngine` directly and route through the execution orchestrator.
- `backtest_engine/catalog.py`
  Add engine-aware metadata such as `logical_run_id`, `requested_execution_mode`, `resolved_execution_mode`, `engine_impl`, and `engine_version`.
- `backtest_engine/metrics.py`
  Continue sharing the same metric calculations where possible so vectorized and reference runs are measured identically.

Recommended test additions:

- `tests/test_vectorized_parity.py`
- `tests/test_execution_mode_resolution.py`
- `tests/test_vectorized_grid_search.py`

## Strategy Output Model for Vectorization

Version 1 vectorized strategies should prefer a compact execution representation.

The primary path is still:

- target position state
- target exposure
- entry/exit masks that can be translated deterministically

But version 1 now also allows a limited richer form:

- a small fixed-size pending order book per run
- explicit stop/limit order types and prices emitted by the adapter
- optional adapter-driven after-fill updates that cancel, replace, or submit multiple pending orders
- bounded same-bar recursive re-execution after fills, up to a configured pass cap

When `one_order_per_signal=true`, the vectorized engine should still mirror the reference engine's pruning behavior and collapse those pending orders down to the first actionable one.

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
- unbounded or highly recursive same-bar fill/recalc behavior
- large or highly recursive pending order books with repeated within-bar state updates

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

- allow independent batched runs for research studies
- allow a narrow shared-capital portfolio mode with same-timeframe, long-only semantics
- keep portfolio ownership rules explicit so portfolio logic does not silently replace strategy-owned sizing

### Later Version

- support shared capital
- support allocation competition
- support rebalance logic
- support portfolio-level attribution in vectorized form
- support ranking and selection modes that are aware of allocation ownership

Until the later phases are complete, complex portfolio backtests should remain on the reference engine.

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
- z-score mean reversion
- synthetic stop/limit cancel-replace parity cases

Do not start parity validation with the most stateful strategy in the codebase.

## Performance Goals

The vectorized engine should be judged on research throughput, not just raw microbenchmarks.

Important measures:

- optimization study wall-clock time
- runs per second for compatible workloads
- memory usage under batched execution
- result parity versus the reference engine

Current implemented performance work:

- memory-aware parameter batching in the vectorized engine so large compatible studies can be executed in chunks instead of building one monolithic order-plan matrix
- reuse of shared market arrays inside each vectorized batch
- adapter-level prepared context reuse across chunks so indicator/preparation work does not need to be rebuilt for every chunk
- batch-level execution benchmarks that record chunk counts, chunk sizes, cache hits, and wall-clock duration for each optimization batch
- independent-asset study aggregation so the same compatible optimization can be run across multiple datasets and summarized as one research study without implying portfolio semantics

The goal is not just "faster."

The goal is:

- materially faster optimization and research workflows without sacrificing trust

## Phased Implementation Plan

### Phase 0: Freeze the First Semantic Slice

Before writing the vectorized backend, lock down the first supported semantic target:

- `SMACrossStrategy`
- long-only
- single asset
- next-bar-open fills
- fixed fee/slippage
- no intrabar or pending-order behavior

Deliverables:

- written compatibility checklist
- one or two golden datasets/runs for parity
- explicit statement of what is reference-only in version 1

This prevents the first implementation from expanding into a hidden rewrite.

### Phase 1: Execution Abstraction

Build:

- `ExecutionRequest` and `ExecutionResult`
- portfolio execution request/result types under the same orchestration layer
- orchestrator and mode resolver
- thin `ReferenceEngine` adapter over the current `BacktestEngine`
- catalog schema support for engine identity

Change these code paths first:

- `backtest_engine/grid_search.py`
- any dashboard or study entry point that currently creates `BacktestEngine` directly
- `backtest_engine/catalog.py`

Acceptance criteria:

- the existing reference engine still runs through the new orchestrator with no behavior change
- a run records requested mode, resolved mode, and engine identity
- higher-level workflows no longer depend directly on `BacktestEngine`

### Phase 2: Vectorized Engine v1

Build:

- vectorized workload preparation for one asset and many parameter combinations
- shared-array parameter broadcasting
- `SMACrossStrategy` vectorized adapter
- first vectorized execution kernel
- trade extraction from target-position changes
- support for explicit same-timeframe vectorized execution on resampled `5m` / `15m` / `1h` bars

Implementation notes:

- keep the first kernel simple and deterministic
- use NumPy first if it keeps the design clean
- add a compiled kernel path later if profiling shows Python overhead remains dominant
- keep `Auto` conservative only for unsupported higher-timeframe workloads; when a study cleanly matches supported same-timeframe vectorized semantics, `Auto` may route it to vectorized execution

Acceptance criteria:

- one-asset SMA studies execute in batch without spinning up one `BacktestEngine` per parameter set
- output includes normalized equity, trades, metrics, and run metadata
- parity is within agreed tolerance against the reference engine
- explicit vectorized studies on resampled higher-timeframe bars run successfully with same-timeframe semantics

### Phase 3: Optimization Integration

Build:

- grid search routed through the orchestrator
- study-level execution mode selection
- reuse of loaded market arrays across parameter combinations
- reuse of prepared adapter context across chunked vectorized batches
- explicit batch benchmark capture for both reference and vectorized optimization batches
- optional batching across independent assets for research studies

Acceptance criteria:

- optimization no longer recreates the whole engine per parameter set for compatible strategies
- the study can run in `Auto`, `Reference`, or `Vectorized`
- resolved execution mode is stored per study and per atomic run
- large compatible parameter grids can be chunked safely without changing results
- batch studies expose enough benchmark metadata to profile chunking behavior and compare research throughput across assets

### Phase 4: More Strategy Coverage

Add:

- more vectorizable adapters beyond SMA crossover
- optional simple short support where semantics remain clean
- improved indicator caching and reuse across studies
- direct chart snapshot support from vectorized outputs
- full vectorized `base_execution` parity as a distinct milestone after same-timeframe higher-timeframe support

Do not add stateful strategies just because they are popular. Add them only when the vectorized semantics are still clear and testable.

### Phase 5: Portfolio-Aware Vectorization

Add later:

- shared-capital multi-asset logic
- rebalance and allocation rules
- portfolio-level vectorized attribution

The first narrow implementation of this phase has now started.

Current implemented portfolio-aware scope:

- shared-cash multi-asset execution in a dedicated vectorized portfolio engine
- same-timeframe only
- long-only only
- adapter-driven signal generation using the existing SMA and z-score vectorized adapters
- explicit allocation ownership modes: `Strategy-Owned`, `Portfolio-Owned`, and `Hybrid`
- explicit portfolio weighting modes, currently `Preserve Strategy Weights`, `Equal Weight Selected`, and `Score-Proportional`
- portfolio construction constraints including minimum active weight, maximum asset weight, and optional cash reserve
- rebalance on target-allocation change, with bounded periodic, drift-threshold, and combined periodic-plus-drift rebalance options
- equal-weight or relative-weight allocation using shared portfolio capital
- basic dashboard study-mode integration for shared-cash portfolio runs
- simple dashboard allocation controls for equal-weight, relative-weight, and fixed-weight portfolio studies
- initial portfolio ranking support through explicit `Top N`, `Score Threshold`, and `Top N Over Threshold` selection modes in `Portfolio-Owned` and `Hybrid` modes
- portfolio-owned weighting overrides that can equal-weight selected assets or weight them proportionally to score
- portfolio construction constraints that can prune tiny allocations, cap single-name concentration, and intentionally leave part of the portfolio in cash
- portfolio runs and trades persisted through the existing batch/run catalog flow
- portfolio execution available through the shared execution/orchestration layer, not just the dashboard worker path
- initial portfolio chart artifact support through Magellan snapshot export using synthetic equity-derived bars plus portfolio weight/cash panes
- initial portfolio attribution/reporting support with per-asset weights, tracking error, realized PnL, turnover, cash-weight, and exposure summaries
- built-in portfolio fallback viewer with equity, exposure, asset-weight, and attribution-table views when Magellan is unavailable

Allocation ownership rules:

- `Strategy-Owned Allocation`
  The strategy owns exposure sizing across its assets, and the portfolio layer should mostly enforce shared-cash accounting, caps, and execution.
- `Portfolio-Owned Allocation`
  The portfolio layer owns ranking, allocation, and rebalance rules, while strategies provide signals or candidates.
- `Hybrid Allocation`
  The strategy proposes weights or exposures and the portfolio layer applies caps, normalization, or risk constraints without replacing the core intent.

Design rule:

- Portfolio ranking or rebalancing must not silently override a strategy that already owns its own capital-allocation logic.
- Portfolio weighting overrides must not silently replace strategy sizing unless the user explicitly selected `Portfolio-Owned Allocation`.
- Portfolio-owned ranking and rebalance rules should only be enabled through an explicit ownership mode or user override.
- This matters especially for multi-asset strategies whose alpha depends on their internal sizing logic.

Current non-goals inside portfolio v1:

- full portfolio UI/editor
- shorting inside portfolio mode
- pending-order portfolio books
- advanced cross-asset ranking and selection rules beyond the initial explicit `Top N` and `Score Threshold` modes
- base-execution parity
- richer portfolio performance decomposition and custom report layouts beyond the initial attribution summary and fallback viewer

This phase should not be treated as "done" until:

- single-asset parity is trusted
- optimization integration is stable
- strategy capability checks are working
- the portfolio engine is available through the shared execution/orchestration layer and higher-level workflows, not just a narrow dashboard path
- richer portfolio controls and richer portfolio charting/reporting exist, including a built-in fallback when Magellan is unavailable

### Phase 6: Compiled Kernel and Performance Tuning

Only after the architecture and semantics are stable, optimize the hottest inner loops further.

Candidates:

- compiled execution kernel
- faster trade extraction
- reusable indicator caches across studies
- memory-aware batching for very large parameter grids

Some of this work has already started:

- the current vectorized engine can split large compatible parameter grids into smaller chunks using an adaptive batch-size heuristic or an explicit config override
- prepared adapter context can now be computed once for a study and reused across chunked vectorized execution

This phase is where we should profile against the local `vectorbt` reference for ideas, but still keep our own contracts and semantics.

## Non-Goals for Version 1

These should stay out of scope for the first implementation:

- rewriting every current strategy for vectorization
- full parity across all order types and execution semantics
- intrabar simulation in vectorized mode
- full portfolio-aware shared-cash vectorization
- silent behavior changes for unsupported strategies

## Remaining Parity Gap

The biggest remaining semantic gap after same-timeframe higher-timeframe support is:

- full `base_execution` parity

Today, the reference engine can generate signals on a higher timeframe while still executing against a lower base timeframe such as `1 minutes`.

That is materially different from:

- running a vectorized study directly on resampled `5m` / `15m` / `1h` bars

Both modes are useful, but they are not identical.

So the plan should be explicit:

- near term: support vectorized same-timeframe runs on higher-timeframe resampled bars
- later: add true vectorized lower-timeframe `base_execution` parity
- until then: let `Auto` use vectorized on supported same-timeframe higher-timeframe studies, but keep falling back to reference for unsupported or clearly base-execution-dependent workloads

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
