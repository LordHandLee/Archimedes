# Quant Backtest Engine - Target System Design

## Purpose

This project is evolving from a single-asset backtest runner into a flexible research platform for strategy development, validation, portfolio construction, and high-performance analysis.

The design must support:

- simple single-asset backtests with minimal friction
- portfolio backtests spanning one or more strategies and one or more assets
- a complete research pipeline from parameter search through robustness testing
- snapshot-based chart review with a custom C++ stock chart visualizer
- a future vectorized execution path for large-scale research workloads

## Core Principles

- Deterministic, reproducible runs with stable hash-based identifiers.
- Separation of concerns across market data, metadata/artifacts, execution, research workflows, and visualization.
- Single-asset workflows remain first-class even as the engine expands to portfolio scope.
- The simplest valid portfolio is still supported: one portfolio, one strategy, one asset.
- One canonical run should feed many downstream consumers: UI, charts, optimization reports, walk-forward reports, and Monte Carlo analysis.
- The event-driven engine remains the reference source of truth for correctness.
- A future vectorized engine is an acceleration layer, not an excuse to weaken execution realism or auditability.

## Flexibility Model

The system must be flexible without forcing the user into a more complex workflow than necessary.

### Single-Asset Backtesting

Single-asset backtesting must remain directly supported in both the engine and the UI.

Conceptually, a single-asset run is represented as the smallest portfolio configuration:

- Portfolio
- one Strategy Allocation
- one Asset Allocation

This means we keep one unified architecture while preserving the current ease of use.

### Portfolio Backtesting

The target hierarchy is:

`Portfolio -> strategy/s -> asset/s`

A portfolio may contain:

- one strategy on one asset
- one strategy across multiple assets
- multiple strategies on one asset
- multiple strategies across multiple assets

This structure enables:

- portfolio-level capital allocation
- strategy-level weight or capital budgets
- asset-level routing and exposure tracking
- consolidated portfolio metrics
- attribution by portfolio, strategy, and asset

## Major Components

### Data Pipeline

- Fetchers ingest raw market data from CSV or supported APIs.
- A cleaner normalizes all bars to a canonical OHLCV schema with UTC timestamps.
- Canonical data is stored in Parquet and registered through DuckDB metadata.
- Resampling remains centralized so all engines and reports consume the same market data definitions.

### Historical Store

- DuckDB plus Parquet remains the system of record for historical bar data.
- Data is organized by dataset and asset, with consistent timeframe derivation rules.
- The store must evolve to support loading multiple assets efficiently for portfolio and vectorized workloads.

### Metadata, Results, and Artifacts Catalog

SQLite remains the metadata and artifact index.

It should track at minimum:

- backtest runs
- batches and optimization jobs
- walk-forward studies
- Monte Carlo studies
- portfolio definitions
- chart snapshots
- generated reports and visual assets

The catalog should preserve strong lineage so every downstream artifact can be traced back to:

- data selection
- engine mode
- strategy version
- parameter set
- portfolio definition
- run timestamps

### Execution Architecture

The system should explicitly support two execution modes over time.

#### 1. Reference Event-Driven Engine

This is the canonical engine for correctness and feature completeness.

Responsibilities:

- realistic order handling
- bar-by-bar or event-by-event strategy lifecycle
- slippage, fees, shorting, and fill rules
- trade log generation
- equity curve generation
- snapshot artifact generation

This engine is the authority used to validate new strategy behavior and confirm parity for any accelerated execution mode.

#### 2. Future Vectorized Engine

This is a planned acceleration layer for compatible workloads such as:

- parameter sweeps
- repeated single-strategy research runs
- broad signal evaluation over large arrays

Goals:

- reduce dependence on slow run-per-process concurrency
- leverage array-based computation with NumPy and related techniques
- support thousands of compatible backtests in seconds when assumptions permit

Constraints:

- we are not depending on `vectorbt`
- we may borrow the performance philosophy, not the library
- vectorized results must be validated against the reference event-driven engine
- strategies that require path-dependent or highly stateful execution may remain on the reference engine
- near-term vectorized support may include same-timeframe runs on resampled higher-timeframe bars
- full lower-timeframe `base_execution` parity for vectorized runs is still future work and must be tracked as a separate milestone

The long-term design should let the orchestrator choose the engine mode based on strategy compatibility and requested workflow.

Detailed execution-layer design lives in [VECTORIZED_ENGINE.md](/home/ethan/quant_backtest_engine/VECTORIZED_ENGINE.md), including the hybrid-engine model and the user-facing `Auto`, `Reference`, and `Vectorized` execution modes.

### Portfolio Layer

The current single-asset broker must evolve into a portfolio-aware execution layer.

Target responsibilities:

- maintain cash and equity at the portfolio level
- track positions per strategy and per asset
- support allocation rules between strategies
- support aggregation of correlated and overlapping exposures
- generate portfolio-level, strategy-level, and asset-level performance metrics
- support portfolio snapshots and reporting

The design should distinguish between:

- execution state
- allocation state
- performance attribution state

That separation will make it easier to support both simple and complex portfolio definitions without turning the broker into a monolith.

The design must also distinguish between allocation ownership modes:

- `Strategy-Owned Allocation`
  The strategy decides its own exposure, target weight, or capital usage across its attached assets. The portfolio layer should mainly provide shared-cash accounting, exposure caps, and execution.
- `Portfolio-Owned Allocation`
  The strategy provides signals, ranks, or candidate entries, and the portfolio layer decides ranking, sizing, and rebalance behavior across assets and strategies.
- `Hybrid Allocation`
  The strategy proposes exposures or weights, and the portfolio layer applies normalization, caps, or risk constraints without replacing the strategy's intent.

Initial portfolio construction modes should stay explicit and simple:

- ranking modes such as `Top N`, `Score Threshold`, and `Top N Over Threshold`
- weighting modes such as `Preserve Strategy Weights`, `Equal Weight Selected`, and `Score-Proportional`
- allocation constraints such as `Min Active Weight`, `Max Asset Weight`, and `Cash Reserve Weight`
- rebalance modes such as `On Change`, `On Change + Periodic`, `On Change + Drift Threshold`, and `On Change + Periodic + Drift Threshold`

Design rule:

- Portfolio ranking or rebalancing must never silently override a strategy that already owns its own allocation logic.
- Portfolio weighting overrides must never silently replace strategy-owned sizing unless the user explicitly chooses `Portfolio-Owned Allocation`.
- If a user wants portfolio-level ranking or rebalancing to replace strategy-owned sizing, that must be an explicit mode choice, not an implicit side effect.
- This distinction is critical for multi-asset strategies whose alpha logic already includes capital-allocation behavior.

### Charting and Visualization

Magellan, the custom stock chart visualizer written in C++, is the planned primary charting tool for this project.

Magellan should support two major usage modes inside the platform:

- artifact mode for reviewing completed backtests, walk-forward studies, Monte Carlo outputs, paper runs, and live runs
- market mode for exploring a ticker with historical bars, live updates, and user-selected indicators

The engine should not rerun a strategy just to open a chart. Instead:

- a completed run produces a `ChartSnapshot` artifact keyed by `run_id`
- Magellan reads the snapshot directly
- the UI opens Magellan against the saved artifact

The chart snapshot contract is defined separately in [CHART_SNAPSHOT_SCHEMA.md](/home/ethan/quant_backtest_engine/CHART_SNAPSHOT_SCHEMA.md).

The Magellan process, launch, and live-session integration model is defined in [MAGELLAN_INTEGRATION.md](/home/ethan/quant_backtest_engine/MAGELLAN_INTEGRATION.md).

Design rules for artifact mode:

- chart rendering is read-only
- chart open must not trigger strategy recomputation
- chart artifacts must work for both single-asset and future portfolio-aware views
- initial portfolio-aware chart artifacts may use synthetic equity-derived bars with cash and weight panes until richer native portfolio chart layouts exist
- Magellan remains the primary chart consumer, but the main UI should also provide a built-in fallback viewer for portfolio runs when Magellan is unavailable
- Matplotlib may remain as a fallback or validation reader, but it is no longer the strategic endpoint

Design rules for portfolio reporting:

- portfolio runs should expose asset-level attribution, not just a single portfolio equity curve
- initial reporting should include asset weights, target weights, tracking error, realized PnL, turnover, cash weight, and exposure summaries
- the main UI should expose a built-in portfolio chart/report view even if Magellan is unavailable
- richer decomposition can come later, but basic attribution must be available once portfolio backtesting exists

Design rules for market mode:

- clicking a ticker symbol in the UI should open that symbol in Magellan
- the chart should load historical bars first, then continue with live updates
- the user should be able to add or remove indicators from a selectable library
- indicator overlays should be computed without rerunning a strategy backtest
- the same market view should later support paper-trade and live-trade overlays
- when this UI launches Magellan, it should own that process, track its PID, and terminate it on clean UI shutdown

### Paper Engine and Live Execution

The platform must include execution pathways beyond research-only backtesting.

#### Paper Engine

The paper engine is the validation stage between research and live deployment.

Responsibilities:

- deploy a selected strategy or portfolio into a simulated execution environment
- subscribe to live or delayed market data
- route orders to a paper broker or internal paper broker model
- record orders, fills, positions, equity, and strategy health
- persist paper-run artifacts for review in Magellan and the main UI

The paper engine should make it easy to compare:

- expected research behavior
- real-time paper behavior
- slippage, fill, and signal timing assumptions

#### Live Deployment

The live deployment layer is the controlled production pathway after paper validation.

Responsibilities:

- promote a paper-tested strategy or portfolio into live execution
- connect to broker or execution adapters
- apply account, risk, and exposure controls
- monitor health, orders, fills, positions, and PnL
- support pause, disable, and kill-switch operations

Design rule:

- no strategy should move from research straight to live deployment without a paper or forward-testing stage unless explicitly overridden

### Research and Validation Pipeline

The research pipeline must be completed in this order because each stage refines the confidence in the prior stage.

#### 1. Parameter Optimization

Purpose:

- search candidate parameter sets
- rank results by selected metrics
- identify promising but not obviously unstable configurations

Detailed version 1 design lives in [PARAMETER_OPTIMIZATION.md](/home/ethan/quant_backtest_engine/PARAMETER_OPTIMIZATION.md).

Outputs:

- optimization batches
- parameter heatmaps and rankings
- candidate parameter sets for deeper validation

#### 2. Walk-Forward Optimization

Purpose:

- test whether optimized parameters generalize across sequential train/test windows
- reduce the chance of selecting a parameter set that only fit one historical segment

Detailed version 1 design lives in [WALK_FORWARD_OPTIMIZATION.md](/home/ethan/quant_backtest_engine/WALK_FORWARD_OPTIMIZATION.md).

Outputs:

- train/test window definitions
- per-window best parameters
- stitched out-of-sample performance
- walk-forward summary metrics and artifacts

#### 3. Monte Carlo Simulation

Purpose:

- stress-test robustness after a strategy or portfolio has already survived optimization and walk-forward validation
- evaluate the distribution of possible outcomes under resampled trade sequences or equity path perturbations

Detailed version 1 design lives in [MONTE_CARLO_SIMULATION.md](/home/ethan/quant_backtest_engine/MONTE_CARLO_SIMULATION.md).

Outputs:

- drawdown and return distributions
- confidence intervals
- probability-oriented risk summaries

Monte Carlo is not a replacement for optimization or walk-forward analysis. It is a later-stage robustness tool.

In practice, the broader research-to-live pipeline is:

1. baseline backtest and sanity checks
2. parameter optimization
3. walk-forward optimization
4. Monte Carlo simulation
5. portfolio construction and sizing if applicable
6. paper or forward test
7. live deployment
8. monitoring and revalidation

For multi-strategy or multi-asset deployment, portfolio construction and sizing should sit between validation and paper/live deployment.

## User Interface Model

The UI should separate workflows into clear tabs while reusing the same underlying catalog and artifact system.

Recommended top-level tabs:

- `Market`
- `Backtest`
- `Optimization`
- `Walk Forward`
- `Monte Carlo`
- `Portfolio`
- `Paper Engine`
- `Live Deployment`
- `Runs / Artifacts`

Design rules:

- Walk-forward optimization and Monte Carlo should be separate tabs.
- The `Market` tab should let the user browse or search ticker symbols and open them in Magellan.
- The `Market` tab should combine historical data with live updates and allow user-selected indicators on the chart.
- Single-asset backtests should remain easy to launch from the `Backtest` tab.
- The `Portfolio` tab should allow the user to define one-strategy/one-asset cases as well as richer multi-strategy, multi-asset structures.
- The `Paper Engine` tab should allow validated strategies or portfolios to be deployed into paper testing and monitored in real time.
- The `Live Deployment` tab should be a separate controlled area for live promotion, runtime controls, and health monitoring.
- All tabs should read and write shared run metadata, snapshots, and study records rather than inventing separate storage paths.

## Typical Workflows

### Single-Asset Backtest

1. Select one dataset or asset and a timeframe.
2. Select one strategy and a parameter set.
3. Run a backtest through the reference engine.
4. Persist results, trades, metrics, and chart snapshot artifacts.
5. Review performance and open Magellan.

### Portfolio Backtest

1. Define a portfolio.
2. Attach one or more strategies.
3. Attach one or more assets to each strategy as allowed by the strategy design.
4. Choose the allocation ownership mode for the portfolio or strategy block.
5. Define allocation, ranking, rebalance, and risk rules that are valid for that ownership mode.
6. Run the portfolio through the portfolio-aware engine.
7. Persist portfolio, strategy, and asset attribution outputs.

Portfolio rule:

- If a strategy already contains meaningful asset-allocation logic, the default should be `Strategy-Owned Allocation` or `Hybrid Allocation`, not forced portfolio-level ranking or equal weighting.

### Research Pipeline

1. Run parameter optimization to identify candidates.
2. Promote selected candidates into walk-forward studies.
3. Promote robust walk-forward candidates into Monte Carlo analysis.
4. Compare results at the strategy and portfolio level before adoption.

### Market Analysis

1. Search for or click a ticker symbol.
2. Open the ticker in Magellan.
3. Load historical data for the selected timeframe.
4. Continue updating the chart with live market data.
5. Add or remove indicators from the chart workspace as needed.

### Paper Testing

1. Select a validated strategy or portfolio candidate.
2. Deploy it to the paper engine.
3. Monitor orders, fills, positions, equity, and logs in real time.
4. Review the run in Magellan with execution overlays.
5. Decide whether the strategy is ready for live promotion.

### Live Deployment

1. Promote a paper-tested strategy or portfolio.
2. Confirm deployment account, risk limits, and execution settings.
3. Launch the live deployment.
4. Monitor live health, orders, positions, and PnL.
5. Pause or disable the deployment if risk or behavior deviates from expectations.

## Extensibility Hooks

- New data sources: implement a fetcher that emits normalized canonical data.
- New strategies: implement the strategy contract once and allow the orchestrator to decide whether it is event-driven only or vectorization-compatible.
- New execution models: add them behind a stable orchestration interface, not by bypassing the catalog and artifact pipeline.
- New portfolio policies: extend allocation, exposure, and attribution layers without breaking simple single-asset usage.
- New visualizers: consume `ChartSnapshot` artifacts without requiring strategy reruns.
- New market feeds: add historical and live data adapters that can feed the `Market`, `Paper Engine`, and `Live Deployment` sections.
- New broker adapters: add paper or live execution connectors without changing the higher-level workflow model.

## Testing and Validation Strategy

- Unit tests should cover accounting, metric correctness, run identity stability, and snapshot schema validity.
- Integration tests should confirm deterministic single-asset and portfolio runs on fixed data.
- Parity tests should compare vectorized-engine outputs against the reference event-driven engine for supported strategy classes.
- Walk-forward and Monte Carlo studies should be testable as first-class workflow artifacts, not ad hoc scripts.
- Chart snapshot tests should verify that the C++ visualizer can open generated artifacts without recomputation.

## Near-Term Architectural Priorities

1. Update the system model from single-asset only to flexible portfolio-aware design while preserving the current single-asset workflow.
2. Formalize `ChartSnapshot` production in the backtest pipeline and make Magellan the primary chart consumer.
3. Add a ticker-driven market workspace with historical plus live charting and selectable indicators.
4. Add a paper engine deployment and monitoring workflow.
5. Add a separate live deployment and monitoring workflow.
6. Finish parameter optimization as a durable workflow with cataloged outputs.
7. Add walk-forward optimization as the next research stage.
8. Add Monte Carlo simulation after walk-forward is in place.
9. Design and phase in a vectorized execution architecture for compatible workloads.

## Summary

The target platform remains practical for simple one-strategy, one-asset testing, but it must scale upward into a portfolio research system with:

- Magellan-based historical and live charting
- optimization and robustness workflows
- paper testing and live deployment pathways
- portfolio-aware execution
- and a future vectorized engine for high-throughput research

The key design decision is to make the broader architecture more capable without sacrificing the simplicity of the current single-asset use case.
