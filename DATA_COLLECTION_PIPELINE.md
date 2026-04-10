# Data Collection Pipeline

## Purpose

This document describes:

- how the data download and ingestion flow behaves today
- where the current gaps are
- what a smarter data collection pipeline should do next
- how reusable user-defined universes should fit into the pipeline
- how to handle multiple data sources without creating unnecessary complexity

This is a design and status document only. It does not imply that all of the behavior below is already implemented.

## Current Behavior

### 1. Download Source And Scope

The download path is now source-aware, but still early.

What exists today:

- a small provider registry resolves the selected source into a fetch script and default request settings
- the dashboard and scheduler both use that registry instead of hardcoding the download command inline
- the current registry contains two real providers:
  - `Massive`
  - fetch script: `scripts/fetch_massive.py`
  - default request shape: roughly `1-minute`, `2 years`
  - `Stooq`
  - fetch script: `scripts/fetch_stooq.py`
  - default request shape: roughly `1-day`, `max history`

So the active flow is no longer "completely source-unaware," but it is also not a finished multi-source engine yet. The current flow is effectively:

1. choose ticker(s) or a saved universe
2. choose a source explicitly or inherit a universe source preference
3. resolve that into a provider fetch command
4. evaluate a simple acquisition policy
5. either download, ingest an existing raw CSV, or skip because the dataset is already fresh
6. ingest into the canonical store when needed

### 2. Interactive UI Download Flow

The interactive download queue in the dashboard currently does a few useful things already:

- it creates one queue item per ticker
- it launches downloads one ticker at a time
- it writes per-ticker log files into `data/download_logs/`
- it tracks row status in the UI
- it continues to the next ticker even if one ticker fails
- it can mark failed tickers in the progress table
- it supports a resume state file per ticker under `data/download_state/`
- it records durable acquisition runs and per-symbol attempts in SQLite
- it auto-ingests successful downloads into the local DuckDB/Parquet store
- it can inherit source preference from a saved universe
- it can show acquisition history through the acquisition catalog UI
- it now has a `Force Refresh` override so a user can re-download even when the local dataset looks fresh
- dataset-level acquisition detail can now be opened directly from research-facing screens such as Orchestrate, Universe Builder, and the catalog itself
- the acquisition policy can now produce an initial refresh plan such as `skip fresh`, `incremental refresh`, `backfill`, or `full rebuild`

Current constraints:

- concurrency is effectively disabled right now
- the provider list is still small and intentionally simple
- the policy layer now has an initial suspicious-gap check for local datasets, plus first refresh planning for incremental refresh, backfill, and full rebuild
- the policy layer now supports same-source gap repair for isolated gappy datasets, including non-contiguous multi-window repair windows when needed
- the policy layer can now build broader compound repair plans that combine internal gap windows with stale-right-edge refresh and/or left-edge backfill windows instead of defaulting straight to a full rebuild
- the acquisition layer now has an initial cross-source parity check and secondary-source gap-fill path for isolated gap repair when a matching alternate source already exists locally
- the policy layer now has a first hybrid multi-step repair path for `cross-source gap-fill + primary incremental refresh` when a matching secondary source can repair internal gaps and the primary source still needs a right-edge refresh
- broader hybrid multi-step planning is still unfinished, especially for more complex mixed refresh/backfill/gap-fill combinations
- provider settings now exist as a first-class concept: non-secret provider config such as Interactive Brokers `host` / `port` / `client_id` is persisted in SQLite, while secrets such as the Massive API key are saved locally outside the code path and injected at runtime

### 3. Scheduled Download Flow

There is also a scheduler service path that launches downloads for scheduled tasks through the same provider registry.

Current strengths:

- scheduled tasks and scheduled task runs are persisted in SQLite metadata tables
- scheduled runs now continue after individual ticker failures
- scheduled runs auto-ingest successful downloads into the canonical store
- scheduled runs also record durable acquisition run and per-symbol attempt history
- scheduled tasks can now persist a `force_refresh` preference instead of always relying on freshness skip behavior

Current gap:

- scheduled acquisition is now visible and editable from the dashboard, but it still needs deeper lifecycle controls and richer run analysis over time

### 4. Download Output

The current provider fetch script currently:

- downloads bars
- writes a CSV file to `data/`
- emits JSON progress / done / error events when launched with `--progress`

The fetcher is still provider-specific, but the UI/scheduler command construction is now centralized through the provider registry instead of being hardcoded all over the application.

### 5. Database / Local Store Ingestion

Today, successful automated download **does** automatically ingest data into the local history store.

The default automated flow is now:

1. fetch raw CSV
2. normalize and ingest into the DuckDB/Parquet store
3. record acquisition metadata and attempt history

Manual CSV import still exists and is still useful for:

- one-off imports
- imported datasets from external vendors
- recovery / debugging workflows
- power-user dataset creation

### 6. Local Dataset Discovery

The current system now has both:

- file-based local dataset discovery from `data/parquet/*.parquet`
- a first acquisition catalog for dataset metadata, acquisition runs, attempts, and task-run lineage

So the system can now track, in a durable structured way:

- source/provider
- logical dataset id
- last attempt / last success / last ingest
- date coverage
- bar count
- last failure reason
- task and universe lineage

What it still does **not** do well enough yet:

- authoritative freshness scoring
- incremental vs backfill reasoning
- full gap detection and repair inside covered ranges
- parity / conflict analysis across multiple providers
- fully source-aware canonical dataset policies

### 7. Universe Layer Exists, But Is Still Early

There is now a first-class reusable `Universe Builder` tab.

Today a universe can already drive:

- data download
- scheduled data refresh
- single-strategy multi-dataset backtests
- shared-strategy portfolio studies
- a universe draft can also show an acquisition summary with freshness, coverage, recent failures, and source visibility before it is saved

The main remaining gap is deeper portfolio integration, especially:

- auto-populating more complex fixed strategy-block portfolio definitions
- richer universe-aware portfolio authoring flows

## Current Gaps

### Durable Acquisition Catalog Exists, But Is Not Finished

There is now a first acquisition catalog that answers many of the basics:

- which requested tickers were downloaded successfully
- which failed
- why they failed
- when they were last attempted
- when they were last updated successfully
- what time range is covered
- whether they were ingested into the canonical local store

What is still missing:

- smarter freshness and staleness logic
- gap-awareness inside a covered date range
- source-level parity analysis
- a deeper per-dataset / per-universe metadata review workflow from everywhere in the app

### Download And Ingest Are Unified By Default, But Still Need Smarts

The current automated pipeline now unifies:

- raw download
- canonical ingest
- catalog update

The remaining gap is not the basic wiring anymore. The remaining gap is intelligence:

- deeper incremental update detection
- deeper backfill detection
- safe overlap merging
- deeper gap detection and repair beyond the first same-source repair window
- freshness and stale-state reasoning

### Batch Outcome Summary Exists, But Needs More Analysis Views

The interactive and scheduled flows now record durable run and attempt history, but the UI still needs richer summary and comparison views that clearly surface:

- succeeded tickers
- failed tickers
- skipped tickers
- ingested tickers
- failure reasons

### No Full Multi-Source Model Yet

The pipeline is no longer completely source-naive. What exists now:

- a provider registry
- source selection in the UI
- universe source preference
- source-aware scheduled tasks
- source-aware dataset ids and acquisition metadata

What does **not** exist yet:

- broad provider coverage beyond the initial `Massive` + `Stooq` + `Interactive Brokers` set
- richer provider-specific symbol mapping layers at scale
- broader source comparison / parity tooling beyond the first local gap-fill path
- explicit merged datasets
- a broader source-aware acquisition policy layer for deeper hybrid multi-step repair coverage and more advanced provider selection

### No Canonical Freshness / Coverage Logic

The system now has a first freshness pass, but it still does not reason deeply enough about:

- whether local data already covers the requested range
- whether only a backfill is needed
- whether only an incremental update is needed
- whether gaps exist inside the covered range

### Universe Definitions Exist, But Need Deeper Workflow Integration

The reusable universe layer now exists.

The remaining work is:

- deeper universe-driven portfolio authoring
- saved reusable strategy-aware universe templates, not just editor shortcuts but actual reusable strategy-block presets attached to a universe
- richer acquisition health/reporting at the universe level
- more dashboard views for universe/task/dataset health over time
- broader cross-study/data lineage visibility from acquisition through research workflows
- stronger universe-to-task and universe-to-strategy-block shortcuts

## Desired Future Behavior

## 1. A Real Acquisition Catalog

The pipeline should have a durable catalog that tracks each logical symbol / dataset and each download attempt.

At minimum, the catalog should answer:

- `symbol`
- `source`
- `provider_symbol`
- `timeframe`
- `adjusted/unadjusted`
- `dataset_id`
- `raw artifact path`
- `canonical store path`
- `coverage_start`
- `coverage_end`
- `bar_count`
- `last_attempt_at`
- `last_success_at`
- `last_ingest_at`
- `last_status`
- `last_failure_reason`
- `ingest_status`
- `freshness_state`

Suggested status values:

- `queued`
- `running`
- `downloaded`
- `ingested`
- `partial`
- `failed`
- `stale`
- `skipped`

### Recommended Catalog Shape

This does not need to be overdesigned, but a practical v1 would likely need separate records for:

- logical assets / symbols
- source-backed datasets
- download attempts
- ingest attempts
- optional gap / parity checks

A simple mental model:

1. `Asset`
   Example: `SPY`
2. `Source Dataset`
   Example: `massive:SPY:1m:adjusted`
3. `Download Attempt`
   Example: `attempt_2026_04_09_...`
4. `Canonical Dataset`
   Example: local backtest dataset id stored in DuckDB/Parquet

## 2. Smart Download + Ingest Pipeline

The system should behave like a pipeline, not like unrelated tools.

Recommended default flow:

1. create download request
2. fetch raw source data
3. validate / normalize the result
4. ingest into the canonical local store automatically
5. update catalog state
6. surface success / failure in the UI

In other words:

- download should usually imply ingest
- manual CSV import should remain available as a fallback or power-user path, not the primary path

### Smart Ingestor Expectations

The ingestor should be able to:

- detect whether a dataset already exists
- detect existing coverage start/end
- decide whether a request is a full load, backfill, or incremental update
- append or merge bars safely by timestamp
- deduplicate overlaps
- mark gaps when they remain
- mark datasets as stale or fresh
- refuse to silently corrupt existing canonical data

## 3. Universe Builder

The platform should support a first-class `Universe Builder` tab.

A universe is a reusable named collection of assets or datasets that can be used by:

- data download requests
- scheduled data refresh tasks
- single-strategy multi-asset backtests
- portfolio construction
- optimization, walk-forward, and Monte Carlo studies derived from those runs

Important design rule:

- a universe is not a portfolio
- a universe is an input scope
- portfolio construction still decides strategy blocks, weighting, ranking, rebalance, and capital rules

### Why This Matters

This gives the user a much simpler mental model:

- `Backtest -> strategy -> universe`
- `Portfolio -> strategy/s -> universe`
- `Download -> source -> universe`
- `Schedule -> source -> universe`

The user should still be allowed to manually choose assets when needed, but universes should become the fast default for repeated workflows.

### Recommended Universe Definition

A universe record should be able to store:

- `universe_id`
- `name`
- `description`
- `asset_ids` or `dataset_ids`
- optional source preference
- optional tags such as `equities`, `ETFs`, `momentum`, `watchlist`
- created / updated timestamps

Universes should be lightweight and reusable. They do not need to encode portfolio logic.

### Universe Resolution Rules

At launch time, a universe should resolve into concrete dataset selections.

Examples:

- for download: provider symbols to fetch
- for ingestion: source datasets to normalize / update
- for backtest: datasets to run independently or as a portfolio input set
- for scheduling: the symbol list attached to the scheduled task

The user should still be able to override this manually.

### Suggested UI Uses

The `Universe Builder` tab should allow the user to:

- create a named universe
- add / remove assets manually
- save and reuse universes later
- optionally clone a universe from the current selection
- use a universe in download, scheduling, backtest, and portfolio flows

This would remove a lot of repetitive asset picking from the current UX.

## 3. Better Batch Queue Visibility

When a user downloads many tickers at once, the system should clearly show:

- which are queued
- which are running
- which succeeded
- which failed
- why they failed
- which were ingested successfully
- which completed only partially

The queue should continue processing the rest of the batch after individual failures unless the user explicitly requests fail-fast behavior.

Recommended batch summary at completion:

- `total requested`
- `downloaded`
- `ingested`
- `failed`
- `partial`
- `skipped`
- a compact list of failed symbols with reasons

## 4. Source-Aware Data Management

The system should treat source as a first-class concept.

Examples:

- Massive
- Alpaca
- IBKR
- CSV imports
- future broker/vendor feeds

Each dataset should know where it came from.

That means the catalog should preserve:

- source name
- provider-specific symbol
- source-specific settings
- download timestamps
- coverage and provenance

## Recommendations For Multiple Sources

## Short Answer

Yes, the UI should expose source selection.

No, sources should **not** be silently merged by default.

Maybe, later, the system can support explicit gap-fill or source-comparison workflows.

## Recommended Default Model

The simplest model that stays clean is:

1. user chooses `source`
2. user chooses `asset / ticker`
3. system downloads raw source data
4. system ingests it into a source-backed canonical dataset

That keeps provenance clear and avoids hidden mixed-source datasets.

## Why Automatic Merge Should Not Be The Default

Different providers can disagree in meaningful ways:

- session boundaries
- timestamps
- missing bars
- corporate action handling
- adjusted vs unadjusted values
- bad prints / vendor cleanup differences
- volume differences

If the system silently mixes them, then later it becomes hard to answer:

- which bar came from which provider
- why a backtest changed
- whether the merged dataset is trustworthy

So the default should be:

- one canonical dataset should come from one chosen source

## Better Multi-Source Progression

### Phase 1: Source Selection

This phase has now started.

What exists today:

- UI source selectors
- provider registry plumbing
- universe source preference
- source-aware scheduled tasks and acquisition history
- initial acquisition policy decisions for `download`, `ingest existing CSV`, and `skip fresh`

What is still unfinished inside this phase:

- broader provider coverage
- richer provider metadata and symbol translation
- deeper source-aware acquisition policy rules

Keep datasets source-specific.

Examples:

- `massive:SPY:1m`
- `alpaca:SPY:1m`
- `csv:SPY:1m`

### Phase 2: Source Comparison / Parity Check

Add a tool that can compare overlapping ranges across sources and report:

- matching bar count
- missing bar count
- price deltas
- volume deltas
- session differences

This is useful and relatively low-risk.

### Phase 3: Explicit Gap Fill

Only after source comparison exists, support:

- `fill missing bars in primary source using secondary source`

This should be explicit, auditable, and provenance-preserving.

### Phase 4: Derived Merged Datasets

If needed later, support derived datasets that intentionally combine sources.

Important rule:

- merged datasets should be explicit derived artifacts, not hidden defaults

They should record:

- primary source
- fallback source
- merge time
- merge rules
- parity / tolerance rules used

## Recommended UI Direction

The UI should eventually support a simple acquisition workflow like this:

### Download / Ingest Request

- `Source`
- `Ticker(s)`
- `Timeframe`
- `Date range`
- `Adjusted / unadjusted`
- `Ingest automatically` toggle
- `Update existing dataset if present` toggle

### Dataset Catalog View

For each tracked dataset, show:

- source
- ticker
- local dataset id
- coverage start/end
- bar count
- last updated
- freshness
- last status
- last failure reason
- raw file presence
- canonical ingest presence
- which universes currently include the dataset

### Universe Catalog View

The UI should eventually expose a universe catalog that shows:

- universe name
- asset count
- source preference if any
- last download attempt across the universe
- last successful refresh
- freshness summary
- failed assets in the most recent refresh

### Backtest / Portfolio Launch View

Backtest and portfolio setup should allow either:

- manual asset / dataset selection
- or selecting a saved universe

That way a user can quickly launch:

- one strategy across a chosen universe
- or one or more strategies over that same universe inside a portfolio workflow

### Batch Run View

For each batch request, show:

- all requested tickers
- per-ticker status
- failure reasons
- ingest results
- final batch summary

### Scheduled Task View

Scheduled acquisition should be visible as a first-class catalog-backed workflow.

The UI should expose:

- scheduled task definition
- source
- selected universe or manual symbol list
- frequency / timing
- active / paused / stopped status
- last run outcome
- next run time
- edit / delete / enable / disable controls
- historical run log list

This is especially important because the scheduler is already partially persisted in SQLite. The next step is to make it fully visible and manageable from the dashboard.

## Recommended Architecture Direction

To avoid overcomplicating this, the internal model should stay linear:

1. `Provider Adapter`
2. `Raw Artifact`
3. `Normalizer`
4. `Canonical Ingestor`
5. `Catalog Update`

### Provider Adapter

Responsible for:

- symbol translation
- API authentication
- provider-specific fetch logic
- provider-specific pagination / rate limiting

Current implementation note:

- the provider adapter layer has now started as a registry plus fetch-command builder
- it currently contains one real provider: `Massive`
- it is not yet a full provider abstraction with comparison, reconciliation, or merge policies

### Raw Artifact

Responsible for:

- saving the untouched source file or raw normalized response
- preserving provenance for debugging and auditability

### Normalizer

Responsible for:

- converting source output into canonical OHLCV bars
- standardizing timestamp and schema
- identifying bad rows, duplicates, and coverage

### Canonical Ingestor

Responsible for:

- writing into DuckDB/Parquet
- append / replace / merge logic
- dedupe
- gap tracking

### Catalog Update

Responsible for:

- durable status updates
- success / failure audit trail
- coverage metadata
- freshness metadata
- universe membership references
- scheduled task and task-run lineage

## Recommended Phased Rollout

## Phase 1: Catalog + Auto-Ingest

Highest-value near-term changes:

- add a durable acquisition catalog
- auto-ingest successful downloads into DuckDB/Parquet
- record per-ticker failure reasons
- make batch queues finish all tickers even when some fail
- show coverage start/end and last updated in the UI
- add first-class visibility for scheduled tasks and scheduled task runs that already live in SQLite

This is the most important phase.

## Phase 2: Universe Builder + Source-Aware Downloading

This phase has partially landed:

- the `Universe Builder` tab exists
- download, scheduling, backtest, and shared-strategy portfolio flows can target a saved universe
- source selection exists in the UI
- `Massive`, `Stooq`, and `Interactive Brokers` are now routed through the initial provider registry
- the Interactive Brokers provider uses the official TWS / IB Gateway API with conservative client-side chunking for minute history; the current official docs confirm that the old hard limits for `1 min` and larger bars were lifted, but requests remain subject to soft throttling and earliest history still varies by contract
- acquisition catalog provenance fields exist
- an initial acquisition policy layer exists
- the strategy-block editor can now seed one block per dataset or expand the current strategy-template set across a selected universe

Still unfinished inside this phase:

- more providers
- saved reusable strategy-aware universe templates
- a broader source-aware acquisition policy layer for deeper hybrid multi-step planning
- broader acquisition health/reporting and source-aware UI review/comparison
- broader cross-study/data lineage visibility

## Phase 3: Smart Updates

- incremental updates
- backfill handling
- dedupe / overlap merging by timestamp
- gap detection
- stale/fresh labeling
- universe-level freshness summaries
- universe-level batch refresh reporting

## Phase 4: Source Comparison And Gap Fill

- parity report across sources
- deeper cross-source gap-fill and hybrid repair flow beyond the first isolated and hybrid incremental paths
- explicit derived merged datasets if needed

## Decisions / Recommendations

### Recommended Decisions

- keep source selection explicit in the UI
- keep downloads and ingest as one default pipeline
- keep raw artifacts and canonical datasets distinct
- add a reusable `Universe Builder` instead of forcing repeated manual asset selection
- let universes drive backtest, portfolio, download, and schedule workflows
- build a durable catalog before building fancy source merge logic
- do not silently combine sources by default
- make gap-fill and source merge explicit later features
- keep scheduled tasks and task runs as catalog-backed first-class objects, not just background scripts

### Explicitly Deferred For Later

- saved reusable strategy-aware universe templates and presets attached to universes
- richer acquisition health/reporting
- more dashboard views for universe/task/dataset health over time
- broader cross-study/data lineage visibility
- strategy health/decay tracking over time so validated strategies and portfolios can be monitored against their historical baselines

The following multi-source capabilities are still intentionally deferred and should remain documented as unfinished:

- full multi-source provider abstraction beyond the initial registry
- source-comparison and parity tooling
- gap-fill from a secondary provider
- explicit derived merged datasets
- automatic source blending or silent fallback between providers

### Not Recommended As A Default

- automatic source blending during download
- silent fallback to another source without provenance
- source merging before parity / comparison tooling exists

## Practical Summary

The current system already has the beginning of a usable downloader:

- per-ticker queueing
- log files
- failure marking
- manual CSV import

But it is still missing the most important thing:

- a unified, durable acquisition pipeline that knows what exists locally, what failed, what is fresh, what is stale, and what has actually been ingested into the local backtest store

The best next design direction is:

1. add a data acquisition catalog
2. make successful downloads auto-ingest by default
3. add a reusable `Universe Builder` so the same asset scope can drive download, scheduling, backtest, and portfolio flows
4. make source a first-class field
5. postpone cross-source merge logic until parity / gap analysis exists
