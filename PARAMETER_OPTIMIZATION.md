# Parameter Optimization Design

## Purpose

This document defines the version 1 parameter optimization module for the quant backtest engine.

It is intentionally practical:

- simple enough to build and use without overengineering
- strong enough to support a retail/quant hybrid research workflow
- separate from walk-forward optimization and Monte Carlo simulation

This module answers:

- which parameter sets perform well in-sample
- which parameter sets are broadly useful across assets
- which parameter areas look stable enough to promote into walk-forward validation

This module does **not** answer whether a parameter set is truly robust out of sample. That is the job of walk-forward optimization, which remains a separate module.

The recommended next-stage roadmap lives in [PARAMETER_OPTIMIZATION_V2.md](/home/ethan/quant_backtest_engine/PARAMETER_OPTIMIZATION_V2.md).

## Scope

Version 1 parameter optimization should focus on:

- deterministic grid search
- multi-asset parameter surface exploration
- full storage of all atomic runs
- aggregated ranking of parameter sets across assets
- visual inspection through 2D heatmap slices
- candidate selection for downstream walk-forward validation

Version 1 should **not** include:

- automatic robust-region detection
- clustering of profitable zones
- Bayesian optimization
- genetic algorithms
- PCA, UMAP, or other dimensionality reduction views
- walk-forward logic mixed into the optimization run itself

Those ideas are useful later, but they are not required for a solid first implementation.

Their recommended sequencing is documented in [PARAMETER_OPTIMIZATION_V2.md](/home/ethan/quant_backtest_engine/PARAMETER_OPTIMIZATION_V2.md).

## Design Goals

- Keep single-asset optimization possible, but default to multi-asset studies.
- Store the full parameter surface, not just the best result.
- Rank parameter sets by robustness, not by raw peak performance.
- Prefer broad plateaus over sharp peaks.
- Make results queryable and explainable from the UI.
- Reuse the existing run catalog and heatmap workflow where possible.

## Core Mental Model

The optimization module should behave like a queryable map of parameter behavior.

Instead of asking:

- "What is the best parameter set?"

We want to ask:

- "Which parameter structures work broadly?"
- "Which parameter sets remain decent across many assets?"
- "Where are the stable plateaus?"

That is the right retail/quant hybrid mindset.

## Recommended Version 1 Choices

These are the recommended default choices for the first durable implementation.

### Search Method

Use deterministic grid search.

Why:

- easy to reason about
- easy to cache
- easy to debug
- easy to visualize as a complete parameter surface

Do not start with Bayesian optimization or genetic algorithms.

### Study Universe

Default to multi-asset optimization.

A study should usually evaluate one strategy and one parameter grid across:

- an asset universe
- one in-sample period
- one or more timeframes

Single-asset optimization should still be supported by simply using an asset universe of size one.

### Surface Representation

Store every atomic run, then aggregate by parameter set.

For three parameters, the main visualization should be:

- X axis: parameter 1
- Y axis: parameter 2
- fixed-value selector for parameter 3

For four or more parameters, continue using 2D slices with fixed-value filters rather than trying to build 3D or 4D plots.

### Ranking Method

Rank by a robustness score, not by best Sharpe on one asset.

The default rank view should be based on aggregated cross-asset behavior.

### Visual Outputs

Use only these version 1 views:

- 2D heatmap slices
- top-N parameter table
- asset distribution boxplot for a selected parameter set
- parameter-set detail table by asset

This is enough for version 1.

## Separation from Walk-Forward

Parameter optimization and walk-forward optimization should remain separate.

Parameter optimization produces:

- in-sample parameter surface data
- aggregated rankings
- candidate parameter sets

Walk-forward will later consume those candidates and test them out of sample.

This separation matters because:

- optimization is for search
- walk-forward is for validation

Do not blend those responsibilities in version 1.

## Inputs

Each optimization study should define:

- `study_id`
- strategy name
- strategy version or code identity
- parameter grid
- asset universe
- timeframe list
- in-sample start
- in-sample end
- backtest configuration
- primary ranking metric bundle
- optional study description

Recommended assumptions:

- one strategy per study
- one asset universe per study
- one in-sample period per study

That keeps the study easy to understand and compare.

## Output Contract

Each optimization study should produce:

- all atomic runs stored in the run catalog
- aggregated results per parameter set
- ranked candidate parameter sets
- saved heatmap artifacts
- saved distribution views for selected candidates

Each promoted candidate should carry:

- `study_id`
- `param_set_id`
- parameter values
- aggregate metrics
- asset-level metric distribution
- artifact references

This is the handoff contract to the later walk-forward module described in [WALK_FORWARD_OPTIMIZATION.md](/home/ethan/quant_backtest_engine/WALK_FORWARD_OPTIMIZATION.md).

## Data Model

Version 1 should build on the existing `runs`, `batches`, and `heatmaps` storage patterns rather than replacing them.

Recommended additions:

### `optimization_studies`

Purpose:

- one row per user-defined optimization job

Suggested fields:

- `study_id`
- `created_at`
- `strategy`
- `strategy_version`
- `asset_universe_json`
- `timeframes_json`
- `in_sample_start`
- `in_sample_end`
- `params_json`
- `score_formula_version`
- `status`
- `description`

### `optimization_aggregates`

Purpose:

- one row per parameter set per study and timeframe after asset-level aggregation

Suggested fields:

- `study_id`
- `param_set_id`
- `timeframe`
- `params_json`
- `asset_count`
- `run_count`
- `median_sharpe`
- `median_total_return`
- `worst_max_drawdown`
- `sharpe_std`
- `profitable_asset_ratio`
- `robust_score`
- `rank`

Notes:

- atomic runs still live in the normal run catalog
- this table is the durable summary layer used by the optimization UI

### Artifact Handling

Version 1 can continue using saved heatmap images plus any later saved boxplot images.

Longer term, this can evolve into a more general artifact table, but that is not required for the first pass.

## Atomic Run Model

An atomic run is one complete backtest for:

- one strategy
- one parameter set
- one asset
- one timeframe
- one in-sample period

Every atomic run should be individually persisted.

This is important because it gives us:

- auditability
- rerankable studies
- asset-level inspection
- future reuse for walk-forward and portfolio research

## Aggregation Model

After all atomic runs are complete, aggregate by `param_set_id`.

Version 1 should aggregate across assets within the study.

Recommended aggregate metrics:

- `median_sharpe`
- `median_total_return`
- `worst_max_drawdown`
- `sharpe_std`
- `profitable_asset_ratio`
- `asset_count`

Recommended meanings:

- `median_sharpe`: central tendency of risk-adjusted performance
- `median_total_return`: central tendency of absolute outcome
- `worst_max_drawdown`: the ugliest drawdown seen across the asset universe
- `sharpe_std`: how inconsistent the parameter set is across assets
- `profitable_asset_ratio`: share of assets with positive total return

Why medians first:

- more resistant to one-symbol distortions
- better aligned with the goal of broad usefulness

## Recommended Robustness Score

Version 1 should rank parameter sets with a simple cross-asset robustness score.

Recommended default:

```text
robust_score =
    median_sharpe
    - 0.25 * sharpe_std
    - 0.50 * abs(worst_max_drawdown)
    + 0.20 * profitable_asset_ratio
```

Assumptions:

- `worst_max_drawdown` is stored as a decimal magnitude such as `0.18`
- `profitable_asset_ratio` is stored from `0.0` to `1.0`

Why this is a good default:

- rewards decent Sharpe across the universe
- penalizes inconsistency
- penalizes ugly tail risk
- rewards broad usefulness across assets

Important note:

- this score is a ranking tool, not a law of nature
- the coefficients should be versioned and kept stable for study comparability
- if changed later, the formula version should be recorded in study metadata

## Ranking Rules

Version 1 should use these rules:

1. Filter out invalid parameter combinations before running.
2. Run and store every valid atomic run.
3. Aggregate by `param_set_id`.
4. Rank by `robust_score`.
5. Inspect the surrounding heatmap region before promoting a candidate.

Promotion rule:

- do not automatically promote the single best point
- prefer candidates that live inside a visibly broad, decent-performing plateau

This is one of the most important decision rules in the whole module.

## Parameter Constraints

The optimization system should support strategy-specific validity constraints before execution.

Examples:

- `fast_ma < slow_ma`
- `entry_len > exit_len`
- `stop_atr > 0`

This prevents wasting time on nonsensical runs and keeps the parameter surface cleaner.

## Visual Design

Version 1 optimization UI should stay simple and concrete.

### Main Views

#### 1. Heatmap Slice

Primary use:

- explore parameter surfaces visually

Controls:

- study selector
- metric selector
- X parameter selector
- Y parameter selector
- fixed-value selector for remaining parameter(s)
- timeframe selector

Default metric view:

- `robust_score`

Other useful views:

- `median_sharpe`
- `median_total_return`
- `worst_max_drawdown`
- `profitable_asset_ratio`

#### 2. Top-N Table

Primary use:

- inspect the strongest candidate parameter sets quickly

Columns:

- rank
- parameter values
- robust score
- median Sharpe
- median return
- worst drawdown
- profitable asset ratio
- asset count

#### 3. Boxplot Across Assets

Primary use:

- inspect distribution quality for a selected parameter set

Recommended boxplots:

- Sharpe across assets
- total return across assets

This helps reveal whether one parameter set is broadly decent or carried by a few outliers.

#### 4. Selected Parameter Detail Table

Primary use:

- inspect asset-level results for one chosen parameter set

Columns:

- asset
- timeframe
- total return
- Sharpe
- max drawdown
- trade count
- run_id

## Recommended User Workflow

1. Choose a strategy, asset universe, in-sample period, timeframe set, and parameter grid.
2. Run the optimization study.
3. Review the `robust_score` heatmap first.
4. Open the top-N table.
5. Inspect the asset distribution boxplot for promising candidates.
6. Prefer candidates from broad plateaus over isolated spikes.
7. Promote a small set of candidates into walk-forward validation.

Version 1 recommendation:

- promote a short list of candidates, not just one

That keeps the later walk-forward stage honest.

## Default Candidate Selection Guidance

When deciding what to promote, prefer:

- broad plateaus over sharp peaks
- median behavior over mean behavior
- consistency over extreme best-case return
- acceptable drawdowns over flashy but fragile performance
- good-enough performance across many assets over excellent performance on one asset

This is the practical sweet spot for the platform.

## Version 1 Non-Goals

These ideas should be explicitly deferred:

- connected-region detection
- DBSCAN or KMeans clustering
- automatic profitable-zone labeling
- overlap scoring across time windows
- regime-conditioned parameter selection
- dynamic adaptive reoptimization

Those can become version 2 or version 3 features after the simpler foundation proves useful.

## Relationship to Future Walk-Forward Optimization

Walk-forward optimization is a separate module and document.

Its job will be to:

- consume promoted candidates from this module
- optimize or rank on train windows
- validate on the next out-of-sample window
- track parameter drift over time

This parameter optimization document should stop at the in-sample candidate-selection boundary.

## Relationship to Future Vectorized Execution

This design is compatible with a future vectorized engine.

That execution-layer design is defined in [VECTORIZED_ENGINE.md](/home/ethan/quant_backtest_engine/VECTORIZED_ENGINE.md).

That future engine should accelerate:

- parameter sweeps
- multi-asset study execution
- repeated compatible runs

But the optimization design itself should not depend on vectorization to be correct. The workflow and storage model should remain valid whether execution is event-driven, vectorized, or mixed.

## Summary

Version 1 parameter optimization should be:

- grid-search based
- multi-asset by default
- fully stored at the atomic-run level
- aggregated by parameter set
- ranked by a simple robustness score
- explored with 2D heatmap slices, top-N tables, and boxplots
- clearly separated from walk-forward validation

That gives the project a professional and durable optimization workflow without overbuilding the first version.
