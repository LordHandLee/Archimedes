# Parameter Optimization Version 2 Roadmap

## Purpose

This document outlines the recommended version 2 plan for the parameter optimization module, plus later upgrade paths beyond version 2.

Version 1 is intentionally simple and durable. Version 2 should improve how we interpret and promote parameter candidates without turning the system into an academic science project.

This roadmap stays separate from walk-forward optimization. Optimization is still for in-sample search and candidate formation. Walk-forward remains the out-of-sample validation stage.

## Relationship to Version 1

Version 1 gives us:

- deterministic grid search
- multi-asset aggregation
- a simple robustness score
- heatmap slices
- top-N tables
- boxplots
- manual candidate promotion

Version 2 should build on that foundation rather than replacing it.

The main upgrade in version 2 is this:

- move from ranking isolated parameter points
- toward identifying and promoting stable parameter regions

## Recommended Start Criteria

Do not start version 2 until version 1 is stable enough to trust.

Recommended gates:

- optimization studies run reliably end to end
- atomic runs are fully stored and queryable
- aggregate rankings are durable in SQLite
- heatmap slices and top-N inspection are usable in the UI
- manual promotion into walk-forward is already possible

If those are not true yet, version 2 will add complexity too early.

## Version 2 Goals

- detect broad robust regions instead of relying on single-point winners
- identify multiple profitable parameter behaviors when they exist
- improve candidate promotion for downstream walk-forward testing
- make the parameter surface explorer more interactive and more comparative
- keep the workflow explainable to a human user

## Recommended Version 2 Scope

These are the best solid choices for version 2.

### 1. Robust Region Detection

This should be the first major version 2 feature.

Goal:

- detect connected areas of consistently decent performance

Recommended default method:

- threshold the aggregated surface by a selected metric such as `robust_score`
- run connected-component or flood-fill detection on the thresholded grid

Why this is the best first choice:

- simple to implement
- easy to explain
- fits naturally with grid-search heatmaps
- directly supports the "plateaus over peaks" philosophy

Recommended default workflow:

1. Build the aggregated parameter surface for a chosen timeframe and fixed-value slice.
2. Normalize the metric selection if needed.
3. Apply a threshold such as:
   `robust_score >= study_percentile_80`
   or
   `robust_score >= absolute_threshold`
4. Find connected regions on the surviving grid cells.
5. Rank regions by size, median score, and minimum score.

Recommended region outputs:

- `region_id`
- parameter bounds
- cell count
- coverage ratio
- median robust score
- minimum robust score
- best point in region

Recommended promotion rule:

- prefer candidates from the best broad region
- do not promote an isolated one-cell spike unless manually justified

### 2. Clustering of Profitable Zones

This should be the second version 2 feature, after region detection.

Goal:

- identify distinct profitable behaviors when the parameter space contains multiple good zones

Recommended default method:

- filter to high-quality parameter sets
- standardize numeric parameter dimensions
- run `DBSCAN`

Why `DBSCAN` is a good choice here:

- no need to pre-specify cluster count
- handles irregular shapes better than `KMeans`
- naturally leaves noisy points unclustered

Important design rule:

- clustering should be descriptive, not the primary ranking mechanism

Use clustering to answer:

- are there multiple good families of parameter behavior?
- are those families compact or diffuse?
- should we promote one candidate from each family into walk-forward?

Do not use clustering to replace the basic robustness score.

### 3. Candidate Promotion Workflow

Version 2 should formalize candidate promotion.

Instead of promoting:

- the single top-ranked parameter set

We should promote:

- 1 to 3 representatives from the strongest robust region
- optionally 1 representative from each clearly distinct profitable cluster

Recommended candidate types:

- `region_center`
- `region_best_point`
- `cluster_representative`
- `manual_override`

This gives the later walk-forward module better inputs without exploding the candidate count.

### 4. Enhanced Parameter Surface Explorer

Version 2 should improve the UI without making it visually noisy.

Recommended additions:

- region-overlay toggle on heatmaps
- cluster-overlay toggle on filtered views
- compare mode across assets
- compare mode across timeframes
- percentile-threshold slider for region detection
- click-through from heatmap cell to asset distribution view

Keep the explorer grounded in 2D slices. Do not jump to 3D plots as a default interface.

### 5. Richer Aggregate Statistics

Version 2 should add a few stronger summary statistics, but only ones that are easy to understand.

Recommended additions:

- `sharpe_p25`
- `return_p25`
- `drawdown_p75`
- `asset_win_rate`
- `asset_fail_count`

Why:

- percentiles are often more informative than means
- they help distinguish a broad plateau from a fragile edge case

Recommended rule:

- keep the version 1 `robust_score` stable
- add a version 2 score only if it clearly improves decisions
- always version the formula

## Recommended Data Model Additions

Version 2 should extend the optimization storage model with explicit region and candidate tables.

### `optimization_regions`

Purpose:

- store robust regions detected from aggregated study surfaces

Suggested fields:

- `study_id`
- `timeframe`
- `slice_key`
- `metric_name`
- `threshold_type`
- `threshold_value`
- `region_id`
- `bounds_json`
- `cell_count`
- `coverage_ratio`
- `median_score`
- `min_score`
- `best_param_set_id`

### `optimization_region_members`

Purpose:

- map each aggregated parameter set into a detected region

Suggested fields:

- `study_id`
- `timeframe`
- `slice_key`
- `region_id`
- `param_set_id`

### `optimization_clusters`

Purpose:

- store optional cluster summaries for filtered profitable zones

Suggested fields:

- `study_id`
- `timeframe`
- `metric_name`
- `filter_rule`
- `cluster_id`
- `member_count`
- `params_center_json`
- `median_score`
- `notes`

### `optimization_candidates`

Purpose:

- store promoted candidates for downstream validation

Suggested fields:

- `candidate_id`
- `study_id`
- `timeframe`
- `param_set_id`
- `source_type`
- `source_region_id`
- `source_cluster_id`
- `promotion_reason`
- `status`

## Recommended UI Additions

Version 2 optimization UI should add structure, not clutter.

### Region Summary Panel

Show:

- region id
- parameter bounds
- cell count
- median score
- minimum score
- recommended representative candidate

### Cluster Summary Panel

Show:

- cluster id
- member count
- parameter center
- score range
- representative candidate

### Compare Mode

Allow side-by-side comparison of:

- asset vs asset
- timeframe vs timeframe
- metric vs metric

This should stay table-and-heatmap based.

### Candidate Queue

Allow the user to:

- add a candidate to a validation queue
- mark a candidate as promoted to walk-forward
- record notes on why a candidate was chosen

That creates a clean bridge to the later validation workflow.

## Recommended Version 2 Workflow

1. Run a version 1 optimization study.
2. Aggregate parameter-set metrics.
3. Detect robust regions on the aggregated surface.
4. Review the region summary and region overlays.
5. Optionally cluster the filtered high-quality parameter sets.
6. Promote a small candidate set from the strongest region or from distinct clusters.
7. Hand those candidates off to walk-forward validation.

This keeps optimization and walk-forward separate while making the handoff much better.

## Version 2 Non-Goals

These should still stay out of scope for version 2:

- live adaptive reoptimization
- automatic regime switching in production
- Bayesian optimization as the default search path
- genetic algorithms as the default search path
- blending walk-forward execution into the optimization job
- automatic deployment of selected candidates

## Potential Upgrades Beyond Version 2

These are strong future options once versions 1 and 2 are working well.

### 1. Advanced Search Methods

- Bayesian optimization for very large parameter spaces
- genetic algorithms for expensive irregular search spaces
- coarse-to-fine search that starts wide and then zooms into robust regions

### 2. Smarter Region Analysis

- morphology-based smoothing before region detection
- overlap scoring between regions across studies
- region persistence analysis across different assets or timeframes

This can be powerful, but it should come after basic region detection proves useful.

### 3. Portfolio-Aware Optimization

- evaluate parameter sets against portfolio-level aggregation, not just asset-level aggregation
- score based on cross-strategy or cross-asset diversification behavior
- promote candidates that remain strong inside portfolio construction rules

### 4. Better Research UX

- interactive artifact browser for heatmaps, boxplots, and future region views
- saved user layouts
- pinned candidates and analyst notes
- export packages for later review

### 5. Performance Upgrades

- resumable optimization jobs
- more efficient batch scheduling
- vectorized evaluation for compatible strategies
- distributed execution for very large studies

These upgrades should improve throughput without changing the core study contract.

### 6. Score Model Upgrades

- multiple named score formulas
- strategy-specific score presets
- downside-focused scores
- percentile-based robustness scores

If this is added, score versioning becomes mandatory.

## Recommended Upgrade Order

The best upgrade order after version 1 is:

1. robust region detection
2. formal candidate promotion
3. cluster summaries
4. richer explorer compare modes
5. performance and execution upgrades
6. advanced search methods

This order keeps the system concrete and useful at each step.

## Decision Summary

The best version 2 plan is not "more complexity everywhere."

It is:

- detect robust regions first
- add clustering second
- formalize candidate promotion
- improve the surface explorer
- defer heavier search and performance ideas until the workflow proves itself

That gives you a strong next version without losing the simple, solid foundation of version 1.
