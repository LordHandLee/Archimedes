# Walk-Forward Optimization Design

## Purpose

This document defines the version 1 walk-forward optimization module for the quant backtest engine.

Walk-forward optimization is a validation stage, not a search stage.

Its job is to answer:

- do parameter choices that looked good in-sample remain acceptable out of sample
- does strategy behavior degrade reasonably from train to test
- do selected parameters remain somewhat stable across time

This module should remain separate from the standalone parameter optimization module.

## Role in the Pipeline

Walk-forward sits after parameter optimization and before Monte Carlo simulation.

Parameter optimization helps us understand the in-sample surface and nominate candidates.

Walk-forward then tests whether those candidates, or the optimization process itself, can survive sequential out-of-sample windows.

Monte Carlo comes later and stress-tests the resulting validated trade or equity path.

## Scope

Version 1 walk-forward optimization should focus on:

- anchored walk-forward schedules
- expanding training windows
- fixed-size out-of-sample test blocks
- reuse of the parameter optimization engine on each train fold
- fold-by-fold parameter selection
- stitched out-of-sample equity and metrics
- simple parameter drift tracking

Version 1 should **not** include:

- rolling-window optimization as the default mode
- nested cross-validation
- regime-conditioned parameter switching
- adaptive weekly or daily reoptimization
- Monte Carlo inside the walk-forward module
- production auto-deployment decisions

## Design Goals

- keep the workflow easy to understand
- make train/test boundaries explicit and auditable
- favor stable parameter behavior over single-fold peaks
- preserve every fold result as a first-class artifact
- produce a stitched out-of-sample result that can be reviewed like a real validation track record

## Recommended Version 1 Choice

Use anchored walk-forward first.

That means:

- the train window expands over time
- the test window is the next fixed out-of-sample block
- each fold moves forward in chronological order

Example:

```text
Fold 1: Train 2018-01-01 -> 2020-12-31, Test 2021-01-01 -> 2021-03-31
Fold 2: Train 2018-01-01 -> 2021-03-31, Test 2021-04-01 -> 2021-06-30
Fold 3: Train 2018-01-01 -> 2021-06-30, Test 2021-07-01 -> 2021-09-30
```

Why this is the best version 1 choice:

- easy to explain
- easy to visualize
- more natural for a retail/quant hybrid workflow than more complex validation schemes
- gives a realistic view of how a strategy would have been reselected over time

## Separation from Parameter Optimization

Walk-forward optimization is a separate module, but it reuses the parameter optimization engine inside each train fold.

That means:

- parameter optimization remains the reusable search component
- walk-forward is the orchestration and validation layer around repeated train/test folds

This separation keeps the system clean:

- optimization is for finding candidate structures
- walk-forward is for checking whether those structures persist out of sample

## Inputs

Each walk-forward study should define:

- `wf_study_id`
- strategy name
- strategy version or code identity
- parameter grid or candidate set
- asset universe
- timeframe list
- fold schedule definition
- backtest configuration
- candidate selection rule
- optional study description

Version 1 recommended defaults:

- anchored schedule
- fixed out-of-sample test block length
- one strategy per study
- one asset universe per study
- one consistent execution model across all folds

Current implemented scope now also includes a narrow portfolio path:

- shared-strategy portfolio walk-forward across multiple assets
- fixed strategy-block portfolio walk-forward
- reduced-candidate support for shared-strategy portfolios
- reduced-candidate support for fixed strategy-block portfolios through promoted fixed definitions
- portfolio-specific fold analysis/reporting on saved walk-forward studies
- portfolio validation-chain views that can jump back to promoted portfolio candidates and forward to linked Monte Carlo studies
- this still depends on the vectorized portfolio backend because a reference-engine portfolio fallback has not been implemented yet

Execution-mode behavior should follow the hybrid-engine design in [VECTORIZED_ENGINE.md](/home/ethan/quant_backtest_engine/VECTORIZED_ENGINE.md).

## Candidate Source

Version 1 should support two candidate source modes:

### 1. Full Grid Per Fold

Run the normal parameter optimization process inside each training window, then choose a candidate for the next test block.

This is the most faithful walk-forward process.

### 2. Reduced Candidate Set Per Fold

Use a short list of candidates produced by the standalone parameter optimization module, then re-rank only those inside each train fold.

This is faster and easier when the full grid is large.

Recommended default:

- allow both
- use reduced candidate sets when the full grid is too expensive
- use full-grid per fold when the study size is manageable

## Candidate Selection Rule

Version 1 should not blindly choose the single highest Sharpe point in each train fold.

Recommended default selection rule:

1. Rank train-fold parameter sets by the same robustness logic used in the parameter optimization module.
2. Keep only the top candidate set or top few acceptable candidates.
3. Prefer the most stable reasonable choice, not the most extreme one.

For version 1, a practical default is:

- select the highest `robust_score` candidate from the train fold
- optionally constrain selection to a user-supplied promoted-candidate list

Later versions can add stronger plateau and region-based selection logic.

## Fold Execution Model

Each fold should run in this order:

1. Define the train start, train end, test start, and test end.
2. Run parameter optimization on the train window.
3. Select one candidate parameter set using the fold selection rule.
4. Run a test-only backtest on the next out-of-sample window using the chosen parameters.
5. Persist train results, selected parameters, test results, and fold artifacts.

This should happen for every fold in chronological order.

## Output Contract

Each walk-forward study should produce:

- a durable study record
- a fold table
- chosen parameters for each fold
- train metrics per fold
- test metrics per fold
- stitched out-of-sample equity curve
- stitched out-of-sample trade log
- parameter drift summary

This output becomes the default source for later Monte Carlo analysis.

## Fold Metrics

Each fold should save at minimum:

- selected params
- train robust score
- train median Sharpe or selected ranking metrics
- test total return
- test Sharpe
- test max drawdown
- test trade count
- test win rate if available

Useful derived fields:

- train-to-test return degradation
- train-to-test Sharpe degradation
- selection rank inside the train fold
- source mode: full-grid or reduced-candidate

## Stitched Out-of-Sample Result

The most important walk-forward artifact is the stitched out-of-sample result.

This should represent the combined chronological test windows only.

It should exclude:

- all train-period equity
- all train-period trades

The stitched out-of-sample result should be reviewable like a real validation track record.

Recommended stitched outputs:

- stitched OOS equity curve
- stitched OOS trades
- stitched OOS drawdown curve
- summary metrics across the combined OOS path

## Parameter Stability Tracking

Version 1 should track parameter drift in a simple, readable way.

For each fold, save:

- selected parameter values
- previous fold parameter values
- change magnitude by parameter

Useful summary views:

- parameter values by fold
- absolute change by fold
- count of parameter switches

What we want to learn:

- do chosen parameters stay in the same general neighborhood
- or do they jump wildly from fold to fold

Wild instability is usually a bad sign even if some single test blocks look good.

## Pass / Fail Guidance

A strategy should look promising only if most of these are true:

- out-of-sample performance remains acceptable across many folds
- train-to-test degradation is noticeable but not catastrophic
- drawdowns remain tolerable in test windows
- selected parameters do not jump randomly every fold
- the stitched OOS curve is still investable-looking after costs and slippage assumptions

Red flags:

- one or two great test folds with many weak ones
- train metrics much stronger than test metrics in most folds
- parameter choices jumping all over the grid
- stitched OOS equity that collapses despite attractive train results

## Data Model

Version 1 should store walk-forward studies separately from optimization studies.

Recommended tables:

### `walk_forward_studies`

Purpose:

- one row per walk-forward job

Suggested fields:

- `wf_study_id`
- `created_at`
- `strategy`
- `strategy_version`
- `asset_universe_json`
- `timeframes_json`
- `schedule_json`
- `candidate_source_mode`
- `params_json`
- `status`
- `description`

### `walk_forward_folds`

Purpose:

- one row per fold

Suggested fields:

- `wf_study_id`
- `fold_index`
- `train_start`
- `train_end`
- `test_start`
- `test_end`
- `selected_param_set_id`
- `selected_params_json`
- `train_rank`
- `train_robust_score`
- `test_run_id`
- `status`

### `walk_forward_fold_metrics`

Purpose:

- detailed per-fold train and test summaries

Suggested fields:

- `wf_study_id`
- `fold_index`
- `train_metrics_json`
- `test_metrics_json`
- `degradation_json`
- `param_drift_json`

### Artifact Handling

Recommended saved artifacts:

- stitched OOS equity image
- fold summary table export
- parameter drift plot
- optional fold-by-fold heatmap snapshots

## User Interface Model

Version 1 walk-forward UI should stay straightforward.

Recommended views:

### 1. Study Setup

Controls:

- strategy selector
- asset universe selector
- timeframe selector
- parameter grid or candidate-list selector
- train start
- first test start
- test block size
- number of folds
- candidate source mode

### 2. Fold Table

Columns:

- fold index
- train period
- test period
- selected params
- train score
- test return
- test Sharpe
- test drawdown
- test run id

Version 1 implementation should also allow:

- opening the saved train-fold optimization study for the selected fold
- opening the saved out-of-sample test run for the selected fold

### 3. Stitched OOS Equity View

Primary use:

- review the combined out-of-sample path as the main validation result

### 4. Parameter Drift View

Primary use:

- see whether chosen parameters remain stable or keep jumping
- review selected parameter values by fold
- count parameter switches and total numeric drift by fold

### 5. Train vs Test Degradation View

Primary use:

- inspect how much performance falls from train to test
- compare train ranking quality to realized out-of-sample Sharpe

## Recommended Version 1 Workflow

1. Start from a strategy and candidate universe that already survived baseline backtesting and parameter research.
2. Define an anchored walk-forward schedule.
3. Run fold-by-fold train optimization and test evaluation.
4. Review the fold table.
5. Review the stitched OOS equity curve.
6. Review parameter drift and train-to-test degradation.
7. Promote only the strategies that remain acceptable across the full stitched OOS result.

## Non-Goals

These should stay out of scope for version 1:

- automatic approval for live deployment
- intra-fold adaptive parameter switching
- complex rolling retraining policies
- automatic regime labeling
- Monte Carlo overlays inside the same module

## Relationship to Monte Carlo

Monte Carlo should come after walk-forward, not before.

Why:

- walk-forward tells us whether the strategy survives sequential out-of-sample testing
- Monte Carlo then tells us how fragile that validated path still is under randomized path stress

The default Monte Carlo input should be the stitched OOS result from this module. The next-stage robustness design lives in [MONTE_CARLO_SIMULATION.md](/home/ethan/quant_backtest_engine/MONTE_CARLO_SIMULATION.md).

## Summary

Version 1 walk-forward optimization should be:

- anchored
- fold-based
- sequential
- auditable
- focused on stitched out-of-sample results
- separate from standalone parameter optimization
- and separate from Monte Carlo stress testing

That gives the platform a clean and professional validation stage without overcomplicating the first version.
