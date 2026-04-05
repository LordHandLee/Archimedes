# Monte Carlo Simulation Design

## Purpose

This document defines the version 1 Monte Carlo simulation module for the quant backtest engine.

Monte Carlo is a later-stage robustness tool.

Its job is to answer:

- how sensitive the strategy or portfolio is to path variation
- what range of returns and drawdowns is plausible if the future resembles the validated historical distribution
- how ugly the tail outcomes can get even after a strategy passes walk-forward validation

Monte Carlo is not a replacement for parameter optimization or walk-forward optimization.

## Role in the Pipeline

Monte Carlo should come after walk-forward optimization and before live deployment or paper deployment.

The intended sequence is:

1. baseline backtest and research sanity checks
2. parameter optimization
3. walk-forward optimization
4. Monte Carlo simulation
5. paper or forward test
6. deployment

This ordering matters because Monte Carlo should stress-test something that has already survived out-of-sample validation.

## Scope

Version 1 Monte Carlo should focus on:

- trade-path reshuffling
- trade bootstrap resampling
- percentile-based risk summaries
- equity fan charts and distribution plots
- drawdown probability reporting
- use of stitched out-of-sample walk-forward results as the default input

Version 1 should **not** include:

- synthetic price-path generation as the default mode
- options-style stochastic process modeling
- regime-switching simulation
- Monte Carlo embedded inside walk-forward folds
- automatic deployment decisions

## Design Goals

- keep the method easy to explain
- stress-test validated results instead of in-sample fantasy paths
- quantify path risk, not just average outcome
- produce distributions that help with capital allocation and deployment decisions
- keep the first version practical and auditable

## Recommended Version 1 Choices

Version 1 should use simple trade-based Monte Carlo methods first.

### 1. Trade Order Reshuffle

Method:

- take the realized trade return sequence
- randomly permute the order without replacement
- rebuild the equity path

What it tells us:

- how much drawdown depends on trade ordering
- how much path shape matters even when the same trade outcomes occur

Why it is useful:

- simple
- intuitive
- good for showing sequence risk

### 2. Trade Bootstrap

Method:

- sample trades from the realized trade return set with replacement
- rebuild a synthetic path with the same number of trades

What it tells us:

- how the strategy might behave if the future resembles the observed trade distribution but not the exact historical sequence

Why it is useful:

- still simple
- gives a wider distribution than pure reshuffling
- useful for return and drawdown percentiles

Recommended default:

- support both methods
- make trade bootstrap the main distribution view
- keep trade reshuffle as a sequence-risk companion view

## Preferred Input Source

The default Monte Carlo input should be:

- stitched out-of-sample trades from the walk-forward module

Why:

- these trades already passed a stronger validation step than raw in-sample optimization results
- this keeps Monte Carlo connected to the actual validation pipeline

Fallback input sources can include:

- a single backtest run
- a single portfolio run

But the preferred path is post-walk-forward.

## Input Contract

Each Monte Carlo study should define:

- `mc_study_id`
- source type
- source identifier such as `wf_study_id` or `run_id`
- resampling mode
- number of simulations
- random seed
- optional cost-stress assumptions
- optional study description

Recommended defaults:

- source type: stitched walk-forward OOS trades
- resampling mode: trade bootstrap
- simulation count: enough to stabilize percentiles without becoming impractical

## Output Contract

Each Monte Carlo study should produce:

- simulation summary table
- return distribution
- max drawdown distribution
- terminal equity distribution
- percentile table
- probability summaries
- representative sample paths

Recommended headline outputs:

- median terminal return
- 5th percentile return
- 95th percentile return
- median max drawdown
- 95th percentile max drawdown
- probability of loss
- probability drawdown exceeds chosen threshold

## Simulation Unit

Version 1 should use realized trade returns as the simulation unit.

That means we need a standardized trade-return representation such as:

- return per trade
- pnl per trade normalized by equity or capital base
- trade timestamp ordering from the original validated path

Why trade returns first:

- simple to compute
- strategy-agnostic enough for version 1
- matches how many traders think about path and sequence risk

Future versions can add block-bootstrapped daily returns or more advanced units if needed.

## Equity Reconstruction

Each simulation should rebuild an equity curve from the resampled trade sequence.

Recommended assumptions for version 1:

- same starting capital as the source study
- same number of trades as the source sequence
- same position-sizing convention embedded in the source trade returns

This makes the simulations easy to compare against the original validated result.

## Recommended Summary Metrics

Each Monte Carlo study should report at minimum:

- terminal return percentiles
- terminal equity percentiles
- max drawdown percentiles
- Sharpe percentiles if meaningful for the chosen unit
- probability of negative total return
- probability of exceeding drawdown thresholds such as 10%, 20%, or 30%

Recommended percentiles:

- 5th
- 25th
- 50th
- 75th
- 95th

## Cost Stress

Version 1 may optionally support a simple cost-stress overlay.

Example:

- increase slippage assumption
- increase fees
- reduce average trade return by a small fixed amount

This should stay simple and explicit.

Do not turn version 1 Monte Carlo into a microstructure simulator.

## Data Model

Monte Carlo studies should be stored separately from runs and walk-forward studies.

Recommended tables:

### `monte_carlo_studies`

Purpose:

- one row per Monte Carlo job

Suggested fields:

- `mc_study_id`
- `created_at`
- `source_type`
- `source_id`
- `resampling_mode`
- `simulation_count`
- `seed`
- `cost_stress_json`
- `status`
- `description`

### `monte_carlo_summary`

Purpose:

- store the high-level summary metrics of the study

Suggested fields:

- `mc_study_id`
- `terminal_return_p05`
- `terminal_return_p50`
- `terminal_return_p95`
- `max_drawdown_p50`
- `max_drawdown_p95`
- `loss_probability`
- `drawdown_thresholds_json`

### `monte_carlo_paths`

Purpose:

- store representative path references, not necessarily every full path inline

Suggested fields:

- `mc_study_id`
- `path_id`
- `path_type`
- `file_path`
- `summary_json`

Recommended path types:

- `median_path`
- `p05_path`
- `p95_path`
- `worst_drawdown_path`

### Artifact Handling

Recommended saved artifacts:

- equity fan chart
- histogram of terminal returns
- histogram of max drawdowns
- percentile summary report

## User Interface Model

Version 1 Monte Carlo UI should stay focused on interpretation.

Recommended views:

### 1. Study Setup

Controls:

- source selector
- resampling mode selector
- simulation count
- random seed
- optional cost-stress settings

### 2. Summary Panel

Show:

- median outcome
- downside percentiles
- loss probability
- drawdown exceedance probabilities

### 3. Equity Fan Chart

Primary use:

- visualize the spread of simulated paths against the original validated path

### 4. Distribution Views

Recommended plots:

- terminal return histogram
- max drawdown histogram

### 5. Risk Threshold Panel

Show:

- probability drawdown exceeds selected thresholds
- probability terminal return is below selected thresholds

## Recommended Version 1 Workflow

1. Start from a stitched out-of-sample walk-forward result.
2. Extract the normalized trade return sequence.
3. Run Monte Carlo with trade bootstrap and optionally trade reshuffle.
4. Review return and drawdown distributions.
5. Review tail outcomes and drawdown exceedance probabilities.
6. Decide whether the strategy or portfolio still looks deployable given the risk distribution.

## How to Use the Results

Monte Carlo should influence:

- position sizing
- capital allocation
- whether to require a paper-trading phase first
- whether the validated strategy still looks too fragile to deploy

Monte Carlo should not by itself:

- approve a strategy for live trading
- override failed walk-forward results
- rescue an overfit system

## Non-Goals

These should stay out of scope for version 1:

- complex stochastic price modeling
- agent-based market simulation
- parameter reselection during Monte Carlo
- full intraday execution simulation inside each Monte Carlo path

## Relationship to Deployment

Monte Carlo should not be the final switch that turns deployment on.

A stronger simplified pipeline is:

1. baseline backtest and sanity checks
2. parameter optimization
3. walk-forward optimization
4. Monte Carlo simulation
5. paper or forward test
6. deployment
7. monitoring and revalidation

If you are deploying multiple strategies or assets, portfolio construction and sizing decisions should happen before paper testing and deployment.

## Summary

Version 1 Monte Carlo should be:

- trade-based
- post-walk-forward
- focused on path and drawdown risk
- summarized with percentiles and probabilities
- simple enough to trust and explain

That gives the platform a strong robustness stage without overbuilding the first version.
