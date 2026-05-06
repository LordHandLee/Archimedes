# Deployment Tab Guide

This dashboard treats deployment as a handoff layer between validated backtests/portfolio studies and the external execution engine.

## Short Answer

Most of the time you should not type `Params JSON` or `Structure JSON` by hand.

If you create a deployment from a validated candidate, the dashboard copies the strategy parameters and portfolio structure from the backtest/optimization artifacts. In that workflow, the engine already knows the strategy recipe and the Deployment tab is just packaging it for live/paper handoff.

Manual Deployment is the advanced escape hatch. Use it only when you want to draft a deployment that did not come from a saved candidate yet. In manual mode:

- `Params JSON` means strategy settings.
- `Structure JSON` means portfolio/asset layout.
- Neither field is the broker order payload.
- The actual order payload should be built later by the deployment runner from the saved deployment record, current live bars, account state, and sizing model.

Current status: the Deployment tab is now a draft/monitor/live-runner layer. It stores the deployment recipe, syncs external account/order state, opens live charts, and can run supported deployed strategies against completed live bars. When a supported deployment is armed, the dashboard starts the live runner, marks the deployment `live`, and sends per-signal ENTRY/EXIT JSON webhooks to the external execution engine.

## Basic Workflow

1. Build and validate a strategy or portfolio in Optimization, Walk Forward, or Monte Carlo.
2. Open the Deployment tab.
3. Choose a validated candidate. Prefer this path because params and structure are copied automatically.
4. Select a deployment target, usually `Algo Engine Live` or `Algo Engine Paper`.
5. Choose the deployment-level sizing type and value.
6. Click `Create Draft Deployment`.
7. Review the draft in `Deployed / Draft Strategies`.
8. Click `Arm` to start the live runner. A successful arm validates the webhook target/secret, starts an Interactive Brokers live stream for each deployment symbol, marks the deployment `live`, and waits for the next completed strategy bar before sending any order signal.
9. Use `Sync External State` to reconcile account equity, buying power, positions, orders, fills, and equity curve data from the external engine.
10. Use Live Monitor to open Magellan charts for the deployment symbols and inspect live prices plus trade markers.

`Pause` and `Stop` deactivate the dashboard-side live runner for that deployment. They do not cancel broker orders that already reached the external execution engine.

The execution-engine webhook secret is a target setting. In Live Monitor, select a deployment, enter the `External Engine URL` and `Webhook Secret`, then click `Save Target Settings`. The secret is saved in the dashboard catalog database with the target record. The dashboard does not scan local SFTP/GVFS mounts or remote `.env` files to discover secrets.

The Deployment tab does not rewrite historical price data. Live Interactive Brokers bars are stored in the separate live-market store and stitched into charts at display time.

## Params JSON

`Params JSON` is the strategy parameter object. It answers: "How should this strategy calculate signals?"

You normally do not enter this when using a validated candidate. The candidate already has the params from the backtest that produced the Sharpe/return you accepted.

Example:

```json
{
  "fast_sma": 20,
  "slow_sma": 50,
  "risk_mode": "annual_volatility",
  "annual_vol_window": 252
}
```

For manual deployments, enter only a JSON object. Arrays or plain strings are rejected.

Examples of things that belong in `Params JSON`:

- moving average lengths
- z-score windows
- stop or trailing-stop settings
- signal thresholds
- future strategy-level sizing settings once dynamic sizing is implemented

Examples of things that do not belong in `Params JSON`:

- account buying power
- current positions
- order IDs
- broker endpoint URLs
- final share quantity for one specific order

## Structure JSON

`Structure JSON` describes portfolio composition and construction rules. It answers: "Which assets or strategy blocks does this deployment contain?"

You normally do not enter this when using a validated candidate. Portfolio optimization and fixed portfolio definitions already generate it.

Shared-strategy portfolio example:

```json
{
  "portfolio_dataset_ids": [
    "interactive_brokers:AAPL:1min",
    "interactive_brokers:MSFT:1min",
    "interactive_brokers:NVDA:1min"
  ],
  "construction_config": {
    "allocation_ownership": "strategy",
    "weighting_mode": "equal_selected",
    "rebalance_mode": "on_change_or_periodic"
  }
}
```

Strategy-block portfolio example:

```json
{
  "strategy_blocks": [
    {
      "block_id": "trend",
      "strategy_name": "SMACrossStrategy",
      "strategy_params": {"fast_sma": 20, "slow_sma": 50},
      "asset_dataset_ids": ["interactive_brokers:SPY:1min", "interactive_brokers:QQQ:1min"]
    }
  ],
  "portfolio_dataset_ids": ["interactive_brokers:SPY:1min", "interactive_brokers:QQQ:1min"]
}
```

The dashboard resolves dataset IDs to ticker symbols when it opens Live Monitor charts, starts IB live streams, runs Live Monitor historical sync, or matches external positions/orders. Provider-style IDs such as `TQQQ_interactive_brokers_10y_1m` must be reduced to `TQQQ` before any Interactive Brokers contract request; IB error 200 usually means a dataset id leaked into a symbol field.

For a single manual strategy on one ticker, `Structure JSON` can usually be `{}` because the symbol/dataset scope fields already identify the asset.

## Sizing Hierarchy

The safest design is to make dynamic sizing visible in backtests before it reaches live deployment. That means annual-volatility sizing should be modeled where Sharpe and return are calculated, not hidden only in the external execution engine.

Recommended hierarchy:

1. Strategy-level signal decides whether the asset should be long, short, flat, or adjusted.
2. Strategy-level sizing model can adjust the desired exposure for that asset, including annual-volatility sizing.
3. Portfolio-level construction combines assets/blocks and applies portfolio constraints.
4. Deployment-level sizing limits the account slice exposed to that deployment.
5. External engine validates broker constraints, buying power, margin, and order rules.

This keeps the important research question answerable: "What happened to Sharpe, return, drawdown, and exposure when annual-volatility sizing was enabled?"

The external engine accepts a `position_size_override` from this dashboard so the dashboard can own steps 1-4 while the external engine owns broker validation and execution. The live runner stores deployment lineage in the raw webhook payload so external fills/orders can be tied back to deployment, child leg, strategy block, dataset, timeframe, and sizing trace.

## Annual Volatility Sizing

Do not use a portfolio target weight of `2.0` as the dynamic-sizing switch. That would mean "base target is 200%" before the volatility adjustment and could fight portfolio caps, deployment sizing, and margin limits.

Use `2.0` only as the maximum volatility multiplier.

For the planned annual-volatility overlay:

```text
base_target_notional = deployable_equity * base_target_weight
volatility_multiplier = min(2.0, 1.0 / annual_volatility)
target_notional = base_target_notional * volatility_multiplier
target_shares = floor(target_notional / latest_price)
target_shares = max(1, target_shares) when the signal is active and buying power allows it
```

Using your example with `$10,000`, ten stocks, and a base 10% target:

```text
base_target_notional = 10000 * 0.10 = 1000
annual_volatility = 2.0  -> target_notional = 1000 * 0.5 = 500
annual_volatility = 0.5  -> target_notional = 1000 * 2.0 = 2000
```

The `2.0` cap is the leverage ceiling for the volatility overlay. Deployment-level controls should still cap total gross exposure and account usage so the portfolio cannot accidentally exceed the intended margin envelope.

## Implemented Backtest Sizing Controls

The Backtest Settings panel now exposes the sizing controls directly:

- `Margin Enabled`: allows the backtest broker/vectorized engines to hold gross exposure above cash.
- `Max Gross Leverage`: account/portfolio gross exposure cap. Use `2.0` for the double-exposure ceiling you described.
- `Position Sizing`: choose `Fixed / None` or `Annual Volatility Target`.
- `Annual Vol Window Days`: trading-day lookback used to estimate annualized volatility. The engine converts this to the correct bar count for the selected timeframe, so `252` means about one trading year on daily, 15-minute, hourly, and other intraday charts.
- `Annual Vol Min Days`: minimum trading-day history before the volatility estimate becomes active. This is also converted to bars for the selected timeframe, so the default `20` means about 20 trading days, not 20 intraday bars.
- `Annual Vol Floor`: lower bound on volatility so a quiet series cannot explode position size.
- `Max Vol Multiplier`: cap on `1 / annual_volatility`; use `2.0` for the maximum doubling behavior.
- `Min Position Shares`: minimum quantity for an active dynamically sized signal when buying power/margin allows it.

The default behavior is unchanged: fixed sizing, no margin. Existing studies should continue to behave the same unless these controls are enabled.

## Margin And Exposure

Annual-volatility sizing should not be hard-wired to one margin assumption. Keep it as an exposure-sizing model that outputs a desired target weight or notional. Then let a separate margin/exposure policy decide whether that target can be filled.

Implemented backtest policy:

1. `margin_enabled = false`: long buys are cash-limited and portfolio gross exposure is clipped at `1.0` after cash reserve.
2. `margin_enabled = true`: target weights above `1.0` are allowed up to `max_gross_leverage`, for example `2.0`.
3. Strategy sizing can request dynamic exposure, but the execution/risk layer enforces buying power, max asset weight, max gross exposure, short rules, borrow costs, and maintenance-margin checks.
4. The annual-volatility `max_volatility_multiplier` is not the same thing as account leverage. The multiplier scales one asset's base target; `max_gross_leverage` caps the whole account or portfolio.

This keeps the design open-ended. Annual volatility becomes one sizing plugin, and later methods such as ATR risk, Kelly fraction, drawdown throttling, or regime-based sizing can use the same target-weight/target-notional contract.

The practical rule for research is: dynamic sizing should work without margin by clipping exposure, but the full `2.0` upside case requires margin enabled and an explicit portfolio/account leverage cap. That way Sharpe and return comparisons can show both "cash account" and "margin account" behavior without rewriting strategies.

## Dynamic Sizing Design

The sizing system is deliberately open-ended. Annual volatility is implemented as the first model, not as a one-off special case in the SMA or z-score strategies.

Current implementation:

1. A reusable sizing module computes per-bar sizing multipliers.
2. Reference broker backtests apply the multiplier when a strategy calls `target_percent`.
3. Single-asset vectorized runs apply the same multiplier before generating target order quantities.
4. Vectorized portfolio runs apply the multiplier to asset target weights before portfolio construction caps are enforced.
5. Run IDs include the sizing and margin settings so cached results cannot mix fixed-sizing and dynamic-sizing studies.

Live deployment behavior:

1. The promoted deployment stores sizing settings in deployment `sizing_json`.
2. The live runner reads completed bars from the live-market store and evaluates supported strategy classes.
3. For `fixed` deployment sizing it sends a `target_qty` override.
4. For `cash` deployment sizing it sends a `target_notional` override.
5. For `percent_equity` sizing it now computes the deployment slice locally, applies the optional annual-volatility overlay from the stored execution config, caps the result by the deployment gross-leverage rule, and sends the final `target_notional` override to the execution engine.

Proposed settings:

```json
{
  "sizing_model": "annual_volatility_target",
  "annual_vol_window": 252,
  "annual_vol_min_periods": 20,
  "annual_vol_floor": 0.05,
  "max_volatility_multiplier": 2.0,
  "min_shares": 1,
  "base_target_weight_source": "strategy_or_portfolio"
}
```

`annual_vol_window` and `annual_vol_min_periods` are trading-day values. The backtest engine converts them to bars from the selected timeframe before calculating rolling volatility.

How this fits existing allocation ownership:

- `Strategy-Owned`: the strategy's own base sizing remains primary; annual-volatility sizing scales that strategy exposure.
- `Portfolio-Owned`: the portfolio chooses base weights first; annual-volatility sizing scales each portfolio target before final caps.
- `Hybrid`: strategy signals decide active/inactive exposure, while portfolio construction and the annual-volatility model shape target weights.

Implementation added this as an optional model instead of changing current default behavior. Existing backtests and deployments should behave exactly the same unless the model is explicitly enabled.

## Live Data Reuse

Yes, the same Interactive Brokers live stream can feed charts and real-time strategy execution. The intended architecture is:

```text
Interactive Brokers reqRealTimeBars
  -> LiveMarketDataStore
  -> Charts / Magellan snapshots
  -> Deployment live runner thread
  -> External engine webhook/order API
```

That keeps live data isolated from historical data. Strategy execution reads completed bars from `LiveMarketDataStore` and sends orders only when the strategy's desired state changes. A 5-minute deployment, for example, does not trade on the still-forming 10:55 bar; it evaluates that bar only once the next 5-minute bucket begins.

The GUI thread does not run strategy evaluation or webhook POSTs. The Qt live stream emits bar records, and the GUI enqueues those records into `LiveDeploymentRunnerWorker`. The worker thread loads the recent historical/live bars, initializes the same strategy class used by backtests with the saved deployment params, calls `strategy.on_bar(...)` with a small signal broker that captures `target_percent(...)`, and translates state changes into execution-engine webhooks.

Historical data can be requested while real-time subscriptions are active, but it should be throttled. IBKR real-time bars use `reqRealTimeBars`, which creates 5-second bar subscriptions and is subject to both market-data-line limits and small-bar pacing limits. Historical bars use `reqHistoricalData`, with `useRTH=0` to include all available regular plus extended-hours data. IBKR allows multiple historical requests, but their documentation still warns about pacing and load-balancing, so the dashboard should gap-fill a bounded window and avoid large duplicate downloads while live strategy streams are running.

Operational rule:

- Do not use `Stop Live` in Charts as a global kill switch for deployment feeds.
- Charts can pause their own watchlist/chart subscriptions.
- Deployment strategy feeds should continue unless the deployment itself is paused or stopped.
- If a historical gap fill is needed during live operation, queue it with pacing limits and merge the returned bars into the separate live-market store, not the historical research store.
- The live-market SQLite store must be opened in WAL mode, use a busy timeout, retry transient `database is locked` errors, and batch historical-sync writes so a gap repair cannot kill an active real-time chart stream.
