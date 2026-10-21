import pandas as pd
from backtest_engine import (
    BacktestConfig,
    DuckDBStore,
    ZScoreMeanReversionStrategy,
)
from backtest_engine.execution import ExecutionMode, ExecutionOrchestrator, ExecutionRequest

def main() -> None:
    dataset_id = "SPY_interactive_brokers_10y_1m"
    duck = DuckDBStore()
    try:
        bars_15m = duck.resample(dataset_id, timeframe="15 minutes")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    base_config = BacktestConfig(
        timeframe="15min",
        starting_cash=5000.0,
        fee_rate=0.0002,
        fee_schedule={"buy": 0.0003, "sell": 0.0005},
        slippage=0.0002,
        slippage_schedule={"buy": 0.0003, "sell": 0.0001},
        borrow_rate=0.03,
        fill_ratio=0.7,
        fill_on_close=False,
        recalc_on_fill=True,
        allow_short=False,
        use_cache=False,
        base_execution=False,
        time_horizon_start=pd.Timestamp("2025-04-10 14:30:00+00:00"),
        time_horizon_end=pd.Timestamp("2026-04-10 14:15:00+00:00"),
    )

    params = {"atr_mult": 1.0, "half_life_factor": 1.0, "half_life_lookback": 10, "long_entry_z": -0.5, "long_exit_z": 0.5, "std_len": 20, "target": 1.0, "use_vol_norm": True, "vol_len": 14, "vol_type": "ATR", "z_smooth_len": 30, "z_smooth_type": "SMA"}

    orchestrator = ExecutionOrchestrator()
    req_ref = ExecutionRequest(
        data=bars_15m,
        base_data=None,
        dataset_id=dataset_id,
        strategy_cls=ZScoreMeanReversionStrategy,
        strategy_params=params,
        config=base_config,
        requested_execution_mode=ExecutionMode.REFERENCE,
    )
    res_ref = orchestrator.execute(req_ref)

    req_vec = ExecutionRequest(
        data=bars_15m,
        base_data=None,
        dataset_id=dataset_id,
        strategy_cls=ZScoreMeanReversionStrategy,
        strategy_params=params,
        config=base_config,
        requested_execution_mode=ExecutionMode.VECTORIZED,
    )
    res_vec = orchestrator.execute(req_vec)

    print("Reference Metrics:", res_ref.metrics.as_dict())
    print("Vectorized Metrics:", res_vec.metrics.as_dict())
    print(f"Ref Trades: {len(res_ref.trades)} | Vec Trades: {len(res_vec.trades)}")

    mismatch_count = 0
    for i in range(min(len(res_ref.trades), len(res_vec.trades))):
        rt = res_ref.trades[i]
        vt = res_vec.trades[i]
        if abs(rt.qty - vt.qty) > 1e-4 or rt.timestamp != vt.timestamp or rt.side != vt.side:
            print(f"Mismatch at trade {i}!")
            print(f"Ref: {rt}")
            print(f"Vec: {vt}")
            mismatch_count += 1
            if mismatch_count >= 5:
                break
    
    if len(res_ref.trades) != len(res_vec.trades):
        print(f"Total trades differ! Ref: {len(res_ref.trades)}, Vec: {len(res_vec.trades)}")

if __name__ == "__main__":
    main()
