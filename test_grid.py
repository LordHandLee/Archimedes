import os
import pandas as pd
from backtest_engine import (
    BacktestConfig,
    DuckDBStore,
    ZScoreMeanReversionStrategy,
    GridSearch,
    GridSpec,
    ResultCatalog,
)
from backtest_engine.execution import ExecutionMode

def main() -> None:
    dataset_id = "SPY_interactive_brokers_10y_1m"
    duck = DuckDBStore()

    def load_bars(timeframe: str):
        return duck.resample(dataset_id, timeframe=timeframe)

    db_path = "temp_grid.sqlite"
    if os.path.exists(db_path):
        os.remove(db_path)
        
    catalog = ResultCatalog(db_path)
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

    grid = GridSearch(
        dataset_id=dataset_id,
        data_loader=load_bars,
        strategy_cls=ZScoreMeanReversionStrategy,
        base_config=base_config,
        catalog=catalog,
    )

    spec_ref = GridSpec(
        params={"half_life_lookback": [10], "long_entry_z": [-0.5], "long_exit_z": [0.5], "target": [1.0], "use_vol_norm": [True], "z_smooth_len": [30], "z_smooth_type": ["SMA"]},
        timeframes=["15 minutes"],
        horizons=[(pd.Timestamp("2025-04-10 14:30:00+00:00"), pd.Timestamp("2026-04-10 14:15:00+00:00"))],
        metric="total_return",
        execution_mode=ExecutionMode.REFERENCE,
    )
    df_ref = grid.run(spec_ref, make_heatmap=False)

    spec_vec = GridSpec(
        params={"half_life_lookback": [10], "long_entry_z": [-0.5], "long_exit_z": [0.5], "target": [1.0], "use_vol_norm": [True], "z_smooth_len": [30], "z_smooth_type": ["SMA"]},
        timeframes=["15 minutes"],
        horizons=[(pd.Timestamp("2025-04-10 14:30:00+00:00"), pd.Timestamp("2026-04-10 14:15:00+00:00"))],
        metric="total_return",
        execution_mode=ExecutionMode.VECTORIZED,
    )
    df_vec = grid.run(spec_vec, make_heatmap=False)

    print("Reference:")
    print(df_ref[['total_return', 'cagr', 'sharpe', 'max_drawdown', 'engine_impl']])
    print("Vectorized:")
    print(df_vec[['total_return', 'cagr', 'sharpe', 'max_drawdown', 'engine_impl']])

if __name__ == "__main__":
    main()
