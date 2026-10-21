import pandas as pd
from backtest_engine import (
    BacktestConfig,
    DuckDBStore,
    SMACrossStrategy,
    GridSearch,
    GridSpec,
    ResultCatalog,
    build_horizons,
)
from backtest_engine.execution import ExecutionMode

def main() -> None:
    dataset_id = "AAPL_2026_Jan_1m"
    duck = DuckDBStore()

    def load_bars(timeframe: str):
        return duck.resample(dataset_id, timeframe=timeframe)

    raw = duck.load(dataset_id)
    end_ts = raw.index[-1]

    catalog = ResultCatalog(":memory:")
    base_config = BacktestConfig(
        timeframe="5min",
        starting_cash=100_000,
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
        base_execution=False
    )

    grid = GridSearch(
        dataset_id=dataset_id,
        data_loader=load_bars,
        strategy_cls=SMACrossStrategy,
        base_config=base_config,
        catalog=catalog,
    )

    spec_ref = GridSpec(
        params={"fast": [5], "slow": [20]},
        timeframes=["5 minutes"],
        horizons=[(None, None)],
        metric="total_return",
        execution_mode=ExecutionMode.REFERENCE,
    )
    df_ref = grid.run(spec_ref, make_heatmap=False)

    spec_vec = GridSpec(
        params={"fast": [5], "slow": [20]},
        timeframes=["5 minutes"],
        horizons=[(None, None)],
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
