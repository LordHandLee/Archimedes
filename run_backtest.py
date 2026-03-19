from pathlib import Path

import pandas as pd

from backtest_engine import (
    BacktestConfig,
    BacktestEngine,
    DuckDBStore,
    GridSearch,
    GridSpec,
    ResultCatalog,
    SMACrossStrategy,
    build_horizons,
    load_csv_prices,
)


def main() -> None:
    csv_path = Path("AAPL.USUSD_Candlestick_1_M_BID_12.01.2026-17.01.2026.csv")
    dataset_id = "AAPL_2026_Jan_1m"

    # 1) Load CSV and persist to DuckDB/parquet once.
    loaded = load_csv_prices(csv_path)
    duck = DuckDBStore()
    duck.write_parquet(dataset_id, loaded.data.reset_index())

    # Loader callable for grid/backtest modules.
    def load_bars(timeframe: str):
        return duck.resample(dataset_id, timeframe=timeframe)

    raw = duck.load(dataset_id)
    end_ts = raw.index[-1]

    catalog = ResultCatalog("backtests.sqlite")
    base_config = BacktestConfig(
        timeframe="5T",
        starting_cash=100_000,
        fee_rate=0.0002,
        fee_schedule={"buy": 0.0003, "sell": 0.0005},
        slippage=0.0002,
        slippage_schedule={"buy": 0.0003, "sell": 0.0001},
        borrow_rate=0.03,  # 3% annualized on short notional
        fill_ratio=0.7,  # simulate partial fills per bar
        fill_on_close=False,
        recalc_on_fill=True,
        allow_short=False,
        use_cache=True,
    )

    # Example: single backtest (no grid).
    bars = load_bars("5 minutes")
    engine = BacktestEngine(
        data=bars,
        base_data=load_bars("1 minutes"),
        dataset_id=dataset_id,
        strategy_cls=SMACrossStrategy,
        catalog=catalog,
        config=replace(base_config, base_execution=True, base_timeframe="1 minutes"),
    )
    single = engine.run({"fast": 10, "slow": 30, "target": 1.0})
    print("Single backtest:", single.run_id, single.metrics.as_dict())

    # Example: optional grid search (OOP module).
    grid = GridSearch(
        dataset_id=dataset_id,
        data_loader=load_bars,
        strategy_cls=SMACrossStrategy,
        base_config=base_config,
        catalog=catalog,
    )
    spec = GridSpec(
        params={"fast": [5, 10, 15], "slow": [20, 30, 40]},
        timeframes=["1 minutes", "5 minutes", "15 minutes", "1 hours"],
        horizons=build_horizons(end_ts, [pd.Timedelta(days=7), pd.Timedelta(days=30)]),
        metric="total_return",
        heatmap_rows="slow",
        heatmap_cols="fast",
        description="SMA grid",
    )
    df = grid.run(spec, make_heatmap=True)
    print(df.head())



if __name__ == "__main__":
    main()
