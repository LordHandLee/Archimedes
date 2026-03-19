from __future__ import annotations

import hashlib
import itertools
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Type

import pandas as pd
import matplotlib.pyplot as plt

from .engine import BacktestConfig, BacktestEngine
from .strategy import Strategy
from .catalog import ResultCatalog
from .reporting import plot_param_heatmap


def _hash_heatmap(payload: Dict) -> str:
    return hashlib.sha256(repr(sorted(payload.items())).encode()).hexdigest()


@dataclass
class GridSpec:
    params: Dict[str, Iterable]
    timeframes: Iterable[str]
    horizons: Iterable[tuple[pd.Timestamp | None, pd.Timestamp | None]]
    metric: str = "total_return"
    heatmap_rows: str = "param1"
    heatmap_cols: str = "param2"
    description: str = ""
    batch_id: str | None = None


def build_horizons(end: pd.Timestamp, windows: Iterable[pd.Timedelta]) -> List[tuple[pd.Timestamp, pd.Timestamp]]:
    return [(end - w, end) for w in windows]


class GridSearch:
    """
    Orchestrates parameter/timeframe/horizon sweeps optionally producing heatmaps.
    """

    def __init__(
        self,
        dataset_id: str,
        data_loader,  # callable that returns bars for a timeframe
        strategy_cls: Type[Strategy],
        base_config: BacktestConfig,
            catalog: Optional[ResultCatalog],
        ) -> None:
            self.dataset_id = dataset_id
            self.data_loader = data_loader
            self.strategy_cls = strategy_cls
            self.base_config = base_config
            self.catalog = catalog

    def run(
        self,
        grid: GridSpec,
        make_heatmap: bool = True,
        stop_cb: Optional[Callable[[], bool]] = None,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> pd.DataFrame:
        records: List[Dict] = []
        tf_list = list(grid.timeframes)
        horizon_list = list(grid.horizons)
        param_lists = list(grid.params.values()) if grid.params else []
        total = len(tf_list) * len(horizon_list)
        for lst in param_lists:
            total *= len(lst)
        batch_id = grid.batch_id or f"batch_{uuid.uuid4().hex[:8]}"
        if progress_cb:
            progress_cb(0, total)
        done = 0
        for timeframe in tf_list:
            bars = self.data_loader(timeframe)
            base_bars = None
            if timeframe != "1 minutes":
                base_bars = self.data_loader("1 minutes")
            for (start, end) in horizon_list:
                config = replace(
                    self.base_config,
                    batch_id=batch_id,
                    timeframe=timeframe,
                    time_horizon_start=start,
                    time_horizon_end=end,
                    base_execution=True if timeframe != "1 minutes" else False,
                )
                for combo in itertools.product(*grid.params.values()):
                    params = dict(zip(grid.params.keys(), combo))
                    if stop_cb and stop_cb():
                        return pd.DataFrame(records)
                    engine = BacktestEngine(
                        data=bars,
                        base_data=base_bars,
                        dataset_id=self.dataset_id,
                        strategy_cls=self.strategy_cls,
                        catalog=self.catalog,
                        config=config,
                    )
                    result = engine.run(params)
                    metrics = result.metrics.as_dict()
                    records.append(
                        {
                            **params,
                            "timeframe": timeframe,
                            "start": start,
                            "end": end,
                            grid.metric: metrics[grid.metric],
                            "run_id": result.run_id,
                        }
                    )
                    done += 1
                    if progress_cb:
                        progress_cb(done, total)

        df = pd.DataFrame(records)
        if make_heatmap and len(grid.params) >= 2:
            p1 = grid.heatmap_cols or list(grid.params.keys())[0]
            p2 = grid.heatmap_rows or list(grid.params.keys())[1]
            heatmap_id = _hash_heatmap(
                {"params": grid.params, "timeframes": list(grid.timeframes), "horizons": list(grid.horizons), "metric": grid.metric}
            )
            fig = plot_param_heatmap(df, value_col=grid.metric, row=p2, col=p1, title=f"{grid.metric} heatmap")
            heatmap_dir = "heatmaps"
            Path(heatmap_dir).mkdir(parents=True, exist_ok=True)
            file_path = str(Path(heatmap_dir) / f"heatmap_{heatmap_id[:8]}.png")
            fig.savefig(file_path)
            plt_close = getattr(__import__("matplotlib.pyplot"), "close", None)
            if plt_close:
                plt_close(fig)
            if self.catalog:
                self.catalog.save_heatmap(heatmap_id, {"params": grid.params, "metric": grid.metric}, file_path, grid.description)
        return df
