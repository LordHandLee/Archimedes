from __future__ import annotations

import pandas as pd


class Strategy:
    """
    Base class for strategies.

    Override `initialize` to prepare indicators.
    Override `on_bar` to submit orders each bar.
    Optionally override `on_after_fill` to react immediately after fills when
    `recalc_on_fill=True` is set in BacktestConfig.
    """

    def __init__(self, **params) -> None:
        self.params = params

    def initialize(self, data: pd.DataFrame) -> None:  # pragma: no cover - intended override
        pass

    def on_bar(self, timestamp: pd.Timestamp, bar: pd.Series, broker) -> None:  # pragma: no cover
        raise NotImplementedError

    def on_after_fill(self, timestamp: pd.Timestamp, bar: pd.Series, broker) -> None:  # pragma: no cover
        pass
