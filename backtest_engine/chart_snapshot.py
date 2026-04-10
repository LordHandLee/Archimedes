from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather

from .catalog import ResultCatalog
from .sample_strategies import compute_zscore_mean_reversion_features


DEFAULT_SNAPSHOT_ROOT = Path("data/chart_snapshots")


def sanitize_series_name(name: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_]+", "_", name.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "Series"


@dataclass
class ChartSnapshotArtifact:
    run_id: str
    snapshot_root: Path
    manifest_path: Path


class ChartSnapshotExporter:
    def __init__(self, root_dir: str | Path = DEFAULT_SNAPSHOT_ROOT) -> None:
        self.root_dir = Path(root_dir)

    def export_backtest_snapshot(
        self,
        *,
        run,
        bars: pd.DataFrame,
        overlays: Mapping[str, pd.Series],
        panes: Mapping[str, pd.Series],
        series_styles: Mapping[str, dict] | None,
        equity_curve: pd.Series,
        trades_df: pd.DataFrame,
        overwrite: bool = True,
    ) -> ChartSnapshotArtifact:
        snapshot_root = self.root_dir / str(run.run_id)
        return self._export_backtest_snapshot_to_root(
            run=run,
            snapshot_root=snapshot_root,
            bars=bars,
            overlays=overlays,
            panes=panes,
            series_styles=series_styles,
            equity_curve=equity_curve,
            trades_df=trades_df,
            overwrite=overwrite,
        )

    def _export_backtest_snapshot_to_root(
        self,
        *,
        run,
        snapshot_root: Path,
        bars: pd.DataFrame,
        overlays: Mapping[str, pd.Series],
        panes: Mapping[str, pd.Series],
        series_styles: Mapping[str, dict] | None,
        equity_curve: pd.Series,
        trades_df: pd.DataFrame,
        overwrite: bool = True,
    ) -> ChartSnapshotArtifact:
        if bars is None or bars.empty:
            raise ValueError("Cannot export a snapshot without price bars.")
        if bars.index.tz is None:
            raise ValueError("Snapshot bars must be indexed by UTC timestamps.")

        if overwrite and snapshot_root.exists():
            shutil.rmtree(snapshot_root)
        snapshot_root.mkdir(parents=True, exist_ok=True)

        price_bars_df = self._build_price_bars_dataframe(bars)
        overlays_df, overlay_order = self._build_series_dataframe(bars, overlays)
        panes_df, pane_order = self._build_series_dataframe(bars, panes)
        equity_df = self._build_equity_dataframe(bars, equity_curve)
        trades_out_df = trades_df.copy() if trades_df is not None else pd.DataFrame()

        price_bars_path = snapshot_root / "price_bars.feather"
        overlays_path = snapshot_root / "overlays.feather"
        panes_path = snapshot_root / "panes.feather"
        equity_path = snapshot_root / "equity.feather"
        trades_path = snapshot_root / "trades.feather"

        self._write_feather(price_bars_df, price_bars_path)
        self._write_feather(overlays_df, overlays_path)
        self._write_feather(panes_df, panes_path)
        self._write_feather(equity_df, equity_path)
        if not trades_out_df.empty:
            self._write_feather(trades_out_df, trades_path)

        manifest = self._build_manifest(
            run=run,
            bars=price_bars_df,
            overlay_order=overlay_order,
            pane_order=pane_order,
            equity_df=equity_df,
            trades_df=trades_out_df,
            files={
                "price_bars": price_bars_path.name,
                "trades": trades_path.name if not trades_out_df.empty else "",
                "equity": equity_path.name,
                "overlays": overlays_path.name,
                "panes": panes_path.name,
            },
            series_styles=series_styles or {},
        )
        manifest_path = snapshot_root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8")

        return ChartSnapshotArtifact(
            run_id=str(run.run_id),
            snapshot_root=snapshot_root.resolve(),
            manifest_path=manifest_path.resolve(),
        )

    def export_portfolio_snapshot(
        self,
        *,
        run,
        portfolio_result,
        overwrite: bool = True,
    ) -> ChartSnapshotArtifact:
        equity_curve = pd.to_numeric(portfolio_result.portfolio_equity_curve, errors="coerce").astype(float)
        if equity_curve.empty:
            raise ValueError("Cannot export a portfolio snapshot without a portfolio equity curve.")
        if equity_curve.index.tz is None:
            raise ValueError("Portfolio equity curve must be indexed by UTC timestamps.")

        bars = self._build_portfolio_bars(equity_curve)
        overlays: dict[str, pd.Series] = {}
        panes, styles = self._build_portfolio_panes(portfolio_result, equity_curve)
        trades_df = self.build_portfolio_trade_frame(portfolio_result, bars)
        return self.export_backtest_snapshot(
            run=run,
            bars=bars,
            overlays=overlays,
            panes=panes,
            series_styles=styles,
            equity_curve=equity_curve.rename("equity"),
            trades_df=trades_df,
            overwrite=overwrite,
        )

    def export_portfolio_asset_snapshots(
        self,
        *,
        run,
        portfolio_result,
        source_bars: Mapping[str, pd.DataFrame],
        strategy_contexts: Mapping[str, Sequence[tuple[str | None, str, Mapping[str, object]]]] | None = None,
        overwrite: bool = True,
    ) -> list[ChartSnapshotArtifact]:
        if not source_bars:
            raise ValueError("Cannot export portfolio asset snapshots without source asset bars.")

        run_root = self.root_dir / str(run.run_id)
        assets_root = run_root / "assets"
        if overwrite and run_root.exists():
            shutil.rmtree(run_root)
        assets_root.mkdir(parents=True, exist_ok=True)

        portfolio_equity = pd.to_numeric(portfolio_result.portfolio_equity_curve, errors="coerce").astype(float)
        if portfolio_equity.empty:
            raise ValueError("Cannot export portfolio asset snapshots without a portfolio equity curve.")

        artifacts: list[ChartSnapshotArtifact] = []
        for source_dataset_id, bars in source_bars.items():
            if bars is None or bars.empty:
                continue
            aligned_bars = bars.sort_index().copy()
            if aligned_bars.index.tz is None:
                aligned_bars.index = aligned_bars.index.tz_localize("UTC")
            else:
                aligned_bars.index = aligned_bars.index.tz_convert("UTC")

            overlays, panes, styles = self._build_portfolio_asset_strategy_series(
                aligned_bars,
                list((strategy_contexts or {}).get(str(source_dataset_id), [])),
            )
            trades_df = self.build_portfolio_asset_trade_frame(
                portfolio_result,
                bars=aligned_bars,
                source_dataset_id=source_dataset_id,
            )
            equity_curve = pd.to_numeric(portfolio_equity.reindex(aligned_bars.index), errors="coerce").ffill().bfill()
            asset_run = SimpleNamespace(
                run_id=str(run.run_id),
                dataset_id=str(source_dataset_id),
                strategy=str(getattr(run, "strategy", "PortfolioExecution") or "PortfolioExecution"),
                params=getattr(run, "params", {}),
                timeframe=str(getattr(run, "timeframe", "") or ""),
                start=str(aligned_bars.index[0]),
                end=str(aligned_bars.index[-1]),
                starting_cash=getattr(run, "starting_cash", None),
                metrics=getattr(run, "metrics", {}),
            )
            snapshot_root = assets_root / sanitize_series_name(str(source_dataset_id))
            artifacts.append(
                self._export_backtest_snapshot_to_root(
                    run=asset_run,
                    snapshot_root=snapshot_root,
                    bars=aligned_bars,
                    overlays=overlays,
                    panes=panes,
                    series_styles=styles,
                    equity_curve=equity_curve.rename("equity"),
                    trades_df=trades_df,
                    overwrite=True,
                )
            )
        if not artifacts:
            raise ValueError("No portfolio asset snapshots could be exported.")
        return artifacts

    @staticmethod
    def build_trade_frame(run, catalog_path: str | Path, bars: pd.DataFrame) -> pd.DataFrame:
        if bars is None or bars.empty:
            return pd.DataFrame()

        catalog = ResultCatalog(catalog_path)
        trades = catalog.load_trades(str(run.run_id)) or []
        if not trades:
            return pd.DataFrame(
                columns=[
                    "seq",
                    "ts_utc_ns",
                    "side",
                    "qty",
                    "price",
                    "fee",
                    "realized_pnl",
                    "equity_after",
                    "bar_index",
                    "event",
                    "event_type",
                    "position_after",
                    "label",
                ]
            )

        rows: list[dict] = []
        position = 0.0
        first_ts = bars.index[0]
        last_ts = bars.index[-1]
        for seq, trade in enumerate(trades, start=1):
            trade_ts = pd.to_datetime(trade.get("timestamp"), utc=True, errors="coerce")
            if pd.isna(trade_ts) or trade_ts < first_ts or trade_ts > last_ts:
                continue

            bar_index = int(bars.index.get_indexer([trade_ts], method="nearest")[0])
            if bar_index < 0:
                continue

            qty = float(trade.get("qty", 0.0))
            side = str(trade.get("side") or ("buy" if qty > 0 else "sell")).lower()
            position_before = position
            position += qty

            if position_before * position < 0:
                event = "flip"
            elif abs(position) > abs(position_before):
                event = "entry"
            elif position == position_before:
                event = "adjust"
            else:
                event = "exit"

            rows.append(
                {
                    "seq": seq,
                    "ts_utc_ns": int(trade_ts.value),
                    "side": side,
                    "qty": qty,
                    "price": float(trade.get("price", 0.0)),
                    "fee": float(trade.get("fee", 0.0)),
                    "realized_pnl": float(trade.get("realized_pnl", 0.0)),
                    "equity_after": float(trade.get("equity_after", 0.0)),
                    "bar_index": bar_index,
                    "event": event,
                    "event_type": event,
                    "position_after": position,
                    "label": f"{event.title()} {side.title()} {seq}",
                }
            )

        return pd.DataFrame(rows)

    @staticmethod
    def build_equity_curve(run, bars: pd.DataFrame, trades_df: pd.DataFrame) -> pd.Series:
        if bars is None or bars.empty:
            return pd.Series(dtype=float)

        price_bars = bars.sort_index()
        cash = float(run.starting_cash) if getattr(run, "starting_cash", None) is not None else 100_000.0
        position = 0.0
        trade_rows = trades_df.sort_values(["bar_index", "seq"]).to_dict("records") if trades_df is not None and not trades_df.empty else []
        trade_cursor = 0
        equity_values: list[float] = []

        closes = price_bars["close"].to_numpy(dtype=float)
        for bar_index in range(len(price_bars)):
            while trade_cursor < len(trade_rows) and int(trade_rows[trade_cursor]["bar_index"]) == bar_index:
                trade = trade_rows[trade_cursor]
                qty = float(trade["qty"])
                price = float(trade["price"])
                fee = float(trade["fee"])
                cash -= qty * price
                cash -= fee
                position += qty
                trade_cursor += 1

            equity_values.append(cash + position * closes[bar_index])

        return pd.Series(equity_values, index=price_bars.index, dtype=float, name="equity")

    @staticmethod
    def build_portfolio_trade_frame(portfolio_result, bars: pd.DataFrame) -> pd.DataFrame:
        if bars is None or bars.empty:
            return pd.DataFrame()

        trades = list(getattr(portfolio_result, "trades", []) or [])
        if not trades:
            return pd.DataFrame(
                columns=[
                    "seq",
                    "ts_utc_ns",
                    "side",
                    "qty",
                    "price",
                    "fee",
                    "realized_pnl",
                    "equity_after",
                    "bar_index",
                    "event",
                    "event_type",
                    "position_after",
                    "label",
                ]
            )

        first_ts = bars.index[0]
        last_ts = bars.index[-1]
        asset_positions: dict[str, float] = {}
        equity_curve = pd.to_numeric(getattr(portfolio_result, "portfolio_equity_curve", pd.Series(dtype=float)), errors="coerce")
        rows: list[dict] = []
        for seq, trade in enumerate(trades, start=1):
            trade_ts = pd.to_datetime(getattr(trade, "timestamp", None), utc=True, errors="coerce")
            if pd.isna(trade_ts) or trade_ts < first_ts or trade_ts > last_ts:
                continue

            bar_index = int(bars.index.get_indexer([trade_ts], method="nearest")[0])
            if bar_index < 0:
                continue

            dataset_id = str(getattr(trade, "dataset_id", "asset"))
            qty = float(getattr(trade, "qty", 0.0))
            side = str(getattr(trade, "side", "") or ("buy" if qty > 0 else "sell")).lower()
            position_before = asset_positions.get(dataset_id, 0.0)
            position_after = position_before + qty
            asset_positions[dataset_id] = position_after

            if position_before * position_after < 0:
                event = "flip"
            elif abs(position_after) > abs(position_before):
                event = "entry"
            elif position_after == position_before:
                event = "adjust"
            else:
                event = "exit"

            equity_after = float(getattr(trade, "equity_after", np.nan))
            if not np.isfinite(equity_after) and not equity_curve.empty and 0 <= bar_index < len(equity_curve):
                equity_after = float(equity_curve.iloc[bar_index])

            rows.append(
                {
                    "seq": seq,
                    "ts_utc_ns": int(trade_ts.value),
                    "side": side,
                    "qty": qty,
                    "price": float(getattr(trade, "price", 0.0)),
                    "fee": float(getattr(trade, "fee", 0.0)),
                    "realized_pnl": float(getattr(trade, "realized_pnl", 0.0)),
                    "equity_after": equity_after,
                    "bar_index": bar_index,
                    "event": event,
                    "event_type": event,
                    "position_after": position_after,
                    "label": f"{dataset_id} {event.title()} {seq}",
                }
            )

        return pd.DataFrame(rows)

    @staticmethod
    def build_portfolio_asset_trade_frame(
        portfolio_result,
        bars: pd.DataFrame,
        source_dataset_id: str,
    ) -> pd.DataFrame:
        if bars is None or bars.empty:
            return pd.DataFrame()

        trades = list(getattr(portfolio_result, "trades", []) or [])
        if not trades:
            return pd.DataFrame(
                columns=[
                    "seq",
                    "ts_utc_ns",
                    "side",
                    "qty",
                    "price",
                    "fee",
                    "realized_pnl",
                    "equity_after",
                    "bar_index",
                    "event",
                    "event_type",
                    "position_after",
                    "label",
                ]
            )

        first_ts = bars.index[0]
        last_ts = bars.index[-1]
        position = 0.0
        rows: list[dict] = []
        for seq, trade in enumerate(trades, start=1):
            trade_ts = pd.to_datetime(getattr(trade, "timestamp", None), utc=True, errors="coerce")
            if pd.isna(trade_ts) or trade_ts < first_ts or trade_ts > last_ts:
                continue

            trade_source_dataset_id = str(getattr(trade, "source_dataset_id", "") or getattr(trade, "dataset_id", "") or "")
            if trade_source_dataset_id != str(source_dataset_id):
                continue

            bar_index = int(bars.index.get_indexer([trade_ts], method="nearest")[0])
            if bar_index < 0:
                continue

            qty = float(getattr(trade, "qty", 0.0))
            side = str(getattr(trade, "side", "") or ("buy" if qty > 0 else "sell")).lower()
            position_before = position
            position_after = position_before + qty
            position = position_after

            if position_before * position_after < 0:
                event = "flip"
            elif abs(position_after) > abs(position_before):
                event = "entry"
            elif position_after == position_before:
                event = "adjust"
            else:
                event = "exit"

            display_label = str(getattr(trade, "dataset_id", source_dataset_id) or source_dataset_id)
            rows.append(
                {
                    "seq": seq,
                    "ts_utc_ns": int(trade_ts.value),
                    "side": side,
                    "qty": qty,
                    "price": float(getattr(trade, "price", 0.0)),
                    "fee": float(getattr(trade, "fee", 0.0)),
                    "realized_pnl": float(getattr(trade, "realized_pnl", 0.0)),
                    "equity_after": float(getattr(trade, "equity_after", np.nan)),
                    "bar_index": bar_index,
                    "event": event,
                    "event_type": event,
                    "position_after": position_after,
                    "label": f"{display_label} {event.title()} {seq}",
                }
            )

        return pd.DataFrame(rows)

    @staticmethod
    def _build_price_bars_dataframe(bars: pd.DataFrame) -> pd.DataFrame:
        frame = bars.copy()
        if frame.index.tz is None:
            frame.index = frame.index.tz_localize("UTC")
        else:
            frame.index = frame.index.tz_convert("UTC")
        volume = frame["volume"].to_numpy(dtype=float) if "volume" in frame.columns else np.zeros(len(frame), dtype=float)
        return pd.DataFrame(
            {
                "ts_utc_ns": frame.index.view("int64").astype("int64"),
                "open": frame["open"].to_numpy(dtype=float),
                "high": frame["high"].to_numpy(dtype=float),
                "low": frame["low"].to_numpy(dtype=float),
                "close": frame["close"].to_numpy(dtype=float),
                "volume": volume,
                "bar_index": np.arange(len(frame), dtype=np.int32),
            }
        )

    @staticmethod
    def _build_portfolio_bars(equity_curve: pd.Series) -> pd.DataFrame:
        series = pd.to_numeric(equity_curve, errors="coerce").astype(float).copy()
        if series.index.tz is None:
            series.index = series.index.tz_localize("UTC")
        else:
            series.index = series.index.tz_convert("UTC")
        opens = series.shift(1).fillna(series.iloc[0])
        highs = np.maximum(opens.to_numpy(dtype=float), series.to_numpy(dtype=float))
        lows = np.minimum(opens.to_numpy(dtype=float), series.to_numpy(dtype=float))
        return pd.DataFrame(
            {
                "open": opens.to_numpy(dtype=float),
                "high": highs,
                "low": lows,
                "close": series.to_numpy(dtype=float),
                "volume": np.zeros(len(series), dtype=float),
            },
            index=series.index,
        )

    @staticmethod
    def _build_portfolio_panes(portfolio_result, equity_curve: pd.Series) -> tuple[dict[str, pd.Series], dict[str, dict]]:
        panes: dict[str, pd.Series] = {}
        styles: dict[str, dict] = {}
        palette = [
            "#4da3ff",
            "#27d07d",
            "#ffcc66",
            "#ff6b6b",
            "#8b7bff",
            "#59c3c3",
            "#f59e0b",
            "#ef476f",
        ]

        equity = pd.to_numeric(equity_curve, errors="coerce").astype(float)
        cash_curve = pd.to_numeric(getattr(portfolio_result, "cash_curve", pd.Series(dtype=float)), errors="coerce").reindex(equity.index)
        cash_weight = (cash_curve / equity.replace(0.0, np.nan)).fillna(0.0)
        panes["Cash Weight"] = cash_weight
        styles["Cash Weight"] = {"color": "#9aa5b1", "line_width": 1.2}

        asset_weights = getattr(portfolio_result, "asset_weights", pd.DataFrame(index=equity.index))
        target_weights = getattr(portfolio_result, "target_weights", pd.DataFrame(index=equity.index))
        for idx, dataset_id in enumerate(asset_weights.columns):
            color = palette[idx % len(palette)]
            actual_name = f"Weight {dataset_id}"
            target_name = f"Target Weight {dataset_id}"
            panes[actual_name] = pd.to_numeric(asset_weights[dataset_id].reindex(equity.index), errors="coerce").fillna(0.0)
            panes[target_name] = pd.to_numeric(target_weights[dataset_id].reindex(equity.index), errors="coerce").fillna(0.0)
            styles[actual_name] = {"color": color, "line_width": 1.5}
            styles[target_name] = {"color": color, "line_width": 1.0}

        return panes, styles

    @staticmethod
    def _build_portfolio_asset_panes(
        portfolio_result,
        *,
        source_dataset_id: str,
        index: pd.Index,
    ) -> tuple[dict[str, pd.Series], dict[str, dict]]:
        panes: dict[str, pd.Series] = {}
        styles: dict[str, dict] = {}
        palette = [
            "#4da3ff",
            "#27d07d",
            "#ffcc66",
            "#ff6b6b",
            "#8b7bff",
            "#59c3c3",
            "#f59e0b",
            "#ef476f",
        ]

        asset_source_dataset_ids = dict(getattr(portfolio_result, "asset_source_dataset_ids", {}) or {})
        asset_weights = getattr(portfolio_result, "asset_weights", pd.DataFrame(index=index)).reindex(index).fillna(0.0)
        target_weights = getattr(portfolio_result, "target_weights", pd.DataFrame(index=index)).reindex(index).fillna(0.0)
        positions = getattr(portfolio_result, "positions", pd.DataFrame(index=index)).reindex(index).fillna(0.0)
        matching_columns = [
            str(column)
            for column in asset_weights.columns
            if str(asset_source_dataset_ids.get(str(column), column)) == str(source_dataset_id)
        ]
        if not matching_columns:
            matching_columns = [str(column) for column in asset_weights.columns if str(column) == str(source_dataset_id)]

        if matching_columns:
            panes["Weight"] = asset_weights[matching_columns].sum(axis=1)
            panes["Target Weight"] = target_weights[matching_columns].sum(axis=1)
            panes["Position"] = positions[matching_columns].sum(axis=1)
            styles["Weight"] = {"color": "#4da3ff", "line_width": 1.6}
            styles["Target Weight"] = {"color": "#ffcc66", "line_width": 1.2}
            styles["Position"] = {"color": "#27d07d", "line_width": 1.2}
            if len(matching_columns) > 1:
                for idx, column in enumerate(matching_columns):
                    color = palette[idx % len(palette)]
                    actual_name = f"Component Weight {column}"
                    target_name = f"Component Target {column}"
                    panes[actual_name] = pd.to_numeric(asset_weights[column], errors="coerce").fillna(0.0)
                    panes[target_name] = pd.to_numeric(target_weights[column], errors="coerce").fillna(0.0)
                    styles[actual_name] = {"color": color, "line_width": 1.1}
                    styles[target_name] = {"color": color, "line_width": 0.9}
        return panes, styles

    @staticmethod
    def build_portfolio_strategy_contexts(request) -> dict[str, list[tuple[str | None, str, Mapping[str, object]]]]:
        contexts: dict[str, list[tuple[str | None, str, Mapping[str, object]]]] = {}
        for asset in list(getattr(request, "assets", []) or []):
            contexts.setdefault(str(asset.dataset_id), []).append(
                (
                    None,
                    str(getattr(asset.strategy_cls, "__name__", "")),
                    dict(getattr(asset, "strategy_params", {}) or {}),
                )
            )
        for block in list(getattr(request, "strategy_blocks", []) or []):
            block_label = str(getattr(block, "display_name", "") or getattr(block, "block_id", "") or getattr(block.strategy_cls, "__name__", "") or "").strip() or None
            strategy_name = str(getattr(block.strategy_cls, "__name__", ""))
            strategy_params = dict(getattr(block, "strategy_params", {}) or {})
            for asset in list(getattr(block, "assets", []) or []):
                contexts.setdefault(str(asset.dataset_id), []).append((block_label, strategy_name, strategy_params))
        return contexts

    @classmethod
    def _build_portfolio_asset_strategy_series(
        cls,
        bars: pd.DataFrame,
        contexts: Sequence[tuple[str | None, str, Mapping[str, object]]],
    ) -> tuple[dict[str, pd.Series], dict[str, pd.Series], dict[str, dict]]:
        overlays: dict[str, pd.Series] = {}
        panes: dict[str, pd.Series] = {}
        styles: dict[str, dict] = {}
        if not contexts:
            return overlays, panes, styles

        multi_context = len(contexts) > 1
        for label, strategy_name, params in contexts:
            prefix = f"{label} | " if multi_context and label else ""
            indicator_map = cls._compute_strategy_indicators(strategy_name, bars, dict(params or {}))
            for name, series in indicator_map.items():
                display_name = f"{prefix}{name}" if prefix else name
                aligned = pd.to_numeric(series.reindex(bars.index), errors="coerce")
                if name in {"ATR", "Z-Score"}:
                    panes[display_name] = aligned
                else:
                    overlays[display_name] = aligned
                styles[display_name] = {"color": cls._indicator_color(name), "line_width": 1.0}
        return overlays, panes, styles

    @staticmethod
    def _indicator_color(name: str) -> str:
        return {
            "SMA Fast": "#4da3ff",
            "SMA Slow": "#ffd166",
            "Half-Life Mean": "#4da3ff",
            "Z-Score": "#ffcc66",
            "Upper": "#27d07d",
            "Lower": "#ff6b6b",
            "Exit Upper": "#7ee787",
            "Exit Lower": "#a28bff",
            "ATR": "#a28bff",
        }.get(name, "#4da3ff")

    @staticmethod
    def _compute_strategy_indicators(strategy_name: str, bars: pd.DataFrame, params: Mapping[str, object]) -> dict[str, pd.Series]:
        out: dict[str, pd.Series] = {}
        if strategy_name == "SMACrossStrategy":
            fast = int(params.get("fast", 10))
            slow = int(params.get("slow", 30))
            out["SMA Fast"] = bars["close"].rolling(fast).mean()
            out["SMA Slow"] = bars["close"].rolling(slow).mean()
        elif strategy_name == "ZScoreMeanReversionStrategy":
            features = compute_zscore_mean_reversion_features(bars, dict(params))
            out["Half-Life Mean"] = features["half_life_mean"]
            out["Z-Score"] = features["z_score"]
        elif strategy_name == "InverseTurtleStrategy":
            entry_len = int(params.get("entry_len", 20))
            exit_len = int(params.get("exit_len", 10))
            atr_len = int(params.get("atr_len", 14))
            use_prev = bool(params.get("use_prev_channels", True))
            upper = bars["high"].rolling(entry_len).max()
            lower = bars["low"].rolling(entry_len).min()
            exit_upper = bars["high"].rolling(exit_len).max()
            exit_lower = bars["low"].rolling(exit_len).min()
            tr = pd.concat(
                [
                    (bars["high"] - bars["low"]).abs(),
                    (bars["high"] - bars["close"].shift(1)).abs(),
                    (bars["low"] - bars["close"].shift(1)).abs(),
                ],
                axis=1,
            ).max(axis=1)
            atr = tr.rolling(atr_len).mean()
            if use_prev:
                upper = upper.shift(1)
                lower = lower.shift(1)
                exit_upper = exit_upper.shift(1)
                exit_lower = exit_lower.shift(1)
                atr = atr.shift(1)
            out["Upper"] = upper
            out["Lower"] = lower
            out["Exit Upper"] = exit_upper
            out["Exit Lower"] = exit_lower
            out["ATR"] = atr
        return out

    @staticmethod
    def _build_series_dataframe(
        bars: pd.DataFrame,
        series_map: Mapping[str, pd.Series],
    ) -> tuple[pd.DataFrame, list[str]]:
        frame = pd.DataFrame(
            {
                "ts_utc_ns": bars.index.view("int64").astype("int64"),
                "bar_index": np.arange(len(bars), dtype=np.int32),
            }
        )
        order: list[str] = []
        used_names: set[str] = set(frame.columns)

        for raw_name, series in series_map.items():
            column_name = sanitize_series_name(raw_name)
            suffix = 2
            while column_name in used_names:
                column_name = f"{column_name}_{suffix}"
                suffix += 1
            used_names.add(column_name)
            order.append(column_name)
            aligned = pd.to_numeric(series.reindex(bars.index), errors="coerce")
            frame[column_name] = aligned.to_numpy(dtype=float)

        return frame, order

    @staticmethod
    def _build_equity_dataframe(bars: pd.DataFrame, equity_curve: pd.Series) -> pd.DataFrame:
        aligned = pd.to_numeric(equity_curve.reindex(bars.index), errors="coerce").astype(float)
        drawdown = (aligned / aligned.cummax()) - 1.0
        return pd.DataFrame(
            {
                "ts_utc_ns": bars.index.view("int64").astype("int64"),
                "bar_index": np.arange(len(bars), dtype=np.int32),
                "equity": aligned.to_numpy(dtype=float),
                "drawdown": drawdown.to_numpy(dtype=float),
            }
        )

    @staticmethod
    def _write_feather(frame: pd.DataFrame, path: Path) -> None:
        table = pa.Table.from_pandas(frame, preserve_index=False)
        feather.write_feather(table, str(path), compression="lz4")

    @staticmethod
    def _iso_utc(ts: pd.Timestamp | str | None) -> str | None:
        if ts is None:
            return None
        timestamp = pd.to_datetime(ts, utc=True, errors="coerce")
        if pd.isna(timestamp):
            return None
        return timestamp.isoformat().replace("+00:00", "Z")

    def _build_manifest(
        self,
        *,
        run,
        bars: pd.DataFrame,
        overlay_order: list[str],
        pane_order: list[str],
        equity_df: pd.DataFrame,
        trades_df: pd.DataFrame,
        files: dict[str, str],
        series_styles: Mapping[str, dict],
    ) -> dict:
        params = json.loads(run.params) if isinstance(run.params, str) else (run.params or {})
        metrics = run.metrics if isinstance(run.metrics, dict) else {}
        overlay_styles = {sanitize_series_name(name): style for name, style in series_styles.items()}
        chart_styles = dict(overlay_styles)
        chart_styles.setdefault("equity", {"color": "#4da3ff", "line_width": 1.3})
        chart_styles.setdefault("drawdown", {"color": "#ff6b6b", "line_width": 1.0})

        return {
            "schema_version": 1,
            "created_at_utc": self._iso_utc(pd.Timestamp.now("UTC")),
            "run_id": str(run.run_id),
            "dataset_id": str(run.dataset_id),
            "strategy": str(run.strategy),
            "params": params,
            "timeframe": str(run.timeframe),
            "base_timeframe": "1 minutes",
            "timezone_display": "America/New_York",
            "start_utc": self._iso_utc(run.start),
            "end_utc": self._iso_utc(run.end),
            "starting_cash": float(run.starting_cash) if getattr(run, "starting_cash", None) is not None else 100_000.0,
            "metrics": metrics,
            "data_format": "feather",
            "files": files,
            "counts": {
                "bars": int(len(bars)),
                "trades": int(len(trades_df)),
                "overlay_series": int(len(overlay_order)),
                "pane_series": int(len(pane_order)),
                "equity_points": int(len(equity_df)),
            },
            "chart_layout": {
                "main_price_series": "candles",
                "overlay_order": overlay_order,
                "pane_order": pane_order,
                "pane_heights": {
                    "price": 3.0,
                    "ATR": 0.8,
                    "equity": 1.0,
                },
                "series_styles": chart_styles,
            },
        }
