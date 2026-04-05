# import pandas as pd
# import numpy as np

# # -----------------------------
# # Config
# # -----------------------------
# CSV_PATH = "AAPLprices_1m_2years.csv"
# OUT_TRADES_CSV = "trades_sma50_150_5mSignal_1mOpenExec.csv"
# TS_COL = "timestamp"
# O,H,L,C = "open","high","low","close"

# FAST, SLOW, TARGET = 50, 150, 1.0   # 100% equity

# # -----------------------------
# # Load 1-minute data
# # -----------------------------
# df_1m = pd.read_csv(CSV_PATH, parse_dates=[TS_COL]).set_index(TS_COL).sort_index()
# df_1m = df_1m[[O,H,L,C]].dropna()

# # -----------------------------
# # Build 5-minute bars for signals
# # -----------------------------
# df_5m = df_1m.resample("4H").agg({O:"first", H:"max", L:"min", C:"last"}).dropna()

# # Indicators on 5-minute close
# close_5m = df_5m[C]
# fast = close_5m.rolling(FAST).mean()
# slow = close_5m.rolling(SLOW).mean()

# # Signal on 5m: 1=long, 0=flat
# signal_5m = (fast > slow).astype(int)

# # Expand to 1-minute timeline (hold until next 5m bar completes)
# signal_1m = signal_5m.reindex(df_1m.index, method="ffill").fillna(0)

# # Execution: position becomes active on the NEXT 1-minute bar
# position = signal_1m.shift(1).fillna(0) * TARGET

# # -----------------------------
# # Return models
# # -----------------------------
# close_1m = df_1m[C]
# open_1m  = df_1m[O]

# ret_c2c = close_1m.pct_change().fillna(0)                 # close->close (approx)
# ret_o2o = open_1m.pct_change().fillna(0)                  # open->open (execution-accurate)
# ret_o2c = ((close_1m - open_1m) / open_1m).fillna(0)      # open->close (mark at close)

# # -----------------------------
# # Metrics
# # -----------------------------
# def compute_metrics(strategy_ret: pd.Series, signal_1m: pd.Series) -> dict:
#     equity = (1.0 + strategy_ret).cumprod()

#     total_return = float(equity.iloc[-1] - 1.0)

#     rolling_max = equity.cummax()
#     drawdown = (equity - rolling_max) / rolling_max
#     max_drawdown = float(drawdown.min())

#     # Sharpe: annualized using ~252 trading days and 390 1-min bars per day
#     bars_per_year = 252 * 390
#     mu = float(strategy_ret.mean())
#     sigma = float(strategy_ret.std(ddof=0))
#     sharpe = float((mu / sigma) * np.sqrt(bars_per_year)) if sigma != 0 else np.nan

#     trades = int((signal_1m.diff().abs() == 1).sum())

#     return {
#         "Total Return": total_return,
#         "Max Drawdown": max_drawdown,
#         "Sharpe": sharpe,
#         "Trades": trades,
#     }

# results = pd.DataFrame([
#     {"PnL Model":"Close→Close (approx)", **compute_metrics(position * ret_c2c, signal_1m)},
#     {"PnL Model":"Open→Open (execution-accurate)", **compute_metrics(position * ret_o2o, signal_1m)},
#     {"PnL Model":"Open→Close (mark at close)", **compute_metrics(position * ret_o2c, signal_1m)},
# ])

# print(results)


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# # -----------------------------
# # USER CONFIG
# # -----------------------------
# CSV_PATH = "AAPLprices_1m_2years.csv"  # <-- change if needed
# OUT_TRADES_CSV = "trades_sma50_150_5mSignal_1mOpenExec.csv"
# OUT_CHART_PNG  = "sma50_150_full_chart.png"

# FAST = 50
# SLOW = 150
# TARGET = 1.0  # 100% equity

# TS_COL = "timestamp"
# O, H, L, C = "open", "high", "low", "close"

# # Full-chart performance controls (optional)
# DOWNSAMPLE_PLOT = False    # True = plot every Nth 5m bar for speed/readability
# DOWNSAMPLE_EVERY = 5        # plot every 5th bar (only used if DOWNSAMPLE_PLOT = True)
# MAX_X_TICKS = 20            # keep x-axis labels readable on huge charts

# # -----------------------------
# # LOAD 1-MINUTE DATA
# # -----------------------------
# df_1m = pd.read_csv(CSV_PATH, parse_dates=[TS_COL]).set_index(TS_COL).sort_index()
# df_1m = df_1m[[O, H, L, C]].dropna()

# # -----------------------------
# # BUILD 5-MINUTE BARS FOR SIGNALS
# # -----------------------------
# df_5m = df_1m.resample("5min").agg({
#     O: "first",
#     H: "max",
#     L: "min",
#     C: "last",
# }).dropna()

# # -----------------------------
# # 5-MINUTE SMA SIGNALS
# # -----------------------------
# close_5m = df_5m[C]
# sma_fast = close_5m.rolling(FAST).mean()
# sma_slow = close_5m.rolling(SLOW).mean()
# signal_5m = (sma_fast > sma_slow).astype(int)  # 1=long, 0=flat

# # Expand to 1m; hold last completed 5m signal until next 5m bar completes
# signal_1m = signal_5m.reindex(df_1m.index, method="ffill").fillna(0)

# # Execution: position becomes active on the NEXT 1-minute bar open
# position = signal_1m.shift(1).fillna(0) * TARGET

# # -----------------------------
# # TRADE LIST (FILLS AT 1M OPEN)
# # -----------------------------
# open_1m = df_1m[O]

# pos_prev = position.shift(1).fillna(0)
# entries = (position > 0) & (pos_prev == 0)
# exits   = (position == 0) & (pos_prev > 0)

# entry_times = df_1m.index[entries.values]
# exit_times  = df_1m.index[exits.values]

# # If strategy ends long, close on last bar open for export completeness
# if len(entry_times) > len(exit_times):
#     exit_times = exit_times.append(pd.Index([df_1m.index[-1]]))

# trades = []
# for et, xt in zip(entry_times, exit_times):
#     ep = float(open_1m.loc[et])
#     xp = float(open_1m.loc[xt])
#     trades.append({
#         "entry_time": et,
#         "exit_time": xt,
#         "entry_price": ep,
#         "exit_price": xp,
#         "return_pct": ((xp / ep) - 1.0) * 100.0 if ep else np.nan,
#         "holding_minutes": int((xt - et).total_seconds() // 60),
#     })

# trades_df = pd.DataFrame(trades)
# trades_df.to_csv(OUT_TRADES_CSV, index=False)
# print(f"Saved trades CSV: {OUT_TRADES_CSV}  (rows={len(trades_df)})")

# # -----------------------------
# # FULL CHART (ALL 5M CANDLES + TRADE MARKERS)
# # -----------------------------
# plot_5m = df_5m.copy()

# # Optional downsampling for plotting speed/readability
# if DOWNSAMPLE_PLOT and DOWNSAMPLE_EVERY > 1:
#     plot_5m = plot_5m.iloc[::DOWNSAMPLE_EVERY].copy()

# x = np.arange(len(plot_5m))

# fig = plt.figure(figsize=(18, 7))
# ax = plt.gca()

# # "Candles" without explicit colors: wick + body as vertical lines
# ax.vlines(x, plot_5m[L].to_numpy(), plot_5m[H].to_numpy(), linewidth=1)
# ax.vlines(x, plot_5m[O].to_numpy(), plot_5m[C].to_numpy(), linewidth=3)

# # Map 1m fill times to 5m bar timestamps (floor)
# entry_5m = pd.to_datetime(trades_df["entry_time"]).dt.floor("5min")
# exit_5m  = pd.to_datetime(trades_df["exit_time"]).dt.floor("5min")

# # Only keep markers that exist in the (possibly downsampled) plot index
# entry_5m = entry_5m[entry_5m.isin(plot_5m.index)]
# exit_5m  = exit_5m[exit_5m.isin(plot_5m.index)]

# # Convert timestamps to x positions and y values (use 5m close for marker y)
# if len(entry_5m) > 0:
#     entry_x = plot_5m.index.get_indexer(entry_5m)
#     entry_y = plot_5m.loc[entry_5m, C].to_numpy()
#     ax.scatter(entry_x, entry_y, marker="^", s=20, label="Entry (long)")

# if len(exit_5m) > 0:
#     exit_x = plot_5m.index.get_indexer(exit_5m)
#     exit_y = plot_5m.loc[exit_5m, C].to_numpy()
#     ax.scatter(exit_x, exit_y, marker="v", s=20, label="Exit")

# # X-axis ticks: keep readable on huge charts
# n = len(plot_5m)
# tick_count = min(MAX_X_TICKS, n)
# tick_step = max(1, n // tick_count)

# ticks = x[::tick_step]
# labels = [plot_5m.index[i].strftime("%Y-%m-%d") for i in range(0, n, tick_step)]
# ax.set_xticks(ticks)
# ax.set_xticklabels(labels, rotation=30, ha="right")

# ax.set_title("Full history: AAPL 5-min candles with SMA(50/150) trades\nSignals on 5m, filled at next 1m open")
# ax.set_ylabel("Price")
# ax.legend()
# plt.tight_layout()

# plt.savefig(OUT_CHART_PNG, dpi=150)
# plt.close(fig)

# print(f"Saved full chart PNG: {OUT_CHART_PNG}")


import pandas as pd
import numpy as np

# -----------------------------
# Config
# -----------------------------
CSV_PATH = "AAPLprices_1m_2years.csv"
OUT_TRADES_CSV = "trades_sma50_150_5mSignal_1mOpenExec.csv"

TS_COL = "timestamp"
O, H, L, C = "open", "high", "low", "close"

FAST, SLOW, TARGET = 50, 150, 1.0   # 100% equity

# -----------------------------
# Load 1-minute data
# -----------------------------
df_1m = pd.read_csv(CSV_PATH, parse_dates=[TS_COL]).set_index(TS_COL).sort_index()
df_1m = df_1m[[O, H, L, C]].dropna()

# -----------------------------
# Build higher-timeframe bars for signals
# -----------------------------
df_5m = df_1m.resample("5min").agg({
    O: "first",
    H: "max",
    L: "min",
    C: "last"
}).dropna()

# -----------------------------
# Indicators
# -----------------------------
close_5m = df_5m[C]
fast = close_5m.rolling(FAST).mean()
slow = close_5m.rolling(SLOW).mean()

# Signal: 1 = long, 0 = flat
signal_5m = (fast > slow).astype(int)

# Expand to 1-minute timeline
signal_1m = signal_5m.reindex(df_1m.index, method="ffill").fillna(0)

# Execution: position active on NEXT 1-minute bar
position = signal_1m.shift(1).fillna(0) * TARGET

# -----------------------------
# Return models
# -----------------------------
close_1m = df_1m[C]
open_1m  = df_1m[O]

ret_c2c = close_1m.pct_change().fillna(0)
ret_o2o = open_1m.pct_change().fillna(0)
ret_o2c = ((close_1m - open_1m) / open_1m).fillna(0)

# -----------------------------
# Metrics
# -----------------------------
def compute_metrics(strategy_ret: pd.Series, signal_1m: pd.Series) -> dict:
    equity = (1.0 + strategy_ret).cumprod()

    total_return = float(equity.iloc[-1] - 1.0)

    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    max_drawdown = float(drawdown.min())

    bars_per_year = 252 * 390
    mu = float(strategy_ret.mean())
    sigma = float(strategy_ret.std(ddof=0))
    sharpe = float((mu / sigma) * np.sqrt(bars_per_year)) if sigma != 0 else np.nan

    trades = int((signal_1m.diff().abs() == 1).sum())

    return {
        "Total Return": total_return,
        "Max Drawdown": max_drawdown,
        "Sharpe": sharpe,
        "Trades": trades,
    }

results = pd.DataFrame([
    {"PnL Model": "Close→Close (approx)", **compute_metrics(position * ret_c2c, signal_1m)},
    {"PnL Model": "Open→Open (execution-accurate)", **compute_metrics(position * ret_o2o, signal_1m)},
    {"PnL Model": "Open→Close (mark at close)", **compute_metrics(position * ret_o2c, signal_1m)},
])

print(results)

# ============================================================
# TRADE EXTRACTION + CSV EXPORT  (NEW)
# ============================================================
pos_prev = position.shift(1).fillna(0)

entries = (position > 0) & (pos_prev == 0)
exits   = (position == 0) & (pos_prev > 0)

entry_times = df_1m.index[entries.values]
exit_times  = df_1m.index[exits.values]

# If strategy ends in a long position, force close at last bar
if len(entry_times) > len(exit_times):
    exit_times = exit_times.append(pd.Index([df_1m.index[-1]]))

trades = []
for et, xt in zip(entry_times, exit_times):
    entry_price = float(open_1m.loc[et])
    exit_price  = float(open_1m.loc[xt])

    trades.append({
        "entry_time": et,
        "exit_time": xt,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "return_pct": ((exit_price / entry_price) - 1.0) * 100.0 if entry_price else np.nan,
        "holding_minutes": int((xt - et).total_seconds() // 60),
    })

trades_df = pd.DataFrame(trades)
trades_df.to_csv(OUT_TRADES_CSV, index=False)

print(f"\nExported {len(trades_df)} trades to {OUT_TRADES_CSV}")
