from __future__ import annotations

import numpy as np
import pandas as pd

from .strategy import Strategy


def _coerce_bool(value, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _rma(series: pd.Series, length: int) -> pd.Series:
    length = max(int(length), 1)
    return series.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def _moving_average(series: pd.Series, length: int, ma_type: str, volume: pd.Series | None = None) -> pd.Series:
    length = max(int(length), 1)
    kind = str(ma_type or "SMA").strip().upper()
    src = series.astype(float)
    if kind == "EMA":
        return src.ewm(span=length, adjust=False, min_periods=length).mean()
    if kind == "RMA":
        return _rma(src, length)
    if kind == "WMA":
        weights = np.arange(1, length + 1, dtype=float)
        weight_sum = float(weights.sum())
        return src.rolling(length).apply(lambda values: float(np.dot(values, weights) / weight_sum), raw=True)
    if kind == "VWMA":
        if volume is None:
            return src.rolling(length).mean()
        vol = volume.astype(float)
        denom = vol.rolling(length).sum().replace(0.0, np.nan)
        return ((src * vol).rolling(length).sum() / denom).replace([np.inf, -np.inf], np.nan)
    return src.rolling(length).mean()


def _linreg_last(series: pd.Series, length: int) -> pd.Series:
    length = max(int(length), 2)
    x = np.arange(length, dtype=float)
    x_mean = float(x.mean())
    x_dev = x - x_mean
    denom = float(np.dot(x_dev, x_dev))
    projection = float(length - 1) - x_mean

    def _calc(values: np.ndarray) -> float:
        if not np.isfinite(values).all():
            return np.nan
        slope = float(np.dot(x_dev, values) / denom)
        return float(values.mean() + slope * projection)

    return series.astype(float).rolling(length).apply(_calc, raw=True)


def _normalize_timeframe_rule(timeframe: str) -> str:
    value = str(timeframe or "").strip().lower()
    if not value:
        return ""
    value = value.replace("minutes", "min").replace("minute", "min").replace("mins", "min")
    value = value.replace("hours", "h").replace("hour", "h").replace("hrs", "h")
    value = value.replace("days", "d").replace("day", "d")
    value = value.replace("weeks", "w").replace("week", "w")
    tokens = value.replace(" ", "")
    num = ""
    unit = ""
    for char in tokens:
        if char.isdigit():
            num += char
        else:
            unit += char
    if not num:
        num = "1"
    if unit in {"min", "m"}:
        return f"{num}min"
    if unit in {"h", "hr"}:
        return f"{num}h"
    if unit in {"d"}:
        return f"{num}D"
    if unit in {"w"}:
        return f"{num}W"
    return tokens


def _resample_ohlcv_left(data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    rule = _normalize_timeframe_rule(timeframe)
    if not rule:
        return data
    try:
        return (
            data.resample(rule, label="left", closed="left")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            .dropna()
        )
    except Exception as exc:
        raise ValueError(f"Invalid regime_timeframe '{timeframe}'.") from exc


def _percent_rank_last(values: np.ndarray) -> float:
    current = values[-1]
    if not np.isfinite(current):
        return np.nan
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.nan
    return float((np.sum(finite <= current) / finite.size) * 100.0)


def _compute_adx(data: pd.DataFrame, length: int, smoothing: int) -> pd.Series:
    length = max(int(length), 1)
    smoothing = max(int(smoothing), 1)
    high = data["high"].astype(float)
    low = data["low"].astype(float)
    close = data["close"].astype(float)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0),
        index=data.index,
        dtype=float,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0),
        index=data.index,
        dtype=float,
    )
    atr = _rma(_true_range(high, low, close), length).replace(0.0, np.nan)
    plus_di = 100.0 * (_rma(plus_dm, length) / atr)
    minus_di = 100.0 * (_rma(minus_dm, length) / atr)
    dx = (100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)).replace(
        [np.inf, -np.inf],
        np.nan,
    )
    return _rma(dx, smoothing)


def compute_zscore_mean_reversion_features(data: pd.DataFrame, params: dict) -> pd.DataFrame:
    src = data["close"].astype(float)
    high = data["high"].astype(float)
    low = data["low"].astype(float)
    prev_close = src.shift(1)

    half_life_lookback = max(int(params.get("half_life_lookback", 100)), 10)
    half_life_factor = max(float(params.get("half_life_factor", 1.5)), 1.0)
    std_len = max(int(params.get("std_len", 20)), 1)
    z_smooth_type = str(params.get("z_smooth_type", "None")).strip().lower()
    z_smooth_len = max(int(params.get("z_smooth_len", 5)), 1)
    use_vol_norm = _coerce_bool(params.get("use_vol_norm", True), True)
    vol_type = str(params.get("vol_type", "ATR")).strip().lower()
    vol_len = max(int(params.get("vol_len", 14)), 1)
    atr_mult = float(params.get("atr_mult", 1.0))
    long_entry_z = float(params.get("long_entry_z", -1.0))
    long_exit_z = float(params.get("long_exit_z", 0.0))
    short_entry_z = float(params.get("short_entry_z", 1.0))
    short_exit_z = float(params.get("short_exit_z", 0.0))

    lag = src.shift(1)
    delta = src - lag
    lag_std = lag.rolling(half_life_lookback).std(ddof=0)
    delta_std = delta.rolling(half_life_lookback).std(ddof=0)
    corr = lag.rolling(half_life_lookback).corr(delta)
    beta = (corr * (delta_std / lag_std.replace(0.0, np.nan))).replace([np.inf, -np.inf], np.nan)
    estimated_half_life = (-np.log(2.0) / beta).where(beta < 0.0)

    fallback_half_life = max(2.0, half_life_lookback / 2.0)
    base_half_life = estimated_half_life.where((estimated_half_life > 0.0) & estimated_half_life.notna(), fallback_half_life)
    effective_half_life = np.maximum(2.0, base_half_life.to_numpy(dtype=float) * half_life_factor)
    half_life_alpha = 1.0 - np.power(0.5, 1.0 / effective_half_life)

    src_values = src.to_numpy(dtype=float)
    half_life_mean = np.empty(len(src_values), dtype=float)
    half_life_mean[0] = src_values[0]
    for i in range(1, len(src_values)):
        alpha = float(half_life_alpha[i]) if np.isfinite(half_life_alpha[i]) else 1.0
        half_life_mean[i] = alpha * src_values[i] + (1.0 - alpha) * half_life_mean[i - 1]
    half_life_mean_series = pd.Series(half_life_mean, index=data.index, name="half_life_mean")

    spread_raw = src - half_life_mean_series
    true_range = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = _rma(true_range, vol_len) * atr_mult
    log_returns = np.log(src / prev_close)
    return_vol = log_returns.rolling(vol_len).std(ddof=0)
    price_vol = return_vol * src
    vol = atr if vol_type == "atr" else price_vol
    if use_vol_norm:
        spread = spread_raw / vol.replace(0.0, np.nan)
    else:
        spread = spread_raw

    spread_mean = spread.rolling(std_len).mean()
    spread_std = spread.rolling(std_len).std(ddof=0).replace(0.0, np.nan)
    spread_z_raw = (spread - spread_mean) / spread_std
    if z_smooth_type == "sma":
        z_score = spread_z_raw.rolling(z_smooth_len).mean()
    elif z_smooth_type == "ema":
        z_score = spread_z_raw.ewm(span=z_smooth_len, adjust=False, min_periods=z_smooth_len).mean()
    else:
        z_score = spread_z_raw

    prev_z = z_score.shift(1)
    long_entry_signal = ((z_score < long_entry_z) & (prev_z >= long_entry_z)).fillna(False)
    long_exit_signal = ((z_score > long_exit_z) & (prev_z <= long_exit_z)).fillna(False)
    short_entry_signal = ((z_score > short_entry_z) & (prev_z <= short_entry_z)).fillna(False)
    short_exit_signal = ((z_score < short_exit_z) & (prev_z >= short_exit_z)).fillna(False)

    long_state_values = np.zeros(len(data), dtype=bool)
    short_state_values = np.zeros(len(data), dtype=bool)
    in_position = False
    in_short = False
    for i, (long_exit_now, long_entry_now, short_exit_now, short_entry_now) in enumerate(
        zip(
            long_exit_signal.to_numpy(dtype=bool),
            long_entry_signal.to_numpy(dtype=bool),
            short_exit_signal.to_numpy(dtype=bool),
            short_entry_signal.to_numpy(dtype=bool),
        )
    ):
        if short_exit_now and in_short:
            in_short = False
        if long_exit_now and in_position:
            in_position = False
        if long_entry_now and not in_position and not in_short:
            in_position = True
        elif short_entry_now and not in_short and not in_position:
            in_short = True
        long_state_values[i] = in_position
        short_state_values[i] = in_short

    return pd.DataFrame(
        {
            "half_life_mean": half_life_mean_series,
            "spread": spread,
            "z_score": z_score,
            "long_entry_signal": long_entry_signal.astype(bool),
            "long_exit_signal": long_exit_signal.astype(bool),
            "short_entry_signal": short_entry_signal.astype(bool),
            "short_exit_signal": short_exit_signal.astype(bool),
            "long_state": pd.Series(long_state_values, index=data.index),
            "short_state": pd.Series(short_state_values, index=data.index),
            "estimated_half_life_bars": estimated_half_life,
            "effective_half_life_bars": pd.Series(effective_half_life, index=data.index),
            "atr": atr,
            "vol": vol,
        },
        index=data.index,
    )


def compute_adaptive_ou_kalman_zscore_features(data: pd.DataFrame, params: dict) -> pd.DataFrame:
    src = data["close"].astype(float)

    ou_lookback = max(int(params.get("ou_lookback", params.get("half_life_lookback", 100))), 10)
    half_life_factor = max(float(params.get("half_life_factor", 1.0)), 0.1)
    min_half_life = max(float(params.get("min_half_life", 2.0)), 1.0)
    max_half_life = max(float(params.get("max_half_life", 250.0)), min_half_life)
    std_half_life_factor = max(float(params.get("std_half_life_factor", 1.0)), 0.1)
    kalman_process_var_ratio = max(float(params.get("kalman_process_var_ratio", 0.01)), 0.0)
    kalman_initial_var_mult = max(float(params.get("kalman_initial_var_mult", 1.0)), 0.001)
    ou_mean_pull = max(float(params.get("ou_mean_pull", 1.0)), 0.0)
    ou_mean_clip_std = max(float(params.get("ou_mean_clip_std", 5.0)), 0.1)
    variance_shock_clip = max(float(params.get("variance_shock_clip", 25.0)), 1.0)
    min_std = max(float(params.get("min_std", 1e-8)), 1e-12)
    max_z_abs = max(float(params.get("max_z_abs", 20.0)), 1.0)
    z_smooth_type = str(params.get("z_smooth_type", "None")).strip().lower()
    z_smooth_len = max(int(params.get("z_smooth_len", 3)), 1)
    exit_signal_mode = str(params.get("exit_signal_mode", "Level")).strip().lower()
    long_entry_z = float(params.get("long_entry_z", -1.0))
    long_exit_z = float(params.get("long_exit_z", 0.0))
    short_entry_z = float(params.get("short_entry_z", 1.0))
    short_exit_z = float(params.get("short_exit_z", 0.0))

    lag = src.shift(1)
    delta = src - lag
    mean_lag = lag.rolling(ou_lookback).mean()
    mean_delta = delta.rolling(ou_lookback).mean()
    lag_var = lag.rolling(ou_lookback).var(ddof=0).replace(0.0, np.nan)
    beta = (lag.rolling(ou_lookback).cov(delta, ddof=0) / lag_var).replace([np.inf, -np.inf], np.nan)
    intercept = mean_delta - beta * mean_lag
    phi = (1.0 + beta).replace([np.inf, -np.inf], np.nan)
    valid_phi = (phi > 0.0) & (phi < 1.0)
    clipped_phi = phi.clip(lower=1e-8, upper=1.0 - 1e-8)
    estimated_half_life = (-np.log(2.0) / np.log(clipped_phi)).where(valid_phi)

    fallback_half_life = max(min_half_life, min(max_half_life, ou_lookback / 2.0))
    base_half_life = estimated_half_life.where(
        (estimated_half_life > 0.0) & estimated_half_life.notna(),
        fallback_half_life,
    )
    effective_half_life = np.clip(
        base_half_life.to_numpy(dtype=float) * half_life_factor,
        min_half_life,
        max_half_life,
    )

    rolling_mean = src.rolling(ou_lookback).mean()
    rolling_var = src.rolling(ou_lookback).var(ddof=0)
    rolling_std = np.sqrt(rolling_var).replace(0.0, np.nan)
    ou_equilibrium = (-intercept / beta.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    ou_equilibrium = ou_equilibrium.where(beta < 0.0)
    ou_equilibrium = ou_equilibrium.clip(
        lower=rolling_mean - rolling_std * ou_mean_clip_std,
        upper=rolling_mean + rolling_std * ou_mean_clip_std,
    )

    residual = (delta - (intercept + beta * lag)).replace([np.inf, -np.inf], np.nan)
    residual_var = residual.rolling(ou_lookback).var(ddof=0)
    stationary_var = (residual_var / (1.0 - phi.pow(2))).where(valid_phi)
    obs_var_hint = stationary_var.where((stationary_var > min_std**2) & stationary_var.notna(), rolling_var)
    obs_var_hint = obs_var_hint.replace([np.inf, -np.inf], np.nan)

    values = src.to_numpy(dtype=float)
    eq_values = ou_equilibrium.to_numpy(dtype=float)
    hint_values = obs_var_hint.to_numpy(dtype=float)
    n = len(values)
    prior_mean = np.full(n, np.nan, dtype=float)
    kalman_mean = np.full(n, np.nan, dtype=float)
    kalman_var = np.full(n, np.nan, dtype=float)
    innovation = np.full(n, np.nan, dtype=float)
    innovation_std = np.full(n, np.nan, dtype=float)
    z_raw_values = np.full(n, np.nan, dtype=float)

    finite_prices = values[np.isfinite(values)]
    initial_price = float(finite_prices[0]) if finite_prices.size else 0.0
    finite_hints = hint_values[np.isfinite(hint_values) & (hint_values > min_std**2)]
    if finite_hints.size:
        initial_obs_var = float(finite_hints[0])
    else:
        initial_obs_var = max(float(np.nanvar(finite_prices)) if finite_prices.size > 1 else 1.0, min_std**2)
    state_mean = initial_price
    state_var = max(initial_obs_var * kalman_initial_var_mult, min_std**2)
    obs_var = max(initial_obs_var, min_std**2)

    for i, price in enumerate(values):
        if not np.isfinite(price):
            if i > 0:
                prior_mean[i] = prior_mean[i - 1]
                kalman_mean[i] = state_mean
                kalman_var[i] = state_var
                innovation_std[i] = np.sqrt(max(obs_var, min_std**2))
            continue

        half_life = float(effective_half_life[i]) if np.isfinite(effective_half_life[i]) else fallback_half_life
        mean_alpha = 1.0 - np.power(0.5, 1.0 / max(half_life, min_half_life))
        eq = float(eq_values[i]) if np.isfinite(eq_values[i]) else np.nan
        if np.isfinite(eq):
            pred_mean = state_mean + min(1.0, ou_mean_pull * mean_alpha) * (eq - state_mean)
        else:
            pred_mean = state_mean

        hint_var = float(hint_values[i]) if np.isfinite(hint_values[i]) and hint_values[i] > min_std**2 else np.nan
        if np.isfinite(hint_var) and i < ou_lookback:
            obs_var = max(hint_var, min_std**2)

        process_var = max(obs_var * kalman_process_var_ratio, 0.0)
        pred_var = max(state_var + process_var, min_std**2)
        denom_var = max(pred_var + obs_var, min_std**2)
        innov = price - pred_mean
        z_raw = innov / np.sqrt(denom_var)
        z_raw = float(np.clip(z_raw, -max_z_abs, max_z_abs))

        gain = pred_var / denom_var
        state_mean = pred_mean + gain * innov
        state_var = max((1.0 - gain) * pred_var, min_std**2)

        var_alpha = 1.0 - np.power(0.5, 1.0 / max(half_life * std_half_life_factor, 1.0))
        shock_limited = min(innov * innov, obs_var * variance_shock_clip)
        obs_var = max((1.0 - var_alpha) * obs_var + var_alpha * shock_limited, min_std**2)

        prior_mean[i] = pred_mean
        kalman_mean[i] = state_mean
        kalman_var[i] = state_var
        innovation[i] = innov
        innovation_std[i] = np.sqrt(denom_var)
        if i >= ou_lookback:
            z_raw_values[i] = z_raw

    z_raw_series = pd.Series(z_raw_values, index=data.index, name="z_score_raw")
    if z_smooth_type == "sma":
        z_score = z_raw_series.rolling(z_smooth_len).mean()
    elif z_smooth_type == "ema":
        z_score = z_raw_series.ewm(span=z_smooth_len, adjust=False, min_periods=z_smooth_len).mean()
    else:
        z_score = z_raw_series

    prev_z = z_score.shift(1)
    long_entry_signal = ((z_score < long_entry_z) & (prev_z >= long_entry_z)).fillna(False)
    short_entry_signal = ((z_score > short_entry_z) & (prev_z <= short_entry_z)).fillna(False)
    if exit_signal_mode in {"cross", "crossing", "crossover"}:
        long_exit_signal = ((z_score > long_exit_z) & (prev_z <= long_exit_z)).fillna(False)
        short_exit_signal = ((z_score < short_exit_z) & (prev_z >= short_exit_z)).fillna(False)
    else:
        long_exit_signal = (z_score >= long_exit_z).fillna(False)
        short_exit_signal = (z_score <= short_exit_z).fillna(False)

    long_state_values = np.zeros(len(data), dtype=bool)
    short_state_values = np.zeros(len(data), dtype=bool)
    in_position = False
    in_short = False
    for i, (long_exit_now, long_entry_now, short_exit_now, short_entry_now) in enumerate(
        zip(
            long_exit_signal.to_numpy(dtype=bool),
            long_entry_signal.to_numpy(dtype=bool),
            short_exit_signal.to_numpy(dtype=bool),
            short_entry_signal.to_numpy(dtype=bool),
        )
    ):
        if short_exit_now and in_short:
            in_short = False
        if long_exit_now and in_position:
            in_position = False
        if long_entry_now and not in_position and not in_short:
            in_position = True
        elif short_entry_now and not in_short and not in_position:
            in_short = True
        long_state_values[i] = in_position
        short_state_values[i] = in_short

    return pd.DataFrame(
        {
            "prior_mean": pd.Series(prior_mean, index=data.index),
            "kalman_mean": pd.Series(kalman_mean, index=data.index),
            "half_life_mean": pd.Series(kalman_mean, index=data.index),
            "ou_equilibrium_mean": ou_equilibrium,
            "spread": pd.Series(innovation, index=data.index),
            "innovation_std": pd.Series(innovation_std, index=data.index),
            "z_score_raw": z_raw_series,
            "z_score": z_score,
            "long_entry_signal": long_entry_signal.astype(bool),
            "long_exit_signal": long_exit_signal.astype(bool),
            "short_entry_signal": short_entry_signal.astype(bool),
            "short_exit_signal": short_exit_signal.astype(bool),
            "long_state": pd.Series(long_state_values, index=data.index),
            "short_state": pd.Series(short_state_values, index=data.index),
            "ou_beta": beta,
            "ou_phi": phi,
            "estimated_half_life_bars": estimated_half_life,
            "effective_half_life_bars": pd.Series(effective_half_life, index=data.index),
            "observation_var": obs_var_hint,
            "kalman_var": pd.Series(kalman_var, index=data.index),
        },
        index=data.index,
    )


def compute_prism_regime_filter_features(data: pd.DataFrame, params: dict) -> pd.DataFrame:
    source = _resample_ohlcv_left(data, str(params.get("regime_timeframe", "")))
    close = source["close"].astype(float)

    use_regime_filter = _coerce_bool(params.get("use_regime_filter", True), True)
    use_bbw = _coerce_bool(params.get("use_bbw_regime", True), True)
    use_adx = _coerce_bool(params.get("use_adx_regime", True), True)
    use_er = _coerce_bool(params.get("use_er_regime", True), True)
    bbw_len = max(int(params.get("bbw_len", params.get("bbwInput", 200))), 1)
    bbw_basis_len = max(int(params.get("bbw_basis_len", 20)), 1)
    bbw_threshold = float(params.get("bbw_percentile_threshold", 50.0))
    adx_len = max(int(params.get("adx_len", 14)), 1)
    adx_smoothing_len = max(int(params.get("adx_smoothing_len", 14)), 1)
    adx_threshold = float(params.get("adx_threshold", 20.0))
    adx_slope_threshold = float(params.get("adx_slope_threshold", 0.0))
    er_len = max(int(params.get("er_len", params.get("erInput", 14))), 1)
    er_threshold = float(params.get("er_threshold", 0.3))

    basis = close.rolling(bbw_basis_len).mean()
    dev = 2.0 * close.rolling(bbw_basis_len).std(ddof=0)
    bbw = (dev / basis.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    bbw_percentile = bbw.rolling(bbw_len, min_periods=bbw_len).apply(_percent_rank_last, raw=True)
    is_bbw_choppy = bbw_percentile < bbw_threshold

    adx = _compute_adx(source, adx_len, adx_smoothing_len)
    adx_slope = adx - adx.shift(1)
    is_adx_choppy = (adx_slope < adx_slope_threshold) | (adx < adx_threshold)

    er_change = (close - close.shift(er_len)).abs()
    er_volatility = close.diff().abs().rolling(er_len).sum()
    er = (er_change / er_volatility.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    is_er_choppy = er < er_threshold

    active_any = bool(use_bbw or use_adx or use_er)
    if not use_regime_filter:
        regime_is_choppy = pd.Series(True, index=source.index)
    else:
        regime_is_choppy = pd.Series(active_any, index=source.index)
        if use_bbw:
            regime_is_choppy &= is_bbw_choppy.fillna(False)
        if use_adx:
            regime_is_choppy &= is_adx_choppy.fillna(False)
        if use_er:
            regime_is_choppy &= is_er_choppy.fillna(False)

    regime = pd.DataFrame(
        {
            "regime_is_choppy": regime_is_choppy.astype(bool),
            "regime_bbw_percentile": bbw_percentile,
            "regime_adx": adx,
            "regime_adx_slope": adx_slope,
            "regime_er": er,
            "regime_bbw_choppy": is_bbw_choppy.fillna(False).astype(bool),
            "regime_adx_choppy": is_adx_choppy.fillna(False).astype(bool),
            "regime_er_choppy": is_er_choppy.fillna(False).astype(bool),
        },
        index=source.index,
    )
    if not regime.index.equals(data.index):
        regime = regime.reindex(data.index, method="ffill")
        bool_cols = ["regime_is_choppy", "regime_bbw_choppy", "regime_adx_choppy", "regime_er_choppy"]
        for column in bool_cols:
            regime[column] = regime[column].fillna(False).astype(bool)
    return regime.reindex(data.index)


def compute_regime_filtered_adaptive_ou_kalman_zscore_features(data: pd.DataFrame, params: dict) -> pd.DataFrame:
    features = compute_adaptive_ou_kalman_zscore_features(data, params)
    regime = compute_prism_regime_filter_features(data, params)
    high = data["high"].astype(float)
    low = data["low"].astype(float)
    close = data["close"].astype(float)
    atr_stop_len = max(int(params.get("atr_stop_len", params.get("atr_len", 14))), 1)
    use_prev_atr = _coerce_bool(params.get("use_prev_atr_for_stop", True), True)
    atr = _rma(_true_range(high, low, close), atr_stop_len)
    atr_stop_distance = atr.shift(1) if use_prev_atr else atr
    atr_stop_mult = max(float(params.get("atr_stop_mult", params.get("atr_mult", 2.0))), 0.0)
    atr_stop_update_mode = str(params.get("atr_stop_update_mode", "Trailing")).strip().lower()

    long_entry_signal = (features["long_entry_signal"] & regime["regime_is_choppy"]).fillna(False)
    short_entry_signal = (features["short_entry_signal"] & regime["regime_is_choppy"]).fillna(False)
    long_exit_signal = features["long_exit_signal"].fillna(False)
    short_exit_signal = features["short_exit_signal"].fillna(False)
    if _coerce_bool(params.get("exit_on_regime_loss", True), True):
        regime_loss = (~regime["regime_is_choppy"]).fillna(True)
        long_exit_signal = (long_exit_signal | regime_loss).fillna(False)
        short_exit_signal = (short_exit_signal | regime_loss).fillna(False)

    long_state_values = np.zeros(len(data), dtype=bool)
    short_state_values = np.zeros(len(data), dtype=bool)
    atr_stop_price_values = np.full(len(data), np.nan, dtype=float)
    in_position = False
    in_short = False
    active_atr_stop = np.nan
    close_values = close.to_numpy(dtype=float)
    atr_distance_values = atr_stop_distance.to_numpy(dtype=float)
    for i, (long_exit_now, long_entry_now, short_exit_now, short_entry_now) in enumerate(
        zip(
            long_exit_signal.to_numpy(dtype=bool),
            long_entry_signal.to_numpy(dtype=bool),
            short_exit_signal.to_numpy(dtype=bool),
            short_entry_signal.to_numpy(dtype=bool),
        )
    ):
        if short_exit_now and in_short:
            in_short = False
        if long_exit_now and in_position:
            in_position = False
            active_atr_stop = np.nan
        if long_entry_now and not in_position and not in_short:
            in_position = True
            atr_distance = atr_distance_values[i]
            if np.isfinite(atr_distance) and atr_distance > 0.0 and atr_stop_mult > 0.0:
                active_atr_stop = close_values[i] - atr_distance * atr_stop_mult
        elif short_entry_now and not in_short and not in_position:
            in_short = True
        elif in_position and atr_stop_update_mode in {"rolling", "dynamic"}:
            atr_distance = atr_distance_values[i]
            if np.isfinite(atr_distance) and atr_distance > 0.0 and atr_stop_mult > 0.0:
                active_atr_stop = close_values[i] - atr_distance * atr_stop_mult
        elif in_position and atr_stop_update_mode in {"trailing", "trail"}:
            atr_distance = atr_distance_values[i]
            if np.isfinite(atr_distance) and atr_distance > 0.0 and atr_stop_mult > 0.0:
                candidate_stop = close_values[i] - atr_distance * atr_stop_mult
                if np.isfinite(active_atr_stop):
                    active_atr_stop = max(active_atr_stop, candidate_stop)
                else:
                    active_atr_stop = candidate_stop
        long_state_values[i] = in_position
        short_state_values[i] = in_short
        if in_position:
            atr_stop_price_values[i] = active_atr_stop

    output = features.copy()
    output["long_entry_signal"] = long_entry_signal.astype(bool)
    output["long_exit_signal"] = long_exit_signal.astype(bool)
    output["short_entry_signal"] = short_entry_signal.astype(bool)
    output["short_exit_signal"] = short_exit_signal.astype(bool)
    output["long_state"] = pd.Series(long_state_values, index=data.index)
    output["short_state"] = pd.Series(short_state_values, index=data.index)
    for column in regime.columns:
        output[column] = regime[column]
    output["atr_stop_distance"] = atr_stop_distance
    output["atr_stop_price"] = pd.Series(atr_stop_price_values, index=data.index)
    return output


def compute_sma_regime_filtered_zscore_mean_reversion_features(data: pd.DataFrame, params: dict) -> pd.DataFrame:
    features = compute_zscore_mean_reversion_features(data, params)
    regime = compute_sma_trend_regime_features(data, params)

    enable_longs = _coerce_bool(params.get("enable_longs", True), True)
    enable_shorts = _coerce_bool(params.get("enable_shorts", True), True)
    use_sma_regime_filter = _coerce_bool(params.get("use_sma_regime_filter", True), True)
    exit_on_regime_loss = _coerce_bool(params.get("exit_on_regime_loss", True), True)
    use_hard_stop_loss = _coerce_bool(
        _param_alias(params, ("use_hard_stop_loss", "useHardStopLoss"), False),
        False,
    )
    hard_stop_loss_pct = max(float(_param_alias(params, ("hard_stop_loss_pct", "hardStopLossPct"), 10.0)), 0.0)
    hard_stop_loss_fraction = hard_stop_loss_pct / 100.0

    if use_sma_regime_filter:
        long_regime_ok = regime["trend_is_bullish"].fillna(False)
        short_regime_ok = regime["trend_is_bearish"].fillna(False)
    else:
        long_regime_ok = pd.Series(True, index=data.index)
        short_regime_ok = pd.Series(True, index=data.index)

    raw_long_entry = (features["long_entry_signal"] & enable_longs & long_regime_ok).fillna(False)
    raw_short_entry = (features["short_entry_signal"] & enable_shorts & short_regime_ok).fillna(False)
    raw_long_exit = features["long_exit_signal"].fillna(False)
    raw_short_exit = features["short_exit_signal"].fillna(False)
    if exit_on_regime_loss:
        raw_long_exit = (raw_long_exit | ~long_regime_ok).fillna(False)
        raw_short_exit = (raw_short_exit | ~short_regime_ok).fillna(False)
    if not enable_longs:
        raw_long_exit = pd.Series(True, index=data.index)
    if not enable_shorts:
        raw_short_exit = pd.Series(True, index=data.index)

    long_entry_values = np.zeros(len(data), dtype=bool)
    long_exit_values = np.zeros(len(data), dtype=bool)
    short_entry_values = np.zeros(len(data), dtype=bool)
    short_exit_values = np.zeros(len(data), dtype=bool)
    long_state_values = np.zeros(len(data), dtype=bool)
    short_state_values = np.zeros(len(data), dtype=bool)
    long_hard_stop_hit_values = np.zeros(len(data), dtype=bool)
    short_hard_stop_hit_values = np.zeros(len(data), dtype=bool)
    long_hard_stop_values = np.full(len(data), np.nan, dtype=float)
    short_hard_stop_values = np.full(len(data), np.nan, dtype=float)
    close_values = data["close"].astype(float).to_numpy(dtype=float)
    in_long = False
    in_short = False
    long_entry_price = np.nan
    short_entry_price = np.nan
    for i, (long_exit_raw, long_entry_raw, short_exit_raw, short_entry_raw) in enumerate(
        zip(
            raw_long_exit.to_numpy(dtype=bool),
            raw_long_entry.to_numpy(dtype=bool),
            raw_short_exit.to_numpy(dtype=bool),
            raw_short_entry.to_numpy(dtype=bool),
        )
    ):
        close_value = float(close_values[i])
        long_hard_stop = np.nan
        short_hard_stop = np.nan
        long_hard_stop_hit = False
        short_hard_stop_hit = False
        if use_hard_stop_loss and hard_stop_loss_fraction > 0.0:
            if in_long and np.isfinite(long_entry_price):
                long_hard_stop = long_entry_price * (1.0 - hard_stop_loss_fraction)
                long_hard_stop_hit = bool(np.isfinite(close_value) and close_value <= long_hard_stop)
            if in_short and np.isfinite(short_entry_price):
                short_hard_stop = short_entry_price * (1.0 + hard_stop_loss_fraction)
                short_hard_stop_hit = bool(np.isfinite(close_value) and close_value >= short_hard_stop)

        long_exit_now = bool(long_exit_raw or long_hard_stop_hit)
        short_exit_now = bool(short_exit_raw or short_hard_stop_hit)
        long_hard_stop_hit_values[i] = bool(long_hard_stop_hit)
        short_hard_stop_hit_values[i] = bool(short_hard_stop_hit)
        long_hard_stop_values[i] = long_hard_stop
        short_hard_stop_values[i] = short_hard_stop

        exited = False
        if short_exit_now and in_short:
            in_short = False
            short_entry_price = np.nan
            short_exit_values[i] = True
            exited = True
        if long_exit_now and in_long:
            in_long = False
            long_entry_price = np.nan
            long_exit_values[i] = True
            exited = True
        if not exited:
            if long_entry_raw and not in_long and not in_short:
                in_long = True
                long_entry_price = close_value if np.isfinite(close_value) else np.nan
                long_entry_values[i] = True
            elif short_entry_raw and not in_short and not in_long:
                in_short = True
                short_entry_price = close_value if np.isfinite(close_value) else np.nan
                short_entry_values[i] = True
        long_state_values[i] = in_long
        short_state_values[i] = in_short

    hard_stop_long_exit = pd.Series(long_hard_stop_hit_values, index=data.index)
    hard_stop_short_exit = pd.Series(short_hard_stop_hit_values, index=data.index)
    final_long_exit = (raw_long_exit | hard_stop_long_exit).fillna(False)
    final_short_exit = (raw_short_exit | hard_stop_short_exit).fillna(False)
    long_hard_stop_price = pd.Series(long_hard_stop_values, index=data.index)
    short_hard_stop_price = pd.Series(short_hard_stop_values, index=data.index)

    output = features.copy()
    output["long_entry_signal"] = raw_long_entry.astype(bool)
    output["long_exit_signal"] = final_long_exit.astype(bool)
    output["short_entry_signal"] = raw_short_entry.astype(bool)
    output["short_exit_signal"] = final_short_exit.astype(bool)
    output["state_long_entry_signal"] = pd.Series(long_entry_values, index=data.index)
    output["state_long_exit_signal"] = pd.Series(long_exit_values, index=data.index)
    output["state_short_entry_signal"] = pd.Series(short_entry_values, index=data.index)
    output["state_short_exit_signal"] = pd.Series(short_exit_values, index=data.index)
    output["long_state"] = pd.Series(long_state_values, index=data.index)
    output["short_state"] = pd.Series(short_state_values, index=data.index)
    output["raw_long_entry_signal"] = raw_long_entry.astype(bool)
    output["raw_short_entry_signal"] = raw_short_entry.astype(bool)
    output["raw_long_exit_signal"] = raw_long_exit.astype(bool)
    output["raw_short_exit_signal"] = raw_short_exit.astype(bool)
    output["long_hard_stop_loss_price"] = long_hard_stop_price
    output["short_hard_stop_loss_price"] = short_hard_stop_price
    output["hard_stop_loss_price"] = long_hard_stop_price.combine_first(short_hard_stop_price)
    output["long_hard_stop_loss_hit"] = hard_stop_long_exit.astype(bool)
    output["short_hard_stop_loss_hit"] = hard_stop_short_exit.astype(bool)
    output["sma_fastest"] = regime["fastest_sma"]
    output["sma_fast"] = regime["fast_sma"]
    output["sma_medium"] = regime["med_sma"]
    output["sma_slow"] = regime["slow_sma"]
    output["sma_slowest"] = regime["slowest_sma"]
    output["sma_regime_state"] = regime["trend_state"]
    output["sma_bull_regime"] = long_regime_ok.astype(bool)
    output["sma_bear_regime"] = short_regime_ok.astype(bool)
    output["regime_filter_long_ok"] = (long_regime_ok & enable_longs).astype(bool)
    output["regime_filter_short_ok"] = (short_regime_ok & enable_shorts).astype(bool)
    return output


def _param_alias(params: dict, names: tuple[str, ...], default):
    for name in names:
        if name in params:
            return params.get(name)
    return default


def _mode_token(value: object) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


RIBBON_TREND_LENGTHS: tuple[int, ...] = tuple(range(5, 105, 5))


def compute_trend_following_ribbon_features(data: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Feature set for the Trend Following Ribbon strategy.

    This ports the TradingView ribbon engine, HTF two-SMA regime filter, price
    distance filter, ribbon-count long lockout, optional close-based short ATR
    trail lockout, and optional close-based hard stop loss.
    """

    close = data["close"].astype(float)
    high = data["high"].astype(float)
    low = data["low"].astype(float)
    volume = data["volume"].astype(float) if "volume" in data else pd.Series(1.0, index=data.index)

    ma_type = str(_param_alias(params, ("ma_type", "matype"), "EMA"))
    trend_check_period = max(int(_param_alias(params, ("trend_check_period", "prd"), 20)), 5)
    trend_channel_rate_pct = max(float(_param_alias(params, ("trend_channel_rate_pct", "rateinp"), 1.0)), 0.1)
    use_linear_regression = _coerce_bool(
        _param_alias(params, ("use_linear_regression", "ulinreg"), True),
        True,
    )
    linear_regression_period = max(int(_param_alias(params, ("linear_regression_period", "linprd"), 10)), 2)
    entry_threshold = max(
        1,
        min(20, int(_param_alias(params, ("entry_ribbon_threshold", "threshold"), 14))),
    )
    exit_threshold = max(
        1,
        min(20, int(_param_alias(params, ("exit_ribbon_threshold", "exitWeakThreshold"), 4))),
    )

    regime_timeframe = str(_param_alias(params, ("regime_timeframe", "regimeTf"), "D"))
    regime_fast_len = max(int(_param_alias(params, ("regime_fast_len", "regimeFastLen"), 50)), 1)
    regime_slow_len = max(int(_param_alias(params, ("regime_slow_len", "regimeSlowLen"), 200)), 1)
    use_regime_lockout = _coerce_bool(
        _param_alias(params, ("use_regime_lockout", "useRegimeLockout", "use_long_regime_lockout", "useLongRegimeLockout"), True),
        True,
    )

    use_short_atr_trail = _coerce_bool(
        _param_alias(params, ("use_short_atr_trail", "useShortAtrTrail"), False),
        False,
    )
    short_atr_len = max(int(_param_alias(params, ("short_atr_len", "shortAtrLen"), 14)), 1)
    short_atr_mult = max(float(_param_alias(params, ("short_atr_mult", "shortAtrMult"), 3.0)), 0.1)

    use_hard_stop_loss = _coerce_bool(
        _param_alias(params, ("use_hard_stop_loss", "useHardStopLoss"), False),
        False,
    )
    hard_stop_loss_pct = max(float(_param_alias(params, ("hard_stop_loss_pct", "hardStopLossPct"), 10.0)), 0.0)
    hard_stop_loss_fraction = hard_stop_loss_pct / 100.0

    use_price_distance_filter = _coerce_bool(
        _param_alias(params, ("use_price_distance_filter", "usePriceDistanceFilter"), True),
        True,
    )
    distance_ma_type = str(_param_alias(params, ("distance_ma_type", "distanceMaType"), "SMA"))
    distance_ma_len = max(int(_param_alias(params, ("distance_ma_len", "distanceMaLen"), 50)), 1)
    distance_norm_mode = _mode_token(_param_alias(params, ("distance_norm_mode", "distanceNormMode"), "ATR"))
    distance_atr_len = max(int(_param_alias(params, ("distance_atr_len", "distanceAtrLen"), 14)), 1)
    min_price_distance = float(_param_alias(params, ("min_price_distance", "minPriceDistance"), -1.0))

    regime_source = _resample_ohlcv_left(data, regime_timeframe)
    regime_close = regime_source["close"].astype(float)
    regime_fast_sma = regime_close.rolling(regime_fast_len).mean()
    regime_slow_sma = regime_close.rolling(regime_slow_len).mean()
    regime = pd.DataFrame(
        {
            "regime_fast_sma": regime_fast_sma,
            "regime_slow_sma": regime_slow_sma,
            "bull_regime": (regime_fast_sma > regime_slow_sma).fillna(False),
            "bear_regime": (regime_fast_sma < regime_slow_sma).fillna(False),
        },
        index=regime_source.index,
    )
    if not regime.index.equals(data.index):
        regime = regime.reindex(data.index, method="ffill")
        for column in ("bull_regime", "bear_regime"):
            regime[column] = regime[column].fillna(False).astype(bool)

    distance_ma = _moving_average(close, distance_ma_len, distance_ma_type, volume=volume)
    distance_atr = _rma(_true_range(high, low, close), distance_atr_len)
    price_distance_raw = close - distance_ma
    price_distance_pct = ((close - distance_ma) / distance_ma.replace(0.0, np.nan) * 100.0).replace(
        [np.inf, -np.inf],
        np.nan,
    )
    price_distance_atr = ((close - distance_ma) / distance_atr.replace(0.0, np.nan)).replace(
        [np.inf, -np.inf],
        np.nan,
    )
    if distance_norm_mode in {"raw"}:
        price_distance = price_distance_raw
    elif distance_norm_mode in {"percent", "pct"}:
        price_distance = price_distance_pct
    else:
        price_distance = price_distance_atr
    if use_price_distance_filter:
        price_distance_ok = (price_distance >= min_price_distance).fillna(False)
    else:
        price_distance_ok = pd.Series(True, index=data.index)

    price_range = high.rolling(280).max() - low.rolling(280).min()
    channel = price_range * (trend_channel_rate_pct / 100.0)
    bull_count = pd.Series(0, index=data.index, dtype=int)
    bear_count = pd.Series(0, index=data.index, dtype=int)
    output: dict[str, pd.Series] = {}
    for length in RIBBON_TREND_LENGTHS:
        ma_source = _moving_average(close, length, ma_type, volume=volume)
        ma = _linreg_last(ma_source, linear_regression_period) if use_linear_regression else ma_source
        highest_ma = ma.rolling(trend_check_period).max()
        lowest_ma = ma.rolling(trend_check_period).min()
        diff = (highest_ma - lowest_ma).abs()
        active = diff > channel
        bullish = (active & (ma > lowest_ma + channel)).fillna(False)
        bearish = (active & (ma < highest_ma - channel)).fillna(False)
        trend = pd.Series(np.select([bullish, bearish], [1, -1], default=0), index=data.index, dtype=int)
        output[f"ma_{length}"] = ma
        output[f"trend_{length}"] = trend
        bull_count = bull_count + (trend == 1).astype(int)
        bear_count = bear_count + (trend == -1).astype(int)

    short_atr = _rma(_true_range(high, low, close), short_atr_len)
    n = len(data)
    long_signal_values = np.zeros(n, dtype=bool)
    short_signal_values = np.zeros(n, dtype=bool)
    long_exit_values = np.zeros(n, dtype=bool)
    short_exit_values = np.zeros(n, dtype=bool)
    long_state_values = np.zeros(n, dtype=bool)
    short_state_values = np.zeros(n, dtype=bool)
    long_locked_values = np.zeros(n, dtype=bool)
    short_locked_values = np.zeros(n, dtype=bool)
    ribbon_bull_values = np.zeros(n, dtype=bool)
    ribbon_bear_values = np.zeros(n, dtype=bool)
    new_ribbon_bull_values = np.zeros(n, dtype=bool)
    close_below_regime_slow_values = np.zeros(n, dtype=bool)
    long_hard_stop_hit_values = np.zeros(n, dtype=bool)
    short_hard_stop_hit_values = np.zeros(n, dtype=bool)
    long_hard_stop_values = np.full(n, np.nan, dtype=float)
    short_hard_stop_values = np.full(n, np.nan, dtype=float)
    short_trail_hit_values = np.zeros(n, dtype=bool)
    short_lowest_values = np.full(n, np.nan, dtype=float)
    short_trail_values = np.full(n, np.nan, dtype=float)

    bull_count_arr = bull_count.to_numpy(dtype=int)
    bear_count_arr = bear_count.to_numpy(dtype=int)
    bull_regime_arr = regime["bull_regime"].fillna(False).to_numpy(dtype=bool)
    bear_regime_arr = regime["bear_regime"].fillna(False).to_numpy(dtype=bool)
    regime_slow_arr = regime["regime_slow_sma"].to_numpy(dtype=float)
    distance_ok_arr = price_distance_ok.fillna(False).to_numpy(dtype=bool)
    low_arr = low.to_numpy(dtype=float)
    close_arr = close.to_numpy(dtype=float)
    short_atr_arr = short_atr.to_numpy(dtype=float)

    in_long = False
    in_short = False
    long_locked = False
    short_locked = False
    short_lowest = np.nan
    long_entry_price = np.nan
    short_entry_price = np.nan
    prev_ribbon_bull_regime = False
    prev_bear_regime = False
    for i in range(n):
        bull_regime_now = bool(bull_regime_arr[i])
        bear_regime_now = bool(bear_regime_arr[i])
        ribbon_bull_now = bool(bull_count_arr[i] >= entry_threshold)
        ribbon_bear_now = bool(bear_count_arr[i] >= entry_threshold)
        new_ribbon_bull_now = ribbon_bull_now and not prev_ribbon_bull_regime
        if use_regime_lockout and new_ribbon_bull_now:
            long_locked = False
        close_value = float(close_arr[i])
        regime_slow_value = float(regime_slow_arr[i])
        close_below_regime_slow = bool(
            np.isfinite(close_value)
            and np.isfinite(regime_slow_value)
            and close_value < regime_slow_value
        )
        if use_regime_lockout and close_below_regime_slow:
            long_locked = True
        if not use_regime_lockout:
            long_locked = False
        if bear_regime_now and not prev_bear_regime:
            short_locked = False
        if not bear_regime_now:
            short_locked = False

        short_trail_stop = np.nan
        short_trail_hit = False
        if in_short:
            low_value = float(low_arr[i])
            if np.isfinite(low_value):
                short_lowest = low_value if not np.isfinite(short_lowest) else min(short_lowest, low_value)
            atr_value = float(short_atr_arr[i])
            if use_short_atr_trail and np.isfinite(short_lowest) and np.isfinite(atr_value):
                short_trail_stop = short_lowest + atr_value * short_atr_mult
                close_value = float(close_arr[i])
                short_trail_hit = bool(np.isfinite(close_value) and close_value > short_trail_stop)
                if short_trail_hit:
                    short_locked = True
        else:
            short_lowest = np.nan

        long_hard_stop = np.nan
        short_hard_stop = np.nan
        long_hard_stop_hit = False
        short_hard_stop_hit = False
        if use_hard_stop_loss and hard_stop_loss_fraction > 0.0:
            if in_long and np.isfinite(long_entry_price):
                long_hard_stop = long_entry_price * (1.0 - hard_stop_loss_fraction)
                long_hard_stop_hit = bool(np.isfinite(close_value) and close_value <= long_hard_stop)
            if in_short and np.isfinite(short_entry_price):
                short_hard_stop = short_entry_price * (1.0 + hard_stop_loss_fraction)
                short_hard_stop_hit = bool(np.isfinite(close_value) and close_value >= short_hard_stop)

        long_signal = (
            ribbon_bull_now
            and bull_regime_now
            and not long_locked
            and not close_below_regime_slow
            and bool(distance_ok_arr[i])
            and not long_hard_stop_hit
        )
        short_signal = (
            ribbon_bear_now
            and bear_regime_now
            and not short_locked
            and bool(distance_ok_arr[i])
            and not short_hard_stop_hit
        )
        long_exit = in_long and (
            bull_count_arr[i] < exit_threshold
            or not bull_regime_now
            or close_below_regime_slow
            or long_hard_stop_hit
        )
        short_exit = in_short and (
            bear_count_arr[i] < exit_threshold
            or not bear_regime_now
            or short_trail_hit
            or short_hard_stop_hit
        )

        long_signal_values[i] = bool(long_signal)
        short_signal_values[i] = bool(short_signal)
        long_exit_values[i] = bool(long_exit)
        short_exit_values[i] = bool(short_exit)
        long_locked_values[i] = bool(long_locked)
        short_locked_values[i] = bool(short_locked)
        ribbon_bull_values[i] = bool(ribbon_bull_now)
        ribbon_bear_values[i] = bool(ribbon_bear_now)
        new_ribbon_bull_values[i] = bool(new_ribbon_bull_now)
        close_below_regime_slow_values[i] = bool(close_below_regime_slow)
        long_hard_stop_hit_values[i] = bool(long_hard_stop_hit)
        short_hard_stop_hit_values[i] = bool(short_hard_stop_hit)
        long_hard_stop_values[i] = long_hard_stop
        short_hard_stop_values[i] = short_hard_stop
        short_trail_hit_values[i] = bool(short_trail_hit)
        short_lowest_values[i] = short_lowest
        short_trail_values[i] = short_trail_stop

        if long_exit:
            in_long = False
            long_entry_price = np.nan
        if short_exit:
            in_short = False
            short_entry_price = np.nan
            short_lowest = np.nan
        if long_signal:
            in_long = True
            in_short = False
            long_entry_price = close_value if np.isfinite(close_value) else np.nan
            short_entry_price = np.nan
            short_lowest = np.nan
        if short_signal:
            in_short = True
            in_long = False
            short_entry_price = close_value if np.isfinite(close_value) else np.nan
            long_entry_price = np.nan

        long_state_values[i] = bool(in_long)
        short_state_values[i] = bool(in_short)
        prev_ribbon_bull_regime = ribbon_bull_now
        prev_bear_regime = bear_regime_now

    output.update(
        {
            "bull_count": bull_count,
            "bear_count": bear_count,
            "regime_fast_sma": regime["regime_fast_sma"],
            "regime_slow_sma": regime["regime_slow_sma"],
            "bull_regime": regime["bull_regime"].fillna(False).astype(bool),
            "bear_regime": regime["bear_regime"].fillna(False).astype(bool),
            "ribbon_bull_regime": pd.Series(ribbon_bull_values, index=data.index),
            "ribbon_bear_regime": pd.Series(ribbon_bear_values, index=data.index),
            "new_ribbon_bull_regime": pd.Series(new_ribbon_bull_values, index=data.index),
            "close_below_regime_slow_sma": pd.Series(close_below_regime_slow_values, index=data.index),
            "distance_ma": distance_ma,
            "distance_atr": distance_atr,
            "price_distance": price_distance,
            "price_distance_raw": price_distance_raw,
            "price_distance_pct": price_distance_pct,
            "price_distance_atr": price_distance_atr,
            "price_distance_ok": price_distance_ok.astype(bool),
            "short_atr": short_atr,
            "short_lowest_since_entry": pd.Series(short_lowest_values, index=data.index),
            "short_atr_trail_stop": pd.Series(short_trail_values, index=data.index),
            "short_atr_trail_hit": pd.Series(short_trail_hit_values, index=data.index),
            "long_hard_stop_loss_price": pd.Series(long_hard_stop_values, index=data.index),
            "short_hard_stop_loss_price": pd.Series(short_hard_stop_values, index=data.index),
            "hard_stop_loss_price": pd.Series(long_hard_stop_values, index=data.index).combine_first(
                pd.Series(short_hard_stop_values, index=data.index)
            ),
            "long_hard_stop_loss_hit": pd.Series(long_hard_stop_hit_values, index=data.index),
            "short_hard_stop_loss_hit": pd.Series(short_hard_stop_hit_values, index=data.index),
            "long_locked_out_bull_regime": pd.Series(long_locked_values, index=data.index),
            "short_locked_out_bear_regime": pd.Series(short_locked_values, index=data.index),
            "long_signal": pd.Series(long_signal_values, index=data.index),
            "short_signal": pd.Series(short_signal_values, index=data.index),
            "long_entry_signal": pd.Series(long_signal_values, index=data.index),
            "short_entry_signal": pd.Series(short_signal_values, index=data.index),
            "long_exit_signal": pd.Series(long_exit_values, index=data.index),
            "short_exit_signal": pd.Series(short_exit_values, index=data.index),
            "long_state": pd.Series(long_state_values, index=data.index),
            "short_state": pd.Series(short_state_values, index=data.index),
        }
    )
    return pd.DataFrame(output, index=data.index)


def compute_sma_trend_regime_features(data: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    SMA trend-state regime features inspired by the TradingView 5-SMA indicator.

    The default regime is bullish when the fastest SMA is above the slowest SMA
    and bearish when it is below. Set regime_mode="stacked" to require the full
    fastest > fast > medium > slow > slowest ordering.
    """

    source = _resample_ohlcv_left(data, str(params.get("regime_timeframe", "")))
    close = source["close"].astype(float)

    fastest_len = max(int(_param_alias(params, ("fastest_len", "fastestLen"), 10)), 1)
    fast_len = max(int(_param_alias(params, ("fast_len", "fastLen", "fast"), 20)), 1)
    med_len = max(int(_param_alias(params, ("med_len", "medium_len", "medLen"), 30)), 1)
    slow_len = max(int(_param_alias(params, ("slow_len", "slowLen", "slow"), 50)), 1)
    slowest_len = max(int(_param_alias(params, ("slowest_len", "slowestLen"), 200)), 1)

    fastest_sma = close.rolling(fastest_len).mean()
    fast_sma = close.rolling(fast_len).mean()
    med_sma = close.rolling(med_len).mean()
    slow_sma = close.rolling(slow_len).mean()
    slowest_sma = close.rolling(slowest_len).mean()

    regime_mode = _mode_token(params.get("regime_mode", "fastest_slowest"))
    if regime_mode in {"stacked", "full_stack", "five_sma_stack", "ordered"}:
        uptrend = (fastest_sma > fast_sma) & (fast_sma > med_sma) & (med_sma > slow_sma) & (slow_sma > slowest_sma)
        downtrend = (fastest_sma < fast_sma) & (fast_sma < med_sma) & (med_sma < slow_sma) & (slow_sma < slowest_sma)
    elif regime_mode in {"fast_slow_confirmed", "confirmed"}:
        uptrend = (fastest_sma > slowest_sma) & (fast_sma > slow_sma) & (med_sma > slow_sma)
        downtrend = (fastest_sma < slowest_sma) & (fast_sma < slow_sma) & (med_sma < slow_sma)
    else:
        uptrend = fastest_sma > slowest_sma
        downtrend = fastest_sma < slowest_sma

    atr_len = max(int(params.get("atr_len", params.get("atr_stop_len", 14))), 1)
    source_atr = _rma(_true_range(source["high"].astype(float), source["low"].astype(float), close), atr_len)
    if _coerce_bool(params.get("confirm_slowest_slope", False), False):
        slope_lookback = max(int(params.get("slope_lookback", 5)), 1)
        min_slope_atr = max(float(params.get("min_slope_atr", 0.0)), 0.0)
        slowest_slope_atr = ((slowest_sma - slowest_sma.shift(slope_lookback)) / source_atr.replace(0.0, np.nan)).replace(
            [np.inf, -np.inf],
            np.nan,
        )
        uptrend &= slowest_slope_atr >= min_slope_atr
        downtrend &= slowest_slope_atr <= -min_slope_atr
    else:
        slowest_slope_atr = ((slowest_sma - slowest_sma.shift(1)) / source_atr.replace(0.0, np.nan)).replace(
            [np.inf, -np.inf],
            np.nan,
        )

    trend_state = pd.Series(np.select([uptrend.fillna(False), downtrend.fillna(False)], [1, -1], default=0), index=source.index)
    allow_neutral_bridge = _coerce_bool(params.get("allow_neutral_bridge", True), True)
    if allow_neutral_bridge:
        prev_direction = trend_state.replace(0, np.nan).ffill().shift(1)
    else:
        prev_direction = trend_state.shift(1)

    long_entry_signal = (trend_state == 1) & (prev_direction == -1)
    short_entry_signal = (trend_state == -1) & (prev_direction == 1)
    if _coerce_bool(params.get("enter_on_initial_state", False), False):
        long_entry_signal |= (trend_state == 1) & prev_direction.isna()
        short_entry_signal |= (trend_state == -1) & prev_direction.isna()

    exit_on_neutral = _coerce_bool(params.get("exit_on_neutral", False), False)
    long_exit_signal = pd.Series(False, index=source.index)
    short_exit_signal = pd.Series(False, index=source.index)
    if exit_on_neutral:
        neutral = trend_state == 0
        long_exit_signal = neutral & (prev_direction == 1)
        short_exit_signal = neutral & (prev_direction == -1)

    regime = pd.DataFrame(
        {
            "fastest_sma": fastest_sma,
            "fast_sma": fast_sma,
            "med_sma": med_sma,
            "slow_sma": slow_sma,
            "slowest_sma": slowest_sma,
            "slowest_slope_atr": slowest_slope_atr,
            "trend_state": trend_state,
            "trend_is_bullish": trend_state == 1,
            "trend_is_bearish": trend_state == -1,
            "long_entry_signal": long_entry_signal.astype(bool),
            "short_entry_signal": short_entry_signal.astype(bool),
            "long_exit_signal": long_exit_signal.astype(bool),
            "short_exit_signal": short_exit_signal.astype(bool),
        },
        index=source.index,
    )
    if not regime.index.equals(data.index):
        regime = regime.reindex(data.index, method="ffill")
        bool_cols = [
            "trend_is_bullish",
            "trend_is_bearish",
            "long_entry_signal",
            "short_entry_signal",
            "long_exit_signal",
            "short_exit_signal",
        ]
        for column in bool_cols:
            regime[column] = regime[column].fillna(False).astype(bool)
        regime["trend_state"] = regime["trend_state"].fillna(0).astype(int)

    high = data["high"].astype(float)
    low = data["low"].astype(float)
    data_close = data["close"].astype(float)
    atr_stop_len = max(int(params.get("atr_stop_len", params.get("atr_len", 14))), 1)
    use_prev_atr = _coerce_bool(params.get("use_prev_atr_for_stop", True), True)
    atr = _rma(_true_range(high, low, data_close), atr_stop_len)
    atr_stop_distance = atr.shift(1) if use_prev_atr else atr
    atr_stop_mult = max(float(params.get("atr_stop_mult", params.get("atr_mult", 3.0))), 0.0)
    atr_stop_update_mode = _mode_token(params.get("atr_stop_update_mode", "Trailing"))
    atr_stop_basis = _mode_token(params.get("atr_stop_basis", "HighLow"))

    long_state_values = np.zeros(len(data), dtype=bool)
    short_state_values = np.zeros(len(data), dtype=bool)
    long_stop_values = np.full(len(data), np.nan, dtype=float)
    short_stop_values = np.full(len(data), np.nan, dtype=float)
    in_long = False
    in_short = False
    active_long_stop = np.nan
    active_short_stop = np.nan
    high_watermark = np.nan
    low_watermark = np.nan
    close_values = data_close.to_numpy(dtype=float)
    high_values = high.to_numpy(dtype=float)
    low_values = low.to_numpy(dtype=float)
    atr_distance_values = atr_stop_distance.to_numpy(dtype=float)
    for i, (long_exit_now, long_entry_now, short_exit_now, short_entry_now) in enumerate(
        zip(
            regime["long_exit_signal"].to_numpy(dtype=bool),
            regime["long_entry_signal"].to_numpy(dtype=bool),
            regime["short_exit_signal"].to_numpy(dtype=bool),
            regime["short_entry_signal"].to_numpy(dtype=bool),
        )
    ):
        if long_exit_now and in_long:
            in_long = False
            active_long_stop = np.nan
            high_watermark = np.nan
        if short_exit_now and in_short:
            in_short = False
            active_short_stop = np.nan
            low_watermark = np.nan
        if long_entry_now:
            in_long = True
            in_short = False
            active_short_stop = np.nan
            low_watermark = np.nan
            high_watermark = close_values[i]
            atr_distance = atr_distance_values[i]
            if np.isfinite(atr_distance) and atr_distance > 0.0 and atr_stop_mult > 0.0:
                active_long_stop = high_watermark - atr_distance * atr_stop_mult
        elif short_entry_now:
            in_short = True
            in_long = False
            active_long_stop = np.nan
            high_watermark = np.nan
            low_watermark = close_values[i]
            atr_distance = atr_distance_values[i]
            if np.isfinite(atr_distance) and atr_distance > 0.0 and atr_stop_mult > 0.0:
                active_short_stop = low_watermark + atr_distance * atr_stop_mult
        elif in_long:
            atr_distance = atr_distance_values[i]
            if np.isfinite(atr_distance) and atr_distance > 0.0 and atr_stop_mult > 0.0:
                basis = close_values[i] if atr_stop_basis in {"close", "closing"} else high_values[i]
                if atr_stop_update_mode in {"trailing", "trail"}:
                    high_watermark = max(high_watermark, basis) if np.isfinite(high_watermark) else basis
                    basis = high_watermark
                candidate_stop = basis - atr_distance * atr_stop_mult
                if atr_stop_update_mode in {"rolling", "dynamic"} or not np.isfinite(active_long_stop):
                    active_long_stop = candidate_stop
                elif atr_stop_update_mode in {"trailing", "trail"}:
                    active_long_stop = max(active_long_stop, candidate_stop)
        elif in_short:
            atr_distance = atr_distance_values[i]
            if np.isfinite(atr_distance) and atr_distance > 0.0 and atr_stop_mult > 0.0:
                basis = close_values[i] if atr_stop_basis in {"close", "closing"} else low_values[i]
                if atr_stop_update_mode in {"trailing", "trail"}:
                    low_watermark = min(low_watermark, basis) if np.isfinite(low_watermark) else basis
                    basis = low_watermark
                candidate_stop = basis + atr_distance * atr_stop_mult
                if atr_stop_update_mode in {"rolling", "dynamic"} or not np.isfinite(active_short_stop):
                    active_short_stop = candidate_stop
                elif atr_stop_update_mode in {"trailing", "trail"}:
                    active_short_stop = min(active_short_stop, candidate_stop)
        long_state_values[i] = in_long
        short_state_values[i] = in_short
        if in_long:
            long_stop_values[i] = active_long_stop
        if in_short:
            short_stop_values[i] = active_short_stop

    output = regime.reindex(data.index)
    output["atr"] = atr
    output["atr_stop_distance"] = atr_stop_distance
    output["long_state"] = pd.Series(long_state_values, index=data.index)
    output["short_state"] = pd.Series(short_state_values, index=data.index)
    output["long_atr_stop_price"] = pd.Series(long_stop_values, index=data.index)
    output["short_atr_stop_price"] = pd.Series(short_stop_values, index=data.index)
    output["atr_stop_price"] = output["long_atr_stop_price"].combine_first(output["short_atr_stop_price"])
    return output


class SMACrossStrategy(Strategy):
    """
    Basic moving-average crossover.

    Parameters:
    - fast: int, fast SMA window
    - slow: int, slow SMA window
    - target: float, position size as fraction of equity (default 1.0)
    """

    def initialize(self, data: pd.DataFrame) -> None:
        fast = int(self.params.get("fast", 10))
        slow = int(self.params.get("slow", 30))
        if fast >= slow:
            raise ValueError("fast window must be < slow window")

        self.target = float(self.params.get("target", 1.0))
        self.data = pd.DataFrame(index=data.index)
        self.data["fast"] = data["close"].rolling(fast).mean()
        self.data["slow"] = data["close"].rolling(slow).mean()

    def on_bar(self, timestamp: pd.Timestamp, bar: pd.Series, broker) -> None:
        row = self.data.loc[timestamp]
        if pd.isna(row["fast"]) or pd.isna(row["slow"]):
            return
        loc = self.data.index.get_loc(timestamp)
        if isinstance(loc, slice) or isinstance(loc, np.ndarray):
            loc = int(np.asarray(loc).nonzero()[0][-1]) if not isinstance(loc, slice) else int(loc.stop - 1)
        if int(loc) <= 0:
            return
        prev = self.data.iloc[int(loc) - 1]
        if pd.isna(prev["fast"]) or pd.isna(prev["slow"]):
            return

        crossed_above = row["fast"] > row["slow"] and prev["fast"] <= prev["slow"]
        crossed_below = row["fast"] < row["slow"] and prev["fast"] >= prev["slow"]

        if crossed_above and broker.position_qty <= 0:
            broker.target_percent(self.target, bar["close"])
        elif crossed_below:
            if getattr(broker, "allow_short", False):
                if broker.position_qty >= 0:
                    broker.target_percent(-self.target, bar["close"])
            elif broker.position_qty > 0:
                broker.target_percent(0.0, bar["close"])


class SMATrendRegimeStrategy(Strategy):
    """
    SMA trend-following regime flip strategy with optional ATR trailing stops.

    Parameters:
    - fastest_len, fast_len, med_len, slow_len, slowest_len: SMA windows
    - regime_mode: fastest_slowest, stacked, or fast_slow_confirmed
    - target: position size as fraction of equity
    - atr_stop_len, atr_stop_mult: trailing stop distance
    - reenter_after_stop: re-enter after an ATR stop while the same regime is still active
    - stop_reentry_mode: Immediate/NextBar or StopRecapture/Recapture
    """

    def initialize(self, data: pd.DataFrame) -> None:
        self.target = float(self.params.get("target", 1.0))
        if self.target < 0:
            raise ValueError("target must be >= 0 for SMA trend regime")
        self.use_atr_stop = _coerce_bool(self.params.get("use_atr_stop", True), True)
        self.reenter_after_stop = _coerce_bool(self.params.get("reenter_after_stop", False), False)
        self.stop_reentry_mode = _mode_token(self.params.get("stop_reentry_mode", self.params.get("reentry_mode", "Immediate")))
        self.atr_stop_mult = max(float(self.params.get("atr_stop_mult", self.params.get("atr_mult", 3.0))), 0.0)
        self.atr_stop_update_mode = _mode_token(self.params.get("atr_stop_update_mode", "Trailing"))
        self.atr_stop_basis = _mode_token(self.params.get("atr_stop_basis", "HighLow"))
        self.atr_stop_tag = "sma_trend_regime_atr_stop"
        self.entry_order_tag = "sma_trend_regime_entry"
        self.active_atr_stop: float | None = None
        self.active_stop_side: str | None = None
        self.pending_reentry_side: str | None = None
        self.pending_reentry_stop: float | None = None
        self.high_watermark: float | None = None
        self.low_watermark: float | None = None
        self.features = compute_sma_trend_regime_features(data, self.params)

    def on_bar(self, timestamp: pd.Timestamp, bar: pd.Series, broker) -> None:
        row = self._feature_row_at(timestamp)
        if row is None:
            return
        self._clear_stale_reentry(row)
        if abs(float(broker.position_qty)) <= 1e-12 and self.active_atr_stop is not None:
            self._reset_stop_state(broker, clear_reentry=False)

        if bool(row.get("long_entry_signal", False)) and broker.position_qty <= 0:
            self._reset_stop_state(broker)
            broker.target_percent(self.target, bar["close"], tag=self.entry_order_tag)
            return
        if bool(row.get("short_entry_signal", False)) and broker.position_qty >= 0:
            self._reset_stop_state(broker)
            broker.target_percent(-self.target, bar["close"], tag=self.entry_order_tag)
            return
        if bool(row.get("long_exit_signal", False)) and broker.position_qty > 0:
            self._reset_stop_state(broker)
            broker.target_percent(0.0, bar["close"])
            return
        if bool(row.get("short_exit_signal", False)) and broker.position_qty < 0:
            self._reset_stop_state(broker)
            broker.target_percent(0.0, bar["close"])
            return

        reentry_target = self._stop_reentry_target(row, bar, broker)
        if reentry_target is not None:
            self._reset_stop_state(broker)
            earliest_ts = pd.Timestamp(timestamp) + pd.Timedelta(nanoseconds=1)
            broker.target_percent(reentry_target, bar["close"], earliest_ts=earliest_ts, tag=self.entry_order_tag)
            return

        if broker.position_qty != 0:
            self._place_or_refresh_atr_stop(timestamp, bar, broker, from_fill=False)

    def on_after_fill(self, timestamp: pd.Timestamp, bar: pd.Series, broker) -> None:
        if broker.position_qty != 0:
            self._place_or_refresh_atr_stop(timestamp, bar, broker, from_fill=True)
            return
        self._remember_stop_reentry()
        self._reset_stop_state(broker, clear_reentry=False)

    def _place_or_refresh_atr_stop(self, timestamp: pd.Timestamp, bar: pd.Series, broker, *, from_fill: bool) -> None:
        if not self.use_atr_stop or self.atr_stop_mult <= 0.0 or broker.position_qty == 0:
            return
        row = self._feature_row_at(timestamp)
        if row is None:
            return
        atr_distance = float(row.get("atr_stop_distance", np.nan))
        if not np.isfinite(atr_distance) or atr_distance <= 0.0:
            return
        if broker.position_qty > 0:
            if self.active_stop_side != "long":
                self.active_atr_stop = None
                self.low_watermark = None
            if from_fill or self.high_watermark is None:
                self.high_watermark = float(broker.avg_price)
            if not from_fill:
                basis_value = float(bar["close"]) if self.atr_stop_basis in {"close", "closing"} else float(bar["high"])
                self.high_watermark = max(float(self.high_watermark), basis_value)
            candidate_stop = float(self.high_watermark) - atr_distance * self.atr_stop_mult
            if self.active_atr_stop is None:
                self.active_atr_stop = candidate_stop
            elif self.atr_stop_update_mode in {"rolling", "dynamic"}:
                self.active_atr_stop = candidate_stop
            elif self.atr_stop_update_mode in {"trailing", "trail"}:
                self.active_atr_stop = max(float(self.active_atr_stop), candidate_stop)
            broker.cancel_orders(tag=self.atr_stop_tag)
            broker.sell_stop(abs(float(broker.position_qty)), float(self.active_atr_stop), tag=self.atr_stop_tag)
            self.active_stop_side = "long"
            return

        if self.active_stop_side != "short":
            self.active_atr_stop = None
            self.high_watermark = None
        if from_fill or self.low_watermark is None:
            self.low_watermark = float(broker.avg_price)
        if not from_fill:
            basis_value = float(bar["close"]) if self.atr_stop_basis in {"close", "closing"} else float(bar["low"])
            self.low_watermark = min(float(self.low_watermark), basis_value)
        candidate_stop = float(self.low_watermark) + atr_distance * self.atr_stop_mult
        if self.active_atr_stop is None:
            self.active_atr_stop = candidate_stop
        elif self.atr_stop_update_mode in {"rolling", "dynamic"}:
            self.active_atr_stop = candidate_stop
        elif self.atr_stop_update_mode in {"trailing", "trail"}:
            self.active_atr_stop = min(float(self.active_atr_stop), candidate_stop)
        broker.cancel_orders(tag=self.atr_stop_tag)
        broker.buy_stop(abs(float(broker.position_qty)), float(self.active_atr_stop), tag=self.atr_stop_tag)
        self.active_stop_side = "short"

    def _remember_stop_reentry(self) -> None:
        if not self.reenter_after_stop:
            return
        if self.active_stop_side not in {"long", "short"} or self.active_atr_stop is None:
            return
        if not np.isfinite(float(self.active_atr_stop)):
            return
        self.pending_reentry_side = self.active_stop_side
        self.pending_reentry_stop = float(self.active_atr_stop)

    def _stop_reentry_target(self, row: pd.Series, bar: pd.Series, broker) -> float | None:
        if not self.reenter_after_stop or broker.position_qty != 0:
            return None
        if self.pending_reentry_side not in {"long", "short"} or self.pending_reentry_stop is None:
            return None
        stop_price = float(self.pending_reentry_stop)
        if not np.isfinite(stop_price):
            return None
        close_price = float(bar["close"])
        immediate_modes = {"immediate", "nextbar", "next_bar", "next_bar_immediate", "same_regime"}
        if self.pending_reentry_side == "long":
            if not bool(row.get("trend_is_bullish", False)):
                return None
            if self.stop_reentry_mode in immediate_modes or close_price > stop_price:
                return self.target
            return None
        if not bool(row.get("trend_is_bearish", False)):
            return None
        if self.stop_reentry_mode in immediate_modes or close_price < stop_price:
            return -self.target
        return None

    def _clear_stale_reentry(self, row: pd.Series) -> None:
        if self.pending_reentry_side == "long" and not bool(row.get("trend_is_bullish", False)):
            self.pending_reentry_side = None
            self.pending_reentry_stop = None
        elif self.pending_reentry_side == "short" and not bool(row.get("trend_is_bearish", False)):
            self.pending_reentry_side = None
            self.pending_reentry_stop = None

    def _reset_stop_state(self, broker, *, clear_reentry: bool = True) -> None:
        broker.cancel_orders(tag=self.atr_stop_tag)
        self.active_atr_stop = None
        self.active_stop_side = None
        self.high_watermark = None
        self.low_watermark = None
        if clear_reentry:
            self.pending_reentry_side = None
            self.pending_reentry_stop = None

    def _feature_row_at(self, timestamp: pd.Timestamp):
        try:
            return self.features.loc[timestamp]
        except KeyError:
            index = self.features.index
            if len(index) == 0:
                return None
            pos = index.searchsorted(pd.Timestamp(timestamp), side="right") - 1
            if pos < 0:
                return None
            return self.features.iloc[int(pos)]


class TrendFollowingRibbonStrategy(Strategy):
    """
    Trend-following ribbon strategy with an HTF SMA regime filter.

    Parameters mirror the TradingView version, using snake_case names:
    - ma_type, trend_check_period, trend_channel_rate_pct
    - use_linear_regression, linear_regression_period
    - entry_ribbon_threshold, exit_ribbon_threshold
    - regime_timeframe, regime_fast_len, regime_slow_len
    - use_short_atr_trail, short_atr_len, short_atr_mult
    - use_hard_stop_loss, hard_stop_loss_pct
    - use_price_distance_filter, distance_ma_type, distance_ma_len
    - distance_norm_mode, distance_atr_len, min_price_distance
    - target: position size as a fraction of equity
    """

    def initialize(self, data: pd.DataFrame) -> None:
        self.target = float(self.params.get("target", 1.0))
        if self.target < 0:
            raise ValueError("target must be >= 0 for trend following ribbon")
        self.features = compute_trend_following_ribbon_features(data, self.params)

    def on_bar(self, timestamp: pd.Timestamp, bar: pd.Series, broker) -> None:
        row = self._feature_row_at(timestamp)
        if row is None:
            return

        if bool(row.get("long_entry_signal", False)) and broker.position_qty <= 0:
            broker.target_percent(self.target, bar["close"])
            return
        if bool(row.get("short_entry_signal", False)) and broker.position_qty >= 0:
            broker.target_percent(-self.target, bar["close"])
            return
        if bool(row.get("long_exit_signal", False)) and broker.position_qty > 0:
            broker.target_percent(0.0, bar["close"])
            return
        if bool(row.get("short_exit_signal", False)) and broker.position_qty < 0:
            broker.target_percent(0.0, bar["close"])

    def _feature_row_at(self, timestamp: pd.Timestamp):
        try:
            return self.features.loc[timestamp]
        except KeyError:
            index = self.features.index
            if len(index) == 0:
                return None
            pos = index.searchsorted(pd.Timestamp(timestamp), side="right") - 1
            if pos < 0:
                return None
            return self.features.iloc[int(pos)]


class InverseTurtleStrategy(Strategy):
    """
    Mean-reversion variant of the Turtle system using breakout channels with ATR stops.

    Parameters:
    - entry_len: int, lookback for breakout channel (default 20, uses prior bars)
    - exit_len: int, lookback for exit channel (default 10, uses prior bars)
    - atr_len: int, ATR length (default 14)
    - atr_mult: float, ATR multiplier for stop distance (default 2.0)
    - target: float, position size as fraction of equity (default 0.1 to mirror TV 10% equity)
    - use_atr_stop: bool, whether to apply ATR-based stops (default True)
    - use_prev_channels: bool, whether to shift channels/ATR by one bar to avoid lookahead (default True)
    """

    def initialize(self, data: pd.DataFrame) -> None:
        entry_len = int(self.params.get("entry_len", 20))
        exit_len = int(self.params.get("exit_len", 10))
        atr_len = int(self.params.get("atr_len", 14))
        self.atr_mult = float(self.params.get("atr_mult", 2.0))
        self.target = float(self.params.get("target", 1.0))
        self.use_atr_stop = bool(self.params.get("use_atr_stop", True))
        self.use_prev_channels = bool(self.params.get("use_prev_channels", True))

        df = pd.DataFrame(index=data.index)
        # Use prior-bar channels to better match Pine's non-repainting behaviour.
        upper = data["high"].rolling(entry_len).max()
        lower = data["low"].rolling(entry_len).min()
        exit_upper = data["high"].rolling(exit_len).max()
        exit_lower = data["low"].rolling(exit_len).min()
        if self.use_prev_channels:
            upper = upper.shift(1)
            lower = lower.shift(1)
            exit_upper = exit_upper.shift(1)
            exit_lower = exit_lower.shift(1)
        df["upper"] = upper
        df["lower"] = lower
        df["exit_upper"] = exit_upper
        df["exit_lower"] = exit_lower

        high = data["high"]
        low = data["low"]
        close = data["close"]
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(atr_len).mean()
        if self.use_prev_channels:
            atr = atr.shift(1)
        df["atr"] = atr

        self.indicators = df

    def on_bar(self, timestamp: pd.Timestamp, bar: pd.Series, broker) -> None:
        row = self.indicators.loc[timestamp]
        if row.isna().any():
            return

        price = bar["close"]
        # Entries
        if bar["low"] <= row["lower"] and broker.position_qty <= 0:
            broker.target_percent(self.target, price)
        elif bar["high"] >= row["upper"] and broker.position_qty >= 0:
            broker.target_percent(-self.target, price)

        # Channel exits
        if broker.position_qty > 0 and bar["high"] >= row["exit_upper"]:
            broker.target_percent(0.0, price)
        elif broker.position_qty < 0 and bar["low"] <= row["exit_lower"]:
            broker.target_percent(0.0, price)

        # ATR stops
        if self.use_atr_stop and broker.position_qty != 0:
            broker.cancel_orders(tag="atr_stop")
            if broker.position_qty > 0:
                stop = broker.avg_price - row["atr"] * self.atr_mult
                broker.sell_stop(abs(broker.position_qty), stop, tag="atr_stop")
            elif broker.position_qty < 0:
                stop = broker.avg_price + row["atr"] * self.atr_mult
                broker.buy_stop(abs(broker.position_qty), stop, tag="atr_stop")


class ZScoreMeanReversionStrategy(Strategy):
    """
    Long-only, fixed-target simplification of the TradingView z-score mean-reversion strategy.

    Intentionally omitted for version 1:
    - regime filter
    - dynamic equity / annual-vol sizing
    - short entries
    """

    def initialize(self, data: pd.DataFrame) -> None:
        self.target = float(self.params.get("target", 1.0))
        if self.target < 0:
            raise ValueError("target must be >= 0 for long-only z-score mean reversion")
        self.features = compute_zscore_mean_reversion_features(data, self.params)

    def on_bar(self, timestamp: pd.Timestamp, bar: pd.Series, broker) -> None:
        row = self.features.loc[timestamp]
        if bool(row.get("long_exit_signal", False)) and broker.position_qty > 0:
            broker.target_percent(0.0, bar["close"])
            return
        if bool(row.get("long_entry_signal", False)) and broker.position_qty <= 0:
            broker.target_percent(self.target, bar["close"])


class SMARegimeFilteredZScoreMeanReversionStrategy(Strategy):
    """
    Z-score mean-reversion strategy gated by an SMA trend regime.

    Long and short z-score thresholds are independent, and either side can be
    disabled with enable_longs / enable_shorts.
    """

    def initialize(self, data: pd.DataFrame) -> None:
        self.target = float(self.params.get("target", 1.0))
        if self.target < 0:
            raise ValueError("target must be >= 0 for SMA regime-filtered z-score mean reversion")
        self.enable_longs = _coerce_bool(self.params.get("enable_longs", True), True)
        self.enable_shorts = _coerce_bool(self.params.get("enable_shorts", True), True)
        self.features = compute_sma_regime_filtered_zscore_mean_reversion_features(data, self.params)

    def on_bar(self, timestamp: pd.Timestamp, bar: pd.Series, broker) -> None:
        row = self._feature_row_at(timestamp)
        if row is None:
            return
        if (not self.enable_longs) and broker.position_qty > 0:
            broker.target_percent(0.0, bar["close"])
            return
        if (not self.enable_shorts) and broker.position_qty < 0:
            broker.target_percent(0.0, bar["close"])
            return
        if bool(row.get("long_exit_signal", False)) and broker.position_qty > 0:
            broker.target_percent(0.0, bar["close"])
            return
        if bool(row.get("short_exit_signal", False)) and broker.position_qty < 0:
            broker.target_percent(0.0, bar["close"])
            return
        if bool(row.get("long_entry_signal", False)) and broker.position_qty <= 0:
            broker.target_percent(self.target, bar["close"])
            return
        if bool(row.get("short_entry_signal", False)) and broker.position_qty >= 0:
            broker.target_percent(-self.target, bar["close"])

    def _feature_row_at(self, timestamp: pd.Timestamp):
        try:
            return self.features.loc[timestamp]
        except KeyError:
            index = self.features.index
            if len(index) == 0:
                return None
            pos = index.searchsorted(pd.Timestamp(timestamp), side="right") - 1
            if pos < 0:
                return None
            return self.features.iloc[int(pos)]


class AdaptiveOUKalmanZScoreStrategy(Strategy):
    """
    Long-only z-score mean-reversion strategy using a rolling OU fit and Kalman fair value.

    This is intentionally a sibling of ZScoreMeanReversionStrategy rather than a replacement.
    """

    def initialize(self, data: pd.DataFrame) -> None:
        self.target = float(self.params.get("target", 1.0))
        if self.target < 0:
            raise ValueError("target must be >= 0 for long-only adaptive OU Kalman z-score mean reversion")
        self.features = compute_adaptive_ou_kalman_zscore_features(data, self.params)

    def on_bar(self, timestamp: pd.Timestamp, bar: pd.Series, broker) -> None:
        row = self.features.loc[timestamp]
        if bool(row.get("long_exit_signal", False)) and broker.position_qty > 0:
            broker.target_percent(0.0, bar["close"])
            return
        if bool(row.get("long_entry_signal", False)) and broker.position_qty <= 0:
            broker.target_percent(self.target, bar["close"])


class RegimeFilteredAdaptiveOUKalmanZScoreStrategy(Strategy):
    """
    Adaptive OU/Kalman z-score mean reversion gated by a choppy-regime filter with ATR stops.
    """

    def initialize(self, data: pd.DataFrame) -> None:
        self.target = float(self.params.get("target", 1.0))
        if self.target < 0:
            raise ValueError("target must be >= 0 for long-only regime-filtered adaptive OU Kalman z-score mean reversion")
        self.use_atr_stop = _coerce_bool(self.params.get("use_atr_stop", True), True)
        self.atr_stop_mult = max(float(self.params.get("atr_stop_mult", self.params.get("atr_mult", 2.0))), 0.0)
        self.atr_stop_update_mode = str(self.params.get("atr_stop_update_mode", "Trailing")).strip().lower()
        self.atr_stop_tag = "regime_ou_kalman_atr_stop"
        self.entry_order_tag = "regime_ou_kalman_entry"
        self.require_regime_on_fill = _coerce_bool(self.params.get("require_regime_on_fill", True), True)
        self.active_atr_stop: float | None = None
        self.features = compute_regime_filtered_adaptive_ou_kalman_zscore_features(data, self.params)

    def on_before_fill(self, timestamp: pd.Timestamp, bar: pd.Series, broker) -> None:
        if not self.require_regime_on_fill:
            return
        row = self._feature_row_at(timestamp)
        if row is None:
            return
        if not bool(row.get("regime_is_choppy", False)):
            broker.cancel_orders(tag=self.entry_order_tag)

    def on_bar(self, timestamp: pd.Timestamp, bar: pd.Series, broker) -> None:
        row = self._feature_row_at(timestamp)
        if row is None:
            return
        if broker.position_qty <= 0 and self.active_atr_stop is not None:
            broker.cancel_orders(tag=self.atr_stop_tag)
            self.active_atr_stop = None

        if bool(row.get("long_exit_signal", False)) and broker.position_qty > 0:
            broker.cancel_orders(tag=self.atr_stop_tag)
            self.active_atr_stop = None
            broker.target_percent(0.0, bar["close"])
            return

        if broker.position_qty > 0:
            self._place_or_refresh_atr_stop(timestamp, broker)
            return

        if bool(row.get("long_entry_signal", False)) and broker.position_qty <= 0:
            broker.target_percent(self.target, bar["close"], tag=self.entry_order_tag)

    def on_after_fill(self, timestamp: pd.Timestamp, bar: pd.Series, broker) -> None:
        if broker.position_qty > 0:
            self._place_or_refresh_atr_stop(timestamp, broker)
            return
        broker.cancel_orders(tag=self.atr_stop_tag)
        self.active_atr_stop = None

    def _place_or_refresh_atr_stop(self, timestamp: pd.Timestamp, broker) -> None:
        if not self.use_atr_stop or self.atr_stop_mult <= 0.0 or broker.position_qty <= 0:
            return
        row = self._feature_row_at(timestamp)
        if row is None:
            return
        atr_distance = float(row.get("atr_stop_distance", np.nan))
        if not np.isfinite(atr_distance) or atr_distance <= 0.0:
            return
        candidate_stop = float(broker.avg_price) - atr_distance * self.atr_stop_mult
        if self.active_atr_stop is None:
            self.active_atr_stop = candidate_stop
        elif self.atr_stop_update_mode in {"rolling", "dynamic"}:
            self.active_atr_stop = candidate_stop
        elif self.atr_stop_update_mode in {"trailing", "trail"}:
            self.active_atr_stop = max(float(self.active_atr_stop), candidate_stop)
        broker.cancel_orders(tag=self.atr_stop_tag)
        broker.sell_stop(abs(float(broker.position_qty)), float(self.active_atr_stop), tag=self.atr_stop_tag)

    def _feature_row_at(self, timestamp: pd.Timestamp):
        try:
            return self.features.loc[timestamp]
        except KeyError:
            index = self.features.index
            if len(index) == 0:
                return None
            pos = index.searchsorted(pd.Timestamp(timestamp), side="right") - 1
            if pos < 0:
                return None
            return self.features.iloc[int(pos)]
