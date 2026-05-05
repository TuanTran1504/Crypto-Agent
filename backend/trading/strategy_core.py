"""
strategy_core.py - Technical analysis, gate logic, and trade planning.

Shared between the live engine and the backtester.
No exchange calls here - only pure DataFrame computation.

Public API:
  compute_indicators(df_h1, df_m15, df_m5) -> dict
  check_macro_bias(ml_pred, fear_greed, funding_rate, symbol) -> (bool, str, str)
  check_technical_gates(context) -> (bool, str)
  find_sr_levels(df_h1, current_price, df_m15) -> dict
  build_trade_plan(signal, setup_name, context, df_exec) -> (dict | None, str)
  validate_ai_trade_decision(decision, context, df_exec) -> (bool, str)
  build_rule_based_signal_decision(context, df_exec) -> dict
  compute_score(context) -> (int, list[str])
  classify_trend(gap_pct, ema34, ema89, threshold, atr_pct) -> str
"""

import os
from typing import Any

import numpy as np
import pandas as pd

H1_TREND_GAP = 0.8
M15_TREND_GAP = 0.4
BTC_TREND_GAP = 1.0
ATR_VOLATILE_PCT = 0.8
TREND_HYSTERESIS_PCT = 0.10
TREND_CONFIRM_BARS = 2
FUNDING_THRESHOLD = 0.05
SCORE_THRESHOLD = 3
ML_CONFIDENCE_MIN = 0.55

SL_MAX_PCT = 0.008
SL_MIN_PCT = 0.002
BREAKOUT_SL_MAX_PCT = 0.013
MIN_RR = 1.5

ROLLOVER_ATR_MULT = 0.75


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(default)


def _sr_buffer_distance(entry: float, atr: float, is_range: bool) -> float:
    default_mult = 0.30 if is_range else 0.25
    buffer_mult = max(0.0, _env_float("SL_SR_BUFFER_ATR_MULT", default_mult))
    return atr * buffer_mult if atr > 0 and entry > 0 else entry * (0.0018 if is_range else 0.0015)


def _min_risk_distance(entry: float, atr: float, is_range: bool) -> float:
    base_floor = max(entry * SL_MIN_PCT, atr * (0.45 if is_range else 0.55))
    atr_floor_mult = max(0.0, _env_float("SL_ATR_MULTIPLIER", 1.8))
    atr_floor = atr * atr_floor_mult if atr > 0 else 0.0
    return max(base_floor, atr_floor)


def _tp_extension_distance(entry: float, atr: float, is_range: bool) -> float:
    default_mult = 0.10 if is_range else 0.15
    extension_mult = max(0.0, _env_float("TP_EXTENSION_ATR_MULT", default_mult))
    min_pct = max(0.0, _env_float("TP_EXTENSION_MIN_PCT", 0.0))
    atr_extension = atr * extension_mult if atr > 0 else 0.0
    pct_extension = entry * min_pct if entry > 0 else 0.0
    return max(atr_extension, pct_extension)


def compute_max_stop_pct(setup_name: str, context: dict,
                         entry: float | None = None,
                         atr: float | None = None) -> float:
    """Shared stop-distance cap so planning and execution use identical limits."""
    setup_code = _setup_code(setup_name)
    base_max_stop = BREAKOUT_SL_MAX_PCT if setup_code in ("B", "C") else SL_MAX_PCT

    resolved_entry = float(entry if entry is not None else (context.get("current_price") or 0.0))
    resolved_atr = float(
        atr if atr is not None else (context.get("atr_m15") or context.get("atr") or 0.0)
    )
    atr_pct = resolved_atr / resolved_entry if resolved_entry > 0 and resolved_atr > 0 else 0.0
    dynamic_mult = max(1.0, _env_float("SL_ATR_DYNAMIC_MULT", 3.0))
    ceiling = max(base_max_stop, _env_float("SL_MAX_PCT_CEILING", 0.04))
    return min(max(base_max_stop, atr_pct * dynamic_mult), ceiling)


def classify_trend(gap_pct: float, ema34: float, ema89: float,
                   threshold: float, atr_pct: float = 0.0,
                   prev_gap_pct: float | None = None,
                   prev_ema34: float | None = None,
                   prev_ema89: float | None = None) -> str:
    """
    UPTREND / DOWNTREND / VOLATILE_RANGE / SIDEWAY
    """
    hysteresis = max(0.0, TREND_HYSTERESIS_PCT)
    enter_threshold = threshold + (hysteresis * 0.5)
    exit_threshold = max(0.0, threshold - (hysteresis * 0.5))

    if (
        TREND_CONFIRM_BARS >= 2
        and prev_gap_pct is not None
        and prev_ema34 is not None
        and prev_ema89 is not None
    ):
        now_up = ema34 > ema89
        now_down = ema34 < ema89
        prev_up = prev_ema34 > prev_ema89
        prev_down = prev_ema34 < prev_ema89

        prev_was_uptrend = prev_up and prev_gap_pct >= enter_threshold
        prev_was_downtrend = prev_down and prev_gap_pct >= enter_threshold

        if now_up and gap_pct >= exit_threshold:
            if prev_was_uptrend or (
                prev_up and prev_gap_pct >= exit_threshold and gap_pct >= enter_threshold
            ):
                return "UPTREND"

        if now_down and gap_pct >= exit_threshold:
            if prev_was_downtrend or (
                prev_down and prev_gap_pct >= exit_threshold and gap_pct >= enter_threshold
            ):
                return "DOWNTREND"
    elif gap_pct >= threshold:
        return "UPTREND" if ema34 > ema89 else "DOWNTREND"

    if atr_pct >= ATR_VOLATILE_PCT:
        return "VOLATILE_RANGE"
    return "SIDEWAY"


def _nearest_liquidity_levels(df_m5: pd.DataFrame, current_price: float,
                              lookback: int = 90, swing_window: int = 2,
                              tol_pct: float = 0.12, min_touches: int = 2) -> tuple[float | None, float | None]:
    """
    Lightweight liquidity hint from repeated swing highs/lows.
    Returns (nearest_ssl_below_price, nearest_bsl_above_price).
    """
    if len(df_m5) < max(12, swing_window * 2 + 3) or current_price <= 0:
        return None, None

    src = df_m5.tail(max(lookback, swing_window * 2 + 3)).reset_index(drop=True)
    highs = src["high"].astype(float).to_numpy()
    lows = src["low"].astype(float).to_numpy()

    swing_highs: list[float] = []
    swing_lows: list[float] = []
    n = len(src)
    for i in range(swing_window, n - swing_window):
        left_h = highs[i - swing_window:i]
        right_h = highs[i + 1:i + swing_window + 1]
        left_l = lows[i - swing_window:i]
        right_l = lows[i + 1:i + swing_window + 1]
        if highs[i] >= left_h.max() and highs[i] >= right_h.max():
            swing_highs.append(float(highs[i]))
        if lows[i] <= left_l.min() and lows[i] <= right_l.min():
            swing_lows.append(float(lows[i]))

    def _cluster(values: list[float]) -> list[dict]:
        if not values:
            return []
        sorted_vals = sorted(values)
        clusters: list[list[float]] = [[sorted_vals[0]]]
        for v in sorted_vals[1:]:
            prev = clusters[-1][-1]
            rel = abs(v - prev) / max(abs(prev), 1e-9) * 100
            if rel <= tol_pct:
                clusters[-1].append(v)
            else:
                clusters.append([v])
        return [{"price": float(np.mean(c)), "touches": len(c)} for c in clusters]

    high_clusters = _cluster(swing_highs)
    low_clusters = _cluster(swing_lows)

    bsl_candidates = [c["price"] for c in high_clusters if c["touches"] >= min_touches and c["price"] > current_price]
    ssl_candidates = [c["price"] for c in low_clusters if c["touches"] >= min_touches and c["price"] < current_price]

    nearest_bsl = min(bsl_candidates, key=lambda p: p - current_price) if bsl_candidates else None
    nearest_ssl = max(ssl_candidates, key=lambda p: p) if ssl_candidates else None
    return nearest_ssl, nearest_bsl



def detect_m15_rollover(raw_trend: str, df_m15: pd.DataFrame,
                        atr: float, rsi: float) -> tuple[bool, str]:
    """
    Detects when an M15 trend still qualifies by slow EMA rules but has already
    deteriorated into a short-term rollover.
    """
    if raw_trend not in ("UPTREND", "DOWNTREND") or len(df_m15) < 5:
        return False, ""

    closes = df_m15["close"].astype(float)
    highs = df_m15["high"].astype(float)
    lows = df_m15["low"].astype(float)
    ema34_series = closes.ewm(span=34, adjust=False).mean()
    ema89_series = closes.ewm(span=89, adjust=False).mean()

    last_close = float(closes.iloc[-1])
    ema34_last = float(ema34_series.iloc[-1])
    ema34_prev2 = float(ema34_series.iloc[-3])
    ema89_last = float(ema89_series.iloc[-1])
    three_bar_move = float(closes.iloc[-1] - closes.iloc[-4])

    if raw_trend == "UPTREND":
        below_ema34 = last_close < ema34_last
        below_ema89 = last_close < ema89_last
        ema_slope_down = ema34_last <= ema34_prev2
        lower_highs = bool(highs.iloc[-1] < highs.iloc[-2] <= highs.iloc[-3])
        sharp_drop = atr > 0 and three_bar_move <= -(atr * ROLLOVER_ATR_MULT)
        weak_rsi = rsi < 48

        if below_ema89 and sharp_drop:
            return True, (
                f"M15 uptrend rolling over: close {last_close:.4f} below EMA89 "
                f"{ema89_last:.4f} after {abs(three_bar_move):.4f} drop"
            )
        if below_ema34 and (ema_slope_down or lower_highs) and (sharp_drop or weak_rsi):
            return True, (
                f"M15 uptrend rolling over: close {last_close:.4f} below EMA34 "
                f"{ema34_last:.4f} with weakening structure"
            )
        return False, ""

    above_ema34 = last_close > ema34_last
    above_ema89 = last_close > ema89_last
    ema_slope_up = ema34_last >= ema34_prev2
    higher_lows = bool(lows.iloc[-1] > lows.iloc[-2] >= lows.iloc[-3])
    sharp_rally = atr > 0 and three_bar_move >= atr * ROLLOVER_ATR_MULT
    strong_rsi = rsi > 52

    if above_ema89 and sharp_rally:
        return True, (
            f"M15 downtrend rolling over: close {last_close:.4f} above EMA89 "
            f"{ema89_last:.4f} after {abs(three_bar_move):.4f} rally"
        )
    if above_ema34 and (ema_slope_up or higher_lows) and (sharp_rally or strong_rsi):
        return True, (
            f"M15 downtrend rolling over: close {last_close:.4f} above EMA34 "
            f"{ema34_last:.4f} with improving structure"
        )
    return False, ""


def compute_indicators(df_h1: pd.DataFrame, df_m15: pd.DataFrame,
                       df_m5: pd.DataFrame) -> dict:
    """
    Computes all indicators needed for gates + scoring + chart.
    Returns a flat context dict.
    """
    df_h1 = df_h1.copy()
    df_h1["ema34"] = df_h1["close"].ewm(span=34, adjust=False).mean()
    df_h1["ema89"] = df_h1["close"].ewm(span=89, adjust=False).mean()

    df_h1["tr"] = np.maximum(
        df_h1["high"] - df_h1["low"],
        np.maximum(
            abs(df_h1["high"] - df_h1["close"].shift(1)),
            abs(df_h1["low"] - df_h1["close"].shift(1)),
        ),
    )
    df_h1["atr_h1"] = df_h1["tr"].ewm(span=14, adjust=False).mean()

    h1_ema34 = df_h1["ema34"].iloc[-1]
    h1_ema89 = df_h1["ema89"].iloc[-1]
    h1_ema34_prev = df_h1["ema34"].iloc[-2] if len(df_h1) >= 2 else h1_ema34
    h1_ema89_prev = df_h1["ema89"].iloc[-2] if len(df_h1) >= 2 else h1_ema89
    h1_close = df_h1["close"].iloc[-1]
    h1_atr = df_h1["atr_h1"].iloc[-1]
    h1_atr_pct = h1_atr / h1_close * 100
    h1_gap = abs(h1_ema34 - h1_ema89) / h1_ema89 * 100
    h1_gap_prev = abs(h1_ema34_prev - h1_ema89_prev) / h1_ema89_prev * 100

    df_m15 = df_m15.copy()
    df_m15["ema34"] = df_m15["close"].ewm(span=34, adjust=False).mean()
    df_m15["ema89"] = df_m15["close"].ewm(span=89, adjust=False).mean()
    df_m15["tr"] = np.maximum(
        df_m15["high"] - df_m15["low"],
        np.maximum(
            abs(df_m15["high"] - df_m15["close"].shift(1)),
            abs(df_m15["low"] - df_m15["close"].shift(1)),
        ),
    )
    df_m15["atr"] = df_m15["tr"].ewm(span=14, adjust=False).mean()

    m15_ema34 = df_m15["ema34"].iloc[-1]
    m15_ema89 = df_m15["ema89"].iloc[-1]
    m15_ema34_prev = df_m15["ema34"].iloc[-2] if len(df_m15) >= 2 else m15_ema34
    m15_ema89_prev = df_m15["ema89"].iloc[-2] if len(df_m15) >= 2 else m15_ema89
    m15_gap = abs(m15_ema34 - m15_ema89) / m15_ema89 * 100
    atr_m15 = df_m15["atr"].iloc[-1]
    close_m15 = float(df_m15["close"].iloc[-1])
    m15_atr_pct = float(atr_m15 / close_m15 * 100) if close_m15 > 0 else 0.0
    m15_gap_prev_for_trend = abs(m15_ema34_prev - m15_ema89_prev) / m15_ema89_prev * 100

    h1_trend = classify_trend(
        h1_gap, h1_ema34, h1_ema89, H1_TREND_GAP, h1_atr_pct,
        prev_gap_pct=h1_gap_prev, prev_ema34=h1_ema34_prev, prev_ema89=h1_ema89_prev,
    )
    raw_m15_trend = classify_trend(
        m15_gap, m15_ema34, m15_ema89, M15_TREND_GAP, m15_atr_pct,
        prev_gap_pct=m15_gap_prev_for_trend, prev_ema34=m15_ema34_prev, prev_ema89=m15_ema89_prev,
    )

    delta = df_m15["close"].diff()
    gain = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = (100 - 100 / (1 + rs)).iloc[-1]

    m15_rollover, m15_rollover_reason = detect_m15_rollover(
        raw_m15_trend, df_m15, float(atr_m15), float(rsi)
    )
    m15_trend = raw_m15_trend

    # ADX(14) on m15 using Wilder smoothing (span = 2*14-1 = 27)
    _w = {"span": 27, "adjust": False}
    _prev_h = df_m15["high"].shift(1)
    _prev_l = df_m15["low"].shift(1)
    _dm_p   = (df_m15["high"] - _prev_h).clip(lower=0)
    _dm_m   = (_prev_l - df_m15["low"]).clip(lower=0)
    _dm_p_w = _dm_p.where(_dm_p > _dm_m, 0.0)
    _dm_m_w = _dm_m.where(_dm_m > _dm_p, 0.0)
    _tr_s   = df_m15["tr"].ewm(**_w).mean()
    _di_p   = _dm_p_w.ewm(**_w).mean() / _tr_s * 100
    _di_m   = _dm_m_w.ewm(**_w).mean() / _tr_s * 100
    _dx     = (abs(_di_p - _di_m) / (_di_p + _di_m).replace(0, np.nan) * 100)
    adx_m15 = float(_dx.ewm(**_w).mean().iloc[-1])

    vol_ma = df_m15["volume"].rolling(20).mean().iloc[-1]
    vol_spike = df_m15["volume"].iloc[-1] > vol_ma * 1.5

    price_above_both_emas = close_m15 > float(m15_ema34) and close_m15 > float(m15_ema89)
    price_below_both_emas = close_m15 < float(m15_ema34) and close_m15 < float(m15_ema89)

    m15_gap_prev = (
        abs(float(df_m15["ema34"].iloc[-4]) - float(df_m15["ema89"].iloc[-4]))
        / float(df_m15["ema89"].iloc[-4]) * 100
        if len(df_m15) >= 4 else m15_gap
    )
    ema_gap_widening = m15_gap > m15_gap_prev

    current_price = float(df_m5["close"].iloc[-1])
    nearest_ssl, nearest_bsl = _nearest_liquidity_levels(df_m5, current_price)

    is_range = (h1_trend == "VOLATILE_RANGE" or m15_trend == "VOLATILE_RANGE")
    is_aligned = (h1_trend == m15_trend and h1_trend not in ("SIDEWAY", "VOLATILE_RANGE"))
    primary_trend = m15_trend
    market_mode = "VOLATILE_RANGE" if is_range else primary_trend

    return {
        "h1_trend": h1_trend,
        "m15_trend": m15_trend,
        "m15_trend_raw": raw_m15_trend,
        "primary_trend": primary_trend,
        "market_mode": market_mode,
        "is_aligned": is_aligned,
        "is_range": is_range,
        "trend_rollover": bool(m15_rollover),
        "trend_rollover_reason": m15_rollover_reason,
        "h1_gap": round(h1_gap, 3),
        "m15_gap": round(m15_gap, 3),
        "h1_atr_pct": round(h1_atr_pct, 4),
        "m15_atr_pct": round(m15_atr_pct, 4),
        "atr_m15": round(atr_m15, 4),
        "adx_m15": round(adx_m15, 2),
        "rsi": round(float(rsi), 2),
        "vol_spike": bool(vol_spike),
        "vol_ma": round(float(vol_ma), 2),
        "price_above_both_emas": bool(price_above_both_emas),
        "price_below_both_emas": bool(price_below_both_emas),
        "ema_gap_widening": bool(ema_gap_widening),
        "current_price": current_price,
        "nearest_ssl": round(float(nearest_ssl), 4) if nearest_ssl is not None else None,
        "nearest_bsl": round(float(nearest_bsl), 4) if nearest_bsl is not None else None,
        "_df_h1_ema34": round(float(h1_ema34), 4),
        "_df_h1_ema89": round(float(h1_ema89), 4),
    }


def _collect_swing_levels(df: pd.DataFrame, width: int) -> tuple[list[float], list[float]]:
    swing_highs: list[float] = []
    swing_lows: list[float] = []
    if df is None or df.empty or len(df) < width * 2 + 1:
        return swing_highs, swing_lows

    source = df.reset_index(drop=True)
    for i in range(width, len(source) - width):
        window_h = source["high"].iloc[i - width:i + width + 1]
        window_l = source["low"].iloc[i - width:i + width + 1]
        if source["high"].iloc[i] == window_h.max():
            swing_highs.append(float(source["high"].iloc[i]))
        if source["low"].iloc[i] == window_l.min():
            swing_lows.append(float(source["low"].iloc[i]))
    return swing_highs, swing_lows


def _build_sr_snapshot(df_h1: pd.DataFrame, ref_price: float, df_m15: pd.DataFrame | None = None) -> dict:
    swing_highs, swing_lows = _collect_swing_levels(df_h1.tail(50), width=5)

    if df_m15 is not None and len(df_m15) >= 30:
        m15_highs, m15_lows = _collect_swing_levels(df_m15.tail(40), width=3)
        swing_highs.extend(m15_highs)
        swing_lows.extend(m15_lows)

    resistances = sorted(h for h in set(swing_highs) if h > ref_price * 1.002)
    supports = sorted((lv for lv in set(swing_lows) if lv < ref_price * 0.998), reverse=True)

    resistance_levels = [round(x, 2) for x in resistances[:5]]
    support_levels = [round(x, 2) for x in supports[:5]]

    if not resistance_levels:
        resistance_levels = [round(ref_price * 1.015, 2)]
    if not support_levels:
        support_levels = [round(ref_price * 0.985, 2)]

    return {
        "resistance": resistance_levels[0],
        "support": support_levels[0],
        "resistance_levels": resistance_levels,
        "support_levels": support_levels,
    }


def _bars_since_breakout(df_h1: pd.DataFrame, level: float, direction: str, max_bars: int = 12) -> int | None:
    if df_h1 is None or df_h1.empty or len(df_h1) < 2 or level <= 0:
        return None

    closes = [float(x) for x in df_h1["close"].tail(max_bars + 1)]
    if len(closes) < 2:
        return None

    for idx in range(len(closes) - 1, 0, -1):
        prev_close = closes[idx - 1]
        cur_close = closes[idx]
        if direction == "up" and cur_close > level and prev_close <= level:
            return len(closes) - 1 - idx
        if direction == "down" and cur_close < level and prev_close >= level:
            return len(closes) - 1 - idx
    return None


def _consecutive_outside_closes(df_h1: pd.DataFrame, level: float, direction: str, max_bars: int = 6) -> int:
    if df_h1 is None or df_h1.empty or level <= 0:
        return 0

    count = 0
    closes = [float(x) for x in df_h1["close"].tail(max_bars)]
    for close in reversed(closes):
        if direction == "up" and close > level:
            count += 1
        elif direction == "down" and close < level:
            count += 1
        else:
            break
    return count


def find_sr_levels(df_h1: pd.DataFrame, current_price: float,
                   df_m15: pd.DataFrame = None) -> dict:
    """
    Swing high/low S/R from H1 supplemented by M15 swing levels.
    Returns the active box plus a previous box snapshot so breakout state
    can stay visually stable even while the current box drifts upward/downward.
    """
    current_sr = _build_sr_snapshot(df_h1, current_price, df_m15)

    prev_support = None
    prev_resistance = None
    prev_ref_price = None
    breakout_state = "INSIDE_PREV_BOX"
    breakout_distance_pct = 0.0
    breakout_confirmed = False
    bars_since_breakout = None
    outside_prev_box_closes = 0

    h1_shift = 6
    m15_shift = 24
    if df_h1 is not None and len(df_h1) > h1_shift + 20:
        prev_h1 = df_h1.iloc[:-h1_shift].copy()
        prev_m15 = df_m15.iloc[:-m15_shift].copy() if df_m15 is not None and len(df_m15) > m15_shift + 20 else None
        prev_ref_price = float(prev_h1["close"].iloc[-1])
        prev_sr = _build_sr_snapshot(prev_h1, prev_ref_price, prev_m15)
        prev_support = prev_sr.get("support")
        prev_resistance = prev_sr.get("resistance")

        tol_pct = 0.08
        if prev_resistance is not None and current_price > float(prev_resistance) * (1 + tol_pct / 100.0):
            breakout_state = "ABOVE_PREV_BOX"
            breakout_distance_pct = (current_price - float(prev_resistance)) / float(prev_resistance) * 100
            outside_prev_box_closes = _consecutive_outside_closes(df_h1, float(prev_resistance), "up")
            bars_since_breakout = _bars_since_breakout(df_h1, float(prev_resistance), "up")
            breakout_confirmed = outside_prev_box_closes >= 2
        elif prev_support is not None and current_price < float(prev_support) * (1 - tol_pct / 100.0):
            breakout_state = "BELOW_PREV_BOX"
            breakout_distance_pct = (float(prev_support) - current_price) / float(prev_support) * 100
            outside_prev_box_closes = _consecutive_outside_closes(df_h1, float(prev_support), "down")
            bars_since_breakout = _bars_since_breakout(df_h1, float(prev_support), "down")
            breakout_confirmed = outside_prev_box_closes >= 2
        else:
            breakout_state = "INSIDE_PREV_BOX"
            breakout_distance_pct = 0.0

    return {
        "resistance": current_sr["resistance"],
        "support": current_sr["support"],
        "resistance_levels": current_sr["resistance_levels"],
        "support_levels": current_sr["support_levels"],
        "prev_resistance": round(float(prev_resistance), 2) if prev_resistance is not None else None,
        "prev_support": round(float(prev_support), 2) if prev_support is not None else None,
        "prev_ref_price": round(float(prev_ref_price), 2) if prev_ref_price is not None else None,
        "breakout_state": breakout_state,
        "breakout_confirmed": bool(breakout_confirmed),
        "breakout_distance_pct": round(float(breakout_distance_pct), 3),
        "bars_since_breakout": bars_since_breakout,
        "outside_prev_box_closes": int(outside_prev_box_closes),
    }


def compute_score(context: dict) -> tuple[int, list[str]]:
    """
    5-point quantitative score.
    """
    score = 0
    details = []
    trend = context.get("primary_trend") or context["m15_trend"]
    is_range = context["is_range"]

    if context.get("ema_gap_widening"):
        score += 1
        details.append("EMA gap widening (+1)")

    rsi = context["rsi"]
    if (trend == "UPTREND" and rsi > 55) or (trend == "DOWNTREND" and rsi < 45):
        score += 1
        details.append(f"RSI={rsi:.1f} (+1)")
    elif is_range and (rsi > 60 or rsi < 40):
        score += 1
        details.append(f"RSI={rsi:.1f} extreme in range (+1)")

    if context["vol_spike"]:
        score += 1
        details.append("Vol spike 1.5x (+1)")

    if (trend == "UPTREND" and context.get("price_above_both_emas")) or \
       (trend == "DOWNTREND" and context.get("price_below_both_emas")):
        score += 1
        details.append("Price aligned with both EMAs (+1)")

    if context.get("h1_atr_pct", 0) > ATR_VOLATILE_PCT * 0.6:
        score += 1
        details.append(f"ATR%={context['h1_atr_pct']:.2f} elevated (+1)")

    if context.get("trend_rollover"):
        details.append("M15 rollover caution")

    return score, details


def check_macro_bias(ml_direction: str, ml_confidence: float,
                     fear_greed: int, funding_rate: float,
                     symbol: str) -> tuple[bool, str, str]:
    """
    Returns macro context. Blocks shorts in extreme fear and longs in extreme greed.
    Thresholds are configurable via FG_EXTREME_FEAR_THRESHOLD / FG_EXTREME_GREED_THRESHOLD.
    Set FG_EXTREME_BLOCK=false to revert to advisory-only mode.
    """
    fg_block       = _env_bool("FG_EXTREME_BLOCK", True)
    fear_threshold = _env_int("FG_EXTREME_FEAR_THRESHOLD", 15)
    greed_threshold = _env_int("FG_EXTREME_GREED_THRESHOLD", 85)

    notes = []

    if ml_direction:
        notes.append(f"ML={ml_direction}({ml_confidence:.0%}) advisory")

    if fear_greed is not None:
        if fear_greed <= fear_threshold:
            if fg_block:
                notes.append(f"F&G={fear_greed} extreme fear — LONG only (bounce risk)")
                return True, " | ".join(notes), "UP"
            notes.append(f"F&G={fear_greed} extreme fear")
        elif fear_greed >= greed_threshold:
            if fg_block:
                notes.append(f"F&G={fear_greed} extreme greed — SHORT only (reversal risk)")
                return True, " | ".join(notes), "DOWN"
            notes.append(f"F&G={fear_greed} extreme greed")
        else:
            notes.append(f"F&G={fear_greed}")

    if funding_rate is not None:
        if funding_rate > FUNDING_THRESHOLD:
            notes.append(f"funding {funding_rate:+.4f}% crowded longs")
        elif funding_rate < -FUNDING_THRESHOLD:
            notes.append(f"funding {funding_rate:+.4f}% crowded shorts")
        else:
            notes.append(f"funding {funding_rate:+.4f}%")

    return True, " | ".join(notes) if notes else "advisory only", "BOTH"


def check_technical_gates(context: dict) -> tuple[bool, str]:
    """
    Runs all technical gates in order.
    """
    symbol_text = str(context.get("symbol", ""))
    symbol_base = symbol_text.split("/", 1)[0].upper() if symbol_text else ""
    primary = context.get("primary_trend") or context["m15_trend"]
    score = context["score"]
    is_range = context["is_range"]
    allow_sideway = _env_bool("TECH_ALLOW_SIDEWAY", False)
    price = float(context.get("current_price") or 0.0)
    sr = context.get("sr", {}) or {}
    rsi = float(context.get("rsi") or 50.0)
    adx = float(context.get("adx_m15") or 0.0)
    atr_pct = float(context.get("m15_atr_pct") or 0.0)
    edge = max(0.35, float(context.get("h1_atr_pct") or 0.8))
    near_edge = False
    if price > 0 and sr:
        dist_r = abs(float(sr["resistance"]) - price) / price * 100
        dist_s = abs(price - float(sr["support"])) / price * 100
        near_edge = dist_r <= edge or dist_s <= edge
    rsi_extreme = rsi <= 42 or rsi >= 58

    if symbol_base == "BTC":
        btc_high_atr_pct = max(0.0, _env_float("BTC_HIGH_ATR_PCT", 1.1))
        btc_min_adx = max(0.0, _env_float("BTC_HIGH_ATR_MIN_ADX", 25.0))
        if atr_pct >= btc_high_atr_pct and adx < btc_min_adx:
            return False, (
                f"GATE0: BTC unstable volatility "
                f"(ATR%={atr_pct:.2f} >= {btc_high_atr_pct:.2f}, ADX={adx:.1f} < {btc_min_adx:.1f})"
            )

    if primary == "SIDEWAY" and not allow_sideway and not (near_edge or rsi_extreme):
        return False, "GATE1: M15 SIDEWAY"

    if not is_range:
        if primary == "VOLATILE_RANGE":
            return False, "GATE2: M15 VOLATILE_RANGE"

    base_threshold = max(1, _env_int("TECH_SCORE_THRESHOLD", 2))
    range_threshold = max(1, _env_int("TECH_SCORE_THRESHOLD_RANGE", 1))
    rollover_threshold = max(1, _env_int("TECH_ROLLOVER_SCORE_THRESHOLD", 1))

    threshold = range_threshold if is_range else base_threshold
    if primary == "SIDEWAY" and (near_edge or rsi_extreme):
        threshold = min(threshold, 1)
    if context.get("trend_rollover"):
        threshold = min(threshold, rollover_threshold)
    if score < threshold:
        return False, f"GATE3: score {score}/{threshold} insufficient"

    return True, "OK"


def get_range_bias(context: dict) -> str:
    """Returns 'NEAR_SUPPORT', 'NEAR_RESISTANCE', or 'MIDDLE'.

    Price is split into thirds of the support→resistance range:
      bottom 30% → NEAR_SUPPORT
      top    30% → NEAR_RESISTANCE
      middle 40% → MIDDLE (no edge, gate blocks trade)
    """
    price = context["current_price"]
    sr = context.get("sr", {})
    if not sr:
        return "MIDDLE"
    resistance = float(sr["resistance"])
    support = float(sr["support"])
    range_size = resistance - support
    if range_size <= 0:
        return "MIDDLE"
    position = (price - support) / range_size  # 0.0 = at support, 1.0 = at resistance
    if position < 0.30:
        return "NEAR_SUPPORT"
    if position > 0.70:
        return "NEAR_RESISTANCE"
    return "MIDDLE"


def detect_candle_pattern(df: pd.DataFrame, idx: int) -> dict:
    """
    Rule-based pattern detection on a single candle at df.iloc[idx].
    """
    if idx < 1 or idx >= len(df):
        return {"pattern": "none", "direction": "NONE"}

    row = df.iloc[idx]
    prev = df.iloc[idx - 1]

    open_price = float(row["open"])
    high_price = float(row["high"])
    low_price = float(row["low"])
    close_price = float(row["close"])
    body = abs(close_price - open_price)
    total = high_price - low_price
    wick_up = high_price - max(open_price, close_price)
    wick_down = min(open_price, close_price) - low_price

    if total == 0:
        return {"pattern": "none", "direction": "NONE"}

    if (
        wick_down >= 2 * body
        and close_price > (low_price + total * 0.5)
        and wick_up <= wick_down * 0.4
    ):
        return {"pattern": "bullish_pinbar", "direction": "BUY"}

    if (
        wick_up >= 2 * body
        and close_price < (low_price + total * 0.5)
        and wick_down <= wick_up * 0.4
    ):
        return {"pattern": "bearish_pinbar", "direction": "SELL"}

    prev_open = float(prev["open"])
    prev_close = float(prev["close"])
    if (
        close_price > open_price
        and prev_close < prev_open
        and open_price <= prev_close
        and close_price >= prev_open
    ):
        return {"pattern": "bullish_engulfing", "direction": "BUY"}

    if (
        close_price < open_price
        and prev_close > prev_open
        and open_price >= prev_close
        and close_price <= prev_open
    ):
        return {"pattern": "bearish_engulfing", "direction": "SELL"}

    return {"pattern": "none", "direction": "NONE"}




def validate_ai_trade_decision(decision: dict, context: dict,
                               df_exec: pd.DataFrame = None) -> tuple[bool, str]:
    """
    Confirms the selected signal follows the current market structure.
    """
    signal = str(decision.get("signal", "")).upper()
    if signal not in ("BUY", "SELL"):
        return False, f"invalid signal {signal!r}"

    allowed_direction = context.get("allowed_direction", "BOTH")
    if allowed_direction == "UP" and signal != "BUY":
        fg = context.get("fear_greed", "?")
        return False, f"F&G={fg} extreme fear — only LONG allowed, got {signal}"
    if allowed_direction == "DOWN" and signal != "SELL":
        fg = context.get("fear_greed", "?")
        return False, f"F&G={fg} extreme greed — only SHORT allowed, got {signal}"

    primary = context.get("primary_trend") or context.get("m15_trend", "")
    if primary == "UPTREND" and signal != "BUY":
        return False, f"M15={primary} requires BUY"
    if primary == "DOWNTREND" and signal != "SELL":
        return False, f"M15={primary} requires SELL"

    setup_code = _setup_code(decision.get("analysis", {}).get("setup_identified", ""))
    if setup_code not in ("A", "B", "C", "D"):
        return False, f"unsupported setup {setup_code!r}; only Setup A/B/C/D allowed"

    symbol_text = str(context.get("symbol", ""))
    symbol_base = symbol_text.split("/", 1)[0].upper() if symbol_text else ""
    if symbol_base == "BTC":
        btc_macro = str(context.get("btc_macro_trend", "NEUTRAL")).upper()
        if btc_macro == "BULL" and signal != "BUY":
            return False, "BTC 4h macro is BULL — blocking countertrend SELL"
        if btc_macro == "BEAR" and signal != "SELL":
            return False, "BTC 4h macro is BEAR — blocking countertrend BUY"

        atr_pct = float(context.get("m15_atr_pct") or 0.0)
        btc_range_disable_atr = max(0.0, _env_float("BTC_RANGE_DISABLE_ATR_PCT", 0.9))
        if setup_code == "D" and atr_pct >= btc_range_disable_atr:
            return False, (
                f"BTC Setup D disabled when M15 ATR%={atr_pct:.2f} "
                f">= {btc_range_disable_atr:.2f}"
            )

    if setup_code == "A":
        h1 = context.get("h1_trend", "")
        m15 = context.get("m15_trend", "")
        if h1 != m15 or h1 in ("SIDEWAY", "VOLATILE_RANGE"):
            return False, f"Setup A requires H1/M15 alignment — H1={h1} M15={m15}"

    timing_ok, timing_reason = _setup_timing_ok(setup_code, signal, context, df_exec)
    if not timing_ok:
        return False, timing_reason

    return True, "OK"


def _setup_code(setup_name: str) -> str:
    text = str(setup_name or "").strip().lower()
    if "setup " in text:
        return text.split("setup ", 1)[1][:1].upper()
    if "setup_" in text:
        return text.split("setup_", 1)[1][:1].upper()
    return text[:1].upper() if text else "?"



def _dedupe_levels(values: list[float], reverse: bool = False) -> list[float]:
    cleaned = []
    for value in values:
        if value is None or not np.isfinite(value):
            continue
        rounded = round(float(value), 6)
        if rounded not in cleaned:
            cleaned.append(rounded)
    return sorted(cleaned, reverse=reverse)


def _recent_box_levels(df_m5: pd.DataFrame, window: int = 6,
                       exclude_last: int = 0) -> tuple[float, float]:
    source = df_m5.iloc[:-exclude_last] if exclude_last > 0 and len(df_m5) > exclude_last else df_m5
    box = source.tail(window) if not source.empty else df_m5.tail(window)
    return float(box["low"].min()), float(box["high"].max())


def _pullback_cluster_levels(df_m5: pd.DataFrame, signal: str, lookback: int = 12) -> tuple[float, float]:
    cluster = df_m5.tail(lookback).copy()
    if signal == "BUY":
        red = cluster[cluster["close"] < cluster["open"]]
        lows = red["low"] if not red.empty else cluster["low"]
        return float(lows.min()), float(cluster["high"].max())
    green = cluster[cluster["close"] > cluster["open"]]
    highs = green["high"] if not green.empty else cluster["high"]
    return float(cluster["low"].min()), float(highs.max())


def _deep_structure_levels(df_m5: pd.DataFrame, lookback: int = 20) -> tuple[float, float]:
    cluster = df_m5.tail(lookback).copy()
    return float(cluster["low"].min()), float(cluster["high"].max())


def _is_small_consolidation(df_exec: pd.DataFrame, ema34_series: pd.Series,
                            atr: float, current_price: float) -> bool:
    if df_exec is None or len(df_exec) < 6:
        return False

    recent = df_exec.tail(6).copy()
    range_6 = float(recent["high"].max() - recent["low"].min())
    closes = recent["close"].astype(float).to_numpy()
    ema_recent = ema34_series.tail(6).astype(float).to_numpy()
    mixed_sides = np.any(closes > ema_recent) and np.any(closes < ema_recent)
    body_sizes = (recent["close"].astype(float) - recent["open"].astype(float)).abs().to_numpy()
    candle_ranges = (recent["high"].astype(float) - recent["low"].astype(float)).replace(0, np.nan).to_numpy()
    small_body_count = int(np.nansum(body_sizes <= (candle_ranges * 0.45)))
    tight_threshold = max(atr * 1.25, current_price * 0.003)
    return range_6 <= tight_threshold and mixed_sides and small_body_count >= 4


def _pinbar_confirmation_ok(signal: str, df_exec: pd.DataFrame) -> tuple[bool, str]:
    if df_exec is None or len(df_exec) < 3:
        return True, "OK"

    latest_idx = len(df_exec) - 1
    prev_idx = len(df_exec) - 2
    latest_pattern = detect_candle_pattern(df_exec, latest_idx)
    prev_pattern = detect_candle_pattern(df_exec, prev_idx)
    prev_high = float(df_exec["high"].iloc[prev_idx])
    prev_low = float(df_exec["low"].iloc[prev_idx])
    latest_close = float(df_exec["close"].iloc[latest_idx])

    if signal == "BUY":
        if latest_pattern.get("pattern") == "bullish_pinbar":
            return False, "bullish pinbar needs next candle confirmation above its high"
        if prev_pattern.get("pattern") == "bullish_pinbar" and latest_close <= prev_high:
            return False, "BUY missing close above prior pinbar high"
    else:
        if latest_pattern.get("pattern") == "bearish_pinbar":
            return False, "bearish pinbar needs next candle confirmation below its low"
        if prev_pattern.get("pattern") == "bearish_pinbar" and latest_close >= prev_low:
            return False, "SELL missing close below prior pinbar low"

    return True, "OK"


def _breakout_acceptance_ok(signal: str, context: dict,
                            df_exec: pd.DataFrame | None) -> tuple[bool, str]:
    if df_exec is None or len(df_exec) < 8:
        return False, "not enough execution candles for breakout setup"

    latest = df_exec.iloc[-1]
    latest_open = float(latest["open"])
    latest_high = float(latest["high"])
    latest_low = float(latest["low"])
    latest_close = float(latest["close"])
    latest_volume = float(latest.get("volume", 0.0))
    latest_range = max(latest_high - latest_low, 1e-9)
    latest_body = abs(latest_close - latest_open)
    atr = float(context.get("atr_m15") or context.get("atr") or 0.0)
    prev_vol_mean = float(df_exec.tail(7).iloc[:-1]["volume"].mean())
    vol_ok = latest_volume >= prev_vol_mean * 1.05 if prev_vol_mean > 0 else True
    box_low, box_high = _recent_box_levels(df_exec, window=6, exclude_last=1)
    closes = df_exec["close"].astype(float)
    ema34 = float(closes.ewm(span=34, adjust=False).mean().iloc[-1])
    body_ok = latest_body >= latest_range * 0.50
    momentum_ok = latest_body >= max(atr * 0.20, latest_close * 0.0012) if atr > 0 else body_ok

    if signal == "BUY":
        if latest_close <= box_high:
            return False, f"BUY breakout not accepted above {box_high:.4f}"
        if latest_close <= ema34:
            return False, f"BUY breakout still below EMA34 {ema34:.4f}"
        if latest_close < latest_high - latest_range * 0.35:
            return False, "BUY breakout candle closed too far off its highs"
    else:
        if latest_close >= box_low:
            return False, f"SELL breakout not accepted below {box_low:.4f}"
        if latest_close >= ema34:
            return False, f"SELL breakout still above EMA34 {ema34:.4f}"
        if latest_close > latest_low + latest_range * 0.35:
            return False, "SELL breakout candle closed too far off its lows"

    if not body_ok or not momentum_ok:
        return False, "breakout candle lacks decisive body expansion"
    if not vol_ok:
        return False, "breakout volume not strong enough"
    return True, "OK"


def _retest_hold_ok(signal: str, context: dict,
                    df_exec: pd.DataFrame | None) -> tuple[bool, str]:
    if df_exec is None or len(df_exec) < 9:
        return False, "not enough execution candles for retest setup"

    latest = df_exec.iloc[-1]
    prev = df_exec.iloc[-2]
    latest_open = float(latest["open"])
    latest_close = float(latest["close"])
    latest_high = float(latest["high"])
    latest_low = float(latest["low"])
    prev_close = float(prev["close"])
    prev_high = float(prev["high"])
    prev_low = float(prev["low"])
    atr = float(context.get("atr_m15") or context.get("atr") or 0.0)
    retest_buffer = max(atr * 0.25, latest_close * 0.0012) if atr > 0 else latest_close * 0.0012
    box_low, box_high = _recent_box_levels(df_exec, window=6, exclude_last=4)
    prev_vol_mean = float(df_exec.tail(8).iloc[:-1]["volume"].mean())
    latest_volume = float(latest.get("volume", 0.0))
    closes = df_exec["close"].astype(float)
    ema34 = float(closes.ewm(span=34, adjust=False).mean().iloc[-1])

    if signal == "BUY":
        breakout_slice = df_exec.iloc[-4:-1]
        breakout_seen = bool(
            (breakout_slice["close"].astype(float) > box_high).any()
            or (breakout_slice["high"].astype(float) > box_high).any()
        )
        retest_seen = min(prev_low, latest_low) <= box_high + retest_buffer
        if not breakout_seen:
            return False, "BUY retest setup missing prior breakout above local structure"
        if not retest_seen:
            return False, "BUY retest setup did not revisit broken resistance"
        if latest_close <= box_high or latest_close <= latest_open:
            return False, "BUY retest hold missing bullish close back above breakout level"
        if latest_close <= ema34:
            return False, f"BUY retest close still below EMA34 {ema34:.4f}"
        if latest_close < latest_high - max((latest_high - latest_low) * 0.45, 1e-9):
            return False, "BUY retest hold candle closed too weakly"
    else:
        breakout_slice = df_exec.iloc[-4:-1]
        breakout_seen = bool(
            (breakout_slice["close"].astype(float) < box_low).any()
            or (breakout_slice["low"].astype(float) < box_low).any()
        )
        retest_seen = max(prev_high, latest_high) >= box_low - retest_buffer
        if not breakout_seen:
            return False, "SELL retest setup missing prior breakdown below local structure"
        if not retest_seen:
            return False, "SELL retest setup did not revisit broken support"
        if latest_close >= box_low or latest_close >= latest_open:
            return False, "SELL retest fail missing bearish close back below breakdown level"
        if latest_close >= ema34:
            return False, f"SELL retest close still above EMA34 {ema34:.4f}"
        if latest_close > latest_low + max((latest_high - latest_low) * 0.45, 1e-9):
            return False, "SELL retest fail candle closed too weakly"

    if prev_vol_mean > 0 and latest_volume < prev_vol_mean:
        return False, "retest confirmation volume below recent average"
    return True, "OK"


def _setup_timing_ok(setup_code: str, signal: str, context: dict,
                     df_exec: pd.DataFrame | None) -> tuple[bool, str]:
    if setup_code in ("A", "D"):
        return _exec_entry_timing_ok(signal, context, df_exec)
    if setup_code == "B":
        return _breakout_acceptance_ok(signal, context, df_exec)
    if setup_code == "C":
        return _retest_hold_ok(signal, context, df_exec)
    return False, f"unsupported setup {setup_code!r}"


def _exec_entry_timing_ok(signal: str, context: dict,
                          df_exec: pd.DataFrame | None) -> tuple[bool, str]:
    """
    Fast veto so higher-timeframe trend does not force entries during an active
    execution-timeframe counter-impulse.
    """
    if df_exec is None or len(df_exec) < 6:
        return True, "OK"

    is_range = bool(context.get("is_range", False))
    closes = df_exec["close"].astype(float)
    ema34_series = closes.ewm(span=34, adjust=False).mean()
    ema34_m5 = float(ema34_series.iloc[-1])
    ema34_prev = float(ema34_series.iloc[-2])
    latest = df_exec.iloc[-1]
    latest_open = float(latest["open"])
    latest_close = float(latest["close"])
    latest_volume = float(latest.get("volume", 0.0))
    prev_close = float(closes.iloc[-2])
    atr = float(context.get("atr_m15") or context.get("atr") or 0.0)
    current_price = float(context.get("current_price") or latest_close or 0.0)
    box_low, box_high = _recent_box_levels(df_exec, window=6, exclude_last=1)

    recent = df_exec.tail(4).copy().reset_index(drop=True)
    three_bar_move = float(recent["close"].iloc[-1] - recent["close"].iloc[0])
    pre_reclaim_move = float(closes.iloc[-2] - closes.iloc[-4])
    prev_vol_mean = float(df_exec.tail(6).iloc[:-1]["volume"].mean())
    vol_expanding = latest_volume > prev_vol_mean * 1.10 if prev_vol_mean > 0 else False

    latest_bearish = latest_close < latest_open
    latest_bullish = latest_close > latest_open
    prior_below_ema = prev_close <= ema34_prev
    prior_above_ema = prev_close >= ema34_prev
    pinbar_ok, pinbar_reason = _pinbar_confirmation_ok(signal, df_exec)
    if not pinbar_ok:
        return False, pinbar_reason
    if _is_small_consolidation(df_exec, ema34_series, atr, current_price):
        prev_pattern = detect_candle_pattern(df_exec, len(df_exec) - 2)
        latest_pattern = detect_candle_pattern(df_exec, len(df_exec) - 1)
        pinbar_names = {"bullish_pinbar", "bearish_pinbar"}
        if prev_pattern.get("pattern") in pinbar_names or latest_pattern.get("pattern") in pinbar_names:
            return False, "pinbar formed inside small consolidation/chop"

    if signal == "BUY":
        if not is_range:
            if latest_bearish and latest_close < ema34_m5:
                return False, f"exec bearish impulse: close {latest_close:.4f} below EMA34 {ema34_m5:.4f}"
            if atr > 0 and three_bar_move <= -atr and latest_close <= ema34_m5:
                return False, f"exec bearish impulse: 3-candle drop {abs(three_bar_move):.4f} >= ATR {atr:.4f}"
            if (
                atr > 0
                and pre_reclaim_move <= -(atr * 0.75)
                and latest_bullish
                and latest_close > ema34_m5
                and prior_below_ema
            ):
                return False, (
                    f"exec flush recovery not confirmed: fresh reclaim above EMA34 {ema34_m5:.4f} "
                    f"after {abs(pre_reclaim_move):.4f} drop; wait for hold"
                )
        if latest_close < box_low:
            return False, f"exec lost micro support {box_low:.4f}"
        if vol_expanding and latest_bearish and latest_close <= ema34_m5:
            return False, "exec sell pressure expanding on the latest candle"
        return True, "OK"

    # SELL
    if not is_range:
        if latest_bullish and latest_close > ema34_m5:
            return False, f"exec bullish impulse: close {latest_close:.4f} above EMA34 {ema34_m5:.4f}"
        if atr > 0 and three_bar_move >= atr and latest_close >= ema34_m5:
            return False, f"exec bullish impulse: 3-candle rally {abs(three_bar_move):.4f} >= ATR {atr:.4f}"
        if (
            atr > 0
            and pre_reclaim_move >= atr * 0.75
            and latest_bearish
            and latest_close < ema34_m5
            and prior_above_ema
        ):
            return False, (
                f"exec squeeze-down not confirmed: fresh reclaim below EMA34 {ema34_m5:.4f} "
                f"after {abs(pre_reclaim_move):.4f} rally; wait for hold"
            )
    if latest_close > box_high:
        return False, f"exec broke above micro resistance {box_high:.4f}"
    if vol_expanding and latest_bullish and latest_close >= ema34_m5:
        return False, "exec buy pressure expanding on the latest candle"
    return True, "OK"


def _pattern_name(pattern: dict[str, str] | None) -> str:
    text = str((pattern or {}).get("pattern") or "none").strip().replace("_", " ")
    return text if text else "none"


def _pattern_matches_signal(pattern: dict[str, str] | None, signal: str) -> bool:
    return str((pattern or {}).get("direction") or "").upper() == str(signal or "").upper()


def _build_analysis_payload(
    *,
    signal: str,
    setup_name: str,
    context: dict,
    df_exec: pd.DataFrame,
    setup_reason: str,
) -> dict[str, str]:
    closes = df_exec["close"].astype(float)
    ema34_series = closes.ewm(span=34, adjust=False).mean()
    ema89_series = closes.ewm(span=89, adjust=False).mean()
    ema34 = float(ema34_series.iloc[-1])
    ema89 = float(ema89_series.iloc[-1])
    latest = df_exec.iloc[-1]
    latest_open = float(latest["open"])
    latest_close = float(latest["close"])
    latest_high = float(latest["high"])
    latest_low = float(latest["low"])
    latest_volume = float(latest.get("volume", 0.0))
    prev_vol_mean = float(df_exec.tail(6).iloc[:-1]["volume"].mean()) if len(df_exec) >= 6 else latest_volume
    volume_state = "above" if prev_vol_mean > 0 and latest_volume >= prev_vol_mean else "below"
    pattern_latest = detect_candle_pattern(df_exec, len(df_exec) - 1)
    pattern_prev = detect_candle_pattern(df_exec, len(df_exec) - 2)
    primary = str(context.get("primary_trend") or context.get("m15_trend") or "?")
    gap = float(context.get("m15_gap") or 0.0)
    gap_move = "widening" if context.get("ema_gap_widening") else "stable/narrowing"
    candle_side = "bullish" if latest_close >= latest_open else "bearish"
    candle_range = max(latest_high - latest_low, 1e-9)
    body = abs(latest_close - latest_open)
    upper_wick = latest_high - max(latest_open, latest_close)
    lower_wick = min(latest_open, latest_close) - latest_low

    return {
        "setup_identified": setup_name,
        "ema_check": (
            f"EMA34 {'above' if ema34 > ema89 else 'below' if ema34 < ema89 else 'near'} "
            f"EMA89. M15 trend={primary}. Gap {gap_move} at {gap:.3f}%."
        ),
        "price_action": (
            f"Latest candle {candle_side}; body={body:.4f}, upper_wick={upper_wick:.4f}, "
            f"lower_wick={lower_wick:.4f}. "
            f"Patterns latest={_pattern_name(pattern_latest)}, prev={_pattern_name(pattern_prev)}."
        ),
        "volume_check": (
            f"Latest volume {volume_state} recent mean "
            f"({latest_volume:.0f} vs {prev_vol_mean:.0f})."
        ),
        "rr_check": "Python will calculate entry, stop loss, take profit, and final R:R.",
        "pattern_match": (
            f"{setup_reason} Latest candle range={candle_range:.4f}. "
            f"Signal direction={signal}."
        ),
    }


def _setup_a_precheck(signal: str, context: dict, df_exec: pd.DataFrame) -> tuple[bool, str]:
    if df_exec is None or len(df_exec) < 6:
        return False, "not enough candles for EMA pullback setup"

    closes = df_exec["close"].astype(float)
    highs = df_exec["high"].astype(float)
    lows = df_exec["low"].astype(float)
    ema34_series = closes.ewm(span=34, adjust=False).mean()
    ema89_series = closes.ewm(span=89, adjust=False).mean()
    ema34 = float(ema34_series.iloc[-1])
    ema89 = float(ema89_series.iloc[-1])
    latest_close = float(closes.iloc[-1])
    latest_open = float(df_exec["open"].iloc[-1])
    prev_close = float(closes.iloc[-2])
    atr = float(context.get("atr_m15") or context.get("atr") or 0.0)
    price = float(context.get("current_price") or latest_close or 0.0)
    zone_buffer = max(atr * 0.35, price * 0.0015) if price > 0 else max(atr * 0.35, 0.0)
    recent_low = float(lows.tail(2).min())
    recent_high = float(highs.tail(2).max())
    latest_pattern = detect_candle_pattern(df_exec, len(df_exec) - 1)
    prev_pattern = detect_candle_pattern(df_exec, len(df_exec) - 2)

    if signal == "BUY":
        touched_zone = recent_low <= ema34 + zone_buffer or recent_low <= ema89 + zone_buffer
        if not touched_zone:
            return False, "pullback did not revisit EMA34/EMA89 demand zone"
        if latest_close < ema34 and latest_close <= prev_close:
            return False, "latest close has not reclaimed the EMA pullback area"
        if not (
            latest_close > latest_open
            or _pattern_matches_signal(latest_pattern, signal)
            or _pattern_matches_signal(prev_pattern, signal)
        ):
            return False, "no bullish reaction candle from EMA pullback zone"
        return True, "EMA pullback touched support zone and printed bullish reaction"

    touched_zone = recent_high >= ema34 - zone_buffer or recent_high >= ema89 - zone_buffer
    if not touched_zone:
        return False, "rally did not revisit EMA34/EMA89 supply zone"
    if latest_close > ema34 and latest_close >= prev_close:
        return False, "latest close has not rejected the EMA rally area"
    if not (
        latest_close < latest_open
        or _pattern_matches_signal(latest_pattern, signal)
        or _pattern_matches_signal(prev_pattern, signal)
    ):
        return False, "no bearish reaction candle from EMA rally zone"
    return True, "EMA rally touched supply zone and printed bearish reaction"


def _setup_d_precheck(signal: str, context: dict, df_exec: pd.DataFrame) -> tuple[bool, str]:
    if df_exec is None or len(df_exec) < 6:
        return False, "not enough candles for range reaction setup"

    sr = dict(context.get("sr") or {})
    support = sr.get("support")
    resistance = sr.get("resistance")
    if support is None or resistance is None:
        return False, "missing support/resistance levels for range setup"

    latest = df_exec.iloc[-1]
    latest_open = float(latest["open"])
    latest_close = float(latest["close"])
    latest_low = float(latest["low"])
    latest_high = float(latest["high"])
    prev = df_exec.iloc[-2]
    prev_low = float(prev["low"])
    prev_high = float(prev["high"])
    atr = float(context.get("atr_m15") or context.get("atr") or 0.0)
    price = float(context.get("current_price") or latest_close or 0.0)
    edge_buffer = max(atr * 0.45, price * 0.0020) if price > 0 else max(atr * 0.45, 0.0)
    latest_pattern = detect_candle_pattern(df_exec, len(df_exec) - 1)
    prev_pattern = detect_candle_pattern(df_exec, len(df_exec) - 2)
    range_bias = str(context.get("range_bias") or get_range_bias(context))
    rsi = float(context.get("rsi") or 50.0)

    if signal == "BUY":
        if range_bias != "NEAR_SUPPORT":
            return False, f"range BUY requires NEAR_SUPPORT, got {range_bias}"
        if min(latest_low, prev_low) > float(support) + edge_buffer:
            return False, "price did not test support closely enough for range BUY"
        if latest_close < float(support):
            return False, "latest close is still below support"
        if not (
            latest_close > latest_open
            or _pattern_matches_signal(latest_pattern, signal)
            or _pattern_matches_signal(prev_pattern, signal)
            or rsi <= 42
        ):
            return False, "no bullish reaction from range support"
        return True, "range support tested with bullish reaction"

    if range_bias != "NEAR_RESISTANCE":
        return False, f"range SELL requires NEAR_RESISTANCE, got {range_bias}"
    if max(latest_high, prev_high) < float(resistance) - edge_buffer:
        return False, "price did not test resistance closely enough for range SELL"
    if latest_close > float(resistance):
        return False, "latest close is still above resistance"
    if not (
        latest_close < latest_open
        or _pattern_matches_signal(latest_pattern, signal)
        or _pattern_matches_signal(prev_pattern, signal)
        or rsi >= 58
    ):
        return False, "no bearish reaction from range resistance"
    return True, "range resistance tested with bearish reaction"


def _setup_precheck(setup_code: str, signal: str, context: dict, df_exec: pd.DataFrame) -> tuple[bool, str]:
    if setup_code == "A":
        return _setup_a_precheck(signal, context, df_exec)
    if setup_code == "D":
        return _setup_d_precheck(signal, context, df_exec)
    if setup_code == "B":
        return True, "candidate breakout setup selected for structural validation"
    if setup_code == "C":
        return True, "candidate retest setup selected for structural validation"
    return False, f"unsupported setup {setup_code!r}"


def _candidate_setups_for_context(context: dict) -> list[tuple[str, str, str]]:
    primary = str(context.get("primary_trend") or context.get("m15_trend") or "").upper()
    allowed_direction = str(context.get("allowed_direction") or "BOTH").upper()
    is_range = bool(context.get("is_range", False))
    range_bias = str(context.get("range_bias") or "")

    candidates: list[tuple[str, str, str]] = []
    if is_range or primary in {"SIDEWAY", "VOLATILE_RANGE"}:
        if allowed_direction != "DOWN" and range_bias == "NEAR_SUPPORT":
            candidates.append(("D", "BUY", "Range support bounce candidate"))
        if allowed_direction != "UP" and range_bias == "NEAR_RESISTANCE":
            candidates.append(("D", "SELL", "Range resistance rejection candidate"))
        return candidates

    if primary == "UPTREND" and allowed_direction != "DOWN":
        return [
            ("C", "BUY", "Trend retest-hold candidate"),
            ("B", "BUY", "Trend breakout continuation candidate"),
            ("A", "BUY", "Trend EMA pullback candidate"),
        ]

    if primary == "DOWNTREND" and allowed_direction != "UP":
        return [
            ("C", "SELL", "Trend retest-fail candidate"),
            ("B", "SELL", "Trend breakdown continuation candidate"),
            ("A", "SELL", "Trend EMA rally-sell candidate"),
        ]

    return candidates


def build_rule_based_signal_decision(context: dict, df_exec: pd.DataFrame) -> dict[str, Any]:
    """
    Deterministic replacement for the old chart+LLM setup selection.

    It proposes a setup candidate from the current market regime, then runs the
    same structural/timing validator and trade planner already used by the live
    engine. This keeps execution deterministic and replayable.
    """
    base_context = dict(context or {})
    if "range_bias" not in base_context and base_context.get("is_range"):
        base_context["range_bias"] = get_range_bias(base_context)

    if df_exec is None or len(df_exec) < 8:
        return {
            "analysis": {
                "setup_identified": "None",
                "ema_check": "Insufficient execution candles.",
                "price_action": "Not enough candles to classify a setup.",
                "volume_check": "N/A",
                "rr_check": "Python would calculate R:R after a valid setup appears.",
                "pattern_match": "No setup candidate evaluated.",
            },
            "signal": "WAIT",
            "reason": "Not enough execution candles for deterministic setup selection.",
            "entry_price": float(base_context.get("current_price") or float(df_exec["close"].iloc[-1])),
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "_meta": {"selector": "rule_based"},
        }

    candidate_checks: list[dict[str, str]] = []
    current_price = float(base_context.get("current_price") or float(df_exec["close"].iloc[-1]))

    for setup_code, signal, candidate_reason in _candidate_setups_for_context(base_context):
        setup_name = f"Setup {setup_code}"
        precheck_ok, precheck_reason = _setup_precheck(setup_code, signal, base_context, df_exec)
        candidate_checks.append(
            {
                "setup": setup_name,
                "signal": signal,
                "stage": "precheck",
                "result": "PASS" if precheck_ok else "FAIL",
                "reason": precheck_reason,
            }
        )
        if not precheck_ok:
            continue

        decision = {
            "analysis": _build_analysis_payload(
                signal=signal,
                setup_name=setup_name,
                context=base_context,
                df_exec=df_exec,
                setup_reason=f"{candidate_reason}. {precheck_reason}.",
            ),
            "signal": signal,
            "reason": candidate_reason,
            "entry_price": current_price,
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "_meta": {"selector": "rule_based"},
        }

        valid, valid_reason = validate_ai_trade_decision(decision, base_context, df_exec)
        candidate_checks.append(
            {
                "setup": setup_name,
                "signal": signal,
                "stage": "validation",
                "result": "PASS" if valid else "FAIL",
                "reason": valid_reason,
            }
        )
        if not valid:
            continue

        plan, plan_reason = build_trade_plan(signal, setup_name, base_context, df_exec)
        candidate_checks.append(
            {
                "setup": setup_name,
                "signal": signal,
                "stage": "planning",
                "result": "PASS" if plan else "FAIL",
                "reason": plan_reason,
            }
        )
        if not plan:
            continue

        decision["reason"] = f"{candidate_reason}. {valid_reason}. {plan_reason}"
        decision["analysis"]["rr_check"] = (
            f"Python previewed Entry={plan['entry_price']} SL={plan['stop_loss']} "
            f"TP={plan['take_profit']} net R:R={plan['rr']:.2f}. Final levels will be recalculated before execution."
        )
        decision["entry_price"] = float(plan["entry_price"])
        decision["stop_loss"] = float(plan["stop_loss"])
        decision["take_profit"] = float(plan["take_profit"])
        decision["take_profit_1"] = float(plan["take_profit_1"])
        decision["take_profit_2"] = float(plan["take_profit_2"])
        decision["target_mode"] = str(plan["target_mode"])
        decision["planned_rr"] = float(plan["rr"])
        decision["_meta"]["candidate_checks"] = candidate_checks
        return decision

    return {
        "analysis": _build_analysis_payload(
            signal="WAIT",
            setup_name="None",
            context=base_context,
            df_exec=df_exec,
            setup_reason="No candidate setup passed deterministic precheck, validation, and trade planning.",
        ),
        "signal": "WAIT",
        "reason": "No deterministic setup passed validation and R:R planning.",
        "entry_price": current_price,
        "stop_loss": 0.0,
        "take_profit": 0.0,
        "_meta": {
            "selector": "rule_based",
            "candidate_checks": candidate_checks,
        },
    }


def build_trade_plan(signal: str, setup_name: str, context: dict,
                     df_exec: pd.DataFrame) -> tuple[dict | None, str]:
    """
    Deterministically builds entry, stop loss, and take profit from structure.
    Setup selection happens upstream; Python owns the trade levels.
    The caller chooses the execution frame used for structure.
    """
    signal = str(signal or "").upper()
    if signal not in ("BUY", "SELL"):
        return None, f"invalid signal {signal!r}"

    entry = float(context.get("current_price") or float(df_exec["close"].iloc[-1]))
    if entry <= 0:
        return None, "missing entry price"

    setup_code = _setup_code(setup_name)
    # Only Setup D should inherit range-style RR/TP/SL economics.
    # Other setups may appear during a sideways market, but they should still
    # be planned with their own trend-style trade math.
    is_range = setup_code == "D"
    if setup_code not in ("A", "B", "C", "D"):
        return None, f"unsupported setup {setup_code!r}; only Setup A/B/C/D allowed"

    atr = float(context.get("atr_m15") or context.get("atr") or 0.0)
    sr = context.get("sr") or {}
    support = sr.get("support")
    resistance = sr.get("resistance")
    support_levels = _dedupe_levels(
        list(sr.get("support_levels") or []) + ([support] if support is not None else []),
        reverse=True,
    )
    resistance_levels = _dedupe_levels(
        list(sr.get("resistance_levels") or []) + ([resistance] if resistance is not None else []),
    )

    exec_window = df_exec.tail(8).copy()
    recent_low = float(exec_window["low"].min())
    recent_high = float(exec_window["high"].max())
    pullback_low, pullback_high = _pullback_cluster_levels(df_exec, signal)
    deep_low, deep_high = _deep_structure_levels(df_exec, lookback=20)
    ema89 = float(df_exec["close"].astype(float).ewm(span=89, adjust=False).mean().iloc[-1])

    min_rr_default = max(0.5, _env_float("TRADE_MIN_RR", MIN_RR))
    min_rr_range = max(0.5, _env_float("TRADE_MIN_RR_RANGE", min_rr_default))
    fee_cost = entry * 0.0018
    if is_range:
        min_rr = min_rr_range
        base_buffer = _sr_buffer_distance(entry, atr, is_range=True)
        min_risk = _min_risk_distance(entry, atr, is_range=True)
        tp_extension = _tp_extension_distance(entry, atr, is_range=True)
    else:
        min_rr = min_rr_default
        base_buffer = _sr_buffer_distance(entry, atr, is_range=False)
        min_risk = _min_risk_distance(entry, atr, is_range=False)
        tp_extension = _tp_extension_distance(entry, atr, is_range=False)

    if signal == "BUY":
        if setup_code == "A":
            stop_refs = [pullback_low, recent_low, support, ema89]
        elif setup_code == "B":
            stop_refs = [recent_low, pullback_low, support, ema89]
        elif setup_code == "C":
            stop_refs = [pullback_low, recent_low, support, ema89]
        elif setup_code == "D":
            stop_refs = [deep_low, support, recent_low]
        else:
            stop_refs = [deep_low, pullback_low, recent_low, support, ema89]
        stop_candidates = _dedupe_levels([x for x in stop_refs if x is not None and x < entry], reverse=True)
        if not stop_candidates:
            return None, "no valid bullish invalidation below entry"

        if setup_code in ("A", "B", "C"):
            stop_anchor = stop_candidates[0]
        else:
            stop_anchor = stop_candidates[-1] if len(stop_candidates) > 1 else stop_candidates[0]
        raw_sl = stop_anchor - base_buffer
        if support is not None:
            raw_sl = min(raw_sl, float(support) - base_buffer)
        risk = entry - raw_sl
        if risk < min_risk:
            raw_sl = entry - min_risk

        target_candidates = [] if setup_code in ("B", "C") else [recent_high]
        target_candidates.extend(resistance_levels[:5])
        if setup_code in ("B", "C") and atr > 0:
            target_candidates.append(entry + atr * 3.5)
            target_candidates.append(entry + atr * 5.0)
            target_candidates.append(entry + atr * 6.5)
        target_levels = _dedupe_levels([x for x in target_candidates if x > entry])
        if tp_extension > 0:
            if is_range:
                # Range: exit just BEFORE resistance (price bounces back, don't need breakout)
                target_levels = _dedupe_levels([x - tp_extension for x in target_levels if (x - tp_extension) > entry])
            else:
                # Trend: push slightly past resistance (expect continuation)
                target_levels = _dedupe_levels([x + tp_extension for x in target_levels if (x + tp_extension) > entry])
        if not target_levels:
            return None, "no valid bullish target above entry"

        sl = raw_sl
        risk = entry - sl
        net_risk = risk + fee_cost
        tp1 = target_levels[0]
        tp2 = target_levels[1] if len(target_levels) > 1 else target_levels[0]

        def _buy_rr(target: float) -> tuple[float, float]:
            gross_reward = target - entry
            net_reward = gross_reward - fee_cost
            rr = net_reward / net_risk if net_risk > 0 else 0.0
            gross_rr = gross_reward / risk if risk > 0 else 0.0
            return rr, gross_rr

        tp1_rr, tp1_gross_rr = _buy_rr(tp1)
        tp2_rr, tp2_gross_rr = _buy_rr(tp2)

        chosen_tp = None
        best_target = tp1
        best_rr = tp1_rr
        best_gross_rr = tp1_gross_rr
        target_mode = "TP1"

        for idx, target in enumerate(target_levels):
            rr, gross_rr = _buy_rr(target)
            if rr > best_rr:
                best_target = target
                best_rr = rr
                best_gross_rr = gross_rr
            if rr >= min_rr:
                chosen_tp = target
                best_target = target
                best_rr = rr
                best_gross_rr = gross_rr
                target_mode = "TP1" if idx == 0 else "TP2"
                break
        if chosen_tp is None:
            return None, (
                f"TP1 RR {tp1_rr:.2f} < {min_rr:.2f}"
                + (f"; TP2 RR {tp2_rr:.2f}" if len(target_levels) > 1 else "")
                + f" (tp1={tp1:.4f})"
            )

    else:
        if setup_code == "A":
            stop_refs = [pullback_high, recent_high, resistance, ema89]
        elif setup_code == "B":
            stop_refs = [recent_high, pullback_high, resistance, ema89]
        elif setup_code == "C":
            stop_refs = [pullback_high, recent_high, resistance, ema89]
        elif setup_code == "D":
            stop_refs = [deep_high, resistance, recent_high]
        else:
            stop_refs = [deep_high, pullback_high, recent_high, resistance, ema89]
        stop_candidates = _dedupe_levels([x for x in stop_refs if x is not None and x > entry])
        if not stop_candidates:
            return None, "no valid bearish invalidation above entry"

        if setup_code in ("A", "B", "C"):
            stop_anchor = stop_candidates[0]
        else:
            stop_anchor = stop_candidates[-1] if len(stop_candidates) > 1 else stop_candidates[0]
        raw_sl = stop_anchor + base_buffer
        if resistance is not None:
            raw_sl = max(raw_sl, float(resistance) + base_buffer)
        risk = raw_sl - entry
        if risk < min_risk:
            raw_sl = entry + min_risk

        target_candidates = [] if setup_code in ("B", "C") else [recent_low]
        target_candidates.extend(support_levels[:5])
        if setup_code in ("B", "C") and atr > 0:
            target_candidates.append(entry - atr * 3.5)
            target_candidates.append(entry - atr * 5.0)
            target_candidates.append(entry - atr * 6.5)
        target_levels = _dedupe_levels([x for x in target_candidates if x < entry], reverse=True)
        if tp_extension > 0:
            if is_range:
                # Range: exit just BEFORE support (price bounces back up, don't need breakdown)
                target_levels = _dedupe_levels(
                    [x + tp_extension for x in target_levels if (x + tp_extension) < entry],
                    reverse=True,
                )
            else:
                target_levels = _dedupe_levels(
                [x - tp_extension for x in target_levels if (x - tp_extension) < entry],
                reverse=True,
            )
        if not target_levels:
            return None, "no valid bearish target below entry"

        sl = raw_sl
        risk = sl - entry
        net_risk = risk + fee_cost
        tp1 = target_levels[0]
        tp2 = target_levels[1] if len(target_levels) > 1 else target_levels[0]

        def _sell_rr(target: float) -> tuple[float, float]:
            gross_reward = entry - target
            net_reward = gross_reward - fee_cost
            rr = net_reward / net_risk if net_risk > 0 else 0.0
            gross_rr = gross_reward / risk if risk > 0 else 0.0
            return rr, gross_rr

        tp1_rr, tp1_gross_rr = _sell_rr(tp1)
        tp2_rr, tp2_gross_rr = _sell_rr(tp2)

        chosen_tp = None
        best_target = tp1
        best_rr = tp1_rr
        best_gross_rr = tp1_gross_rr
        target_mode = "TP1"

        for idx, target in enumerate(target_levels):
            rr, gross_rr = _sell_rr(target)
            if rr > best_rr:
                best_target = target
                best_rr = rr
                best_gross_rr = gross_rr
            if rr >= min_rr:
                chosen_tp = target
                best_target = target
                best_rr = rr
                best_gross_rr = gross_rr
                target_mode = "TP1" if idx == 0 else "TP2"
                break
        if chosen_tp is None:
            return None, (
                f"TP1 RR {tp1_rr:.2f} < {min_rr:.2f}"
                + (f"; TP2 RR {tp2_rr:.2f}" if len(target_levels) > 1 else "")
                + f" (tp1={tp1:.4f})"
            )

    final_risk_pct = abs(entry - sl) / entry
    max_stop_pct = compute_max_stop_pct(setup_name, context, entry=entry, atr=atr)
    if final_risk_pct > max_stop_pct:
        return None, f"stop distance {final_risk_pct*100:.3f}% exceeds max {max_stop_pct*100:.2f}%"

    plan = {
        "entry_price": round(entry, 6),
        "stop_loss": round(sl, 6),
        "take_profit": round(chosen_tp, 6),
        "take_profit_1": round(tp1, 6),
        "take_profit_2": round(tp2, 6),
        "target_mode": target_mode,
        "rr": round(best_rr, 2),
        "gross_rr": round(best_gross_rr, 2),
        "tp1_rr": round(tp1_rr, 2),
        "tp2_rr": round(tp2_rr, 2),
        "min_rr": round(min_rr, 2),
        "stop_anchor": round(stop_anchor, 6),
        "target_anchor": round(best_target, 6),
        "tp_extension": round(tp_extension, 6),
        "estimated_cost": round(fee_cost, 6),
    }
    reason = (
        f"stop@{plan['stop_anchor']} TP1@{plan['take_profit_1']} TP2@{plan['take_profit_2']} "
        f"selected={plan['target_mode']} RRnet={plan['rr']:.2f} RRgross={plan['gross_rr']:.2f} "
        f"RR1={plan['tp1_rr']:.2f} RR2={plan['tp2_rr']:.2f} min={plan['min_rr']:.2f} "
        f"tp_ext={plan['tp_extension']:.6f}"
    )
    return plan, reason
