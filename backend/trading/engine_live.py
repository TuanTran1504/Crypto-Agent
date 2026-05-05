"""
engine_live.py — Live account trading engine (BTC, ETH, SOL only).

Identical strategy to engine.py but:
  - Uses BINANCE_LIVE_API_KEY / BINANCE_LIVE_SECRET_KEY (real Binance, not testnet)
  - Trades BTC, ETH, SOL only (XRP excluded — min notional risk at small balance)
  - Exchange-side STOP_MARKET / TAKE_PROFIT_MARKET for SL/TP (works on live)
  - Software monitor as backup SL/TP in case exchange orders fail
  - account_type='live' on all DB records — separate from testnet trades
  - Min notional check: skips trades where order value is below Binance minimum
"""

import argparse
import hashlib
import hmac
import json
import math
import os
import re
import time
import threading
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone, timedelta
UTC = timezone.utc
VN_TZ = timezone(timedelta(hours=7))
from pathlib import Path
from typing import Any

import psycopg2
from psycopg2 import errors
from psycopg2 import OperationalError, InterfaceError
import redis
from binance.um_futures import UMFutures
from dotenv import load_dotenv

from strategy_core  import (
    compute_indicators, compute_score, find_sr_levels,
    check_macro_bias, check_technical_gates, get_range_bias,
    validate_ai_trade_decision, build_trade_plan, compute_max_stop_pct,
    build_rule_based_signal_decision,
)
from strategy_policy import (
    ActiveStrategyPolicy,
    apply_policy_to_environment,
    capture_managed_environment,
    guard_risk_increase_overrides,
    load_active_strategy_policy,
    policy_allows_risk_increase,
    reset_managed_environment,
    setup_enabled_in_policy,
    symbol_enabled_in_policy,
)
from trade_logger   import (
    get_logger, log_cycle_start, log_gate_pass, log_gate_fail,
    log_trade_open, log_trade_close,
    log_skip, log_error, log_cycle_summary,
)

ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / ".env")

log = get_logger()

API_KEY      = os.getenv("BINANCE_LIVE_API_KEY", "")
API_SECRET   = os.getenv("BINANCE_LIVE_SECRET_KEY", "")
DATABASE_URL = os.getenv("DATABASE_URL", "")
REDIS_URL    = os.getenv("REDIS_URL", "")

ACCOUNT_TYPE   = "live"
TRADES_TABLE   = "trades_live"
FILLS_TABLE    = "trade_fills_live"
OPPORTUNITIES_TABLE = "strategy_opportunities_live"
SYMBOLS        = ["BTC", "ETH", "SOL"]
STRATEGY_POLICY_ENGINE_NAME = os.getenv("STRATEGY_POLICY_ENGINE_NAME", "llm_live")
LEVERAGE       = 5
try:
    DEFAULT_MIN_NOTIONAL = max(
        0.0,
        float(os.getenv("BINANCE_LIVE_MIN_NOTIONAL", os.getenv("BINANCE_MIN_NOTIONAL", "100"))),
    )
except (TypeError, ValueError):
    DEFAULT_MIN_NOTIONAL = 100.0

QTY_PRECISION  = {
    "BTC": 3,
    "ETH": 2,
    "SOL": 1,
}
PRICE_PRECISION = {
    "BTC": 1,   # tick size 0.1  → e.g. 84234.5
    "ETH": 2,   # tick size 0.01 → e.g. 2113.95
    "SOL": 2,   # tick size 0.01 → e.g. 79.87
    "XRP": 4,   # tick size 0.0001 → e.g. 2.1234
}
STOP_LOSS_PCT         = 0.008
BREAKOUT_STOP_LOSS_PCT = 0.013
SL_MIN_PCT            = 0.002
try:
    SL_ATR_MULTIPLIER = max(0.0, float(os.getenv("SL_ATR_MULTIPLIER", "1.8")))
except (TypeError, ValueError):
    SL_ATR_MULTIPLIER = 1.8
try:
    SL_SR_BUFFER_ATR_MULT = max(0.0, float(os.getenv("SL_SR_BUFFER_ATR_MULT", "0.25")))
except (TypeError, ValueError):
    SL_SR_BUFFER_ATR_MULT = 0.25
try:
    TAKE_PROFIT_MIN_RR = max(0.5, float(os.getenv("TRADE_MIN_RR", "1.5")))
except (TypeError, ValueError):
    TAKE_PROFIT_MIN_RR = 1.5
try:
    RANGE_MIN_RR = max(0.5, float(os.getenv("TRADE_MIN_RR_RANGE", str(TAKE_PROFIT_MIN_RR))))
except (TypeError, ValueError):
    RANGE_MIN_RR = TAKE_PROFIT_MIN_RR
try:
    TP_EXTENSION_ATR_MULT = max(0.0, float(os.getenv("TP_EXTENSION_ATR_MULT", "0.15")))
except (TypeError, ValueError):
    TP_EXTENSION_ATR_MULT = 0.15
try:
    SETUP_E_MIN_RR = max(0.5, float(os.getenv("SETUP_E_MIN_RR", "1.0")))
except (TypeError, ValueError):
    SETUP_E_MIN_RR = 1.0
try:
    POSITION_RISK_PCT = max(0.0, float(os.getenv("POSITION_RISK_PCT", "0.01")))
except (TypeError, ValueError):
    POSITION_RISK_PCT = 0.01
try:
    BTC_POSITION_RISK_PCT = max(0.0, float(os.getenv("BTC_POSITION_RISK_PCT", "0.005")))
except (TypeError, ValueError):
    BTC_POSITION_RISK_PCT = 0.005
POSITION_RISK_PCT_BY_SYMBOL = {
    "BTC": BTC_POSITION_RISK_PCT,
}
try:
    MAX_POSITION_FRACTION = max(0.0, float(os.getenv("MAX_POSITION_FRACTION", "0.10")))
except (TypeError, ValueError):
    MAX_POSITION_FRACTION = 0.10
try:
    BTC_MAX_POSITION_FRACTION = max(0.0, float(os.getenv("BTC_MAX_POSITION_FRACTION", "0.15")))
except (TypeError, ValueError):
    BTC_MAX_POSITION_FRACTION = 0.15
MAX_POSITION_FRACTION_BY_SYMBOL = {
    "BTC": BTC_MAX_POSITION_FRACTION,
}
try:
    BREAK_EVEN_TRIGGER_R_MULT = max(0.0, float(os.getenv("BREAK_EVEN_TRIGGER_R_MULT", "0.5")))
except (TypeError, ValueError):
    BREAK_EVEN_TRIGGER_R_MULT = 0.5
try:
    BTC_BREAK_EVEN_TRIGGER_R_MULT = max(
        0.0, float(os.getenv("BTC_BREAK_EVEN_TRIGGER_R_MULT", "1.0"))
    )
except (TypeError, ValueError):
    BTC_BREAK_EVEN_TRIGGER_R_MULT = 1.0
BREAK_EVEN_TRIGGER_R_MULT_BY_SYMBOL = {
    "BTC": BTC_BREAK_EVEN_TRIGGER_R_MULT,
}
try:
    BTC_TRAIL_LOOKBACK_BARS = max(2, int(os.getenv("BTC_TRAIL_LOOKBACK_BARS", "2")))
except (TypeError, ValueError):
    BTC_TRAIL_LOOKBACK_BARS = 2
try:
    BTC_TRAIL_STOP_ATR_MULT = max(0.0, float(os.getenv("BTC_TRAIL_STOP_ATR_MULT", "0.35")))
except (TypeError, ValueError):
    BTC_TRAIL_STOP_ATR_MULT = 0.35
try:
    BTC_TRAIL_MARK_BUFFER_PCT = max(0.0, float(os.getenv("BTC_TRAIL_MARK_BUFFER_PCT", "0.0015")))
except (TypeError, ValueError):
    BTC_TRAIL_MARK_BUFFER_PCT = 0.0015
CYCLE_INTERVAL        = 5 * 60
MONITOR_INTERVAL      = 5
ALLOWED_ML_HORIZONS = {"4h", "1d"}
_SYMBOL_MIN_NOTIONAL_CACHE: dict[str, float] = {}
# ML predictions are intentionally disabled in live trading for now.
# Keep the plumbing for later experiments, but do not let ML influence entries.
ML_EXPERIMENTAL_ENABLED = False
_BASE_MANAGED_ENV = capture_managed_environment()
_ACTIVE_STRATEGY_POLICY: ActiveStrategyPolicy | None = None


def _runtime_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _runtime_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(default)


def _refresh_runtime_config_from_env():
    global SL_ATR_MULTIPLIER
    global SL_SR_BUFFER_ATR_MULT
    global TAKE_PROFIT_MIN_RR
    global RANGE_MIN_RR
    global TP_EXTENSION_ATR_MULT
    global SETUP_E_MIN_RR
    global POSITION_RISK_PCT
    global BTC_POSITION_RISK_PCT
    global POSITION_RISK_PCT_BY_SYMBOL
    global MAX_POSITION_FRACTION
    global BTC_MAX_POSITION_FRACTION
    global MAX_POSITION_FRACTION_BY_SYMBOL
    global BREAK_EVEN_TRIGGER_R_MULT
    global BTC_BREAK_EVEN_TRIGGER_R_MULT
    global BREAK_EVEN_TRIGGER_R_MULT_BY_SYMBOL
    global BTC_TRAIL_LOOKBACK_BARS
    global BTC_TRAIL_STOP_ATR_MULT
    global BTC_TRAIL_MARK_BUFFER_PCT

    SL_ATR_MULTIPLIER = max(0.0, _runtime_float("SL_ATR_MULTIPLIER", 1.8))
    SL_SR_BUFFER_ATR_MULT = max(0.0, _runtime_float("SL_SR_BUFFER_ATR_MULT", 0.25))
    TAKE_PROFIT_MIN_RR = max(0.5, _runtime_float("TRADE_MIN_RR", 1.5))
    RANGE_MIN_RR = max(0.5, _runtime_float("TRADE_MIN_RR_RANGE", TAKE_PROFIT_MIN_RR))
    TP_EXTENSION_ATR_MULT = max(0.0, _runtime_float("TP_EXTENSION_ATR_MULT", 0.15))
    SETUP_E_MIN_RR = max(0.5, _runtime_float("SETUP_E_MIN_RR", 1.0))
    POSITION_RISK_PCT = max(0.0, _runtime_float("POSITION_RISK_PCT", 0.01))
    BTC_POSITION_RISK_PCT = max(0.0, _runtime_float("BTC_POSITION_RISK_PCT", 0.005))
    POSITION_RISK_PCT_BY_SYMBOL = {"BTC": BTC_POSITION_RISK_PCT}
    MAX_POSITION_FRACTION = max(0.0, _runtime_float("MAX_POSITION_FRACTION", 0.10))
    BTC_MAX_POSITION_FRACTION = max(0.0, _runtime_float("BTC_MAX_POSITION_FRACTION", 0.15))
    MAX_POSITION_FRACTION_BY_SYMBOL = {"BTC": BTC_MAX_POSITION_FRACTION}
    BREAK_EVEN_TRIGGER_R_MULT = max(0.0, _runtime_float("BREAK_EVEN_TRIGGER_R_MULT", 0.5))
    BTC_BREAK_EVEN_TRIGGER_R_MULT = max(
        0.0, _runtime_float("BTC_BREAK_EVEN_TRIGGER_R_MULT", 1.0)
    )
    BREAK_EVEN_TRIGGER_R_MULT_BY_SYMBOL = {"BTC": BTC_BREAK_EVEN_TRIGGER_R_MULT}
    BTC_TRAIL_LOOKBACK_BARS = max(2, _runtime_int("BTC_TRAIL_LOOKBACK_BARS", 2))
    BTC_TRAIL_STOP_ATR_MULT = max(0.0, _runtime_float("BTC_TRAIL_STOP_ATR_MULT", 0.35))
    BTC_TRAIL_MARK_BUFFER_PCT = max(0.0, _runtime_float("BTC_TRAIL_MARK_BUFFER_PCT", 0.0015))


def _refresh_runtime_strategy_policy() -> tuple[ActiveStrategyPolicy | None, dict[str, str], list[str], str]:
    global _ACTIVE_STRATEGY_POLICY
    reset_managed_environment(_BASE_MANAGED_ENV)
    policy = load_active_strategy_policy(STRATEGY_POLICY_ENGINE_NAME, ACCOUNT_TYPE)
    applied: dict[str, str] = {}
    blocked: list[str] = []
    guard_reason = "risk governor not needed"
    if policy:
        applied = apply_policy_to_environment(policy.policy_json)
        allow_risk_increase, guard_reason = policy_allows_risk_increase(policy)
        applied, blocked = guard_risk_increase_overrides(
            applied,
            _BASE_MANAGED_ENV,
            allow_risk_increase=allow_risk_increase,
        )
    _refresh_runtime_config_from_env()
    _ACTIVE_STRATEGY_POLICY = policy
    return policy, applied, blocked, guard_reason


def _normalize_horizon_token(token: str) -> str:
    t = str(token).strip().lower()
    if len(t) < 2 or t[-1] not in {"d", "h"} or not t[:-1].isdigit():
        raise ValueError(f"Invalid horizon token '{token}'. Use values like 1d, 4h.")
    return f"{int(t[:-1])}{t[-1]}"


def _parse_horizon_tokens(raw: str) -> list[str]:
    tokens: list[str] = []
    for part in raw.split(","):
        p = part.strip()
        if not p:
            continue
        token = _normalize_horizon_token(p)
        if token not in ALLOWED_ML_HORIZONS:
            continue
        if token not in tokens:
            tokens.append(token)
    return tokens or ["4h", "1d"]


def _sanitize_horizon(token: str, fallback: str) -> str:
    return token if token in ALLOWED_ML_HORIZONS else fallback


ML_HORIZONS = _parse_horizon_tokens(os.getenv("ML_HORIZONS", "4h,1d"))
ML_PRIMARY_HORIZON = _sanitize_horizon(
    _normalize_horizon_token(os.getenv("ML_PRIMARY_HORIZON", ML_HORIZONS[0])),
    ML_HORIZONS[0],
)
if ML_PRIMARY_HORIZON not in ML_HORIZONS:
    ML_HORIZONS = [ML_PRIMARY_HORIZON] + ML_HORIZONS
ML_TREND_HORIZON = _sanitize_horizon(
    _normalize_horizon_token(os.getenv("ML_TREND_HORIZON", "1d")),
    "1d",
)
ML_REQUIRE_TREND_ALIGNMENT = os.getenv("ML_REQUIRE_TREND_ALIGNMENT", "1") == "1"
ML_CONFLICT_MODE = os.getenv("ML_CONFLICT_MODE", "higher_confidence").strip().lower()
if ML_CONFLICT_MODE not in {"skip", "higher_confidence"}:
    ML_CONFLICT_MODE = "higher_confidence"
try:
    ML_CONFLICT_MIN_RR = max(TAKE_PROFIT_MIN_RR, float(os.getenv("ML_CONFLICT_MIN_RR", "2.0")))
except (TypeError, ValueError):
    ML_CONFLICT_MIN_RR = 2.0


def get_client() -> UMFutures:
    return UMFutures(key=API_KEY, secret=API_SECRET)  # no base_url = live


def _algo_order(symbol: str, side: str, order_type: str, trigger_price: float,
                quantity: float | None = None) -> dict:
    """Place SL/TP via Binance Algo Order API (/fapi/v1/algoOrder).

    quantity=None  → closePosition=true (closes entire position — used for SL and single TP)
    quantity=N     → partial close of N units (used for staged TP1 / TP2)
    """
    base      = symbol.replace("USDT", "")
    precision = PRICE_PRECISION.get(base, 2)
    qty_prec  = QTY_PRECISION.get(base, 3)
    params = {
        "algoType":     "CONDITIONAL",
        "symbol":       symbol,
        "side":         side,
        "type":         order_type,
        "triggerPrice": f"{trigger_price:.{precision}f}",
        "workingType":  "MARK_PRICE",
        "timestamp":    int(time.time() * 1000),
    }
    if quantity is not None:
        params["quantity"] = f"{quantity:.{qty_prec}f}"
    else:
        params["closePosition"] = "true"
    query = urllib.parse.urlencode(params)
    sig   = hmac.new(API_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
    url   = f"https://fapi.binance.com/fapi/v1/algoOrder?{query}&signature={sig}"
    req   = urllib.request.Request(url, method="POST", headers={"X-MBX-APIKEY": API_KEY})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def _list_open_algo_orders(symbol: str) -> list:
    """Return open algo orders for a symbol from Binance."""
    params = {"symbol": symbol, "timestamp": int(time.time() * 1000)}
    query = urllib.parse.urlencode(params)
    sig   = hmac.new(API_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
    url   = f"https://fapi.binance.com/fapi/v1/openAlgoOrders?{query}&signature={sig}"
    req   = urllib.request.Request(url, headers={"X-MBX-APIKEY": API_KEY})
    with urllib.request.urlopen(req, timeout=10) as r:
        data = json.loads(r.read())
    if isinstance(data, list):
        return data
    return data.get("orders", [])


def _cancel_algo_order(algo_id: int) -> dict:
    """Cancel an algo order by algoId."""
    params = {"algoId": algo_id, "timestamp": int(time.time() * 1000)}
    query = urllib.parse.urlencode(params)
    sig   = hmac.new(API_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
    url   = f"https://fapi.binance.com/fapi/v1/algoOrder?{query}&signature={sig}"
    req   = urllib.request.Request(url, method="DELETE", headers={"X-MBX-APIKEY": API_KEY})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def _get_position_risk_pct(symbol: str) -> float:
    return POSITION_RISK_PCT_BY_SYMBOL.get(symbol, POSITION_RISK_PCT)


def _get_max_position_fraction(symbol: str) -> float:
    return MAX_POSITION_FRACTION_BY_SYMBOL.get(symbol, MAX_POSITION_FRACTION)


def _get_break_even_trigger_r_mult(symbol: str) -> float:
    return BREAK_EVEN_TRIGGER_R_MULT_BY_SYMBOL.get(symbol, BREAK_EVEN_TRIGGER_R_MULT)


def _round_price(symbol: str, price: float) -> float:
    base = symbol.replace("USDT", "")
    return round(price, PRICE_PRECISION.get(base, 2))


def _replace_stop_loss_algo(symbol: str, trade_side: str, mark: float, new_sl: float):
    """
    Replace the active protective stop.

    We identify the stop as the close-side trigger that still sits on the loss
    side of the current mark price; TP orders remain on the profit side.
    """
    close_side = "SELL" if trade_side == "BUY" else "BUY"
    algo_orders = _list_open_algo_orders(f"{symbol}USDT")
    for ao in algo_orders:
        ao_trigger = float(ao.get("triggerPrice") or 0)
        ao_side = ao.get("side", "")
        if not ao_trigger or ao_side != close_side:
            continue
        is_stop = (
            (trade_side == "BUY" and ao_trigger < mark)
            or (trade_side == "SELL" and ao_trigger > mark)
        )
        if not is_stop:
            continue
        _cancel_algo_order(ao["algoId"])
        log.info(f"[MONITOR] {symbol} cancelled stop algo {ao['algoId']} @ {ao_trigger}")
    _algo_order(f"{symbol}USDT", close_side, "STOP_MARKET", new_sl)


def _get_conn():
    conn = psycopg2.connect(
        DATABASE_URL,
        sslmode="require",
        connect_timeout=8,
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=3,
    )
    with conn.cursor() as cur:
        cur.execute("SET TIME ZONE 'Asia/Ho_Chi_Minh'")
    conn.commit()
    return conn


def _is_transient_db_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    transient_markers = (
        "ssl connection has been closed unexpectedly",
        "server closed the connection unexpectedly",
        "connection not open",
        "terminating connection due to administrator command",
        "could not receive data from server",
    )
    return any(marker in msg for marker in transient_markers)


def _json_default(value: Any):
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC).isoformat()
    return str(value)


def _compact_json(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=True, default=_json_default))
    except Exception:
        return {"serialization_error": str(value)}


def _extract_rr_diagnostics(plan_reason: str) -> dict[str, float] | None:
    text = str(plan_reason or "")
    match = re.search(r"TP1 RR ([0-9.]+) < ([0-9.]+)", text)
    if not match:
        return None
    out = {
        "tp1_rr": float(match.group(1)),
        "min_rr": float(match.group(2)),
    }
    match_tp2 = re.search(r"TP2 RR ([0-9.]+)", text)
    if match_tp2:
        out["tp2_rr"] = float(match_tp2.group(1))
        out["best_rr"] = max(out["tp1_rr"], out["tp2_rr"])
    else:
        out["best_rr"] = out["tp1_rr"]
    return out


def _build_context_snapshot(ctx: dict[str, Any]) -> dict[str, Any]:
    sr = dict(ctx.get("sr") or {})
    return {
        "symbol": ctx.get("symbol"),
        "market_mode": ctx.get("market_mode"),
        "score": ctx.get("score"),
        "score_details": list(ctx.get("score_details") or []),
        "current_price": ctx.get("current_price"),
        "primary_trend": ctx.get("primary_trend"),
        "h1_trend": ctx.get("h1_trend"),
        "m15_trend": ctx.get("m15_trend"),
        "btc_trend": ctx.get("btc_trend"),
        "btc_macro_trend": ctx.get("btc_macro_trend"),
        "allowed_direction": ctx.get("allowed_direction"),
        "is_range": ctx.get("is_range"),
        "range_bias": ctx.get("range_bias"),
        "rsi": ctx.get("rsi"),
        "atr_m15": ctx.get("atr_m15"),
        "adx_m15": ctx.get("adx_m15"),
        "m15_gap": ctx.get("m15_gap"),
        "fear_greed": ctx.get("fear_greed"),
        "funding_rate": ctx.get("funding_rate"),
        "sr": {
            "support": sr.get("support"),
            "resistance": sr.get("resistance"),
            "prev_support": sr.get("prev_support"),
            "prev_resistance": sr.get("prev_resistance"),
            "breakout_state": sr.get("breakout_state"),
            "breakout_confirmed": sr.get("breakout_confirmed"),
            "bars_since_breakout": sr.get("bars_since_breakout"),
        },
    }


def _build_preview_plan_payload(plan: dict[str, Any] | None, reason: str) -> dict[str, Any]:
    if plan:
        payload = dict(plan)
        payload["status"] = "READY"
        payload["reason"] = reason
        return _compact_json(payload)
    payload = {
        "status": "REJECTED",
        "reason": reason,
    }
    rr_diagnostics = _extract_rr_diagnostics(reason)
    if rr_diagnostics:
        payload.update(rr_diagnostics)
    return payload


def _decision_for_snapshot(decision: dict[str, Any], plan: dict[str, Any] | None = None) -> dict[str, Any]:
    snapshot = _compact_json(decision if isinstance(decision, dict) else {})
    if not isinstance(snapshot, dict):
        snapshot = {}
    if plan:
        snapshot["entry_price"] = plan.get("entry_price")
        snapshot["stop_loss"] = plan.get("stop_loss")
        snapshot["take_profit"] = plan.get("take_profit")
        snapshot["take_profit_1"] = plan.get("take_profit_1")
        snapshot["take_profit_2"] = plan.get("take_profit_2")
        snapshot["target_mode"] = plan.get("target_mode")
        snapshot["planned_rr"] = plan.get("rr")
    return snapshot


def db_record_opportunity(
    *,
    symbol: str,
    signal: str,
    setup: str,
    status: str,
    status_reason: str,
    context: dict[str, Any],
    decision: dict[str, Any],
    preview_plan_payload: dict[str, Any],
    linked_trade_id: int | None = None,
    execution_ref: str | None = None,
):
    policy_id = _ACTIVE_STRATEGY_POLICY.id if _ACTIVE_STRATEGY_POLICY else None
    policy_version = _ACTIVE_STRATEGY_POLICY.version if _ACTIVE_STRATEGY_POLICY else None
    decision_analysis = dict(decision.get("analysis") or {})
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {OPPORTUNITIES_TABLE}
                  (engine_name, account_type, symbol, cycle_at, signal, setup, status,
                   status_reason, decision_reason, market_mode, primary_trend,
                   allowed_direction, is_range, range_bias, score, current_price,
                   entry_price, stop_loss, take_profit, take_profit_2, planned_rr, target_mode,
                   strategy_policy_id, strategy_policy_version, linked_trade_id, execution_ref,
                   preview_plan_json, context_json, decision_json)
                VALUES
                  (%s, %s, %s, NOW(), %s, %s, %s,
                   %s, %s, %s, %s,
                   %s, %s, %s, %s, %s,
                   %s, %s, %s, %s, %s, %s,
                   %s, %s, %s, %s,
                   %s::jsonb, %s::jsonb, %s::jsonb)
                """,
                (
                    STRATEGY_POLICY_ENGINE_NAME,
                    ACCOUNT_TYPE,
                    symbol,
                    signal,
                    setup,
                    status,
                    status_reason[:500],
                    str(decision.get("reason") or "")[:500],
                    context.get("market_mode"),
                    context.get("primary_trend") or context.get("m15_trend"),
                    context.get("allowed_direction"),
                    bool(context.get("is_range", False)),
                    context.get("range_bias"),
                    context.get("score"),
                    context.get("current_price"),
                    decision.get("entry_price"),
                    decision.get("stop_loss"),
                    decision.get("take_profit"),
                    decision.get("take_profit_2"),
                    decision.get("planned_rr"),
                    decision.get("target_mode"),
                    policy_id,
                    policy_version,
                    linked_trade_id,
                    execution_ref,
                    json.dumps(_compact_json(preview_plan_payload), ensure_ascii=True, default=_json_default),
                    json.dumps(_build_context_snapshot(context), ensure_ascii=True, default=_json_default),
                    json.dumps(
                        _compact_json(
                            {
                                "signal": decision.get("signal"),
                                "reason": decision.get("reason"),
                                "analysis": decision_analysis,
                                "entry_price": decision.get("entry_price"),
                                "stop_loss": decision.get("stop_loss"),
                                "take_profit": decision.get("take_profit"),
                                "take_profit_1": decision.get("take_profit_1"),
                                "take_profit_2": decision.get("take_profit_2"),
                                "target_mode": decision.get("target_mode"),
                                "planned_rr": decision.get("planned_rr"),
                                "_meta": decision.get("_meta"),
                            }
                        ),
                        ensure_ascii=True,
                        default=_json_default,
                    ),
                ),
            )
        conn.commit()
    except Exception as e:
        log.warning(f"[DB] Opportunity snapshot failed {symbol} status={status}: {e}")
    finally:
        conn.close()


def db_ensure_trades_table():
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {TRADES_TABLE} (
                    id                BIGSERIAL PRIMARY KEY,
                    symbol            TEXT        NOT NULL,
                    side              TEXT        NOT NULL,
                    status            TEXT        NOT NULL DEFAULT 'OPEN',
                    entry_price       FLOAT,
                    exit_price        FLOAT,
                    quantity          FLOAT,
                    leverage          INTEGER     DEFAULT 5,
                    stop_loss         FLOAT,
                    take_profit       FLOAT,
                    pnl_usdt          FLOAT,
                    pnl_pct           FLOAT,
                    confidence        FLOAT,
                    horizon           INTEGER     DEFAULT 1,
                    binance_order_id  TEXT,
                    close_reason      TEXT,
                    setup             TEXT,
                    notes             TEXT,
                    opened_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    closed_at         TIMESTAMPTZ
                )
            """)
            for col, definition in [
                ("setup",            "TEXT"),
                ("notes",            "TEXT"),
                ("confidence",       "FLOAT"),
                ("binance_order_id", "TEXT"),
                ("close_reason",     "TEXT"),
                ("account_type",     "TEXT DEFAULT 'testnet'"),
                ("take_profit_2",    "FLOAT"),
                ("close_order_id",   "TEXT"),
                ("tp1_recorded",     "BOOLEAN DEFAULT FALSE"),
                ("strategy_policy_id", "BIGINT"),
                ("strategy_policy_version", "INTEGER"),
            ]:
                cur.execute(f"""
                    ALTER TABLE {TRADES_TABLE} ADD COLUMN IF NOT EXISTS {col} {definition}
                """)
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {FILLS_TABLE} (
                    id          BIGSERIAL   PRIMARY KEY,
                    trade_id    BIGINT      NOT NULL,
                    fill_type   TEXT        NOT NULL,
                    price       FLOAT       NOT NULL,
                    quantity    FLOAT       NOT NULL,
                    fee_usdt    FLOAT       NOT NULL DEFAULT 0,
                    pnl_usdt    FLOAT       NOT NULL DEFAULT 0,
                    order_id    TEXT,
                    filled_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {OPPORTUNITIES_TABLE} (
                    id                      BIGSERIAL PRIMARY KEY,
                    engine_name             TEXT NOT NULL,
                    account_type            TEXT NOT NULL,
                    symbol                  TEXT NOT NULL,
                    cycle_at                TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    signal                  TEXT,
                    setup                   TEXT,
                    status                  TEXT NOT NULL,
                    status_reason           TEXT,
                    decision_reason         TEXT,
                    market_mode             TEXT,
                    primary_trend           TEXT,
                    allowed_direction       TEXT,
                    is_range                BOOLEAN DEFAULT FALSE,
                    range_bias              TEXT,
                    score                   INTEGER,
                    current_price           FLOAT,
                    entry_price             FLOAT,
                    stop_loss               FLOAT,
                    take_profit             FLOAT,
                    take_profit_2           FLOAT,
                    planned_rr              FLOAT,
                    target_mode             TEXT,
                    strategy_policy_id      BIGINT,
                    strategy_policy_version INTEGER,
                    linked_trade_id         BIGINT,
                    execution_ref           TEXT,
                    preview_plan_json       JSONB DEFAULT '{{}}'::jsonb,
                    context_json            JSONB DEFAULT '{{}}'::jsonb,
                    decision_json           JSONB DEFAULT '{{}}'::jsonb
                )
            """)
            cur.execute(f"""
                CREATE UNIQUE INDEX IF NOT EXISTS {TRADES_TABLE}_one_open_position_per_account_symbol_idx
                ON {TRADES_TABLE} (account_type, symbol)
                WHERE status = 'OPEN'
            """)
            cur.execute(f"""
                CREATE UNIQUE INDEX IF NOT EXISTS {TRADES_TABLE}_one_pending_per_account_symbol_idx
                ON {TRADES_TABLE} (account_type, symbol)
                WHERE status = 'PENDING'
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {OPPORTUNITIES_TABLE}_account_cycle_idx
                ON {OPPORTUNITIES_TABLE} (account_type, cycle_at DESC)
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {OPPORTUNITIES_TABLE}_policy_cycle_idx
                ON {OPPORTUNITIES_TABLE} (strategy_policy_id, cycle_at DESC)
            """)
        conn.commit()
    finally:
        conn.close()


def _classify_fill_type(price: float, entry: float, sl: float, tp: float,
                         tp2: float, side: str) -> str:
    if not price or not entry:
        return "CLOSE"
    # Pick the closest stored price level within a reasonable tolerance.
    # Using closest-wins avoids mislabeling when SL and TP are near each other
    # (e.g. break-even SL = entry, TP1 just below for a SHORT).
    tol = entry * 0.005  # 0.5% of entry price
    candidates = []
    if sl:
        candidates.append(("SL", abs(price - sl)))
    if tp2:
        candidates.append(("TP2", abs(price - tp2)))
    if tp:
        candidates.append(("TP1" if tp2 else "TP", abs(price - tp)))
    if candidates:
        closest_type, closest_dist = min(candidates, key=lambda x: x[1])
        if closest_dist <= tol:
            return closest_type
    if side == "BUY":
        return "TP" if price > entry else "SL"
    return "TP" if price < entry else "SL"


def db_record_fill(trade_id: int, fill_type: str, price: float, qty: float,
                   order_id: str | None, entry_price: float, trade_side: str,
                   filled_at_ms: int | None = None):
    fee_usdt = price * qty * 0.0004
    if fill_type == "ENTRY":
        pnl_usdt = -fee_usdt
    else:
        gross = (price - entry_price) * qty if trade_side == "BUY" else (entry_price - price) * qty
        pnl_usdt = gross - fee_usdt
    filled_at = datetime.fromtimestamp(filled_at_ms / 1000, tz=UTC) if filled_at_ms else None
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                INSERT INTO {FILLS_TABLE}
                  (trade_id, fill_type, price, quantity, fee_usdt, pnl_usdt, order_id, filled_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (trade_id, fill_type, round(price, 8), round(qty, 8),
                  round(fee_usdt, 6), round(pnl_usdt, 6),
                  order_id, filled_at or datetime.now(UTC)))
        conn.commit()
        log.info(f"[DB] Fill trade={trade_id} type={fill_type} price={price} qty={qty} pnl={pnl_usdt:+.4f}")
    except Exception as e:
        log.warning(f"[DB] Record fill failed trade={trade_id} type={fill_type}: {e}")
    finally:
        conn.close()


def db_mark_tp1_recorded(trade_id: int):
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE {TRADES_TABLE} SET tp1_recorded=TRUE WHERE id=%s", (trade_id,)
            )
        conn.commit()
    except Exception as e:
        log.warning(f"[DB] Mark tp1_recorded failed trade={trade_id}: {e}")
    finally:
        conn.close()


def db_create_pending(symbol: str, side: str, setup: str,
                      confidence: float = 0.0,
                      strategy_policy_id: int | None = None,
                      strategy_policy_version: int | None = None) -> int:
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            try:
                cur.execute(f"""
                    INSERT INTO {TRADES_TABLE}
                      (symbol, side, status, entry_price, quantity, leverage,
                       stop_loss, take_profit, confidence, setup, horizon, account_type,
                       strategy_policy_id, strategy_policy_version)
                    VALUES (%s,%s,'PENDING', 0, 0, %s, 0, 0, %s,%s, 1, %s, %s, %s)
                    RETURNING id
                """, (
                    symbol, side, LEVERAGE, confidence, setup, ACCOUNT_TYPE,
                    strategy_policy_id, strategy_policy_version,
                ))
            except errors.UniqueViolation:
                conn.rollback()
                raise RuntimeError(f"[{symbol}] Duplicate PENDING — another instance already entering")
            tid = cur.fetchone()[0]
        conn.commit()
    finally:
        conn.close()
    return tid


def db_confirm_open(trade_id: int, entry_price: float, quantity: float,
                    stop_loss: float, take_profit: float, take_profit_2: float,
                    order_id: str, notes: str, opened_at_ms: int | None = None):
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            opened_at = (
                datetime.fromtimestamp(opened_at_ms / 1000, tz=UTC)
                if opened_at_ms else None
            )
            if opened_at:
                cur.execute(f"""
                    UPDATE {TRADES_TABLE} SET
                        status='OPEN', entry_price=%s, quantity=%s,
                        stop_loss=%s, take_profit=%s, take_profit_2=%s,
                        binance_order_id=%s, notes=%s, opened_at=%s
                    WHERE id=%s
                """, (entry_price, quantity, stop_loss, take_profit, take_profit_2,
                      order_id, notes[:200], opened_at, trade_id))
            else:
                cur.execute(f"""
                    UPDATE {TRADES_TABLE} SET
                        status='OPEN', entry_price=%s, quantity=%s,
                        stop_loss=%s, take_profit=%s, take_profit_2=%s,
                        binance_order_id=%s, notes=%s
                    WHERE id=%s
                """, (entry_price, quantity, stop_loss, take_profit, take_profit_2,
                      order_id, notes[:200], trade_id))
            cur.execute(f"SELECT side FROM {TRADES_TABLE} WHERE id=%s", (trade_id,))
            side_row = cur.fetchone()
            trade_side = side_row[0] if side_row else None
        conn.commit()
    finally:
        conn.close()
    if trade_side:
        db_record_fill(trade_id, "ENTRY", entry_price, quantity, order_id,
                       entry_price, trade_side, opened_at_ms)


def db_cancel_pending(trade_id: int):
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(f"DELETE FROM {TRADES_TABLE} WHERE id=%s AND status='PENDING'", (trade_id,))
        conn.commit()
    finally:
        conn.close()


def db_update_stop_loss(trade_id: int, new_sl: float):
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE {TRADES_TABLE} SET stop_loss=%s WHERE id=%s",
                (new_sl, trade_id),
            )
        conn.commit()
    finally:
        conn.close()


def db_close_trade(trade_id: int, exit_price: float, reason: str,
                   closed_at_ms: int | None = None,
                   close_order_id: str | None = None):
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT entry_price, quantity, side FROM {TRADES_TABLE} WHERE id=%s", (trade_id,)
            )
            row = cur.fetchone()
            if not row:
                return
            entry, qty, side = row
            if side == "BUY":
                pnl_pct = (exit_price - entry) / entry * LEVERAGE
            else:
                pnl_pct = (entry - exit_price) / entry * LEVERAGE
            margin   = qty * entry / LEVERAGE
            raw_pnl  = pnl_pct * margin
            fees     = (entry + exit_price) * qty * 0.0004
            pnl_usdt = raw_pnl - fees
            # Override with fills-based PnL if fills are recorded (more accurate for staged trades)
            cur.execute(
                f"SELECT SUM(pnl_usdt), COUNT(*) FROM {FILLS_TABLE} WHERE trade_id=%s",
                (trade_id,)
            )
            fills_row = cur.fetchone()
            if fills_row and fills_row[1] and fills_row[1] > 0 and fills_row[0] is not None:
                pnl_usdt = float(fills_row[0])
                pnl_pct  = pnl_usdt / margin if margin else pnl_pct
            closed_at = (
                datetime.fromtimestamp(closed_at_ms / 1000, tz=UTC)
                if closed_at_ms else None
            )
            if closed_at:
                cur.execute(f"""
                    UPDATE {TRADES_TABLE} SET
                        status='CLOSED', exit_price=%s, pnl_usdt=%s,
                        pnl_pct=%s, close_reason=%s, closed_at=%s, close_order_id=%s
                    WHERE id=%s
                """, (exit_price, round(pnl_usdt, 4), round(pnl_pct * 100, 4), reason,
                      closed_at, close_order_id, trade_id))
            else:
                cur.execute(f"""
                    UPDATE {TRADES_TABLE} SET
                        status='CLOSED', exit_price=%s, pnl_usdt=%s,
                        pnl_pct=%s, close_reason=%s, closed_at=NOW(), close_order_id=%s
                    WHERE id=%s
                """, (exit_price, round(pnl_usdt, 4), round(pnl_pct * 100, 4), reason,
                      close_order_id, trade_id))
        conn.commit()
    finally:
        conn.close()
    log.info(f"[DB] Trade {trade_id} closed — {reason} @ {exit_price} pnl={pnl_usdt:+.4f} USDT ({pnl_pct*100:.2f}% raw, fees={fees:.4f})")


def db_get_open_trades() -> list[dict]:
    retries = 2
    for attempt in range(retries + 1):
        conn = None
        try:
            conn = _get_conn()
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT id, symbol, side, entry_price, quantity,
                           stop_loss, take_profit, take_profit_2, binance_order_id, opened_at,
                           COALESCE(tp1_recorded, FALSE) AS tp1_recorded
                    FROM {TRADES_TABLE} WHERE status='OPEN' AND account_type=%s
                """, (ACCOUNT_TYPE,))
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, r)) for r in cur.fetchall()]
        except (OperationalError, InterfaceError) as e:
            if attempt < retries and _is_transient_db_error(e):
                wait_s = 0.25 * (attempt + 1)
                log.warning(
                    f"[DB] get_open_trades transient error ({e}); retrying in {wait_s:.2f}s "
                    f"({attempt + 1}/{retries})"
                )
                time.sleep(wait_s)
                continue
            log.warning(f"[DB] get_open_trades failed: {e}")
            return []
        except Exception as e:
            log.warning(f"[DB] get_open_trades failed: {e}")
            return []
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
    return []


def db_cleanup_stale_pending(max_age_seconds: int = 120):
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                DELETE FROM {TRADES_TABLE}
                WHERE status='PENDING'
                  AND account_type=%s
                  AND opened_at < NOW() - INTERVAL '%s seconds'
                RETURNING id, symbol
            """, (ACCOUNT_TYPE, max_age_seconds))
            rows = cur.fetchall()
            for tid, sym in rows:
                log.warning(f"[DB] Cleaned up stale PENDING id={tid} symbol={sym}")
        conn.commit()
    except Exception as e:
        log.warning(f"[DB] cleanup_stale_pending failed: {e}")
    finally:
        conn.close()


def _get_exit_fill_from_binance(client: UMFutures, symbol: str, trade_side: str,
                                after_ms: int | None = None) -> dict | None:
    try:
        closing_side = "SELL" if trade_side == "BUY" else "BUY"
        kwargs: dict = {"symbol": f"{symbol}USDT", "limit": 20 if after_ms else 10}
        if after_ms:
            kwargs["startTime"] = after_ms
        trades = client.get_account_trades(**kwargs)
        if not trades:
            return None

        all_closing = [
            t for t in trades
            if t.get("side") == closing_side
            and abs(float(t.get("qty", 0) or 0)) > 0
            and float(t.get("price", 0) or 0) > 0
        ]

        if after_ms:
            # Return all closing fills since trade opened, chronological order
            closing_fills = all_closing
        else:
            # Group only the most recent fill cluster (within 5s of each other)
            closing_fills = []
            for t in reversed(all_closing):
                if not closing_fills:
                    closing_fills.append(t)
                    continue
                first_time   = int(closing_fills[0].get("time", 0) or 0)
                current_time = int(t.get("time", 0) or 0)
                if first_time and current_time and abs(first_time - current_time) > 5000:
                    break
                closing_fills.append(t)

        if not closing_fills:
            return None

        total_qty = sum(abs(float(t.get("qty", 0) or 0)) for t in closing_fills)
        if total_qty <= 0:
            return None

        weighted_exit = sum(
            float(t.get("price", 0) or 0) * abs(float(t.get("qty", 0) or 0))
            for t in closing_fills
        ) / total_qty
        realized_pnl  = sum(float(t.get("realizedPnl", 0) or 0) for t in closing_fills)
        latest_time   = max(int(t.get("time", 0) or 0) for t in closing_fills)
        most_recent   = max(closing_fills, key=lambda t: int(t.get("time", 0) or 0))
        close_order_id = str(most_recent.get("orderId", ""))
        return {
            "exit_price":   weighted_exit,
            "qty":          total_qty,
            "realized_pnl": realized_pnl,
            "time":         latest_time,
            "fills":        len(closing_fills),
            "order_id":     close_order_id,
            "fills_list":   closing_fills,
        }
    except Exception as e:
        log.warning(f"[MONITOR] Could not fetch exit fills for {symbol}: {e}")
        return None


_ORPHAN_SL_PCT = 0.005
_ORPHAN_TP_PCT = 0.010
_POSITION_MISS_COUNTS: dict[int, int] = {}


def _recover_orphaned_positions(client: UMFutures, open_trades: list[dict]):
    tracked_symbols = {t["symbol"] for t in open_trades}
    try:
        all_positions = client.get_position_risk()
    except Exception as e:
        log.warning(f"[MONITOR] Could not fetch Binance positions: {e}")
        return

    for pos in all_positions:
        amt = float(pos.get("positionAmt", 0))
        if amt == 0:
            continue
        sym = pos["symbol"].replace("USDT", "")
        if sym not in SYMBOLS or sym in tracked_symbols:
            continue

        side  = "BUY" if amt > 0 else "SELL"
        entry = float(pos["entryPrice"])
        qty   = abs(amt)
        mark  = float(pos["markPrice"])

        if side == "BUY":
            sl = round(entry * (1 - _ORPHAN_SL_PCT), 2)
            tp = round(entry * (1 + _ORPHAN_TP_PCT), 2)
        else:
            sl = round(entry * (1 + _ORPHAN_SL_PCT), 2)
            tp = round(entry * (1 - _ORPHAN_TP_PCT), 2)

        log.warning(
            f"[MONITOR] Orphaned position: {sym} {side} qty={qty} "
            f"entry={entry} mark={mark} — recovering SL={sl} TP={tp}"
        )

        claimed_trade_id = None
        recovered_setup = "RECOVERED_ORPHAN"
        recovered_note = "Recovered Binance position without original setup context"
        conn = _get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(f"""
                    INSERT INTO {TRADES_TABLE}
                      (symbol, side, status, entry_price, quantity, leverage,
                       stop_loss, take_profit, confidence, setup, notes, horizon, account_type)
                    VALUES (%s,%s,'OPEN',%s,%s,%s,%s,%s, 0, %s, %s, 1, %s)
                    RETURNING id
                """, (
                    sym,
                    side,
                    entry,
                    qty,
                    LEVERAGE,
                    sl,
                    tp,
                    recovered_setup,
                    recovered_note,
                    ACCOUNT_TYPE,
                ))
                claimed_trade_id = cur.fetchone()[0]
            conn.commit()
            log.info(
                f"[MONITOR] Recovery record inserted for {sym} {side} @ {entry} "
                f"id={claimed_trade_id} setup={recovered_setup}"
            )
        except errors.UniqueViolation:
            conn.rollback()
            log.info(f"[MONITOR] Orphan {sym} already claimed by another monitor - skipping recovery insert")
            continue
        except Exception as e:
            conn.rollback()
            log.warning(f"[MONITOR] Failed to insert recovery record for {sym}: {e}")
            continue
        finally:
            conn.close()


def _monitor_loop(client: UMFutures):
    log.info("[MONITOR] Live monitor thread started (5s interval)")
    while True:
        try:
            db_cleanup_stale_pending()
            open_trades = db_get_open_trades()

            # --- Phase 1: software SL/TP backup ---
            for trade in list(open_trades):
                sl = trade.get("stop_loss") or 0
                tp = trade.get("take_profit") or 0
                tp2 = trade.get("take_profit_2") or 0
                # Staged exits (tp2 set and different from tp1) rely on exchange
                # reduceOnly orders for TP. Software backup only enforces SL to
                # avoid prematurely closing the remaining half at TP1 price.
                is_staged = bool(tp2 and abs(tp2 - tp) / max(tp, 1) > 0.0005)
                if not sl and not tp:
                    continue

                sym = trade["symbol"]
                mark = get_symbol_mark_price(client, sym)
                if mark is None:
                    continue

                trade_side = trade["side"]

                # --- Break-even: move SL to entry after the configured R-multiple is reached ---
                entry = trade.get("entry_price") or 0
                if entry and sl:
                    be_applied = (
                        (trade_side == "BUY"  and sl >= entry * 0.9999) or
                        (trade_side == "SELL" and sl <= entry * 1.0001)
                    )
                    if not be_applied:
                        risk = abs(entry - sl)
                        be_trigger_r = _get_break_even_trigger_r_mult(sym)
                        be_trigger = (
                            entry + risk * be_trigger_r
                            if trade_side == "BUY"
                            else entry - risk * be_trigger_r
                        )
                        be_reached = (
                            (trade_side == "BUY"  and mark >= be_trigger) or
                            (trade_side == "SELL" and mark <= be_trigger)
                        )
                        if be_reached:
                            log.info(
                                f"[MONITOR] {sym} break-even triggered "
                                f"(mark={mark:.4f} trigger={be_trigger:.4f} r_mult={be_trigger_r:.2f}) "
                                f"— moving SL to entry {entry}"
                            )
                            try:
                                _replace_stop_loss_algo(sym, trade_side, mark, entry)
                                log.info(f"[MONITOR] {sym} new break-even SL placed at entry {entry}")
                            except Exception as e:
                                log.warning(f"[MONITOR] {sym} break-even SL replace failed ({e}) — software monitor will enforce via DB")
                            db_update_stop_loss(trade["id"], entry)
                            trade["stop_loss"] = entry
                            sl = entry

                # --- TP1 detection: check if position halved for staged trades ---
                if is_staged and not trade.get("tp1_recorded"):
                    tp1_price = tp or 0
                    tp1_reached = (
                        (trade.get("side") == "BUY"  and tp1_price and mark >= tp1_price) or
                        (trade.get("side") == "SELL" and tp1_price and mark <= tp1_price)
                    )
                    if tp1_reached:
                        try:
                            pos_check = get_open_position(client, sym)
                            if pos_check is not None:
                                half_qty = trade["quantity"] / 2
                                if pos_check["amount"] <= half_qty * 1.1:
                                    db_mark_tp1_recorded(trade["id"])
                                    trade["tp1_recorded"] = True
                                    log.info(f"[MONITOR] {sym} TP1 confirmed: position={pos_check['amount']:.6f} (half={half_qty:.6f})")
                        except Exception as e:
                            log.warning(f"[MONITOR] {sym} TP1 position check failed: {e}")

                if sym == "BTC" and is_staged and trade.get("tp1_recorded"):
                    btc_trail_sl = _btc_tp1_trailing_stop(client, trade, mark)
                    if btc_trail_sl is not None:
                        try:
                            _replace_stop_loss_algo(sym, trade_side, mark, btc_trail_sl)
                            log.info(
                                f"[MONITOR] {sym} TP1 trail updated stop "
                                f"{trade.get('stop_loss')} → {btc_trail_sl}"
                            )
                        except Exception as e:
                            log.warning(
                                f"[MONITOR] {sym} TP1 trail stop replace failed ({e}) "
                                "— software monitor will enforce via DB"
                            )
                        db_update_stop_loss(trade["id"], btc_trail_sl)
                        trade["stop_loss"] = btc_trail_sl
                        sl = btc_trail_sl

                # For staged trades use tp2 as software fallback so we still
                # take profit even if the TP algo orders failed to place.
                tp_check = tp2 if is_staged and tp2 else tp
                hit = None
                if trade_side == "BUY":
                    if sl and mark <= sl:
                        hit = "stop_loss"
                    elif tp_check and mark >= tp_check:
                        hit = "take_profit"
                else:
                    if sl and mark >= sl:
                        hit = "stop_loss"
                    elif tp_check and mark <= tp_check:
                        hit = "take_profit"

                if hit:
                    close_side = "SELL" if trade_side == "BUY" else "BUY"
                    log.warning(f"[MONITOR] {sym} {hit} hit (mark={mark}) - closing position")
                    try:
                        live_pos = get_open_position(client, sym)
                        if live_pos is None:
                            log.info(f"[MONITOR] {sym} fallback close skipped - no live position remains")
                            continue
                        qty = live_pos["amount"]
                        if qty <= 0:
                            log.info(f"[MONITOR] {sym} fallback close skipped - live position size is zero")
                            continue
                        close_resp = client.new_order(
                            symbol=f"{sym}USDT",
                            side=close_side,
                            type="MARKET",
                            quantity=qty,
                            reduceOnly="true",
                        )
                        close_time_ms = int(
                            close_resp.get("transactTime") or close_resp.get("updateTime") or 0
                        ) or None
                        close_oid = str(close_resp.get("orderId", "")) or None
                        fill_price = float(close_resp.get("avgPrice") or 0)
                        if not fill_price and close_oid:
                            time.sleep(0.3)
                            try:
                                filled = client.query_order(symbol=f"{sym}USDT", orderId=close_oid)
                                fill_price = float(filled.get("avgPrice") or 0)
                            except Exception:
                                pass
                        fill_price = fill_price or mark
                        # Record individual fills (captures TP1 + this close for staged trades)
                        try:
                            opened_at_ms = int(trade["opened_at"].timestamp() * 1000) if trade.get("opened_at") else None
                            time.sleep(0.2)  # ensure fill registered in Binance history
                            all_exit = _get_exit_fill_from_binance(client, sym, trade_side, after_ms=opened_at_ms)
                            if all_exit:
                                entry_px = trade.get("entry_price") or 0
                                for f in all_exit.get("fills_list", []):
                                    f_price = float(f.get("price", 0) or 0)
                                    f_qty   = abs(float(f.get("qty", 0) or 0))
                                    f_time  = int(f.get("time", 0) or 0) or None
                                    f_oid   = str(f.get("orderId", "")) or None
                                    ftype   = _classify_fill_type(f_price, entry_px,
                                                                   trade.get("stop_loss") or 0,
                                                                   trade.get("take_profit") or 0,
                                                                   trade.get("take_profit_2") or 0,
                                                                   trade_side)
                                    db_record_fill(trade["id"], ftype, f_price, f_qty, f_oid,
                                                   entry_px, trade_side, f_time)
                        except Exception as e:
                            log.warning(f"[MONITOR] {sym} fills recording failed (Phase 1): {e}")
                        db_close_trade(trade["id"], fill_price, hit,
                                       closed_at_ms=close_time_ms, close_order_id=close_oid)
                        open_trades = [t for t in open_trades if t["id"] != trade["id"]]
                    except Exception as e:
                        log.warning(f"[MONITOR] Failed to close {sym} on {hit}: {e}")

            # --- Phase 2: recover orphaned positions ---
            _recover_orphaned_positions(client, open_trades)

            # --- Phase 3: close DB records when Binance position is gone ---
            for trade in open_trades:
                sym = trade["symbol"]
                trade_id = trade["id"]
                pos = get_open_position(client, sym)
                if pos is not None:
                    _POSITION_MISS_COUNTS.pop(trade_id, None)
                    continue

                miss_count = _POSITION_MISS_COUNTS.get(trade_id, 0) + 1
                _POSITION_MISS_COUNTS[trade_id] = miss_count
                if miss_count < 3:
                    log.info(
                        f"[MONITOR] {sym} position miss {miss_count}/3 - waiting before DB close"
                    )
                    continue

                opened_at_ms = int(trade["opened_at"].timestamp() * 1000) if trade.get("opened_at") else None
                exit_fill = _get_exit_fill_from_binance(client, sym, trade["side"], after_ms=opened_at_ms)
                exit_time_ms = None
                exit_order_id = None
                if exit_fill is not None:
                    exit_price = float(exit_fill["exit_price"])
                    exit_time_ms = exit_fill.get("time") or None
                    exit_order_id = exit_fill.get("order_id") or None
                    log.info(
                        f"[MONITOR] {sym} exit fills={exit_fill['fills']} "
                        f"qty={exit_fill['qty']:.6f} pnl={exit_fill['realized_pnl']:.4f} "
                        f"orderId={exit_order_id} time={exit_time_ms}"
                    )
                    # Record individual fills
                    entry_px = trade.get("entry_price") or 0
                    side_t   = trade["side"]
                    for f in exit_fill.get("fills_list", []):
                        f_price = float(f.get("price", 0) or 0)
                        f_qty   = abs(float(f.get("qty", 0) or 0))
                        f_time  = int(f.get("time", 0) or 0) or None
                        f_oid   = str(f.get("orderId", "")) or None
                        ftype   = _classify_fill_type(f_price, entry_px,
                                                       trade.get("stop_loss") or 0,
                                                       trade.get("take_profit") or 0,
                                                       trade.get("take_profit_2") or 0,
                                                       side_t)
                        db_record_fill(trade_id, ftype, f_price, f_qty, f_oid,
                                       entry_px, side_t, f_time)
                else:
                    exit_price = get_symbol_mark_price(client, sym)
                    if exit_price is None:
                        continue
                sl   = trade["stop_loss"]
                tp   = trade["take_profit"]
                side = trade["side"]
                if side == "BUY":
                    reason = ("take_profit" if exit_price >= tp
                              else "stop_loss" if exit_price <= sl
                              else "native_close")
                else:
                    reason = ("take_profit" if exit_price <= tp
                              else "stop_loss" if exit_price >= sl
                              else "native_close")
                log.info(f"[MONITOR] {sym} closed → reason={reason} exit={exit_price} orderId={exit_order_id}")
                db_close_trade(trade["id"], exit_price, reason,
                               closed_at_ms=exit_time_ms, close_order_id=exit_order_id)
                _POSITION_MISS_COUNTS.pop(trade_id, None)

        except Exception as e:
            log.warning(f"[MONITOR] Sync error: {e}")

        time.sleep(MONITOR_INTERVAL)


def _get_redis():
    url = REDIS_URL
    if not url:
        return None
    if url.startswith("redis://"):
        url = "rediss://" + url[8:]
    try:
        r = redis.from_url(url, decode_responses=True, socket_timeout=3)
        r.ping()
        return r
    except Exception:
        return None


def get_ml_predictions() -> dict:
    try:
        r = _get_redis()
        if r is None:
            return {}
        result = {}
        for sym in SYMBOLS:
            for horizon_token in ML_HORIZONS:
                val = r.get(f"prediction:{sym}:{horizon_token}")
                if val:
                    result[f"{sym}_{horizon_token}"] = json.loads(val)
        return result
    except Exception as e:
        log.warning(f"Redis ML read failed: {e}")
        return {}


def get_market_signals() -> dict:
    signals = {}
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT value FROM fear_greed_index ORDER BY date DESC LIMIT 1")
            row = cur.fetchone()
            if row:
                signals["fear_greed"] = int(row[0])
            for sym in SYMBOLS:
                cur.execute(
                    "SELECT rate_avg FROM funding_rates WHERE symbol=%s "
                    "ORDER BY date DESC LIMIT 1", (sym,)
                )
                row = cur.fetchone()
                if row:
                    signals[f"{sym.lower()}_funding"] = float(row[0])
    except Exception as e:
        log.warning(f"DB market signals failed: {e}")
    finally:
        conn.close()
    return signals


import pandas as pd


def fetch_ohlcv(client: UMFutures, symbol: str, interval: str,
                limit: int = 200) -> pd.DataFrame:
    raw = client.klines(symbol=f"{symbol}USDT", interval=interval, limit=limit)
    df = pd.DataFrame(raw, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col])
    df["timestamp"] = pd.to_numeric(df["timestamp"])
    return df.iloc[:-1].reset_index(drop=True)


def _compute_m15_atr(df_m15: pd.DataFrame) -> float:
    if df_m15 is None or len(df_m15) < 2:
        return 0.0
    prev_close = df_m15["close"].shift(1)
    tr = pd.concat(
        [
            (df_m15["high"] - df_m15["low"]).abs(),
            (df_m15["high"] - prev_close).abs(),
            (df_m15["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(span=14, adjust=False).mean().iloc[-1]
    return float(atr) if pd.notna(atr) else 0.0


def _btc_tp1_trailing_stop(client: UMFutures, trade: dict, mark: float) -> float | None:
    if trade.get("symbol") != "BTC" or not trade.get("tp1_recorded"):
        return None

    entry = float(trade.get("entry_price") or 0.0)
    current_sl = float(trade.get("stop_loss") or 0.0)
    if entry <= 0 or current_sl <= 0 or mark <= 0:
        return None

    try:
        df_m15 = fetch_ohlcv(client, "BTC", "15m", 40)
    except Exception as e:
        log.warning(f"[MONITOR] BTC trailing fetch failed: {e}")
        return None

    if len(df_m15) < BTC_TRAIL_LOOKBACK_BARS + 14:
        return None

    atr = _compute_m15_atr(df_m15)
    if atr <= 0:
        return None

    side = trade.get("side")
    if side == "BUY":
        structure = float(df_m15["low"].tail(BTC_TRAIL_LOOKBACK_BARS).min())
        candidate = max(entry, structure - atr * BTC_TRAIL_STOP_ATR_MULT)
        candidate = _round_price("BTC", candidate)
        max_safe_stop = mark * (1 - BTC_TRAIL_MARK_BUFFER_PCT)
        if candidate <= current_sl or candidate >= max_safe_stop:
            return None
        return candidate

    if side == "SELL":
        structure = float(df_m15["high"].tail(BTC_TRAIL_LOOKBACK_BARS).max())
        candidate = min(entry, structure + atr * BTC_TRAIL_STOP_ATR_MULT)
        candidate = _round_price("BTC", candidate)
        min_safe_stop = mark * (1 + BTC_TRAIL_MARK_BUFFER_PCT)
        if candidate >= current_sl or candidate <= min_safe_stop:
            return None
        return candidate

    return None


def _btc_macro_trend(df_4h: pd.DataFrame) -> str:
    """
    Returns 'BULL', 'BEAR', or 'NEUTRAL' based on BTC 4h EMA50.
      BULL:    close > EMA50 and EMA50 rising  → only block SHORTs on ETH/SOL
      BEAR:    close < EMA50 and EMA50 falling → only block LONGs  on ETH/SOL
      NEUTRAL: everything else                 → both directions allowed
    """
    if len(df_4h) < 55:
        return "NEUTRAL"
    close  = df_4h["close"].astype(float)
    ema50  = close.ewm(span=50, adjust=False).mean()
    last_close = float(close.iloc[-1])
    last_ema   = float(ema50.iloc[-1])
    slope      = float(ema50.iloc[-1]) - float(ema50.iloc[-3])  # 3-bar slope
    if last_close > last_ema and slope > 0:
        return "BULL"
    if last_close < last_ema and slope < 0:
        return "BEAR"
    return "NEUTRAL"


def get_account_balance(client: UMFutures) -> float:
    try:
        account = client.account()
        for asset in account.get("assets", []):
            if asset["asset"] == "USDT":
                wallet = float(asset["walletBalance"])
                unrealized = float(asset.get("unrealizedProfit", 0))
                return round(wallet + unrealized, 2)
    except Exception as e:
        log.warning(f"Balance fetch failed: {e}")
    return 0.0


def get_open_position(client: UMFutures, symbol: str) -> dict | None:
    try:
        risk = client.get_position_risk(symbol=f"{symbol}USDT")
        positions = risk if isinstance(risk, list) else [risk]
        for pos in positions:
            amt = float(pos.get("positionAmt", 0))
            if abs(amt) > 0:
                return {
                    "symbol":      symbol,
                    "side":        "BUY" if amt > 0 else "SELL",
                    "amount":      abs(amt),
                    "entry_price": float(pos.get("entryPrice", 0)),
                    "mark_price":  float(pos.get("markPrice", 0)),
                    "upnl":        float(pos.get("unRealizedProfit", 0)),
                }
    except Exception as e:
        log.warning(f"Position fetch failed for {symbol}: {e}")
    return None


def get_symbol_mark_price(client: UMFutures, symbol: str) -> float | None:
    try:
        risk = client.get_position_risk(symbol=f"{symbol}USDT")
        positions = risk if isinstance(risk, list) else [risk]
        for pos in positions:
            mark_price = float(pos.get("markPrice", 0) or 0)
            if mark_price > 0:
                return mark_price
    except Exception as e:
        log.warning(f"Mark price fetch failed for {symbol}: {e}")

    try:
        ticker = client.ticker_price(symbol=f"{symbol}USDT")
        return float(ticker["price"])
    except Exception:
        return None


def get_symbol_min_notional(client: UMFutures, symbol: str) -> float:
    cached = _SYMBOL_MIN_NOTIONAL_CACHE.get(symbol)
    if cached is not None:
        return cached

    sym_pair = f"{symbol}USDT"
    min_notional = DEFAULT_MIN_NOTIONAL
    try:
        info = client.exchange_info()
        for sym_info in info.get("symbols", []):
            if sym_info.get("symbol") != sym_pair:
                continue
            candidates: list[float] = []
            for filt in sym_info.get("filters", []):
                if filt.get("filterType") not in {"MIN_NOTIONAL", "NOTIONAL"}:
                    continue
                for key in ("notional", "minNotional"):
                    raw = filt.get(key)
                    try:
                        value = float(raw)
                    except (TypeError, ValueError):
                        continue
                    if value > 0:
                        candidates.append(value)
            if candidates:
                min_notional = max(candidates)
            break
    except Exception as e:
        log.warning(f"[{symbol}] exchange_info min notional lookup failed, using fallback {DEFAULT_MIN_NOTIONAL:.2f}: {e}")

    _SYMBOL_MIN_NOTIONAL_CACHE[symbol] = min_notional
    return min_notional


def calc_quantity(balance: float, entry: float, sl: float, symbol: str,
                  min_notional: float) -> float:
    risk_usdt     = balance * _get_position_risk_pct(symbol)
    sl_distance   = abs(entry - sl)
    if sl_distance == 0:
        return 0.0
    position_value = (risk_usdt / sl_distance) * entry
    max_fraction = _get_max_position_fraction(symbol)
    position_value = min(position_value, balance * max_fraction * LEVERAGE)
    # Binance rejects orders below its minimum order notional.
    if position_value < min_notional:
        return 0.0
    qty       = position_value / entry
    precision = QTY_PRECISION.get(symbol, 2)
    factor = 10 ** precision
    qty = math.floor(qty * factor) / factor
    if qty <= 0 or (qty * entry) < min_notional:
        return 0.0
    return qty


def min_balance_required_for_symbol(symbol: str, entry_price: float, min_notional: float) -> float:
    precision = QTY_PRECISION.get(symbol, 2)
    min_qty_step = 10 ** (-precision)
    min_notional_for_step = entry_price * min_qty_step
    max_fraction = _get_max_position_fraction(symbol)
    denom = max_fraction * LEVERAGE
    if denom <= 0:
        return float("inf")
    return max(min_notional, min_notional_for_step) / denom

def _signal_follows_trend(signal: str, context: dict) -> tuple[bool, str]:
    """Hard guard: selected signal must align with directional trend."""
    if signal not in ("BUY", "SELL"):
        return False, f"invalid signal {signal}"

    primary = context.get("primary_trend") or context.get("m15_trend", "")
    if primary == "UPTREND" and signal != "BUY":
        return False, f"M15={primary} requires BUY"
    if primary == "DOWNTREND" and signal != "SELL":
        return False, f"M15={primary} requires SELL"

    return True, "OK"


def execute_trade(client: UMFutures, symbol: str, decision: dict,
                  balance: float, context: dict, dry_run: bool = False) -> tuple[bool, dict[str, Any]]:
    execution_meta: dict[str, Any] = {"trade_id": None, "execution_ref": None, "pending_id": None}
    signal = decision.get("signal")
    if signal not in ("BUY", "SELL"):
        return False, execution_meta

    try:
        ai_sl  = float(decision.get("stop_loss", 0))
        ai_tp  = float(decision.get("take_profit", 0))
        reason = decision.get("reason", "")
        setup  = decision.get("analysis", {}).get("setup_identified", "?")
    except (TypeError, ValueError) as e:
        log_error(f"[{symbol}] Invalid SL/TP from selector", e)
        return False, execution_meta

    entry = float(decision.get("entry_price") or context.get("current_price", 0))
    if entry == 0:
        try:
            ticker = client.ticker_price(symbol=f"{symbol}USDT")
            entry  = float(ticker["price"])
        except Exception as e:
            log_error(f"[{symbol}] Ticker fetch failed", e)
            return False, execution_meta

    if signal == "BUY" and not (ai_tp > entry > ai_sl):
        log_gate_fail("DIRECTION", f"BUY needs TP({ai_tp}) > entry({entry}) > SL({ai_sl})", symbol)
        return False, execution_meta
    if signal == "SELL" and not (ai_sl > entry > ai_tp):
        log_gate_fail("DIRECTION", f"SELL needs SL({ai_sl}) > entry({entry}) > TP({ai_tp})", symbol)
        return False, execution_meta

    risk_pct = abs(entry - ai_sl) / entry * 100
    if risk_pct < SL_MIN_PCT * 100:
        log_gate_fail("SL_MIN", f"SL {risk_pct:.3f}% < min {SL_MIN_PCT*100:.2f}%", symbol)
        return False, execution_meta
    max_stop_pct = compute_max_stop_pct(setup, context, entry=entry)
    if risk_pct > max_stop_pct * 100:
        log_gate_fail("SL_MAX", f"SL {risk_pct:.3f}% > max {max_stop_pct*100:.2f}%", symbol)
        return False, execution_meta

    risk      = abs(entry - ai_sl)
    reward    = abs(ai_tp - entry)
    rr        = float(decision.get("planned_rr") or (reward / risk if risk > 0 else 0))
    is_setup_e = "setup_e" in setup.lower() if setup else False
    is_range_setup = any(tag in setup.lower() for tag in ("setup d", "setup_d")) if setup else False
    if is_setup_e:
        min_rr = SETUP_E_MIN_RR
    elif is_range_setup:
        min_rr = RANGE_MIN_RR
    else:
        min_rr = TAKE_PROFIT_MIN_RR
    if rr < min_rr:
        log_gate_fail("RR", f"R:R={rr:.2f} < {min_rr}", symbol)
        return False, execution_meta

    min_notional = get_symbol_min_notional(client, symbol)
    qty = calc_quantity(balance, entry, ai_sl, symbol, min_notional)
    if qty <= 0:
        min_balance = min_balance_required_for_symbol(symbol, entry, min_notional)
        log_gate_fail(
            "QTY",
            (
                f"Quantity too small or below min notional ${min_notional:.2f}. "
                f"balance={balance:.2f} requires>={min_balance:.2f} with leverage={LEVERAGE} "
                f"and cap={_get_max_position_fraction(symbol):.2f}"
            ),
            symbol,
        )
        return False, execution_meta

    # --- Staged exit: partial TP1 + TP2 closePosition to avoid residual dust ---
    tp1_price = float(decision.get("take_profit_1") or ai_tp)
    tp2_price = float(decision.get("take_profit_2") or ai_tp)
    price_prec = PRICE_PRECISION.get(symbol, 2)
    qty_prec_factor = 10 ** QTY_PRECISION.get(symbol, 2)
    tp1_qty = math.floor((qty / 2) * qty_prec_factor) / qty_prec_factor
    staged = (
        tp1_qty > 0
        and tp1_qty < qty
        and (tp1_qty * entry) >= min_notional
        and abs(tp2_price - tp1_price) / entry > 0.0005  # TPs must differ by >0.05%
    )

    log.info(
        f"  [EXEC] {signal} {symbol} | entry≈{entry} SL={ai_sl} "
        f"TP1={tp1_price} TP2={tp2_price} qty={qty} tp1_qty={tp1_qty} staged={staged} "
        f"R:R={rr:.2f} | dry_run={dry_run}"
    )

    if dry_run:
        log_trade_open(symbol, signal, entry, ai_sl, tp1_price if staged else ai_tp,
                       rr, setup, reason, trade_id="DRY_RUN", context=context)
        execution_meta["execution_ref"] = "DRY_RUN"
        return True, execution_meta

    side       = "BUY" if signal == "BUY" else "SELL"
    close_side = "SELL" if signal == "BUY" else "BUY"
    sym_pair   = f"{symbol}USDT"
    confidence = context.get("ml_confidence", 0.0)

    # --- Step 1: reserve DB slot ---
    pending_id = None
    try:
        policy_id = _ACTIVE_STRATEGY_POLICY.id if _ACTIVE_STRATEGY_POLICY else None
        policy_version = _ACTIVE_STRATEGY_POLICY.version if _ACTIVE_STRATEGY_POLICY else None
        pending_id = db_create_pending(
            symbol,
            signal,
            setup,
            confidence,
            strategy_policy_id=policy_id,
            strategy_policy_version=policy_version,
        )
        execution_meta["pending_id"] = pending_id
        log.info(f"  [EXEC] PENDING record created id={pending_id}")
    except Exception as e:
        log_error(f"[{symbol}] Could not create PENDING record — aborting", e)
        return False, execution_meta

    try:
        client.change_leverage(symbol=sym_pair, leverage=LEVERAGE)

        # --- Step 2: entry order ---
        order    = client.new_order(symbol=sym_pair, side=side, type="MARKET", quantity=qty)
        order_id = str(order.get("orderId", ""))
        fill_time_ms = int(order.get("transactTime") or order.get("updateTime") or 0)
        actual_price = float(order.get("avgPrice") or 0)
        if not actual_price:
            # avgPrice can be "0" in the immediate response — query for the real fill
            time.sleep(0.3)
            try:
                filled = client.query_order(symbol=sym_pair, orderId=order_id)
                actual_price = float(filled.get("avgPrice") or 0) or entry
                if not fill_time_ms:
                    fill_time_ms = int(filled.get("updateTime") or 0)
            except Exception:
                actual_price = entry
        log.info(f"  [EXEC] Entry filled @ {actual_price} orderId={order_id} time={fill_time_ms}")

        time.sleep(0.5)

        # --- Step 3: SL via Algo Order API (closePosition=true covers full remaining qty) ---
        try:
            resp = _algo_order(sym_pair, close_side, "STOP_MARKET", ai_sl)
            log.info(f"  [EXEC] SL algo order placed @ {ai_sl} id={resp.get('algoId','?')}")
        except Exception as e:
            log.warning(f"  [EXEC] SL algo order failed ({e}) — software monitor will enforce")

        # --- Step 4: TP orders via Algo Order API ---
        if staged:
            # TP1 — close a partial quantity at nearer target
            try:
                resp = _algo_order(sym_pair, close_side, "TAKE_PROFIT_MARKET", tp1_price, quantity=tp1_qty)
                log.info(f"  [EXEC] TP1 algo order placed @ {tp1_price} qty={tp1_qty} id={resp.get('algoId','?')}")
            except Exception as e:
                log.warning(f"  [EXEC] TP1 order failed ({e}) — software monitor will enforce")
            # TP2 — close whatever remains at farther target (prevents rounding residue)
            try:
                resp = _algo_order(sym_pair, close_side, "TAKE_PROFIT_MARKET", tp2_price)
                log.info(f"  [EXEC] TP2 algo order placed @ {tp2_price} closePosition=true id={resp.get('algoId','?')}")
            except Exception as e:
                log.warning(f"  [EXEC] TP2 order failed ({e}) — software monitor will enforce")
        else:
            # Fallback: single TP via Algo Order (closePosition=true — position too small to split)
            try:
                resp = _algo_order(sym_pair, close_side, "TAKE_PROFIT_MARKET", ai_tp)
                log.info(f"  [EXEC] TP algo order placed @ {ai_tp} id={resp.get('algoId','?')}")
            except Exception as e:
                log.warning(f"  [EXEC] TP algo order failed ({e}) — software monitor will enforce")

        # --- Step 5: confirm OPEN ---
        db_tp1  = tp1_price if staged else ai_tp
        db_tp2  = tp2_price if staged else ai_tp
        db_confirm_open(pending_id, actual_price, qty, ai_sl, db_tp1, db_tp2, order_id, reason,
                        opened_at_ms=fill_time_ms or None)
        log_trade_open(symbol, signal, actual_price, ai_sl, db_tp1, rr,
                       setup, reason, trade_id=order_id, context=context)
        execution_meta["trade_id"] = pending_id
        execution_meta["execution_ref"] = order_id
        return True, execution_meta

    except Exception as e:
        log_error(f"[{symbol}] Order execution failed", e)
        db_cancel_pending(pending_id)
        return False, execution_meta


def run_symbol_cycle(client: UMFutures, symbol: str,
                     market_signals: dict,
                     balance: float, dry_run: bool = False):

    log.info(f"\n{'─'*50}")
    log.info(f"  CYCLE  {symbol}  {datetime.now(VN_TZ).strftime('%H:%M:%S ICT')}")

    pos = get_open_position(client, symbol)
    if pos:
        upnl_pct = pos["upnl"] / (pos["entry_price"] * pos["amount"]) * 100
        log_skip(
            "OPEN_POSITION",
            (
                f"{symbol} has open {pos['side']} position | "
                f"entry={pos['entry_price']} mark={pos['mark_price']} "
                f"uPnL={upnl_pct:+.2f}%"
            ),
        )
        return

    try:
        df_h1  = fetch_ohlcv(client, symbol, "1h",  200)
        df_m15 = fetch_ohlcv(client, symbol, "15m", 100)
        df_m5  = fetch_ohlcv(client, symbol, "5m",  100)
        df_btc_h1  = fetch_ohlcv(client, "BTC", "1h",  100) if symbol != "BTC" else df_h1
        df_btc_m15 = fetch_ohlcv(client, "BTC", "15m", 100) if symbol != "BTC" else df_m15
        df_btc_m5  = fetch_ohlcv(client, "BTC", "5m",  100) if symbol != "BTC" else df_m5
        df_btc_4h  = fetch_ohlcv(client, "BTC", "4h",  100)
    except Exception as e:
        log_error(f"[{symbol}] OHLCV fetch failed", e)
        return

    ctx = compute_indicators(df_h1, df_m15, df_m5)
    ctx["symbol"] = f"{symbol}/USDT"

    if symbol != "BTC":
        btc_ctx        = compute_indicators(df_btc_h1, df_btc_m15, df_btc_m5)
        btc_trend      = btc_ctx["m15_trend"]
        btc_macro      = _btc_macro_trend(df_btc_4h)
        ctx["btc_trend"]       = btc_trend
        ctx["btc_macro_trend"] = btc_macro
    else:
        ctx["btc_trend"]       = ctx["m15_trend"]
        ctx["btc_macro_trend"] = _btc_macro_trend(df_btc_4h)

    ctx["sr"] = find_sr_levels(df_h1, ctx["current_price"], df_m15)

    score, score_details = compute_score(ctx)
    ctx["score"]         = score
    ctx["score_details"] = score_details

    log_cycle_start(symbol, ctx["market_mode"], score)
    log.info(f"  H1={ctx['h1_trend']} M15={ctx['m15_trend']} BTC_M15={ctx.get('btc_trend','?')} BTC_4H={ctx.get('btc_macro_trend','?')}")
    log.info(f"  S/R  R={ctx['sr']['resistance']} S={ctx['sr']['support']}")
    log.info(
        "  PrevBox "
        f"state={ctx['sr'].get('breakout_state', 'INSIDE_PREV_BOX')} "
        f"R={ctx['sr'].get('prev_resistance')} S={ctx['sr'].get('prev_support')} "
        f"confirm={ctx['sr'].get('breakout_confirmed', False)} "
        f"bars_since={ctx['sr'].get('bars_since_breakout')}"
    )
    log.info(f"  Score {score}/5: {', '.join(score_details) or 'none'}")
    log.info(f"  RSI={ctx['rsi']:.1f} ATR={ctx['atr_m15']:.4f} ADX={ctx.get('adx_m15', 0):.1f} EMA_gap={ctx.get('m15_gap', 0):.3f}%")

    # ML is experimental and intentionally disabled for live trading.
    # Keep context keys explicit so logs and prompts remain deterministic.
    ml_dir = ""
    ml_conf = 0.0
    trend_dir = ""
    trend_conf = 0.0
    trend_pred = {}
    fear_greed = market_signals.get("fear_greed")
    funding    = market_signals.get(f"{symbol.lower()}_funding", 0)

    ctx["ml_direction"]  = ml_dir
    ctx["ml_confidence"] = ml_conf
    ctx["ml_entry_horizon"] = "disabled"
    ctx["ml_trend_horizon"] = "disabled"
    ctx["ml_trend_direction"] = trend_dir
    ctx["ml_trend_confidence"] = trend_conf
    ctx["ml_conflict_resolved"] = False
    ctx["ml_conflict_min_rr"] = 0.0
    ctx["fear_greed"]    = fear_greed
    ctx["funding_rate"]  = funding
    ctx["llm_exec_tf"]   = "15m"

    macro_ok, macro_reason, allowed_dir = check_macro_bias(
        ml_dir, ml_conf, fear_greed, funding, symbol
    )
    ctx["allowed_direction"] = allowed_dir

    if not macro_ok:
        log_gate_fail("MACRO", macro_reason, symbol, ctx)
        return
    trend_str = " trend=disabled(experimental)"
    log_gate_pass(
        "MACRO",
        f"allowed={allowed_dir} ML=disabled(experimental){trend_str} F&G={fear_greed}",
    )

    if symbol != "BTC":
        btc_macro = ctx.get("btc_macro_trend", "NEUTRAL")
        primary   = ctx.get("primary_trend") or ctx["m15_trend"]
        if btc_macro == "BEAR" and primary == "UPTREND":
            log_gate_fail(
                "BTC_MACRO",
                f"BTC 4h strongly BEAR (below EMA50, slope down) — blocking LONG on {symbol}",
                symbol, ctx,
            )
            return
        if btc_macro == "BULL" and primary == "DOWNTREND":
            log_gate_fail(
                "BTC_MACRO",
                f"BTC 4h strongly BULL (above EMA50, slope up) — blocking SHORT on {symbol}",
                symbol, ctx,
            )
            return
        log_gate_pass("BTC_MACRO", f"BTC 4h macro={btc_macro} | {symbol} trend={primary} — no conflict")

    tech_ok, tech_reason = check_technical_gates(ctx)
    if not tech_ok:
        log_gate_fail("TECHNICAL", tech_reason, symbol, ctx)
        return
    log_gate_pass("TECHNICAL", tech_reason)

    # --- Choppy/ranging market gate ---
    adx  = ctx.get("adx_m15", 30.0)
    gap  = ctx.get("m15_gap", 1.0)
    if adx < 22 and gap < 0.3:
        log_gate_fail(
            "CHOPPY",
            f"ADX={adx:.1f}<22 and EMA_gap={gap:.3f}%<0.3% — no trend strength, skipping",
            symbol, ctx,
        )
        return
    log_gate_pass("CHOPPY", f"ADX={adx:.1f} EMA_gap={gap:.3f}%")

    if ctx["is_range"]:
        ctx["range_bias"] = get_range_bias(ctx)
        log.info(f"  Range bias: {ctx['range_bias']}")
        if ctx["range_bias"] == "MIDDLE":
            log_gate_fail("RANGE_POSITION", "price in middle of range — no edge", symbol, ctx)
            return
        rsi = ctx["rsi"]
        if ctx["range_bias"] == "NEAR_SUPPORT" and rsi > 55:
            log_gate_fail("RANGE_POSITION", f"near support but RSI={rsi:.1f} — sellers not exhausted", symbol, ctx)
            return
        if ctx["range_bias"] == "NEAR_RESISTANCE" and rsi < 45:
            log_gate_fail("RANGE_POSITION", f"near resistance but RSI={rsi:.1f} — buyers not exhausted", symbol, ctx)
            return

    log.info("  All gates passed → evaluating deterministic setup selector...")
    decision = build_rule_based_signal_decision(ctx, df_m15)
    signal = decision.get("signal", "WAIT")
    analysis = decision.get("analysis", {}) if isinstance(decision, dict) else {}
    setup_name = analysis.get("setup_identified", "?")
    log.info(f"  [RULE] setup={setup_name} signal={signal}")
    log.info(f"  [RULE] EMA  : {analysis.get('ema_check', '')}")
    log.info(f"  [RULE] PA   : {analysis.get('price_action', '')}")
    log.info(f"  [RULE] VOL  : {analysis.get('volume_check', '')}")
    log.info(f"  [RULE] R:R  : {analysis.get('rr_check', '')}")
    log.info(f"  [RULE] Match: {analysis.get('pattern_match', '')}")
    log.info(f"  [RULE] Reason: {decision.get('reason', '')}")

    if signal == "WAIT":
        db_record_opportunity(
            symbol=symbol,
            signal="WAIT",
            setup=setup_name,
            status="RULE_WAIT",
            status_reason=decision.get("reason", ""),
            context=ctx,
            decision=_decision_for_snapshot(decision),
            preview_plan_payload={"status": "NO_SIGNAL", "reason": decision.get("reason", "")},
        )
        log.info(f"  [RULE] WAIT — {decision.get('reason','')}")
        log_skip("RULE_WAIT", decision.get("reason", ""), ctx, decision)
        log_cycle_summary(symbol, "WAIT", False, balance, ctx, decision)
        return

    ai_ok, ai_reason = validate_ai_trade_decision(decision, ctx, df_m15)
    if not ai_ok:
        db_record_opportunity(
            symbol=symbol,
            signal=signal,
            setup=setup_name,
            status="RULE_INVALID",
            status_reason=ai_reason,
            context=ctx,
            decision=_decision_for_snapshot(decision),
            preview_plan_payload={"status": "RULE_INVALID", "reason": ai_reason},
        )
        log_gate_fail("RULE_VALIDATION", ai_reason, symbol, ctx)
        log_skip("RULE_INVALID", ai_reason, ctx, decision)
        log_cycle_summary(symbol, "WAIT", False, balance, ctx, decision)
        return

    setup_name = decision.get("analysis", {}).get("setup_identified", "")
    preview_plan, preview_plan_reason = build_trade_plan(signal, setup_name, ctx, df_m15)
    preview_plan_payload = _build_preview_plan_payload(preview_plan, preview_plan_reason)
    symbol_policy_enabled = symbol_enabled_in_policy(
        _ACTIVE_STRATEGY_POLICY.policy_json if _ACTIVE_STRATEGY_POLICY else {},
        symbol,
    )
    setup_policy_enabled = setup_enabled_in_policy(
        _ACTIVE_STRATEGY_POLICY.policy_json if _ACTIVE_STRATEGY_POLICY else {},
        setup_name,
    )
    if not symbol_policy_enabled:
        policy_reason = f"{symbol} disabled by active strategy policy"
        db_record_opportunity(
            symbol=symbol,
            signal=signal,
            setup=setup_name,
            status="POLICY_BLOCKED",
            status_reason=policy_reason,
            context=ctx,
            decision=_decision_for_snapshot(decision, preview_plan),
            preview_plan_payload=preview_plan_payload,
        )
        log_gate_fail("POLICY_SYMBOL", policy_reason, symbol, ctx)
        log_skip("POLICY_SYMBOL", policy_reason, ctx, decision)
        log_cycle_summary(symbol, "WAIT", False, balance, ctx, decision)
        return
    if not setup_policy_enabled:
        policy_reason = f"{setup_name or 'unknown setup'} disabled by active strategy policy"
        db_record_opportunity(
            symbol=symbol,
            signal=signal,
            setup=setup_name,
            status="POLICY_BLOCKED",
            status_reason=policy_reason,
            context=ctx,
            decision=_decision_for_snapshot(decision, preview_plan),
            preview_plan_payload=preview_plan_payload,
        )
        log_gate_fail("POLICY_SETUP", policy_reason, symbol, ctx)
        log_skip("POLICY_SETUP", policy_reason, ctx, decision)
        log_cycle_summary(symbol, "WAIT", False, balance, ctx, decision)
        return
    if not preview_plan:
        db_record_opportunity(
            symbol=symbol,
            signal=signal,
            setup=setup_name,
            status="PLAN_REJECTED",
            status_reason=preview_plan_reason,
            context=ctx,
            decision=_decision_for_snapshot(decision),
            preview_plan_payload=preview_plan_payload,
        )
        log_gate_fail("TRADE_PLAN", preview_plan_reason, symbol, ctx)
        log_skip("TRADE_PLAN", preview_plan_reason, ctx, decision)
        log_cycle_summary(symbol, "WAIT", False, balance, ctx, decision)
        return

    # Refresh price before final plan so SL/TP/RR are calculated from actual entry price.
    try:
        ticker = client.ticker_price(symbol=f"{symbol}USDT")
        fresh_price = float(ticker["price"])
        if fresh_price and fresh_price > 0:
            log.info(f"  [PLAN] Price refresh: {ctx['current_price']} → {fresh_price}")
            ctx["current_price"] = fresh_price
    except Exception as e:
        log.warning(f"  [PLAN] Price refresh failed ({e}) — using original price")
    # Keep planning on the same 15m execution frame used for setup detection.
    plan, plan_reason = build_trade_plan(signal, setup_name, ctx, df_m15)
    if not plan:
        db_record_opportunity(
            symbol=symbol,
            signal=signal,
            setup=setup_name,
            status="PLAN_REFRESH_REJECTED",
            status_reason=plan_reason,
            context=ctx,
            decision=_decision_for_snapshot(decision),
            preview_plan_payload=preview_plan_payload,
        )
        log_gate_fail("TRADE_PLAN", plan_reason, symbol, ctx)
        log_skip("TRADE_PLAN", plan_reason, ctx, decision)
        log_cycle_summary(symbol, "WAIT", False, balance, ctx, decision)
        return

    decision["entry_price"] = plan["entry_price"]
    decision["stop_loss"] = plan["stop_loss"]
    decision["take_profit"] = plan["take_profit"]
    decision["take_profit_1"] = plan["take_profit_1"]
    decision["take_profit_2"] = plan["take_profit_2"]
    decision["target_mode"] = plan["target_mode"]
    decision["planned_rr"] = plan["rr"]
    decision.setdefault("analysis", {})
    decision["analysis"]["rr_check"] = (
        f"Python planned Entry={plan['entry_price']} SL={plan['stop_loss']} "
        f"TP={plan['take_profit']} ({plan['target_mode']}, TP1={plan['take_profit_1']}, TP2={plan['take_profit_2']}) "
        f"net R:R={plan['rr']:.2f} gross R:R={plan['gross_rr']:.2f}. Pass."
    )
    log.info(
        f"  [PLAN] entry={plan['entry_price']} SL={plan['stop_loss']} TP={plan['take_profit']} "
        f"({plan['target_mode']}; TP1={plan['take_profit_1']} TP2={plan['take_profit_2']}) "
        f"netR:R={plan['rr']:.2f} grossR:R={plan['gross_rr']:.2f} | {plan_reason}"
    )

    # Guard: skip if Binance already has an open position for this symbol
    if not dry_run and get_open_position(client, symbol) is not None:
        db_record_opportunity(
            symbol=symbol,
            signal=signal,
            setup=setup_name,
            status="BINANCE_POSITION_EXISTS",
            status_reason=f"{symbol} already has an open Binance position",
            context=ctx,
            decision=_decision_for_snapshot(decision, plan),
            preview_plan_payload=_build_preview_plan_payload(plan, plan_reason),
        )
        log.info(f"  [{symbol}] Binance position already open — skipping to avoid double entry")
        return

    executed, execution_meta = execute_trade(client, symbol, decision, balance, ctx, dry_run=dry_run)
    db_record_opportunity(
        symbol=symbol,
        signal=signal,
        setup=setup_name,
        status="DRY_RUN_EXECUTED" if dry_run and executed else "EXECUTED" if executed else "EXECUTION_FAILED",
        status_reason="trade executed" if executed else "trade execution failed",
        context=ctx,
        decision=_decision_for_snapshot(decision, plan),
        preview_plan_payload=_build_preview_plan_payload(plan, plan_reason),
        linked_trade_id=execution_meta.get("trade_id"),
        execution_ref=execution_meta.get("execution_ref"),
    )
    log_cycle_summary(symbol, signal, executed, balance, ctx, decision)


def run_once(dry_run: bool = False):
    log.info("=" * 60)
    log.info(f"ENGINE LIVE  {'DRY RUN' if dry_run else 'LIVE'}  {datetime.now(VN_TZ).isoformat()}")
    log.info("=" * 60)

    policy, applied_overrides, blocked_risk_overrides, guard_reason = _refresh_runtime_strategy_policy()
    client         = get_client()
    market_signals = get_market_signals()
    balance        = get_account_balance(client)

    log.info(f"Balance: {balance:.2f} USDT")
    if policy:
        log.info(
            f"Strategy policy: {policy.policy_name} v{policy.version} "
            f"(id={policy.id}, engine={policy.engine_name}, source={policy.source})"
        )
        if applied_overrides:
            summary = " ".join(f"{k}={v}" for k, v in sorted(applied_overrides.items()))
            log.info(f"Policy overrides: {summary}")
        log.info(f"Policy risk-governor: {guard_reason}")
        if blocked_risk_overrides:
            log.warning(
                "Blocked risk-increasing overrides without clear signal: "
                + " | ".join(blocked_risk_overrides)
            )
    else:
        log.info("Strategy policy: none active — using .env defaults")
    ml_status = "enabled (experimental)" if ML_EXPERIMENTAL_ENABLED else "disabled (experimental)"
    log.info(f"ML mode: {ml_status}")
    log.info(
        "Filter mode: "
        f"TECH_ALLOW_SIDEWAY={os.getenv('TECH_ALLOW_SIDEWAY', '0')} "
        f"TECH_SCORE_THRESHOLD={os.getenv('TECH_SCORE_THRESHOLD', '3')} "
        f"TECH_SCORE_THRESHOLD_RANGE={os.getenv('TECH_SCORE_THRESHOLD_RANGE', os.getenv('TECH_SCORE_THRESHOLD', '3'))} "
        f"SETUP_C_RELAXED={os.getenv('SETUP_C_RELAXED', '0')} "
        f"SL_ATR_MULTIPLIER={SL_ATR_MULTIPLIER:.2f} "
        f"SL_SR_BUFFER_ATR_MULT={SL_SR_BUFFER_ATR_MULT:.2f} "
        f"TRADE_MIN_RR={TAKE_PROFIT_MIN_RR:.2f} "
        f"TRADE_MIN_RR_RANGE={RANGE_MIN_RR:.2f} "
        f"TP_EXTENSION_ATR_MULT={TP_EXTENSION_ATR_MULT:.2f} "
        f"SETUP_E_MIN_RR={SETUP_E_MIN_RR:.2f} "
        f"BTC_POSITION_RISK_PCT={BTC_POSITION_RISK_PCT:.4f} "
        f"BTC_MAX_POSITION_FRACTION={BTC_MAX_POSITION_FRACTION:.2f} "
        f"BTC_BREAK_EVEN_TRIGGER_R_MULT={BTC_BREAK_EVEN_TRIGGER_R_MULT:.2f} "
        f"BTC_HIGH_ATR_PCT={os.getenv('BTC_HIGH_ATR_PCT', '1.1')} "
        f"BTC_HIGH_ATR_MIN_ADX={os.getenv('BTC_HIGH_ATR_MIN_ADX', '25')} "
        f"BTC_RANGE_DISABLE_ATR_PCT={os.getenv('BTC_RANGE_DISABLE_ATR_PCT', '0.9')}"
    )
    funding_str = "  ".join(
        f"{s} funding={market_signals.get(f'{s.lower()}_funding', 0):+.4f}%"
        for s in SYMBOLS
    )
    log.info(f"Fear&Greed={market_signals.get('fear_greed','?')}  {funding_str}")

    latest_balance = balance
    for symbol in SYMBOLS:
        try:
            latest_balance = get_account_balance(client)
            run_symbol_cycle(client, symbol, market_signals, latest_balance, dry_run)
        except Exception as e:
            log_error(f"[{symbol}] Unhandled cycle error", e)


def run_loop(dry_run: bool = False):
    log.info("Engine LIVE started — 5-min cycle loop")

    try:
        db_ensure_trades_table()
    except Exception as e:
        log.warning(f"DB table init failed: {e}")

    client_monitor = get_client()
    t = threading.Thread(target=_monitor_loop, args=(client_monitor,), daemon=True)
    t.start()

    from trade_logger import _tg_send
    _tg_send("🟢 *Live engine started* — monitoring BTC / ETH / SOL")

    while True:
        try:
            run_once(dry_run=dry_run)
        except Exception as e:
            log_error("Loop-level error", e)
            _tg_send(f"⚠️ *Live engine error*: {e}")
        log.info(f"  Sleeping {CYCLE_INTERVAL}s until next cycle...")
        time.sleep(CYCLE_INTERVAL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Trading Engine — BTC/ETH/SOL")
    parser.add_argument("--loop",    action="store_true")
    parser.add_argument("--once",    action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not API_KEY or not API_SECRET:
        print("ERROR: BINANCE_LIVE_API_KEY / BINANCE_LIVE_SECRET_KEY not set in .env")
        exit(1)

    if args.loop:
        run_loop(dry_run=args.dry_run)
    else:
        run_once(dry_run=args.dry_run)
