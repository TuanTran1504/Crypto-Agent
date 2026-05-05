"""
run_policy_review.py — guarded strategy policy reviewer.

Flow:
  1) Evaluate deterministic maturity gate (hours + sample size + cooldown).
  2) If HOLD -> persist run and exit (no LLM call).
  3) If ALLOW_REVIEW -> summarize recent trade performance and ask LLM for a JSON patch.
  4) Validate patch with strict bounds and conservative risk rules.
  5) Optionally auto-activate validated policy (hot-reload picked up by engine on next cycle).
  6) Persist review run details to policy_review_runs.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import psycopg2
from dotenv import load_dotenv
from openai import OpenAI
from psycopg2.extras import Json, RealDictCursor

from policy_review_guard import ReviewGuardConfig, evaluate_policy_review_guard


UTC = timezone.utc
ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / ".env")

log = logging.getLogger(__name__)


def _env_bool(value: str | bool | int | None, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _clean_optional_text(value: Any, max_len: int = 500) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    return text[:max_len]


def _resolve_force_review(force_review: bool | None = None) -> bool:
    if force_review is not None:
        return bool(force_review)
    return _env_bool(os.getenv("POLICY_REVIEW_FORCE_REVIEW", "0"), default=False)


def _resolve_force_review_reason(force_review_reason: str | None = None) -> str | None:
    if force_review_reason is not None:
        return _clean_optional_text(force_review_reason)
    return _clean_optional_text(os.getenv("POLICY_REVIEW_FORCE_REVIEW_REASON", ""))


def _resolve_auto_apply_override(auto_apply_override: bool | None = None) -> bool | None:
    if auto_apply_override is not None:
        return bool(auto_apply_override)
    raw = os.getenv("POLICY_REVIEW_AUTO_APPLY_OVERRIDE")
    if raw is None:
        return None
    return _env_bool(raw, default=False)


def _build_manual_override_payload(
    *,
    force_review: bool,
    force_review_reason: str | None,
    auto_apply_override: bool | None,
    policy_review_enabled: bool,
    guard_decision: str,
) -> dict[str, Any]:
    guard_decision_text = str(guard_decision or "").strip().upper() or "UNKNOWN"
    return {
        "force_review": bool(force_review),
        "reason": force_review_reason if force_review else None,
        "auto_apply_override": bool(auto_apply_override) if auto_apply_override is not None else None,
        "bypassed_policy_review_enabled": bool(force_review and not policy_review_enabled),
        "bypassed_guard_gate": bool(force_review and guard_decision_text != "ALLOW_REVIEW"),
    }


def _attach_manual_override(result: dict[str, Any], manual_override: dict[str, Any]) -> dict[str, Any]:
    if not manual_override.get("force_review"):
        return result
    enriched = dict(result)
    enriched["manual_override"] = manual_override
    return enriched


def _json_default(value: Any):
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC).isoformat()
    return str(value)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, default=_json_default)


def _to_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _utc_now(now_dt: datetime | None = None) -> datetime:
    return _to_utc(now_dt) or datetime.now(UTC)


def _safe_table_name(name: str, fallback: str = "trades_live") -> str:
    text = str(name or "").strip()
    if not text:
        return fallback
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", text):
        return text
    return fallback


def _has_column(conn, table_name: str, column_name: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = %s
              AND column_name = %s
            LIMIT 1
            """,
            (table_name, column_name),
        )
        return cur.fetchone() is not None


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _get_path(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _set_path(data: dict[str, Any], path: tuple[str, ...], value: Any):
    cur = data
    for key in path[:-1]:
        child = cur.get(key)
        if not isinstance(child, dict):
            child = {}
            cur[key] = child
        cur = child
    cur[path[-1]] = value


def _del_path(data: dict[str, Any], path: tuple[str, ...]):
    if not path:
        return
    cur: Any = data
    for key in path[:-1]:
        if not isinstance(cur, dict) or key not in cur:
            return
        cur = cur[key]
    if isinstance(cur, dict):
        cur.pop(path[-1], None)


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(_json_dumps(base))
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _flatten_patch(data: Any, prefix: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], Any]]:
    rows: list[tuple[tuple[str, ...], Any]] = []
    if isinstance(data, dict):
        for k, v in data.items():
            rows.extend(_flatten_patch(v, prefix + (str(k),)))
        return rows
    rows.append((prefix, data))
    return rows


@dataclass(frozen=True)
class PolicyKeySpec:
    kind: str
    min_value: float | int | None = None
    max_value: float | int | None = None


ALLOWED_POLICY_KEYS: dict[tuple[str, ...], PolicyKeySpec] = {
    ("global", "fg_extreme_block"): PolicyKeySpec("bool"),
    ("global", "fg_extreme_fear_threshold"): PolicyKeySpec("int", 0, 100),
    ("global", "fg_extreme_greed_threshold"): PolicyKeySpec("int", 0, 100),
    ("global", "tech_allow_sideway"): PolicyKeySpec("bool"),
    ("global", "tech_score_threshold"): PolicyKeySpec("int", 1, 5),
    ("global", "tech_score_threshold_range"): PolicyKeySpec("int", 1, 5),
    ("global", "tech_rollover_score_threshold"): PolicyKeySpec("int", 0, 5),
    ("global", "sl_atr_multiplier"): PolicyKeySpec("float", 0.5, 4.0),
    ("global", "sl_sr_buffer_atr_mult"): PolicyKeySpec("float", 0.0, 1.0),
    ("global", "sl_atr_dynamic_mult"): PolicyKeySpec("float", 0.5, 6.0),
    ("global", "sl_max_pct_ceiling"): PolicyKeySpec("float", 0.003, 0.08),
    ("global", "trade_min_rr"): PolicyKeySpec("float", 0.8, 4.0),
    ("global", "trade_min_rr_range"): PolicyKeySpec("float", 0.8, 4.0),
    ("global", "tp_extension_atr_mult"): PolicyKeySpec("float", 0.0, 1.5),
    ("global", "setup_e_min_rr"): PolicyKeySpec("float", 0.8, 3.0),
    ("symbols", "BTC", "enabled"): PolicyKeySpec("bool"),
    ("symbols", "BTC", "position_risk_pct"): PolicyKeySpec("float", 0.001, 0.03),
    ("symbols", "BTC", "max_position_fraction"): PolicyKeySpec("float", 0.02, 0.40),
    ("symbols", "BTC", "break_even_trigger_r_mult"): PolicyKeySpec("float", 0.1, 3.0),
    ("symbols", "BTC", "high_atr_pct"): PolicyKeySpec("float", 0.005, 0.20),
    ("symbols", "BTC", "high_atr_min_adx"): PolicyKeySpec("float", 5.0, 80.0),
    ("symbols", "BTC", "range_disable_atr_pct"): PolicyKeySpec("float", 0.002, 0.15),
    ("symbols", "BTC", "trail_lookback_bars"): PolicyKeySpec("int", 2, 8),
    ("symbols", "BTC", "trail_stop_atr_mult"): PolicyKeySpec("float", 0.05, 2.0),
    ("symbols", "BTC", "trail_mark_buffer_pct"): PolicyKeySpec("float", 0.0, 0.02),
    ("symbols", "ETH", "enabled"): PolicyKeySpec("bool"),
    ("symbols", "SOL", "enabled"): PolicyKeySpec("bool"),
}

RISK_UP_DIRECTION_HIGHER = "higher_is_riskier"
RISK_UP_DIRECTION_LOWER = "lower_is_riskier"

RISK_RULES: dict[tuple[str, ...], str] = {
    ("global", "tech_allow_sideway"): RISK_UP_DIRECTION_HIGHER,
    ("global", "fg_extreme_block"): RISK_UP_DIRECTION_LOWER,
    ("global", "tech_score_threshold"): RISK_UP_DIRECTION_LOWER,
    ("global", "tech_score_threshold_range"): RISK_UP_DIRECTION_LOWER,
    ("global", "tech_rollover_score_threshold"): RISK_UP_DIRECTION_LOWER,
    ("global", "trade_min_rr"): RISK_UP_DIRECTION_LOWER,
    ("global", "trade_min_rr_range"): RISK_UP_DIRECTION_LOWER,
    ("global", "setup_e_min_rr"): RISK_UP_DIRECTION_LOWER,
    ("symbols", "BTC", "position_risk_pct"): RISK_UP_DIRECTION_HIGHER,
    ("symbols", "BTC", "max_position_fraction"): RISK_UP_DIRECTION_HIGHER,
    ("symbols", "BTC", "break_even_trigger_r_mult"): RISK_UP_DIRECTION_HIGHER,
    ("symbols", "BTC", "trail_stop_atr_mult"): RISK_UP_DIRECTION_HIGHER,
}


def _coerce_by_spec(value: Any, spec: PolicyKeySpec) -> Any:
    if spec.kind == "bool":
        return _env_bool(value)
    if spec.kind == "int":
        parsed = int(value)
        if spec.min_value is not None and parsed < int(spec.min_value):
            raise ValueError(f"value {parsed} < min {spec.min_value}")
        if spec.max_value is not None and parsed > int(spec.max_value):
            raise ValueError(f"value {parsed} > max {spec.max_value}")
        return parsed
    if spec.kind == "float":
        parsed = float(value)
        if spec.min_value is not None and parsed < float(spec.min_value):
            raise ValueError(f"value {parsed} < min {spec.min_value}")
        if spec.max_value is not None and parsed > float(spec.max_value):
            raise ValueError(f"value {parsed} > max {spec.max_value}")
        return round(parsed, 8)
    return value


def _is_risk_increase(base_value: Any, new_value: Any, direction: str) -> bool:
    if base_value is None or new_value is None:
        return False
    if direction == RISK_UP_DIRECTION_HIGHER:
        return new_value > base_value
    return new_value < base_value


def _is_allowed_dynamic_key(path: tuple[str, ...]) -> bool:
    if len(path) != 3:
        return False
    if path[0] == "setups" and path[2] == "enabled":
        return bool(re.match(r"^[A-Z]$", path[1]))
    return False


def _normalize_dynamic_key(path: tuple[str, ...], value: Any) -> tuple[tuple[str, ...], Any] | None:
    if len(path) != 3:
        return None
    head, mid, tail = path
    if head == "setups" and tail == "enabled":
        key = str(mid).upper().strip()[:1]
        if not key or not re.match(r"^[A-Z]$", key):
            return None
        return ("setups", key, "enabled"), _env_bool(value)
    return None


def validate_policy_patch(
    patch: dict[str, Any],
    active_policy_json: dict[str, Any],
) -> tuple[dict[str, Any], list[str], list[str]]:
    errors: list[str] = []
    sanitized: dict[str, Any] = {}
    risk_increase_changes: list[str] = []

    for path, raw_value in _flatten_patch(patch):
        if not path:
            continue
        spec = ALLOWED_POLICY_KEYS.get(path)
        norm_path = path
        value = raw_value

        if spec is None:
            if _is_allowed_dynamic_key(path):
                dynamic = _normalize_dynamic_key(path, raw_value)
                if dynamic is None:
                    errors.append(f"Invalid dynamic key path={'.'.join(path)}")
                    continue
                norm_path, value = dynamic
                _set_path(sanitized, norm_path, value)
                continue
            errors.append(f"Unknown policy key path={'.'.join(path)}")
            continue

        try:
            value = _coerce_by_spec(raw_value, spec)
        except Exception as exc:
            errors.append(f"Invalid value for {'.'.join(path)}: {exc}")
            continue

        _set_path(sanitized, norm_path, value)

    merged = _deep_merge(active_policy_json or {}, sanitized)
    for path, direction in RISK_RULES.items():
        old_value = _get_path(active_policy_json or {}, path)
        new_value = _get_path(merged, path)
        if _is_risk_increase(old_value, new_value, direction):
            risk_increase_changes.append(f"{'.'.join(path)}: {old_value!r} -> {new_value!r}")

    return sanitized, errors, risk_increase_changes


def strip_risk_increase_changes(
    sanitized_patch: dict[str, Any],
    active_policy_json: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    out = json.loads(_json_dumps(sanitized_patch))
    stripped: list[str] = []
    merged = _deep_merge(active_policy_json or {}, out)
    for path, direction in RISK_RULES.items():
        old_value = _get_path(active_policy_json or {}, path)
        new_value = _get_path(merged, path)
        if not _is_risk_increase(old_value, new_value, direction):
            continue
        _del_path(out, path)
        stripped.append(f"{'.'.join(path)}: {old_value!r} -> {new_value!r}")
    return out, stripped


def _is_patch_empty(patch: dict[str, Any]) -> bool:
    return len(_flatten_patch(patch)) == 0


def _policy_snapshot_from_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return None
    return {
        "id": int(row["id"]),
        "policy_name": row.get("policy_name"),
        "engine_name": row.get("engine_name"),
        "account_type": row.get("account_type"),
        "version": int(row.get("version") or 0),
        "status": row.get("status"),
        "policy_json": dict(row.get("policy_json") or {}),
        "validation_report": dict(row.get("validation_report") or {}),
        "source": row.get("source"),
        "reason": row.get("reason"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "effective_from": row.get("effective_from"),
        "activated_at": row.get("activated_at"),
    }


def _summarize_patch_paths(patch: dict[str, Any]) -> list[str]:
    return sorted(".".join(path) for path, _ in _flatten_patch(patch) if path)


def _build_allowed_key_specs() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path, spec in sorted(ALLOWED_POLICY_KEYS.items(), key=lambda item: ".".join(item[0])):
        out.append(
            {
                "path": ".".join(path),
                "kind": spec.kind,
                "min": spec.min_value,
                "max": spec.max_value,
            }
        )
    out.append(
        {
            "path": "setups.<A..Z>.enabled",
            "kind": "bool",
            "min": None,
            "max": None,
        }
    )
    return out


def _fetch_active_policy_row(
    conn,
    engine_name: str,
    account_type: str,
    *,
    as_of: datetime | None = None,
) -> dict[str, Any] | None:
    bound_dt = _utc_now(as_of)
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT id, policy_name, engine_name, account_type, version, status,
                   policy_json, validation_report, source, reason,
                   created_at, updated_at, effective_from, activated_at
            FROM strategy_policies
            WHERE engine_name = %s
              AND account_type = %s
              AND status = 'active'
              AND created_at <= %s
              AND (effective_from IS NULL OR effective_from <= %s)
            ORDER BY COALESCE(effective_from, created_at) DESC, version DESC
            LIMIT 1
            """,
            (engine_name, account_type, bound_dt, bound_dt),
        )
        row = cur.fetchone()
        return dict(row) if row else None


def _fetch_latest_policy_row(
    conn,
    engine_name: str,
    account_type: str,
    *,
    as_of: datetime | None = None,
) -> dict[str, Any] | None:
    bound_dt = _utc_now(as_of)
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT id, policy_name, engine_name, account_type, version, status,
                   policy_json, validation_report, source, reason,
                   created_at, updated_at, effective_from, activated_at
            FROM strategy_policies
            WHERE engine_name = %s
              AND account_type = %s
              AND created_at <= %s
            ORDER BY created_at DESC, version DESC
            LIMIT 1
            """,
            (engine_name, account_type, bound_dt),
        )
        row = cur.fetchone()
        return dict(row) if row else None


def _fetch_previous_policy_row(
    conn,
    active_row: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not active_row:
        return None
    created_at = _to_utc(active_row.get("created_at"))
    if created_at is None:
        return None
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT id, policy_name, engine_name, account_type, version, status,
                   policy_json, validation_report, source, reason,
                   created_at, updated_at, effective_from, activated_at
            FROM strategy_policies
            WHERE engine_name = %s
              AND account_type = %s
              AND created_at < %s
            ORDER BY created_at DESC, version DESC
            LIMIT 1
            """,
            (
                active_row["engine_name"],
                active_row["account_type"],
                created_at,
            ),
        )
        row = cur.fetchone()
        return dict(row) if row else None


def _normalize_setup_label(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "UNKNOWN"
    if text.upper() in {"UNKNOWN", "NONE", "?"}:
        return "UNKNOWN"
    return text


def _normalize_trade_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["setup"] = _normalize_setup_label(out.get("setup"))
    for key in ("opened_at", "closed_at"):
        out[key] = _to_utc(out.get(key))
    for key in (
        "pnl_usdt",
        "pnl_pct",
        "entry_price",
        "exit_price",
        "stop_loss",
        "take_profit",
        "take_profit_2",
        "confidence",
    ):
        if out.get(key) is not None:
            out[key] = float(out[key])
    for key in ("id", "strategy_policy_id", "strategy_policy_version"):
        if out.get(key) is not None:
            out[key] = int(out[key])
    return out


def _fetch_closed_trade_rows(
    conn,
    trades_table: str,
    account_type: str,
    *,
    since_dt: datetime | None = None,
    until_dt: datetime | None = None,
    limit_rows: int | None = None,
    policy_id: int | None = None,
    policy_version: int | None = None,
    descending: bool = True,
) -> list[dict[str, Any]]:
    has_tp2 = _has_column(conn, trades_table, "take_profit_2")
    has_confidence = _has_column(conn, trades_table, "confidence")
    has_close_reason = _has_column(conn, trades_table, "close_reason")
    has_policy_id = _has_column(conn, trades_table, "strategy_policy_id")
    has_policy_version = _has_column(conn, trades_table, "strategy_policy_version")

    if policy_id is not None and not has_policy_id:
        return []
    if policy_version is not None and not has_policy_version:
        return []

    select_cols = [
        "id",
        "symbol",
        "side",
        "setup",
        "pnl_usdt",
        "pnl_pct",
        "entry_price",
        "exit_price",
        "stop_loss",
        "take_profit",
        "opened_at",
        "closed_at",
        (
            "take_profit_2"
            if has_tp2
            else "NULL::DOUBLE PRECISION AS take_profit_2"
        ),
        (
            "confidence"
            if has_confidence
            else "NULL::DOUBLE PRECISION AS confidence"
        ),
        (
            "close_reason"
            if has_close_reason
            else "NULL::TEXT AS close_reason"
        ),
        (
            "strategy_policy_id"
            if has_policy_id
            else "NULL::BIGINT AS strategy_policy_id"
        ),
        (
            "strategy_policy_version"
            if has_policy_version
            else "NULL::INTEGER AS strategy_policy_version"
        ),
    ]

    where_clauses = [
        "account_type = %s",
        "status = 'CLOSED'",
        "pnl_usdt IS NOT NULL",
        "closed_at IS NOT NULL",
    ]
    params: list[Any] = [account_type]
    if since_dt is not None:
        where_clauses.append("closed_at >= %s")
        params.append(_to_utc(since_dt))
    if until_dt is not None:
        where_clauses.append("closed_at <= %s")
        params.append(_to_utc(until_dt))
    if policy_id is not None:
        where_clauses.append("strategy_policy_id = %s")
        params.append(int(policy_id))
    if policy_version is not None:
        where_clauses.append("strategy_policy_version = %s")
        params.append(int(policy_version))

    order_direction = "DESC" if descending else "ASC"
    query = f"""
        SELECT {", ".join(select_cols)}
        FROM {trades_table}
        WHERE {' AND '.join(where_clauses)}
        ORDER BY closed_at {order_direction}
    """
    if limit_rows is not None:
        query += " LIMIT %s"
        params.append(int(limit_rows))

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(query, tuple(params))
        rows = cur.fetchall()
    return [_normalize_trade_row(dict(row)) for row in rows]


def _safe_avg(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _compute_max_drawdown(pnl_values: list[float]) -> float:
    running = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for pnl in pnl_values:
        running += pnl
        peak = max(peak, running)
        max_drawdown = max(max_drawdown, peak - running)
    return max_drawdown


def _compute_streaks(pnl_values: list[float]) -> tuple[int, int]:
    max_win = 0
    max_loss = 0
    cur_win = 0
    cur_loss = 0
    for pnl in pnl_values:
        if pnl > 0:
            cur_win += 1
            cur_loss = 0
        elif pnl < 0:
            cur_loss += 1
            cur_win = 0
        else:
            cur_win = 0
            cur_loss = 0
        max_win = max(max_win, cur_win)
        max_loss = max(max_loss, cur_loss)
    return max_win, max_loss


def _summarize_trade_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    normalized_rows = [dict(row) for row in rows]
    if not normalized_rows:
        return {
            "closed_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate_pct": 0.0,
            "total_pnl_usdt": 0.0,
            "avg_pnl_usdt": 0.0,
            "avg_win_usdt": 0.0,
            "avg_loss_usdt": 0.0,
            "expectancy_usdt": 0.0,
            "profit_factor": None,
            "max_drawdown_usdt": 0.0,
            "max_win_streak": 0,
            "max_loss_streak": 0,
            "avg_duration_hours": 0.0,
            "max_duration_hours": 0.0,
            "best_trade_usdt": 0.0,
            "worst_trade_usdt": 0.0,
            "start_closed_at": None,
            "end_closed_at": None,
        }

    pnls_desc = [float(row.get("pnl_usdt") or 0.0) for row in normalized_rows]
    wins = [pnl for pnl in pnls_desc if pnl > 0]
    losses = [pnl for pnl in pnls_desc if pnl < 0]
    rows_asc = sorted(
        normalized_rows,
        key=lambda row: row.get("closed_at") or datetime.min.replace(tzinfo=UTC),
    )
    pnls_asc = [float(row.get("pnl_usdt") or 0.0) for row in rows_asc]
    durations = []
    for row in normalized_rows:
        opened_at = row.get("opened_at")
        closed_at = row.get("closed_at")
        if opened_at and closed_at:
            durations.append((closed_at - opened_at).total_seconds() / 3600.0)
    max_win_streak, max_loss_streak = _compute_streaks(pnls_asc)
    gross_profit = float(sum(wins))
    gross_loss = abs(float(sum(losses)))
    profit_factor = None if gross_loss <= 0 else round(gross_profit / gross_loss, 4)
    return {
        "closed_trades": len(normalized_rows),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round((len(wins) / len(normalized_rows)) * 100.0, 2),
        "total_pnl_usdt": round(float(sum(pnls_desc)), 4),
        "avg_pnl_usdt": round(_safe_avg(pnls_desc), 4),
        "avg_win_usdt": round(_safe_avg(wins), 4),
        "avg_loss_usdt": round(_safe_avg(losses), 4),
        "expectancy_usdt": round(_safe_avg(pnls_desc), 4),
        "profit_factor": profit_factor,
        "max_drawdown_usdt": round(_compute_max_drawdown(pnls_asc), 4),
        "max_win_streak": int(max_win_streak),
        "max_loss_streak": int(max_loss_streak),
        "avg_duration_hours": round(_safe_avg(durations), 4),
        "max_duration_hours": round(max(durations), 4) if durations else 0.0,
        "best_trade_usdt": round(max(pnls_desc), 4),
        "worst_trade_usdt": round(min(pnls_desc), 4),
        "start_closed_at": rows_asc[0].get("closed_at"),
        "end_closed_at": rows_asc[-1].get("closed_at"),
    }


def _summarize_trade_groups(
    rows: list[dict[str, Any]],
    key_names: tuple[str, ...],
    *,
    top_n: int = 12,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(str(row.get(name) or "UNKNOWN") for name in key_names)
        grouped[key].append(row)

    summaries: list[dict[str, Any]] = []
    for key, group_rows in grouped.items():
        item = {name: key[idx] for idx, name in enumerate(key_names)}
        item.update(_summarize_trade_rows(group_rows))
        summaries.append(item)

    summaries.sort(
        key=lambda item: (
            float(item.get("total_pnl_usdt") or 0.0),
            float(item.get("expectancy_usdt") or 0.0),
            -int(item.get("closed_trades") or 0),
            tuple(str(item.get(name) or "") for name in key_names),
        )
    )
    return summaries[:top_n]


def _setup_code_from_name(setup_name: Any) -> str:
    text = str(setup_name or "").strip().upper()
    if not text:
        return "UNKNOWN"
    if text.startswith("SETUP "):
        return text.split("SETUP ", 1)[1][:1]
    if text.startswith("SETUP_"):
        return text.split("SETUP_", 1)[1][:1]
    return text[:1]


def _build_regime_proxy_slices(rows: list[dict[str, Any]]) -> dict[str, Any]:
    range_rows = [row for row in rows if _setup_code_from_name(row.get("setup")) == "D"]
    trend_rows = [
        row
        for row in rows
        if _setup_code_from_name(row.get("setup")) in {"A", "B", "C"}
    ]
    other_rows = [
        row
        for row in rows
        if _setup_code_from_name(row.get("setup")) not in {"A", "B", "C", "D"}
    ]
    return {
        "range_proxy": _summarize_trade_rows(range_rows),
        "trend_proxy": _summarize_trade_rows(trend_rows),
        "other_or_unknown": _summarize_trade_rows(other_rows),
    }


def _compute_data_quality(rows: list[dict[str, Any]]) -> dict[str, Any]:
    unknown_setups = 0
    noncanonical_setups = 0
    legacy_unlabeled_setups = 0
    missing_risk_levels = 0
    missing_close_reason = 0
    for row in rows:
        setup_label = _normalize_setup_label(row.get("setup"))
        if setup_label == "UNKNOWN":
            unknown_setups += 1
        if _setup_code_from_name(setup_label) not in {"A", "B", "C", "D"}:
            noncanonical_setups += 1
        if setup_label in {"LEGACY_UNLABELED", "RECOVERED_ORPHAN"}:
            legacy_unlabeled_setups += 1
        if not row.get("stop_loss") or not row.get("take_profit"):
            missing_risk_levels += 1
        if not str(row.get("close_reason") or "").strip():
            missing_close_reason += 1
    return {
        "rows_reviewed": len(rows),
        "unknown_setup_trades": int(unknown_setups),
        "noncanonical_setup_trades": int(noncanonical_setups),
        "legacy_unlabeled_setup_trades": int(legacy_unlabeled_setups),
        "missing_risk_level_trades": int(missing_risk_levels),
        "missing_close_reason_trades": int(missing_close_reason),
    }


def _metric_delta(current_value: Any, previous_value: Any, digits: int = 4) -> Any:
    if current_value is None or previous_value is None:
        return None
    try:
        return round(float(current_value) - float(previous_value), digits)
    except (TypeError, ValueError):
        return None


def _build_policy_change_comparison(
    conn,
    trades_table: str,
    account_type: str,
    active_row: dict[str, Any] | None,
    *,
    as_of: datetime | None = None,
) -> dict[str, Any]:
    if not active_row:
        return {
            "active_policy": None,
            "previous_policy": None,
            "matched_recent_sample": None,
        }

    previous_row = _fetch_previous_policy_row(conn, active_row)
    active_started_at = _to_utc(
        active_row.get("activated_at")
        or active_row.get("effective_from")
        or active_row.get("created_at")
    )
    active_rows = _fetch_closed_trade_rows(
        conn,
        trades_table,
        account_type,
        since_dt=active_started_at,
        until_dt=as_of,
        limit_rows=250,
        policy_id=int(active_row["id"]),
    )
    previous_rows: list[dict[str, Any]] = []
    if previous_row:
        previous_started_at = _to_utc(
            previous_row.get("activated_at")
            or previous_row.get("effective_from")
            or previous_row.get("created_at")
        )
        previous_rows = _fetch_closed_trade_rows(
            conn,
            trades_table,
            account_type,
            since_dt=previous_started_at,
            until_dt=as_of,
            limit_rows=250,
            policy_id=int(previous_row["id"]),
        )

    active_summary = _summarize_trade_rows(active_rows)
    previous_summary = _summarize_trade_rows(previous_rows)
    matched_n = min(len(active_rows), len(previous_rows), 30)
    matched_comparison = None
    if matched_n > 0:
        active_matched = _summarize_trade_rows(active_rows[:matched_n])
        previous_matched = _summarize_trade_rows(previous_rows[:matched_n])
        matched_comparison = {
            "matched_trade_count": matched_n,
            "active_policy": active_matched,
            "previous_policy": previous_matched,
            "deltas": {
                "total_pnl_usdt": _metric_delta(
                    active_matched.get("total_pnl_usdt"),
                    previous_matched.get("total_pnl_usdt"),
                ),
                "win_rate_pct": _metric_delta(
                    active_matched.get("win_rate_pct"),
                    previous_matched.get("win_rate_pct"),
                    digits=2,
                ),
                "expectancy_usdt": _metric_delta(
                    active_matched.get("expectancy_usdt"),
                    previous_matched.get("expectancy_usdt"),
                ),
            },
        }

    return {
        "active_policy": {
            "policy": _policy_snapshot_from_row(active_row),
            "summary": active_summary,
        },
        "previous_policy": (
            {
                "policy": _policy_snapshot_from_row(previous_row),
                "summary": previous_summary,
            }
            if previous_row
            else None
        ),
        "matched_recent_sample": matched_comparison,
    }


def _build_engine_behavior_contract() -> dict[str, Any]:
    return {
        "signal_source": "rule_based_setup_selector",
        "selector_function": "build_rule_based_signal_decision",
        "supported_setups": [
            {
                "code": "A",
                "name": "Setup A",
                "intent": "trend EMA pullback continuation",
                "policy_surface": ["setups.A.enabled"],
            },
            {
                "code": "B",
                "name": "Setup B",
                "intent": "trend breakout continuation",
                "policy_surface": ["setups.B.enabled"],
            },
            {
                "code": "C",
                "name": "Setup C",
                "intent": "trend retest hold or retest fail continuation",
                "policy_surface": ["setups.C.enabled"],
            },
            {
                "code": "D",
                "name": "Setup D",
                "intent": "range edge reaction near support or resistance",
                "policy_surface": [
                    "setups.D.enabled",
                    "global.trade_min_rr_range",
                    "symbols.BTC.range_disable_atr_pct",
                    "symbols.BTC.high_atr_pct",
                    "symbols.BTC.high_atr_min_adx",
                ],
            },
        ],
        "candidate_order_by_regime": {
            "UPTREND": ["Setup C", "Setup B", "Setup A"],
            "DOWNTREND": ["Setup C", "Setup B", "Setup A"],
            "SIDEWAY_OR_RANGE": ["Setup D"],
        },
        "always_enforced_validations": [
            "Signal must remain BUY, SELL, or WAIT after deterministic selection.",
            "Only Setup A/B/C/D are valid; policy cannot invent new setups.",
            "validate_ai_trade_decision enforces structural alignment and setup timing.",
            "Setup A requires H1 and M15 trend alignment.",
            "BTC Setup D can be blocked in high ATR or strong ADX conditions.",
            "build_trade_plan recomputes entry, stop, take-profit, and minimum R:R before execution.",
        ],
        "policy_effects": {
            "symbols.<SYMBOL>.enabled": "disable all trades for that symbol",
            "setups.<A..Z>.enabled": "disable one setup family without changing others",
            "global.trade_min_rr": "raise or lower trend-style minimum R:R acceptance",
            "global.trade_min_rr_range": "raise or lower range-style minimum R:R acceptance",
            "symbols.BTC.position_risk_pct": "changes BTC position sizing only",
            "symbols.BTC.max_position_fraction": "caps BTC exposure only",
            "symbols.BTC.break_even_trigger_r_mult": "changes BTC break-even trigger only",
            "symbols.BTC.trail_stop_atr_mult": "changes BTC trailing-stop looseness only",
        },
        "non_policy_truths": [
            "Policy patches cannot create new trade opportunities that the deterministic selector never saw.",
            "Policy patches mostly remove, tighten, or resize deterministic trades.",
            "A good proposal should explain exactly which existing trade families it expects to remove or retain.",
        ],
    }


def _build_review_evidence_pack(
    conn,
    trades_table: str,
    account_type: str,
    active_row: dict[str, Any] | None,
    guard_context: dict[str, Any],
    *,
    as_of: datetime | None = None,
) -> dict[str, Any]:
    as_of_dt = _utc_now(as_of)
    recent_rows = _fetch_closed_trade_rows(
        conn,
        trades_table,
        account_type,
        until_dt=as_of_dt,
        limit_rows=200,
    )
    rows_7d = _fetch_closed_trade_rows(
        conn,
        trades_table,
        account_type,
        since_dt=as_of_dt - timedelta(days=7),
        until_dt=as_of_dt,
        limit_rows=300,
    )
    rows_30d = _fetch_closed_trade_rows(
        conn,
        trades_table,
        account_type,
        since_dt=as_of_dt - timedelta(days=30),
        until_dt=as_of_dt,
        limit_rows=600,
    )
    policy_started_at = _to_utc(
        ((guard_context or {}).get("policy_timing") or {}).get("policy_started_at")
    ) or _to_utc(
        (active_row or {}).get("activated_at")
        or (active_row or {}).get("effective_from")
        or (active_row or {}).get("created_at")
    )
    since_policy_rows = (
        _fetch_closed_trade_rows(
            conn,
            trades_table,
            account_type,
            since_dt=policy_started_at,
            until_dt=as_of_dt,
            limit_rows=400,
        )
        if policy_started_at is not None
        else []
    )

    recent_scope = recent_rows[:100]
    recent_losses = [row for row in recent_rows if float(row.get("pnl_usdt") or 0.0) < 0][:50]

    return {
        "metadata": {
            "as_of": as_of_dt,
            "recent_closed_trades_considered": len(recent_rows),
            "policy_started_at": policy_started_at,
        },
        "windows": {
            "last_20_trades": _summarize_trade_rows(recent_rows[:20]),
            "last_50_trades": _summarize_trade_rows(recent_rows[:50]),
            "last_100_trades": _summarize_trade_rows(recent_rows[:100]),
            "last_7_days": _summarize_trade_rows(rows_7d),
            "last_30_days": _summarize_trade_rows(rows_30d),
            "since_policy_start": _summarize_trade_rows(since_policy_rows),
        },
        "by_symbol": _summarize_trade_groups(recent_scope, ("symbol",)),
        "by_setup": _summarize_trade_groups(recent_scope, ("setup",)),
        "by_symbol_setup": _summarize_trade_groups(recent_scope, ("symbol", "setup")),
        "recent_loss_patterns": _summarize_trade_groups(
            recent_losses,
            ("symbol", "setup"),
            top_n=12,
        ),
        "regime_proxy_slices": _build_regime_proxy_slices(recent_scope),
        "data_quality": _compute_data_quality(recent_rows[:100]),
        "policy_change_comparison": _build_policy_change_comparison(
            conn,
            trades_table,
            account_type,
            active_row,
            as_of=as_of_dt,
        ),
    }


def _build_deterministic_review_checks(
    review_input: dict[str, Any],
    cfg: ReviewGuardConfig,
    *,
    bypass_maturity_gate: bool = False,
) -> dict[str, Any]:
    guard = dict(review_input.get("guard") or {})
    guard_context = dict(guard.get("context") or {})
    evidence_pack = dict(review_input.get("evidence_pack") or {})
    windows = dict(evidence_pack.get("windows") or {})
    data_quality = dict(evidence_pack.get("data_quality") or {})
    last_50 = dict(windows.get("last_50_trades") or {})
    since_policy = dict(windows.get("since_policy_start") or {})
    policy_comparison = dict(evidence_pack.get("policy_change_comparison") or {})

    issues: list[str] = []
    warnings: list[str] = []
    strengths: list[str] = []

    maturity_gate_passed = str(guard.get("decision") or "").upper() == "ALLOW_REVIEW"
    if not maturity_gate_passed:
        issues.append(str(guard.get("reason") or "maturity gate failed"))

    if int(data_quality.get("unknown_setup_trades") or 0) > 0:
        warnings.append(
            f"{int(data_quality.get('unknown_setup_trades') or 0)} recent trades have unknown setup labels"
        )
    if int(data_quality.get("legacy_unlabeled_setup_trades") or 0) > 0:
        warnings.append(
            f"{int(data_quality.get('legacy_unlabeled_setup_trades') or 0)} recent trades are legacy or recovered rows without canonical setup labels"
        )
    if int(data_quality.get("noncanonical_setup_trades") or 0) > 0:
        warnings.append(
            f"{int(data_quality.get('noncanonical_setup_trades') or 0)} recent trades use non-canonical setup labels outside Setup A/B/C/D"
        )
    if int(last_50.get("closed_trades") or 0) < max(20, cfg.min_closed_trades_since_update):
        warnings.append(
            "recent trade sample is thin relative to the review threshold"
        )
    if float(last_50.get("max_drawdown_usdt") or 0.0) > 0.0:
        strengths.append(
            f"last_50 sample captures drawdown up to {float(last_50.get('max_drawdown_usdt') or 0.0):.2f} USDT"
        )
    if int(since_policy.get("closed_trades") or 0) >= cfg.min_closed_trades_since_update:
        strengths.append(
            f"post-update sample includes {int(since_policy.get('closed_trades') or 0)} closed trades"
        )

    matched_sample = policy_comparison.get("matched_recent_sample")
    if matched_sample:
        deltas = dict((matched_sample or {}).get("deltas") or {})
        pnl_delta = deltas.get("total_pnl_usdt")
        if pnl_delta is not None:
            direction = "improved" if float(pnl_delta) > 0 else "weakened" if float(pnl_delta) < 0 else "matched"
            strengths.append(
                f"compared with the previous policy on matched sample, recent pnl {direction} by {float(pnl_delta):.2f} USDT"
            )

    clear_signal, clear_signal_evidence = _compute_clear_signal_for_risk_increase(
        guard_context,
        min_evidence_trades=max(cfg.min_closed_trades_since_update, 30),
    )

    qa_verdict = "OK"
    if issues:
        qa_verdict = "FAIL"
    elif warnings:
        qa_verdict = "WARN"

    summary_parts = []
    if issues:
        summary_parts.append("Guard failed or evidence immature.")
    if warnings:
        summary_parts.append("Use extra caution because the evidence pack still has noise.")
    if not summary_parts:
        summary_parts.append("Evidence pack is mature enough for a conservative review.")

    return {
        "qa_verdict": qa_verdict,
        "maturity_gate_passed": maturity_gate_passed,
        "bypass_maturity_gate": bool(bypass_maturity_gate),
        "allow_llm_proposal": bool(maturity_gate_passed or bypass_maturity_gate),
        "issues": issues,
        "warnings": warnings,
        "strengths": strengths,
        "risk_constraints": {
            "allow_risk_increase": bool(clear_signal),
            "max_changed_keys": 2,
            "prefer_no_change_without_strong_edge": True,
        },
        "clear_signal_for_risk_increase": bool(clear_signal),
        "clear_signal_evidence": clear_signal_evidence,
        "summary": " ".join(summary_parts),
    }


def build_policy_review_context(
    *,
    database_url: str,
    engine_name: str,
    account_type: str,
    trades_table: str,
    config: ReviewGuardConfig | None = None,
    as_of: datetime | None = None,
    include_latest_policy_fallback: bool = False,
    bypass_maturity_gate: bool = False,
) -> tuple[dict[str, Any], dict[str, Any] | None, Any, ReviewGuardConfig]:
    cfg = config or ReviewGuardConfig(trades_table=trades_table)
    guard = evaluate_policy_review_guard(
        database_url=database_url,
        engine_name=engine_name,
        account_type=account_type,
        config=cfg,
        as_of=as_of,
    )
    guard_payload = asdict(guard)
    guard_context = dict(guard_payload.get("context") or {})

    conn = psycopg2.connect(database_url, sslmode="require")
    try:
        active_row = _fetch_active_policy_row(
            conn,
            engine_name,
            account_type,
            as_of=as_of,
        )
        if not active_row and include_latest_policy_fallback:
            active_row = _fetch_latest_policy_row(
                conn,
                engine_name,
                account_type,
                as_of=as_of,
            )
        evidence_pack = _build_review_evidence_pack(
            conn,
            trades_table,
            account_type,
            active_row,
            guard_context,
            as_of=as_of,
        )
    finally:
        conn.close()

    review_input = {
        "engine_name": engine_name,
        "account_type": account_type,
        "as_of": _utc_now(as_of),
        "guard": guard_payload,
        "active_policy": _policy_snapshot_from_row(active_row),
        "allowed_keys": _build_allowed_key_specs(),
        "engine_behavior": _build_engine_behavior_contract(),
        "evidence_pack": evidence_pack,
    }
    review_input["deterministic_checks"] = _build_deterministic_review_checks(
        review_input,
        cfg,
        bypass_maturity_gate=bypass_maturity_gate,
    )
    return review_input, active_row, guard, cfg


def _fetch_recent_setup_symbol_stats(
    conn,
    trades_table: str,
    account_type: str,
    limit_rows: int = 50,
    *,
    as_of: datetime | None = None,
) -> dict[str, Any]:
    rows = _fetch_closed_trade_rows(
        conn,
        trades_table,
        account_type,
        until_dt=as_of,
        limit_rows=limit_rows,
    )
    loss_rows = [row for row in rows if float(row.get("pnl_usdt") or 0.0) < 0][:limit_rows]
    return {
        "last_closed_trades_by_symbol_setup": _summarize_trade_groups(
            rows,
            ("symbol", "setup"),
            top_n=limit_rows,
        ),
        "last_losing_trades_patterns": _summarize_trade_groups(
            loss_rows,
            ("symbol", "setup"),
            top_n=limit_rows,
        ),
    }


def _fetch_opportunity_rows(
    conn,
    opportunities_table: str,
    account_type: str,
    *,
    since_dt: datetime | None = None,
    until_dt: datetime | None = None,
    limit_rows: int | None = None,
    engine_name: str | None = None,
) -> list[dict[str, Any]]:
    has_trade_id = _has_column(conn, opportunities_table, "linked_trade_id")
    has_decision_json = _has_column(conn, opportunities_table, "decision_json")
    has_context_json = _has_column(conn, opportunities_table, "context_json")
    has_preview_plan = _has_column(conn, opportunities_table, "preview_plan_json")

    select_cols = [
        "id",
        "cycle_at",
        "symbol",
        "signal",
        "setup",
        "status",
        "status_reason",
        "decision_reason",
        "market_mode",
        "primary_trend",
        "allowed_direction",
        "is_range",
        "range_bias",
        "score",
        "current_price",
        "entry_price",
        "stop_loss",
        "take_profit",
        "take_profit_2",
        "planned_rr",
        "target_mode",
        "strategy_policy_id",
        "strategy_policy_version",
        (
            "linked_trade_id"
            if has_trade_id
            else "NULL::BIGINT AS linked_trade_id"
        ),
        (
            "decision_json"
            if has_decision_json
            else "'{}'::jsonb AS decision_json"
        ),
        (
            "context_json"
            if has_context_json
            else "'{}'::jsonb AS context_json"
        ),
        (
            "preview_plan_json"
            if has_preview_plan
            else "'{}'::jsonb AS preview_plan_json"
        ),
    ]

    where_clauses = ["account_type = %s"]
    params: list[Any] = [account_type]
    if engine_name:
        where_clauses.append("engine_name = %s")
        params.append(engine_name)
    if since_dt is not None:
        where_clauses.append("cycle_at >= %s")
        params.append(_to_utc(since_dt))
    if until_dt is not None:
        where_clauses.append("cycle_at <= %s")
        params.append(_to_utc(until_dt))

    query = f"""
        SELECT {", ".join(select_cols)}
        FROM {opportunities_table}
        WHERE {' AND '.join(where_clauses)}
        ORDER BY cycle_at ASC
    """
    if limit_rows is not None:
        query += " LIMIT %s"
        params.append(int(limit_rows))

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(query, tuple(params))
        rows = cur.fetchall()

    normalized: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["cycle_at"] = _to_utc(item.get("cycle_at"))
        for key in (
            "current_price",
            "entry_price",
            "stop_loss",
            "take_profit",
            "take_profit_2",
            "planned_rr",
        ):
            if item.get(key) is not None:
                item[key] = float(item[key])
        for key in ("id", "strategy_policy_id", "strategy_policy_version", "linked_trade_id"):
            if item.get(key) is not None:
                item[key] = int(item[key])
        item["decision_json"] = dict(item.get("decision_json") or {})
        item["context_json"] = dict(item.get("context_json") or {})
        item["preview_plan_json"] = dict(item.get("preview_plan_json") or {})
        normalized.append(item)
    return normalized


def _get_llm_client_and_model() -> tuple[OpenAI, str, str]:
    provider = os.getenv("POLICY_REVIEW_LLM_PROVIDER", "openai").strip().lower()

    if provider in {"google", "gemini"}:
        api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is required for POLICY_REVIEW_LLM_PROVIDER=google")
        base_url = os.getenv(
            "POLICY_REVIEW_LLM_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        ).strip()
        model = os.getenv("POLICY_REVIEW_MODEL", "gemini-2.5-pro").strip()
        return OpenAI(api_key=api_key, base_url=base_url), provider, model

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for policy review")
    base_url = os.getenv("POLICY_REVIEW_LLM_BASE_URL", "").strip() or None
    model = os.getenv("POLICY_REVIEW_MODEL", "gpt-5-mini").strip()
    return OpenAI(api_key=api_key, base_url=base_url), provider, model


def _set_nested_from_dot_path(target: dict[str, Any], raw_path: Any, value: Any) -> bool:
    path_text = str(raw_path or "").strip()
    if not path_text:
        return False
    parts = tuple(part.strip() for part in path_text.split(".") if part.strip())
    if not parts:
        return False
    _set_path(target, parts, value)
    return True


def _normalize_structured_changes(changes: Any) -> tuple[dict[str, Any], list[str], str]:
    if isinstance(changes, dict):
        out: dict[str, Any] = {}
        skipped: list[str] = []
        for raw_key, raw_value in changes.items():
            if "." in str(raw_key):
                if not _set_nested_from_dot_path(out, raw_key, raw_value):
                    skipped.append(f"invalid path={raw_key!r}")
                continue
            if isinstance(raw_value, dict):
                out[str(raw_key)] = copy.deepcopy(raw_value)
            else:
                out[str(raw_key)] = raw_value
        return out, skipped, "dict"

    if isinstance(changes, list):
        out = {}
        skipped = []
        for idx, item in enumerate(changes):
            if not isinstance(item, dict):
                skipped.append(f"row[{idx}] not an object")
                continue
            if "path" not in item or "value" not in item:
                skipped.append(f"row[{idx}] missing path or value")
                continue
            if not _set_nested_from_dot_path(out, item.get("path"), item.get("value")):
                skipped.append(f"row[{idx}] invalid path")
        return out, skipped, "list"

    return {}, ["unsupported changes payload"], "unknown"


def extract_policy_patch_from_response(payload: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(payload, dict):
        return {}, {"shape": "non_object", "source": "none", "skipped": ["payload not an object"]}

    changes_payload = payload.get("changes")
    if "changes" in payload and changes_payload is not None:
        patch, skipped, shape = _normalize_structured_changes(changes_payload)
        return patch, {"shape": shape, "source": "changes", "skipped": skipped}

    patch_payload = payload.get("patch")
    if isinstance(patch_payload, dict):
        patch_changes = patch_payload.get("changes")
        if "changes" in patch_payload and patch_changes is not None:
            patch, skipped, shape = _normalize_structured_changes(patch_changes)
            return patch, {"shape": shape, "source": "patch.changes", "skipped": skipped}
        if "changes" in patch_payload and patch_changes is None:
            patch_payload = {key: value for key, value in patch_payload.items() if key != "changes"}
        patch, skipped, shape = _normalize_structured_changes(patch_payload)
        return patch, {"shape": shape, "source": "patch", "skipped": skipped}

    policy_roots = {"global", "symbols", "setups"}
    if any(key in payload for key in policy_roots) or any("." in str(key) for key in payload):
        patch, skipped, shape = _normalize_structured_changes(payload)
        return patch, {"shape": shape, "source": "direct", "skipped": skipped}

    return {}, {"shape": "object", "source": "none", "skipped": ["no patch payload found"]}


def _build_reviewer_prompts(review_input: dict[str, Any]) -> tuple[str, str]:
    system_prompt = (
        "You are a conservative crypto strategy reviewer for a deterministic trading engine. "
        "Return strict JSON only. Prefer NO_CHANGE unless evidence is strong, stable, and tied to the real engine behavior. "
        "Do not optimize for trade frequency. Avoid overfitting short windows. "
        "Treat the engine_behavior and deterministic_checks fields as ground truth."
    )

    user_prompt = f"""Review this strategy context and propose either NO_CHANGE or a minimal PATCH.

Scope:
- Engine/account: {review_input.get("engine_name")} / {review_input.get("account_type")}
- Review timestamp: {review_input.get("as_of")}
- Policy changes should be rare, evidence-based, and easy to replay.
- Deterministic rules:
  1) The engine is rule-based; do not assume hidden discretionary behavior.
  2) Use only allowed keys that can actually influence the deterministic selector or trade planner.
  3) If deterministic_checks.allow_llm_proposal is false, return NO_CHANGE.
  4) Prefer one or two key changes. Only exceed that if the evidence pack clearly demands it.
  5) Cite evidence from the evidence pack and mention at least one disconfirming fact.

Allowed patch paths:
- global.<key> from current runtime policy config
- symbols.BTC.<key> for BTC risk controls
- symbols.<BTC|ETH|SOL>.enabled
- setups.<A..Z>.enabled

Return JSON exactly in this schema:
{{
  "decision": "NO_CHANGE or PROPOSE_CHANGE",
  "reason": "short explanation",
  "confidence": 0.0,
  "evidence_used": ["short evidence bullets"],
  "disconfirming_evidence": ["facts that argue against changing policy"],
  "expected_effect": {{
    "trade_count": "higher/lower/flat",
    "risk_profile": "higher/lower/flat",
    "targeted_trade_families": ["symbol/setup groups impacted"]
  }},
  "changes": [
    {{
      "path": "global.trade_min_rr",
      "value": 1.6,
      "because": "why this specific change is justified"
    }}
  ]
}}

Context JSON:
{_json_dumps(review_input)}
"""
    return system_prompt, user_prompt


def _call_llm_reviewer(review_input: dict[str, Any]) -> dict[str, Any]:
    client, provider, model = _get_llm_client_and_model()
    system_prompt, user_prompt = _build_reviewer_prompts(review_input)

    request_kwargs: dict[str, Any] = {
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    # Some models (including certain GPT-5 variants) only accept default temperature.
    # Keep temperature optional via env; if unset, omit it entirely.
    temp_raw = os.getenv("POLICY_REVIEW_TEMPERATURE", "").strip()
    if temp_raw:
        request_kwargs["temperature"] = float(temp_raw)

    response = client.chat.completions.create(**request_kwargs)
    raw = (response.choices[0].message.content or "").strip()
    parsed = json.loads(raw) if raw else {}
    parsed["_meta"] = {
        "provider": provider,
        "model": model,
    }
    return parsed


def _compute_clear_signal_for_risk_increase(
    guard_context: dict[str, Any],
    min_evidence_trades: int,
) -> tuple[bool, dict[str, Any]]:
    post_update = dict(guard_context.get("post_update_sample") or {})
    windows = dict(guard_context.get("performance_windows") or {})
    last_50 = dict(windows.get("last_50_trades") or {})
    last_7d = dict(windows.get("last_7_days") or {})

    closed_since_update = _to_int(post_update.get("closed_trades_since_update"), 0)
    closed_50 = _to_int(last_50.get("closed_trades"), 0)
    win_rate_50 = _to_float(last_50.get("win_rate_pct"), 0.0)
    pnl_50 = _to_float(last_50.get("total_pnl_usdt"), 0.0)
    pnl_7d = _to_float(last_7d.get("total_pnl_usdt"), 0.0)

    clear_signal = (
        closed_since_update >= min_evidence_trades
        and closed_50 >= min(50, min_evidence_trades)
        and win_rate_50 >= 57.0
        and pnl_50 > 0.0
        and pnl_7d >= 0.0
    )
    evidence = {
        "closed_trades_since_update": closed_since_update,
        "last_50_closed_trades": closed_50,
        "last_50_win_rate_pct": round(win_rate_50, 2),
        "last_50_total_pnl_usdt": round(pnl_50, 4),
        "last_7d_total_pnl_usdt": round(pnl_7d, 4),
        "min_evidence_trades_required": int(min_evidence_trades),
    }
    return clear_signal, evidence


def _is_symbol_enabled(policy_json: dict[str, Any], symbol: str) -> bool:
    value = _get_path(policy_json or {}, ("symbols", str(symbol or "").upper(), "enabled"))
    return True if value is None else _env_bool(value)


def _is_setup_enabled(policy_json: dict[str, Any], setup_name: Any) -> bool:
    code = _setup_code_from_name(setup_name)
    value = _get_path(policy_json or {}, ("setups", code, "enabled"))
    return True if value is None else _env_bool(value)


def _estimate_trade_rr(trade: dict[str, Any]) -> float | None:
    side = str(trade.get("side") or "").upper()
    entry = _to_float(trade.get("entry_price"), 0.0)
    stop_loss = _to_float(trade.get("stop_loss"), 0.0)
    if entry <= 0 or stop_loss <= 0:
        return None
    risk = abs(entry - stop_loss)
    if risk <= 0:
        return None

    candidates = []
    for key in ("take_profit", "take_profit_2"):
        tp = trade.get(key)
        if tp is None:
            continue
        tp_value = _to_float(tp, 0.0)
        if tp_value <= 0:
            continue
        reward = (tp_value - entry) if side == "BUY" else (entry - tp_value)
        if reward > 0:
            candidates.append(reward / risk)
    if not candidates:
        return None
    return max(candidates)


def _trade_allowed_under_policy(
    policy_json: dict[str, Any],
    trade: dict[str, Any],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    symbol = str(trade.get("symbol") or "").upper()
    setup_name = trade.get("setup")
    if symbol and not _is_symbol_enabled(policy_json, symbol):
        reasons.append(f"symbol_disabled:{symbol}")
    if not _is_setup_enabled(policy_json, setup_name):
        reasons.append(f"setup_disabled:{_setup_code_from_name(setup_name)}")

    estimated_rr = _estimate_trade_rr(trade)
    if estimated_rr is not None:
        setup_code = _setup_code_from_name(setup_name)
        rr_key = ("global", "trade_min_rr_range") if setup_code == "D" else ("global", "trade_min_rr")
        rr_threshold = _to_float(_get_path(policy_json or {}, rr_key), 0.0)
        if rr_threshold > 0 and estimated_rr < rr_threshold:
            reasons.append(f"rr_below_threshold:{estimated_rr:.3f}<{rr_threshold:.3f}")

    return len(reasons) == 0, reasons


def score_policy_patch_on_trades(
    active_policy_json: dict[str, Any],
    sanitized_patch: dict[str, Any],
    forward_trades: list[dict[str, Any]],
) -> dict[str, Any]:
    merged_policy = _deep_merge(active_policy_json or {}, sanitized_patch or {})
    kept_rows: list[dict[str, Any]] = []
    removed_rows: list[dict[str, Any]] = []
    blocked_reasons: dict[str, int] = defaultdict(int)
    removed_reasons_by_trade: list[dict[str, Any]] = []

    for trade in forward_trades:
        allowed, reasons = _trade_allowed_under_policy(merged_policy, trade)
        if allowed:
            kept_rows.append(trade)
            continue
        removed_rows.append(trade)
        for reason in reasons:
            blocked_reasons[reason] += 1
        removed_reasons_by_trade.append(
            {
                "trade_id": trade.get("id"),
                "symbol": trade.get("symbol"),
                "setup": trade.get("setup"),
                "pnl_usdt": trade.get("pnl_usdt"),
                "reasons": reasons,
            }
        )

    baseline = _summarize_trade_rows(forward_trades)
    candidate = _summarize_trade_rows(kept_rows)
    removed = _summarize_trade_rows(removed_rows)
    baseline_closed = int(baseline.get("closed_trades") or 0)
    candidate_closed = int(candidate.get("closed_trades") or 0)
    trade_delta = candidate_closed - baseline_closed

    changed_paths = _summarize_patch_paths(sanitized_patch)
    modeled_prefixes = {
        "global.trade_min_rr",
        "global.trade_min_rr_range",
    }
    unmodeled_paths = [
        path
        for path in changed_paths
        if not path.endswith(".enabled") and path not in modeled_prefixes
    ]

    return {
        "method": "trade_filter_counterfactual",
        "changed_paths": changed_paths,
        "unmodeled_paths": unmodeled_paths,
        "baseline": baseline,
        "candidate": candidate,
        "removed": removed,
        "deltas": {
            "closed_trades": trade_delta,
            "trade_coverage_pct": round((candidate_closed / baseline_closed) * 100.0, 2)
            if baseline_closed
            else 0.0,
            "total_pnl_usdt": _metric_delta(
                candidate.get("total_pnl_usdt"),
                baseline.get("total_pnl_usdt"),
            ),
            "win_rate_pct": _metric_delta(
                candidate.get("win_rate_pct"),
                baseline.get("win_rate_pct"),
                digits=2,
            ),
            "expectancy_usdt": _metric_delta(
                candidate.get("expectancy_usdt"),
                baseline.get("expectancy_usdt"),
            ),
        },
        "blocked_reason_counts": dict(sorted(blocked_reasons.items())),
        "removed_trade_families": _summarize_trade_groups(removed_rows, ("symbol", "setup"), top_n=8),
        "removed_trade_examples": removed_reasons_by_trade[:10],
    }


def _rr_threshold_for_setup(policy_json: dict[str, Any], setup_name: Any) -> float:
    setup_code = _setup_code_from_name(setup_name)
    rr_key = ("global", "trade_min_rr_range") if setup_code == "D" else ("global", "trade_min_rr")
    return _to_float(_get_path(policy_json or {}, rr_key), 0.0)


def _best_preview_rr(opportunity: dict[str, Any]) -> float | None:
    planned_rr = opportunity.get("planned_rr")
    if planned_rr is not None:
        return float(planned_rr)
    preview_plan = dict(opportunity.get("preview_plan_json") or {})
    rr_candidates = []
    for key in ("planned_rr", "rr", "tp1_rr", "tp2_rr", "best_rr"):
        value = preview_plan.get(key)
        if value is None:
            continue
        try:
            rr_candidates.append(float(value))
        except (TypeError, ValueError):
            continue
    if rr_candidates:
        return max(rr_candidates)
    return None


def _opportunity_policy_allowance(
    policy_json: dict[str, Any],
    opportunity: dict[str, Any],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    signal = str(opportunity.get("signal") or "").upper()
    if signal not in {"BUY", "SELL"}:
        reasons.append(f"non_trade_signal:{signal or 'UNKNOWN'}")
        return False, reasons

    symbol = str(opportunity.get("symbol") or "").upper()
    setup_name = opportunity.get("setup")
    if symbol and not _is_symbol_enabled(policy_json, symbol):
        reasons.append(f"symbol_disabled:{symbol}")
    if not _is_setup_enabled(policy_json, setup_name):
        reasons.append(f"setup_disabled:{_setup_code_from_name(setup_name)}")

    best_rr = _best_preview_rr(opportunity)
    rr_threshold = _rr_threshold_for_setup(policy_json, setup_name)
    if best_rr is not None and rr_threshold > 0 and best_rr < rr_threshold:
        reasons.append(f"rr_below_threshold:{best_rr:.3f}<{rr_threshold:.3f}")

    preview_plan = dict(opportunity.get("preview_plan_json") or {})
    preview_status = str(preview_plan.get("status") or "").upper()
    if preview_status == "REJECTED" and not best_rr:
        reasons.append("preview_plan_rejected_without_rr_preview")

    return len(reasons) == 0, reasons


def _baseline_policy_allowed(opportunity: dict[str, Any]) -> bool:
    status = str(opportunity.get("status") or "").upper()
    return status in {
        "PLAN_READY",
        "EXECUTED",
        "DRY_RUN_EXECUTED",
        "EXECUTION_FAILED",
        "BINANCE_POSITION_EXISTS",
    }


def score_policy_patch_on_opportunities(
    active_policy_json: dict[str, Any],
    sanitized_patch: dict[str, Any],
    opportunity_rows: list[dict[str, Any]],
    *,
    closed_trade_by_id: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    merged_policy = _deep_merge(active_policy_json or {}, sanitized_patch or {})
    baseline_candidates = 0
    baseline_allowed = 0
    candidate_allowed = 0
    newly_allowed = 0
    newly_blocked = 0
    unscored_new_candidates = 0
    blocked_reason_counts: dict[str, int] = defaultdict(int)
    newly_allowed_examples: list[dict[str, Any]] = []
    newly_blocked_examples: list[dict[str, Any]] = []

    baseline_realized_rows: list[dict[str, Any]] = []
    candidate_realized_rows: list[dict[str, Any]] = []

    for row in opportunity_rows:
        signal = str(row.get("signal") or "").upper()
        if signal not in {"BUY", "SELL"}:
            continue
        baseline_candidates += 1
        baseline_ok = _baseline_policy_allowed(row)
        if baseline_ok:
            baseline_allowed += 1

        candidate_ok, reasons = _opportunity_policy_allowance(merged_policy, row)
        if candidate_ok:
            candidate_allowed += 1
        else:
            for reason in reasons:
                blocked_reason_counts[reason] += 1

        trade_id = row.get("linked_trade_id")
        trade_row = None
        if trade_id is not None and closed_trade_by_id is not None:
            trade_row = closed_trade_by_id.get(int(trade_id))

        if baseline_ok and not candidate_ok:
            newly_blocked += 1
            if len(newly_blocked_examples) < 10:
                newly_blocked_examples.append(
                    {
                        "opportunity_id": row.get("id"),
                        "symbol": row.get("symbol"),
                        "setup": row.get("setup"),
                        "trade_id": trade_id,
                        "reasons": reasons,
                    }
                )
        if candidate_ok and not baseline_ok:
            newly_allowed += 1
            if trade_row is None:
                unscored_new_candidates += 1
            if len(newly_allowed_examples) < 10:
                newly_allowed_examples.append(
                    {
                        "opportunity_id": row.get("id"),
                        "symbol": row.get("symbol"),
                        "setup": row.get("setup"),
                        "trade_id": trade_id,
                        "status": row.get("status"),
                    }
                )

        if baseline_ok and trade_row is not None:
            baseline_realized_rows.append(trade_row)
        if candidate_ok and trade_row is not None:
            candidate_realized_rows.append(trade_row)

    baseline_trade_summary = _summarize_trade_rows(baseline_realized_rows)
    candidate_trade_summary = _summarize_trade_rows(candidate_realized_rows)

    return {
        "method": "opportunity_snapshot_counterfactual",
        "changed_paths": _summarize_patch_paths(sanitized_patch),
        "baseline": {
            "candidate_rows": baseline_candidates,
            "policy_allowed_rows": baseline_allowed,
            "realized_trade_summary": baseline_trade_summary,
        },
        "candidate": {
            "policy_allowed_rows": candidate_allowed,
            "realized_trade_summary": candidate_trade_summary,
        },
        "deltas": {
            "candidate_rows": candidate_allowed - baseline_allowed,
            "trade_coverage_pct": round((candidate_allowed / baseline_allowed) * 100.0, 2)
            if baseline_allowed
            else 0.0,
            "total_pnl_usdt": _metric_delta(
                candidate_trade_summary.get("total_pnl_usdt"),
                baseline_trade_summary.get("total_pnl_usdt"),
            ),
            "win_rate_pct": _metric_delta(
                candidate_trade_summary.get("win_rate_pct"),
                baseline_trade_summary.get("win_rate_pct"),
                digits=2,
            ),
            "expectancy_usdt": _metric_delta(
                candidate_trade_summary.get("expectancy_usdt"),
                baseline_trade_summary.get("expectancy_usdt"),
            ),
        },
        "newly_allowed_candidates": int(newly_allowed),
        "newly_blocked_candidates": int(newly_blocked),
        "unscored_new_candidates": int(unscored_new_candidates),
        "blocked_reason_counts": dict(sorted(blocked_reason_counts.items())),
        "newly_allowed_examples": newly_allowed_examples,
        "newly_blocked_examples": newly_blocked_examples,
    }


def score_policy_review_candidate(
    review_input: dict[str, Any],
    sanitized_patch: dict[str, Any],
    validation_errors: list[str],
    risk_increase_changes: list[str],
    *,
    counterfactual: dict[str, Any] | None = None,
) -> dict[str, Any]:
    deterministic_checks = dict(review_input.get("deterministic_checks") or {})
    score = 100
    reasons: list[str] = []
    verdict = "ACCEPT"
    changed_paths = _summarize_patch_paths(sanitized_patch)

    if validation_errors:
        score = 0
        verdict = "REJECT"
        reasons.append("validator rejected one or more policy keys")
    if not changed_paths:
        score = min(score, 5)
        verdict = "REJECT"
        reasons.append("patch is empty after normalization")
    if not deterministic_checks.get("allow_llm_proposal", False):
        score -= 45
        reasons.append("deterministic maturity checks do not support a policy change")
    if len(changed_paths) > int((deterministic_checks.get("risk_constraints") or {}).get("max_changed_keys", 2)):
        score -= 15
        reasons.append("proposal changes too many keys for a conservative policy update")
    if risk_increase_changes and not deterministic_checks.get("clear_signal_for_risk_increase", False):
        score = min(score, 20)
        verdict = "REJECT"
        reasons.append("proposal attempts a risk increase without clear evidence")

    if counterfactual:
        deltas = dict(counterfactual.get("deltas") or {})
        pnl_delta = deltas.get("total_pnl_usdt")
        coverage = _to_float(deltas.get("trade_coverage_pct"), 100.0)
        if pnl_delta is not None:
            if float(pnl_delta) < 0:
                score -= 30
                reasons.append("counterfactual forward pnl is worse than baseline")
            elif float(pnl_delta) > 0:
                reasons.append("counterfactual forward pnl improves over baseline")
        if coverage < 40.0:
            score -= 10
            reasons.append("proposal removes too much trade coverage in counterfactual replay")

    score = max(0, min(100, score))
    if verdict != "REJECT":
        if score >= 75:
            verdict = "ACCEPT"
        elif score >= 50:
            verdict = "REVIEW"
        else:
            verdict = "REJECT"

    return {
        "verdict": verdict,
        "score": int(score),
        "changed_paths": changed_paths,
        "reasons": reasons,
    }


def _insert_policy_review_run(conn, payload: dict[str, Any]) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO policy_review_runs
              (engine_name, account_type, reviewer_model,
               guard_decision, guard_reason,
               llm_called, llm_decision, llm_reason,
               proposed_patch, guard_payload, llm_payload,
               active_policy_id, active_policy_version,
               hours_since_policy_update, closed_trades_since_update)
            VALUES
              (%s, %s, %s,
               %s, %s,
               %s, %s, %s,
               %s::jsonb, %s::jsonb, %s::jsonb,
               %s, %s,
               %s, %s)
            RETURNING id
            """,
            (
                payload["engine_name"],
                payload["account_type"],
                payload.get("reviewer_model"),
                payload["guard_decision"],
                payload["guard_reason"],
                bool(payload.get("llm_called", False)),
                payload.get("llm_decision"),
                payload.get("llm_reason"),
                Json(payload.get("proposed_patch") or {}, dumps=_json_dumps),
                Json(payload.get("guard_payload") or {}, dumps=_json_dumps),
                Json(payload.get("llm_payload") or {}, dumps=_json_dumps),
                payload.get("active_policy_id"),
                payload.get("active_policy_version"),
                payload.get("hours_since_policy_update"),
                payload.get("closed_trades_since_update"),
            ),
        )
        row = cur.fetchone()
    return int(row[0])


def _create_policy_version(
    conn,
    *,
    active_policy: dict[str, Any],
    merged_policy_json: dict[str, Any],
    validation_report: dict[str, Any],
    reason: str,
    reviewer_model: str,
    auto_activate: bool,
) -> tuple[int, int, str]:
    policy_name = str(active_policy.get("policy_name") or "default")
    engine_name = str(active_policy["engine_name"])
    account_type = str(active_policy["account_type"])

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COALESCE(MAX(version), 0)
            FROM strategy_policies
            WHERE policy_name = %s
              AND engine_name = %s
              AND account_type = %s
            """,
            (policy_name, engine_name, account_type),
        )
        next_version = int(cur.fetchone()[0]) + 1

        status = "active" if auto_activate else "validated"
        if auto_activate:
            cur.execute(
                """
                UPDATE strategy_policies
                SET status = 'retired', updated_at = NOW()
                WHERE engine_name = %s
                  AND account_type = %s
                  AND status = 'active'
                """,
                (engine_name, account_type),
            )

        cur.execute(
            """
            INSERT INTO strategy_policies
              (policy_name, engine_name, account_type, version, status,
               policy_json, validation_report, reason, source, created_by,
               base_policy_id, effective_from, activated_at)
            VALUES
              (%s, %s, %s, %s, %s,
               %s::jsonb, %s::jsonb, %s, %s, %s,
               %s, NOW(), %s)
            RETURNING id
            """,
            (
                policy_name,
                engine_name,
                account_type,
                next_version,
                status,
                Json(merged_policy_json, dumps=_json_dumps),
                Json(validation_report, dumps=_json_dumps),
                reason[:500],
                "llm_reviewer",
                reviewer_model[:120],
                int(active_policy["id"]),
                datetime.now(UTC) if auto_activate else None,
            ),
        )
        new_policy_id = int(cur.fetchone()[0])
    return new_policy_id, next_version, status


def run_policy_review_once(
    *,
    force_review: bool | None = None,
    force_review_reason: str | None = None,
    auto_apply_override: bool | None = None,
) -> dict[str, Any]:
    force_review = _resolve_force_review(force_review)
    force_review_reason = _resolve_force_review_reason(force_review_reason)
    auto_apply_override = _resolve_auto_apply_override(auto_apply_override)
    policy_review_enabled = _env_bool(os.getenv("POLICY_REVIEW_ENABLED", "1"), default=True)
    if not policy_review_enabled and not force_review:
        return {
            "status": "disabled",
            "message": "POLICY_REVIEW_ENABLED=0",
        }
    if force_review and not policy_review_enabled:
        log.warning("POLICY_REVIEW_FORCE_REVIEW bypassed POLICY_REVIEW_ENABLED=0")

    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError("DATABASE_URL is required")

    engine_name = os.getenv("STRATEGY_POLICY_ENGINE_NAME", "llm_live").strip()
    account_type = os.getenv("STRATEGY_POLICY_ACCOUNT_TYPE", "live").strip()
    trades_table = _safe_table_name(os.getenv("POLICY_REVIEW_TRADES_TABLE", "trades_live"))

    cfg = ReviewGuardConfig(
        min_hours_since_update=_to_float(
            os.getenv("POLICY_REVIEW_MIN_HOURS_SINCE_UPDATE", "24"),
            24.0,
        ),
        min_closed_trades_since_update=_to_int(
            os.getenv("POLICY_REVIEW_MIN_CLOSED_TRADES_SINCE_UPDATE", "20"),
            20,
        ),
        cooldown_hours_since_last_change=_to_float(
            os.getenv("POLICY_REVIEW_COOLDOWN_HOURS", "12"),
            12.0,
        ),
        trades_table=trades_table,
    )
    review_input, active_row, guard, cfg = build_policy_review_context(
        database_url=database_url,
        engine_name=engine_name,
        account_type=account_type,
        trades_table=trades_table,
        config=cfg,
        bypass_maturity_gate=force_review,
    )
    guard_payload = asdict(guard)
    manual_override = _build_manual_override_payload(
        force_review=force_review,
        force_review_reason=force_review_reason,
        auto_apply_override=auto_apply_override,
        policy_review_enabled=policy_review_enabled,
        guard_decision=guard.decision,
    )
    guard_payload["manual_override"] = manual_override
    guard_ctx = dict(guard_payload.get("context") or {})
    policy_timing = dict(guard_ctx.get("policy_timing") or {})
    post_update = dict(guard_ctx.get("post_update_sample") or {})
    active_policy_ctx = dict(review_input.get("active_policy") or guard_ctx.get("active_policy") or {})

    base_run_payload = {
        "engine_name": engine_name,
        "account_type": account_type,
        "reviewer_model": None,
        "guard_decision": guard.decision,
        "guard_reason": guard.reason,
        "llm_called": False,
        "llm_decision": "NO_CALL",
        "llm_reason": "",
        "proposed_patch": {},
        "guard_payload": guard_payload,
        "active_policy_id": active_policy_ctx.get("id"),
        "active_policy_version": active_policy_ctx.get("version"),
        "hours_since_policy_update": policy_timing.get("hours_since_update"),
        "closed_trades_since_update": post_update.get("closed_trades_since_update"),
        "llm_payload": {
            "manual_override": manual_override,
        },
    }

    conn = psycopg2.connect(database_url, sslmode="require")
    try:
        if not active_row:
            payload = dict(base_run_payload)
            payload["llm_reason"] = "No active policy found in DB scope"
            run_id = _insert_policy_review_run(conn, payload)
            conn.commit()
            return _attach_manual_override(
                {"status": "hold", "run_id": run_id, "reason": payload["llm_reason"]},
                manual_override,
            )

        if guard.decision != "ALLOW_REVIEW" and not force_review:
            payload = dict(base_run_payload)
            payload["llm_reason"] = guard.reason
            run_id = _insert_policy_review_run(conn, payload)
            conn.commit()
            return {"status": "hold", "run_id": run_id, "reason": guard.reason}

        if guard.decision != "ALLOW_REVIEW" and force_review:
            log.warning(
                "POLICY_REVIEW_FORCE_REVIEW bypassed guard decision=%s reason=%s",
                guard.decision,
                guard.reason,
            )

        try:
            reviewer_output = _call_llm_reviewer(review_input)
        except Exception as exc:
            payload = dict(base_run_payload)
            payload["reviewer_model"] = os.getenv("POLICY_REVIEW_MODEL", "unknown")
            payload["llm_called"] = False
            payload["llm_decision"] = "ERROR"
            payload["llm_reason"] = str(exc)[:500]
            payload["llm_payload"] = {
                "manual_override": manual_override,
                "error": str(exc),
                "review_input": review_input,
            }
            run_id = _insert_policy_review_run(conn, payload)
            conn.commit()
            return _attach_manual_override(
                {
                    "status": "error",
                    "run_id": run_id,
                    "reason": payload["llm_reason"],
                },
                manual_override,
            )
        meta = dict(reviewer_output.get("_meta") or {})
        reviewer_model = str(meta.get("model") or os.getenv("POLICY_REVIEW_MODEL", "unknown"))
        decision = str(reviewer_output.get("decision") or "NO_CHANGE").strip().upper()
        llm_reason = str(reviewer_output.get("reason") or "").strip()[:500]
        normalized_patch, patch_debug = extract_policy_patch_from_response(reviewer_output)

        payload = dict(base_run_payload)
        payload["reviewer_model"] = reviewer_model
        payload["llm_called"] = True
        payload["llm_decision"] = decision
        payload["llm_reason"] = llm_reason
        payload["proposed_patch"] = normalized_patch
        payload["llm_payload"] = {
            "manual_override": manual_override,
            "reviewer_output": reviewer_output,
            "patch_debug": patch_debug,
        }

        if decision != "PROPOSE_CHANGE":
            run_id = _insert_policy_review_run(conn, payload)
            conn.commit()
            return _attach_manual_override(
                {"status": "no_change", "run_id": run_id, "reason": llm_reason or "NO_CHANGE"},
                manual_override,
            )

        active_policy_json = dict(active_row.get("policy_json") or {})
        sanitized_patch, validation_errors, risk_increase_changes = validate_policy_patch(
            patch=normalized_patch,
            active_policy_json=active_policy_json,
        )

        deterministic_checks = dict(review_input.get("deterministic_checks") or {})
        clear_signal = bool(deterministic_checks.get("clear_signal_for_risk_increase", False))
        evidence = dict(deterministic_checks.get("clear_signal_evidence") or {})

        stripped_risk_changes: list[str] = []
        if risk_increase_changes and not clear_signal:
            sanitized_patch, stripped_risk_changes = strip_risk_increase_changes(
                sanitized_patch=sanitized_patch,
                active_policy_json=active_policy_json,
            )
        effective_risk_increase_changes = [
            change for change in risk_increase_changes if change not in stripped_risk_changes
        ]

        proposal_score = score_policy_review_candidate(
            review_input,
            sanitized_patch,
            validation_errors,
            effective_risk_increase_changes,
        )

        if validation_errors or _is_patch_empty(sanitized_patch) or proposal_score["verdict"] == "REJECT":
            payload["llm_decision"] = "REJECTED_VALIDATION"
            notes = {
                "validation_errors": validation_errors,
                "risk_increase_changes": risk_increase_changes,
                "effective_risk_increase_changes": effective_risk_increase_changes,
                "risk_increase_stripped": stripped_risk_changes,
                "clear_signal_for_risk_increase": clear_signal,
                "clear_signal_evidence": evidence,
                "proposal_score": proposal_score,
            }
            payload["llm_reason"] = (
                "No effective validated patch"
                if not validation_errors and proposal_score["verdict"] != "REJECT"
                else ("; ".join(validation_errors or proposal_score.get("reasons") or []))[:500]
            )
            payload["proposed_patch"] = sanitized_patch
            payload["llm_payload"] = {
                "manual_override": manual_override,
                "reviewer_output": reviewer_output,
                "patch_debug": patch_debug,
                "validation": notes,
            }
            run_id = _insert_policy_review_run(conn, payload)
            conn.commit()
            return _attach_manual_override(
                {
                    "status": "rejected",
                    "run_id": run_id,
                    "reason": payload["llm_reason"],
                    "validation": notes,
                },
                manual_override,
            )

        merged_policy = _deep_merge(active_policy_json, sanitized_patch)
        auto_apply = (
            bool(auto_apply_override)
            if auto_apply_override is not None
            else _env_bool(os.getenv("POLICY_REVIEW_AUTO_APPLY", "0"), default=False)
        )
        allow_risk_increase = bool(risk_increase_changes) and clear_signal
        validation_report = {
            "review_type": "llm_policy_review",
            "allow_risk_increase": bool(allow_risk_increase),
            "clear_signal": bool(clear_signal),
            "evidence_trades": int(evidence.get("closed_trades_since_update", 0)),
            "risk_increase_changes": risk_increase_changes,
            "effective_risk_increase_changes": effective_risk_increase_changes,
            "risk_increase_stripped": stripped_risk_changes,
            "validation_errors": validation_errors,
            "proposal_score": proposal_score,
            "changed_paths": _summarize_patch_paths(sanitized_patch),
            "reviewer_model": reviewer_model,
            "reviewed_at": datetime.now(UTC),
            "guard_decision": guard.decision,
            "guard_reason": guard.reason,
            "guard_config": guard.guard_config,
            "manual_override": manual_override,
        }
        new_policy_id, new_policy_version, new_status = _create_policy_version(
            conn,
            active_policy=active_row,
            merged_policy_json=merged_policy,
            validation_report=validation_report,
            reason=llm_reason or "LLM policy review proposal",
            reviewer_model=reviewer_model,
            auto_activate=auto_apply,
        )

        payload["llm_decision"] = "APPLIED" if auto_apply else "VALIDATED"
        payload["llm_reason"] = (
            f"policy_id={new_policy_id} version={new_policy_version} status={new_status}"
        )
        payload["proposed_patch"] = sanitized_patch
        payload["llm_payload"] = {
            "manual_override": manual_override,
            "reviewer_output": reviewer_output,
            "patch_debug": patch_debug,
            "validation": {
                "validation_errors": validation_errors,
                "risk_increase_changes": risk_increase_changes,
                "effective_risk_increase_changes": effective_risk_increase_changes,
                "risk_increase_stripped": stripped_risk_changes,
                "clear_signal_for_risk_increase": clear_signal,
                "clear_signal_evidence": evidence,
                "proposal_score": proposal_score,
            },
            "new_policy": {
                "id": new_policy_id,
                "version": new_policy_version,
                "status": new_status,
                "auto_applied": auto_apply,
            },
        }

        run_id = _insert_policy_review_run(conn, payload)
        conn.commit()
        return _attach_manual_override(
            {
                "status": "applied" if auto_apply else "validated",
                "run_id": run_id,
                "policy_id": new_policy_id,
                "policy_version": new_policy_version,
                "policy_status": new_status,
                "reason": payload["llm_reason"],
            },
            manual_override,
        )

    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Run one guarded strategy policy review")
    parser.add_argument("--json-only", action="store_true")
    parser.add_argument(
        "--force-review",
        action="store_true",
        help=(
            "Manually bypass POLICY_REVIEW_ENABLED and the maturity/cooldown guard. "
            "An active policy and normal validation checks still apply."
        ),
    )
    parser.add_argument(
        "--force-review-reason",
        default="",
        help="Optional audit note for a manual override, e.g. telegram:/review",
    )
    parser.add_argument(
        "--force-apply",
        action="store_true",
        help="Auto-activate a validated policy for this run, regardless of POLICY_REVIEW_AUTO_APPLY.",
    )
    args = parser.parse_args()

    result = run_policy_review_once(
        force_review=True if args.force_review else None,
        force_review_reason=args.force_review_reason or None,
        auto_apply_override=True if args.force_apply else None,
    )
    output = _json_dumps(result)
    if not args.json_only:
        log.info(f"policy review result: {output}")
    print(output)


if __name__ == "__main__":
    main()
