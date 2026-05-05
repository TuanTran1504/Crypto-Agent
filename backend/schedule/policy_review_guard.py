from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import psycopg2
from psycopg2.extras import RealDictCursor

UTC = timezone.utc


@dataclass
class ReviewGuardConfig:
    min_hours_since_update: float = 24.0
    min_closed_trades_since_update: int = 20
    cooldown_hours_since_last_change: float = 12.0
    trades_table: str = "trades_live"


@dataclass
class TradeWindowSummary:
    closed_trades: int
    wins: int
    win_rate_pct: float
    total_pnl_usdt: float


@dataclass
class ReviewGuardResult:
    decision: str  # HOLD | ALLOW_REVIEW
    reason: str
    guard_config: dict[str, Any]
    context: dict[str, Any]


def _to_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _utc_now(now_dt: datetime | None = None) -> datetime:
    return _to_utc(now_dt) or datetime.now(UTC)


def _hours_between(now_dt: datetime, then_dt: datetime | None) -> float:
    if then_dt is None:
        return 1e9
    return max(0.0, (now_dt - then_dt).total_seconds() / 3600.0)


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


def _fetch_active_policy(
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
            SELECT id, policy_name, version, status, policy_json, validation_report,
                   source, created_at, updated_at, effective_from, activated_at
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


def _fetch_recent_policy_changes(
    conn,
    engine_name: str,
    account_type: str,
    *,
    as_of: datetime | None = None,
    limit_rows: int = 3,
) -> list[dict[str, Any]]:
    bound_dt = _utc_now(as_of)
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT id, policy_name, version, status, source, reason,
                   created_at, updated_at, effective_from, activated_at
            FROM strategy_policies
            WHERE engine_name = %s
              AND account_type = %s
              AND created_at <= %s
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (engine_name, account_type, bound_dt, limit_rows),
        )
        return [dict(r) for r in cur.fetchall()]


def _fetch_latest_policy_change_time(
    conn,
    engine_name: str,
    account_type: str,
    *,
    as_of: datetime | None = None,
) -> datetime | None:
    bound_dt = _utc_now(as_of)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT MAX(created_at)
            FROM strategy_policies
            WHERE engine_name = %s
              AND account_type = %s
              AND created_at <= %s
            """,
            (engine_name, account_type, bound_dt),
        )
        row = cur.fetchone()
        ts = row[0] if row else None
        return _to_utc(ts)


def _fetch_open_positions_count(
    conn,
    trades_table: str,
    account_type: str,
    *,
    as_of: datetime | None = None,
) -> int:
    with conn.cursor() as cur:
        if as_of is None:
            cur.execute(
                f"""
                SELECT COUNT(*)
                FROM {trades_table}
                WHERE account_type = %s
                  AND status = 'OPEN'
                """,
                (account_type,),
            )
        else:
            bound_dt = _utc_now(as_of)
            cur.execute(
                f"""
                SELECT COUNT(*)
                FROM {trades_table}
                WHERE account_type = %s
                  AND opened_at <= %s
                  AND (closed_at IS NULL OR closed_at > %s)
                """,
                (account_type, bound_dt, bound_dt),
            )
        row = cur.fetchone()
        return int(row[0] if row else 0)


def _fetch_closed_trades_since_update(
    conn,
    trades_table: str,
    account_type: str,
    policy_id: int,
    policy_started_at: datetime | None,
    *,
    as_of: datetime | None = None,
) -> int:
    has_policy_id = _has_column(conn, trades_table, "strategy_policy_id")
    params: list[Any]
    with conn.cursor() as cur:
        if has_policy_id:
            query = f"""
                SELECT COUNT(*)
                FROM {trades_table}
                WHERE account_type = %s
                  AND status = 'CLOSED'
                  AND strategy_policy_id = %s
            """
            params = [account_type, policy_id]
            if policy_started_at is not None:
                query += " AND closed_at >= %s"
                params.append(policy_started_at)
            if as_of is not None:
                query += " AND closed_at <= %s"
                params.append(_utc_now(as_of))
            cur.execute(query, tuple(params))
        else:
            query = f"""
                SELECT COUNT(*)
                FROM {trades_table}
                WHERE account_type = %s
                  AND status = 'CLOSED'
                  AND closed_at >= %s
            """
            params = [account_type, policy_started_at]
            if as_of is not None:
                query += " AND closed_at <= %s"
                params.append(_utc_now(as_of))
            cur.execute(query, tuple(params))
        row = cur.fetchone()
        return int(row[0] if row else 0)


def _fetch_window_summary(
    conn,
    trades_table: str,
    account_type: str,
    *,
    since_dt: datetime | None = None,
    as_of: datetime | None = None,
    limit_rows: int | None = None,
) -> TradeWindowSummary:
    where_clauses = [
        "account_type = %s",
        "status = 'CLOSED'",
        "pnl_usdt IS NOT NULL",
        "closed_at IS NOT NULL",
    ]
    params: list[Any] = [account_type]
    if since_dt is not None:
        where_clauses.append("closed_at >= %s")
        params.append(since_dt)
    if as_of is not None:
        where_clauses.append("closed_at <= %s")
        params.append(_utc_now(as_of))

    base_query = f"""
        SELECT pnl_usdt
        FROM {trades_table}
        WHERE {' AND '.join(where_clauses)}
        ORDER BY closed_at DESC
    """
    if limit_rows is not None:
        base_query += " LIMIT %s"
        params.append(limit_rows)

    with conn.cursor() as cur:
        cur.execute(base_query, tuple(params))
        rows = cur.fetchall()

    pnl_values = [float(r[0]) for r in rows]
    closed = len(pnl_values)
    wins = sum(1 for pnl in pnl_values if pnl > 0)
    win_rate = (wins / closed * 100.0) if closed else 0.0
    total_pnl = float(sum(pnl_values)) if pnl_values else 0.0
    return TradeWindowSummary(
        closed_trades=closed,
        wins=wins,
        win_rate_pct=round(win_rate, 2),
        total_pnl_usdt=round(total_pnl, 4),
    )


def evaluate_policy_review_guard(
    database_url: str,
    engine_name: str,
    account_type: str,
    config: ReviewGuardConfig | None = None,
    *,
    as_of: datetime | None = None,
) -> ReviewGuardResult:
    cfg = config or ReviewGuardConfig()
    now_dt = _utc_now(as_of)

    conn = psycopg2.connect(database_url, sslmode="require")
    try:
        active = _fetch_active_policy(conn, engine_name, account_type, as_of=now_dt)
        if not active:
            return ReviewGuardResult(
                decision="HOLD",
                reason="No active strategy policy in DB scope",
                guard_config=asdict(cfg),
                context={
                    "engine_name": engine_name,
                    "account_type": account_type,
                    "as_of": now_dt,
                    "active_policy": None,
                },
            )

        policy_started_at = _to_utc(
            active.get("activated_at")
            or active.get("effective_from")
            or active.get("created_at")
        )
        hours_since_update = _hours_between(now_dt, policy_started_at)
        latest_change_at = _fetch_latest_policy_change_time(
            conn,
            engine_name,
            account_type,
            as_of=now_dt,
        )
        hours_since_last_change = _hours_between(now_dt, latest_change_at)
        closed_since_update = _fetch_closed_trades_since_update(
            conn,
            cfg.trades_table,
            account_type,
            int(active["id"]),
            policy_started_at,
            as_of=now_dt,
        )
        open_positions = _fetch_open_positions_count(
            conn,
            cfg.trades_table,
            account_type,
            as_of=as_of,
        )

        last_50 = _fetch_window_summary(
            conn,
            cfg.trades_table,
            account_type,
            as_of=now_dt,
            limit_rows=50,
        )
        last_3d = _fetch_window_summary(
            conn,
            cfg.trades_table,
            account_type,
            since_dt=now_dt - timedelta(days=3),
            as_of=now_dt,
        )
        last_7d = _fetch_window_summary(
            conn,
            cfg.trades_table,
            account_type,
            since_dt=now_dt - timedelta(days=7),
            as_of=now_dt,
        )
        recent_policy_changes = _fetch_recent_policy_changes(
            conn,
            engine_name,
            account_type,
            as_of=now_dt,
            limit_rows=3,
        )

        context = {
            "engine_name": engine_name,
            "account_type": account_type,
            "as_of": now_dt,
            "active_policy": {
                "id": int(active["id"]),
                "policy_name": active.get("policy_name"),
                "version": int(active.get("version") or 0),
                "status": active.get("status"),
                "source": active.get("source"),
                "created_at": active.get("created_at"),
                "updated_at": active.get("updated_at"),
                "effective_from": active.get("effective_from"),
                "activated_at": active.get("activated_at"),
            },
            "policy_timing": {
                "policy_started_at": policy_started_at,
                "hours_since_update": round(hours_since_update, 2),
                "latest_policy_change_at": latest_change_at,
                "hours_since_last_policy_change": round(hours_since_last_change, 2),
            },
            "post_update_sample": {
                "closed_trades_since_update": int(closed_since_update),
                "open_positions_count": int(open_positions),
            },
            "performance_windows": {
                "last_50_trades": asdict(last_50),
                "last_3_days": asdict(last_3d),
                "last_7_days": asdict(last_7d),
            },
            "recent_policy_changes": recent_policy_changes,
        }

        hold_reasons: list[str] = []
        if hours_since_update < cfg.min_hours_since_update:
            hold_reasons.append(
                f"hours_since_update={hours_since_update:.2f} < {cfg.min_hours_since_update:.2f}"
            )
        if closed_since_update < cfg.min_closed_trades_since_update:
            hold_reasons.append(
                f"closed_trades_since_update={closed_since_update} < {cfg.min_closed_trades_since_update}"
            )
        if hours_since_last_change < cfg.cooldown_hours_since_last_change:
            hold_reasons.append(
                "hours_since_last_policy_change="
                f"{hours_since_last_change:.2f} < {cfg.cooldown_hours_since_last_change:.2f}"
            )

        if hold_reasons:
            return ReviewGuardResult(
                decision="HOLD",
                reason="; ".join(hold_reasons),
                guard_config=asdict(cfg),
                context=context,
            )

        return ReviewGuardResult(
            decision="ALLOW_REVIEW",
            reason="All maturity gates passed",
            guard_config=asdict(cfg),
            context=context,
        )
    finally:
        conn.close()
