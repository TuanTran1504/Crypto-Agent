#!/usr/bin/env python3
"""
Walk-forward evaluator for policy review proposals.

This script:
  - picks historical anchor timestamps
  - rebuilds the review context as-of each anchor
  - runs the shadow multi-agent reviewer in read-only mode
  - scores the candidate patch on the next window of closed trades

The forward scorer is a trade-filter counterfactual, not a full market replay.
It is still useful because it tells us whether a proposal would have removed
mostly losing trades, mostly winning trades, or little at all.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor


UTC = timezone.utc
ROOT = Path(__file__).resolve().parent.parent.parent
SCHEDULE_DIR = ROOT / "backend" / "schedule"
TESTING_DIR = ROOT / "backend" / "testing"
if str(SCHEDULE_DIR) not in sys.path:
    sys.path.insert(0, str(SCHEDULE_DIR))
if str(TESTING_DIR) not in sys.path:
    sys.path.insert(0, str(TESTING_DIR))

import run_policy_review as prod  # noqa: E402
import run_policy_review_shadow_multi_agent as shadow  # noqa: E402

load_dotenv(ROOT / ".env")


def _json_default(value: Any):
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC).isoformat()
    return str(value)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, default=_json_default)


def _parse_as_of(raw: str | None) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _print_step(enabled: bool, title: str, payload: Any | None = None):
    if not enabled:
        return
    ts = datetime.now(tz=UTC).isoformat(timespec="seconds")
    print(f"[{ts}] {title}", file=sys.stderr)
    if payload is not None:
        pretty = json.dumps(payload, ensure_ascii=True, default=_json_default, indent=2)
        for line in pretty.splitlines():
            print(f"  {line}", file=sys.stderr)


def _fetch_policy_change_anchors(
    conn,
    engine_name: str,
    account_type: str,
    *,
    max_anchors: int,
    start_at: datetime | None = None,
    end_at: datetime | None = None,
) -> list[dict[str, Any]]:
    where_clauses = [
        "engine_name = %s",
        "account_type = %s",
        "COALESCE(activated_at, effective_from, created_at) IS NOT NULL",
    ]
    params: list[Any] = [engine_name, account_type]
    if start_at is not None:
        where_clauses.append("COALESCE(activated_at, effective_from, created_at) >= %s")
        params.append(start_at)
    if end_at is not None:
        where_clauses.append("COALESCE(activated_at, effective_from, created_at) <= %s")
        params.append(end_at)

    query = f"""
        SELECT
            id,
            version,
            status,
            COALESCE(activated_at, effective_from, created_at) AS anchor_at,
            created_at,
            effective_from,
            activated_at
        FROM strategy_policies
        WHERE {' AND '.join(where_clauses)}
        ORDER BY anchor_at DESC
        LIMIT %s
    """
    params.append(int(max_anchors))
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(query, tuple(params))
        return [dict(row) for row in cur.fetchall()]


def _fetch_forward_trades(
    conn,
    trades_table: str,
    account_type: str,
    *,
    anchor_at: datetime,
    forward_trades: int,
    forward_days: int,
) -> list[dict[str, Any]]:
    until_dt = anchor_at + timedelta(days=max(1, forward_days))
    return prod._fetch_closed_trade_rows(
        conn,
        trades_table,
        account_type,
        since_dt=anchor_at + timedelta(microseconds=1),
        until_dt=until_dt,
        limit_rows=forward_trades,
        descending=False,
    )


def _fetch_forward_opportunities(
    conn,
    opportunities_table: str,
    engine_name: str,
    account_type: str,
    *,
    anchor_at: datetime,
    forward_days: int,
    limit_rows: int,
) -> list[dict[str, Any]]:
    until_dt = anchor_at + timedelta(days=max(1, forward_days))
    return prod._fetch_opportunity_rows(
        conn,
        opportunities_table,
        account_type,
        engine_name=engine_name,
        since_dt=anchor_at + timedelta(microseconds=1),
        until_dt=until_dt,
        limit_rows=limit_rows,
    )


def run_walkforward_eval(
    *,
    verbose_steps: bool = True,
    max_anchors: int = 5,
    forward_trades: int = 30,
    forward_days: int = 7,
    bypass_maturity_gate: bool = False,
    start_at: datetime | None = None,
    end_at: datetime | None = None,
) -> dict[str, Any]:
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError("DATABASE_URL is required")

    engine_name = os.getenv("STRATEGY_POLICY_ENGINE_NAME", "llm_live").strip()
    account_type = os.getenv("STRATEGY_POLICY_ACCOUNT_TYPE", "live").strip()
    trades_table = prod._safe_table_name(os.getenv("POLICY_REVIEW_TRADES_TABLE", "trades_live"))
    opportunities_table = prod._safe_table_name(
        os.getenv("POLICY_REVIEW_OPPORTUNITIES_TABLE", "strategy_opportunities_live"),
        "strategy_opportunities_live",
    )

    conn = psycopg2.connect(database_url, sslmode="require")
    try:
        anchors = _fetch_policy_change_anchors(
            conn,
            engine_name,
            account_type,
            max_anchors=max_anchors,
            start_at=start_at,
            end_at=end_at,
        )
    finally:
        conn.close()

    _print_step(
        verbose_steps,
        "Walk-forward anchors selected",
        {
            "engine_name": engine_name,
            "account_type": account_type,
            "anchor_count": len(anchors),
            "forward_trades": forward_trades,
            "forward_days": forward_days,
            "bypass_maturity_gate": bypass_maturity_gate,
        },
    )

    runs: list[dict[str, Any]] = []
    for anchor in anchors:
        anchor_at = anchor.get("anchor_at")
        if anchor_at is None:
            continue
        anchor_at = anchor_at.astimezone(UTC) if anchor_at.tzinfo else anchor_at.replace(tzinfo=UTC)
        _print_step(
            verbose_steps,
            "Evaluate anchor",
            {
                "policy_id": anchor.get("id"),
                "policy_version": anchor.get("version"),
                "anchor_at": anchor_at,
            },
        )

        shadow_result = shadow.run_multi_agent_shadow_preview(
            verbose_steps=False,
            as_of=anchor_at,
            bypass_maturity_gate=bypass_maturity_gate,
        )
        run_item: dict[str, Any] = {
            "anchor": anchor,
            "shadow_result": shadow_result,
        }

        if shadow_result.get("status") != "ok":
            runs.append(run_item)
            continue

        active_policy = dict((shadow_result.get("review_input") or {}).get("active_policy") or {})
        active_policy_json = dict(active_policy.get("policy_json") or {})
        sanitized_patch = dict(shadow_result.get("candidate_patch_sanitized") or {})
        validation_errors = list(shadow_result.get("validation_errors") or [])
        risk_increase_changes = list(shadow_result.get("effective_risk_increase_changes") or [])

        conn = psycopg2.connect(database_url, sslmode="require")
        try:
            forward_rows = _fetch_forward_trades(
                conn,
                trades_table,
                account_type,
                anchor_at=anchor_at,
                forward_trades=forward_trades,
                forward_days=forward_days,
            )
            try:
                forward_opportunities = _fetch_forward_opportunities(
                    conn,
                    opportunities_table,
                    engine_name,
                    account_type,
                    anchor_at=anchor_at,
                    forward_days=forward_days,
                    limit_rows=max(forward_trades * 8, 100),
                )
            except Exception:
                forward_opportunities = []
        finally:
            conn.close()

        run_item["forward_trade_count"] = len(forward_rows)
        run_item["forward_opportunity_count"] = len(forward_opportunities)
        if shadow_result.get("final_decision") == "PROPOSE_CHANGE" and sanitized_patch:
            forward_score = (
                prod.score_policy_patch_on_trades(
                    active_policy_json,
                    sanitized_patch,
                    forward_rows,
                )
                if forward_rows
                else None
            )
            closed_trade_by_id = {
                int(row["id"]): row
                for row in forward_rows
                if row.get("id") is not None
            }
            opportunity_score = (
                prod.score_policy_patch_on_opportunities(
                    active_policy_json,
                    sanitized_patch,
                    forward_opportunities,
                    closed_trade_by_id=closed_trade_by_id,
                )
                if forward_opportunities
                else None
            )
            counterfactual = opportunity_score or forward_score
            proposal_score = prod.score_policy_review_candidate(
                dict(shadow_result.get("review_input") or {}),
                sanitized_patch,
                validation_errors,
                risk_increase_changes,
                counterfactual=counterfactual,
            )
        else:
            if shadow_result.get("final_decision") != "PROPOSE_CHANGE":
                verdict = "NO_EFFECTIVE_PROPOSAL"
                score_value = 100
            elif not sanitized_patch:
                verdict = "NO_PATCH"
                score_value = 100
            elif not forward_rows and not forward_opportunities:
                verdict = "NO_FORWARD_EVIDENCE"
                score_value = 0
            else:
                verdict = "NO_FORWARD_TRADES"
                score_value = 0
            forward_score = None
            opportunity_score = None
            proposal_score = {
                "verdict": verdict,
                "score": score_value,
                "changed_paths": prod._summarize_patch_paths(sanitized_patch),
                "reasons": ["no effective proposed patch or no forward evidence available"],
            }

        run_item["forward_score"] = forward_score
        run_item["opportunity_score"] = opportunity_score
        run_item["walkforward_score"] = proposal_score
        runs.append(run_item)

    improving = 0
    harmful = 0
    proposed = 0
    for run in runs:
        shadow_result = dict(run.get("shadow_result") or {})
        if shadow_result.get("final_decision") == "PROPOSE_CHANGE":
            proposed += 1
        counterfactual = dict(run.get("opportunity_score") or run.get("forward_score") or {})
        pnl_delta = dict(counterfactual.get("deltas") or {}).get("total_pnl_usdt")
        if pnl_delta is None:
            continue
        if float(pnl_delta) > 0:
            improving += 1
        elif float(pnl_delta) < 0:
            harmful += 1

    return {
        "status": "ok",
        "mode": "walkforward_eval",
        "engine_name": engine_name,
        "account_type": account_type,
        "summary": {
            "anchors_evaluated": len(runs),
            "proposed_changes": proposed,
            "improving_counterfactuals": improving,
            "harmful_counterfactuals": harmful,
            "forward_trades": forward_trades,
            "forward_days": forward_days,
        },
        "runs": runs,
    }


def main():
    parser = argparse.ArgumentParser(description="Run walk-forward policy review evaluation")
    parser.add_argument("--json-only", action="store_true")
    parser.add_argument("--quiet-steps", action="store_true")
    parser.add_argument("--max-anchors", type=int, default=5)
    parser.add_argument("--forward-trades", type=int, default=30)
    parser.add_argument("--forward-days", type=int, default=7)
    parser.add_argument("--bypass-maturity-gate", action="store_true")
    parser.add_argument("--start-at", type=str, default="")
    parser.add_argument("--end-at", type=str, default="")
    args = parser.parse_args()

    result = run_walkforward_eval(
        verbose_steps=not args.quiet_steps,
        max_anchors=max(1, args.max_anchors),
        forward_trades=max(1, args.forward_trades),
        forward_days=max(1, args.forward_days),
        bypass_maturity_gate=args.bypass_maturity_gate,
        start_at=_parse_as_of(args.start_at),
        end_at=_parse_as_of(args.end_at),
    )
    print(_json_dumps(result))


if __name__ == "__main__":
    main()
