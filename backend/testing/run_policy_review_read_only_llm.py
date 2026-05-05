#!/usr/bin/env python3
"""
Read-only LLM policy-review preview.

Purpose:
  - Build the same shared review context used by production and shadow review
  - Optionally bypass maturity gating for testing
  - Call the single LLM reviewer
  - Apply shared patch normalization, validation, and proposal scoring
  - Print JSON result only

Safety:
  - NEVER writes to DB
  - NEVER changes strategy_policies
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


UTC = timezone.utc
ROOT = Path(__file__).resolve().parent.parent.parent
SCHEDULE_DIR = ROOT / "backend" / "schedule"
if str(SCHEDULE_DIR) not in sys.path:
    sys.path.insert(0, str(SCHEDULE_DIR))

import run_policy_review as prod  # noqa: E402
from policy_review_guard import ReviewGuardConfig  # noqa: E402

load_dotenv(ROOT / ".env")


def _json_default(value: Any):
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC).isoformat()
    return str(value)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, default=_json_default)


def _env_bool(value: str | bool | int | None, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


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


def _parse_as_of(raw: str | None) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def run_read_only_llm_preview(*, as_of: datetime | None = None) -> dict[str, Any]:
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError("DATABASE_URL is required")

    engine_name = os.getenv("STRATEGY_POLICY_ENGINE_NAME", "llm_live").strip()
    account_type = os.getenv("STRATEGY_POLICY_ACCOUNT_TYPE", "live").strip()
    trades_table = prod._safe_table_name(os.getenv("POLICY_REVIEW_TRADES_TABLE", "trades_live"))
    bypass_maturity_gate = _env_bool(
        os.getenv("POLICY_REVIEW_PREVIEW_BYPASS_TIME", "1"),
        default=True,
    )

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
    review_input, active_row, guard, cfg = prod.build_policy_review_context(
        database_url=database_url,
        engine_name=engine_name,
        account_type=account_type,
        trades_table=trades_table,
        config=cfg,
        as_of=as_of,
        include_latest_policy_fallback=True,
        bypass_maturity_gate=bypass_maturity_gate,
    )

    if not active_row:
        return {
            "status": "no_policy",
            "mode": "read_only_llm",
            "engine_name": engine_name,
            "account_type": account_type,
            "guard_decision": guard.decision,
            "guard_reason": guard.reason,
            "bypass_maturity_gate": bypass_maturity_gate,
            "review_input": review_input,
            "reason": "No strategy policy rows found in DB scope",
        }

    reviewer_output = prod._call_llm_reviewer(review_input)
    meta = dict(reviewer_output.get("_meta") or {})
    reviewer_model = str(meta.get("model") or os.getenv("POLICY_REVIEW_MODEL", "unknown"))
    decision = str(reviewer_output.get("decision") or "NO_CHANGE").strip().upper()
    llm_reason = str(reviewer_output.get("reason") or "").strip()[:500]
    normalized_patch, patch_debug = prod.extract_policy_patch_from_response(reviewer_output)

    active_policy_json = dict(active_row.get("policy_json") or {})
    sanitized_patch, validation_errors, risk_increase_changes = prod.validate_policy_patch(
        patch=normalized_patch,
        active_policy_json=active_policy_json,
    )

    deterministic_checks = dict(review_input.get("deterministic_checks") or {})
    clear_signal = bool(deterministic_checks.get("clear_signal_for_risk_increase", False))
    clear_signal_evidence = dict(deterministic_checks.get("clear_signal_evidence") or {})
    stripped_risk_changes: list[str] = []
    if risk_increase_changes and not clear_signal:
        sanitized_patch, stripped_risk_changes = prod.strip_risk_increase_changes(
            sanitized_patch=sanitized_patch,
            active_policy_json=active_policy_json,
        )
    effective_risk_increase_changes = [
        change for change in risk_increase_changes if change not in stripped_risk_changes
    ]
    proposal_score = prod.score_policy_review_candidate(
        review_input,
        sanitized_patch,
        validation_errors,
        effective_risk_increase_changes,
    )
    merged_preview = prod._deep_merge(active_policy_json, sanitized_patch)

    final_decision = "NO_CHANGE"
    if (
        decision == "PROPOSE_CHANGE"
        and not validation_errors
        and not prod._is_patch_empty(sanitized_patch)
        and proposal_score.get("verdict") != "REJECT"
        and deterministic_checks.get("allow_llm_proposal", False)
    ):
        final_decision = "PROPOSE_CHANGE"

    return {
        "status": "ok",
        "mode": "read_only_llm",
        "guard_decision": guard.decision,
        "guard_reason": guard.reason,
        "bypass_maturity_gate": bypass_maturity_gate,
        "reviewer_model": reviewer_model,
        "llm_decision": decision,
        "llm_reason": llm_reason,
        "active_policy_id": int(active_row["id"]),
        "active_policy_version": int(active_row.get("version") or 0),
        "active_policy_status": str(active_row.get("status") or ""),
        "review_input": review_input,
        "proposed_patch_raw": normalized_patch,
        "patch_debug": patch_debug,
        "proposed_patch_sanitized": sanitized_patch,
        "validation_errors": validation_errors,
        "risk_increase_changes": risk_increase_changes,
        "effective_risk_increase_changes": effective_risk_increase_changes,
        "risk_increase_stripped": stripped_risk_changes,
        "clear_signal_for_risk_increase": clear_signal,
        "clear_signal_evidence": clear_signal_evidence,
        "proposal_score": proposal_score,
        "merged_policy_preview": merged_preview,
        "final_decision": final_decision,
        "reviewer_output": reviewer_output,
        "notes": "Read-only preview only. No DB writes were performed.",
    }


def main():
    parser = argparse.ArgumentParser(description="Run read-only LLM policy review preview")
    parser.add_argument("--json-only", action="store_true")
    parser.add_argument("--as-of", type=str, default="")
    args = parser.parse_args()

    result = run_read_only_llm_preview(as_of=_parse_as_of(args.as_of))
    print(_json_dumps(result))


if __name__ == "__main__":
    main()
