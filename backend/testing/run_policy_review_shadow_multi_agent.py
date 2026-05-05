#!/usr/bin/env python3
"""
Multi-agent shadow policy review (read-only).

This script is for testing only:
  - Reads DB context through the same shared builder used by production
  - Runs a simplified analyst -> proposer -> critic workflow
  - Applies deterministic validation, risk stripping, and proposal scoring
  - Prints JSON output

Safety:
  - No DB writes
  - No strategy policy activation
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
BACKEND_DIR = ROOT / "backend"
SCHEDULE_DIR = ROOT / "backend" / "schedule"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(SCHEDULE_DIR) not in sys.path:
    sys.path.insert(0, str(SCHEDULE_DIR))

from observability import (  # noqa: E402
    JsonlTraceSink,
    LocalBlobStore,
    PostgresTraceSink,
    RedactionRule,
    TraceRedactor,
    TraceRun,
    Tracer,
    artifact,
    record_artifacts,
    trace_llm_call,
    traced_step,
)
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


def _print_step(
    *,
    enabled: bool,
    title: str,
    description: str | None = None,
    payload: Any | None = None,
) -> None:
    if not enabled:
        return
    ts = datetime.now(tz=UTC).isoformat(timespec="seconds")
    print(f"[{ts}] {title}", file=sys.stderr)
    if description:
        print(f"  {description}", file=sys.stderr)
    if payload is not None:
        pretty = json.dumps(payload, ensure_ascii=True, default=_json_default, indent=2)
        print("  details:", file=sys.stderr)
        for line in pretty.splitlines():
            print(f"    {line}", file=sys.stderr)


def _default_trace_blob_dir(trace_output_path: str | None) -> str | None:
    if not trace_output_path:
        return None
    output_path = Path(trace_output_path)
    return str(output_path.parent / f"{output_path.stem}_blobs")


def _build_trace_run(
    *,
    trace_output_path: str | None,
    trace_blob_dir: str | None,
    trace_database_url: str | None,
    engine_name: str,
    account_type: str,
    trades_table: str,
    bypass_maturity_gate: bool,
    as_of: datetime | None,
) -> tuple[Tracer, TraceRun, str | None]:
    output_path = str(Path(trace_output_path).expanduser()) if trace_output_path else None
    resolved_blob_dir = (
        str(Path(trace_blob_dir).expanduser())
        if trace_blob_dir
        else _default_trace_blob_dir(output_path)
    )
    sinks = [JsonlTraceSink(output_path)] if output_path else []
    if trace_database_url:
        sinks.append(PostgresTraceSink(trace_database_url))
    blob_store = LocalBlobStore(resolved_blob_dir) if resolved_blob_dir else None
    tracer = Tracer(
        sinks=sinks,
        redactor=TraceRedactor(
            [
                RedactionRule("llm_request.messages.*.content", "summary_only"),
                RedactionRule("llm_response.raw_text", "summary_only"),
            ]
        ),
        blob_store=blob_store,
        blob_threshold_bytes=8_192,
        best_effort=True,
    )
    trace_run = tracer.start_run(
        "policy_review.shadow_multi_agent",
        workflow_version="v1",
        metadata={
            "engine_name": engine_name,
            "account_type": account_type,
            "trades_table": trades_table,
            "bypass_maturity_gate": bool(bypass_maturity_gate),
            "as_of": as_of,
            "read_only": True,
        },
    )
    return tracer, trace_run, resolved_blob_dir


def _attach_trace_metadata(
    result: dict[str, Any],
    *,
    trace_run: TraceRun,
    trace_output_path: str | None,
    trace_blob_dir: str | None,
    postgres_enabled: bool,
) -> dict[str, Any]:
    if not trace_output_path and not postgres_enabled:
        return result
    out = dict(result)
    out["trace"] = {
        "run_id": trace_run.run_id,
        "output_path": str(Path(trace_output_path).expanduser()) if trace_output_path else None,
        "blob_dir": trace_blob_dir,
        "postgres_enabled": bool(postgres_enabled),
    }
    return out


def _call_llm_agent(
    *,
    agent_name: str,
    system_prompt: str,
    payload: dict[str, Any],
    verbose_steps: bool = False,
    trace_run: TraceRun | None = None,
    trace_step_key: str | None = None,
) -> dict[str, Any]:
    client, provider, model = prod._get_llm_client_and_model()
    _print_step(
        enabled=verbose_steps,
        title=f"Agent [{agent_name}] call",
        description="Send the shared review context plus prior agent output to this role.",
        payload={"provider": provider, "model": model, "agent": agent_name},
    )
    user_prompt = (
        f"Agent: {agent_name}\n"
        "Return JSON only.\n\n"
        f"Input JSON:\n{_json_dumps(payload)}"
    )
    request_kwargs: dict[str, Any] = {
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    temp_raw = os.getenv("POLICY_REVIEW_TEMPERATURE", "").strip()
    if temp_raw:
        request_kwargs["temperature"] = float(temp_raw)

    parsed = trace_llm_call(
        trace_run,
        step_key=trace_step_key or f"llm_{agent_name}",
        agent_name=agent_name,
        provider=provider,
        model=model,
        messages=request_kwargs["messages"],
        prompt_input=payload,
        invoke=lambda: (client.chat.completions.create(**request_kwargs).choices[0].message.content or "").strip(),
        response_format=request_kwargs.get("response_format"),
        temperature=request_kwargs.get("temperature"),
    )

    _print_step(
        enabled=verbose_steps,
        title=f"Agent [{agent_name}] response",
        description="Parsed JSON response returned by this agent.",
        payload=parsed,
    )
    return parsed


def run_multi_agent_shadow_preview(
    *,
    verbose_steps: bool = True,
    as_of: datetime | None = None,
    bypass_maturity_gate: bool | None = None,
    trace_output_path: str | None = None,
    trace_blob_dir: str | None = None,
    trace_database_url: str | None = None,
) -> dict[str, Any]:
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError("DATABASE_URL is required")

    engine_name = os.getenv("STRATEGY_POLICY_ENGINE_NAME", "llm_live").strip()
    account_type = os.getenv("STRATEGY_POLICY_ACCOUNT_TYPE", "live").strip()
    trades_table = prod._safe_table_name(os.getenv("POLICY_REVIEW_TRADES_TABLE", "trades_live"))
    bypass_flag = (
        _env_bool(os.getenv("POLICY_REVIEW_PREVIEW_BYPASS_TIME", "1"), default=True)
        if bypass_maturity_gate is None
        else bool(bypass_maturity_gate)
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
    tracer, trace_run, resolved_trace_blob_dir = _build_trace_run(
        trace_output_path=trace_output_path,
        trace_blob_dir=trace_blob_dir,
        trace_database_url=trace_database_url,
        engine_name=engine_name,
        account_type=account_type,
        trades_table=trades_table,
        bypass_maturity_gate=bypass_flag,
        as_of=as_of,
    )

    try:
        with trace_run:
            _print_step(
                enabled=verbose_steps,
                title="Step 1/7: Build shared review context",
                description="Load guard state, active policy, engine contract, and the richer evidence pack.",
                payload={
                    "engine_name": engine_name,
                    "account_type": account_type,
                    "trades_table": trades_table,
                    "bypass_maturity_gate": bypass_flag,
                    "as_of": as_of,
                },
            )
            with traced_step(
                trace_run,
                step_key="build_review_context",
                kind="prepare",
                metadata={
                    "engine_name": engine_name,
                    "account_type": account_type,
                },
                input_artifacts=[
                    artifact(
                        "review_scope",
                        {
                            "engine_name": engine_name,
                            "account_type": account_type,
                            "trades_table": trades_table,
                            "bypass_maturity_gate": bypass_flag,
                            "as_of": as_of,
                        },
                        role="input",
                        redaction_path=("review_scope",),
                    )
                ],
            ) as trace_step:
                review_input, active_row, guard, cfg = prod.build_policy_review_context(
                    database_url=database_url,
                    engine_name=engine_name,
                    account_type=account_type,
                    trades_table=trades_table,
                    config=cfg,
                    as_of=as_of,
                    include_latest_policy_fallback=True,
                    bypass_maturity_gate=bypass_flag,
                )
                record_artifacts(
                    trace_step,
                    [
                        artifact(
                            "review_input",
                            review_input,
                            role="output",
                            redaction_path=("review_input",),
                        ),
                        artifact(
                            "active_policy",
                            prod._policy_snapshot_from_row(active_row),
                            role="output",
                            redaction_path=("active_policy",),
                        ),
                        artifact(
                            "guard",
                            asdict(guard),
                            role="output",
                            redaction_path=("guard",),
                        ),
                    ],
                )

            guard_payload = asdict(guard)
            deterministic_checks = dict(review_input.get("deterministic_checks") or {})

            _print_step(
                enabled=verbose_steps,
                title="Step 2/7: Deterministic gate summary",
                description="Confirm maturity, data-quality, and risk constraints before any agent proposes a patch.",
                payload={
                    "guard_decision": guard_payload.get("decision"),
                    "guard_reason": guard_payload.get("reason"),
                    "qa_verdict": deterministic_checks.get("qa_verdict"),
                    "allow_llm_proposal": deterministic_checks.get("allow_llm_proposal"),
                    "allow_risk_increase": (deterministic_checks.get("risk_constraints") or {}).get("allow_risk_increase"),
                    "issues": deterministic_checks.get("issues"),
                    "warnings": deterministic_checks.get("warnings"),
                },
            )
            with traced_step(trace_run, step_key="deterministic_gate_summary", kind="decision") as trace_step:
                gate_summary = {
                    "guard_decision": guard_payload.get("decision"),
                    "guard_reason": guard_payload.get("reason"),
                    "qa_verdict": deterministic_checks.get("qa_verdict"),
                    "allow_llm_proposal": deterministic_checks.get("allow_llm_proposal"),
                    "allow_risk_increase": (deterministic_checks.get("risk_constraints") or {}).get("allow_risk_increase"),
                    "issues": deterministic_checks.get("issues"),
                    "warnings": deterministic_checks.get("warnings"),
                }
                record_artifacts(
                    trace_step,
                    [
                        artifact(
                            "gate_summary",
                            gate_summary,
                            role="output",
                            redaction_path=("gate_summary",),
                        )
                    ],
                )
                trace_step.record_decision(
                    "guard_decision",
                    str(guard_payload.get("decision") or "UNKNOWN"),
                    reason=str(guard_payload.get("reason") or "")[:500] or None,
                )
                trace_step.record_validation(
                    "deterministic_checks",
                    str(deterministic_checks.get("qa_verdict") or "UNKNOWN"),
                    errors=list(deterministic_checks.get("issues") or []),
                    warnings=list(deterministic_checks.get("warnings") or []),
                    metrics={
                        "allow_llm_proposal": bool(deterministic_checks.get("allow_llm_proposal")),
                        "allow_risk_increase": bool(
                            (deterministic_checks.get("risk_constraints") or {}).get("allow_risk_increase")
                        ),
                    },
                )

            if not active_row:
                result = {
                    "status": "no_policy",
                    "mode": "shadow_multi_agent",
                    "guard_decision": guard_payload.get("decision"),
                    "guard_reason": guard_payload.get("reason"),
                    "review_input": review_input,
                    "reason": "No strategy policy rows found in DB scope",
                }
                trace_run.record_decision(
                    "final_decision",
                    "NO_POLICY",
                    reason=result["reason"],
                )
                trace_run.record_artifact(
                    "final_result",
                    result,
                    role="output",
                    redaction_path=("final_result",),
                )
                trace_run.complete(
                    status="succeeded",
                    outcome_code="NO_POLICY",
                    summary={"reason": result["reason"]},
                )
                return _attach_trace_metadata(
                    result,
                    trace_run=trace_run,
                    trace_output_path=trace_output_path,
                    trace_blob_dir=resolved_trace_blob_dir,
                    postgres_enabled=bool(trace_database_url),
                )

            common_input = {
                "mode": "shadow_multi_agent",
                "review_input": review_input,
                "deterministic_role": (
                    "The Python controller already handled data QA, maturity gating, and risk checks. "
                    "Focus only on interpretation and proposal quality."
                ),
            }

            analyst_system = (
                "You are the Analyst agent for policy review. "
                "Interpret the evidence pack for a deterministic trading engine. "
                "Do not invent changes. Explain which symbol/setup groups are weak, what evidence supports NO_CHANGE, "
                "and which minimal hypotheses are worth testing. "
                "Return JSON with: summary, key_findings(list), weak_trade_families(list), "
                "no_change_case(list), candidate_hypotheses(list), confidence."
            )
            analyst_out = _call_llm_agent(
                agent_name="analyst",
                system_prompt=analyst_system,
                payload=common_input,
                verbose_steps=verbose_steps,
                trace_run=trace_run,
                trace_step_key="agent_analyst",
            )

            proposer_system = (
                "You are the Proposer agent for policy review. "
                "Use the deterministic checks and analyst findings to propose a minimal, testable patch. "
                "If review_input.deterministic_checks.allow_llm_proposal is false, return NO_CHANGE. "
                "Return JSON with: decision(NO_CHANGE|PROPOSE_CHANGE), reason, confidence, evidence_used(list), "
                "disconfirming_evidence(list), expected_effect(object), changes(list of {path,value,because})."
            )
            proposer_out = _call_llm_agent(
                agent_name="proposer",
                system_prompt=proposer_system,
                payload={**common_input, "analyst_output": analyst_out},
                verbose_steps=verbose_steps,
                trace_run=trace_run,
                trace_step_key="agent_proposer",
            )

            critic_system = (
                "You are the Critic agent for policy review. "
                "Act as an anti-overfitting filter. You may APPROVE, REJECT, or TRIM the proposer's changes, but do not invent a broad new patch. "
                "Return JSON with: decision(APPROVE|REJECT|TRIM), reason, confidence, failure_modes(list), "
                "why_no_change_might_be_better(string), approved_changes(list of {path,value,because})."
            )
            critic_out = _call_llm_agent(
                agent_name="critic",
                system_prompt=critic_system,
                payload={**common_input, "analyst_output": analyst_out, "proposer_output": proposer_out},
                verbose_steps=verbose_steps,
                trace_run=trace_run,
                trace_step_key="agent_critic",
            )

            proposer_decision = str(proposer_out.get("decision") or "NO_CHANGE").strip().upper()
            proposer_patch, proposer_patch_debug = prod.extract_policy_patch_from_response(proposer_out)
            critic_decision = str(critic_out.get("decision") or "REJECT").strip().upper()
            critic_patch, critic_patch_debug = prod.extract_policy_patch_from_response(
                {
                    "changes": critic_out.get("approved_changes"),
                    "patch": critic_out.get("approved_patch"),
                }
            )

            if critic_decision == "TRIM":
                candidate_patch = critic_patch
                candidate_source = "critic_trimmed"
            elif critic_decision == "APPROVE":
                candidate_patch = proposer_patch
                candidate_source = "proposer_approved"
            else:
                candidate_patch = {}
                candidate_source = "critic_rejected"

            _print_step(
                enabled=verbose_steps,
                title="Step 3/7: Choose candidate patch",
                description="Take the proposer's structured patch and let the critic either approve, reject, or trim it.",
                payload={
                    "proposer_decision": proposer_decision,
                    "critic_decision": critic_decision,
                    "candidate_source": candidate_source,
                    "proposer_patch_debug": proposer_patch_debug,
                    "critic_patch_debug": critic_patch_debug,
                    "candidate_patch": candidate_patch,
                },
            )
            with traced_step(trace_run, step_key="candidate_patch_selection", kind="decision") as trace_step:
                record_artifacts(
                    trace_step,
                    [
                        artifact(
                            "candidate_patch_debug",
                            {
                                "proposer_patch_debug": proposer_patch_debug,
                                "critic_patch_debug": critic_patch_debug,
                            },
                            role="output",
                            redaction_path=("candidate_patch_debug",),
                        ),
                        artifact(
                            "candidate_patch_raw",
                            candidate_patch,
                            role="output",
                            redaction_path=("candidate_patch_raw",),
                        ),
                    ],
                )
                trace_step.record_decision(
                    "candidate_source",
                    candidate_source,
                    reason=f"proposer={proposer_decision} critic={critic_decision}",
                    metadata={
                        "proposer_decision": proposer_decision,
                        "critic_decision": critic_decision,
                    },
                )

            active_policy_json = dict(active_row.get("policy_json") or {})
            sanitized_patch, validation_errors, risk_increase_changes = prod.validate_policy_patch(
                patch=candidate_patch,
                active_policy_json=active_policy_json,
            )

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

            _print_step(
                enabled=verbose_steps,
                title="Step 4/7: Deterministic validation",
                description="Apply shared patch validation, strip risk increases when evidence is insufficient, and score proposal quality.",
                payload={
                    "validation_errors": validation_errors,
                    "risk_increase_changes": risk_increase_changes,
                    "effective_risk_increase_changes": effective_risk_increase_changes,
                    "risk_increase_stripped": stripped_risk_changes,
                    "clear_signal_for_risk_increase": clear_signal,
                    "clear_signal_evidence": clear_signal_evidence,
                    "proposal_score": proposal_score,
                    "candidate_patch_sanitized": sanitized_patch,
                },
            )
            with traced_step(trace_run, step_key="patch_validation", kind="validator") as trace_step:
                validation_payload = {
                    "validation_errors": validation_errors,
                    "risk_increase_changes": risk_increase_changes,
                    "effective_risk_increase_changes": effective_risk_increase_changes,
                    "risk_increase_stripped": stripped_risk_changes,
                    "clear_signal_for_risk_increase": clear_signal,
                    "clear_signal_evidence": clear_signal_evidence,
                    "proposal_score": proposal_score,
                }
                record_artifacts(
                    trace_step,
                    [
                        artifact(
                            "candidate_patch_sanitized",
                            sanitized_patch,
                            role="output",
                            redaction_path=("candidate_patch_sanitized",),
                        ),
                        artifact(
                            "validation_summary",
                            validation_payload,
                            role="output",
                            redaction_path=("validation_summary",),
                        ),
                    ],
                )
                trace_step.record_validation(
                    "policy_patch_validation",
                    "PASS" if not validation_errors else "FAIL",
                    errors=validation_errors,
                    warnings=stripped_risk_changes,
                    metrics={
                        "risk_increase_changes": len(risk_increase_changes),
                        "effective_risk_increase_changes": len(effective_risk_increase_changes),
                        "proposal_score": proposal_score.get("score"),
                        "proposal_verdict": proposal_score.get("verdict"),
                    },
                )

            merged_policy_preview = prod._deep_merge(active_policy_json, sanitized_patch)
            patch_effective = not validation_errors and not prod._is_patch_empty(sanitized_patch)
            final_decision = "NO_CHANGE"
            if (
                proposer_decision == "PROPOSE_CHANGE"
                and critic_decision != "REJECT"
                and patch_effective
                and proposal_score.get("verdict") != "REJECT"
                and deterministic_checks.get("allow_llm_proposal", False)
            ):
                final_decision = "PROPOSE_CHANGE"

            _print_step(
                enabled=verbose_steps,
                title="Step 5/7: Final shadow decision",
                description="Compute the read-only shadow decision after agent output meets deterministic validation.",
                payload={
                    "final_decision": final_decision,
                    "patch_effective": patch_effective,
                    "merged_policy_preview": merged_policy_preview,
                },
            )
            with traced_step(trace_run, step_key="final_shadow_decision", kind="decision") as trace_step:
                record_artifacts(
                    trace_step,
                    [
                        artifact(
                            "merged_policy_preview",
                            merged_policy_preview,
                            role="output",
                            redaction_path=("merged_policy_preview",),
                        )
                    ],
                )
                trace_step.record_decision(
                    "final_decision",
                    final_decision,
                    metadata={
                        "patch_effective": patch_effective,
                        "proposal_verdict": proposal_score.get("verdict"),
                    },
                )

            result = {
                "status": "ok",
                "mode": "shadow_multi_agent",
                "guard_decision": guard_payload.get("decision"),
                "guard_reason": guard_payload.get("reason"),
                "bypass_maturity_gate": bypass_flag,
                "active_policy_id": int(active_row["id"]),
                "active_policy_version": int(active_row.get("version") or 0),
                "review_input": review_input,
                "agents": {
                    "analyst": analyst_out,
                    "proposer": proposer_out,
                    "critic": critic_out,
                },
                "candidate_patch_source": candidate_source,
                "candidate_patch_raw": candidate_patch,
                "candidate_patch_sanitized": sanitized_patch,
                "validation_errors": validation_errors,
                "risk_increase_changes": risk_increase_changes,
                "effective_risk_increase_changes": effective_risk_increase_changes,
                "risk_increase_stripped": stripped_risk_changes,
                "clear_signal_for_risk_increase": clear_signal,
                "clear_signal_evidence": clear_signal_evidence,
                "proposal_score": proposal_score,
                "merged_policy_preview": merged_policy_preview,
                "final_decision": final_decision,
                "notes": "Read-only shadow run only. No DB writes were performed.",
            }
            trace_run.record_artifact(
                "final_result",
                result,
                role="output",
                redaction_path=("final_result",),
            )
            trace_run.complete(
                status="succeeded",
                outcome_code=final_decision,
                summary={
                    "candidate_source": candidate_source,
                    "proposal_verdict": proposal_score.get("verdict"),
                },
            )
            return _attach_trace_metadata(
                result,
                trace_run=trace_run,
                trace_output_path=trace_output_path,
                trace_blob_dir=resolved_trace_blob_dir,
                postgres_enabled=bool(trace_database_url),
            )
    finally:
        tracer.close()


def main():
    parser = argparse.ArgumentParser(description="Run multi-agent shadow policy review (read-only)")
    parser.add_argument("--json-only", action="store_true")
    parser.add_argument("--quiet-steps", action="store_true")
    parser.add_argument("--as-of", type=str, default="")
    parser.add_argument("--trace-output", type=str, default="")
    parser.add_argument("--trace-blob-dir", type=str, default="")
    parser.add_argument("--trace-postgres", action="store_true")
    parser.add_argument("--trace-database-url", type=str, default="")
    args = parser.parse_args()

    resolved_trace_db_url = None
    if args.trace_postgres:
        resolved_trace_db_url = (
            args.trace_database_url.strip()
            or os.getenv("TRACE_DATABASE_URL", "").strip()
            or os.getenv("DATABASE_URL", "").strip()
        )
        if not resolved_trace_db_url:
            parser.error("--trace-postgres requires --trace-database-url, TRACE_DATABASE_URL, or DATABASE_URL")

    result = run_multi_agent_shadow_preview(
        verbose_steps=not args.quiet_steps,
        as_of=_parse_as_of(args.as_of),
        trace_output_path=args.trace_output or None,
        trace_blob_dir=args.trace_blob_dir or None,
        trace_database_url=resolved_trace_db_url or None,
    )
    output = _json_dumps(result)
    print(output)


if __name__ == "__main__":
    main()
