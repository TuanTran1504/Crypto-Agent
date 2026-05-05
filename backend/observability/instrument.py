from __future__ import annotations

import json
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Iterator, TypeVar

from .serialization import json_dumps
from .tracer import TraceRun, TraceStep


T = TypeVar("T")
RedactionPath = str | tuple[str, ...] | list[str] | None


@dataclass(frozen=True)
class ArtifactSpec:
    name: str
    value: Any
    role: str = "artifact"
    content_type: str = "application/json"
    redaction_path: RedactionPath = None


def artifact(
    name: str,
    value: Any,
    *,
    role: str = "artifact",
    content_type: str = "application/json",
    redaction_path: RedactionPath = None,
) -> ArtifactSpec:
    return ArtifactSpec(
        name=name,
        value=value,
        role=role,
        content_type=content_type,
        redaction_path=redaction_path,
    )


def record_artifacts(step: TraceStep | None, artifacts: list[ArtifactSpec] | None) -> None:
    if step is None:
        return
    for item in artifacts or []:
        step.record_artifact(
            item.name,
            item.value,
            role=item.role,
            content_type=item.content_type,
            redaction_path=item.redaction_path,
        )


@contextmanager
def traced_step(
    trace_run: TraceRun | None,
    *,
    step_key: str,
    kind: str,
    metadata: dict[str, Any] | None = None,
    input_artifacts: list[ArtifactSpec] | None = None,
) -> Iterator[TraceStep | None]:
    context = (
        trace_run.step(step_key, kind=kind, metadata=metadata or {})
        if trace_run is not None
        else nullcontext(None)
    )
    with context as step:
        record_artifacts(step, input_artifacts)
        yield step


def trace_llm_call(
    trace_run: TraceRun | None,
    *,
    step_key: str,
    agent_name: str,
    provider: str,
    model: str,
    messages: list[dict[str, Any]],
    prompt_input: Any | None,
    invoke: Callable[[], str],
    parser: Callable[[str], T] | None = None,
    response_format: dict[str, Any] | None = None,
    temperature: Any | None = None,
    metadata: dict[str, Any] | None = None,
    request_redaction_path: RedactionPath = ("llm_request",),
    raw_text_redaction_path: RedactionPath = ("llm_response", "raw_text"),
    parsed_redaction_path: RedactionPath = ("llm_response", "parsed"),
    decision_name: str = "agent_decision",
    decision_path: str = "decision",
    reason_path: str = "reason",
) -> T:
    llm_metadata = {
        "agent_name": agent_name,
        "provider": provider,
        "model": model,
    }
    if metadata:
        llm_metadata.update(metadata)

    request_payload = {
        "provider": provider,
        "model": model,
        "agent_name": agent_name,
        "messages": messages,
        "prompt_input": prompt_input,
        "response_format": response_format,
        "temperature": temperature,
    }

    with traced_step(
        trace_run,
        step_key=step_key,
        kind="llm",
        metadata=llm_metadata,
        input_artifacts=[
            artifact(
                "request",
                request_payload,
                role="input",
                redaction_path=request_redaction_path,
            )
        ],
    ) as step:
        if step is not None:
            step.record_metric(
                "prompt_chars",
                _measure_message_chars(messages),
                unit="chars",
            )

        raw_text = invoke()
        parsed: Any
        if parser is not None:
            parsed = parser(raw_text)
        else:
            parsed = json.loads(raw_text) if raw_text else {}

        if isinstance(parsed, dict):
            meta = dict(parsed.get("_meta") or {})
            meta.update(
                {
                    "provider": provider,
                    "model": model,
                    "agent": agent_name,
                }
            )
            parsed["_meta"] = meta

        if step is not None:
            step.record_metric("response_chars", len(raw_text), unit="chars")
            step.record_artifact(
                "response.raw_text",
                raw_text,
                role="output",
                content_type="text/plain",
                redaction_path=raw_text_redaction_path,
            )
            step.record_artifact(
                "response.parsed",
                parsed,
                role="output",
                redaction_path=parsed_redaction_path,
            )
            if isinstance(parsed, dict):
                decision = _get_from_dot_path(parsed, decision_path)
                if decision is not None:
                    step.record_decision(
                        decision_name,
                        str(decision),
                        reason=_string_or_none(_get_from_dot_path(parsed, reason_path)),
                        metadata={"agent_name": agent_name},
                    )

        return parsed


def trace_tool_call(
    trace_run: TraceRun | None,
    *,
    step_key: str,
    tool_name: str,
    request: Any,
    invoke: Callable[[], T],
    metadata: dict[str, Any] | None = None,
    request_redaction_path: RedactionPath = ("tool_request",),
    result_redaction_path: RedactionPath = ("tool_result",),
    result_name: str = "result",
    result_content_type: str = "application/json",
    decision_extractor: Callable[[T], dict[str, Any] | None] | None = None,
) -> T:
    tool_metadata = {"tool_name": tool_name}
    if metadata:
        tool_metadata.update(metadata)

    with traced_step(
        trace_run,
        step_key=step_key,
        kind="tool",
        metadata=tool_metadata,
        input_artifacts=[
            artifact(
                "request",
                request,
                role="input",
                redaction_path=request_redaction_path,
            )
        ],
    ) as step:
        result = invoke()
        if step is not None:
            step.record_artifact(
                result_name,
                result,
                role="output",
                content_type=result_content_type,
                redaction_path=result_redaction_path,
            )
            if decision_extractor is not None:
                decision = decision_extractor(result)
                if decision:
                    step.record_decision(
                        str(decision.get("name") or "tool_decision"),
                        str(decision.get("choice") or "UNKNOWN"),
                        reason=_string_or_none(decision.get("reason")),
                        metadata=dict(decision.get("metadata") or {}),
                    )
        return result


def _measure_message_chars(messages: list[dict[str, Any]]) -> int:
    total = 0
    for message in messages:
        total += len(_stringify_content(message.get("content")))
    return total


def _stringify_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json_dumps(value)


def _get_from_dot_path(payload: dict[str, Any], path: str) -> Any:
    parts = [part for part in str(path or "").split(".") if part]
    cur: Any = payload
    for part in parts:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _string_or_none(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None
