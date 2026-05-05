from __future__ import annotations

import time
import traceback
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from .blob_store import LocalBlobStore
from .redaction import TraceRedactor
from .serialization import json_bytes, summarize_value, to_serializable
from .sinks import TraceSink


UTC = timezone.utc


def _now_utc() -> datetime:
    return datetime.now(UTC)


def _normalize_path(path: str | tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    if path is None:
        return ()
    if isinstance(path, str):
        return tuple(part for part in path.split(".") if part)
    return tuple(str(part) for part in path)


class Tracer:
    def __init__(
        self,
        *,
        sinks: list[TraceSink] | None = None,
        redactor: TraceRedactor | None = None,
        blob_store: LocalBlobStore | None = None,
        blob_threshold_bytes: int = 32_768,
        best_effort: bool = True,
        capture_tracebacks: bool = True,
    ):
        self._sinks = list(sinks or [])
        self.redactor = redactor or TraceRedactor()
        self.blob_store = blob_store
        self.blob_threshold_bytes = int(blob_threshold_bytes)
        self.best_effort = bool(best_effort)
        self.capture_tracebacks = bool(capture_tracebacks)

    def start_run(
        self,
        workflow_name: str,
        *,
        workflow_version: str = "dev",
        metadata: dict[str, Any] | None = None,
        tags: dict[str, Any] | None = None,
        correlation_ids: dict[str, Any] | None = None,
    ) -> "TraceRun":
        return TraceRun(
            tracer=self,
            workflow_name=workflow_name,
            workflow_version=workflow_version,
            metadata=metadata or {},
            tags=tags or {},
            correlation_ids=correlation_ids or {},
        )

    def emit(self, event: dict[str, Any]) -> None:
        if not self._sinks:
            return
        for sink in self._sinks:
            try:
                sink.emit(event)
            except Exception:
                if not self.best_effort:
                    raise

    def flush(self) -> None:
        for sink in self._sinks:
            try:
                sink.flush()
            except Exception:
                if not self.best_effort:
                    raise

    def close(self) -> None:
        for sink in self._sinks:
            try:
                sink.close()
            except Exception:
                if not self.best_effort:
                    raise


class TraceRun:
    def __init__(
        self,
        *,
        tracer: Tracer,
        workflow_name: str,
        workflow_version: str,
        metadata: dict[str, Any],
        tags: dict[str, Any],
        correlation_ids: dict[str, Any],
    ):
        self.tracer = tracer
        self.run_id = uuid.uuid4().hex
        self.workflow_name = workflow_name
        self.workflow_version = workflow_version
        self.metadata = to_serializable(metadata)
        self.tags = to_serializable(tags)
        self.correlation_ids = to_serializable(correlation_ids)
        self.started_at = _now_utc()
        self._seq = 0
        self._step_attempts: defaultdict[str, int] = defaultdict(int)
        self._started = False
        self._completed = False

    def __enter__(self) -> "TraceRun":
        if not self._started:
            self._started = True
            self._emit(
                "run.started",
                payload={
                    "workflow_name": self.workflow_name,
                    "workflow_version": self.workflow_version,
                    "metadata": self.metadata,
                    "tags": self.tags,
                    "correlation_ids": self.correlation_ids,
                },
            )
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is not None:
            self.record_error(exc=exc, tb=tb, message="run failed")
            self.complete(
                status="failed",
                outcome_code=exc.__class__.__name__,
                summary={"message": str(exc)},
            )
            return False
        if not self._completed:
            self.complete(status="succeeded")
        return False

    def step(
        self,
        step_key: str,
        *,
        kind: str,
        metadata: dict[str, Any] | None = None,
        parent_step_id: str | None = None,
    ) -> "TraceStep":
        self._step_attempts[step_key] += 1
        return TraceStep(
            run=self,
            step_key=step_key,
            kind=kind,
            attempt=self._step_attempts[step_key],
            metadata=metadata or {},
            parent_step_id=parent_step_id,
        )

    def record_artifact(
        self,
        name: str,
        value: Any,
        *,
        step: "TraceStep | None" = None,
        role: str = "artifact",
        content_type: str = "application/json",
        redaction_path: str | tuple[str, ...] | list[str] | None = None,
    ) -> dict[str, Any]:
        path = _normalize_path(redaction_path) or _normalize_path(name)
        redacted_value = self.tracer.redactor.redact(value, path=path)
        payload_bytes = json_bytes(redacted_value)
        artifact_payload: dict[str, Any] = {
            "name": name,
            "role": role,
            "content_type": content_type,
            "redaction_path": ".".join(path),
            "size_bytes": len(payload_bytes),
            "summary": summarize_value(redacted_value),
        }
        if self.tracer.blob_store and len(payload_bytes) > self.tracer.blob_threshold_bytes:
            prefix = _artifact_prefix(self.workflow_name, step.step_key if step else "run", name)
            artifact_payload["storage"] = "blob"
            artifact_payload["blob"] = self.tracer.blob_store.put_json(
                redacted_value,
                prefix=prefix,
            )
        else:
            artifact_payload["storage"] = "inline"
            artifact_payload["value"] = redacted_value

        self._emit(
            "artifact.captured",
            step=step,
            payload=artifact_payload,
        )
        return artifact_payload

    def record_decision(
        self,
        name: str,
        choice: str,
        *,
        step: "TraceStep | None" = None,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._emit(
            "decision.recorded",
            step=step,
            payload={
                "name": name,
                "choice": choice,
                "reason": reason,
                "metadata": to_serializable(metadata or {}),
            },
        )

    def record_validation(
        self,
        name: str,
        status: str,
        *,
        step: "TraceStep | None" = None,
        errors: list[str] | None = None,
        warnings: list[str] | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        self._emit(
            "validation.recorded",
            step=step,
            payload={
                "name": name,
                "status": status,
                "errors": list(errors or []),
                "warnings": list(warnings or []),
                "metrics": to_serializable(metrics or {}),
            },
        )

    def record_metric(
        self,
        name: str,
        value: Any,
        *,
        step: "TraceStep | None" = None,
        unit: str | None = None,
    ) -> None:
        self._emit(
            "metric.recorded",
            step=step,
            payload={
                "name": name,
                "value": to_serializable(value),
                "unit": unit,
            },
        )

    def record_error(
        self,
        *,
        exc: BaseException | None = None,
        tb=None,
        step: "TraceStep | None" = None,
        message: str | None = None,
        retryable: bool | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "message": message or (str(exc) if exc is not None else ""),
            "error_type": exc.__class__.__name__ if exc is not None else None,
            "retryable": retryable,
            "details": to_serializable(details or {}),
        }
        if exc is not None and self.tracer.capture_tracebacks:
            payload["traceback"] = "".join(traceback.format_exception(type(exc), exc, tb))
        self._emit("error.recorded", step=step, payload=payload)

    def complete(
        self,
        *,
        status: str,
        outcome_code: str | None = None,
        summary: dict[str, Any] | None = None,
    ) -> None:
        if self._completed:
            return
        ended_at = _now_utc()
        self._completed = True
        self._emit(
            "run.completed",
            payload={
                "status": status,
                "outcome_code": outcome_code,
                "summary": to_serializable(summary or {}),
                "started_at": self.started_at,
                "ended_at": ended_at,
                "duration_ms": int((ended_at - self.started_at).total_seconds() * 1000.0),
            },
        )
        self.tracer.flush()

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _emit(
        self,
        event_type: str,
        *,
        step: "TraceStep | None" = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        event = {
            "event_id": uuid.uuid4().hex,
            "event_type": event_type,
            "run_id": self.run_id,
            "workflow_name": self.workflow_name,
            "workflow_version": self.workflow_version,
            "step_id": step.step_id if step else None,
            "step_key": step.step_key if step else None,
            "seq": self._next_seq(),
            "ts": _now_utc(),
            "payload": to_serializable(payload or {}),
        }
        self.tracer.emit(event)


class TraceStep:
    def __init__(
        self,
        *,
        run: TraceRun,
        step_key: str,
        kind: str,
        attempt: int,
        metadata: dict[str, Any],
        parent_step_id: str | None,
    ):
        self.run = run
        self.step_id = uuid.uuid4().hex
        self.step_key = step_key
        self.kind = kind
        self.attempt = int(attempt)
        self.metadata = to_serializable(metadata)
        self.parent_step_id = parent_step_id
        self.started_at = _now_utc()
        self._started = False
        self._completed = False
        self._start_monotonic = 0.0

    def __enter__(self) -> "TraceStep":
        if not self._started:
            self._started = True
            self._start_monotonic = time.monotonic()
            self.run._emit(
                "step.started",
                step=self,
                payload={
                    "step_key": self.step_key,
                    "kind": self.kind,
                    "attempt": self.attempt,
                    "metadata": self.metadata,
                    "parent_step_id": self.parent_step_id,
                },
            )
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is not None:
            self.run.record_error(exc=exc, tb=tb, step=self, message=f"step failed: {self.step_key}")
            self.finish(
                status="failed",
                summary={"message": str(exc)},
            )
            return False
        if not self._completed:
            self.finish(status="succeeded")
        return False

    def record_artifact(
        self,
        name: str,
        value: Any,
        *,
        role: str = "artifact",
        content_type: str = "application/json",
        redaction_path: str | tuple[str, ...] | list[str] | None = None,
    ) -> dict[str, Any]:
        return self.run.record_artifact(
            name,
            value,
            step=self,
            role=role,
            content_type=content_type,
            redaction_path=redaction_path,
        )

    def record_decision(
        self,
        name: str,
        choice: str,
        *,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.run.record_decision(
            name,
            choice,
            step=self,
            reason=reason,
            metadata=metadata,
        )

    def record_validation(
        self,
        name: str,
        status: str,
        *,
        errors: list[str] | None = None,
        warnings: list[str] | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        self.run.record_validation(
            name,
            status,
            step=self,
            errors=errors,
            warnings=warnings,
            metrics=metrics,
        )

    def record_metric(self, name: str, value: Any, *, unit: str | None = None) -> None:
        self.run.record_metric(name, value, step=self, unit=unit)

    def finish(
        self,
        *,
        status: str,
        summary: dict[str, Any] | None = None,
    ) -> None:
        if self._completed:
            return
        ended_at = _now_utc()
        duration_ms = int(max(0.0, (time.monotonic() - self._start_monotonic) * 1000.0))
        self._completed = True
        self.run._emit(
            "step.completed",
            step=self,
            payload={
                "status": status,
                "summary": to_serializable(summary or {}),
                "started_at": self.started_at,
                "ended_at": ended_at,
                "duration_ms": duration_ms,
                "kind": self.kind,
                "attempt": self.attempt,
            },
        )


def _artifact_prefix(workflow_name: str, step_key: str, artifact_name: str) -> str:
    def _clean(text: str) -> str:
        return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_") or "artifact"

    return "_".join(
        [
            _clean(workflow_name),
            _clean(step_key),
            _clean(artifact_name),
        ]
    )
