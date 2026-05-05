from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Protocol

import psycopg2
from psycopg2.extras import Json

from .serialization import json_dumps


class TraceSink(Protocol):
    def emit(self, event: dict[str, Any]) -> None:
        ...

    def flush(self) -> None:
        ...

    def close(self) -> None:
        ...


class JsonlTraceSink:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")
        self._lock = threading.Lock()

    def emit(self, event: dict[str, Any]) -> None:
        line = json_dumps(event)
        with self._lock:
            self._handle.write(line)
            self._handle.write("\n")
            self._handle.flush()

    def flush(self) -> None:
        with self._lock:
            self._handle.flush()

    def close(self) -> None:
        with self._lock:
            if not self._handle.closed:
                self._handle.flush()
                self._handle.close()


class LoggingTraceSink:
    def __init__(self, logger: logging.Logger | None = None, level: int = logging.INFO):
        self.logger = logger or logging.getLogger("observability.trace")
        self.level = level

    def emit(self, event: dict[str, Any]) -> None:
        self.logger.log(self.level, json_dumps(event))

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


class MemoryTraceSink:
    def __init__(self):
        self.events: list[dict[str, Any]] = []

    def emit(self, event: dict[str, Any]) -> None:
        self.events.append(event)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


class PostgresTraceSink:
    def __init__(
        self,
        database_url: str,
        *,
        sslmode: str = "require",
        application_name: str = "observability-trace",
        connect_timeout: int = 10,
    ):
        self.database_url = str(database_url).strip()
        self.sslmode = sslmode
        self.application_name = application_name
        self.connect_timeout = int(connect_timeout)
        self._conn = None
        self._lock = threading.Lock()

    def emit(self, event: dict[str, Any]) -> None:
        with self._lock:
            try:
                self._emit_internal(event)
            except Exception:
                self._reset_conn()
                raise

    def flush(self) -> None:
        return None

    def close(self) -> None:
        with self._lock:
            self._reset_conn()

    def _emit_internal(self, event: dict[str, Any]) -> None:
        event_type = str(event.get("event_type") or "").strip()
        payload = dict(event.get("payload") or {})

        if event_type == "run.started":
            self._upsert_run_started(event, payload)
        elif event_type == "run.completed":
            self._upsert_run_completed(event, payload)
        elif event_type == "step.started":
            self._upsert_step_started(event, payload)
        elif event_type == "step.completed":
            self._upsert_step_completed(event, payload)

        self._insert_event(event)

    def _connect(self):
        if self._conn is None or getattr(self._conn, "closed", 1):
            self._conn = psycopg2.connect(
                self.database_url,
                sslmode=self.sslmode,
                application_name=self.application_name,
                connect_timeout=self.connect_timeout,
            )
            self._conn.autocommit = True
        return self._conn

    def _reset_conn(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
        self._conn = None

    def _execute(self, query: str, params: tuple[Any, ...]) -> None:
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute(query, params)

    def _upsert_run_started(self, event: dict[str, Any], payload: dict[str, Any]) -> None:
        self._execute(
            """
            INSERT INTO trace_runs
              (run_id, workflow_name, workflow_version, status,
               started_at, metadata, tags, correlation_ids,
               created_at, updated_at)
            VALUES
              (%s, %s, %s, %s,
               %s, %s::jsonb, %s::jsonb, %s::jsonb,
               NOW(), NOW())
            ON CONFLICT (run_id) DO UPDATE
            SET workflow_name = EXCLUDED.workflow_name,
                workflow_version = EXCLUDED.workflow_version,
                status = EXCLUDED.status,
                started_at = COALESCE(trace_runs.started_at, EXCLUDED.started_at),
                metadata = EXCLUDED.metadata,
                tags = EXCLUDED.tags,
                correlation_ids = EXCLUDED.correlation_ids,
                updated_at = NOW()
            """,
            (
                event.get("run_id"),
                event.get("workflow_name"),
                event.get("workflow_version"),
                "running",
                event.get("ts"),
                Json(payload.get("metadata") or {}),
                Json(payload.get("tags") or {}),
                Json(payload.get("correlation_ids") or {}),
            ),
        )

    def _upsert_run_completed(self, event: dict[str, Any], payload: dict[str, Any]) -> None:
        self._execute(
            """
            INSERT INTO trace_runs
              (run_id, workflow_name, workflow_version, status,
               started_at, ended_at, duration_ms, outcome_code, summary,
               metadata, tags, correlation_ids,
               created_at, updated_at)
            VALUES
              (%s, %s, %s, %s,
               %s, %s, %s, %s, %s::jsonb,
               '{}'::jsonb, '{}'::jsonb, '{}'::jsonb,
               NOW(), NOW())
            ON CONFLICT (run_id) DO UPDATE
            SET workflow_name = EXCLUDED.workflow_name,
                workflow_version = EXCLUDED.workflow_version,
                status = EXCLUDED.status,
                ended_at = EXCLUDED.ended_at,
                duration_ms = EXCLUDED.duration_ms,
                outcome_code = EXCLUDED.outcome_code,
                summary = EXCLUDED.summary,
                updated_at = NOW()
            """,
            (
                event.get("run_id"),
                event.get("workflow_name"),
                event.get("workflow_version"),
                payload.get("status"),
                payload.get("started_at") or event.get("ts"),
                payload.get("ended_at") or event.get("ts"),
                payload.get("duration_ms"),
                payload.get("outcome_code"),
                Json(payload.get("summary") or {}),
            ),
        )

    def _upsert_step_started(self, event: dict[str, Any], payload: dict[str, Any]) -> None:
        self._execute(
            """
            INSERT INTO trace_steps
              (step_id, run_id, step_key, kind, attempt, status,
               started_at, parent_step_id, metadata,
               created_at, updated_at)
            VALUES
              (%s, %s, %s, %s, %s, %s,
               %s, %s, %s::jsonb,
               NOW(), NOW())
            ON CONFLICT (step_id) DO UPDATE
            SET run_id = EXCLUDED.run_id,
                step_key = EXCLUDED.step_key,
                kind = EXCLUDED.kind,
                attempt = EXCLUDED.attempt,
                status = EXCLUDED.status,
                started_at = COALESCE(trace_steps.started_at, EXCLUDED.started_at),
                parent_step_id = COALESCE(trace_steps.parent_step_id, EXCLUDED.parent_step_id),
                metadata = EXCLUDED.metadata,
                updated_at = NOW()
            """,
            (
                event.get("step_id"),
                event.get("run_id"),
                event.get("step_key"),
                payload.get("kind"),
                payload.get("attempt"),
                "running",
                event.get("ts"),
                payload.get("parent_step_id"),
                Json(payload.get("metadata") or {}),
            ),
        )

    def _upsert_step_completed(self, event: dict[str, Any], payload: dict[str, Any]) -> None:
        self._execute(
            """
            INSERT INTO trace_steps
              (step_id, run_id, step_key, kind, attempt, status,
               started_at, ended_at, duration_ms, summary,
               created_at, updated_at)
            VALUES
              (%s, %s, %s, %s, %s, %s,
               %s, %s, %s, %s::jsonb,
               NOW(), NOW())
            ON CONFLICT (step_id) DO UPDATE
            SET run_id = EXCLUDED.run_id,
                step_key = EXCLUDED.step_key,
                kind = COALESCE(trace_steps.kind, EXCLUDED.kind),
                attempt = COALESCE(trace_steps.attempt, EXCLUDED.attempt),
                status = EXCLUDED.status,
                started_at = COALESCE(trace_steps.started_at, EXCLUDED.started_at),
                ended_at = EXCLUDED.ended_at,
                duration_ms = EXCLUDED.duration_ms,
                summary = EXCLUDED.summary,
                updated_at = NOW()
            """,
            (
                event.get("step_id"),
                event.get("run_id"),
                event.get("step_key"),
                payload.get("kind"),
                payload.get("attempt"),
                payload.get("status"),
                payload.get("started_at") or event.get("ts"),
                payload.get("ended_at") or event.get("ts"),
                payload.get("duration_ms"),
                Json(payload.get("summary") or {}),
            ),
        )

    def _insert_event(self, event: dict[str, Any]) -> None:
        self._execute(
            """
            INSERT INTO trace_events
              (event_id, run_id, step_id, workflow_name, workflow_version,
               step_key, event_type, seq, ts, payload, created_at)
            VALUES
              (%s, %s, %s, %s, %s,
               %s, %s, %s, %s, %s::jsonb, NOW())
            ON CONFLICT (event_id) DO NOTHING
            """,
            (
                event.get("event_id"),
                event.get("run_id"),
                event.get("step_id"),
                event.get("workflow_name"),
                event.get("workflow_version"),
                event.get("step_key"),
                event.get("event_type"),
                event.get("seq"),
                event.get("ts"),
                Json(event.get("payload") or {}),
            ),
        )
