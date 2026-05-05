-- Migration: generic observability traces
-- Stores reusable run/step/event audit traces for LLM workflows.

CREATE TABLE IF NOT EXISTS trace_runs (
    run_id            TEXT PRIMARY KEY,
    workflow_name     TEXT        NOT NULL,
    workflow_version  TEXT        NOT NULL DEFAULT 'dev',
    status            TEXT        NOT NULL DEFAULT 'running'
                                  CHECK (status IN ('running', 'succeeded', 'failed', 'cancelled')),
    outcome_code      TEXT,
    started_at        TIMESTAMPTZ NOT NULL,
    ended_at          TIMESTAMPTZ,
    duration_ms       BIGINT,
    metadata          JSONB       NOT NULL DEFAULT '{}'::jsonb,
    tags              JSONB       NOT NULL DEFAULT '{}'::jsonb,
    correlation_ids   JSONB       NOT NULL DEFAULT '{}'::jsonb,
    summary           JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_trace_runs_workflow_time
    ON trace_runs (workflow_name, started_at DESC);

CREATE INDEX IF NOT EXISTS idx_trace_runs_status_time
    ON trace_runs (status, started_at DESC);

CREATE INDEX IF NOT EXISTS idx_trace_runs_metadata
    ON trace_runs USING GIN (metadata);

CREATE TABLE IF NOT EXISTS trace_steps (
    step_id           TEXT PRIMARY KEY,
    run_id            TEXT        NOT NULL REFERENCES trace_runs(run_id) ON DELETE CASCADE,
    step_key          TEXT        NOT NULL,
    kind              TEXT        NOT NULL,
    attempt           INTEGER     NOT NULL DEFAULT 1,
    status            TEXT        NOT NULL DEFAULT 'running'
                                  CHECK (status IN ('running', 'succeeded', 'failed', 'skipped')),
    started_at        TIMESTAMPTZ NOT NULL,
    ended_at          TIMESTAMPTZ,
    duration_ms       BIGINT,
    parent_step_id    TEXT,
    metadata          JSONB       NOT NULL DEFAULT '{}'::jsonb,
    summary           JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_trace_steps_run_time
    ON trace_steps (run_id, started_at ASC);

CREATE INDEX IF NOT EXISTS idx_trace_steps_key_attempt
    ON trace_steps (run_id, step_key, attempt);

CREATE INDEX IF NOT EXISTS idx_trace_steps_status
    ON trace_steps (status, started_at DESC);

CREATE TABLE IF NOT EXISTS trace_events (
    event_id          TEXT PRIMARY KEY,
    run_id            TEXT        NOT NULL REFERENCES trace_runs(run_id) ON DELETE CASCADE,
    step_id           TEXT,
    workflow_name     TEXT        NOT NULL,
    workflow_version  TEXT        NOT NULL DEFAULT 'dev',
    step_key          TEXT,
    event_type        TEXT        NOT NULL,
    seq               BIGINT      NOT NULL,
    ts                TIMESTAMPTZ NOT NULL,
    payload           JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_trace_events_run_seq
    ON trace_events (run_id, seq ASC);

CREATE INDEX IF NOT EXISTS idx_trace_events_step_seq
    ON trace_events (step_id, seq ASC);

CREATE INDEX IF NOT EXISTS idx_trace_events_type_time
    ON trace_events (event_type, ts DESC);

CREATE INDEX IF NOT EXISTS idx_trace_events_payload
    ON trace_events USING GIN (payload);
