CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS grid_observations (
    id                    BIGSERIAL,
    timestamp             TIMESTAMPTZ     NOT NULL,
    settlement_period     INTEGER,
    actual_load_mw        FLOAT,
    generation_wind_mw    FLOAT,
    generation_solar_mw   FLOAT,
    generation_gas_mw     FLOAT,
    generation_nuclear_mw FLOAT,
    grid_imbalance_price  FLOAT,
    temperature_c         FLOAT,
    wind_speed_ms         FLOAT,
    cloud_cover_pct       FLOAT,
    hour                  INTEGER,
    weekday               INTEGER,
    month                 INTEGER,
    is_holiday            BOOLEAN DEFAULT FALSE,
    load_lag_1h           FLOAT,
    load_lag_24h          FLOAT,
    load_lag_168h         FLOAT,
    oil_price_brent       FLOAT,
    oil_delta_pct_30m     FLOAT,
    PRIMARY KEY (id, timestamp)
);

CREATE TABLE IF NOT EXISTS forecasts (
    id              BIGSERIAL,
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    forecast_for    TIMESTAMPTZ     NOT NULL,
    forecast_mw     FLOAT           NOT NULL,
    lower_bound_mw  FLOAT           NOT NULL,
    upper_bound_mw  FLOAT           NOT NULL,
    confidence_pct  INTEGER         DEFAULT 90,
    shock_score     FLOAT           NOT NULL,
    shock_active    BOOLEAN         NOT NULL,
    oil_delta_pct   FLOAT,
    headline_score  FLOAT,
    actual_mw       FLOAT,
    mape_pct        FLOAT,
    model_version   TEXT,
    PRIMARY KEY (id, created_at)
);



CREATE TABLE IF NOT EXISTS model_runs (
    id              BIGSERIAL       PRIMARY KEY,
    run_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    mlflow_run_id   TEXT            UNIQUE,
    mape_overall    FLOAT,
    mape_normal     FLOAT,
    mape_shock      FLOAT,
    train_samples   INTEGER,
    val_samples     INTEGER,
    shock_samples   INTEGER,
    promoted        BOOLEAN         DEFAULT FALSE,
    notes           TEXT
);

CREATE TABLE IF NOT EXISTS vnd_rates (
    id          BIGSERIAL PRIMARY KEY,
    fetched_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    usd_to_vnd  FLOAT NOT NULL
);

CREATE INDEX idx_vnd_rates_fetched_at 
ON vnd_rates (fetched_at DESC);

CREATE TABLE IF NOT EXISTS macro_events (
    id           BIGSERIAL PRIMARY KEY,
    event_date   TIMESTAMPTZ NOT NULL,
    event_type   TEXT NOT NULL,  -- 'FED_RATE', 'CPI', 'NFP', 'HALVING'
    description  TEXT,
    expected     TEXT,           -- market consensus before event
    actual       TEXT,           -- filled in after event
    impact_pct   FLOAT,          -- actual BTC move within 24h
    CONSTRAINT macro_events_date_type_unique UNIQUE (event_date, event_type)
);

DROP TABLE shock_events;

CREATE TABLE shock_events (
    id              BIGSERIAL PRIMARY KEY,
    event_date      TIMESTAMPTZ NOT NULL,
    description     TEXT NOT NULL,
    severity        TEXT NOT NULL CHECK (severity IN ('LOW', 'ELEVATED', 'HIGH')),
    oil_impact      FLOAT DEFAULT 0,
    trigger_headline TEXT,
    peak_score      FLOAT,
    notes           TEXT,
    CONSTRAINT shock_events_date_unique UNIQUE (event_date)
);

CREATE INDEX idx_shock_events_date ON shock_events (event_date DESC);

CREATE TABLE IF NOT EXISTS crypto_prices (
    id              BIGSERIAL,
    symbol          TEXT            NOT NULL,
    fetched_at      TIMESTAMPTZ     NOT NULL,
    open_usd        FLOAT           NOT NULL,
    high_usd        FLOAT           NOT NULL,
    low_usd         FLOAT           NOT NULL,
    close_usd       FLOAT           NOT NULL,
    volume_usd      FLOAT           NOT NULL,
    change_pct      FLOAT,
    source          TEXT            DEFAULT 'binance',
    interval_minutes INTEGER        DEFAULT 1440,  -- 1440 minutes = 1 day
    PRIMARY KEY (id, fetched_at)
);
CREATE TABLE crypto_prices (
    id BIGSERIAL PRIMARY KEY,
    symbol TEXT,
    fetched_at TIMESTAMPTZ,
    open_usd FLOAT,
    high_usd FLOAT,
    low_usd FLOAT,
    close_usd FLOAT,
    volume_usd FLOAT,
    change_pct FLOAT,
    source TEXT DEFAULT 'binance',
    interval_minutes INTEGER DEFAULT 1440,  -- 30, 60, 240, 1440, etc.
    created_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_crypto_prices_symbol_time ON crypto_prices (symbol, fetched_at DESC);
CREATE INDEX idx_crypto_prices_interval ON crypto_prices (interval_minutes);

CREATE TABLE IF NOT EXISTS fear_greed_index (
    date        DATE        PRIMARY KEY,
    value       FLOAT       NOT NULL,  -- 0-100 raw value
    fetched_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS funding_rates (
    symbol      TEXT        NOT NULL,
    date        DATE        NOT NULL,
    rate_avg    FLOAT       NOT NULL,  -- daily average of 3 readings
    fetched_at  TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (symbol, date)
);

-- -----------------------------------------------------------------------------
-- Macro + ETF ingestion tables (idempotent upserts with revision tracking)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS macro_releases (
    id                 BIGSERIAL PRIMARY KEY,
    source             TEXT        NOT NULL,    -- fred, tradingeconomics, etc.
    series_code        TEXT        NOT NULL,    -- FED_FUNDS, CPIAUCSL, PAYEMS, ...
    event_name         TEXT        NOT NULL,    -- FED_RATE, CPI, NFP, ...
    country            TEXT        DEFAULT 'US',
    release_time_utc   TIMESTAMPTZ NOT NULL,
    period_date        DATE,                    -- reference period (month/day)
    expected_value     FLOAT,
    actual_value       FLOAT,
    previous_value     FLOAT,
    surprise_value     FLOAT,                   -- typically actual - expected
    unit               TEXT,                    -- %, jobs_k, index, etc.
    importance         TEXT,                    -- low/medium/high
    revision_count           INTEGER     NOT NULL DEFAULT 0,
    row_hash                 TEXT        NOT NULL,    -- hash of key economic payload
    raw_payload              JSONB       NOT NULL DEFAULT '{}'::jsonb,
    -- Set once when actual_value is first observed; NULL until then.
    -- Used for point-in-time (as-of) joins during ML training so that
    -- actual/surprise are only visible after this timestamp.
    actual_first_available_at TIMESTAMPTZ,
    first_seen_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT macro_releases_unique
        UNIQUE (source, series_code, release_time_utc, period_date)
);

CREATE INDEX IF NOT EXISTS idx_macro_releases_time
    ON macro_releases (release_time_utc DESC);
CREATE INDEX IF NOT EXISTS idx_macro_releases_event
    ON macro_releases (event_name, release_time_utc DESC);

CREATE TABLE IF NOT EXISTS etf_flows (
    id                   BIGSERIAL PRIMARY KEY,
    source               TEXT        NOT NULL,  -- farside, coinglass, custom, ...
    flow_date            DATE        NOT NULL,
    ticker               TEXT        NOT NULL,  -- IBIT, FBTC, ...
    net_flow_usd         FLOAT,
    net_flow_btc         FLOAT,
    aum_usd              FLOAT,
    volume_usd           FLOAT,
    premium_discount_pct FLOAT,
    revision_count       INTEGER     NOT NULL DEFAULT 0,
    row_hash             TEXT        NOT NULL,
    raw_payload          JSONB       NOT NULL DEFAULT '{}'::jsonb,
    first_seen_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT etf_flows_unique
        UNIQUE (source, flow_date, ticker)
);

CREATE INDEX IF NOT EXISTS idx_etf_flows_date
    ON etf_flows (flow_date DESC);
CREATE INDEX IF NOT EXISTS idx_etf_flows_ticker_date
    ON etf_flows (ticker, flow_date DESC);

-- -----------------------------------------------------------------------------
-- Runtime strategy policies (versioned, hot-reloadable trading config)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS strategy_policies (
    id                BIGSERIAL PRIMARY KEY,
    policy_name       TEXT        NOT NULL DEFAULT 'default',
    engine_name       TEXT        NOT NULL,  -- e.g. scalp_live, llm_live
    account_type      TEXT        NOT NULL,  -- e.g. live, testnet
    version           INTEGER     NOT NULL,
    status            TEXT        NOT NULL DEFAULT 'draft'
                                  CHECK (status IN ('draft', 'validated', 'active', 'rejected', 'retired')),
    policy_json       JSONB       NOT NULL DEFAULT '{}'::jsonb,
    validation_report JSONB       NOT NULL DEFAULT '{}'::jsonb,
    reason            TEXT,
    source            TEXT        NOT NULL DEFAULT 'manual',
    created_by        TEXT,
    base_policy_id    BIGINT REFERENCES strategy_policies(id) ON DELETE SET NULL,
    effective_from    TIMESTAMPTZ,
    activated_at      TIMESTAMPTZ,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT strategy_policies_version_unique
        UNIQUE (policy_name, engine_name, account_type, version)
);

CREATE INDEX IF NOT EXISTS idx_strategy_policies_lookup
    ON strategy_policies (engine_name, account_type, status, effective_from DESC, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_strategy_policies_created_at
    ON strategy_policies (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_strategy_policies_policy_json
    ON strategy_policies USING GIN (policy_json);

CREATE UNIQUE INDEX IF NOT EXISTS idx_strategy_policies_one_active_per_scope
    ON strategy_policies (engine_name, account_type)
    WHERE status = 'active';

-- -----------------------------------------------------------------------------
-- Policy review run history
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS policy_review_runs (
    id                         BIGSERIAL PRIMARY KEY,
    run_at                     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    engine_name                TEXT        NOT NULL,
    account_type               TEXT        NOT NULL,
    reviewer_model             TEXT,
    guard_decision             TEXT        NOT NULL CHECK (guard_decision IN ('HOLD', 'ALLOW_REVIEW')),
    guard_reason               TEXT        NOT NULL,
    llm_called                 BOOLEAN     NOT NULL DEFAULT FALSE,
    llm_decision               TEXT,
    llm_reason                 TEXT,
    proposed_patch             JSONB       NOT NULL DEFAULT '{}'::jsonb,
    guard_payload              JSONB       NOT NULL DEFAULT '{}'::jsonb,
    llm_payload                JSONB       NOT NULL DEFAULT '{}'::jsonb,
    active_policy_id           BIGINT,
    active_policy_version      INTEGER,
    hours_since_policy_update  FLOAT,
    closed_trades_since_update INTEGER,
    created_at                 TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_policy_review_runs_scope_time
    ON policy_review_runs (engine_name, account_type, run_at DESC);

CREATE INDEX IF NOT EXISTS idx_policy_review_runs_guard_decision
    ON policy_review_runs (guard_decision, run_at DESC);

CREATE INDEX IF NOT EXISTS idx_policy_review_runs_llm_decision
    ON policy_review_runs (llm_decision, run_at DESC);

CREATE INDEX IF NOT EXISTS idx_policy_review_runs_guard_payload
    ON policy_review_runs USING GIN (guard_payload);

CREATE INDEX IF NOT EXISTS idx_policy_review_runs_llm_payload
    ON policy_review_runs USING GIN (llm_payload);

-- -----------------------------------------------------------------------------
-- Generic observability traces
-- -----------------------------------------------------------------------------
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
