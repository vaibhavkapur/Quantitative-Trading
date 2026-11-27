-- Performance Attribution Engine — PostgreSQL Production / Sandbox Schema
-- Version: pae-v1.0.0
-- All timestamps are UTC (TIMESTAMPTZ). Monetary amounts carry explicit ISO-4217 currency.
-- Order execution is intentionally absent from this schema (attribution & reporting only).

CREATE TABLE IF NOT EXISTS pae_input_batches (
    batch_id            TEXT PRIMARY KEY,
    portfolio_id        TEXT NOT NULL,
    as_of_utc           TIMESTAMPTZ NOT NULL,
    availability_utc    TIMESTAMPTZ NOT NULL,
    reporting_currency  CHAR(3) NOT NULL,
    data_source         TEXT NOT NULL,
    entitlement_tag     TEXT NOT NULL,
    content_hash        CHAR(64) NOT NULL,
    payload_json        JSONB NOT NULL,
    created_at_utc      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pae_input_batches_portfolio_asof
    ON pae_input_batches (portfolio_id, as_of_utc);

CREATE TABLE IF NOT EXISTS pae_attribution_reports (
    version_id            TEXT PRIMARY KEY,
    report_id             TEXT NOT NULL,
    portfolio_id          TEXT NOT NULL,
    version_number        INTEGER NOT NULL CHECK (version_number >= 1),
    supersedes_version_id TEXT REFERENCES pae_attribution_reports(version_id),
    correction_reason     TEXT,
    methodology_version   TEXT NOT NULL,
    config_hash           CHAR(64) NOT NULL,
    input_hash            CHAR(64) NOT NULL,
    report_hash           CHAR(64) NOT NULL,
    period_start          DATE NOT NULL,
    period_end            DATE NOT NULL,
    reporting_currency    CHAR(3) NOT NULL,
    environment_mode      TEXT NOT NULL,
    status                TEXT NOT NULL CHECK (status IN ('DRAFT', 'REVIEWED', 'APPROVED', 'SUPERSEDED')),
    author_id             TEXT NOT NULL,
    reviewer_id           TEXT,
    approver_id           TEXT,
    created_at_utc        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    reviewed_at_utc       TIMESTAMPTZ,
    approved_at_utc       TIMESTAMPTZ,
    report_json           JSONB NOT NULL,
    UNIQUE (report_id, version_number),
    CHECK (reviewer_id IS NULL OR reviewer_id <> author_id),
    CHECK (approver_id IS NULL OR approver_id <> author_id)
);

CREATE INDEX IF NOT EXISTS idx_pae_reports_portfolio_period
    ON pae_attribution_reports (portfolio_id, period_start, period_end, version_number DESC);

CREATE TABLE IF NOT EXISTS pae_contribution_provenance (
    provenance_id       BIGSERIAL PRIMARY KEY,
    version_id          TEXT NOT NULL REFERENCES pae_attribution_reports(version_id) ON DELETE CASCADE,
    trade_date          DATE NOT NULL,
    strategy_id         TEXT NOT NULL,
    instrument_id       TEXT NOT NULL,
    sector              TEXT NOT NULL,
    position_ids_json   JSONB NOT NULL,
    trade_ids_json      JSONB NOT NULL,
    gross_contribution  DOUBLE PRECISION NOT NULL,
    fx_contribution     DOUBLE PRECISION NOT NULL,
    fee_drag            DOUBLE PRECISION NOT NULL,
    net_contribution    DOUBLE PRECISION NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pae_provenance_version
    ON pae_contribution_provenance (version_id, trade_date, strategy_id, instrument_id);

CREATE TABLE IF NOT EXISTS pae_data_quality_exceptions (
    exception_id        TEXT PRIMARY KEY,
    version_id          TEXT NOT NULL REFERENCES pae_attribution_reports(version_id) ON DELETE CASCADE,
    code                TEXT NOT NULL,
    severity            TEXT NOT NULL CHECK (severity IN ('INFO', 'WARNING', 'ERROR', 'CRITICAL')),
    trade_date          DATE,
    entity_id           TEXT,
    message             TEXT NOT NULL,
    details_json        JSONB NOT NULL,
    created_at_utc      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS pae_audit_log (
    seq_id              BIGSERIAL PRIMARY KEY,
    event_id            TEXT UNIQUE NOT NULL,
    timestamp_utc       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    actor_id            TEXT NOT NULL,
    actor_role          TEXT NOT NULL,
    action              TEXT NOT NULL,
    resource_id         TEXT NOT NULL,
    outcome             TEXT NOT NULL,
    details_json        JSONB NOT NULL,
    prev_hash           CHAR(64) NOT NULL,
    entry_hash          CHAR(64) NOT NULL
);
