-- PostgreSQL schema for Dynamic Capital Allocation Engine (DCAE)
-- Intended for approved paper/sandbox deployment; local prototype defaults to SQLite.
-- Do not store credentials in SQL or version-controlled configuration files.

CREATE TABLE IF NOT EXISTS dcae_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    schema_version TEXT NOT NULL,
    as_of_utc TIMESTAMPTZ NOT NULL,
    published_at_utc TIMESTAMPTZ NOT NULL,
    content_hash TEXT NOT NULL,
    provenance_json JSONB NOT NULL,
    payload_json JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_dcae_snapshots_as_of
    ON dcae_snapshots (as_of_utc DESC);

CREATE TABLE IF NOT EXISTS dcae_proposals (
    proposal_id TEXT PRIMARY KEY,
    idempotency_key TEXT NOT NULL UNIQUE,
    decision_id TEXT NOT NULL,
    snapshot_id TEXT NOT NULL REFERENCES dcae_snapshots(snapshot_id),
    config_version TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    allocator_name TEXT NOT NULL,
    status TEXT NOT NULL,
    proposer_id TEXT NOT NULL,
    reviewer_id TEXT,
    created_at_utc TIMESTAMPTZ NOT NULL,
    updated_at_utc TIMESTAMPTZ NOT NULL,
    expires_at_utc TIMESTAMPTZ NOT NULL,
    payload_json JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_dcae_proposals_status
    ON dcae_proposals (status, created_at_utc DESC);

CREATE TABLE IF NOT EXISTS dcae_audit_log (
    seq BIGSERIAL PRIMARY KEY,
    event_id TEXT NOT NULL UNIQUE,
    event_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    actor_id TEXT NOT NULL,
    occurred_at_utc TIMESTAMPTZ NOT NULL,
    prev_hash TEXT NOT NULL,
    entry_hash TEXT NOT NULL,
    payload_json JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_dcae_audit_entity
    ON dcae_audit_log (entity_id, seq ASC);

CREATE TABLE IF NOT EXISTS dcae_paper_executions (
    execution_id TEXT PRIMARY KEY,
    proposal_id TEXT NOT NULL REFERENCES dcae_proposals(proposal_id),
    idempotency_key TEXT NOT NULL UNIQUE,
    executed_by TEXT NOT NULL,
    executed_at_utc TIMESTAMPTZ NOT NULL,
    total_nav_usd DOUBLE PRECISION NOT NULL,
    cash_usd DOUBLE PRECISION NOT NULL,
    total_cost_usd DOUBLE PRECISION NOT NULL,
    receipt_json JSONB NOT NULL
);
