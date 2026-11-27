-- Authoritative Multi-Strategy PMS Schema (PostgreSQL 15+ reference DDL)
-- Local prototype runs identical logical constraints on SQLite WAL in src/pms/ledger/store.py.

CREATE TABLE IF NOT EXISTS ledger_events (
    seq_no          BIGSERIAL PRIMARY KEY,
    event_id        TEXT NOT NULL UNIQUE,
    idempotency_key TEXT NOT NULL UNIQUE,
    event_type      TEXT NOT NULL CHECK (event_type IN (
                        'TRADE_FILL', 'CASH_TRANSFER', 'STRATEGY_ALLOCATION',
                        'FX_CONVERSION', 'FEE', 'CORPORATE_ACTION', 'ADJUSTMENT'
                    )),
    event_time      TIMESTAMPTZ NOT NULL,
    recorded_time   TIMESTAMPTZ NOT NULL,
    prev_hash       CHAR(64) NOT NULL,
    event_hash      CHAR(64) NOT NULL UNIQUE,
    payload_json    JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ledger_events_event_time ON ledger_events (event_time);
CREATE INDEX IF NOT EXISTS idx_ledger_events_event_type ON ledger_events (event_type);

-- Database-enforced append-only immutability guard
CREATE OR REPLACE FUNCTION prevent_ledger_event_mutation()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION 'IMMUTABILITY_VIOLATION: ledger_events history cannot be updated or deleted';
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_prevent_ledger_events_update ON ledger_events;
CREATE TRIGGER trg_prevent_ledger_events_update
    BEFORE UPDATE OR DELETE ON ledger_events
    FOR EACH ROW EXECUTE FUNCTION prevent_ledger_event_mutation();

-- Disposable materialized read-model cache (reconstructible deterministically from ledger_events)
CREATE TABLE IF NOT EXISTS materialized_state_cache (
    cache_key     TEXT PRIMARY KEY,
    last_seq_no   BIGINT NOT NULL,
    state_hash    CHAR(64) NOT NULL,
    snapshot_json JSONB NOT NULL,
    updated_at    TIMESTAMPTZ NOT NULL
);
