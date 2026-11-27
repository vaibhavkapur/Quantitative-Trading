# ADR-002: Approval-Gated Paper Execution, Dual Control, and Tamper-Evident Audit Trail

- **Status:** Accepted
- **Date:** 2026-11-27

## Context

Capital allocation changes carry high operational and risk impact. `plan.md` specifies explicit non-goals (no automated live reallocation without approval) and requires reviewable evidence, independent constraint checks, idempotency, and role separation.

## Decision

1. Every allocation run generates a persisted `RebalanceProposal` in `PENDING_APPROVAL` status paired with an independently computed `FeasibilityReport` (`ConstraintVerifier`).
2. `ProposalService.review_proposal` enforces **dual control** (`reviewer_id != proposer_id`) and proposal TTL expiry (`proposal_ttl_minutes`).
3. `PaperPortfolioLedger.apply_approved_proposal` refuses any proposal not in `APPROVED` status, rejects any `mode != "paper"` with `LiveExecutionProhibitedError`, and enforces idempotent execution via persisted `idempotency_key` receipts.
4. All lifecycle events are appended to an immutable SHA-256 hash-chained log (`SQLiteAuditStore`), verifiable at any time via `verify_chain()`.

## Consequences

- Unapproved, expired, or self-approved proposals cannot mutate paper positions.
- Retries after network or process uncertainty cannot double-apply trades or double-charge transaction costs.
- Post-hoc database tampering is deterministically detected.
