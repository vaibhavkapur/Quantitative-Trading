# ADR-001: Clean-Room Adapter Contracts for External Portfolio Foundations

- **Status**: Accepted
- **Date**: 2026-11-27
- **Milestone**: M1 — Foundation and design

## Context

`plan.md` Section 2 references four personal portfolio prototypes (`Quantum-Trading` projects 01, 07, 10 and `Trading-Software` project 14) and requires verifying public access, licenses, background IP, and SAAT ownership terms before copying or adapting code. Those repositories could not be verified for public access or SAAT IP clearance during MVP implementation.

## Decision

Implement clean-room adapter contracts (`ExternalSignalAdapter` in `src/alpha_engine/adapters/`) that mirror the signal boundaries of statistical arbitrage, pairs spread, order-book imbalance, and AI research-agent proposals while recording explicit `AdapterProvenance` (`ip_clearance_status: "unresolved_clean_room_contract_only"`).

## Consequences

- **Positive**: Zero unverified external code or IP contamination; canonical PIT signal contracts and integration tests are established immediately; future approved implementations can swap in behind the adapter protocol without touching core evaluation or catalog logic.
- **Trade-off**: Adapter implementations represent baseline statistical/proxy formulations rather than tuned upstream prototypes.
