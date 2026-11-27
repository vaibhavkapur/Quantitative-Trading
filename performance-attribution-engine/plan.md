# Performance Attribution Engine — Implementation Plan

| Field | Value |
|---|---|
| Project | 12 of 25 |
| Classification | Extension / integration of existing portfolio foundations |
| Priority | High |
| Proposed owner | CTO / trading engineering (reviewers to be assigned) |
| Suggested stack | Python, Polars, NumPy, PostgreSQL, FastAPI, dashboard framework |
| Initial deployment | Local reproducible prototype, then approved paper/sandbox environment |
| Status | Proposed; not an assertion of SAAT's existing capability |

## 1. Objective and scope

Explain portfolio returns and execution costs by strategy, instrument, sector and selected risk factors using reconciled position history.

**MVP deliverables**
- Return and benchmark calculation
- Strategy/instrument contribution
- Optional Brinson or factor attribution
- Execution-cost attribution
- Reconciliation and report API

**Explicit non-goals**
- No assertion that factor correlation proves causation
- no misleading attribution of gross as net returns

## 2. Existing foundations and integration dependencies

- [Trading-Software — 09-transaction-cost-analysis](https://github.com/vaibhavkapur/Trading-Software/tree/main/projects/09-transaction-cost-analysis)
- [High-Frequency-Trading-Engine](https://github.com/vaibhavkapur/High-Frequency-Trading-Engine)
- [Quantum-Trading — 02-hybrid-quantum-portfolio-optimizer](https://github.com/vaibhavkapur/Quantum-Trading/tree/main/projects/02-hybrid-quantum-portfolio-optimizer)

Before coding, inspect the linked repository at a pinned commit: inventory reusable contracts, tests and documented limitations. Prefer an adapter or incremental extension over duplicating the implementation. Preserve baseline tests and benchmark evidence.

**Dependencies / related plans:** #10, #17, #21; existing TCA.

Confirm access, licenses, background IP and SAAT ownership terms before copying or adapting any personal repository. A linked prototype is not proof of production readiness. If access or licensing is unresolved, implement against clean, documented contracts instead.

## 3. Proposed architecture and component boundaries

| Component / workstream | Responsibility and boundary |
|---|---|
| Return and benchmark calculation | Own the canonical domain contract and validate incoming data; publish versioned inputs. |
| Strategy/instrument contribution | Implement the core deterministic or research processing; isolate optional external dependencies. |
| Optional Brinson or factor attribution | Persist results or state with reproducible provenance; expose explicit failure modes. |
| Execution-cost attribution | Expose integration or operational interfaces; apply authorization and fail-safe validation. |
| Reconciliation and report API | Deliver reviewable evidence, operator workflow and integration tests; do not bypass approvals. |

### Suggested repository layout

```text
performance-attribution-engine/
  plan.md
  README.md
  docs/              # architecture, ADRs, assumptions, runbook
  src/               # domain contracts, core logic, adapters
  config/            # versioned settings; no secrets
  data/fixtures/     # permitted or synthetic inputs and manifests
  tests/unit/
  tests/integration/
  tests/scenarios/
  scripts/           # setup, demo, reproduce, validate
  reports/           # evaluation, benchmarks, acceptance evidence
```

Adapt the directory layout to the language and existing repository. Define domain types and schemas before choosing a distributed deployment; start with the simplest process boundaries that satisfy the workload. Use stable identifiers, explicit units/currency where relevant, UTC event and availability timestamps, versioned configuration and structured errors. Separate external provider interfaces from business logic.

## 4. Implementation milestones

Milestones are ordered by dependency, not promised calendar dates. Each milestone ends with a runnable demo and passing relevant tests.

### M1 — Foundation and design

- [ ] Define return conventions, cash-flow adjustments, benchmark and attribution identity; audit existing P&L/TCA.
- [ ] Document interfaces, assumptions and design decisions introduced in this milestone.
- [ ] Add automated tests and store a reproducible demonstration or validation report.

### M2 — Core implementation

- [ ] Implement daily contributions and reconciliation to total return; account for fees and FX.
- [ ] Document interfaces, assumptions and design decisions introduced in this milestone.
- [ ] Add automated tests and store a reproducible demonstration or validation report.

### M3 — Integration and validation

- [ ] Add sector/factor analysis when appropriate data exists; link implementation shortfall to trades.
- [ ] Document interfaces, assumptions and design decisions introduced in this milestone.
- [ ] Add automated tests and store a reproducible demonstration or validation report.

### M4 — Hardening and delivery

- [ ] Build report with residuals and data-quality exceptions; verify against hand-worked portfolios.
- [ ] Document interfaces, assumptions and design decisions introduced in this milestone.
- [ ] Add automated tests and store a reproducible demonstration or validation report.

## 5. Testing and validation

Implement unit tests for domain invariants, integration tests for adapter contracts and scenario tests for failure paths. Use fixed seeds where applicable; avoid testing only happy paths.

| Scenario | Required verification |
|---|---|
| External cash flow | Reproduce the case with a fixture or injected fault; assert the documented correct outcome, rejection or recovery and retain evidence. |
| Zero nav | Reproduce the case with a fixture or injected fault; assert the documented correct outcome, rejection or recovery and retain evidence. |
| Fee corrections | Reproduce the case with a fixture or injected fault; assert the documented correct outcome, rejection or recovery and retain evidence. |
| Currency conversion | Reproduce the case with a fixture or injected fault; assert the documented correct outcome, rejection or recovery and retain evidence. |
| Missing benchmark | Reproduce the case with a fixture or injected fault; assert the documented correct outcome, rejection or recovery and retain evidence. |
| Rebalance | Reproduce the case with a fixture or injected fault; assert the documented correct outcome, rejection or recovery and retain evidence. |
| Rounding residual | Reproduce the case with a fixture or injected fault; assert the documented correct outcome, rejection or recovery and retain evidence. |

For quantitative work, use chronological splits, point-in-time inputs, realistic costs where relevant, and report failed experiments as well as successful ones. For execution systems, test idempotency, uncertain outcomes and state recovery. Never infer live performance from a simulated test.

## 6. Acceptance criteria / definition of done

- [ ] Attribution sums to total within documented tolerance or shows explicit residual.
- [ ] Return methodology is versioned.
- [ ] Benchmarks use aligned dates/currency.
- [ ] Every contribution links to positions/trades.
- [ ] Corrections create revised report versions.
- [ ] A clean checkout can run the documented setup, tests and demo with permitted fixtures.
- [ ] Architecture, dependency assumptions, known limitations and runbook are committed.
- [ ] Relevant CI checks pass and raw evaluation or test evidence is retained.
- [ ] Security, data licensing, IP and any deployment approval requirements are reviewed.

## 7. Operational controls and rollout

- Use synthetic or explicitly permitted historical data first; record provenance and entitlements.
- Store credentials outside code, separate roles, and restrict access to sensitive artifacts.
- Keep order execution disabled unless a deterministic, independently reviewed policy explicitly permits paper/sandbox operation. Any real-capital deployment is a separate approval decision requiring applicable legal/regulatory, broker, security and risk checks.
- Capture health, failure and decision metrics; document the operator response and safe fallback.
- Establish baseline measurements before setting throughput, latency, accuracy or availability targets; report environment, sample size and uncertainty rather than claiming unmeasured performance.

## 8. Handoff and evidence

Deliver source code, dependency lockfile or reproducible build instructions, architecture diagram, API/schema documentation, test report, demo script, benchmark/evaluation artifacts where applicable, and an explicit list of unsupported features. The project is complete at MVP scope only after its acceptance checklist has evidence attached.

**Next decision:** confirm the actual SAAT asset class, strategy horizon, data access, operating constraints and code ownership before selecting implementation depth or connecting real providers.
