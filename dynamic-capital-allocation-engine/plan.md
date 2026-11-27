# Dynamic Capital Allocation Engine — Implementation Plan

| Field | Value |
|---|---|
| Project | 09 of 25 |
| Classification | Extension / integration of existing portfolio foundations |
| Priority | Very high |
| Proposed owner | CTO / trading engineering (reviewers to be assigned) |
| Suggested stack | Python, NumPy, SciPy, CVXPY or equivalent, PostgreSQL, FastAPI |
| Initial deployment | Local reproducible prototype, then approved paper/sandbox environment |
| Status | Proposed; not an assertion of SAAT's existing capability |

## 1. Objective and scope

Allocate a fund-level risk and capital budget across independently tracked strategies subject to exposure, liquidity and turnover constraints.

**MVP deliverables**
- Strategy performance and exposure snapshots
- Equal-weight and risk-budget baselines
- Constrained allocator
- Rebalance proposal and feasibility report
- Allocation audit trail

**Explicit non-goals**
- No automated live reallocation without approval
- no guaranteed target Sharpe ratio

## 2. Existing foundations and integration dependencies

- [Quantum-Trading — 02-hybrid-quantum-portfolio-optimizer](https://github.com/vaibhavkapur/Quantum-Trading/tree/main/projects/02-hybrid-quantum-portfolio-optimizer)
- [Quantum-Trading — 11-quantum-rebalancing](https://github.com/vaibhavkapur/Quantum-Trading/tree/main/projects/11-quantum-rebalancing)
- [Quantum-Trading — 08-quantum-risk-engine](https://github.com/vaibhavkapur/Quantum-Trading/tree/main/projects/08-quantum-risk-engine)

Before coding, inspect the linked repository at a pinned commit: inventory reusable contracts, tests and documented limitations. Prefer an adapter or incremental extension over duplicating the implementation. Preserve baseline tests and benchmark evidence.

**Dependencies / related plans:** #10, #12, #17, #22; existing pre-trade risk engine.

Confirm access, licenses, background IP and SAAT ownership terms before copying or adapting any personal repository. A linked prototype is not proof of production readiness. If access or licensing is unresolved, implement against clean, documented contracts instead.

## 3. Proposed architecture and component boundaries

| Component / workstream | Responsibility and boundary |
|---|---|
| Strategy performance and exposure snapshots | Own the canonical domain contract and validate incoming data; publish versioned inputs. |
| Equal-weight and risk-budget baselines | Implement the core deterministic or research processing; isolate optional external dependencies. |
| Constrained allocator | Persist results or state with reproducible provenance; expose explicit failure modes. |
| Rebalance proposal and feasibility report | Expose integration or operational interfaces; apply authorization and fail-safe validation. |
| Allocation audit trail | Deliver reviewable evidence, operator workflow and integration tests; do not bypass approvals. |

### Suggested repository layout

```text
dynamic-capital-allocation-engine/
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

- [ ] Audit optimizer and rebalance contracts; define strategy NAV, returns, capital eligibility and constraint schema.
- [ ] Document interfaces, assumptions and design decisions introduced in this milestone.
- [ ] Add automated tests and store a reproducible demonstration or validation report.

### M2 — Core implementation

- [ ] Implement equal-weight, inverse-volatility and constrained optimizer using point-in-time estimates.
- [ ] Document interfaces, assumptions and design decisions introduced in this milestone.
- [ ] Add automated tests and store a reproducible demonstration or validation report.

### M3 — Integration and validation

- [ ] Add transaction costs, caps, correlation stress, no-feasible-solution fallback and proposal review.
- [ ] Document interfaces, assumptions and design decisions introduced in this milestone.
- [ ] Add automated tests and store a reproducible demonstration or validation report.

### M4 — Hardening and delivery

- [ ] Integrate paper positions and compare allocation alternatives using walk-forward evaluation.
- [ ] Document interfaces, assumptions and design decisions introduced in this milestone.
- [ ] Add automated tests and store a reproducible demonstration or validation report.

## 5. Testing and validation

Implement unit tests for domain invariants, integration tests for adapter contracts and scenario tests for failure paths. Use fixed seeds where applicable; avoid testing only happy paths.

| Scenario | Required verification |
|---|---|
| Non-psd covariance | Reproduce the case with a fixture or injected fault; assert the documented correct outcome, rejection or recovery and retain evidence. |
| Zero volatility | Reproduce the case with a fixture or injected fault; assert the documented correct outcome, rejection or recovery and retain evidence. |
| Insufficient history | Reproduce the case with a fixture or injected fault; assert the documented correct outcome, rejection or recovery and retain evidence. |
| Infeasible caps | Reproduce the case with a fixture or injected fault; assert the documented correct outcome, rejection or recovery and retain evidence. |
| Stale positions | Reproduce the case with a fixture or injected fault; assert the documented correct outcome, rejection or recovery and retain evidence. |
| Turnover limit | Reproduce the case with a fixture or injected fault; assert the documented correct outcome, rejection or recovery and retain evidence. |
| Correlated drawdowns | Reproduce the case with a fixture or injected fault; assert the documented correct outcome, rejection or recovery and retain evidence. |

For quantitative work, use chronological splits, point-in-time inputs, realistic costs where relevant, and report failed experiments as well as successful ones. For execution systems, test idempotency, uncertain outcomes and state recovery. Never infer live performance from a simulated test.

## 6. Acceptance criteria / definition of done

- [ ] Allocation sum and cash reserve reconcile.
- [ ] Every constraint is independently checked.
- [ ] Infeasible problems return safe no-change or cash fallback.
- [ ] Decisions expose inputs and objective.
- [ ] Baselines evaluated with same costs and periods.
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
