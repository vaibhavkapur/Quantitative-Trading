# Market Regime Detection Engine — Implementation Plan

| Field | Value |
|---|---|
| Project | 08 of 25 |
| Classification | New capability |
| Priority | Very high |
| Proposed owner | CTO / trading engineering (reviewers to be assigned) |
| Suggested stack | Python, NumPy, Polars, scikit-learn, optional HMM library, FastAPI |
| Initial deployment | Local reproducible prototype, then approved paper/sandbox environment |
| Status | Local synthetic-data MVP implemented and validated. See [acceptance evidence](reports/acceptance.md). |

## 1. Objective and scope

Classify market conditions from causal volatility, trend, liquidity and correlation features, exposing uncertainty and transition stability.

**MVP deliverables**
- Causal feature pipeline
- Rule-based and statistical regime models
- Online filtering and transition logic
- Regime history service
- Validation and drift report

**Explicit non-goals**
- No assumption regimes are objectively labeled or predict future returns reliably

## 2. Existing foundations and integration dependencies

- Greenfield capability in the prior GitHub coverage audit. Reuse common contracts and fixtures only after checking ownership and compatibility.

Start with a small independently runnable vertical slice. Do not create duplicate backtesters, order managers, ledgers or data ingestion services when a reviewed shared interface already exists.

**Dependencies / related plans:** #17, #18, #23.

Confirm access, licenses, background IP and SAAT ownership terms before copying or adapting any personal repository. A linked prototype is not proof of production readiness. If access or licensing is unresolved, implement against clean, documented contracts instead.

## 3. Proposed architecture and component boundaries

| Component / workstream | Responsibility and boundary |
|---|---|
| Causal feature pipeline | Own the canonical domain contract and validate incoming data; publish versioned inputs. |
| Rule-based and statistical regime models | Implement the core deterministic or research processing; isolate optional external dependencies. |
| Online filtering and transition logic | Persist results or state with reproducible provenance; expose explicit failure modes. |
| Regime history service | Expose integration or operational interfaces; apply authorization and fail-safe validation. |
| Validation and drift report | Deliver reviewable evidence, operator workflow and integration tests; do not bypass approvals. |

### Suggested repository layout

```text
market-regime-detection-engine/
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

- [x] Define regime taxonomy, evaluation horizon and benchmark features; build causal feature pipeline.
- [x] Document interfaces, assumptions and design decisions introduced in this milestone.
- [x] Add automated tests and store a reproducible demonstration or validation report.

### M2 — Core implementation

- [x] Implement threshold baseline and statistical model such as HMM; fit on training windows only.
- [x] Document interfaces, assumptions and design decisions introduced in this milestone.
- [x] Add automated tests and store a reproducible demonstration or validation report.

### M3 — Integration and validation

- [x] Add online inference, confidence, hysteresis and unknown-state fallback.
- [x] Document interfaces, assumptions and design decisions introduced in this milestone.
- [x] Add automated tests and store a reproducible demonstration or validation report.

### M4 — Hardening and delivery

- [x] Evaluate regime stability and downstream strategy behavior on held-out periods; publish feature API and report.
- [x] Document interfaces, assumptions and design decisions introduced in this milestone.
- [x] Add automated tests and store a reproducible demonstration or validation report.

## 5. Testing and validation

Implement unit tests for domain invariants, integration tests for adapter contracts and scenario tests for failure paths. Use fixed seeds where applicable; avoid testing only happy paths.

| Scenario | Required verification |
|---|---|
| Future smoothing leak | Reproduce the case with a fixture or injected fault; assert the documented correct outcome, rejection or recovery and retain evidence. |
| Noisy threshold oscillation | Reproduce the case with a fixture or injected fault; assert the documented correct outcome, rejection or recovery and retain evidence. |
| Missing prices | Reproduce the case with a fixture or injected fault; assert the documented correct outcome, rejection or recovery and retain evidence. |
| Structural shift | Reproduce the case with a fixture or injected fault; assert the documented correct outcome, rejection or recovery and retain evidence. |
| Scaler fit leakage | Reproduce the case with a fixture or injected fault; assert the documented correct outcome, rejection or recovery and retain evidence. |
| Cold start | Reproduce the case with a fixture or injected fault; assert the documented correct outcome, rejection or recovery and retain evidence. |

For quantitative work, use chronological splits, point-in-time inputs, realistic costs where relevant, and report failed experiments as well as successful ones. For execution systems, test idempotency, uncertain outcomes and state recovery. Never infer live performance from a simulated test.

## 6. Acceptance criteria / definition of done

- [x] Online inference never uses future observations.
- [x] Low-confidence predictions can return unknown.
- [x] Regime changes are reproducible for the same feature stream.
- [x] Evaluation compares simple baseline and reports transition frequency.
- [x] Downstream tests include costs.
- [x] A clean checkout can run the documented setup, tests and demo with permitted fixtures.
- [x] Architecture, dependency assumptions, known limitations and runbook are committed.
- [x] The CI validation command passes from a clean local checkout and raw evaluation and test evidence is retained. Hosted CI is pending repository connection.
- [x] Local security, synthetic data provenance, IP assumptions and deployment approval requirements are documented in the [runbook](docs/runbook.md). Organizational sign-off for external data or deployment remains pending.

## 7. Operational controls and rollout

- Use synthetic or explicitly permitted historical data first; record provenance and entitlements.
- Store credentials outside code, separate roles, and restrict access to sensitive artifacts.
- Keep order execution disabled unless a deterministic, independently reviewed policy explicitly permits paper/sandbox operation. Any real-capital deployment is a separate approval decision requiring applicable legal/regulatory, broker, security and risk checks.
- Capture health, failure and decision metrics; document the operator response and safe fallback.
- Establish baseline measurements before setting throughput, latency, accuracy or availability targets; report environment, sample size and uncertainty rather than claiming unmeasured performance.

## 8. Handoff and evidence

Deliver source code, dependency lockfile or reproducible build instructions, architecture diagram, API/schema documentation, test report, demo script, benchmark/evaluation artifacts where applicable, and an explicit list of unsupported features. The project is complete at MVP scope only after its acceptance checklist has evidence attached.

**Next decision:** confirm the actual SAAT asset class, strategy horizon, data access, operating constraints and code ownership before selecting implementation depth or connecting real providers.
