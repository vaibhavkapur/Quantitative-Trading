# Dependency, Licensing & IP Audit

## 1. Linked External Foundations (Plan §2)

`plan.md` §2 references three personal GitHub repositories as candidate foundations:
1. `vaibhavkapur/Personal-Finance-Agents` (`projects/Personal-Trading-Desk`)
2. `vaibhavkapur/High-Frequency-Trading-Engine`
3. `vaibhavkapur/Trading-Software` (`projects/13-broker-reliability-reconciliation`)

Per `plan.md` §2 governance requirements:
> *"Confirm access, licenses, background IP and SAAT ownership terms before copying or adapting any personal repository. ... If access or licensing is unresolved, implement against clean, documented contracts instead."*

**Audit Decision**: None of the three external repositories were copied or vendored into this codebase because formal license clearance and SAAT IP ownership confirmation are pending (and the local build environment operates hermetically without external GitHub access). Instead, clean, well-documented adapter contracts (`PaperOMSAdapter`, `BrokerStatement`, `ReconciliationEngine`) and canonical event schemas were authored from scratch so future adapters to those repositories—once cleared—can plug in cleanly without changing the core accounting engine.

## 2. Third-Party Python Dependencies

All runtime and development dependencies are open-source, pinned via `pyproject.toml` and `uv.lock`, and contain zero proprietary code or embedded secrets:

| Package | Scope | License | Purpose |
|---|---|---|---|
| `pydantic` (`>=2.7,<3`) | Runtime | MIT | Strict schema validation, frozen domain models, Decimal & UTC enforcement |
| `fastapi` (`>=0.115,<1`) | Runtime | MIT | Typed REST API and OpenAPI schema generation |
| `uvicorn` (`>=0.30,<1`) | Runtime | BSD-3-Clause | Local ASGI server for API and Portfolio/Reconciliation Web UI |
| `pytest` (`>=8,<9`) | Dev/Test | MIT | Unit, integration, and 8 required scenario verification tests |
| `httpx` (`>=0.27,<1`) | Dev/Test | BSD-3-Clause | FastAPI `TestClient` integration testing |
| `ruff` (`>=0.6,<1`) | Dev/Lint | MIT | Static analysis and code style verification |

## 3. Data Provenance & Entitlements

All fixtures in `data/fixtures/` (`manifest.json`, `baseline_events.json`, `market_marks.json`, `broker_statement.json`) are 100% synthetic, deterministically authored datasets (`SYNTHETIC_PERMITTED`). Zero proprietary market vendor data, live broker statements, or client PII are included.

## 4. Security & Deployment Approval Gate

- No credentials, API keys, or secrets exist in source code or `config/`.
- `config/settings.json` sets `"live_order_execution_enabled": false` and `"allow_history_mutation": false`.
- Connecting real prime brokers, live OMS gateways, or paid pricing feeds requires a separate security, data-licensing, legal/regulatory, and risk approval sign-off.
