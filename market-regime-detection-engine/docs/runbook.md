# Local operations

## Build and review

1. Run `scripts/setup` from a checkout with Python 3.9+ and its standard SQLite module. Setup checks the synthetic fixture against its generator and manifest. No package manager, network or credentials are required.
2. Run `scripts/validate` for the same setup, test and demo entry point used by CI. The HTTP integration test needs a temporary IPv4 loopback port. Unit and scenario tests use no sockets.
3. Inspect `.local/evaluation.json` and its run ID. Review coverage, reasons for unknown, changes per 100 adjacent observations, direct known changes, median label duration, drift blocks and cost-adjusted strategy outcomes.
4. Start the read service using the token example in the README. The CLI writer needs read access to the input/config and write access to the destination. The service uses SQLite read-only mode. Separate the filesystem identities for readers and writers when sharing a host. Keep tokens outside the repository. The launcher creates private files with umask 077. Rotate a token by restarting the service with a fresh environment value.

## Failure and recovery

- **No ready history or HTTP 503.** Check the configured database path, read permissions and whether a run completed. Rebuild using the original input, settings and engine version. SQLite commits metadata, history and evaluation together. An interrupted transaction rolls back. A retry for a completed run returns the same identity and does not append rows.
- **Corrupt database.** Stop the reader and build a new database path with `scripts/regime run --db .local/recovered.sqlite3`. Compare run ID, counts and the report, then start the service against the verified replacement. Preserve the original file for diagnosis.
- **CSV or order rejection.** The CLI exits nonzero and records the error on stderr. Inspect the named field or line and confirm the provenance and availability times upstream. Replay a corrected immutable snapshot as a new run.
- **Unknown or drift spike.** Inspect decision reasons and recent feature history as of the affected arrival. Check missing closes, warmup length and input units before interpreting a shift. Keep downstream consumers in their documented unknown-state fallback. The example strategy targets cash on unknown once that decision is available. Changes to training cutoff, costs or thresholds belong in a new configuration and run.
- **Cold start.** Wait for the configured training minimum and full trailing window. A frozen model still needs confirmation bars to publish its first known state.
- **Unexpected 401 or 400.** Confirm the bearer token in the reader's environment and use UTC timestamps with the exclusive pagination cursor. The health route reveals readiness only.

## Evidence and rollout

Use `python3 scripts/fixture.py --check` to verify the committed CSV and manifest. Use `PYTHONPATH=src python3 scripts/benchmark.py` to repeat single-process fit and scoring timings. Wall-clock timing varies by host. Compare model digests on matching Python, math library and CPU environments. Compare labels and numerical metrics with tolerances when changing runtimes. Use `scripts/regime run --report path.json` to export another evaluation.

The local inputs have recorded synthetic provenance and the runtime has no external data entitlement or order-routing dependency. The service is scoped to a trusted local host. A shared service needs its own TLS, identity, rate limiting, host hardening, retention rules and security review. Actual SAAT ownership, approved datasets, asset class, strategy horizon and operating constraints require owner confirmation. Provider access, paper deployment and any real-capital use require separate approvals. This prototype has no execution interface.
