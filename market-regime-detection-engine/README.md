# Market Regime Detection Engine

This local prototype labels an instrument's daily market conditions as rising, falling, stressed or unknown. It computes trailing features from prices, benchmark prices and spreads, compares a threshold rule with a fitted Gaussian hidden Markov model (HMM), and serves the decisions available at a given UTC time.

## Run locally

From this directory, with Python 3.9+ including SQLite, run

```sh
scripts/setup
scripts/demo
scripts/validate
```

Setup verifies the bundled synthetic fixture. The demo prints the run ID and transition counts, then writes the full evaluation to `.local/evaluation.json` and history to `.local/history.sqlite3`. Validation runs setup, all tests including a temporary loopback HTTP server, and the demo. All commands work offline.

For another compatible series or a different cutoff, pass a CSV and a versioned settings file. Each run records both inputs' fingerprints.

```sh
scripts/regime run --bars data/fixtures/daily.csv --settings config/daily.v1.json \
  --db .local/history.sqlite3 --report .local/evaluation.json
```

## Query the history

Start the local read service in the same shell that holds the token. This example uses an ephemeral token and stops the server when finished.

```sh
export REGIME_READ_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
scripts/regime serve --port 8765 &
service_pid=$!
run_id=$(python3 -c 'import json; print(json.load(open(".local/evaluation.json"))["run_id"])')
curl --retry 5 --retry-connrefused -fsS http://127.0.0.1:8765/health
curl -fsS -H "Authorization: Bearer $REGIME_READ_TOKEN" \
  "http://127.0.0.1:8765/v1/runs/$run_id/history?as_of=2024-01-05T16:05:00Z&limit=2"
kill "$service_pid"
unset REGIME_READ_TOKEN
```

## Evidence and handoff

The supplied run trains on 260 synthetic daily bars and scores 195 later arrivals. The HMM produced 10 label changes, including transitions through unknown, compared with 11 for the rule. The HMM gave a known label on 153 bars. The rule gave one on 139. These measurements describe the fixture and configured horizon.

- [Design, assumptions and unsupported features](docs/architecture.md)
- [Input, feature, model and HTTP contracts](docs/contracts.md)
- [Operator and recovery runbook](docs/runbook.md)
- [Acceptance evidence and limitations](reports/acceptance.md)
- [Raw holdout evaluation](reports/evaluation.json), [test output](reports/validation.txt), [local timing measurements](reports/benchmark.json)
