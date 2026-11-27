# Contracts

## Bars and versions

`bars.v1` CSV uses the exact ordered header below. Empty numeric fields encode missing observations. A literal NaN, infinity, zero price, negative spread, invalid timestamp, duplicate daily event, regressing availability or series mismatch fails the run. Prices and spread must be finite. UTC timestamps have second precision. One event per UTC date and strictly increasing availability are required. Availability must be at or after event time. Revisions require a new input snapshot and run.

```csv
series,event_at,available_at,close,benchmark_close,spread_bps
```

`series` is the stable asset/benchmark pair identifier. Both closes are in the same currency, USD in the fixture. `spread_bps` is the quoted full bid–ask spread in basis points of midprice. Stored decisions use the composite identity (run ID, availability timestamp). The fixture manifest identifies the schema, generator, seed, entitlement and source SHA-256. Config schema `daily.v1` sets the series, training cutoff, rolling window, gap tolerance, stress thresholds, confidence floor, confirmation run length, support radius, EM iterations and evaluation cost.

## Features and decisions

Version `trailing-daily.v1` uses the following order. With N trailing paired close-to-close log returns, compute

- `annual_vol` from their asset sample standard deviation times the square root of 252, in fractional units.
- `trend_t` from the mean asset return times the square root of N divided by its sample standard deviation.
- `spread_bps` from the arithmetic mean of the N spreads on return-ending bars.
- `benchmark_corr` from the Pearson correlation of the N paired asset and benchmark returns.

A fresh window needs N+1 consecutive usable closes. A missing field or event gap above the configured number of calendar days restarts the window. Degenerate price variance yields `flat_window`. An unavailable feature vector resets both decision gates and the HMM posterior. Input status is one of `ready`, `warmup`, `missing_input`, `gap` or `flat_window`.

With `PYTHONPATH=src`, a caller may use the same incremental interface as the replay command.

```python
from regime.domain import Settings, read_bars
from regime.engine import OnlineEngine

cfg = Settings.load("config/daily.v1.json")
bars = list(read_bars("data/fixtures/daily.csv"))
training = [bar for bar in bars if bar.available_at <= cfg.train_through]
engine, training_features = OnlineEngine.fit(training, cfg)
first_decision = engine.observe(bars[len(training)])
```

Feed subsequent bars to the same instance in arrival order from one caller. Fitting requires at least the greater of 30 or twice the feature window in complete training vectors. A cutoff violation or invalid bar raises `ContractError`.

Every observation returns event and availability timestamps, series, features or null, feature status, HMM support distance and separate `rule` and `hmm` decisions. Each decision carries the published label, raw candidate, candidate confidence, three normalized scores and a reason. Missing inputs return an empty score map, null candidate and zero confidence. The HMM support distance is the root mean squared standardized distance to the closest emission component, accounting for within-state feature variance.

Under the defaults, scores below 0.63 abstain immediately. The HMM also abstains at a support distance above 4 and restarts its forward prior on the next complete observation. Low confidence clears the published label and pending transition. The HMM still carries its filtered distribution for an in-support low-confidence observation. Three consecutive confident candidates are required to establish or change a label. During confirmation the last published label persists, initially unknown. Confidence always belongs to the candidate, which may differ from the published label.

## Read service

All endpoints use HTTP GET with JSON responses on IPv4 loopback. All versioned routes require an `Authorization: Bearer <token>` header. Timestamps in history queries are availability times.

- `/health` returns readiness. A missing database or absent complete run yields HTTP 503.
- `/v1/runs` lists completed run IDs.
- `/v1/runs/{id}/metadata` returns settings, model parameters, feature order, versions, hashes and final training arrival.
- `/v1/runs/{id}/history` returns the full decisions in ascending availability order.
- `/v1/runs/{id}/features` returns the same observations with only timestamps, series, feature status and features.
- `/v1/runs/{id}/report` returns the full retrospective holdout evaluation. Use this for review only. Point-in-time consumers must query history or features.

History and features accept inclusive `as_of`, exclusive `after` and integer `limit` (default 100, range 1–500). Pass the last response's `next_after` as `after` while keeping the same run ID and `as_of`. A null cursor ends the page sequence. Each response contains `items` and `next_after`. Invalid page parameters return HTTP 400, missing credentials return 401, and an unknown run or resource returns 404. Error JSON includes a stable `error` key. The CLI prints structured failure details on stderr and exits nonzero.
