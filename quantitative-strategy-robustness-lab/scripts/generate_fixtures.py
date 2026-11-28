"""Generate deterministic valid, invalid, and boundary synthetic fixtures.

All data produced by this script is purely synthetic (fixed RNG seeds) with
provenance explicitly marked 'synthetic_fixture'. No real market accounts,
transactions, customer data, or proprietary artifacts are included.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from robustness_lab.io_adapters import save_return_series_csv
from robustness_lab.models import (
    CostModel,
    ParameterGridSpec,
    ReturnFrequency,
    ReturnSeriesSpec,
    TrialRecord,
    TrialStatus,
)
from robustness_lab.registry import ImmutableExperimentRegistry

ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = ROOT / "data" / "fixtures"


def _business_dates(n: int, start: str = "2024-01-02") -> tuple[str, ...]:
    idx = pd.bdate_range(start=start, periods=n)
    return tuple(d.strftime("%Y-%m-%d") for d in idx)


def _reset_dir(path: Path) -> None:
    if path.exists():
        for child in sorted(path.rglob("*"), reverse=True):
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
    path.mkdir(parents=True, exist_ok=True)


def build_known_signal_experiment() -> None:
    """3x3 grid with strong plateau signal + 1 logged FAILED trial (100% grid coverage)."""
    out = FIXTURES_DIR / "valid" / "known_signal_experiment"
    _reset_dir(out)
    rng = np.random.default_rng(202601)
    n_obs = 360
    ts = _business_dates(n_obs)

    # Simple synthetic benchmark: modest drift ~5% ann, ~12% vol
    bench = rng.normal(loc=0.0002, scale=0.0075, size=n_obs)
    # Latent signal factor shared weakly with independent strategy noise per config
    latent_alpha = rng.normal(loc=0.0011, scale=0.0055, size=n_obs)

    grid = ParameterGridSpec(
        grid_id="grid-known-signal-v1",
        version="1.0.0",
        parameters={"lookback": (15, 20, 25), "vol_target": (0.08, 0.10, 0.12)},
    )
    (out / "parameter_grid.json").write_text(json.dumps(grid.to_dict(), indent=2) + "\n", encoding="utf-8")
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "experiment_id": "exp-known-signal-v1",
                "dataset_version": "synth-2026.01",
                "frequency": "daily",
                "provenance": "synthetic_fixture",
                "description": "Synthetic 3x3 parameter plateau with true positive net drift and 1 logged failure.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    cost_model = CostModel(commission_bps=2.0, slippage_bps=2.0, borrow_annual_bps=40.0, annualization_factor=252)
    registry = ImmutableExperimentRegistry(out / "registry.jsonl")
    registry.initialize()

    seq = 0
    for cfg in grid.all_configurations():
        seq += 1
        lb = int(cfg["lookback"])
        vt = float(cfg["vol_target"])
        tid = f"trial_lb{lb}_vt{int(round(vt * 100)):02d}"

        # Intentionally log one corner configuration as FAILED to demonstrate full failure tracking
        if lb == 15 and vt == 0.12:
            rec = TrialRecord(
                trial_id=tid,
                sequence_number=seq,
                experiment_id="exp-known-signal-v1",
                strategy_name="synthetic_momentum_plateau",
                parameters=cfg,
                status=TrialStatus.FAILED,
                created_at_utc=f"2026-01-15T10:{seq:02d}:00Z",
                dataset_version="synth-2026.01",
                code_version="0.1.0",
                cost_model=cost_model,
                failure_reason="Risk budget constraint violated during training calibration (max leverage > 2.0).",
                tags={"provenance": "synthetic_fixture"},
            )
            registry.append(rec)
            continue

        # Smooth plateau centered at (lb=20, vt=0.10); meaningful gradient toward grid edges
        d_lb = abs(lb - 20) / 5.0
        d_vt = abs(vt - 0.10) / 0.02
        dist_sq = d_lb ** 2 + d_vt ** 2
        alpha_weight = max(0.25, 1.0 - 0.30 * dist_sq)
        drift_bonus = 0.00060 * max(0.0, 1.0 - 0.45 * dist_sq)
        scale = vt / 0.10
        idio = rng.normal(loc=0.0, scale=0.0028 * scale, size=n_obs)
        gross = np.clip((alpha_weight * latent_alpha + idio) * scale + drift_bonus, -0.5, 0.5)
        turnover = np.full(n_obs, 0.05)
        short_exp = np.full(n_obs, 0.25)

        spec = ReturnSeriesSpec(
            series_id=tid,
            version="synth-2026.01",
            frequency=ReturnFrequency.DAILY,
            timestamps=ts,
            gross_returns=tuple(float(x) for x in gross),
            turnover=tuple(float(x) for x in turnover),
            short_exposure=tuple(float(x) for x in short_exp),
            benchmark_returns=tuple(float(x) for x in bench),
            provenance="synthetic_fixture",
        )
        save_return_series_csv(spec, out / "series" / f"{tid}.csv")

        rec = TrialRecord(
            trial_id=tid,
            sequence_number=seq,
            experiment_id="exp-known-signal-v1",
            strategy_name="synthetic_momentum_plateau",
            parameters=cfg,
            status=TrialStatus.COMPLETED,
            created_at_utc=f"2026-01-15T10:{seq:02d}:00Z",
            dataset_version="synth-2026.01",
            code_version="0.1.0",
            cost_model=cost_model,
            series_id=tid,
            return_series_sha256=spec.content_sha256,
            tags={"provenance": "synthetic_fixture"},
        )
        registry.append(rec)


def build_null_overfit_experiment() -> None:
    """4x4 grid of pure-noise null trials demonstrating multiple-testing rejection."""
    out = FIXTURES_DIR / "valid" / "null_overfit_experiment"
    _reset_dir(out)
    rng = np.random.default_rng(202602)
    n_obs = 360
    ts = _business_dates(n_obs)
    bench = rng.normal(loc=0.0002, scale=0.0075, size=n_obs)

    grid = ParameterGridSpec(
        grid_id="grid-null-noise-v1",
        version="1.0.0",
        parameters={"fast": (5, 10, 15, 20), "slow": (40, 60, 80, 100)},
    )
    (out / "parameter_grid.json").write_text(json.dumps(grid.to_dict(), indent=2) + "\n", encoding="utf-8")
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "experiment_id": "exp-null-noise-v1",
                "dataset_version": "synth-2026.02",
                "frequency": "daily",
                "provenance": "synthetic_fixture",
                "description": "16 independent zero-drift Gaussian null strategies for multiple-testing rejection test.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    cost_model = CostModel(commission_bps=3.0, slippage_bps=3.0, borrow_annual_bps=50.0, annualization_factor=252)
    registry = ImmutableExperimentRegistry(out / "registry.jsonl")
    registry.initialize()

    for seq, cfg in enumerate(grid.all_configurations(), start=1):
        tid = f"null_f{cfg['fast']}_s{cfg['slow']}"
        gross = rng.normal(loc=0.0, scale=0.009, size=n_obs)
        turnover = np.full(n_obs, 0.12)
        short_exp = np.full(n_obs, 0.30)

        spec = ReturnSeriesSpec(
            series_id=tid,
            version="synth-2026.02",
            frequency=ReturnFrequency.DAILY,
            timestamps=ts,
            gross_returns=tuple(float(x) for x in gross),
            turnover=tuple(float(x) for x in turnover),
            short_exposure=tuple(float(x) for x in short_exp),
            benchmark_returns=tuple(float(x) for x in bench),
            provenance="synthetic_fixture",
        )
        save_return_series_csv(spec, out / "series" / f"{tid}.csv")
        rec = TrialRecord(
            trial_id=tid,
            sequence_number=seq,
            experiment_id="exp-null-noise-v1",
            strategy_name="synthetic_null_crossover",
            parameters=cfg,
            status=TrialStatus.COMPLETED,
            created_at_utc=f"2026-01-16T10:{seq:02d}:00Z",
            dataset_version="synth-2026.02",
            code_version="0.1.0",
            cost_model=cost_model,
            series_id=tid,
            return_series_sha256=spec.content_sha256,
            tags={"provenance": "synthetic_fixture"},
        )
        registry.append(rec)


def build_cherry_picked_experiment() -> None:
    """Experiment with omitted grid configurations, sequence gaps, and exact duplicate trials."""
    out = FIXTURES_DIR / "valid" / "cherry_picked_experiment"
    _reset_dir(out)
    rng = np.random.default_rng(202603)
    n_obs = 360
    ts = _business_dates(n_obs)
    bench = rng.normal(loc=0.0002, scale=0.0075, size=n_obs)

    grid = ParameterGridSpec(
        grid_id="grid-cherry-v1",
        version="1.0.0",
        parameters={"lookback": (15, 20, 25), "vol_target": (0.08, 0.10, 0.12)},
    )
    (out / "parameter_grid.json").write_text(json.dumps(grid.to_dict(), indent=2) + "\n", encoding="utf-8")
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "experiment_id": "exp-cherry-picked-v1",
                "dataset_version": "synth-2026.03",
                "frequency": "daily",
                "provenance": "synthetic_fixture",
                "description": "Selective reporting scenario: missing sequence numbers, low grid coverage, and duplicate returns.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    cost_model = CostModel(commission_bps=2.0, slippage_bps=2.0, borrow_annual_bps=40.0, annualization_factor=252)
    registry = ImmutableExperimentRegistry(out / "registry.jsonl")
    registry.initialize()

    # Only log 5 of 9 planned configs with sequence gaps (1, 2, 4, 7, 9) and duplicate series on trial 4 & 7
    chosen = [
        (1, {"lookback": 15, "vol_target": 0.08}),
        (2, {"lookback": 20, "vol_target": 0.10}),
        (4, {"lookback": 20, "vol_target": 0.12}),
        (7, {"lookback": 25, "vol_target": 0.10}),
        (9, {"lookback": 25, "vol_target": 0.12}),
    ]
    saved_dup_gross = None
    for seq, cfg in chosen:
        tid = f"cherry_seq{seq}"
        if seq == 4:
            gross = rng.normal(loc=0.001, scale=0.007, size=n_obs)
            saved_dup_gross = gross.copy()
        elif seq == 7 and saved_dup_gross is not None:
            gross = saved_dup_gross.copy()  # Exact duplicate of seq 4
        else:
            gross = rng.normal(loc=0.0008, scale=0.007, size=n_obs)

        spec = ReturnSeriesSpec(
            series_id=tid,
            version="synth-2026.03",
            frequency=ReturnFrequency.DAILY,
            timestamps=ts,
            gross_returns=tuple(float(x) for x in gross),
            turnover=tuple(0.05 for _ in range(n_obs)),
            short_exposure=tuple(0.20 for _ in range(n_obs)),
            benchmark_returns=tuple(float(x) for x in bench),
            provenance="synthetic_fixture",
        )
        save_return_series_csv(spec, out / "series" / f"{tid}.csv")
        registry.append(
            TrialRecord(
                trial_id=tid,
                sequence_number=seq,
                experiment_id="exp-cherry-picked-v1",
                strategy_name="synthetic_cherry_picked",
                parameters=cfg,
                status=TrialStatus.COMPLETED,
                created_at_utc=f"2026-01-17T10:{seq:02d}:00Z",
                dataset_version="synth-2026.03",
                code_version="0.1.0",
                cost_model=cost_model,
                series_id=tid,
                return_series_sha256=spec.content_sha256,
                tags={"provenance": "synthetic_fixture"},
            )
        )


def build_boundary_fixtures() -> None:
    """Create insufficient-sample and extreme-outlier boundary scenarios."""
    # 1. Insufficient sample (24 observations)
    out_small = FIXTURES_DIR / "boundary" / "insufficient_sample_experiment"
    _reset_dir(out_small)
    rng = np.random.default_rng(202604)
    n_obs = 24
    ts = _business_dates(n_obs)
    cost_model = CostModel()
    registry = ImmutableExperimentRegistry(out_small / "registry.jsonl")
    registry.initialize()
    (out_small / "manifest.json").write_text(
        json.dumps({"experiment_id": "exp-insufficient-sample", "dataset_version": "synth-2026.04", "frequency": "daily", "provenance": "synthetic_fixture"}, indent=2) + "\n",
        encoding="utf-8",
    )
    for seq in range(1, 5):
        tid = f"short_t{seq}"
        gross = rng.normal(loc=0.001, scale=0.008, size=n_obs)
        spec = ReturnSeriesSpec(
            series_id=tid,
            version="synth-2026.04",
            frequency=ReturnFrequency.DAILY,
            timestamps=ts,
            gross_returns=tuple(float(x) for x in gross),
            provenance="synthetic_fixture",
        )
        save_return_series_csv(spec, out_small / "series" / f"{tid}.csv")
        registry.append(
            TrialRecord(
                trial_id=tid,
                sequence_number=seq,
                experiment_id="exp-insufficient-sample",
                strategy_name="synthetic_short",
                parameters={"k": seq},
                status=TrialStatus.COMPLETED,
                created_at_utc=f"2026-01-18T10:{seq:02d}:00Z",
                dataset_version="synth-2026.04",
                code_version="0.1.0",
                cost_model=cost_model,
                series_id=tid,
                return_series_sha256=spec.content_sha256,
            )
        )

    # 2. Extreme outlier driven strategy
    out_outlier = FIXTURES_DIR / "boundary" / "extreme_outlier_experiment"
    _reset_dir(out_outlier)
    rng2 = np.random.default_rng(202605)
    n_obs_out = 360
    ts_out = _business_dates(n_obs_out)
    registry2 = ImmutableExperimentRegistry(out_outlier / "registry.jsonl")
    registry2.initialize()
    (out_outlier / "manifest.json").write_text(
        json.dumps({"experiment_id": "exp-extreme-outliers", "dataset_version": "synth-2026.05", "frequency": "daily", "provenance": "synthetic_fixture"}, indent=2) + "\n",
        encoding="utf-8",
    )
    for seq in range(1, 7):
        tid = f"outlier_t{seq}"
        gross = rng2.normal(loc=0.0, scale=0.005, size=n_obs_out)
        gross = gross - np.mean(gross) - 0.0001
        if seq == 3:
            # Two extreme one-day jackpot spikes driving the entire apparent Sharpe
            gross[100] = 0.22
            gross[240] = 0.25
        spec = ReturnSeriesSpec(
            series_id=tid,
            version="synth-2026.05",
            frequency=ReturnFrequency.DAILY,
            timestamps=ts_out,
            gross_returns=tuple(float(x) for x in gross),
            turnover=tuple(0.02 for _ in range(n_obs_out)),
            provenance="synthetic_fixture",
        )
        save_return_series_csv(spec, out_outlier / "series" / f"{tid}.csv")
        registry2.append(
            TrialRecord(
                trial_id=tid,
                sequence_number=seq,
                experiment_id="exp-extreme-outliers",
                strategy_name="synthetic_outlier_spike",
                parameters={"threshold": 1.0 + 0.1 * seq},
                status=TrialStatus.COMPLETED,
                created_at_utc=f"2026-01-19T10:{seq:02d}:00Z",
                dataset_version="synth-2026.05",
                code_version="0.1.0",
                cost_model=cost_model,
                series_id=tid,
                return_series_sha256=spec.content_sha256,
            )
        )


def build_invalid_fixtures() -> None:
    """Create contract-violating fixtures for negative validation tests."""
    inv = FIXTURES_DIR / "invalid"
    _reset_dir(inv)

    # Non-monotonic timestamps
    (inv / "non_monotonic_timestamps.csv").write_text(
        "timestamp,gross_return\n2024-01-03,0.01\n2024-01-02,0.005\n2024-01-04,-0.002\n",
        encoding="utf-8",
    )
    # NaN returns
    (inv / "nan_returns.csv").write_text(
        "timestamp,gross_return\n2024-01-02,0.01\n2024-01-03,NaN\n2024-01-04,-0.002\n",
        encoding="utf-8",
    )
    # Return < -1.0 (-120%)
    (inv / "below_minus_100pct.csv").write_text(
        "timestamp,gross_return\n2024-01-02,0.01\n2024-01-03,-1.20\n2024-01-04,0.005\n",
        encoding="utf-8",
    )
    # Tampered registry JSONL
    tampered_reg = ImmutableExperimentRegistry(inv / "tampered_registry.jsonl")
    tampered_reg.initialize()
    tampered_reg.append(
        TrialRecord(
            trial_id="t1",
            sequence_number=1,
            experiment_id="exp-tamper",
            strategy_name="s",
            parameters={"a": 1},
            status=TrialStatus.COMPLETED,
            created_at_utc="2026-01-01T00:00:00Z",
            dataset_version="v1",
            code_version="0.1.0",
            cost_model=CostModel(),
            series_id="t1",
        )
    )
    raw = (inv / "tampered_registry.jsonl").read_text(encoding="utf-8")
    (inv / "tampered_registry.jsonl").write_text(raw.replace('"a":1', '"a":999'), encoding="utf-8")


def build_hand_calculated_reference() -> None:
    """Deterministic hand-calculated reference oracle fixture."""
    ref_path = FIXTURES_DIR / "valid" / "hand_calculated_reference.json"
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    # 12 symmetric returns around +0.01 with known exact moments
    # Deviations from 0.01: [-0.02, -0.01, 0.0, 0.01, 0.02, -0.01, 0.01, -0.02, 0.02, 0.0, -0.01, 0.01]
    returns = [-0.01, 0.00, 0.01, 0.02, 0.03, 0.00, 0.02, -0.01, 0.03, 0.01, 0.00, 0.02]
    # Exact sum = 0.12 -> mean = 0.01
    # Squared deviations: [4,1,0,1,4,1,1,4,4,0,1,1] * 1e-4 = 22e-4
    # Sample variance (ddof=1): 22e-4 / 11 = 2e-4 -> std = sqrt(0.0002) = 0.01414213562373095
    # Per-period Sharpe = 0.01 / sqrt(0.0002) = 0.7071067811865475 = 1 / sqrt(2)
    # Annualized Sharpe (252) = sqrt(126) = 11.224972160321824
    # Cost model: 5 bps comm + 5 bps slip = 10 bps = 0.001 per unit turnover; turnover=0.5 -> 0.0005 drag per period
    # Net mean = 0.0095, same std = sqrt(0.0002) -> net per-period Sharpe = 0.0095 / sqrt(0.0002) = 0.6717514421272202
    ref = {
        "provenance": "synthetic_fixture",
        "description": "Hand-calculated analytical oracle for returns, transaction cost drag, and Sharpe ratio.",
        "annualization_factor": 252,
        "gross_returns": returns,
        "turnover": [0.5] * 12,
        "cost_model": {
            "commission_bps": 5.0,
            "slippage_bps": 5.0,
            "borrow_annual_bps": 0.0,
            "fixed_cost_per_period": 0.0,
            "annualization_factor": 252,
        },
        "expected": {
            "gross_mean": 0.01,
            "gross_sample_std": 0.01414213562373095,
            "gross_sharpe_period": 0.7071067811865475,
            "gross_sharpe_annual": 11.224972160321824,
            "period_cost_drag": 0.0005,
            "net_mean": 0.0095,
            "net_sharpe_period": 0.6717514421272202,
            "skewness": 0.0,
        },
        "tolerance": 1e-10,
    }
    ref_path.write_text(json.dumps(ref, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    build_known_signal_experiment()
    build_null_overfit_experiment()
    build_cherry_picked_experiment()
    build_boundary_fixtures()
    build_invalid_fixtures()
    build_hand_calculated_reference()
    print(f"Generated all deterministic fixtures under {FIXTURES_DIR}")


if __name__ == "__main__":
    main()
