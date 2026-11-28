"""File I/O adapters for loading and saving versioned inputs and policy configs.

Isolates all filesystem parsing (CSV/JSON) from pure domain modules.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .models import (
    ContractValidationError,
    CostModel,
    ParameterGridSpec,
    ReturnFrequency,
    ReturnSeriesSpec,
    UnsupportedConfigurationError,
)


def load_policy_config(path: Path) -> dict[str, Any]:
    """Load and validate JSON policy configuration."""
    p = Path(path)
    if not p.exists():
        raise ContractValidationError(f"Policy configuration file not found: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ContractValidationError(f"Invalid JSON in policy config {p}: {exc}") from exc
    required_top = ("policy_version", "seed", "dsr", "pbo", "bootstrap", "walk_forward", "perturbation", "integrity", "promotion")
    missing = [k for k in required_top if k not in data]
    if missing:
        raise ContractValidationError(f"Policy configuration missing required section(s): {missing}")
    return data


def load_parameter_grid(path: Path) -> ParameterGridSpec:
    p = Path(path)
    if not p.exists():
        raise ContractValidationError(f"Parameter grid file not found: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ContractValidationError(f"Invalid JSON in parameter grid {p}: {exc}") from exc
    return ParameterGridSpec.from_dict(data)


def load_cost_model(path: Path) -> CostModel:
    p = Path(path)
    if not p.exists():
        raise ContractValidationError(f"Cost model file not found: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ContractValidationError(f"Invalid JSON in cost model {p}: {exc}") from exc
    return CostModel.from_dict(data)


def load_return_series_csv(
    path: Path,
    series_id: str,
    version: str,
    frequency: str = "daily",
    provenance: str = "synthetic_fixture",
) -> ReturnSeriesSpec:
    """Load a ReturnSeriesSpec from CSV with required columns [timestamp, gross_return].

    Optional columns: turnover, short_exposure, benchmark_return.
    """
    p = Path(path)
    if not p.exists():
        raise ContractValidationError(f"Return series file not found: {p}")
    try:
        freq_enum = ReturnFrequency(frequency.lower())
    except ValueError as exc:
        raise UnsupportedConfigurationError(
            f"Unsupported return frequency '{frequency}'. Supported: {[f.value for f in ReturnFrequency]}"
        ) from exc

    try:
        df = pd.read_csv(p)
    except Exception as exc:
        raise ContractValidationError(f"Failed to read CSV {p}: {exc}") from exc

    required_cols = {"timestamp", "gross_return"}
    missing_cols = required_cols.difference(df.columns)
    if missing_cols:
        raise ContractValidationError(f"CSV {p} missing required columns: {sorted(missing_cols)}")

    timestamps = tuple(str(x) for x in df["timestamp"].tolist())
    gross = tuple(float(x) for x in df["gross_return"].tolist())
    turnover = tuple(float(x) for x in df["turnover"].tolist()) if "turnover" in df.columns else None
    short_exp = (
        tuple(float(x) for x in df["short_exposure"].tolist()) if "short_exposure" in df.columns else None
    )
    benchmark = (
        tuple(float(x) for x in df["benchmark_return"].tolist())
        if "benchmark_return" in df.columns
        else None
    )

    return ReturnSeriesSpec(
        series_id=series_id,
        version=version,
        frequency=freq_enum,
        timestamps=timestamps,
        gross_returns=gross,
        turnover=turnover,
        short_exposure=short_exp,
        benchmark_returns=benchmark,
        provenance=provenance,
    )


def save_return_series_csv(series: ReturnSeriesSpec, path: Path) -> None:
    """Persist a ReturnSeriesSpec to CSV."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        "timestamp": list(series.timestamps),
        "gross_return": list(series.gross_returns),
    }
    if series.turnover is not None:
        data["turnover"] = list(series.turnover)
    if series.short_exposure is not None:
        data["short_exposure"] = list(series.short_exposure)
    if series.benchmark_returns is not None:
        data["benchmark_return"] = list(series.benchmark_returns)
    df = pd.DataFrame(data)
    df.to_csv(p, index=False)


def align_return_series_matrix(
    series_by_id: dict[str, ReturnSeriesSpec],
    ordered_ids: list[str],
) -> tuple[tuple[str, ...], list[ReturnSeriesSpec]]:
    """Verify all series share identical aligned timestamps before matrix construction."""
    if not ordered_ids:
        raise ContractValidationError("ordered_ids cannot be empty")
    missing = [sid for sid in ordered_ids if sid not in series_by_id]
    if missing:
        raise ContractValidationError(f"Missing return series for trial(s): {missing}")

    ref = series_by_id[ordered_ids[0]]
    ref_ts = ref.timestamps
    aligned: list[ReturnSeriesSpec] = []
    for sid in ordered_ids:
        spec = series_by_id[sid]
        if spec.timestamps != ref_ts:
            raise ContractValidationError(
                f"Timestamp misalignment between '{ordered_ids[0]}' and '{sid}': "
                f"len {len(ref_ts)} vs {len(spec.timestamps)}"
            )
        if spec.frequency != ref.frequency:
            raise ContractValidationError(
                f"Frequency mismatch between '{ordered_ids[0]}' ({ref.frequency.value}) "
                f"and '{sid}' ({spec.frequency.value})"
            )
        aligned.append(spec)
    return ref_ts, aligned
