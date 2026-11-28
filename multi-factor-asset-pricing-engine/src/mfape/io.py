"""I/O utilities for loading inputs, hashing artifacts, and writing reproducible reports."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import scipy
import statsmodels

from mfape.errors import ConfigurationError, UnsupportedInputError
from mfape.schemas import EngineConfig, RunOutput, SCHEMA_VERSION


def load_config(config_path: str | Path) -> EngineConfig:
    """Load and validate an EngineConfig from a JSON file."""
    path = Path(config_path)
    if not path.exists():
        raise ConfigurationError(
            f"Configuration file not found: {path}",
            {"path": str(path)},
        )
    if path.suffix.lower() != ".json":
        raise UnsupportedInputError(
            f"Unsupported configuration file format '{path.suffix}'. Expected '.json'.",
            {"path": str(path), "suffix": path.suffix},
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigurationError(
            f"Malformed JSON in configuration file {path}: {exc}",
            {"path": str(path)},
        ) from exc
    return EngineConfig.from_dict(raw)


def resolve_path(raw_path: str, base_dir: Path | None = None) -> Path:
    """Resolve a possibly relative path against an optional base directory."""
    p = Path(raw_path)
    if p.is_absolute() or base_dir is None:
        return p
    return (base_dir / p).resolve()


def load_table(path: str | Path) -> pl.DataFrame:
    """Load a CSV or Parquet table into a Polars DataFrame with explicit error handling."""
    file_path = Path(path)
    if not file_path.exists():
        raise ConfigurationError(
            f"Input data file not found: {file_path}",
            {"path": str(file_path)},
        )
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pl.read_csv(file_path, try_parse_dates=False)
    if suffix == ".parquet":
        return pl.read_parquet(file_path)
    raise UnsupportedInputError(
        f"Unsupported tabular input format '{suffix}' for {file_path}. "
        f"Supported formats: '.csv', '.parquet'.",
        {"path": str(file_path), "suffix": suffix},
    )


def compute_file_sha256(path: str | Path) -> str:
    """Compute a stable SHA-256 hex digest for an input file."""
    file_path = Path(path)
    hasher = hashlib.sha256()
    with file_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_run_id(
    config_dict: dict[str, Any],
    input_fingerprints: dict[str, str],
    engine_version: str,
) -> str:
    """Generate a deterministic, content-addressed run identifier."""
    # Exclude output_dir so moving the destination path does not alter logical run identity
    canonical_cfg = {k: v for k, v in config_dict.items() if k != "output_dir"}
    payload = {
        "schema_version": SCHEMA_VERSION,
        "engine_version": engine_version,
        "config": canonical_cfg,
        "input_fingerprints": input_fingerprints,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(blob).hexdigest()[:16]
    return f"mfape-{digest}"


def collect_runtime_environment() -> dict[str, Any]:
    """Capture runtime and dependency versions for reproducibility logs."""
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {
            "numpy": np.__version__,
            "polars": pl.__version__,
            "statsmodels": statsmodels.__version__,
            "scipy": scipy.__version__,
        },
    }


def write_run_artifacts(
    output: RunOutput,
    output_dir: str | Path,
    include_series: bool = True,
) -> dict[str, str]:
    """Write versioned JSON results, replay manifest, comparison CSV, and Markdown summary."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    result_path = out_path / "run_result.json"
    manifest_path = out_path / "replay_manifest.json"
    comparison_csv_path = out_path / "model_comparison.csv"
    summary_md_path = out_path / "summary.md"

    result_payload = output.to_dict(include_series=include_series)
    result_path.write_text(
        json.dumps(result_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest_payload = {
        "schema_version": output.schema_version,
        "engine_version": output.engine_version,
        "run_id": output.run_id,
        "run_label": output.run_label,
        "config": output.config,
        "input_fingerprints": output.input_fingerprints,
        "runtime_environment": output.runtime_environment,
    }
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    rows = output.comparison.rows
    if rows:
        fieldnames = list(rows[0].keys())
        with comparison_csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    summary_md_path.write_text(_render_summary_markdown(output), encoding="utf-8")

    return {
        "result_json": str(result_path),
        "replay_manifest": str(manifest_path),
        "comparison_csv": str(comparison_csv_path),
        "summary_md": str(summary_md_path),
    }


def _fmt(val: float | None, digits: int = 4) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:.{digits}f}"


def _render_summary_markdown(output: RunOutput) -> str:
    lines: list[str] = [
        f"# Asset Pricing Run Summary — `{output.run_label}`",
        "",
        f"- **Run ID:** `{output.run_id}`",
        f"- **Schema version:** `{output.schema_version}`",
        f"- **Engine version:** `{output.engine_version}`",
        f"- **Baseline model:** `{output.comparison.baseline_model_name}`",
        f"- **Split boundaries:** `{json.dumps(output.split_boundaries, sort_keys=True)}`",
        "",
        "## Model Comparison vs Baseline",
        "",
        "| Model | Factors | TS Mean R² | TS Mean |α| | GRS p-val | FM CS R² | FM Pricing RMSE | OOS CS R² | OOS CS RMSE | OOS TS R² (α=0) | Δ OOS CS RMSE vs Baseline |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in output.comparison.rows:
        lines.append(
            f"| {row['model_name']} | {', '.join(row['factors'])} | "
            f"{_fmt(row['ts_mean_r2'])} | {_fmt(row['ts_mean_abs_alpha'], 5)} | "
            f"{_fmt(row['ts_grs_pvalue'])} | {_fmt(row['fm_cs_r2'])} | "
            f"{_fmt(row['fm_pricing_rmse'], 5)} | {_fmt(row['oos_cs_r2'])} | "
            f"{_fmt(row['oos_cs_rmse'], 5)} | {_fmt(row['oos_ts_r2_zero_alpha'])} | "
            f"{_fmt(row['delta_oos_cs_rmse_vs_baseline'], 5)} |"
        )

    lines.extend(["", "## Fama-MacBeth Cross-Sectional Factor Premia (Training Window)", ""])
    for m in output.models:
        fm = m.fama_macbeth_train
        lines.append(f"### {m.spec.name} ({fm.date_start} to {fm.date_end}, T={fm.n_periods}, N={fm.n_assets})")
        lines.append("")
        lines.append("| Term | Est (period) | Ann. Est | FM SE | FM t-stat | Shanken SE | Shanken t-stat | HAC SE | HAC t-stat |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for p in fm.premia:
            lines.append(
                f"| {p.name} | {_fmt(p.estimate, 5)} | {_fmt(p.annualized_estimate, 4)} | "
                f"{_fmt(p.fm_se, 5)} | {_fmt(p.fm_tstat, 2)} | "
                f"{_fmt(p.shanken_se, 5)} | {_fmt(p.shanken_tstat, 2)} | "
                f"{_fmt(p.hac_se, 5)} | {_fmt(p.hac_tstat, 2)} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"
