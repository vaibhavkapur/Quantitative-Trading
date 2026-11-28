"""I/O boundary layer: scenario loading, structured logging, replay manifests, and report writers."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import platform
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl
import scipy

from optimal_execution import __version__ as ENGINE_VERSION
from optimal_execution.contracts import (
    SCHEMA_VERSION,
    ExecutionScenario,
    MonteCarloSummary,
    RunMetadata,
    ScheduleRequest,
    SimulationResult,
    StrategyType,
    UnsupportedScenarioError,
    ValidationError,
)
from optimal_execution.schedules import generate_schedule
from optimal_execution.simulator import simulate_deterministic, simulate_monte_carlo


class _JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra_fields = getattr(record, "structured_fields", None)
        if isinstance(extra_fields, dict):
            payload.update(extra_fields)
        return json.dumps(payload, sort_keys=True)


def get_logger(name: str = "optimal_execution", level: int = logging.INFO) -> logging.Logger:
    """Return a structured JSON logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_JsonLogFormatter())
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def log_event(logger: logging.Logger, message: str, **fields: Any) -> None:
    """Emit a structured JSON log record."""
    logger.info(message, extra={"structured_fields": fields})


def load_json_file(path: Path) -> Dict[str, Any]:
    """Load and parse a JSON file with explicit error reporting."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration or fixture file not found: {file_path}")
    try:
        raw = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"Invalid JSON in {file_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValidationError(f"Expected JSON object in {file_path}, got {type(raw).__name__}")
    return raw


def load_scenario(path: Path) -> ExecutionScenario:
    """Load and validate an ExecutionScenario from disk."""
    raw = load_json_file(path)
    return ExecutionScenario.from_dict(raw)


def load_schedule_requests(raw_requests: Sequence[Dict[str, Any]]) -> List[ScheduleRequest]:
    """Parse and validate a list of ScheduleRequest dictionaries."""
    requests: List[ScheduleRequest] = []
    for idx, item in enumerate(raw_requests):
        if not isinstance(item, dict) or "strategy" not in item:
            raise ValidationError(f"schedule_requests[{idx}] must be an object containing 'strategy'")
        try:
            strategy = StrategyType(item["strategy"])
        except ValueError as exc:
            raise UnsupportedScenarioError(
                f"Unsupported strategy in schedule_requests[{idx}]: {item['strategy']!r}"
            ) from exc
        req = ScheduleRequest(
            strategy=strategy,
            risk_aversion_lambda=float(item.get("risk_aversion_lambda", 1e-6)),
            target_participation_rate=float(item.get("target_participation_rate", 0.10)),
            enforce_capacity_in_optimizer=bool(item.get("enforce_capacity_in_optimizer", False)),
        )
        req.validate()
        requests.append(req)
    return requests


def compute_config_sha256(payload: Dict[str, Any]) -> str:
    """Compute a stable SHA-256 hex digest of a canonical JSON payload."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def build_run_metadata(
    scenario: ExecutionScenario,
    extra_config: Optional[Dict[str, Any]] = None,
) -> RunMetadata:
    """Create reproducible runtime metadata and a content-addressed stable run_id."""
    hash_input: Dict[str, Any] = {"scenario": scenario.to_dict()}
    if extra_config:
        hash_input["extra_config"] = extra_config
    digest = compute_config_sha256(hash_input)
    run_id = f"run-{digest[:12]}"

    return RunMetadata(
        run_id=run_id,
        schema_version=SCHEMA_VERSION,
        engine_version=ENGINE_VERSION,
        timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        python_version=platform.python_version(),
        platform=platform.platform(),
        package_versions={
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "polars": pl.__version__,
        },
        scenario_id=scenario.scenario_id,
        seed=scenario.seed,
        config_sha256=digest,
    )


def _serialize_dataclass_obj(obj: Any) -> Any:
    """Recursively convert dataclasses and enums into JSON-safe primitives."""
    if hasattr(obj, "__dataclass_fields__"):
        return _serialize_dataclass_obj(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _serialize_dataclass_obj(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize_dataclass_obj(x) for x in obj]
    if hasattr(obj, "value") and isinstance(obj.value, str):
        return obj.value
    return obj


def simulation_result_to_dict(result: SimulationResult) -> Dict[str, Any]:
    return _serialize_dataclass_obj(result)


def mc_summary_to_dict(summary: MonteCarloSummary) -> Dict[str, Any]:
    return _serialize_dataclass_obj(summary)


def save_json_report(
    output_path: Path,
    metadata: RunMetadata,
    payload: Dict[str, Any],
) -> Path:
    """Write a versioned machine-readable JSON report with embedded runtime metadata."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "metadata": _serialize_dataclass_obj(metadata),
        "data": _serialize_dataclass_obj(payload),
    }
    out.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def save_polars_csv(output_path: Path, df: pl.DataFrame) -> Path:
    """Write a Polars DataFrame to CSV."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(out)
    return out


def save_replay_manifest(
    output_path: Path,
    scenario: ExecutionScenario,
    requests: Sequence[ScheduleRequest],
    metadata: RunMetadata,
) -> Path:
    """Persist a self-contained replay manifest to reproduce a run bit-for-bit."""
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": metadata.run_id,
        "config_sha256": metadata.config_sha256,
        "scenario": scenario.to_dict(),
        "schedule_requests": [
            {
                "strategy": req.strategy.value,
                "risk_aversion_lambda": req.risk_aversion_lambda,
                "target_participation_rate": req.target_participation_rate,
                "enforce_capacity_in_optimizer": req.enforce_capacity_in_optimizer,
            }
            for req in requests
        ],
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def replay_from_manifest(
    manifest_path: Path,
) -> Tuple[ExecutionScenario, List[SimulationResult], List[MonteCarloSummary], RunMetadata]:
    """Re-execute an exact run from a saved replay manifest."""
    raw = load_json_file(manifest_path)
    if "scenario" not in raw or "schedule_requests" not in raw:
        raise ValidationError(
            f"Replay manifest {manifest_path} must contain 'scenario' and 'schedule_requests'"
        )
    scenario = ExecutionScenario.from_dict(raw["scenario"])
    requests = load_schedule_requests(raw["schedule_requests"])
    metadata = build_run_metadata(
        scenario,
        extra_config={"schedule_requests": raw["schedule_requests"]},
    )

    det_results: List[SimulationResult] = []
    mc_summaries: List[MonteCarloSummary] = []
    for req in requests:
        sched = generate_schedule(scenario, req)
        det_results.append(simulate_deterministic(scenario, sched))
        mc_sum, _ = simulate_monte_carlo(scenario, sched)
        mc_summaries.append(mc_sum)

    return scenario, det_results, mc_summaries, metadata
