"""Canonical data contracts, validation rules, and typed result models."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
import itertools
import json
import math
from typing import Any, Optional

import numpy as np


class RobustnessLabError(Exception):
    """Base exception for the quantitative strategy robustness lab."""


class ContractValidationError(RobustnessLabError):
    """Raised when input data violates schema, unit, or domain invariants."""


class InsufficientSampleError(RobustnessLabError):
    """Raised when a sample or trial set is too small for valid inference."""


class UnsupportedConfigurationError(RobustnessLabError):
    """Raised when an unsupported method, frequency, or parameter is requested."""


class RegistryIntegrityError(RobustnessLabError):
    """Raised when the immutable experiment registry fails hash-chain or append checks."""


class TrialStatus(str, Enum):
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    REJECTED = "REJECTED"
    ABORTED = "ABORTED"


class PromotionDecision(str, Enum):
    PROMOTE = "PROMOTE"
    HOLD_FOR_REVIEW = "HOLD_FOR_REVIEW"
    REJECT = "REJECT"


class ReturnFrequency(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"

    @property
    def periods_per_year(self) -> int:
        mapping = {
            ReturnFrequency.DAILY: 252,
            ReturnFrequency.WEEKLY: 52,
            ReturnFrequency.MONTHLY: 12,
        }
        return mapping[self]


def canonical_json(obj: Any) -> str:
    """Serialize an object deterministically for hashing and comparison."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    """Stable SHA-256 over canonical 10-decimal formatted values, invariant to CSV round-trip."""
    rounded = np.round(np.asarray(arr, dtype=np.float64), decimals=10)
    # Normalize -0.0 to +0.0
    rounded = np.where(rounded == 0.0, 0.0, rounded)
    canonical = ",".join(f"{float(x):.10f}" for x in rounded.ravel())
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CostModel:
    """Transaction and financing cost specification in basis points (bps).

    Units:
      - commission_bps: one-way commission per unit of turnover (1 bps = 1e-4).
      - slippage_bps: one-way market impact / slippage per unit of turnover.
      - borrow_annual_bps: annualized short-borrow fee applied to short_exposure.
      - fixed_cost_per_period: fixed per-period decimal return drag.
      - annualization_factor: periods per year used to prorate annual borrow cost.
    """

    commission_bps: float = 2.0
    slippage_bps: float = 3.0
    borrow_annual_bps: float = 50.0
    fixed_cost_per_period: float = 0.0
    annualization_factor: int = 252

    def __post_init__(self) -> None:
        for name in (
            "commission_bps",
            "slippage_bps",
            "borrow_annual_bps",
            "fixed_cost_per_period",
        ):
            val = getattr(self, name)
            if not isinstance(val, (int, float)) or not math.isfinite(val) or val < 0:
                raise ContractValidationError(f"CostModel.{name} must be finite and >= 0, got {val!r}")
        if not isinstance(self.annualization_factor, int) or self.annualization_factor <= 0:
            raise ContractValidationError(
                f"CostModel.annualization_factor must be a positive int, got {self.annualization_factor!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CostModel":
        return cls(
            commission_bps=float(data.get("commission_bps", 2.0)),
            slippage_bps=float(data.get("slippage_bps", 3.0)),
            borrow_annual_bps=float(data.get("borrow_annual_bps", 50.0)),
            fixed_cost_per_period=float(data.get("fixed_cost_per_period", 0.0)),
            annualization_factor=int(data.get("annualization_factor", 252)),
        )


@dataclass(frozen=True)
class ReturnSeriesSpec:
    """Versioned simple return series for a single strategy trial or benchmark.

    Units:
      - gross_returns: simple per-period decimal returns (e.g. 0.01 = +1%).
      - turnover: per-period one-way portfolio turnover fraction (>= 0).
      - short_exposure: per-period gross short leverage fraction (>= 0).
      - benchmark_returns: optional aligned simple per-period benchmark returns.
      - timestamps: strictly increasing ISO-8601 date/datetime strings (UTC or date-only).
    """

    series_id: str
    version: str
    frequency: ReturnFrequency
    timestamps: tuple[str, ...]
    gross_returns: tuple[float, ...]
    turnover: Optional[tuple[float, ...]] = None
    short_exposure: Optional[tuple[float, ...]] = None
    benchmark_returns: Optional[tuple[float, ...]] = None
    provenance: str = "synthetic_fixture"

    def __post_init__(self) -> None:
        if not self.series_id or not isinstance(self.series_id, str):
            raise ContractValidationError("series_id must be a non-empty string")
        if not self.version or not isinstance(self.version, str):
            raise ContractValidationError("version must be a non-empty string")
        if not isinstance(self.frequency, ReturnFrequency):
            raise ContractValidationError(f"Unsupported frequency: {self.frequency!r}")
        n = len(self.gross_returns)
        if n == 0:
            raise ContractValidationError(f"Return series '{self.series_id}' is empty")
        if len(self.timestamps) != n:
            raise ContractValidationError(
                f"Length mismatch in '{self.series_id}': timestamps={len(self.timestamps)} vs returns={n}"
            )
        # Check strictly increasing unique timestamps
        for idx in range(1, n):
            if self.timestamps[idx] <= self.timestamps[idx - 1]:
                raise ContractValidationError(
                    f"Timestamps in '{self.series_id}' must be strictly increasing "
                    f"(index {idx}: {self.timestamps[idx - 1]!r} -> {self.timestamps[idx]!r})"
                )
        arr = np.asarray(self.gross_returns, dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            raise ContractValidationError(f"Non-finite values (NaN/Inf) found in gross_returns of '{self.series_id}'")
        if np.any(arr < -1.0):
            raise ContractValidationError(
                f"Simple return < -1.0 (-100%) found in '{self.series_id}': min={float(np.min(arr)):.6f}"
            )

        for opt_name, non_neg in (
            ("turnover", True),
            ("short_exposure", True),
            ("benchmark_returns", False),
        ):
            opt_val = getattr(self, opt_name)
            if opt_val is not None:
                if len(opt_val) != n:
                    raise ContractValidationError(
                        f"Length mismatch in '{self.series_id}': {opt_name}={len(opt_val)} vs returns={n}"
                    )
                opt_arr = np.asarray(opt_val, dtype=np.float64)
                if not np.all(np.isfinite(opt_arr)):
                    raise ContractValidationError(f"Non-finite values in {opt_name} of '{self.series_id}'")
                if non_neg and np.any(opt_arr < 0.0):
                    raise ContractValidationError(f"Negative values in {opt_name} of '{self.series_id}'")
                if opt_name == "benchmark_returns" and np.any(opt_arr < -1.0):
                    raise ContractValidationError(f"Benchmark return < -1.0 in '{self.series_id}'")

    def gross_array(self) -> np.ndarray:
        return np.asarray(self.gross_returns, dtype=np.float64)

    def turnover_array(self) -> np.ndarray:
        if self.turnover is None:
            return np.zeros(len(self.gross_returns), dtype=np.float64)
        return np.asarray(self.turnover, dtype=np.float64)

    def short_exposure_array(self) -> np.ndarray:
        if self.short_exposure is None:
            return np.zeros(len(self.gross_returns), dtype=np.float64)
        return np.asarray(self.short_exposure, dtype=np.float64)

    def benchmark_array(self) -> Optional[np.ndarray]:
        if self.benchmark_returns is None:
            return None
        return np.asarray(self.benchmark_returns, dtype=np.float64)

    @property
    def content_sha256(self) -> str:
        return sha256_array(self.gross_array())


@dataclass(frozen=True)
class ParameterGridSpec:
    """Declarative parameter grid defining the complete planned search space."""

    grid_id: str
    version: str
    parameters: dict[str, tuple[Any, ...]]

    def __post_init__(self) -> None:
        if not self.grid_id or not isinstance(self.grid_id, str):
            raise ContractValidationError("grid_id must be a non-empty string")
        if not self.version or not isinstance(self.version, str):
            raise ContractValidationError("version must be a non-empty string")
        if not self.parameters:
            raise ContractValidationError("ParameterGridSpec.parameters cannot be empty")
        for key, vals in self.parameters.items():
            if not key or not isinstance(key, str):
                raise ContractValidationError(f"Invalid parameter name: {key!r}")
            if not vals:
                raise ContractValidationError(f"Parameter '{key}' has empty candidate list")

    @property
    def expected_trial_count(self) -> int:
        count = 1
        for vals in self.parameters.values():
            count *= len(vals)
        return count

    def all_configurations(self) -> list[dict[str, Any]]:
        keys = sorted(self.parameters.keys())
        combos = itertools.product(*(self.parameters[k] for k in keys))
        return [dict(zip(keys, vals)) for vals in combos]

    def config_keys(self) -> set[str]:
        return {canonical_json(cfg) for cfg in self.all_configurations()}

    @property
    def grid_hash(self) -> str:
        serializable = {k: list(v) for k, v in sorted(self.parameters.items())}
        return sha256_text(canonical_json({"grid_id": self.grid_id, "version": self.version, "parameters": serializable}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "grid_id": self.grid_id,
            "version": self.version,
            "parameters": {k: list(v) for k, v in sorted(self.parameters.items())},
            "expected_trial_count": self.expected_trial_count,
            "grid_hash": self.grid_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ParameterGridSpec":
        raw_params = data.get("parameters", {})
        if not isinstance(raw_params, dict):
            raise ContractValidationError("ParameterGridSpec.parameters must be a dict")
        return cls(
            grid_id=str(data["grid_id"]),
            version=str(data["version"]),
            parameters={str(k): tuple(v) for k, v in raw_params.items()},
        )


@dataclass(frozen=True)
class TrialRecord:
    """Immutable metadata for a single strategy research trial (including failures)."""

    trial_id: str
    sequence_number: int
    experiment_id: str
    strategy_name: str
    parameters: dict[str, Any]
    status: TrialStatus
    created_at_utc: str
    dataset_version: str
    code_version: str
    cost_model: CostModel
    series_id: Optional[str] = None
    return_series_sha256: Optional[str] = None
    failure_reason: Optional[str] = None
    metrics_summary: dict[str, float] = field(default_factory=dict)
    split_boundaries: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    tags: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.trial_id or not isinstance(self.trial_id, str):
            raise ContractValidationError("trial_id must be a non-empty string")
        if not isinstance(self.sequence_number, int) or self.sequence_number < 1:
            raise ContractValidationError(f"sequence_number must be an integer >= 1, got {self.sequence_number!r}")
        if not self.experiment_id:
            raise ContractValidationError("experiment_id must be non-empty")
        if not isinstance(self.status, TrialStatus):
            raise ContractValidationError(f"Invalid TrialStatus: {self.status!r}")
        if self.status == TrialStatus.COMPLETED and not self.series_id:
            raise ContractValidationError(f"COMPLETED trial '{self.trial_id}' must reference a series_id")
        if self.status in (TrialStatus.FAILED, TrialStatus.ABORTED, TrialStatus.REJECTED) and not self.failure_reason:
            raise ContractValidationError(
                f"Non-completed trial '{self.trial_id}' with status {self.status.value} must record failure_reason"
            )
        for k, v in self.metrics_summary.items():
            if not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ContractValidationError(f"Trial '{self.trial_id}' metric '{k}' must be finite, got {v!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "sequence_number": self.sequence_number,
            "experiment_id": self.experiment_id,
            "strategy_name": self.strategy_name,
            "parameters": dict(sorted(self.parameters.items())),
            "status": self.status.value,
            "created_at_utc": self.created_at_utc,
            "dataset_version": self.dataset_version,
            "code_version": self.code_version,
            "cost_model": self.cost_model.to_dict(),
            "series_id": self.series_id,
            "return_series_sha256": self.return_series_sha256,
            "failure_reason": self.failure_reason,
            "metrics_summary": dict(sorted(self.metrics_summary.items())),
            "split_boundaries": list(self.split_boundaries),
            "tags": dict(sorted(self.tags.items())),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrialRecord":
        return cls(
            trial_id=str(data["trial_id"]),
            sequence_number=int(data["sequence_number"]),
            experiment_id=str(data["experiment_id"]),
            strategy_name=str(data["strategy_name"]),
            parameters=dict(data.get("parameters", {})),
            status=TrialStatus(data["status"]),
            created_at_utc=str(data["created_at_utc"]),
            dataset_version=str(data["dataset_version"]),
            code_version=str(data["code_version"]),
            cost_model=CostModel.from_dict(data.get("cost_model", {})),
            series_id=data.get("series_id"),
            return_series_sha256=data.get("return_series_sha256"),
            failure_reason=data.get("failure_reason"),
            metrics_summary={str(k): float(v) for k, v in data.get("metrics_summary", {}).items()},
            split_boundaries=tuple(data.get("split_boundaries", ())),
            tags={str(k): str(v) for k, v in data.get("tags", {}).items()},
        )


@dataclass(frozen=True)
class RegistryEntry:
    """Hash-chained immutable envelope stored in the experiment registry."""

    prev_hash: str
    record: TrialRecord
    entry_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "prev_hash": self.prev_hash,
            "record": self.record.to_dict(),
            "entry_hash": self.entry_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RegistryEntry":
        return cls(
            prev_hash=str(data["prev_hash"]),
            record=TrialRecord.from_dict(data["record"]),
            entry_hash=str(data["entry_hash"]),
        )


@dataclass(frozen=True)
class OutlierDiagnostic:
    max_abs_return: float
    max_abs_zscore: float
    n_extreme_observations: int
    excess_kurtosis: float
    skewness: float
    winsorized_sharpe_annual: float
    raw_sharpe_annual: float
    outlier_impact_pct: float
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class DSRResult:
    candidate_trial_id: str
    n_observations: int
    annualization_factor: int
    observed_sharpe_period: float
    observed_sharpe_annual: float
    benchmark_sharpe_annual: float
    skewness: float
    excess_kurtosis: float
    autocorr_variance_inflation: float
    psr_vs_benchmark: float
    raw_trial_count: int
    effective_trial_count: float
    trial_sharpe_variance_period: float
    expected_max_sharpe_period: float
    expected_max_sharpe_annual: float
    deflated_sharpe_ratio: float
    assumptions: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class PBOResult:
    n_partitions: int
    n_combinations_evaluated: int
    total_possible_combinations: int
    exact_enumeration: bool
    n_trials: int
    n_observations_used: int
    pbo: float
    prob_oos_loss: float
    median_logit: float
    mean_oos_rank: float
    is_to_oos_regression_slope: float
    is_to_oos_regression_intercept: float
    mean_is_sharpe_selected: float
    mean_oos_sharpe_selected: float
    assumptions: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class BootstrapResult:
    method: str
    n_resamples: int
    block_length: int
    seed: int
    confidence_level: float
    observed_sharpe_annual: float
    sharpe_ci_lower: float
    sharpe_ci_upper: float
    sharpe_std_error: float
    mean_return_ci_lower: float
    mean_return_ci_upper: float
    max_drawdown_observed: float
    max_drawdown_ci_upper: float
    p_value_vs_zero: float
    assumptions: tuple[str, ...]


@dataclass(frozen=True)
class WalkForwardFold:
    fold_index: int
    train_start: str
    train_end: str
    purge_end: str
    test_start: str
    test_end: str
    n_train: int
    n_test: int
    selected_trial_id: str
    train_net_sharpe: float
    test_net_sharpe: float
    test_benchmark_sharpe: Optional[float]
    test_mean_return: float


@dataclass(frozen=True)
class WalkForwardResult:
    mode: str
    train_window: int
    test_window: int
    step_size: int
    purge_gap: int
    n_folds: int
    total_oos_observations: int
    mean_is_net_sharpe: float
    aggregate_oos_net_sharpe: float
    aggregate_oos_gross_sharpe: float
    aggregate_oos_benchmark_sharpe: Optional[float]
    is_to_oos_degradation_ratio: float
    oos_hit_rate: float
    folds: tuple[WalkForwardFold, ...]
    assumptions: tuple[str, ...]


@dataclass(frozen=True)
class PerturbationNeighbor:
    trial_id: str
    parameters: dict[str, Any]
    normalized_distance: float
    net_sharpe_annual: float
    relative_sharpe_retention: float


@dataclass(frozen=True)
class PerturbationResult:
    center_trial_id: str
    center_parameters: dict[str, Any]
    center_net_sharpe: float
    relative_step: float
    n_neighbors_evaluated: int
    positive_neighbor_fraction: float
    median_neighbor_sharpe: float
    min_neighbor_sharpe: float
    sharpe_cliff_ratio: float
    stability_score: float
    neighbors: tuple[PerturbationNeighbor, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class IntegrityAuditResult:
    registry_valid: bool
    registry_errors: tuple[str, ...]
    total_logged_trials: int
    completed_trials: int
    failed_or_rejected_trials: int
    exact_duplicate_pairs: tuple[tuple[str, str], ...]
    near_duplicate_pairs: tuple[tuple[str, str, float], ...]
    duplicate_fraction: float
    sequence_gaps: tuple[int, ...]
    expected_grid_trials: Optional[int]
    covered_grid_trials: Optional[int]
    grid_coverage_fraction: Optional[float]
    cherry_picking_flags: tuple[str, ...]
    passed: bool


@dataclass(frozen=True)
class PromotionGateCheck:
    gate_name: str
    category: str
    passed: bool
    required: bool
    observed_value: Any
    threshold: Any
    rationale: str


@dataclass(frozen=True)
class PromotionVerdict:
    decision: PromotionDecision
    candidate_trial_id: str
    evidence_complete: bool
    missing_evidence: tuple[str, ...]
    gates: tuple[PromotionGateCheck, ...]
    summary: str
    review_notes: tuple[str, ...]
