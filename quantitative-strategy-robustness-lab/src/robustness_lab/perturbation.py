"""Parameter perturbation and neighborhood stability diagnostics.

Pure domain module (no file I/O). Checks whether the candidate strategy sits
on a broad performance plateau or an isolated overfit spike in parameter space.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .models import (
    ContractValidationError,
    ParameterGridSpec,
    PerturbationNeighbor,
    PerturbationResult,
    TrialRecord,
)


def _coordinate_distance(
    val: Any,
    center_val: Any,
    grid_values: Optional[tuple[Any, ...]] = None,
) -> tuple[float, bool]:
    """Compute normalized relative distance and whether val is grid-adjacent to center_val."""
    if val == center_val:
        return 0.0, True

    grid_adjacent = False
    if grid_values is not None and val in grid_values and center_val in grid_values:
        idx_v = grid_values.index(val)
        idx_c = grid_values.index(center_val)
        if abs(idx_v - idx_c) <= 1:
            grid_adjacent = True

    if isinstance(val, (int, float)) and isinstance(center_val, (int, float)) and not isinstance(val, bool):
        denom = abs(float(center_val)) if abs(float(center_val)) > 1e-9 else 1.0
        rel_dist = abs(float(val) - float(center_val)) / denom
        return rel_dist, grid_adjacent

    # Categorical mismatch
    return 1.0, grid_adjacent


def evaluate_parameter_perturbation(
    center_trial: TrialRecord,
    completed_trials: list[TrialRecord],
    trial_net_sharpes: dict[str, float],
    relative_step: float = 0.20,
    parameter_grid: Optional[ParameterGridSpec] = None,
) -> PerturbationResult:
    """Evaluate local parameter perturbation stability around center_trial."""
    if relative_step <= 0.0:
        raise ContractValidationError(f"relative_step must be > 0, got {relative_step}")
    if center_trial.trial_id not in trial_net_sharpes:
        raise ContractValidationError(f"Missing net Sharpe for center trial '{center_trial.trial_id}'")

    center_params = center_trial.parameters
    center_sr = float(trial_net_sharpes[center_trial.trial_id])
    param_keys = sorted(center_params.keys())

    neighbors: list[PerturbationNeighbor] = []
    for trial in completed_trials:
        if trial.trial_id == center_trial.trial_id:
            continue
        if trial.trial_id not in trial_net_sharpes:
            continue
        if sorted(trial.parameters.keys()) != param_keys:
            continue

        coord_dists: list[float] = []
        is_neighbor = True
        any_diff = False

        for k in param_keys:
            grid_vals = parameter_grid.parameters.get(k) if parameter_grid is not None else None
            dist, grid_adj = _coordinate_distance(trial.parameters[k], center_params[k], grid_vals)
            if dist > 0.0:
                any_diff = True
            if not (dist <= relative_step or grid_adj):
                is_neighbor = False
                break
            coord_dists.append(dist)

        if is_neighbor and any_diff:
            norm_dist = float(np.sqrt(np.mean(np.square(coord_dists)))) if coord_dists else 0.0
            nbr_sr = float(trial_net_sharpes[trial.trial_id])
            retention = nbr_sr / center_sr if abs(center_sr) > 1e-6 else 0.0
            neighbors.append(
                PerturbationNeighbor(
                    trial_id=trial.trial_id,
                    parameters=dict(trial.parameters),
                    normalized_distance=norm_dist,
                    net_sharpe_annual=nbr_sr,
                    relative_sharpe_retention=retention,
                )
            )

    neighbors.sort(key=lambda n: (n.normalized_distance, n.trial_id))

    warnings: list[str] = []
    if not neighbors:
        warnings.append(
            f"No neighboring parameter configurations found within relative_step={relative_step:.2f} "
            "or adjacent grid steps; local robustness cannot be confirmed."
        )
        return PerturbationResult(
            center_trial_id=center_trial.trial_id,
            center_parameters=dict(center_params),
            center_net_sharpe=center_sr,
            relative_step=relative_step,
            n_neighbors_evaluated=0,
            positive_neighbor_fraction=0.0,
            median_neighbor_sharpe=0.0,
            min_neighbor_sharpe=0.0,
            sharpe_cliff_ratio=1.0,
            stability_score=0.0,
            neighbors=(),
            warnings=tuple(warnings),
        )

    nbr_srs = np.asarray([n.net_sharpe_annual for n in neighbors], dtype=np.float64)
    pos_frac = float(np.mean(nbr_srs > 0.0))
    med_sr = float(np.median(nbr_srs))
    min_sr = float(np.min(nbr_srs))

    if center_sr > 1e-6:
        cliff_ratio = max(0.0, (center_sr - med_sr) / center_sr)
        retention_component = float(np.clip(med_sr / center_sr, 0.0, 1.0))
        stability_score = 0.5 * pos_frac + 0.5 * retention_component
    else:
        cliff_ratio = 1.0
        stability_score = 0.0
        warnings.append("Center candidate has non-positive net Sharpe ratio.")

    if len(neighbors) < 2:
        warnings.append(f"Only {len(neighbors)} neighbor evaluated; perturbation estimate is sparse.")

    return PerturbationResult(
        center_trial_id=center_trial.trial_id,
        center_parameters=dict(center_params),
        center_net_sharpe=center_sr,
        relative_step=relative_step,
        n_neighbors_evaluated=len(neighbors),
        positive_neighbor_fraction=pos_frac,
        median_neighbor_sharpe=med_sr,
        min_neighbor_sharpe=min_sr,
        sharpe_cliff_ratio=float(cliff_ratio),
        stability_score=float(stability_score),
        neighbors=tuple(neighbors),
        warnings=tuple(warnings),
    )
