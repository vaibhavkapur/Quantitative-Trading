"""Point-in-time parameter estimator with data quality guards and stress overlays."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from dcae.domain.models import EngineConfig, FundSnapshot, StrategySnapshot
from dcae.errors import (
    InsufficientHistoryError,
    NonPSDCovarianceError,
    ZeroVolatilityError,
)
from dcae.math.linalg import (
    Matrix,
    Vector,
    apply_correlation_stress,
    check_psd,
    cov_to_vol_and_corr,
    nearest_psd_spectral,
)
from dcae.snapshots.publisher import SnapshotPublisher


@dataclass(frozen=True)
class EstimationResult:
    """Point-in-time annualized parameter estimates and rich diagnostics."""

    strategy_ids: Tuple[str, ...]
    eligible_ids: Tuple[str, ...]
    excluded_reasons: Dict[str, str]
    expected_returns_annual: Dict[str, float]
    volatilities_annual: Dict[str, float]
    covariance_annual: Matrix
    stressed_covariance_annual: Matrix
    correlation_matrix: Matrix
    stressed_correlation_matrix: Matrix
    active_covariance_annual: Matrix
    correlation_stress_active: bool
    psd_repaired: bool
    original_min_eigenvalue: float
    repaired_min_eigenvalue: float
    psd_repair_frobenius_delta: float
    observation_counts: Dict[str, int]
    dropped_future_observations: Dict[str, int]
    warnings: Tuple[str, ...]

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "strategy_ids": list(self.strategy_ids),
            "eligible_ids": list(self.eligible_ids),
            "excluded_reasons": dict(self.excluded_reasons),
            "expected_returns_annual": {
                k: round(v, 6) for k, v in self.expected_returns_annual.items()
            },
            "volatilities_annual": {
                k: round(v, 6) for k, v in self.volatilities_annual.items()
            },
            "correlation_stress_active": self.correlation_stress_active,
            "psd_repaired": self.psd_repaired,
            "original_min_eigenvalue": float(self.original_min_eigenvalue),
            "repaired_min_eigenvalue": float(self.repaired_min_eigenvalue),
            "psd_repair_frobenius_delta": float(self.psd_repair_frobenius_delta),
            "observation_counts": dict(self.observation_counts),
            "dropped_future_observations": dict(self.dropped_future_observations),
            "warnings": list(self.warnings),
        }


class PointInTimeEstimator:
    """Computes annualized expected returns, volatilities, and PSD covariance strictly PIT."""

    def __init__(self, config: EngineConfig) -> None:
        self.config = config

    def estimate(self, snapshot: FundSnapshot) -> EstimationResult:
        pit_series, dropped_future = SnapshotPublisher.point_in_time_filter(snapshot)
        ann = float(self.config.annualization_factor)
        sqrt_ann = math.sqrt(ann)

        all_ids = tuple(s.strategy_id for s in snapshot.strategies)
        strat_by_id: Dict[str, StrategySnapshot] = {s.strategy_id: s for s in snapshot.strategies}

        excluded: Dict[str, str] = {}
        warnings: List[str] = []
        obs_counts: Dict[str, int] = {}
        raw_means: Dict[str, float] = {}
        raw_vols: Dict[str, float] = {}

        # Step 1: Per-strategy eligibility, history, and zero-volatility screening
        for sid in all_ids:
            strat = strat_by_id[sid]
            vals = pit_series[sid]
            obs_counts[sid] = len(vals)

            if dropped_future[sid] > 0:
                warnings.append(
                    f"Filtered {dropped_future[sid]} future/unavailable return observations for {sid}"
                )

            if not strat.eligible:
                excluded[sid] = f"marked_ineligible:{strat.eligibility_reason}"
                raw_means[sid] = 0.0
                raw_vols[sid] = 0.0
                continue

            if len(vals) < self.config.min_history_observations:
                if self.config.insufficient_history_policy == "reject":
                    raise InsufficientHistoryError(
                        f"Strategy {sid} has {len(vals)} point-in-time observations "
                        f"(minimum required: {self.config.min_history_observations})",
                        context={
                            "strategy_id": sid,
                            "observations": len(vals),
                            "min_history_observations": self.config.min_history_observations,
                        },
                    )
                excluded[sid] = (
                    f"insufficient_history:{len(vals)}<{self.config.min_history_observations}"
                )
                warnings.append(
                    f"Excluded {sid} due to insufficient history "
                    f"({len(vals)} < {self.config.min_history_observations})"
                )
                raw_means[sid] = 0.0
                raw_vols[sid] = 0.0
                continue

            mean_p = sum(vals) / len(vals)
            var_p = sum((v - mean_p) ** 2 for v in vals) / max(1, len(vals) - 1)
            vol_p = math.sqrt(max(var_p, 0.0))
            ann_vol = vol_p * sqrt_ann

            if ann_vol <= self.config.zero_volatility_eps:
                if self.config.zero_volatility_policy == "reject":
                    raise ZeroVolatilityError(
                        f"Strategy {sid} has zero/degenerate annualized volatility ({ann_vol:.3e})",
                        context={
                            "strategy_id": sid,
                            "annualized_volatility": ann_vol,
                            "zero_volatility_eps": self.config.zero_volatility_eps,
                        },
                    )
                if self.config.zero_volatility_policy == "exclude":
                    excluded[sid] = f"zero_volatility:{ann_vol:.3e}"
                    warnings.append(
                        f"Excluded {sid} due to zero/near-zero volatility ({ann_vol:.3e})"
                    )
                    raw_means[sid] = 0.0
                    raw_vols[sid] = 0.0
                    continue
                # floor policy
                ann_vol = max(ann_vol, math.sqrt(self.config.psd_min_eigenvalue))
                warnings.append(f"Floored zero volatility for {sid} to {ann_vol:.3e}")

            raw_means[sid] = mean_p * ann
            raw_vols[sid] = ann_vol

        eligible_ids = tuple(sid for sid in all_ids if sid not in excluded)
        if not eligible_ids:
            warnings.append("No eligible strategies passed point-in-time screening")

        # Return shrinkage toward cross-sectional mean of eligible strategies
        if eligible_ids:
            grand_mean = sum(raw_means[sid] for sid in eligible_ids) / len(eligible_ids)
        else:
            grand_mean = 0.0
        alpha_ret = max(0.0, min(1.0, self.config.return_shrinkage))
        expected_returns: Dict[str, float] = {}
        for sid in all_ids:
            if sid in excluded:
                expected_returns[sid] = 0.0
            else:
                expected_returns[sid] = (1.0 - alpha_ret) * raw_means[sid] + alpha_ret * grand_mean

        # Step 2: Build raw annualized covariance matrix over all_ids
        n = len(all_ids)
        if snapshot.external_covariance is not None:
            raw_cov = self._align_external_covariance(snapshot, all_ids)
        else:
            raw_cov = self._estimate_sample_covariance(
                all_ids=all_ids,
                excluded=excluded,
                pit_series=pit_series,
                ann=ann,
                raw_vols=raw_vols,
            )

        # Apply diagonal shrinkage on sample covariance (not when testing raw external non-PSD)
        if snapshot.external_covariance is None and n > 0:
            lam_cov = max(0.0, min(1.0, self.config.covariance_shrinkage))
            shrunk_cov = [[0.0] * n for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    if i == j:
                        shrunk_cov[i][j] = raw_cov[i][j]
                    else:
                        shrunk_cov[i][j] = (1.0 - lam_cov) * raw_cov[i][j]
            cov_to_check = shrunk_cov
        else:
            cov_to_check = raw_cov

        # Step 3: PSD verification & repair/rejection
        is_psd, orig_min_eig, _, _ = check_psd(cov_to_check, tol=1e-10)
        psd_repaired = False
        repair_delta = 0.0
        repaired_min_eig = orig_min_eig

        if not is_psd:
            if self.config.non_psd_policy == "reject":
                raise NonPSDCovarianceError(
                    f"Estimated covariance matrix is non-PSD (min eigenvalue = {orig_min_eig:.6e})",
                    context={
                        "min_eigenvalue": orig_min_eig,
                        "non_psd_policy": self.config.non_psd_policy,
                    },
                )
            repaired_cov, orig_min_eig, repair_delta, clipped_eigs, _ = nearest_psd_spectral(
                cov_to_check,
                min_eigenvalue=self.config.psd_min_eigenvalue,
            )
            cov_annual = repaired_cov
            repaired_min_eig = min(clipped_eigs) if clipped_eigs else 0.0
            psd_repaired = True
            warnings.append(
                f"Repaired non-PSD covariance (min_eig {orig_min_eig:.3e} -> "
                f"{repaired_min_eig:.3e}, frobenius_delta={repair_delta:.3e})"
            )
        else:
            # Ensure small positive diagonal floor for numerical stability on active strategies
            cov_annual = [list(row) for row in cov_to_check]
            for i, sid in enumerate(all_ids):
                if sid not in excluded and cov_annual[i][i] < self.config.psd_min_eigenvalue:
                    cov_annual[i][i] = self.config.psd_min_eigenvalue

        vols_list, corr_matrix = cov_to_vol_and_corr(cov_annual)
        vols_dict = {
            sid: (0.0 if sid in excluded else vols_list[i])
            for i, sid in enumerate(all_ids)
        }

        # Step 4: Correlated drawdown detection & correlation stress overlay
        stressed_cov, stressed_corr = apply_correlation_stress(
            cov_annual,
            blend=self.config.correlation_stress_blend,
            stress_floor=self.config.correlation_stress_floor,
            min_eigenvalue=self.config.psd_min_eigenvalue,
        )

        # Detect correlated drawdown across eligible strategies
        in_drawdown = [
            sid
            for sid in eligible_ids
            if strat_by_id[sid].current_drawdown <= self.config.drawdown_stress_trigger
        ]
        correlated_drawdown_triggered = len(eligible_ids) >= 2 and (
            len(in_drawdown) >= max(2, math.ceil(0.5 * len(eligible_ids)))
        )
        stress_active = self.config.correlation_stress_enabled or correlated_drawdown_triggered
        if correlated_drawdown_triggered:
            warnings.append(
                f"Correlated drawdown regime detected ({len(in_drawdown)}/{len(eligible_ids)} "
                f"eligible strategies below {self.config.drawdown_stress_trigger:.1%}); "
                "activating stressed covariance overlay"
            )

        active_cov = stressed_cov if stress_active else cov_annual

        return EstimationResult(
            strategy_ids=all_ids,
            eligible_ids=eligible_ids,
            excluded_reasons=excluded,
            expected_returns_annual=expected_returns,
            volatilities_annual=vols_dict,
            covariance_annual=cov_annual,
            stressed_covariance_annual=stressed_cov,
            correlation_matrix=corr_matrix,
            stressed_correlation_matrix=stressed_corr,
            active_covariance_annual=active_cov,
            correlation_stress_active=stress_active,
            psd_repaired=psd_repaired,
            original_min_eigenvalue=orig_min_eig,
            repaired_min_eigenvalue=repaired_min_eig,
            psd_repair_frobenius_delta=repair_delta,
            observation_counts=obs_counts,
            dropped_future_observations=dropped_future,
            warnings=tuple(warnings),
        )

    def _align_external_covariance(
        self,
        snapshot: FundSnapshot,
        all_ids: Tuple[str, ...],
    ) -> Matrix:
        ext = snapshot.external_covariance
        ext_ids = snapshot.external_covariance_ids
        assert ext is not None and ext_ids is not None
        idx_map = {sid: i for i, sid in enumerate(ext_ids)}
        n = len(all_ids)
        cov = [[0.0] * n for _ in range(n)]
        for i, si in enumerate(all_ids):
            for j, sj in enumerate(all_ids):
                if si in idx_map and sj in idx_map:
                    cov[i][j] = float(ext[idx_map[si]][idx_map[sj]])
        return cov

    def _estimate_sample_covariance(
        self,
        *,
        all_ids: Tuple[str, ...],
        excluded: Dict[str, str],
        pit_series: Dict[str, List[float]],
        ann: float,
        raw_vols: Dict[str, float],
    ) -> Matrix:
        n = len(all_ids)
        cov = [[0.0] * n for _ in range(n)]
        for i, si in enumerate(all_ids):
            if si in excluded:
                cov[i][i] = self.config.psd_min_eigenvalue
                continue
            xi = pit_series[si]
            for j in range(i, n):
                sj = all_ids[j]
                if sj in excluded:
                    continue
                xj = pit_series[sj]
                # Align on common trailing window
                m = min(len(xi), len(xj))
                if m < 2:
                    cval = 0.0
                else:
                    ai = xi[-m:]
                    aj = xj[-m:]
                    mi = sum(ai) / m
                    mj = sum(aj) / m
                    cval = sum((ai[k] - mi) * (aj[k] - mj) for k in range(m)) / (m - 1) * ann
                if i == j:
                    cval = max(cval, raw_vols[si] ** 2, self.config.psd_min_eigenvalue)
                cov[i][j] = cval
                cov[j][i] = cval
        return cov
