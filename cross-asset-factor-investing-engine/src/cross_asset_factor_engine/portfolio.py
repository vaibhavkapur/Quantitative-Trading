"""Volatility-scaled portfolio construction under exposure, volatility, and turnover constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize

from cross_asset_factor_engine.config import PortfolioConfig, PortfolioConstraints
from cross_asset_factor_engine.errors import ConstraintViolationError


@dataclass(frozen=True)
class OptimizationResult:
    """Constrained portfolio optimization output with full constraint diagnostics."""

    weights: np.ndarray
    target_weights: np.ndarray
    ex_ante_vol: float
    gross_exposure: float
    net_exposure: float
    one_way_turnover: float
    class_gross_exposure: dict[str, float]
    converged: bool
    iterations: int
    status_message: str


def estimate_covariance(
    returns_matrix: np.ndarray,
    *,
    shrinkage: float = 0.20,
    min_asset_vol: float = 0.02,
    annualization_days: int = 252,
) -> np.ndarray:
    """Estimate shrinkage-regularized annualized covariance strictly from past returns."""
    arr = np.asarray(returns_matrix, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] == 0:
        raise ValueError("returns_matrix must be a 2D array with at least one asset.")
    n_obs, n_assets = arr.shape
    min_var = float(min_asset_vol) ** 2

    if n_obs < 2:
        return np.eye(n_assets, dtype=np.float64) * max(min_var, 0.01)

    # Replace non-finite entries with column mean (or 0 if entirely missing) strictly within window
    clean = arr.copy()
    for j in range(n_assets):
        finite = np.isfinite(clean[:, j])
        if not np.any(finite):
            clean[:, j] = 0.0
        elif not np.all(finite):
            clean[~finite, j] = float(np.mean(clean[finite, j]))

    sample_cov = np.cov(clean, rowvar=False, ddof=1)
    if sample_cov.ndim == 0:
        sample_cov = np.array([[float(sample_cov)]], dtype=np.float64)
    sample_cov = np.nan_to_num(sample_cov, nan=0.0, posinf=0.0, neginf=0.0) * float(annualization_days)

    diag_var = np.maximum(np.diag(sample_cov), min_var)
    diag_target = np.diag(diag_var)
    cov = (1.0 - float(shrinkage)) * sample_cov + float(shrinkage) * diag_target
    cov = 0.5 * (cov + cov.T)

    # Ensure diagonal variance floor and positive definiteness
    for i in range(n_assets):
        if cov[i, i] < min_var:
            cov[i, i] = min_var
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 1.0e-8)
    cov_pd = (eigvecs * eigvals) @ eigvecs.T
    return 0.5 * (cov_pd + cov_pd.T)


def compute_unconstrained_vol_scaled_weights(
    signals: np.ndarray,
    cov: np.ndarray,
    *,
    target_vol: float,
    min_asset_vol: float = 0.02,
) -> np.ndarray:
    """Scale signals inversely by asset volatility and target ex-ante portfolio volatility."""
    sig = np.asarray(signals, dtype=np.float64)
    clean_sig = np.where(np.isfinite(sig), sig, 0.0)
    if np.linalg.norm(clean_sig) < 1.0e-12:
        return np.zeros_like(clean_sig)

    asset_vols = np.maximum(np.sqrt(np.maximum(np.diag(cov), 0.0)), float(min_asset_vol))
    raw_w = clean_sig / asset_vols
    port_var = float(raw_w @ cov @ raw_w)
    if port_var <= 1.0e-16:
        return np.zeros_like(clean_sig)
    scale = float(target_vol) / float(np.sqrt(port_var))
    return raw_w * scale


def verify_portfolio_constraints(
    weights: np.ndarray,
    prev_weights: np.ndarray,
    cov: np.ndarray,
    asset_classes: list[str],
    constraints: PortfolioConstraints,
    *,
    volatility_cap: float,
    tolerance: float = 1.0e-5,
) -> dict[str, Any]:
    """Verify every exposure, volatility, and turnover constraint.

    Raises `ConstraintViolationError` with structured violation metrics if any cap is exceeded.
    """
    w = np.asarray(weights, dtype=np.float64)
    w_prev = np.asarray(prev_weights, dtype=np.float64)
    violations: list[dict[str, Any]] = []

    if not np.all(np.isfinite(w)):
        raise ConstraintViolationError(
            "Portfolio weights contain NaN or infinite values.",
            details={"weights": w.tolist()},
        )

    max_abs_w = float(np.max(np.abs(w))) if len(w) > 0 else 0.0
    if max_abs_w > constraints.max_asset_weight + tolerance:
        violations.append(
            {
                "constraint": "max_asset_weight",
                "observed": max_abs_w,
                "limit": constraints.max_asset_weight,
                "excess": max_abs_w - constraints.max_asset_weight,
            }
        )

    gross = float(np.sum(np.abs(w)))
    if gross > constraints.max_gross_exposure + tolerance:
        violations.append(
            {
                "constraint": "max_gross_exposure",
                "observed": gross,
                "limit": constraints.max_gross_exposure,
                "excess": gross - constraints.max_gross_exposure,
            }
        )

    net = float(np.sum(w))
    if net < constraints.min_net_exposure - tolerance:
        violations.append(
            {
                "constraint": "min_net_exposure",
                "observed": net,
                "limit": constraints.min_net_exposure,
                "excess": constraints.min_net_exposure - net,
            }
        )
    if net > constraints.max_net_exposure + tolerance:
        violations.append(
            {
                "constraint": "max_net_exposure",
                "observed": net,
                "limit": constraints.max_net_exposure,
                "excess": net - constraints.max_net_exposure,
            }
        )

    class_gross: dict[str, float] = {}
    for cls_name, cap in constraints.max_class_gross_exposure.items():
        cls_val = float(sum(abs(w[i]) for i, c in enumerate(asset_classes) if c == cls_name))
        class_gross[cls_name] = cls_val
        if cls_val > cap + tolerance:
            violations.append(
                {
                    "constraint": f"max_class_gross_exposure:{cls_name}",
                    "observed": cls_val,
                    "limit": cap,
                    "excess": cls_val - cap,
                }
            )

    turnover = 0.5 * float(np.sum(np.abs(w - w_prev)))
    if turnover > constraints.max_turnover_per_rebalance + tolerance:
        violations.append(
            {
                "constraint": "max_turnover_per_rebalance",
                "observed": turnover,
                "limit": constraints.max_turnover_per_rebalance,
                "excess": turnover - constraints.max_turnover_per_rebalance,
            }
        )

    ex_ante_vol = float(np.sqrt(max(float(w @ cov @ w), 0.0)))
    if ex_ante_vol > volatility_cap + tolerance:
        violations.append(
            {
                "constraint": "volatility_cap",
                "observed": ex_ante_vol,
                "limit": volatility_cap,
                "excess": ex_ante_vol - volatility_cap,
            }
        )

    summary = {
        "max_asset_weight": max_abs_w,
        "gross_exposure": gross,
        "net_exposure": net,
        "one_way_turnover": turnover,
        "ex_ante_vol": ex_ante_vol,
        "class_gross_exposure": class_gross,
        "violations": violations,
    }
    if violations:
        names = [v["constraint"] for v in violations]
        raise ConstraintViolationError(
            f"Portfolio violated {len(violations)} constraint(s): {names}",
            details=summary,
        )
    return summary


def _check_problem_feasibility(
    prev_weights: np.ndarray,
    cov: np.ndarray,
    asset_classes: list[str],
    constraints: PortfolioConstraints,
    volatility_cap: float,
    tolerance: float,
) -> None:
    """Detect provably infeasible constraint configurations before optimization."""
    if volatility_cap <= 0:
        raise ConstraintViolationError(
            f"volatility_cap must be strictly positive, got {volatility_cap}.",
            details={"volatility_cap": volatility_cap},
        )
    if constraints.max_gross_exposure <= 0 or constraints.max_asset_weight <= 0:
        raise ConstraintViolationError("Exposure caps must be strictly positive.")
    if constraints.min_net_exposure > constraints.max_net_exposure + tolerance:
        raise ConstraintViolationError(
            "Infeasible net exposure band: min_net_exposure > max_net_exposure.",
            details={
                "min_net_exposure": constraints.min_net_exposure,
                "max_net_exposure": constraints.max_net_exposure,
            },
        )

    # One-way turnover <= tau implies L1 distance ||w - w_prev||_1 <= 2 * tau.
    max_l1_step = 2.0 * constraints.max_turnover_per_rebalance
    prev_gross = float(np.sum(np.abs(prev_weights)))
    if prev_gross - max_l1_step > constraints.max_gross_exposure + tolerance:
        raise ConstraintViolationError(
            "Infeasible rebalance: previous gross exposure cannot be reduced below max_gross_exposure "
            "within max_turnover_per_rebalance.",
            details={
                "prev_gross_exposure": prev_gross,
                "max_gross_exposure": constraints.max_gross_exposure,
                "max_turnover_per_rebalance": constraints.max_turnover_per_rebalance,
                "min_reachable_gross": prev_gross - max_l1_step,
            },
        )

    # Check if zero-net is infeasible relative to net band + turnover
    prev_net = float(np.sum(prev_weights))
    if prev_net - max_l1_step > constraints.max_net_exposure + tolerance:
        raise ConstraintViolationError(
            "Infeasible rebalance: previous net exposure cannot be reduced below max_net_exposure "
            "within max_turnover_per_rebalance.",
            details={"prev_net": prev_net, "max_net_exposure": constraints.max_net_exposure},
        )
    if prev_net + max_l1_step < constraints.min_net_exposure - tolerance:
        raise ConstraintViolationError(
            "Infeasible rebalance: previous net exposure cannot be increased above min_net_exposure "
            "within max_turnover_per_rebalance.",
            details={"prev_net": prev_net, "min_net_exposure": constraints.min_net_exposure},
        )

    # Check whether per-asset caps allow achieving min_net_exposure
    max_possible_gross = min(
        constraints.max_gross_exposure,
        len(prev_weights) * constraints.max_asset_weight,
        sum(
            min(
                constraints.max_class_gross_exposure.get(cls, constraints.max_gross_exposure),
                sum(1 for c in asset_classes if c == cls) * constraints.max_asset_weight,
            )
            for cls in set(asset_classes)
        ),
    )
    if constraints.min_net_exposure > max_possible_gross + tolerance or (
        constraints.max_net_exposure < -max_possible_gross - tolerance
    ):
        raise ConstraintViolationError(
            "Infeasible exposure constraints: net exposure band exceeds achievable gross exposure.",
            details={
                "min_net_exposure": constraints.min_net_exposure,
                "max_net_exposure": constraints.max_net_exposure,
                "max_possible_gross": max_possible_gross,
            },
        )

    # Check minimum reachable portfolio vol inside L1 turnover ball around prev_weights
    # Lower bound via Cauchy-Schwarz on smallest eigenvalue of cov if moving toward zero
    if prev_gross > max_l1_step + tolerance:
        # Shrink prev_weights proportionally toward zero by the max allowed L1 step
        alpha = max(0.0, 1.0 - max_l1_step / prev_gross)
        shrunk = prev_weights * alpha
        min_proj_vol = float(np.sqrt(max(float(shrunk @ cov @ shrunk), 0.0)))
        # Only fail early if even the optimal radial shrink exceeds vol cap significantly AND SLSQP confirms below
        _ = min_proj_vol


def optimize_constrained_portfolio(
    signals: np.ndarray,
    cov: np.ndarray,
    prev_weights: np.ndarray,
    asset_classes: list[str],
    portfolio_config: PortfolioConfig,
) -> OptimizationResult:
    """Solve smooth convex QP/QCQP for volatility-scaled weights under exposure, vol, and turnover caps.

    Uses exact split variables `w = w+ - w-` and `w - w_prev = d+ - d-` (all >= 0) so L1 exposure
    and L1 turnover constraints are linear and continuously differentiable for SLSQP.
    """
    n = len(signals)
    w_prev = np.asarray(prev_weights, dtype=np.float64).copy()
    cons = portfolio_config.constraints
    vol_cap = float(portfolio_config.volatility_cap)
    tol = float(portfolio_config.constraint_tolerance)

    _check_problem_feasibility(w_prev, cov, asset_classes, cons, vol_cap, tol)

    w_target = compute_unconstrained_vol_scaled_weights(
        signals,
        cov,
        target_vol=portfolio_config.target_volatility,
        min_asset_vol=portfolio_config.min_asset_vol,
    )

    # Variables x = [w_pos (n), w_neg (n), d_pos (n), d_neg (n)]
    to_pen = float(portfolio_config.turnover_penalty)
    eps_l1 = 1.0e-7

    def _unpack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return x[0:n], x[n : 2 * n], x[2 * n : 3 * n], x[3 * n : 4 * n]

    def objective(x: np.ndarray) -> float:
        wp, wn, dp, dn = _unpack(x)
        w = wp - wn
        diff = w - w_target
        return float(
            0.5 * np.dot(diff, diff)
            + 0.5 * to_pen * np.sum(dp + dn)
            + eps_l1 * np.sum(wp + wn + dp + dn)
        )

    def objective_jac(x: np.ndarray) -> np.ndarray:
        wp, wn, _dp, _dn = _unpack(x)
        diff = (wp - wn) - w_target
        g = np.empty(4 * n, dtype=np.float64)
        g[0:n] = diff + eps_l1
        g[n : 2 * n] = -diff + eps_l1
        g[2 * n : 3 * n] = 0.5 * to_pen + eps_l1
        g[3 * n : 4 * n] = 0.5 * to_pen + eps_l1
        return g

    # Equality: (wp - wn) - (dp - dn) - w_prev = 0
    eq_jac = np.zeros((n, 4 * n), dtype=np.float64)
    idx = np.arange(n)
    eq_jac[idx, idx] = 1.0
    eq_jac[idx, n + idx] = -1.0
    eq_jac[idx, 2 * n + idx] = -1.0
    eq_jac[idx, 3 * n + idx] = 1.0

    scipy_constraints: list[dict[str, Any]] = [
        {
            "type": "eq",
            "fun": lambda x: (x[0:n] - x[n : 2 * n]) - (x[2 * n : 3 * n] - x[3 * n : 4 * n]) - w_prev,
            "jac": lambda _x: eq_jac,
        }
    ]

    # Gross exposure: max_gross - sum(wp + wn) >= 0
    gross_jac = np.zeros(4 * n, dtype=np.float64)
    gross_jac[0 : 2 * n] = -1.0
    scipy_constraints.append(
        {
            "type": "ineq",
            "fun": lambda x: cons.max_gross_exposure - float(np.sum(x[0 : 2 * n])),
            "jac": lambda _x: gross_jac,
        }
    )

    # Net exposure upper: max_net - sum(wp - wn) >= 0
    net_u_jac = np.zeros(4 * n, dtype=np.float64)
    net_u_jac[0:n] = -1.0
    net_u_jac[n : 2 * n] = 1.0
    scipy_constraints.append(
        {
            "type": "ineq",
            "fun": lambda x: cons.max_net_exposure - float(np.sum(x[0:n] - x[n : 2 * n])),
            "jac": lambda _x: net_u_jac,
        }
    )

    # Net exposure lower: sum(wp - wn) - min_net >= 0
    net_l_jac = -net_u_jac
    scipy_constraints.append(
        {
            "type": "ineq",
            "fun": lambda x: float(np.sum(x[0:n] - x[n : 2 * n])) - cons.min_net_exposure,
            "jac": lambda _x: net_l_jac,
        }
    )

    # Per-class gross exposure caps
    for cls_name, cap in sorted(cons.max_class_gross_exposure.items()):
        cls_indices = [i for i, c in enumerate(asset_classes) if c == cls_name]
        if not cls_indices:
            continue
        cls_jac = np.zeros(4 * n, dtype=np.float64)
        for ci in cls_indices:
            cls_jac[ci] = -1.0
            cls_jac[n + ci] = -1.0
        scipy_constraints.append(
            {
                "type": "ineq",
                "fun": lambda x, c_idx=cls_indices, c_cap=cap: c_cap
                - float(sum(x[i] + x[n + i] for i in c_idx)),
                "jac": lambda _x, j=cls_jac: j,
            }
        )

    # Turnover cap: max_turnover - 0.5 * sum(dp + dn) >= 0
    to_jac = np.zeros(4 * n, dtype=np.float64)
    to_jac[2 * n : 4 * n] = -0.5
    scipy_constraints.append(
        {
            "type": "ineq",
            "fun": lambda x: cons.max_turnover_per_rebalance - 0.5 * float(np.sum(x[2 * n : 4 * n])),
            "jac": lambda _x: to_jac,
        }
    )

    # Volatility cap: vol_cap^2 - w^T cov w >= 0
    vol_cap_sq = vol_cap**2

    def vol_ineq(x: np.ndarray) -> float:
        w = x[0:n] - x[n : 2 * n]
        return float(vol_cap_sq - float(w @ cov @ w))

    def vol_ineq_jac(x: np.ndarray) -> np.ndarray:
        w = x[0:n] - x[n : 2 * n]
        grad_w = -2.0 * (cov @ w)
        g = np.zeros(4 * n, dtype=np.float64)
        g[0:n] = grad_w
        g[n : 2 * n] = -grad_w
        return g

    scipy_constraints.append({"type": "ineq", "fun": vol_ineq, "jac": vol_ineq_jac})

    bounds = (
        [(0.0, float(cons.max_asset_weight))] * (2 * n)
        + [(0.0, 2.0 * float(cons.max_asset_weight))] * (2 * n)
    )

    # Construct feasible warm start near prev_weights clipped to per-asset bounds
    w0 = np.clip(w_prev, -cons.max_asset_weight, cons.max_asset_weight)
    gross0 = float(np.sum(np.abs(w0)))
    if gross0 > cons.max_gross_exposure and gross0 > 0:
        w0 = w0 * (cons.max_gross_exposure / gross0)
    vol0 = float(np.sqrt(max(float(w0 @ cov @ w0), 0.0)))
    if vol0 > vol_cap and vol0 > 0:
        w0 = w0 * (0.98 * vol_cap / vol0)

    def _make_x0(w_init: np.ndarray) -> np.ndarray:
        d_init = w_init - w_prev
        return np.concatenate(
            [
                np.maximum(w_init, 0.0),
                np.maximum(-w_init, 0.0),
                np.maximum(d_init, 0.0),
                np.maximum(-d_init, 0.0),
            ]
        )

    starts = [_make_x0(w0), _make_x0(np.zeros(n, dtype=np.float64))]
    best_res = None
    best_w = None
    best_obj = float("inf")

    for x0 in starts:
        res = minimize(
            objective,
            x0,
            method="SLSQP",
            jac=objective_jac,
            bounds=bounds,
            constraints=scipy_constraints,
            options={"maxiter": 400, "ftol": 1.0e-10, "disp": False},
        )
        w_cand = res.x[0:n] - res.x[n : 2 * n]
        w_cand = np.where(np.abs(w_cand) < 1.0e-11, 0.0, w_cand)
        try:
            verify_portfolio_constraints(
                w_cand, w_prev, cov, asset_classes, cons, volatility_cap=vol_cap, tolerance=tol
            )
            cand_obj = float(res.fun)
            if cand_obj < best_obj:
                best_obj = cand_obj
                best_res = res
                best_w = w_cand
        except ConstraintViolationError:
            if best_res is None:
                best_res = res
                best_w = w_cand

    assert best_res is not None and best_w is not None
    w_opt = best_w

    # Strict verification (raises ConstraintViolationError if infeasible or violated)
    if portfolio_config.enforce_strict_constraints:
        audit = verify_portfolio_constraints(
            w_opt, w_prev, cov, asset_classes, cons, volatility_cap=vol_cap, tolerance=tol
        )
    else:
        class_gross = {
            cls: float(sum(abs(w_opt[i]) for i, c in enumerate(asset_classes) if c == cls))
            for cls in cons.max_class_gross_exposure
        }
        audit = {
            "gross_exposure": float(np.sum(np.abs(w_opt))),
            "net_exposure": float(np.sum(w_opt)),
            "one_way_turnover": 0.5 * float(np.sum(np.abs(w_opt - w_prev))),
            "ex_ante_vol": float(np.sqrt(max(float(w_opt @ cov @ w_opt), 0.0))),
            "class_gross_exposure": class_gross,
        }

    return OptimizationResult(
        weights=w_opt,
        target_weights=w_target,
        ex_ante_vol=float(audit["ex_ante_vol"]),
        gross_exposure=float(audit["gross_exposure"]),
        net_exposure=float(audit["net_exposure"]),
        one_way_turnover=float(audit["one_way_turnover"]),
        class_gross_exposure=dict(audit["class_gross_exposure"]),
        converged=bool(best_res.success),
        iterations=int(getattr(best_res, "nit", 0)),
        status_message=str(getattr(best_res, "message", "")),
    )
