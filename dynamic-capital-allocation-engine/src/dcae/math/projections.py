"""Exact Euclidean convex projections and Dykstra's alternating projection algorithm."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

from dcae.math.linalg import Matrix, Vector, dot, jacobi_eigen_symmetric, vec_norm2


def project_box(x: Sequence[float], lower: Sequence[float], upper: Sequence[float]) -> Vector:
    """Exact Euclidean projection onto [lower_i, upper_i]."""
    return [min(u, max(l, v)) for v, l, u in zip(x, lower, upper)]


def project_sum_band(x: Sequence[float], sum_min: float, sum_max: float) -> Vector:
    """Exact Euclidean projection onto {w : sum_min <= sum(w) <= sum_max}."""
    n = len(x)
    if n == 0:
        return []
    s = sum(x)
    if s < sum_min:
        shift = (sum_min - s) / n
        return [v + shift for v in x]
    if s > sum_max:
        shift = (sum_max - s) / n
        return [v + shift for v in x]
    return list(x)


def project_halfspace(x: Sequence[float], a: Sequence[float], b: float) -> Vector:
    """Exact Euclidean projection onto {w : a^T w <= b}."""
    val = dot(a, x)
    if val <= b:
        return list(x)
    norm_sq = dot(a, a)
    if norm_sq <= 1e-18:
        return list(x)
    scale = (val - b) / norm_sq
    return [xi - scale * ai for xi, ai in zip(x, a)]


def project_l1_ball(v: Sequence[float], radius: float) -> Vector:
    """Exact Euclidean projection onto {d : ||d||_1 <= radius} (Duchi et al., 2008)."""
    if radius <= 0.0:
        return [0.0 for _ in v]
    abs_v = [abs(x) for x in v]
    if sum(abs_v) <= radius:
        return list(v)

    u = sorted(abs_v, reverse=True)
    cssv = 0.0
    rho = -1
    theta = 0.0
    for j, uj in enumerate(u):
        cssv += uj
        t = (cssv - radius) / (j + 1)
        if uj - t > 0.0:
            rho = j
            theta = t
    if rho < 0:
        return [0.0 for _ in v]

    out: Vector = []
    for x in v:
        mag = max(abs(x) - theta, 0.0)
        out.append(mag if x >= 0.0 else -mag)
    return out


def project_turnover_ball(
    x: Sequence[float],
    w0: Sequence[float],
    max_one_way_turnover: float,
) -> Vector:
    """Exact Euclidean projection onto {w : 0.5 * ||w - w0||_1 <= max_one_way_turnover}."""
    radius = max(0.0, 2.0 * max_one_way_turnover)
    delta = [xi - wi for xi, wi in zip(x, w0)]
    proj_delta = project_l1_ball(delta, radius)
    return [wi + di for wi, di in zip(w0, proj_delta)]


def project_ellipsoid(
    x: Sequence[float],
    eigvals: Sequence[float],
    V: Sequence[Sequence[float]],
    max_quad_value: float,
) -> Vector:
    """Exact Euclidean projection onto {w : w^T Sigma w <= max_quad_value}.

    Sigma = V * diag(eigvals) * V^T with eigvals >= 0.
    """
    n = len(x)
    if n == 0 or max_quad_value <= 0.0:
        return [0.0] * n
    # Rotate into eigenbasis: y0 = V^T x
    y0 = [sum(V[i][k] * x[i] for i in range(n)) for k in range(n)]
    pos_eigs = [max(ev, 0.0) for ev in eigvals]
    curr_quad = sum(pos_eigs[k] * y0[k] * y0[k] for k in range(n))
    if curr_quad <= max_quad_value:
        return list(x)

    def phi(nu: float) -> float:
        total = 0.0
        for k in range(n):
            ek = pos_eigs[k]
            if ek > 0.0:
                denom = 1.0 + 2.0 * nu * ek
                yk = y0[k] / denom
                total += ek * yk * yk
        return total - max_quad_value

    lo = 0.0
    hi = 1.0
    while phi(hi) > 0.0 and hi < 1e14:
        hi *= 2.0
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        if phi(mid) > 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-13 * max(1.0, hi):
            break
    nu = 0.5 * (lo + hi)
    y = [y0[k] / (1.0 + 2.0 * nu * pos_eigs[k]) for k in range(n)]
    return [sum(V[i][k] * y[k] for k in range(n)) for i in range(n)]


@dataclass(frozen=True)
class LinearCap:
    """Linear inequality a^T w <= b."""

    name: str
    coeffs: Tuple[float, ...]
    rhs: float


@dataclass(frozen=True)
class ConvexConstraintSet:
    """Collection of closed convex constraints for Dykstra projection."""

    lower: Tuple[float, ...]
    upper: Tuple[float, ...]
    sum_min: float
    sum_max: float
    w0: Tuple[float, ...]
    max_one_way_turnover: Optional[float] = None
    linear_caps: Tuple[LinearCap, ...] = ()
    cov_eigvals: Optional[Tuple[float, ...]] = None
    cov_eigvecs: Optional[Matrix] = None
    max_variance: Optional[float] = None
    stressed_eigvals: Optional[Tuple[float, ...]] = None
    stressed_eigvecs: Optional[Matrix] = None
    max_stressed_variance: Optional[float] = None

    def max_violation(self, w: Sequence[float]) -> float:
        """Compute maximum scalar violation across all constraints in this set."""
        viol = 0.0
        for wi, lo, hi in zip(w, self.lower, self.upper):
            viol = max(viol, lo - wi, wi - hi)
        s = sum(w)
        viol = max(viol, self.sum_min - s, s - self.sum_max)
        if self.max_one_way_turnover is not None:
            to = 0.5 * sum(abs(wi - w0i) for wi, w0i in zip(w, self.w0))
            viol = max(viol, to - self.max_one_way_turnover)
        for cap in self.linear_caps:
            lhs = dot(cap.coeffs, w)
            viol = max(viol, lhs - cap.rhs)
        if (
            self.max_variance is not None
            and self.cov_eigvals is not None
            and self.cov_eigvecs is not None
        ):
            n = len(w)
            y = [sum(self.cov_eigvecs[i][k] * w[i] for i in range(n)) for k in range(n)]
            var = sum(max(self.cov_eigvals[k], 0.0) * y[k] * y[k] for k in range(n))
            vol = math.sqrt(max(var, 0.0))
            max_vol = math.sqrt(max(self.max_variance, 0.0))
            viol = max(viol, vol - max_vol)
        if (
            self.max_stressed_variance is not None
            and self.stressed_eigvals is not None
            and self.stressed_eigvecs is not None
        ):
            n = len(w)
            y = [sum(self.stressed_eigvecs[i][k] * w[i] for i in range(n)) for k in range(n)]
            var = sum(max(self.stressed_eigvals[k], 0.0) * y[k] * y[k] for k in range(n))
            vol = math.sqrt(max(var, 0.0))
            max_vol = math.sqrt(max(self.max_stressed_variance, 0.0))
            viol = max(viol, vol - max_vol)
        return viol


def dykstra_project(
    x0: Sequence[float],
    cset: ConvexConstraintSet,
    *,
    max_iters: int = 250,
    tol: float = 1e-10,
) -> Tuple[Vector, float, int]:
    """Project x0 onto the intersection of constraints in cset via Dykstra's algorithm.

    Returns (projected_w, max_violation, iterations).
    """
    n = len(x0)
    if n == 0:
        return [], 0.0, 0

    projectors: List[Callable[[Sequence[float]], Vector]] = [
        lambda v: project_box(v, cset.lower, cset.upper),
        lambda v: project_sum_band(v, cset.sum_min, cset.sum_max),
    ]
    if cset.max_one_way_turnover is not None:
        to_lim = cset.max_one_way_turnover
        w0 = cset.w0
        projectors.append(lambda v, lim=to_lim, base=w0: project_turnover_ball(v, base, lim))

    for cap in cset.linear_caps:
        projectors.append(
            lambda v, a=cap.coeffs, b=cap.rhs: project_halfspace(v, a, b)
        )

    if (
        cset.max_variance is not None
        and cset.cov_eigvals is not None
        and cset.cov_eigvecs is not None
    ):
        ev = cset.cov_eigvals
        V = cset.cov_eigvecs
        mv = cset.max_variance
        projectors.append(lambda v, e=ev, vecs=V, r=mv: project_ellipsoid(v, e, vecs, r))

    if (
        cset.max_stressed_variance is not None
        and cset.stressed_eigvals is not None
        and cset.stressed_eigvecs is not None
    ):
        sev = cset.stressed_eigvals
        sV = cset.stressed_eigvecs
        smv = cset.max_stressed_variance
        projectors.append(lambda v, e=sev, vecs=sV, r=smv: project_ellipsoid(v, e, vecs, r))

    m = len(projectors)
    x = list(x0)
    increments = [[0.0] * n for _ in range(m)]
    iters = 0

    for it in range(1, max_iters + 1):
        iters = it
        x_prev = list(x)
        for j, proj in enumerate(projectors):
            y = [x[i] + increments[j][i] for i in range(n)]
            x_new = proj(y)
            increments[j] = [y[i] - x_new[i] for i in range(n)]
            x = x_new
        step = vec_norm2([x[i] - x_prev[i] for i in range(n)])
        viol = cset.max_violation(x)
        if viol <= tol and step <= tol:
            break

    # Final clamp onto box for tiny numerical drift while tracking violation
    final_viol = cset.max_violation(x)
    return x, final_viol, iters


def eigen_cache(cov: Sequence[Sequence[float]]) -> Tuple[Tuple[float, ...], Matrix]:
    eigvals, V = jacobi_eigen_symmetric(cov)
    return tuple(eigvals), V
