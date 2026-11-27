"""Pure-Python numerical linear algebra, Jacobi eigendecomposition, and PSD repair.

Designed for deterministic, zero-dependency execution on strategy counts typical
of multi-strategy funds (N <= 100), with optional NumPy/SciPy parity.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

Vector = List[float]
Matrix = List[List[float]]


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def vec_norm2(a: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def vec_l1_dist(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b))


def mat_vec(A: Sequence[Sequence[float]], x: Sequence[float]) -> Vector:
    return [sum(row[j] * x[j] for j in range(len(x))) for row in A]


def quad_form(x: Sequence[float], A: Sequence[Sequence[float]]) -> float:
    Ax = mat_vec(A, x)
    return dot(x, Ax)


def transpose(A: Sequence[Sequence[float]]) -> Matrix:
    if not A:
        return []
    rows = len(A)
    cols = len(A[0])
    return [[A[i][j] for i in range(rows)] for j in range(cols)]


def mat_mul(A: Sequence[Sequence[float]], B: Sequence[Sequence[float]]) -> Matrix:
    n = len(A)
    k_len = len(B)
    m = len(B[0]) if B else 0
    Bt = transpose(B)
    return [[sum(A[i][k] * Bt[j][k] for k in range(k_len)) for j in range(m)] for i in range(n)]


def symmetrize(A: Sequence[Sequence[float]]) -> Matrix:
    n = len(A)
    return [[0.5 * (A[i][j] + A[j][i]) for j in range(n)] for i in range(n)]


def frobenius_norm(A: Sequence[Sequence[float]]) -> float:
    return math.sqrt(sum(val * val for row in A for val in row))


def max_asymmetry(A: Sequence[Sequence[float]]) -> float:
    n = len(A)
    best = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            diff = abs(A[i][j] - A[j][i])
            if diff > best:
                best = diff
    return best


def jacobi_eigen_symmetric(
    A: Sequence[Sequence[float]],
    *,
    max_sweeps: int = 120,
    tol: float = 1e-13,
) -> Tuple[Vector, Matrix]:
    """Compute eigenvalues and orthonormal eigenvectors of a real symmetric matrix.

    Returns (eigenvalues_sorted_desc, V) where columns of V are eigenvectors
    satisfying A = V * diag(eigenvalues) * V^T.
    """
    n = len(A)
    if n == 0:
        return [], []
    if n == 1:
        return [float(A[0][0])], [[1.0]]

    M = symmetrize(A)
    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    for _ in range(max_sweeps):
        # Find largest off-diagonal element
        max_val = 0.0
        p = 0
        q = 1
        for i in range(n):
            row = M[i]
            for j in range(i + 1, n):
                av = abs(row[j])
                if av > max_val:
                    max_val = av
                    p, q = i, j
        if max_val <= tol:
            break

        app = M[p][p]
        aqq = M[q][q]
        apq = M[p][q]
        if abs(apq) <= tol:
            break

        tau = (aqq - app) / (2.0 * apq)
        if tau >= 0.0:
            t = 1.0 / (tau + math.sqrt(1.0 + tau * tau))
        else:
            t = -1.0 / (-tau + math.sqrt(1.0 + tau * tau))
        c = 1.0 / math.sqrt(1.0 + t * t)
        s = t * c

        # Update M
        M[p][p] = app - t * apq
        M[q][q] = aqq + t * apq
        M[p][q] = 0.0
        M[q][p] = 0.0

        for k in range(n):
            if k != p and k != q:
                mkp = M[k][p]
                mkq = M[k][q]
                new_kp = c * mkp - s * mkq
                new_kq = s * mkp + c * mkq
                M[k][p] = new_kp
                M[p][k] = new_kp
                M[k][q] = new_kq
                M[q][k] = new_kq

        # Update eigenvectors V
        for k in range(n):
            vkp = V[k][p]
            vkq = V[k][q]
            V[k][p] = c * vkp - s * vkq
            V[k][q] = s * vkp + c * vkq

    eigvals = [M[i][i] for i in range(n)]
    order = sorted(range(n), key=lambda idx: eigvals[idx], reverse=True)
    sorted_vals = [eigvals[idx] for idx in order]
    sorted_V = [[V[row][idx] for idx in order] for row in range(n)]
    return sorted_vals, sorted_V


def reconstruct_from_eigen(eigvals: Sequence[float], V: Sequence[Sequence[float]]) -> Matrix:
    """Reconstruct symmetric matrix V * diag(eigvals) * V^T."""
    n = len(eigvals)
    out = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            s = sum(V[i][k] * eigvals[k] * V[j][k] for k in range(n))
            out[i][j] = s
            out[j][i] = s
    return out


def check_psd(
    A: Sequence[Sequence[float]],
    *,
    tol: float = 1e-10,
) -> Tuple[bool, float, Vector, Matrix]:
    """Check if matrix A is symmetric positive semidefinite within tolerance."""
    eigvals, V = jacobi_eigen_symmetric(A)
    min_eig = min(eigvals) if eigvals else 0.0
    return (min_eig >= -tol), min_eig, eigvals, V


def nearest_psd_spectral(
    A: Sequence[Sequence[float]],
    *,
    min_eigenvalue: float = 1e-7,
) -> Tuple[Matrix, float, float, Vector, Matrix]:
    """Project a symmetric matrix onto the PSD cone by spectral eigenvalue clipping.

    Returns (repaired_matrix, original_min_eig, frobenius_repair_delta, clipped_eigs, V).
    """
    sym = symmetrize(A)
    eigvals, V = jacobi_eigen_symmetric(sym)
    orig_min = min(eigvals) if eigvals else 0.0
    clipped = [max(ev, min_eigenvalue) for ev in eigvals]
    repaired = reconstruct_from_eigen(clipped, V)
    diff = [[repaired[i][j] - sym[i][j] for j in range(len(sym))] for i in range(len(sym))]
    delta = frobenius_norm(diff)
    return repaired, orig_min, delta, clipped, V


def cov_to_vol_and_corr(cov: Sequence[Sequence[float]], eps: float = 1e-12) -> Tuple[Vector, Matrix]:
    """Extract standard deviations and correlation matrix from covariance."""
    n = len(cov)
    vols = [math.sqrt(max(cov[i][i], 0.0)) for i in range(n)]
    corr = [[0.0] * n for _ in range(n)]
    for i in range(n):
        corr[i][i] = 1.0
        for j in range(i + 1, n):
            denom = vols[i] * vols[j]
            if denom <= eps:
                rho = 0.0
            else:
                rho = max(-1.0, min(1.0, cov[i][j] / denom))
            corr[i][j] = rho
            corr[j][i] = rho
    return vols, corr


def apply_correlation_stress(
    cov: Sequence[Sequence[float]],
    *,
    blend: float = 0.35,
    stress_floor: float = 0.65,
    min_eigenvalue: float = 1e-7,
) -> Tuple[Matrix, Matrix]:
    """Apply correlation stress overlay (blending off-diagonal correlations upward).

    Models crisis / correlated drawdown regimes where diversification breaks down.
    Returns (stressed_covariance, stressed_correlation).
    """
    n = len(cov)
    if n == 0:
        return [], []
    vols, corr = cov_to_vol_and_corr(cov)
    stressed_corr = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            target = max(corr[i][j], stress_floor)
            rho = (1.0 - blend) * corr[i][j] + blend * target
            rho = max(-0.99, min(0.99, rho))
            stressed_corr[i][j] = rho
            stressed_corr[j][i] = rho
    raw_stressed_cov = [
        [stressed_corr[i][j] * vols[i] * vols[j] for j in range(n)] for i in range(n)
    ]
    repaired_cov, _, _, _, _ = nearest_psd_spectral(
        raw_stressed_cov,
        min_eigenvalue=min_eigenvalue,
    )
    _, final_corr = cov_to_vol_and_corr(repaired_cov)
    return repaired_cov, final_corr
