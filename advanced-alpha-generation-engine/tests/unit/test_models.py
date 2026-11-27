"""Unit tests for statistical and ML predictive baselines, including optional dependency isolation."""

from __future__ import annotations

import numpy as np
import pytest

from alpha_engine.errors import OptionalDependencyError
from alpha_engine.models import (
    EqualRankCompositeBaseline,
    GradientBoostedAlphaBaseline,
    RidgeAlphaBaseline,
    XGBoostAlphaBaseline,
)


def _toy_xy(seed: int = 7) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(300, 4))
    beta = np.array([0.8, -0.4, 0.3, 0.0])
    y = X @ beta + rng.normal(0.0, 0.3, size=300)
    # Inject a few NaNs to verify safe imputation
    X[5, 1] = np.nan
    X[12, 3] = np.nan
    return X, y, ["f1", "f2", "f3", "f4"]


def test_equal_composite_ridge_and_gbdt_baselines_are_deterministic() -> None:
    X, y, names = _toy_xy()

    eq1 = EqualRankCompositeBaseline(feature_signs={"f1": 1.0, "f2": -1.0, "f3": 1.0, "f4": 0.0})
    eq1.fit(X, y, names)
    p1 = eq1.predict(X)
    assert np.all(np.isfinite(p1))
    assert np.corrcoef(p1, y)[0, 1] > 0.7

    r1 = RidgeAlphaBaseline(alpha=10.0).fit(X, y, names)
    r2 = r1.clone_fresh().fit(X, y, names)
    np.testing.assert_allclose(r1.predict(X), r2.predict(X))
    coefs = r1.feature_coefficients()
    assert coefs["f1"] > 0.5 and coefs["f2"] < -0.2

    g1 = GradientBoostedAlphaBaseline(max_iter=25, max_depth=2, seed=42).fit(X, y, names)
    g2 = g1.clone_fresh().fit(X, y, names)
    np.testing.assert_allclose(g1.predict(X), g2.predict(X))
    assert np.corrcoef(g1.predict(X), y)[0, 1] > 0.7
    assert set(g1.feature_coefficients().keys()) == set(names)


def test_xgboost_optional_dependency_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    X, y, names = _toy_xy()

    # If native libomp is present, XGBoost fits deterministically; otherwise it must raise OptionalDependencyError
    try:
        xgb1 = XGBoostAlphaBaseline(n_estimators=20, max_depth=2, seed=42).fit(X, y, names)
        xgb2 = xgb1.clone_fresh().fit(X, y, names)
        np.testing.assert_allclose(xgb1.predict(X), xgb2.predict(X))
        assert set(xgb1.feature_coefficients().keys()) == set(names)
    except OptionalDependencyError as exc:
        assert exc.error_code == "OPTIONAL_DEPENDENCY_ERROR"

    # Explicit monkeypatch guarantees isolation contract is always verified
    def _raise_missing():
        raise OptionalDependencyError("xgboost optional dependency is not installed.")

    monkeypatch.setattr(XGBoostAlphaBaseline, "_import_xgboost", staticmethod(_raise_missing))
    with pytest.raises(OptionalDependencyError):
        XGBoostAlphaBaseline().fit(X, y, names)
