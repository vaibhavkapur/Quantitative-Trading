"""Boundary and failure regression tests for exposure, turnover, and volatility cap violations."""

from __future__ import annotations

import numpy as np
import pytest

from cross_asset_factor_engine import (
    ConstraintViolationError,
    optimize_constrained_portfolio,
    verify_portfolio_constraints,
)
from cross_asset_factor_engine.config import PortfolioConfig, PortfolioConstraints


def _sample_cov() -> np.ndarray:
    vols = np.array([0.18, 0.20, 0.10, 0.12, 0.30, 0.16])
    corr = np.array(
        [
            [1.00, 0.65, 0.20, -0.10, 0.25, 0.05],
            [0.65, 1.00, 0.25, -0.15, 0.20, 0.05],
            [0.20, 0.25, 1.00, -0.30, 0.10, 0.15],
            [-0.10, -0.15, -0.30, 1.00, 0.05, 0.10],
            [0.25, 0.20, 0.10, 0.05, 1.00, 0.30],
            [0.05, 0.05, 0.15, 0.10, 0.30, 1.00],
        ]
    )
    return np.outer(vols, vols) * corr


def test_optimizer_enforces_tight_exposure_vol_and_turnover_caps() -> None:
    cov = _sample_cov()
    classes = ["equity", "equity", "fx", "fx", "commodity", "commodity"]
    # Extreme signals that would violate every cap if unconstrained
    signals = np.array([3.0, 2.8, 2.5, -0.2, 2.9, 2.4])
    prev_w = np.array([0.05, 0.05, 0.0, 0.0, 0.05, 0.0])

    cfg = PortfolioConfig(
        target_volatility=0.10,
        volatility_cap=0.10,
        constraints=PortfolioConstraints(
            max_asset_weight=0.20,
            max_gross_exposure=0.90,
            min_net_exposure=-0.20,
            max_net_exposure=0.20,
            max_turnover_per_rebalance=0.30,
            max_class_gross_exposure={"equity": 0.35, "fx": 0.35, "commodity": 0.35},
        ),
        enforce_strict_constraints=True,
        constraint_tolerance=1e-5,
    )

    res = optimize_constrained_portfolio(signals, cov, prev_w, classes, cfg)
    assert np.max(np.abs(res.weights)) <= cfg.constraints.max_asset_weight + 1e-5
    assert res.gross_exposure <= cfg.constraints.max_gross_exposure + 1e-5
    assert cfg.constraints.min_net_exposure - 1e-5 <= res.net_exposure <= cfg.constraints.max_net_exposure + 1e-5
    assert res.ex_ante_vol <= cfg.volatility_cap + 1e-5
    assert res.one_way_turnover <= cfg.constraints.max_turnover_per_rebalance + 1e-5
    for cls_name, cap in cfg.constraints.max_class_gross_exposure.items():
        assert res.class_gross_exposure[cls_name] <= cap + 1e-5


def test_verify_portfolio_constraints_catches_every_cap_violation() -> None:
    cov = _sample_cov()
    classes = ["equity", "equity", "fx", "fx", "commodity", "commodity"]
    prev_w = np.zeros(6)
    cons = PortfolioConstraints(
        max_asset_weight=0.25,
        max_gross_exposure=1.00,
        min_net_exposure=-0.20,
        max_net_exposure=0.20,
        max_turnover_per_rebalance=0.40,
        max_class_gross_exposure={"equity": 0.40, "fx": 0.40, "commodity": 0.40},
    )

    # 1. Single-asset weight cap violation
    with pytest.raises(ConstraintViolationError) as e1:
        verify_portfolio_constraints(
            np.array([0.35, -0.35, 0.0, 0.0, 0.0, 0.0]), prev_w, cov, classes, cons, volatility_cap=0.50
        )
    assert any(v["constraint"] == "max_asset_weight" for v in e1.value.details["violations"])

    # 2. Gross exposure violation
    with pytest.raises(ConstraintViolationError) as e2:
        verify_portfolio_constraints(
            np.array([0.24, -0.24, 0.24, -0.24, 0.24, -0.24]), prev_w, cov, classes, cons, volatility_cap=0.50
        )
    assert any(v["constraint"] == "max_gross_exposure" for v in e2.value.details["violations"])

    # 3. Net exposure upper violation
    with pytest.raises(ConstraintViolationError) as e3:
        verify_portfolio_constraints(
            np.array([0.15, 0.15, 0.10, 0.0, 0.0, 0.0]), prev_w, cov, classes, cons, volatility_cap=0.50
        )
    assert any(v["constraint"] == "max_net_exposure" for v in e3.value.details["violations"])

    # 4. Asset-class gross cap violation
    with pytest.raises(ConstraintViolationError) as e4:
        verify_portfolio_constraints(
            np.array([0.25, -0.25, 0.0, 0.0, 0.0, 0.0]), prev_w, cov, classes, cons, volatility_cap=0.50
        )
    assert any(v["constraint"] == "max_class_gross_exposure:equity" for v in e4.value.details["violations"])

    # 5. Volatility cap violation
    with pytest.raises(ConstraintViolationError) as e5:
        verify_portfolio_constraints(
            np.array([0.20, 0.20, 0.0, 0.0, 0.20, -0.40]), prev_w, cov, classes, cons, volatility_cap=0.05
        )
    assert any(v["constraint"] == "volatility_cap" for v in e5.value.details["violations"])

    # 6. Turnover cap violation
    with pytest.raises(ConstraintViolationError) as e6:
        verify_portfolio_constraints(
            np.array([0.20, -0.20, 0.20, -0.20, 0.15, -0.15]),
            prev_w,
            cov,
            classes,
            cons,
            volatility_cap=0.50,
        )
    assert any(v["constraint"] == "max_turnover_per_rebalance" for v in e6.value.details["violations"])


def test_infeasible_gross_exposure_under_turnover_cap_raises_explicit_error() -> None:
    """When prev_weights gross exposure cannot be unwound below max_gross_exposure within turnover cap."""
    cov = _sample_cov()
    classes = ["equity", "equity", "fx", "fx", "commodity", "commodity"]
    signals = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    # prev gross = 2.40, max_turnover = 0.10 -> max L1 step = 0.20 -> min reachable gross = 2.20 > 1.00!
    prev_w = np.array([0.40, -0.40, 0.40, -0.40, 0.40, -0.40])

    cfg = PortfolioConfig(
        target_volatility=0.10,
        volatility_cap=0.20,
        constraints=PortfolioConstraints(
            max_asset_weight=0.50,
            max_gross_exposure=1.00,
            min_net_exposure=-0.50,
            max_net_exposure=0.50,
            max_turnover_per_rebalance=0.10,
            max_class_gross_exposure={"equity": 0.90, "fx": 0.90, "commodity": 0.90},
        ),
        enforce_strict_constraints=True,
    )
    with pytest.raises(ConstraintViolationError) as exc_info:
        optimize_constrained_portfolio(signals, cov, prev_w, classes, cfg)
    assert "max_gross_exposure" in str(exc_info.value)


def test_infeasible_volatility_cap_under_turnover_cap_raises_explicit_error() -> None:
    """When prev_weights ex-ante vol cannot be reduced below volatility_cap within turnover cap."""
    cov = _sample_cov()
    classes = ["equity", "equity", "fx", "fx", "commodity", "commodity"]
    signals = np.zeros(6)
    # High-vol concentrated position (vol ~ 0.18), tiny turnover cap (0.01) -> cannot reach vol_cap=0.02!
    prev_w = np.array([0.30, 0.30, 0.0, 0.0, 0.30, 0.0])

    cfg = PortfolioConfig(
        target_volatility=0.015,
        volatility_cap=0.02,
        constraints=PortfolioConstraints(
            max_asset_weight=0.50,
            max_gross_exposure=1.50,
            min_net_exposure=-1.00,
            max_net_exposure=1.00,
            max_turnover_per_rebalance=0.01,
            max_class_gross_exposure={"equity": 0.80, "fx": 0.80, "commodity": 0.80},
        ),
        enforce_strict_constraints=True,
    )
    with pytest.raises(ConstraintViolationError) as exc_info:
        optimize_constrained_portfolio(signals, cov, prev_w, classes, cfg)
    assert any(v["constraint"] == "volatility_cap" for v in exc_info.value.details["violations"])


def test_conflicting_net_and_gross_exposure_band_raises_explicit_error() -> None:
    cov = _sample_cov()
    classes = ["equity", "equity", "fx", "fx", "commodity", "commodity"]
    cfg = PortfolioConfig(
        target_volatility=0.10,
        volatility_cap=0.20,
        constraints=PortfolioConstraints(
            max_asset_weight=0.30,
            max_gross_exposure=0.50,
            min_net_exposure=0.80,
            max_net_exposure=1.00,
            max_turnover_per_rebalance=0.50,
            max_class_gross_exposure={"equity": 0.50, "fx": 0.50, "commodity": 0.50},
        ),
        enforce_strict_constraints=True,
    )
    with pytest.raises(ConstraintViolationError):
        optimize_constrained_portfolio(np.ones(6), cov, np.zeros(6), classes, cfg)
