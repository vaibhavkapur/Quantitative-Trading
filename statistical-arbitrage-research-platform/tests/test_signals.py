"""Unit tests for the pure spread signal state machine and causal z-scores."""

from __future__ import annotations

from datetime import date

import pytest

from statarb.cointegration import SpreadModel
from statarb.config import SignalConfig
from statarb.signals import (
    SpreadPositionState,
    compute_target_leg_weights,
    compute_zscore_causal,
    step_spread_signal,
)


def _signal_cfg() -> SignalConfig:
    return SignalConfig(
        z_entry=2.0,
        z_exit=0.5,
        z_stop=4.0,
        max_holding_days=5,
        zscore_mode="training_frozen",
        zscore_lookback_days=20,
    )


def test_long_entry_hold_and_mean_revert_exit() -> None:
    cfg = _signal_cfg()
    st = SpreadPositionState()

    # Neutral -> stay flat
    st, tr = step_spread_signal(st, -1.2, cfg)
    assert tr.action == "HOLD_FLAT" and st.position == 0

    # Cross below -z_entry -> enter long spread
    st, tr = step_spread_signal(st, -2.3, cfg)
    assert tr.action == "ENTER_LONG" and st.position == 1 and st.holding_days == 1

    # Still below -z_exit -> hold
    st, tr = step_spread_signal(st, -1.1, cfg)
    assert tr.action == "HOLD_LONG" and st.position == 1 and st.holding_days == 2

    # Revert above -z_exit -> exit mean reversion
    st, tr = step_spread_signal(st, -0.3, cfg)
    assert tr.action == "EXIT_MEAN_REVERT" and st.position == 0 and st.holding_days == 0


def test_short_stop_loss_and_cooldown() -> None:
    cfg = _signal_cfg()
    st = SpreadPositionState()

    st, tr = step_spread_signal(st, 2.5, cfg)
    assert tr.action == "ENTER_SHORT" and st.position == -1

    # Breach +z_stop -> exit stop loss with cooldown
    st, tr = step_spread_signal(st, 4.2, cfg)
    assert tr.action == "EXIT_STOP_LOSS" and st.position == 0 and st.stopped_out_direction == -1

    # Back to 2.5 without first reverting inside z_exit -> must NOT re-enter immediately
    st, tr = step_spread_signal(st, 2.5, cfg)
    assert tr.action == "HOLD_FLAT" and st.position == 0

    # Revert inside z_exit -> clears cooldown
    st, tr = step_spread_signal(st, 0.2, cfg)
    assert tr.action == "HOLD_FLAT" and st.stopped_out_direction == 0

    # Next excursion >= z_entry -> allowed to re-enter
    st, tr = step_spread_signal(st, 2.2, cfg)
    assert tr.action == "ENTER_SHORT" and st.position == -1


def test_time_stop_exit() -> None:
    cfg = _signal_cfg()
    st = SpreadPositionState()

    st, _ = step_spread_signal(st, -2.2, cfg)  # holding_days = 1
    for expected_day in (2, 3, 4, 5):
        st, tr = step_spread_signal(st, -1.5, cfg)
        assert tr.action == "HOLD_LONG" and st.holding_days == expected_day

    # On next bar holding_days == max_holding_days -> time stop
    st, tr = step_spread_signal(st, -1.5, cfg)
    assert tr.action == "EXIT_TIME_STOP" and st.position == 0


def test_non_finite_zscore_flattens_position() -> None:
    cfg = _signal_cfg()
    st = SpreadPositionState(position=1, holding_days=2)
    st, tr = step_spread_signal(st, float("nan"), cfg)
    assert tr.action == "EXIT_INVALID_Z" and st.position == 0


def test_causal_zscore_invariance_to_future_values() -> None:
    cfg = SignalConfig(
        z_entry=1.5,
        z_exit=0.25,
        z_stop=3.5,
        max_holding_days=20,
        zscore_mode="causal_rolling",
        zscore_lookback_days=10,
    )
    hist = [0.1, -0.2, 0.3, -0.1, 0.4, -0.5, 0.2, 0.0, -0.3, 0.6, 0.8]
    z_at_t = compute_zscore_causal(hist[:10], 0.0, 1.0, cfg)
    # Changing future observations beyond index 9 cannot alter z_at_t
    hist_mutated = hist[:10] + [999.0, -999.0]
    z_at_t_again = compute_zscore_causal(hist_mutated[:10], 0.0, 1.0, cfg)
    assert z_at_t == pytest.approx(z_at_t_again, abs=1e-12)


def test_target_leg_weights_preserve_hedge_ratio_and_gross_cap() -> None:
    model = SpreadModel(
        spread_id="PAIR:A|B",
        kind="pair",
        symbols=("A", "B"),
        weights=(1.0, -1.5),
        intercept=0.0,
        spread_mean=0.0,
        spread_std=1.0,
        use_log_prices=True,
        estimator="ols",
        adf_statistic=-4.0,
        adf_pvalue=0.002,
        adjusted_pvalue=0.004,
        half_life_days=6.0,
        correlation=0.95,
        train_start=date(2023, 1, 2),
        train_end=date(2023, 6, 30),
    )
    w_long = compute_target_leg_weights(model, position_dir=1, spread_gross_weight=0.25)
    assert abs(w_long["A"]) + abs(w_long["B"]) == pytest.approx(0.25, abs=1e-12)
    assert w_long["B"] / w_long["A"] == pytest.approx(-1.5, abs=1e-12)

    w_flat = compute_target_leg_weights(model, position_dir=0, spread_gross_weight=0.25)
    assert w_flat == {"A": 0.0, "B": 0.0}
