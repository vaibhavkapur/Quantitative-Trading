"""Pure domain signal state machine for mean-reverting spread entry, exit, and stops.

Isolated from I/O, execution, and portfolio accounting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from statarb.cointegration import SpreadModel
from statarb.config import SignalConfig
from statarb.exceptions import UnsupportedInputError


@dataclass(frozen=True)
class SpreadPositionState:
    """State of a single spread position."""

    position: int = 0  # -1 = short spread, 0 = flat, +1 = long spread
    holding_days: int = 0
    stopped_out_direction: int = 0  # prevents immediate re-entry after stop-loss


@dataclass(frozen=True)
class SignalTransition:
    """Deterministic state transition record for one bar."""

    prev_position: int
    new_position: int
    zscore: float
    holding_days: int
    action: str


def step_spread_signal(
    state: SpreadPositionState,
    zscore: float,
    cfg: SignalConfig,
) -> tuple[SpreadPositionState, SignalTransition]:
    """Advance the spread signal state machine by one bar.

    Rules:
    - Entry LONG spread (+1) when z <= -z_entry (and z > -z_stop, not stopped out in +1).
    - Entry SHORT spread (-1) when z >= +z_entry (and z < +z_stop, not stopped out in -1).
    - Exit mean reversion: LONG exits when z >= -z_exit; SHORT exits when z <= +z_exit.
    - Stop-loss: LONG stops when z <= -z_stop; SHORT stops when z >= +z_stop.
    - Time stop: exit when holding_days >= max_holding_days.
    - Stop-out cooldown clears once |z| <= z_exit or z crosses zero to the opposite side.
    """
    prev_pos = state.position

    if not math.isfinite(zscore):
        next_state = SpreadPositionState(position=0, holding_days=0, stopped_out_direction=0)
        action = "EXIT_INVALID_Z" if prev_pos != 0 else "HOLD_FLAT"
        return next_state, SignalTransition(prev_pos, 0, zscore, 0, action)

    if prev_pos == 0:
        stopped_dir = state.stopped_out_direction
        if stopped_dir == 1 and zscore >= -cfg.z_exit:
            stopped_dir = 0
        elif stopped_dir == -1 and zscore <= cfg.z_exit:
            stopped_dir = 0

        if zscore <= -cfg.z_entry and zscore > -cfg.z_stop and stopped_dir != 1:
            next_state = SpreadPositionState(position=1, holding_days=1, stopped_out_direction=0)
            return next_state, SignalTransition(0, 1, zscore, 1, "ENTER_LONG")

        if zscore >= cfg.z_entry and zscore < cfg.z_stop and stopped_dir != -1:
            next_state = SpreadPositionState(position=-1, holding_days=1, stopped_out_direction=0)
            return next_state, SignalTransition(0, -1, zscore, 1, "ENTER_SHORT")

        next_state = SpreadPositionState(
            position=0, holding_days=0, stopped_out_direction=stopped_dir
        )
        return next_state, SignalTransition(0, 0, zscore, 0, "HOLD_FLAT")

    if prev_pos == 1:
        if zscore <= -cfg.z_stop:
            next_state = SpreadPositionState(position=0, holding_days=0, stopped_out_direction=1)
            return next_state, SignalTransition(1, 0, zscore, 0, "EXIT_STOP_LOSS")
        if zscore >= -cfg.z_exit:
            next_state = SpreadPositionState(position=0, holding_days=0, stopped_out_direction=0)
            return next_state, SignalTransition(1, 0, zscore, 0, "EXIT_MEAN_REVERT")
        if state.holding_days >= cfg.max_holding_days:
            next_state = SpreadPositionState(position=0, holding_days=0, stopped_out_direction=0)
            return next_state, SignalTransition(1, 0, zscore, 0, "EXIT_TIME_STOP")
        next_days = state.holding_days + 1
        next_state = SpreadPositionState(
            position=1, holding_days=next_days, stopped_out_direction=0
        )
        return next_state, SignalTransition(1, 1, zscore, next_days, "HOLD_LONG")

    # prev_pos == -1
    if zscore >= cfg.z_stop:
        next_state = SpreadPositionState(position=0, holding_days=0, stopped_out_direction=-1)
        return next_state, SignalTransition(-1, 0, zscore, 0, "EXIT_STOP_LOSS")
    if zscore <= cfg.z_exit:
        next_state = SpreadPositionState(position=0, holding_days=0, stopped_out_direction=0)
        return next_state, SignalTransition(-1, 0, zscore, 0, "EXIT_MEAN_REVERT")
    if state.holding_days >= cfg.max_holding_days:
        next_state = SpreadPositionState(position=0, holding_days=0, stopped_out_direction=0)
        return next_state, SignalTransition(-1, 0, zscore, 0, "EXIT_TIME_STOP")
    next_days = state.holding_days + 1
    next_state = SpreadPositionState(position=-1, holding_days=next_days, stopped_out_direction=0)
    return next_state, SignalTransition(-1, -1, zscore, next_days, "HOLD_SHORT")


def compute_zscore_causal(
    spread_history: list[float],
    train_mean: float,
    train_std: float,
    cfg: SignalConfig,
) -> float:
    """Compute strictly causal z-score using either frozen training stats or trailing window."""
    if not spread_history:
        return 0.0
    current = spread_history[-1]
    if cfg.zscore_mode == "training_frozen":
        if train_std <= 1e-12:
            return 0.0
        return float((current - train_mean) / train_std)
    if cfg.zscore_mode == "causal_rolling":
        window = np.asarray(spread_history[-cfg.zscore_lookback_days :], dtype=np.float64)
        if window.size < 5:
            denom = train_std if train_std > 1e-12 else 1.0
            return float((current - train_mean) / denom)
        mu = float(np.mean(window))
        sd = float(np.std(window, ddof=1))
        if sd <= 1e-12:
            return 0.0
        return float((current - mu) / sd)
    raise UnsupportedInputError(f"Unsupported zscore_mode: '{cfg.zscore_mode}'")


def compute_target_leg_weights(
    model: SpreadModel,
    position_dir: int,
    spread_gross_weight: float,
) -> dict[str, float]:
    """Map spread direction (+1/-1/0) to gross-normalized leg weights preserving hedge ratios."""
    if position_dir == 0 or spread_gross_weight <= 0.0:
        return {sym: 0.0 for sym in model.symbols}
    l1_norm = sum(abs(w) for w in model.weights)
    if l1_norm <= 1e-12:
        return {sym: 0.0 for sym in model.symbols}
    return {
        sym: float(position_dir * w * spread_gross_weight / l1_norm)
        for sym, w in zip(model.symbols, model.weights, strict=True)
    }
