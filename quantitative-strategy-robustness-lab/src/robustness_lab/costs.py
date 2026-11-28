"""Transaction cost, financing drag, and benchmark comparison calculations.

Pure domain module (no file I/O).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .models import CostModel, ReturnSeriesSpec


@dataclass(frozen=True)
class CostAdjustedSeries:
    """Result of applying transaction and borrow costs to a gross return series."""

    series_id: str
    timestamps: tuple[str, ...]
    gross_returns: np.ndarray
    cost_drag: np.ndarray
    net_returns: np.ndarray
    benchmark_returns: Optional[np.ndarray]
    excess_net_returns: Optional[np.ndarray]
    total_gross_compounded: float
    total_net_compounded: float
    mean_period_cost_bps: float
    annualized_cost_drag: float


def apply_transaction_costs(
    series: ReturnSeriesSpec,
    cost_model: CostModel,
) -> CostAdjustedSeries:
    """Compute net returns after commission, slippage, borrow, and fixed costs.

    Formula per period t:
      cost_t = turnover_t * (commission_bps + slippage_bps) * 1e-4
               + short_exposure_t * (borrow_annual_bps * 1e-4 / annualization_factor)
               + fixed_cost_per_period
      net_r_t = max(-1.0, gross_r_t - cost_t)
    """
    gross = series.gross_array()
    turnover = series.turnover_array()
    short_exp = series.short_exposure_array()

    trade_cost_rate = (cost_model.commission_bps + cost_model.slippage_bps) * 1e-4
    borrow_period_rate = (cost_model.borrow_annual_bps * 1e-4) / float(cost_model.annualization_factor)

    cost_drag = turnover * trade_cost_rate + short_exp * borrow_period_rate + cost_model.fixed_cost_per_period
    # Floor at -1.0 for simple return coherence (total equity wipeout)
    net = np.maximum(-1.0, gross - cost_drag)

    benchmark = series.benchmark_array()
    excess = net - benchmark if benchmark is not None else None

    total_gross = float(np.prod(1.0 + gross) - 1.0)
    total_net = float(np.prod(1.0 + net) - 1.0)
    mean_cost = float(np.mean(cost_drag))

    return CostAdjustedSeries(
        series_id=series.series_id,
        timestamps=series.timestamps,
        gross_returns=gross,
        cost_drag=cost_drag,
        net_returns=net,
        benchmark_returns=benchmark,
        excess_net_returns=excess,
        total_gross_compounded=total_gross,
        total_net_compounded=total_net,
        mean_period_cost_bps=mean_cost * 1e4,
        annualized_cost_drag=mean_cost * float(series.frequency.periods_per_year),
    )
