"""Optimal Execution Research Engine.

Research prototype for comparing execution schedules (TWAP, VWAP, POV/participation,
and analytic/numeric Almgren-Chriss) under realistic spread, permanent/temporary
impact, capacity, liquidity, and timing-risk assumptions.

All simulations are offline research models; simulated fills are not real trading
results.
"""

from optimal_execution.contracts import (
    SCHEMA_VERSION,
    CapacityPolicy,
    CostMetrics,
    ExecutionScenario,
    FillSlice,
    ImpactParams,
    MarketProfile,
    MonteCarloSummary,
    ParentOrder,
    PriceShock,
    RunMetadata,
    Schedule,
    ScheduleError,
    ScheduleRequest,
    SensitivityPoint,
    SimulationDiagnostics,
    SimulationResult,
    StrategyType,
    UnsupportedScenarioError,
    ValidationError,
)

__version__ = "0.1.0"

__all__ = [
    "SCHEMA_VERSION",
    "__version__",
    "CapacityPolicy",
    "CostMetrics",
    "ExecutionScenario",
    "FillSlice",
    "ImpactParams",
    "MarketProfile",
    "MonteCarloSummary",
    "ParentOrder",
    "PriceShock",
    "RunMetadata",
    "Schedule",
    "ScheduleError",
    "ScheduleRequest",
    "SensitivityPoint",
    "SimulationDiagnostics",
    "SimulationResult",
    "StrategyType",
    "UnsupportedScenarioError",
    "ValidationError",
]
