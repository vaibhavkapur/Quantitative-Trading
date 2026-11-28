"""Contract, schema validation, and fixture tests."""

from __future__ import annotations

from pathlib import Path
import pytest

from optimal_execution.contracts import (
    ExecutionScenario,
    ImpactParams,
    MarketProfile,
    OrderSide,
    ParentOrder,
    PriceShock,
    ScheduleRequest,
    StrategyType,
    UnsupportedScenarioError,
    ValidationError,
)
from optimal_execution.io import load_scenario

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "data" / "fixtures"
CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def test_load_all_valid_fixtures() -> None:
    valid_paths = [
        CONFIG_DIR / "default_scenario.json",
        FIXTURES_DIR / "reference_deterministic.json",
        FIXTURES_DIR / "zero_liquidity.json",
        FIXTURES_DIR / "partial_fill_capacity.json",
        FIXTURES_DIR / "price_shock.json",
    ]
    for path in valid_paths:
        scenario = load_scenario(path)
        scenario.validate()
        assert scenario.order.quantity > 0
        assert len(scenario.market.volume_curve) == scenario.order.horizon_slices


def test_invalid_negative_quantity_fixture_rejected() -> None:
    with pytest.raises(ValidationError, match="quantity must be > 0"):
        load_scenario(FIXTURES_DIR / "invalid_negative_quantity.json")


def test_invalid_missing_fields_fixture_rejected() -> None:
    with pytest.raises(ValidationError, match="Missing required scenario fields"):
        load_scenario(FIXTURES_DIR / "invalid_missing_fields.json")


def test_unsupported_schema_version_rejected() -> None:
    with pytest.raises(UnsupportedScenarioError, match="Unsupported schema_version"):
        load_scenario(FIXTURES_DIR / "invalid_unsupported_schema.json")


def test_missing_file_raises_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        load_scenario(FIXTURES_DIR / "does_not_exist.json")


def test_volume_curve_length_mismatch_rejected() -> None:
    scenario = ExecutionScenario(
        scenario_id="bad_len",
        order=ParentOrder(
            order_id="O1",
            symbol="SYN",
            side=OrderSide.BUY,
            quantity=1000.0,
            arrival_price=50.0,
            horizon_slices=4,
        ),
        market=MarketProfile(
            volume_curve=(1000.0, 1000.0),  # length 2 != 4
            daily_volatility=0.02,
            bid_ask_spread_bps=5.0,
        ),
        impact=ImpactParams(permanent_impact_gamma=1e-6, temporary_impact_eta=2e-6),
    )
    with pytest.raises(ValidationError, match="volume_curve length"):
        scenario.validate()


def test_price_shock_out_of_bounds_rejected() -> None:
    scenario = ExecutionScenario(
        scenario_id="bad_shock",
        order=ParentOrder(
            order_id="O1",
            symbol="SYN",
            side=OrderSide.BUY,
            quantity=1000.0,
            arrival_price=50.0,
            horizon_slices=3,
        ),
        market=MarketProfile(
            volume_curve=(5000.0, 5000.0, 5000.0),
            daily_volatility=0.02,
            bid_ask_spread_bps=5.0,
            price_shocks=(PriceShock(slice_index=3, shock_bps=50.0),),
        ),
        impact=ImpactParams(permanent_impact_gamma=1e-6, temporary_impact_eta=2e-6),
    )
    with pytest.raises(ValidationError, match="out of bounds"):
        scenario.validate()


def test_invalid_schedule_request_parameters() -> None:
    with pytest.raises(ValidationError, match="risk_aversion_lambda"):
        ScheduleRequest(
            strategy=StrategyType.ALMGREN_CHRISS_ANALYTIC,
            risk_aversion_lambda=-1e-6,
        ).validate()

    with pytest.raises(ValidationError, match="target_participation_rate"):
        ScheduleRequest(
            strategy=StrategyType.PARTICIPATION,
            target_participation_rate=1.5,
        ).validate()


def test_roundtrip_serialization() -> None:
    scenario = load_scenario(FIXTURES_DIR / "price_shock.json")
    restored = ExecutionScenario.from_dict(scenario.to_dict())
    assert restored == scenario
