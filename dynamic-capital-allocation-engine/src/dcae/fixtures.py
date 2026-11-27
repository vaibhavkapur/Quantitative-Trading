"""Deterministic synthetic fixtures for tests, demos, walk-forward, and scenarios.

All generated data is explicitly tagged as synthetic (`synthetic_internal`)
with fixed random seeds so every test and benchmark is bit-for-bit reproducible.
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple

from dcae.domain.models import (
    FundSnapshot,
    PositionSnapshot,
    Provenance,
    ReturnObservation,
    StrategyExposure,
    StrategySnapshot,
    format_utc,
)
from dcae.evaluation.walkforward import WalkForwardStrategySpec

BASE_START_UTC = datetime(2025, 1, 2, 21, 0, 0, tzinfo=timezone.utc)


def _business_dates(n_days: int, start: datetime = BASE_START_UTC) -> List[datetime]:
    dates: List[datetime] = []
    cur = start
    while len(dates) < n_days:
        if cur.weekday() < 5:
            dates.append(cur)
        cur += timedelta(days=1)
    return dates


def generate_walkforward_strategies(
    *,
    seed: int = 20260409,
    n_days: int = 180,
    include_stress_regime: bool = True,
) -> Tuple[List[WalkForwardStrategySpec], Provenance]:
    """Generate 5 multi-strategy synthetic return series with a crisis drawdown window.

    Days 90..115 inject a correlated drawdown shock where cross-strategy factor
    correlations spike sharply so stress-aware allocators can be tested realistically.
    """
    rng = random.Random(seed)
    dates = _business_dates(n_days)

    specs_meta = [
        (
            "strat_eq_statarb_us",
            "US Equity StatArb",
            StrategyExposure(
                strategy_group="equity_statarb",
                asset_class="equities",
                gross_exposure=1.0,
                net_exposure=0.05,
                market_beta=0.10,
                adv_usd=45_000_000.0,
                max_capacity_usd=15_000_000.0,
                transaction_cost_bps=8.0,
                liquidity_days_to_exit=1.0,
            ),
            0.11 / 252.0,  # daily drift
            0.09 / math.sqrt(252.0),  # daily vol
            0.35,  # common factor loading
            0.25,  # risk budget
        ),
        (
            "strat_eq_momentum",
            "Global Equity Cross-Sectional Momentum",
            StrategyExposure(
                strategy_group="equity_statarb",
                asset_class="equities",
                gross_exposure=1.0,
                net_exposure=0.25,
                market_beta=0.35,
                adv_usd=60_000_000.0,
                max_capacity_usd=18_000_000.0,
                transaction_cost_bps=10.0,
                liquidity_days_to_exit=1.0,
            ),
            0.13 / 252.0,
            0.14 / math.sqrt(252.0),
            0.60,
            0.20,
        ),
        (
            "strat_macro_cta",
            "Systematic Macro Trend CTA",
            StrategyExposure(
                strategy_group="macro_trend",
                asset_class="futures_macro",
                gross_exposure=1.0,
                net_exposure=0.15,
                market_beta=-0.05,
                adv_usd=120_000_000.0,
                max_capacity_usd=25_000_000.0,
                transaction_cost_bps=5.0,
                liquidity_days_to_exit=0.5,
            ),
            0.09 / 252.0,
            0.12 / math.sqrt(252.0),
            -0.15,
            0.25,
        ),
        (
            "strat_vol_carry",
            "Index Volatility Carry",
            StrategyExposure(
                strategy_group="vol_carry",
                asset_class="volatility",
                gross_exposure=1.0,
                net_exposure=0.40,
                market_beta=0.45,
                adv_usd=35_000_000.0,
                max_capacity_usd=8_000_000.0,
                transaction_cost_bps=15.0,
                liquidity_days_to_exit=2.0,
            ),
            0.14 / 252.0,
            0.16 / math.sqrt(252.0),
            0.70,
            0.12,
        ),
        (
            "strat_fi_rv",
            "Rates Relative Value",
            StrategyExposure(
                strategy_group="fixed_income_rv",
                asset_class="fixed_income",
                gross_exposure=1.0,
                net_exposure=0.05,
                market_beta=0.05,
                adv_usd=90_000_000.0,
                max_capacity_usd=20_000_000.0,
                transaction_cost_bps=6.0,
                liquidity_days_to_exit=1.0,
            ),
            0.075 / 252.0,
            0.06 / math.sqrt(252.0),
            0.20,
            0.18,
        ),
    ]

    # Common market factor + crisis shock
    common_shocks = [rng.gauss(0.0, 1.0) for _ in range(n_days)]
    strategies: List[WalkForwardStrategySpec] = []

    for sid, name, exp, drift, vol, loading, rb in specs_meta:
        nav = 100.0
        obs_list: List[ReturnObservation] = []
        for idx, dt in enumerate(dates):
            idio = rng.gauss(0.0, 1.0)
            shock = common_shocks[idx]
            if include_stress_regime and 90 <= idx <= 110:
                # Correlated crisis drawdown: common negative drift & higher correlation
                eff_loading = 0.85 if loading >= 0 else 0.30
                eff_vol = vol * 1.8
                eff_drift = -0.0045 if loading >= 0 else -0.001
                r = eff_drift + eff_vol * (
                    eff_loading * (shock - 0.55) + math.sqrt(max(0.0, 1.0 - eff_loading ** 2)) * idio
                )
            else:
                r = drift + vol * (
                    loading * shock + math.sqrt(max(0.0, 1.0 - loading ** 2)) * idio
                )
            r = max(-0.25, min(0.25, r))
            nav *= 1.0 + r
            ev_utc = format_utc(dt)
            av_utc = format_utc(dt + timedelta(hours=1))
            obs_list.append(
                ReturnObservation(
                    event_at_utc=ev_utc,
                    available_at_utc=av_utc,
                    nav_usd=round(nav, 6),
                    period_return=round(r, 8),
                )
            )
        strategies.append(
            WalkForwardStrategySpec(
                strategy_id=sid,
                name=name,
                exposure=exp,
                returns=tuple(obs_list),
                eligible=True,
                risk_budget=rb,
            )
        )

    prov = Provenance(
        source="dcae.fixtures.generate_walkforward_strategies",
        entitlement="synthetic_internal",
        generator_version="1.0.0",
        seed=seed,
        notes=(
            "Synthetic 5-strategy daily returns with correlated drawdown regime "
            "(days 90-110). Not real market data; for deterministic testing only."
        ),
    )
    return strategies, prov


def build_baseline_snapshot(
    *,
    seed: int = 20260409,
    n_days: int = 60,
    total_nav_usd: float = 10_000_000.0,
) -> FundSnapshot:
    """Build a valid, fresh baseline FundSnapshot with 5 strategies and balanced positions."""
    wf_strats, prov = generate_walkforward_strategies(
        seed=seed,
        n_days=n_days,
        include_stress_regime=False,
    )
    as_of_dt = datetime.fromisoformat(
        wf_strats[0].returns[-1].available_at_utc.replace("Z", "+00:00")
    ) + timedelta(hours=1)
    as_of_utc = format_utc(as_of_dt)
    pos_updated = format_utc(as_of_dt - timedelta(hours=1))

    initial_weights = {
        "strat_eq_statarb_us": 0.22,
        "strat_eq_momentum": 0.18,
        "strat_macro_cta": 0.20,
        "strat_vol_carry": 0.12,
        "strat_fi_rv": 0.18,
    }
    strat_notional = {sid: w * total_nav_usd for sid, w in initial_weights.items()}
    cash_usd = total_nav_usd - sum(strat_notional.values())

    positions = PositionSnapshot(
        as_of_utc=as_of_utc,
        updated_at_utc=pos_updated,
        currency="USD",
        total_nav_usd=total_nav_usd,
        cash_usd=cash_usd,
        strategy_notional_usd=strat_notional,
    )

    strat_snaps = tuple(
        StrategySnapshot(
            strategy_id=s.strategy_id,
            name=s.name,
            currency="USD",
            eligible=True,
            eligibility_reason="ok",
            exposure=s.exposure,
            returns=s.returns,
            risk_budget=s.risk_budget,
            current_drawdown=-0.01,
        )
        for s in wf_strats
    )

    return FundSnapshot(
        snapshot_id="snap_baseline_001",
        as_of_utc=as_of_utc,
        published_at_utc=as_of_utc,
        provenance=prov,
        positions=positions,
        strategies=strat_snaps,
    )


def build_scenario_fixtures() -> Dict[str, FundSnapshot]:
    """Build explicit fixtures for all 7 required validation scenarios in plan.md."""
    base = build_baseline_snapshot(seed=20260409, n_days=60)
    scenarios: Dict[str, FundSnapshot] = {}

    # 1. Non-PSD covariance (injected negative eigenvalue via external covariance)
    sids = tuple(s.strategy_id for s in base.strategies)
    non_psd = [
        [0.010, 0.012, 0.012, 0.002, 0.001],
        [0.012, 0.010, -0.012, 0.002, 0.001],
        [0.012, -0.012, 0.010, 0.002, 0.001],
        [0.002, 0.002, 0.002, 0.020, 0.003],
        [0.001, 0.001, 0.001, 0.003, 0.005],
    ]
    scenarios["non_psd_covariance"] = FundSnapshot(
        snapshot_id="snap_scen_non_psd",
        as_of_utc=base.as_of_utc,
        published_at_utc=base.published_at_utc,
        provenance=Provenance(
            source="dcae.fixtures.build_scenario_fixtures",
            entitlement="synthetic_internal",
            seed=20260409,
            notes="Injected non-PSD covariance with negative eigenvalue",
        ),
        positions=base.positions,
        strategies=base.strategies,
        external_covariance=non_psd,
        external_covariance_ids=sids,
    )

    # 2. Zero volatility (one strategy has flat constant returns = 0.0)
    zero_vol_strats: List[StrategySnapshot] = []
    for i, s in enumerate(base.strategies):
        if i == 3:  # strat_vol_carry set to constant zero returns
            flat_rets = tuple(
                ReturnObservation(
                    event_at_utc=r.event_at_utc,
                    available_at_utc=r.available_at_utc,
                    nav_usd=100.0,
                    period_return=0.0,
                )
                for r in s.returns
            )
            zero_vol_strats.append(
                StrategySnapshot(
                    strategy_id=s.strategy_id,
                    name=s.name,
                    currency=s.currency,
                    eligible=s.eligible,
                    eligibility_reason=s.eligibility_reason,
                    exposure=s.exposure,
                    returns=flat_rets,
                    risk_budget=s.risk_budget,
                    current_drawdown=0.0,
                )
            )
        else:
            zero_vol_strats.append(s)
    scenarios["zero_volatility"] = FundSnapshot(
        snapshot_id="snap_scen_zero_vol",
        as_of_utc=base.as_of_utc,
        published_at_utc=base.published_at_utc,
        provenance=Provenance(
            source="dcae.fixtures.build_scenario_fixtures",
            entitlement="synthetic_internal",
            seed=20260409,
            notes="Injected zero-volatility constant return strategy",
        ),
        positions=base.positions,
        strategies=tuple(zero_vol_strats),
    )

    # 3. Insufficient history (one strategy has only 5 observations < 20 min)
    short_strats: List[StrategySnapshot] = []
    for i, s in enumerate(base.strategies):
        if i == 1:
            short_strats.append(
                StrategySnapshot(
                    strategy_id=s.strategy_id,
                    name=s.name,
                    currency=s.currency,
                    eligible=s.eligible,
                    eligibility_reason=s.eligibility_reason,
                    exposure=s.exposure,
                    returns=s.returns[-5:],
                    risk_budget=s.risk_budget,
                    current_drawdown=s.current_drawdown,
                )
            )
        else:
            short_strats.append(s)
    scenarios["insufficient_history"] = FundSnapshot(
        snapshot_id="snap_scen_insufficient_history",
        as_of_utc=base.as_of_utc,
        published_at_utc=base.published_at_utc,
        provenance=Provenance(
            source="dcae.fixtures.build_scenario_fixtures",
            entitlement="synthetic_internal",
            seed=20260409,
            notes="Injected strategy with only 5 observations (< 20 minimum)",
        ),
        positions=base.positions,
        strategies=tuple(short_strats),
    )

    # 4. Infeasible caps (sum of strategy max_weight = 0.25 < min invested 0.50)
    tight_strats = tuple(
        StrategySnapshot(
            strategy_id=s.strategy_id,
            name=s.name,
            currency=s.currency,
            eligible=s.eligible,
            eligibility_reason=s.eligibility_reason,
            exposure=s.exposure,
            returns=s.returns,
            min_weight=0.0,
            max_weight=0.05,  # 5 * 0.05 = 0.25 < 1 - cash_reserve_max (0.50)
            risk_budget=s.risk_budget,
            current_drawdown=s.current_drawdown,
        )
        for s in base.strategies
    )
    scenarios["infeasible_caps"] = FundSnapshot(
        snapshot_id="snap_scen_infeasible_caps",
        as_of_utc=base.as_of_utc,
        published_at_utc=base.published_at_utc,
        provenance=Provenance(
            source="dcae.fixtures.build_scenario_fixtures",
            entitlement="synthetic_internal",
            seed=20260409,
            notes="Injected infeasible strategy caps summing to 25% vs 50% min invested",
        ),
        positions=base.positions,
        strategies=tight_strats,
    )

    # 5. Stale positions (position snapshot 72 hours old > 24h limit)
    as_of_dt = datetime.fromisoformat(base.as_of_utc.replace("Z", "+00:00"))
    stale_ts = format_utc(as_of_dt - timedelta(hours=72))
    stale_pos = PositionSnapshot(
        as_of_utc=stale_ts,
        updated_at_utc=stale_ts,
        currency="USD",
        total_nav_usd=base.positions.total_nav_usd,
        cash_usd=base.positions.cash_usd,
        strategy_notional_usd=dict(base.positions.strategy_notional_usd),
    )
    scenarios["stale_positions"] = FundSnapshot(
        snapshot_id="snap_scen_stale_positions",
        as_of_utc=base.as_of_utc,
        published_at_utc=base.published_at_utc,
        provenance=Provenance(
            source="dcae.fixtures.build_scenario_fixtures",
            entitlement="synthetic_internal",
            seed=20260409,
            notes="Injected stale positions (72h old vs 24h limit)",
        ),
        positions=stale_pos,
        strategies=base.strategies,
    )

    # 6. Turnover limit (highly skewed initial portfolio wanting large rebalance)
    skewed_pos = PositionSnapshot(
        as_of_utc=base.positions.as_of_utc,
        updated_at_utc=base.positions.updated_at_utc,
        currency="USD",
        total_nav_usd=10_000_000.0,
        cash_usd=1_000_000.0,
        strategy_notional_usd={
            "strat_eq_statarb_us": 500_000.0,
            "strat_eq_momentum": 3_400_000.0,
            "strat_macro_cta": 600_000.0,
            "strat_vol_carry": 3_500_000.0,
            "strat_fi_rv": 1_000_000.0,
        },
    )
    scenarios["turnover_limit"] = FundSnapshot(
        snapshot_id="snap_scen_turnover_limit",
        as_of_utc=base.as_of_utc,
        published_at_utc=base.published_at_utc,
        provenance=Provenance(
            source="dcae.fixtures.build_scenario_fixtures",
            entitlement="synthetic_internal",
            seed=20260409,
            notes="Skewed starting weights to verify binding one-way turnover constraint",
        ),
        positions=skewed_pos,
        strategies=base.strategies,
    )

    # 7. Correlated drawdowns (crisis window where 4/5 strategies are in >= 7% drawdown)
    dd_wf, dd_prov = generate_walkforward_strategies(
        seed=20260409,
        n_days=111,
        include_stress_regime=True,
    )
    dd_as_of = datetime.fromisoformat(
        dd_wf[0].returns[-1].available_at_utc.replace("Z", "+00:00")
    ) + timedelta(hours=1)
    dd_as_of_utc = format_utc(dd_as_of)
    dd_pos = PositionSnapshot(
        as_of_utc=dd_as_of_utc,
        updated_at_utc=format_utc(dd_as_of - timedelta(hours=1)),
        currency="USD",
        total_nav_usd=10_000_000.0,
        cash_usd=1_000_000.0,
        strategy_notional_usd={
            "strat_eq_statarb_us": 2_200_000.0,
            "strat_eq_momentum": 1_800_000.0,
            "strat_macro_cta": 2_000_000.0,
            "strat_vol_carry": 1_200_000.0,
            "strat_fi_rv": 1_800_000.0,
        },
    )
    dd_strats = tuple(
        StrategySnapshot(
            strategy_id=s.strategy_id,
            name=s.name,
            currency="USD",
            eligible=True,
            eligibility_reason="ok",
            exposure=s.exposure,
            returns=s.returns[-60:],
            risk_budget=s.risk_budget,
            current_drawdown=-0.085 if s.strategy_id != "strat_macro_cta" else -0.02,
        )
        for s in dd_wf
    )
    scenarios["correlated_drawdowns"] = FundSnapshot(
        snapshot_id="snap_scen_correlated_drawdowns",
        as_of_utc=dd_as_of_utc,
        published_at_utc=dd_as_of_utc,
        provenance=dd_prov,
        positions=dd_pos,
        strategies=dd_strats,
    )

    return scenarios
