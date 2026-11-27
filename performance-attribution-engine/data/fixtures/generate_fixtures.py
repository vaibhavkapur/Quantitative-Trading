#!/usr/bin/env python3
"""Deterministic synthetic fixture generator for Performance Attribution Engine (seed=42).

Generates self-reconciling synthetic datasets with explicit provenance, UTC timestamps,
FX rates, trades with TCA benchmarks, external cash flows, benchmark sector weights/returns,
and risk-factor returns. Writes data/fixtures/manifest.json with SHA-256 digests.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple


FIXTURES_DIR = Path(__file__).resolve().parent


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_multi_strategy_fixture(
    *,
    seed: int = 42,
    apply_fee_correction: bool = False,
) -> Dict[str, Any]:
    """Build a 10-day multi-strategy, multi-currency portfolio with exact accounting bridge."""
    rng = random.Random(seed)
    dates = [
        "2026-01-05",
        "2026-01-06",
        "2026-01-07",
        "2026-01-08",
        "2026-01-09",
        "2026-01-12",
        "2026-01-13",
        "2026-01-14",
        "2026-01-15",
        "2026-01-16",
    ]
    prior_date = "2026-01-02"

    # Instruments across 3 strategies, 4 sectors, and 3 currencies (USD, EUR, GBP)
    instruments = [
        {"strategy": "STRAT_US_ALPHA", "id": "TECH_A", "sector": "TECH", "ccy": "USD", "px0": 150.0, "qty0": 2000.0},
        {"strategy": "STRAT_US_ALPHA", "id": "HLTH_B", "sector": "HEALTHCARE", "ccy": "USD", "px0": 100.0, "qty0": 2000.0},
        {"strategy": "STRAT_EU_VALUE", "id": "FIN_EU", "sector": "FINANCIALS", "ccy": "EUR", "px0": 80.0, "qty0": 2500.0},
        {"strategy": "STRAT_STAT_ARB", "id": "IND_UK", "sector": "INDUSTRIALS", "ccy": "GBP", "px0": 60.0, "qty0": 2000.0},
    ]

    # FX trajectories (base -> USD), including prior_date fixing
    fx_series: Dict[str, Dict[str, float]] = {
        "EUR": {prior_date: 1.1000},
        "GBP": {prior_date: 1.2500},
    }
    fx_rates: List[Dict[str, Any]] = [
        {
            "pair": "EUR/USD",
            "base_currency": "EUR",
            "quote_currency": "USD",
            "trade_date": prior_date,
            "rate": 1.1000,
            "as_of_utc": f"{prior_date}T21:00:00Z",
            "availability_utc": f"{prior_date}T21:05:00Z",
            "source": "SYNTHETIC_FIXING",
        },
        {
            "pair": "GBP/USD",
            "base_currency": "GBP",
            "quote_currency": "USD",
            "trade_date": prior_date,
            "rate": 1.2500,
            "as_of_utc": f"{prior_date}T21:00:00Z",
            "availability_utc": f"{prior_date}T21:05:00Z",
            "source": "SYNTHETIC_FIXING",
        },
    ]

    for d in dates:
        prev_eur = list(fx_series["EUR"].values())[-1]
        prev_gbp = list(fx_series["GBP"].values())[-1]
        eur_r = round(prev_eur * (1.0 + rng.uniform(-0.0045, 0.0050)), 6)
        gbp_r = round(prev_gbp * (1.0 + rng.uniform(-0.0040, 0.0045)), 6)
        fx_series["EUR"][d] = eur_r
        fx_series["GBP"][d] = gbp_r
        for ccy, val in (("EUR", eur_r), ("GBP", gbp_r)):
            fx_rates.append(
                {
                    "pair": f"{ccy}/USD",
                    "base_currency": ccy,
                    "quote_currency": "USD",
                    "trade_date": d,
                    "rate": val,
                    "as_of_utc": f"{d}T21:00:00Z",
                    "availability_utc": f"{d}T21:05:00Z",
                    "source": "SYNTHETIC_FIXING",
                }
            )

    def get_fx(ccy: str, d: str) -> float:
        return 1.0 if ccy == "USD" else fx_series[ccy][d]

    # Initial portfolio state (start cash = $130,000 -> total NAV0 = 300k + 200k + 220k + 150k + 130k = $1,000,000)
    state = {
        inst["id"]: {
            "strategy": inst["strategy"],
            "sector": inst["sector"],
            "ccy": inst["ccy"],
            "px": inst["px0"],
            "qty": inst["qty0"],
        }
        for inst in instruments
    }
    cash_usd = 130000.0
    nav_usd = cash_usd + sum(
        s["qty"] * s["px"] * get_fx(s["ccy"], prior_date) for s in state.values()
    )

    # Planned rebalances (day index -> list of trade specs)
    rebalance_plan: Dict[int, List[Dict[str, Any]]] = {
        2: [
            # Day 3: Trim TECH_A, add FIN_EU
            {"id": "TECH_A", "side": "SELL", "order_qty": 400.0, "qty": 350.0, "comm": 42.0, "exch": 8.0, "tax": 0.0, "venue": "XNAS"},
            {"id": "FIN_EU", "side": "BUY", "order_qty": 500.0, "qty": 500.0, "comm": 35.0, "exch": 7.5, "tax": 12.0, "venue": "XETR"},
        ],
        5: [
            # Day 6: Add IND_UK, trim HLTH_B
            {"id": "IND_UK", "side": "BUY", "order_qty": 600.0, "qty": 500.0, "comm": 28.0, "exch": 6.0, "tax": 15.0, "venue": "XLON"},
            {"id": "HLTH_B", "side": "SELL", "order_qty": 300.0, "qty": 300.0, "comm": 30.0, "exch": 5.0, "tax": 0.0, "venue": "XNYS"},
        ],
    }

    # Planned external cash flows
    cf_plan: Dict[int, Tuple[float, str, str]] = {
        3: (200000.0, "START_OF_DAY", "Institutional investor subscription"),
        7: (-100000.0, "END_OF_DAY", "Scheduled partial investor redemption"),
    }

    nav_snapshots: List[Dict[str, Any]] = []
    positions: List[Dict[str, Any]] = []
    trades: List[Dict[str, Any]] = []
    cash_flows: List[Dict[str, Any]] = []
    fee_adjustments: List[Dict[str, Any]] = []
    benchmark: List[Dict[str, Any]] = []
    factors: List[Dict[str, Any]] = []

    trade_seq = 1

    for idx, d in enumerate(dates):
        prev_d = prior_date if idx == 0 else dates[idx - 1]
        start_nav = nav_usd
        start_cash = cash_usd

        # Generate correlated daily risk factors first so stock & benchmark returns have realistic factor structure
        f_mkt = round(rng.gauss(0.0012, 0.0085), 6)
        f_smb = round(rng.gauss(0.0002, 0.0040), 6)
        f_hml = round(rng.gauss(0.0003, 0.0045), 6)
        f_mom = round(rng.gauss(0.0004, 0.0050), 6)
        rf = 0.00015
        factors.append(
            {
                "trade_date": d,
                "factor_returns": {"MKT": f_mkt, "SMB": f_smb, "HML": f_hml, "MOM": f_mom},
                "risk_free_rate": rf,
                "currency": "USD",
                "availability_utc": f"{d}T21:10:00Z",
            }
        )

        # Cash flow on day idx
        cf_sod = 0.0
        cf_eod = 0.0
        if idx in cf_plan:
            amt, timing, desc = cf_plan[idx]
            cash_flows.append(
                {
                    "cash_flow_id": f"CF-{d.replace('-', '')}-01",
                    "trade_date": d,
                    "portfolio_id": "PORT_SAAT_MULTI",
                    "amount_local": amt,
                    "currency": "USD",
                    "timing": timing,
                    "description": desc,
                    "availability_utc": f"{d}T14:00:00Z",
                }
            )
            if timing == "START_OF_DAY":
                cf_sod += amt
            else:
                cf_eod += amt

        cash_interest = round((start_cash + cf_sod) * rf, 6)

        # Price moves tied to factors + idiosyncratic alpha
        factor_betas = {
            "TECH_A": (1.20, 0.15, -0.35, 0.40, 0.0008),
            "HLTH_B": (0.85, -0.10, 0.10, 0.10, 0.0004),
            "FIN_EU": (1.05, 0.05, 0.55, -0.10, 0.0005),
            "IND_UK": (0.95, 0.25, 0.30, 0.15, 0.0003),
        }

        day_trades_spec = {t["id"]: t for t in rebalance_plan.get(idx, [])}
        day_gross_pnl = cash_interest
        day_total_fees = 0.0

        # Daily fund-level management fee (~1.5% annualized / 252)
        mgmt_fee = round((start_nav + cf_sod) * (0.015 / 252.0), 4)
        fee_id = f"FEE-MGMT-{d.replace('-', '')}"
        fee_adjustments.append(
            {
                "fee_id": fee_id,
                "trade_date": d,
                "portfolio_id": "PORT_SAAT_MULTI",
                "fee_type": "MANAGEMENT_FEE",
                "amount_local": mgmt_fee,
                "currency": "USD",
                "strategy_id": None,
                "instrument_id": None,
                "is_correction": False,
                "supersedes_fee_id": None,
                "correction_reason": None,
                "availability_utc": f"{d}T21:05:00Z",
            }
        )
        day_total_fees += mgmt_fee

        # Optional retroactive fee correction on day 5 (index 4) for v2 revision demo & scenario
        if apply_fee_correction and idx == 4:
            corr_amt = -450.0  # Prime broker commission rebate / over-accrual credit
            fee_adjustments.append(
                {
                    "fee_id": "FEE-CORR-20260109-01",
                    "trade_date": d,
                    "portfolio_id": "PORT_SAAT_MULTI",
                    "fee_type": "FEE_REBATE",
                    "amount_local": corr_amt,
                    "currency": "USD",
                    "strategy_id": "STRAT_US_ALPHA",
                    "instrument_id": "TECH_A",
                    "is_correction": True,
                    "supersedes_fee_id": fee_id,
                    "correction_reason": "Retroactive prime-broker commission rebate and fee restatement",
                    "availability_utc": f"{d}T22:00:00Z",
                }
            )
            day_total_fees += corr_amt

        # Update each position and build exact trades
        cash_delta_trades = 0.0
        for inst_id, st in state.items():
            b_mkt, b_smb, b_hml, b_mom, alpha_i = factor_betas[inst_id]
            loc_ret = (
                rf
                + b_mkt * f_mkt
                + b_smb * f_smb
                + b_hml * f_hml
                + b_mom * f_mom
                + alpha_i
                + rng.gauss(0.0, 0.0020)
            )
            p0 = st["px"]
            p1 = round(p0 * (1.0 + loc_ret), 4)
            q0 = st["qty"]
            fx0 = get_fx(st["ccy"], prev_d)
            fx1 = get_fx(st["ccy"], d)

            dq = 0.0
            trade_pnl = 0.0
            trade_fees_loc = 0.0
            if inst_id in day_trades_spec:
                ts = day_trades_spec[inst_id]
                side = ts["side"]
                qty = ts["qty"]
                dq = qty if side == "BUY" else -qty
                # Realistic execution prices showing positive delay & market impact
                sign = 1.0 if side == "BUY" else -1.0
                dec_px = round(p0 * (1.0 + 0.15 * loc_ret), 4)
                arr_px = round(dec_px * (1.0 + sign * 0.0005), 4)
                exec_px = round(arr_px * (1.0 + sign * 0.0009), 4)
                vwap_px = round((p0 + p1) / 2.0, 4)

                trades.append(
                    {
                        "trade_id": f"TRD-{d.replace('-', '')}-{trade_seq:03d}",
                        "order_id": f"ORD-{d.replace('-', '')}-{trade_seq:03d}",
                        "trade_date": d,
                        "execution_ts_utc": f"{d}T15:30:00Z",
                        "portfolio_id": "PORT_SAAT_MULTI",
                        "strategy_id": st["strategy"],
                        "instrument_id": inst_id,
                        "sector": st["sector"],
                        "side": side,
                        "order_quantity": ts["order_qty"],
                        "quantity": qty,
                        "local_currency": st["ccy"],
                        "decision_price_local": dec_px,
                        "arrival_price_local": arr_px,
                        "execution_price_local": exec_px,
                        "close_price_local": p1,
                        "benchmark_vwap_local": vwap_px,
                        "commission_local": ts["comm"],
                        "exchange_fee_local": ts["exch"],
                        "tax_local": ts["tax"],
                        "venue": ts["venue"],
                        "availability_utc": f"{d}T21:05:00Z",
                    }
                )
                trade_seq += 1
                trade_pnl = dq * (p1 - exec_px) * fx1
                trade_fees_loc = ts["comm"] + ts["exch"] + ts["tax"]
                cash_delta_trades -= dq * exec_px * fx1

            # Small dividend on day 4 for FIN_EU
            income_loc = 125.0 if (idx == 3 and inst_id == "FIN_EU") else 0.0
            pos_fee_loc = 4.0 if inst_id == "IND_UK" else 0.0  # custody/borrow fee

            hold_pnl = q0 * (p1 * fx1 - p0 * fx0)
            pos_gross_pnl = hold_pnl + trade_pnl + income_loc * fx1
            pos_fees_base = (pos_fee_loc + trade_fees_loc) * fx1

            day_gross_pnl += pos_gross_pnl
            day_total_fees += pos_fees_base

            q1 = q0 + dq
            positions.append(
                {
                    "position_id": f"POS-{d.replace('-', '')}-{st['strategy']}-{inst_id}",
                    "trade_date": d,
                    "portfolio_id": "PORT_SAAT_MULTI",
                    "strategy_id": st["strategy"],
                    "instrument_id": inst_id,
                    "sector": st["sector"],
                    "asset_class": "EQUITY",
                    "local_currency": st["ccy"],
                    "start_quantity": q0,
                    "end_quantity": q1,
                    "start_price_local": p0,
                    "end_price_local": p1,
                    "income_local": income_loc,
                    "explicit_fees_local": pos_fee_loc,
                    "start_market_value_base": None,
                    "end_market_value_base": None,
                    "as_of_utc": f"{d}T21:00:00Z",
                    "availability_utc": f"{d}T21:05:00Z",
                }
            )
            st["px"] = p1
            st["qty"] = q1
            cash_delta_trades += income_loc * fx1

        # Update cash & NAV by exact accounting identity
        end_cash = start_cash + cf_sod + cf_eod + cash_interest + cash_delta_trades - day_total_fees
        end_nav = start_nav + cf_sod + cf_eod + day_gross_pnl - day_total_fees

        nav_snapshots.append(
            {
                "nav_id": f"NAV-{d.replace('-', '')}",
                "trade_date": d,
                "portfolio_id": "PORT_SAAT_MULTI",
                "reporting_currency": "USD",
                "start_nav_base": start_nav,
                "end_nav_base": end_nav,
                "start_cash_base": start_cash,
                "end_cash_base": end_cash,
                "cash_interest_base": cash_interest,
                "as_of_utc": f"{d}T21:00:00Z",
                "availability_utc": f"{d}T21:05:00Z",
            }
        )
        cash_usd = end_cash
        nav_usd = end_nav

        # Benchmark with 4 sectors + CASH summing exactly to benchmark total return
        bm_weights = {
            "TECH": 0.30,
            "HEALTHCARE": 0.20,
            "FINANCIALS": 0.25,
            "INDUSTRIALS": 0.20,
            "CASH": 0.05,
        }
        bm_sec_rets = {
            "TECH": round(rf + 1.15 * f_mkt - 0.25 * f_hml + 0.20 * f_mom, 6),
            "HEALTHCARE": round(rf + 0.85 * f_mkt + 0.05 * f_hml, 6),
            "FINANCIALS": round(rf + 1.00 * f_mkt + 0.45 * f_hml, 6),
            "INDUSTRIALS": round(rf + 0.95 * f_mkt + 0.20 * f_smb, 6),
            "CASH": rf,
        }
        bm_tot = sum(bm_weights[s] * bm_sec_rets[s] for s in bm_weights)
        benchmark.append(
            {
                "benchmark_id": "BM_GLOBAL_EQ_USD",
                "trade_date": d,
                "currency": "USD",
                "total_return": bm_tot,
                "sector_weights": bm_weights,
                "sector_returns": bm_sec_rets,
                "as_of_utc": f"{d}T21:00:00Z",
                "availability_utc": f"{d}T21:05:00Z",
            }
        )

    suffix = "V2_FEE_CORRECTION" if apply_fee_correction else "V1_BASELINE"
    return {
        "batch_id": f"BATCH-SAAT-MULTI-2026Q1-{suffix}",
        "portfolio_id": "PORT_SAAT_MULTI",
        "reporting_currency": "USD",
        "as_of_utc": "2026-01-17T00:00:00Z",
        "data_source": "SYNTHETIC_FIXTURE_SEED_42",
        "entitlement_tag": "PERMITTED_SYNTHETIC",
        "benchmark_id": "BM_GLOBAL_EQ_USD",
        "nav_snapshots": nav_snapshots,
        "positions": positions,
        "trades": trades,
        "cash_flows": cash_flows,
        "fee_adjustments": fee_adjustments,
        "fx_rates": fx_rates,
        "benchmark": benchmark,
        "factors": factors,
    }


def build_hand_worked_fixture() -> Dict[str, Any]:
    """Build a 2-day exact rational hand-worked portfolio verified analytically on paper.

    Day 1 (2026-01-05):
      Start NAV = $100,000 ($50,000 US_TECH: 500 @ $100; $40,000 EU_FIN: 400 @ EUR 100, FX=1.00; $10,000 Cash)
      No cash flows.
      Prices move: US_TECH $100 -> $110 (+10%), EU_FIN EUR 100 -> EUR 105 (+5%), FX EUR/USD 1.00 -> 1.10 (+10%).
      US_TECH hold P&L = 500 * 10 = +$5,000 (+5.00% contrib)
      EU_FIN local price P&L = 400 * 5 * 1.00 = +$2,000 (+2.00%)
      EU_FIN FX translation P&L = 400 * 100 * 0.10 = +$4,000 (+4.00%)
      EU_FIN FX interaction P&L = 400 * 5 * 0.10 = +$200 (+0.20%)
      Total EU_FIN gross P&L = $6,200 (+6.20%; exact base ret = 1.05*1.10 - 1 = 15.50% * 40% = 6.20%)
      Cash interest = $0. Management fee = $200 (-0.20%).
      Day 1 Gross P&L = $11,200 -> Gross Return = +11.20%
      Day 1 Net P&L   = $11,000 -> Net Return   = +11.00%
      End NAV Day 1   = $111,000 (MV US=$55,000, MV EU=400*105*1.10=$46,200, Cash=$9,800)

    Day 2 (2026-01-06):
      Start NAV = $111,000; SOD Cash Flow = +$89,000 -> Capital Base B_2 = $200,000!
      Intraday Trade on US_TECH: BUY 500 shares @ exec $112 (decision $110, arrival $111, close $115, comm $100).
      US_TECH hold P&L = 500 * (115 - 110) = +$2,500 (+1.25% of $200k)
      US_TECH trading P&L = +500 * (115 - 112) = +$1,500 (+0.75% of $200k)
      US_TECH gross P&L = +$4,000 (+2.00%); commission = $100 (-0.05%); net = +$3,900 (+1.95%)
      EU_FIN flat at EUR 105, FX flat at 1.10 -> P&L = $0.
      Day 2 Gross P&L = $4,000 -> Gross Return = +2.00%
      Day 2 Net P&L   = $3,900 -> Net Return   = +1.95%
      End NAV Day 2   = $200,000 + $3,900 = $203,900.
      Cumulative Gross = (1.112)*(1.02) - 1 = 0.13424 (+13.424%)
      Cumulative Net   = (1.110)*(1.0195) - 1 = 0.131645 (+13.1645%)
    """
    return {
        "batch_id": "BATCH-HAND-WORKED-01",
        "portfolio_id": "PORT_HAND_WORKED",
        "reporting_currency": "USD",
        "as_of_utc": "2026-01-07T00:00:00Z",
        "data_source": "HAND_WORKED_VERIFICATION_CASE",
        "entitlement_tag": "PERMITTED_SYNTHETIC",
        "benchmark_id": "BM_HAND_2SEC",
        "fx_rates": [
            {
                "pair": "EUR/USD",
                "base_currency": "EUR",
                "quote_currency": "USD",
                "trade_date": "2026-01-02",
                "rate": 1.00,
                "as_of_utc": "2026-01-02T21:00:00Z",
                "availability_utc": "2026-01-02T21:05:00Z",
            },
            {
                "pair": "EUR/USD",
                "base_currency": "EUR",
                "quote_currency": "USD",
                "trade_date": "2026-01-05",
                "rate": 1.10,
                "as_of_utc": "2026-01-05T21:00:00Z",
                "availability_utc": "2026-01-05T21:05:00Z",
            },
            {
                "pair": "EUR/USD",
                "base_currency": "EUR",
                "quote_currency": "USD",
                "trade_date": "2026-01-06",
                "rate": 1.10,
                "as_of_utc": "2026-01-06T21:00:00Z",
                "availability_utc": "2026-01-06T21:05:00Z",
            },
        ],
        "nav_snapshots": [
            {
                "nav_id": "HW-NAV-1",
                "trade_date": "2026-01-05",
                "portfolio_id": "PORT_HAND_WORKED",
                "reporting_currency": "USD",
                "start_nav_base": 100000.0,
                "end_nav_base": 111000.0,
                "start_cash_base": 10000.0,
                "end_cash_base": 9800.0,
                "cash_interest_base": 0.0,
                "as_of_utc": "2026-01-05T21:00:00Z",
                "availability_utc": "2026-01-05T21:05:00Z",
            },
            {
                "nav_id": "HW-NAV-2",
                "trade_date": "2026-01-06",
                "portfolio_id": "PORT_HAND_WORKED",
                "reporting_currency": "USD",
                "start_nav_base": 111000.0,
                "end_nav_base": 203900.0,
                "start_cash_base": 9800.0,
                "end_cash_base": 42700.0,
                "cash_interest_base": 0.0,
                "as_of_utc": "2026-01-06T21:00:00Z",
                "availability_utc": "2026-01-06T21:05:00Z",
            },
        ],
        "positions": [
            {
                "position_id": "HW-POS-1-US",
                "trade_date": "2026-01-05",
                "portfolio_id": "PORT_HAND_WORKED",
                "strategy_id": "STRAT_US",
                "instrument_id": "US_TECH",
                "sector": "TECH",
                "asset_class": "EQUITY",
                "local_currency": "USD",
                "start_quantity": 500.0,
                "end_quantity": 500.0,
                "start_price_local": 100.0,
                "end_price_local": 110.0,
                "as_of_utc": "2026-01-05T21:00:00Z",
                "availability_utc": "2026-01-05T21:05:00Z",
            },
            {
                "position_id": "HW-POS-1-EU",
                "trade_date": "2026-01-05",
                "portfolio_id": "PORT_HAND_WORKED",
                "strategy_id": "STRAT_EU",
                "instrument_id": "EU_FIN",
                "sector": "FINANCIALS",
                "asset_class": "EQUITY",
                "local_currency": "EUR",
                "start_quantity": 400.0,
                "end_quantity": 400.0,
                "start_price_local": 100.0,
                "end_price_local": 105.0,
                "as_of_utc": "2026-01-05T21:00:00Z",
                "availability_utc": "2026-01-05T21:05:00Z",
            },
            {
                "position_id": "HW-POS-2-US",
                "trade_date": "2026-01-06",
                "portfolio_id": "PORT_HAND_WORKED",
                "strategy_id": "STRAT_US",
                "instrument_id": "US_TECH",
                "sector": "TECH",
                "asset_class": "EQUITY",
                "local_currency": "USD",
                "start_quantity": 500.0,
                "end_quantity": 1000.0,
                "start_price_local": 110.0,
                "end_price_local": 115.0,
                "as_of_utc": "2026-01-06T21:00:00Z",
                "availability_utc": "2026-01-06T21:05:00Z",
            },
            {
                "position_id": "HW-POS-2-EU",
                "trade_date": "2026-01-06",
                "portfolio_id": "PORT_HAND_WORKED",
                "strategy_id": "STRAT_EU",
                "instrument_id": "EU_FIN",
                "sector": "FINANCIALS",
                "asset_class": "EQUITY",
                "local_currency": "EUR",
                "start_quantity": 400.0,
                "end_quantity": 400.0,
                "start_price_local": 105.0,
                "end_price_local": 105.0,
                "as_of_utc": "2026-01-06T21:00:00Z",
                "availability_utc": "2026-01-06T21:05:00Z",
            },
        ],
        "trades": [
            {
                "trade_id": "HW-TRD-2-US",
                "order_id": "HW-ORD-2-US",
                "trade_date": "2026-01-06",
                "execution_ts_utc": "2026-01-06T15:00:00Z",
                "portfolio_id": "PORT_HAND_WORKED",
                "strategy_id": "STRAT_US",
                "instrument_id": "US_TECH",
                "sector": "TECH",
                "side": "BUY",
                "order_quantity": 600.0,
                "quantity": 500.0,
                "local_currency": "USD",
                "decision_price_local": 110.0,
                "arrival_price_local": 111.0,
                "execution_price_local": 112.0,
                "close_price_local": 115.0,
                "benchmark_vwap_local": 111.5,
                "commission_local": 100.0,
                "exchange_fee_local": 0.0,
                "tax_local": 0.0,
                "venue": "SIM_XNAS",
                "availability_utc": "2026-01-06T21:05:00Z",
            }
        ],
        "cash_flows": [
            {
                "cash_flow_id": "HW-CF-2",
                "trade_date": "2026-01-06",
                "portfolio_id": "PORT_HAND_WORKED",
                "amount_local": 89000.0,
                "currency": "USD",
                "timing": "START_OF_DAY",
                "description": "SOD cash inflow lifting capital base from $111k to $200k",
                "availability_utc": "2026-01-06T13:30:00Z",
            }
        ],
        "fee_adjustments": [
            {
                "fee_id": "HW-FEE-1",
                "trade_date": "2026-01-05",
                "portfolio_id": "PORT_HAND_WORKED",
                "fee_type": "MANAGEMENT_FEE",
                "amount_local": 200.0,
                "currency": "USD",
                "availability_utc": "2026-01-05T21:05:00Z",
            }
        ],
        "benchmark": [
            {
                "benchmark_id": "BM_HAND_2SEC",
                "trade_date": "2026-01-05",
                "currency": "USD",
                "total_return": 0.072,
                "sector_weights": {"TECH": 0.60, "FINANCIALS": 0.40},
                "sector_returns": {"TECH": 0.08, "FINANCIALS": 0.06},
                "as_of_utc": "2026-01-05T21:00:00Z",
                "availability_utc": "2026-01-05T21:05:00Z",
            },
            {
                "benchmark_id": "BM_HAND_2SEC",
                "trade_date": "2026-01-06",
                "currency": "USD",
                "total_return": 0.018,
                "sector_weights": {"TECH": 0.60, "FINANCIALS": 0.40},
                "sector_returns": {"TECH": 0.03, "FINANCIALS": 0.00},
                "as_of_utc": "2026-01-06T21:00:00Z",
                "availability_utc": "2026-01-06T21:05:00Z",
            },
        ],
        "factors": [],
    }


def main() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    files = {
        "multi_strategy_multi_ccy_baseline.json": (
            build_multi_strategy_fixture(seed=42, apply_fee_correction=False),
            "10-day synthetic multi-strategy, multi-currency (USD/EUR/GBP) portfolio with trades, cash flows, Brinson benchmark, and 4 risk factors.",
        ),
        "multi_strategy_fee_correction_v2.json": (
            build_multi_strategy_fixture(seed=42, apply_fee_correction=True),
            "Revised 10-day dataset incorporating a retroactive prime-broker fee rebate on 2026-01-09 to demonstrate immutable v2 revision superseding v1.",
        ),
        "hand_worked_portfolio.json": (
            build_hand_worked_fixture(),
            "2-day exact rational hand-worked portfolio (USD + EUR, SOD cash flow, intraday BUY rebalance, commission, management fee, Brinson benchmark).",
        ),
    }

    manifest_entries = []
    for fname, (payload, desc) in files.items():
        path = FIXTURES_DIR / fname
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest_entries.append(
            {
                "file": fname,
                "batch_id": payload["batch_id"],
                "portfolio_id": payload["portfolio_id"],
                "data_source": payload["data_source"],
                "entitlement_tag": payload["entitlement_tag"],
                "sha256": _sha256_file(path),
                "description": desc,
            }
        )

    manifest = {
        "manifest_version": "1.0.0",
        "generator": "data/fixtures/generate_fixtures.py",
        "seed": 42,
        "entitlement_policy": "All fixtures are 100% synthetic and explicitly permitted for local testing and CI.",
        "fixtures": manifest_entries,
    }
    (FIXTURES_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Generated {len(manifest_entries)} deterministic fixtures in {FIXTURES_DIR}")


if __name__ == "__main__":
    main()
