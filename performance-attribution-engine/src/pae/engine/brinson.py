"""Brinson-Fachler and Brinson-Hood-Beebower sector attribution with Carino active linking."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

from pae.config import AttributionConfig
from pae.domain.models import AttributionInputBatch, DataQualityException
from pae.engine.contribution import ContributionReport
from pae.engine.returns import ReturnSummary


@dataclass(frozen=True)
class SectorBrinsonEffect:
    """Linked multi-period (or single-day) Brinson attribution for one sector."""

    sector: str
    avg_portfolio_weight: float
    avg_benchmark_weight: float
    avg_active_weight: float
    allocation_effect: float
    selection_effect: float
    interaction_effect: float
    total_effect: float
    position_ids: List[str] = field(default_factory=list)
    trade_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DailyBrinsonRecord:
    """Daily Brinson decomposition verifying Allocation + Selection + Interaction + Intraday = Active Gross."""

    trade_date: str
    portfolio_gross_return: float
    benchmark_return: float
    active_gross_return: float
    total_allocation: float
    total_selection: float
    total_interaction: float
    intraday_rebalance_effect: float
    unexplained_residual: float
    sector_effects: List[SectorBrinsonEffect]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BrinsonReport:
    """Multi-period Brinson attribution report with explicit availability status."""

    status: str  # "AVAILABLE", "PARTIAL", "UNAVAILABLE"
    model: str
    benchmark_id: Optional[str]
    aligned_days: int
    total_days: int
    cumulative_portfolio_gross_return: Optional[float]
    cumulative_benchmark_return: Optional[float]
    cumulative_active_gross_return: Optional[float]
    total_allocation_effect: float
    total_selection_effect: float
    total_interaction_effect: float
    total_intraday_rebalance_effect: float
    total_residual: float
    by_sector: List[SectorBrinsonEffect] = field(default_factory=list)
    daily_records: List[DailyBrinsonRecord] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _active_carino_weights(pairs: List[Tuple[str, float, float]]) -> Dict[str, float]:
    """Compute exact Carino (1999) active-return smoothing factors.

    Given daily (date, r_p, r_b), produces weights w_t such that:
        sum_t w_t * (r_p,t - r_b,t) == R_p - R_b.
    """
    if not pairs:
        return {}
    rp_cum = 1.0
    rb_cum = 1.0
    for _, rp, rb in pairs:
        if 1.0 + rp <= 0 or 1.0 + rb <= 0:
            return {d: 1.0 for d, _, _ in pairs}
        rp_cum *= 1.0 + rp
        rb_cum *= 1.0 + rb
    rp_cum -= 1.0
    rb_cum -= 1.0

    active_total = rp_cum - rb_cum
    if abs(active_total) < 1e-12:
        k_total = 1.0 / (1.0 + rp_cum) if (1.0 + rp_cum) > 0 else 1.0
    else:
        k_total = (math.log1p(rp_cum) - math.log1p(rb_cum)) / active_total

    weights: Dict[str, float] = {}
    for d, rp, rb in pairs:
        diff = rp - rb
        if abs(diff) < 1e-12:
            k_t = 1.0 / (1.0 + rp) if (1.0 + rp) > 0 else 1.0
        else:
            k_t = (math.log1p(rp) - math.log1p(rb)) / diff
        weights[d] = k_t / k_total
    return weights


def compute_brinson_attribution(
    batch: AttributionInputBatch,
    config: AttributionConfig,
    returns: ReturnSummary,
    contributions: ContributionReport,
    exceptions: List[DataQualityException],
) -> BrinsonReport:
    """Compute daily and Carino-linked Brinson-Fachler or BHB sector attribution."""
    bm_by_date = {b.trade_date: b for b in batch.benchmark}
    ret_by_date = {r.trade_date: r for r in returns.daily_returns}
    sum_by_date = {s.trade_date: s for s in contributions.daily_summaries}

    # Index daily position contributions by date & sector
    pos_by_date_sector: Dict[str, Dict[str, List]] = {}
    for pc in contributions.daily_positions:
        pos_by_date_sector.setdefault(pc.trade_date, {}).setdefault(pc.sector, []).append(pc)

    daily_records: List[DailyBrinsonRecord] = []
    active_pairs: List[Tuple[str, float, float]] = []
    notes: List[str] = []

    for dret in returns.daily_returns:
        d = dret.trade_date
        if dret.is_zero_nav_day:
            continue
        bm = bm_by_date.get(d)
        if bm is None or dret.benchmark_return_aligned is None:
            continue
        if not bm.sector_weights or not bm.sector_returns:
            exceptions.append(
                DataQualityException(
                    code="BRINSON_MISSING_SECTOR_DATA",
                    severity="INFO",
                    trade_date=d,
                    entity_id=bm.benchmark_id,
                    message=f"Benchmark {bm.benchmark_id} lacks sector weights/returns on {d}; Brinson skipped for this date.",
                )
            )
            continue

        # Convert benchmark sector returns to reporting currency if needed
        bm_fx_ret = dret.benchmark_fx_return or 0.0
        bm_weights = dict(bm.sector_weights)
        bm_rets = {
            sec: (1.0 + r_loc) * (1.0 + bm_fx_ret) - 1.0
            for sec, r_loc in bm.sector_returns.items()
        }

        # Check benchmark sector consistency
        rb_from_sectors = sum(bm_weights.get(s, 0.0) * bm_rets.get(s, 0.0) for s in set(bm_weights) | set(bm_rets))
        rb_total = dret.benchmark_return_aligned

        # Build portfolio sector weights & returns from daily position contributions
        sec_map = pos_by_date_sector.get(d, {})
        dsum = sum_by_date[d]
        p_weights: Dict[str, float] = {}
        p_rets: Dict[str, float] = {}
        p_pos_ids: Dict[str, List[str]] = {}
        p_trade_ids: Dict[str, List[str]] = {}
        intraday_rebal_effect = 0.0

        invested_weight = 0.0
        for sec, pcs in sec_map.items():
            w_s = sum(p.start_weight for p in pcs)
            c_s = sum(p.gross_contribution for p in pcs)
            p_pos_ids[sec] = sorted({p.position_id for p in pcs})
            p_trade_ids[sec] = sorted({tid for p in pcs for tid in p.trade_ids})
            if abs(w_s) > 1e-12:
                p_weights[sec] = w_s
                p_rets[sec] = c_s / w_s
                invested_weight += w_s
            else:
                # Position entered intraday with 0 start weight: isolate in intraday_rebalance_effect
                intraday_rebal_effect += c_s

        # Residual weight assigned to CASH if portfolio holds cash (so sum(w_p) == 1.0)
        cash_w = 1.0 - invested_weight
        if abs(cash_w) > 1e-9 or "CASH" in bm_weights:
            p_weights["CASH"] = p_weights.get("CASH", 0.0) + cash_w
            if abs(p_weights["CASH"]) > 1e-12:
                p_rets["CASH"] = dsum.cash_contribution / p_weights["CASH"]
            else:
                p_rets["CASH"] = 0.0

        all_sectors = sorted(set(p_weights.keys()) | set(bm_weights.keys()))
        sec_effects: List[SectorBrinsonEffect] = []
        tot_alloc = 0.0
        tot_sel = 0.0
        tot_int = 0.0

        for sec in all_sectors:
            wp = p_weights.get(sec, 0.0)
            wb = bm_weights.get(sec, 0.0)
            rp = p_rets.get(sec, 0.0)
            rb_s = bm_rets.get(sec, 0.0)

            if config.brinson_model == "BRINSON_FACHLER":
                alloc = (wp - wb) * (rb_s - rb_from_sectors)
            else:
                alloc = (wp - wb) * rb_s
            sel = wb * (rp - rb_s)
            inter = (wp - wb) * (rp - rb_s)
            tot = alloc + sel + inter

            tot_alloc += alloc
            tot_sel += sel
            tot_int += inter

            sec_effects.append(
                SectorBrinsonEffect(
                    sector=sec,
                    avg_portfolio_weight=wp,
                    avg_benchmark_weight=wb,
                    avg_active_weight=wp - wb,
                    allocation_effect=alloc,
                    selection_effect=sel,
                    interaction_effect=inter,
                    total_effect=tot,
                    position_ids=p_pos_ids.get(sec, []),
                    trade_ids=p_trade_ids.get(sec, []),
                )
            )

        active_gross = dret.gross_return - rb_total
        explained = tot_alloc + tot_sel + tot_int + intraday_rebal_effect
        residual = active_gross - explained

        daily_records.append(
            DailyBrinsonRecord(
                trade_date=d,
                portfolio_gross_return=dret.gross_return,
                benchmark_return=rb_total,
                active_gross_return=active_gross,
                total_allocation=tot_alloc,
                total_selection=tot_sel,
                total_interaction=tot_int,
                intraday_rebalance_effect=intraday_rebal_effect,
                unexplained_residual=residual,
                sector_effects=sec_effects,
            )
        )
        active_pairs.append((d, dret.gross_return, rb_total))

    if not daily_records:
        notes.append("Brinson attribution unavailable: no aligned dates with benchmark sector weights and returns.")
        return BrinsonReport(
            status="UNAVAILABLE",
            model=config.brinson_model,
            benchmark_id=returns.benchmark_id,
            aligned_days=0,
            total_days=returns.valid_return_days,
            cumulative_portfolio_gross_return=returns.cumulative_gross_return,
            cumulative_benchmark_return=returns.cumulative_benchmark_return,
            cumulative_active_gross_return=returns.cumulative_active_gross_return,
            total_allocation_effect=0.0,
            total_selection_effect=0.0,
            total_interaction_effect=0.0,
            total_intraday_rebalance_effect=0.0,
            total_residual=0.0,
            notes=notes,
        )

    status = "AVAILABLE" if len(daily_records) == returns.valid_return_days else "PARTIAL"
    if status == "PARTIAL":
        notes.append(
            f"Brinson computed over {len(daily_records)} of {returns.valid_return_days} valid trading days."
        )

    # Link multi-period Brinson effects with active Carino weights
    w_act = _active_carino_weights(active_pairs)
    n_aligned = len(daily_records)

    rp_cum = 1.0
    rb_cum = 1.0
    for _, rp, rb in active_pairs:
        rp_cum *= 1.0 + rp
        rb_cum *= 1.0 + rb
    rp_cum -= 1.0
    rb_cum -= 1.0
    act_cum = rp_cum - rb_cum

    sec_acc: Dict[str, Dict[str, object]] = {}
    tot_alloc_l = 0.0
    tot_sel_l = 0.0
    tot_int_l = 0.0
    tot_intra_l = 0.0
    tot_res_l = 0.0

    for rec in daily_records:
        w = w_act[rec.trade_date]
        tot_alloc_l += w * rec.total_allocation
        tot_sel_l += w * rec.total_selection
        tot_int_l += w * rec.total_interaction
        tot_intra_l += w * rec.intraday_rebalance_effect
        tot_res_l += w * rec.unexplained_residual
        for se in rec.sector_effects:
            b = sec_acc.setdefault(
                se.sector,
                {
                    "wp": 0.0,
                    "wb": 0.0,
                    "alloc": 0.0,
                    "sel": 0.0,
                    "inter": 0.0,
                    "pos_ids": [],
                    "trade_ids": [],
                },
            )
            b["wp"] = float(b["wp"]) + se.avg_portfolio_weight
            b["wb"] = float(b["wb"]) + se.avg_benchmark_weight
            b["alloc"] = float(b["alloc"]) + w * se.allocation_effect
            b["sel"] = float(b["sel"]) + w * se.selection_effect
            b["inter"] = float(b["inter"]) + w * se.interaction_effect
            b["pos_ids"].extend(se.position_ids)  # type: ignore[union-attr]
            b["trade_ids"].extend(se.trade_ids)  # type: ignore[union-attr]

    by_sector: List[SectorBrinsonEffect] = []
    for sec in sorted(sec_acc.keys()):
        b = sec_acc[sec]
        wp_avg = float(b["wp"]) / n_aligned
        wb_avg = float(b["wb"]) / n_aligned
        al = float(b["alloc"])
        sl = float(b["sel"])
        it = float(b["inter"])
        by_sector.append(
            SectorBrinsonEffect(
                sector=sec,
                avg_portfolio_weight=wp_avg,
                avg_benchmark_weight=wb_avg,
                avg_active_weight=wp_avg - wb_avg,
                allocation_effect=al,
                selection_effect=sl,
                interaction_effect=it,
                total_effect=al + sl + it,
                position_ids=sorted(set(b["pos_ids"])),  # type: ignore[arg-type]
                trade_ids=sorted(set(b["trade_ids"])),  # type: ignore[arg-type]
            )
        )

    return BrinsonReport(
        status=status,
        model=config.brinson_model,
        benchmark_id=returns.benchmark_id,
        aligned_days=n_aligned,
        total_days=returns.valid_return_days,
        cumulative_portfolio_gross_return=rp_cum,
        cumulative_benchmark_return=rb_cum,
        cumulative_active_gross_return=act_cum,
        total_allocation_effect=tot_alloc_l,
        total_selection_effect=tot_sel_l,
        total_interaction_effect=tot_int_l,
        total_intraday_rebalance_effect=tot_intra_l,
        total_residual=tot_res_l,
        by_sector=by_sector,
        daily_records=daily_records,
        notes=notes,
    )
