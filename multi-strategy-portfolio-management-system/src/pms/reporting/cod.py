"""Deterministic Close-of-Day (COD) report generator (JSON + Markdown)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from pms.domain.models import FrozenModel
from pms.ledger.engine import ReplaySnapshot
from pms.reconciliation.models import ReconciliationRun
from pms.valuation.consolidation import FundValuationReport


def _fmt_money(val: Decimal, ccy: str = "USD") -> str:
    return f"{val:,.2f} {ccy}"


class CODReportPackage(FrozenModel):
    report_id: str
    generated_at: datetime
    fund_id: str
    base_currency: str
    ledger_event_count: int
    ledger_last_seq_no: int
    ledger_state_hash: str
    hash_chain_valid: bool
    valuation: FundValuationReport
    reconciliation: ReconciliationRun | None = None
    replay_verified: bool = True


def build_cod_report(
    *,
    replay: ReplaySnapshot,
    valuation: FundValuationReport,
    hash_chain_valid: bool,
    reconciliation: ReconciliationRun | None = None,
    generated_at: datetime | None = None,
) -> CODReportPackage:
    ts = generated_at or datetime.now(UTC)
    report_id = f"COD-{valuation.snapshot_id}-SEQ{replay.last_seq_no}"
    return CODReportPackage(
        report_id=report_id,
        generated_at=ts,
        fund_id=valuation.fund_id,
        base_currency=valuation.base_currency,
        ledger_event_count=replay.event_count,
        ledger_last_seq_no=replay.last_seq_no,
        ledger_state_hash=replay.state_hash,
        hash_chain_valid=hash_chain_valid,
        valuation=valuation,
        reconciliation=reconciliation,
        replay_verified=True,
    )


def render_cod_markdown(pkg: CODReportPackage) -> str:
    v = pkg.valuation
    ccy = v.base_currency
    lines = [
        f"# Close-of-Day Portfolio Report — {pkg.report_id}",
        "",
        f"- **Fund**: {v.fund_name} (`{v.fund_id}`)",
        f"- **As-Of Time (UTC)**: `{v.as_of_time.isoformat()}`",
        f"- **Market Snapshot**: `{v.snapshot_id}`",
        f"- **Ledger Events / Last Seq**: `{pkg.ledger_event_count}` / `{pkg.ledger_last_seq_no}`",
        f"- **Ledger State Hash (SHA-256)**: `{pkg.ledger_state_hash}`",
        f"- **Event Hash Chain Valid**: `{pkg.hash_chain_valid}`",
        f"- **Valuation Complete**: `{v.valuation_complete}` (Unpriced instruments: `{list(v.unpriced_instruments)}`)",
        f"- **Consolidation Check (Fund == Strategies + Unallocated)**: `{'PASS' if v.consolidation_check.passed else 'FAIL'}`",
        "",
        "## Fund Summary",
        "",
        "| Metric | Amount |",
        "|---|---:|",
        f"| Fund NAV | {_fmt_money(v.fund_nav_base, ccy)} |",
        f"| Net Capital Flows | {_fmt_money(v.net_capital_flows_base, ccy)} |",
        f"| Total Cash (Base) | {_fmt_money(v.total_cash_base, ccy)} |",
        f"| Strategy Cash (Base) | {_fmt_money(v.strategy_cash_base, ccy)} |",
        f"| Unallocated Cash (Base) | {_fmt_money(v.unallocated_cash_base, ccy)} |",
        f"| Priced Market Value | {_fmt_money(v.priced_market_value_base, ccy)} |",
        f"| Unpriced Cost Basis (Flagged) | {_fmt_money(v.unpriced_cost_basis_base, ccy)} |",
        f"| Gross Exposure | {_fmt_money(v.gross_exposure_base, ccy)} |",
        f"| Net Exposure | {_fmt_money(v.net_exposure_base, ccy)} |",
        "",
        "## P&L Explainability",
        "",
        "| Component | Amount |",
        "|---|---:|",
        f"| Realized Trading P&L | {_fmt_money(v.pnl.realized_trading_pnl_base, ccy)} |",
        f"| Unrealized Price P&L | {_fmt_money(v.pnl.unrealized_price_pnl_base, ccy)} |",
        f"| FX Translation P&L | {_fmt_money(v.pnl.fx_translation_pnl_base, ccy)} |",
        f"| Dividend Income | {_fmt_money(v.pnl.dividend_income_base, ccy)} |",
        f"| Fees & Commissions (incl. Late Fees) | -{_fmt_money(v.pnl.fees_and_commissions_base, ccy)} |",
        f"| Late Fees Subset | -{_fmt_money(v.pnl.late_fees_base, ccy)} |",
        f"| Audited Adjustments | {_fmt_money(v.pnl.adjustments_base, ccy)} |",
        f"| **Explained Total P&L** | **{_fmt_money(v.pnl.explained_total_pnl_base, ccy)}** |",
        f"| **NAV - Net Capital** | **{_fmt_money(v.pnl.nav_minus_net_capital_base, ccy)}** |",
        f"| **Unexplained Residual** | **{_fmt_money(v.pnl.unexplained_residual_base, ccy)}** |",
        "",
        "## Strategy & Unallocated Breakdown",
        "",
        "| Account | Strategy | Cash (Base) | NAV (Base) | Total P&L | Unpriced |",
        "|---|---|---:|---:|---:|---|",
    ]
    for acct in (*v.strategy_valuations, *v.unallocated_valuations):
        lines.append(
            f"| `{acct.account_id}` | {acct.strategy_id or 'UNALLOCATED'} "
            f"| {_fmt_money(acct.total_cash_base, ccy)} "
            f"| {_fmt_money(acct.nav_base, ccy)} "
            f"| {_fmt_money(acct.pnl.explained_total_pnl_base, ccy)} "
            f"| {', '.join(acct.unpriced_instruments) or 'None'} |"
        )

    if v.exceptions:
        lines.extend(["", "## Valuation Exceptions", ""])
        for ex in v.exceptions:
            lines.append(f"- **[{ex.code}]** `{ex.account_id}` / `{ex.instrument_id or ex.currency}`: {ex.message}")

    if pkg.reconciliation is not None:
        r = pkg.reconciliation
        lines.extend(
            [
                "",
                "## Broker Reconciliation Summary",
                "",
                f"- **Run ID**: `{r.run_id}` (Statement `{r.statement_id}` from `{r.broker_name}`)",
                f"- **Positions Checked**: `{r.positions_checked}` | **Cash Balances Checked**: `{r.cash_balances_checked}`",
                f"- **Total Breaks**: `{len(r.breaks)}` | **Open Breaks**: `{r.open_break_count}` | **Mutated History**: `{r.mutated_history}`",
                "",
            ]
        )
        if r.breaks:
            lines.extend(
                [
                    "| Break ID | Type | Account | Instrument/CCY | Internal | Broker | Diff | Status |",
                    "|---|---|---|---|---:|---:|---:|---|",
                ]
            )
            for b in r.breaks:
                target = b.instrument_id or b.currency
                lines.append(
                    f"| `{b.break_id}` | `{b.break_type.value}` | `{b.account_id}` | `{target}` "
                    f"| {b.internal_value} | {b.broker_value} | {b.difference} | `{b.status.value}` |"
                )
    return "\n".join(lines) + "\n"


def write_cod_artifacts(
    pkg: CODReportPackage, *, json_path: Path | str, markdown_path: Path | str
) -> None:
    jp = Path(json_path)
    mp = Path(markdown_path)
    jp.parent.mkdir(parents=True, exist_ok=True)
    mp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(
        json.dumps(pkg.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    mp.write_text(render_cod_markdown(pkg), encoding="utf-8")
