"""Exporters rendering immutable AttributionReport payloads to JSON, Markdown, CSV, and HTML."""

from __future__ import annotations

import csv
import html
import io
import json
from pathlib import Path
from typing import Any, Dict


def _pct(v: Any) -> str:
    if v is None:
        return "N/A"
    return f"{float(v) * 100.0:+.4f}%"


def _bps(v: Any) -> str:
    if v is None:
        return "N/A"
    return f"{float(v) * 10000.0:+.2f} bps"


def _num(v: Any) -> str:
    if v is None:
        return "N/A"
    return f"{float(v):,.2f}"


def export_json(report: Dict[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True)


def export_markdown(report: Dict[str, Any]) -> str:
    meta = report["metadata"]
    ret = report["returns"]
    wf = report["waterfall"]
    contrib = report["contributions"]
    br = report["brinson"]
    fac = report["factors"]
    tca = report["execution_costs"]
    rec = report["reconciliation"]

    lines = [
        f"# Performance Attribution Report — `{meta['version_id']}`",
        "",
        f"> **{meta['environment_mode']}** | Methodology: `{meta['methodology_version']}` | "
        f"Currency: `{meta['reporting_currency']}` | Reconciliation: **{rec['status']}**",
        "",
        "## 1. Metadata & Disclosures",
        "",
        f"- **Portfolio:** `{meta['portfolio_id']}` ({meta['period_start']} to {meta['period_end']})",
        f"- **Version:** v{meta['version_number']} (supersedes: `{meta.get('supersedes_version_id') or 'None'}`)",
        f"- **Correction Reason:** {meta.get('correction_reason') or 'Initial release'}",
        f"- **Input Hash:** `{meta['input_hash']}`",
        f"- **Config Hash:** `{meta['config_hash']}`",
        f"- **Report Hash:** `{meta['report_hash']}`",
        "",
    ]
    for disc in meta.get("disclosures", []):
        lines.append(f"- {disc}")

    if report.get("revision_diff"):
        rd = report["revision_diff"]
        lines.extend(
            [
                "",
                "## 1b. Revision Diff vs. Prior Version",
                "",
                f"- **Prior Version:** `{rd.get('prior_version_id')}`",
                f"- **Delta Gross Return:** {_pct(rd.get('delta_cumulative_gross_return'))}",
                f"- **Delta Net Return:** {_pct(rd.get('delta_cumulative_net_return'))}",
                f"- **Delta Total Fees ({meta['reporting_currency']}):** {_num(rd.get('delta_total_fees_base'))}",
            ]
        )

    lines.extend(
        [
            "",
            "## 2. Return & Gross-to-Net Waterfall",
            "",
            "| Metric | Value | Notes |",
            "|---|---:|---|",
            f"| Theoretical Paper Return (Pre-IS) | {_pct(wf['theoretical_paper_return_before_is'])} | Before implicit execution slippage |",
            f"| Implicit Execution Cost Drag | {-wf['implicit_execution_cost_drag'] * 100.0:+.4f}% | Delay + impact + opportunity |",
            f"| **Cumulative Gross Return (TWR)** | **{_pct(ret['cumulative_gross_return'])}** | Strictly excludes explicit fees |",
            f"| Explicit Trading Cost Drag | {-wf['explicit_trade_cost_drag'] * 100.0:+.4f}% | Commissions + exchange fees + tax |",
            f"| Other Fees & Compounding Drag | {-wf['other_fees_and_compounding_drag'] * 100.0:+.4f}% | Management/admin/corrections |",
            f"| **Cumulative Net Return (TWR)** | **{_pct(ret['cumulative_net_return'])}** | Net of all explicit fees |",
            f"| Modified Dietz Net (Cross-Check) | {_pct(ret['modified_dietz_net_return'])} | External cash-flow sensitivity check |",
            f"| Aligned Benchmark Return (`{ret.get('benchmark_id') or 'None'}`) | {_pct(ret['cumulative_benchmark_return'])} | {ret['benchmark_aligned_days']} aligned / {ret['benchmark_missing_days']} missing days |",
            f"| Active Gross Return | {_pct(ret['cumulative_active_gross_return'])} | Gross minus aligned benchmark |",
            f"| Active Net Return | {_pct(ret['cumulative_active_net_return'])} | Net minus aligned benchmark |",
            f"| Explicit Linked Net Residual | {_bps(rec['linked_net_residual'])} | Tolerance: ±{rec['return_tolerance_bps']:.2f} bps |",
            "",
            "## 3. Strategy Contribution (Carino-Linked)",
            "",
            "| Strategy | Avg Weight | Local Price | FX | Trading | Income | Gross Contrib | Fee Drag | Net Contrib | Positions | Trades |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in contrib["by_strategy"]:
        lines.append(
            f"| `{row['group_key']}` | {row['avg_weight']*100:.2f}% | {_pct(row['linked_local_price_contribution'])} | "
            f"{_pct(row['linked_fx_contribution'])} | {_pct(row['linked_trading_contribution'])} | "
            f"{_pct(row['linked_income_contribution'])} | {_pct(row['linked_gross_contribution'])} | "
            f"{_bps(-row['linked_fee_drag_contribution'])} | **{_pct(row['linked_net_contribution'])}** | "
            f"{len(row['position_ids'])} | {len(row['trade_ids'])} |"
        )

    lines.extend(
        [
            "",
            "## 4. Instrument Contribution (Carino-Linked)",
            "",
            "| Instrument | Avg Weight | Gross Contrib | FX Contrib | Fee Drag | Net Contrib | Net P&L | Linked Positions |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in contrib["by_instrument"]:
        pos_preview = ", ".join(row["position_ids"][:3]) + ("..." if len(row["position_ids"]) > 3 else "")
        lines.append(
            f"| `{row['group_key']}` | {row['avg_weight']*100:.2f}% | {_pct(row['linked_gross_contribution'])} | "
            f"{_pct(row['linked_fx_contribution'])} | {_bps(-row['linked_fee_drag_contribution'])} | "
            f"**{_pct(row['linked_net_contribution'])}** | {_num(row['total_net_pnl_base'])} | `{pos_preview}` |"
        )

    lines.extend(
        [
            "",
            f"## 5. Brinson Sector Attribution ({br['model']} — `{br['status']}`)",
            "",
        ]
    )
    if br["by_sector"]:
        lines.extend(
            [
                "| Sector | Port Wt | Bench Wt | Active Wt | Allocation | Selection | Interaction | Total Effect |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for s in br["by_sector"]:
            lines.append(
                f"| `{s['sector']}` | {s['avg_portfolio_weight']*100:.2f}% | {s['avg_benchmark_weight']*100:.2f}% | "
                f"{s['avg_active_weight']*100:+.2f}% | {_bps(s['allocation_effect'])} | {_bps(s['selection_effect'])} | "
                f"{_bps(s['interaction_effect'])} | **{_bps(s['total_effect'])}** |"
            )
    else:
        lines.append("_Brinson attribution unavailable for this period (missing aligned benchmark sector data)._")

    lines.extend(
        [
            "",
            f"## 6. Statistical Risk-Factor Attribution (`{fac['status']}`)",
            "",
            f"> _{fac['disclaimer']}_",
            "",
        ]
    )
    if fac["status"] == "AVAILABLE":
        lines.extend(
            [
                f"- **Sample Size:** {fac['sample_size']} days | **R²:** {fac['r_squared']:.4f} | "
                f"**Specific Return:** {_pct(fac['linked_specific_return'])}",
                "",
                "| Factor | Beta | Std Error | t-stat | Linked Contribution |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for f in fac["factors"]:
            se_str = f"{f['std_error']:.4f}" if f["std_error"] is not None else "N/A"
            t_str = f"{f['t_stat']:.2f}" if f["t_stat"] is not None else "N/A"
            lines.append(
                f"| `{f['factor_name']}` | {f['beta']:+.4f} | {se_str} | {t_str} | {_pct(f['linked_contribution'])} |"
            )
    else:
        lines.append(f"_Factor attribution status: {fac['status']}._")

    lines.extend(
        [
            "",
            "## 7. Execution-Cost Attribution (Implementation Shortfall)",
            "",
            f"- **Trades:** {tca['trade_count']} | **Executed Notional:** {_num(tca['total_executed_notional_base'])} {meta['reporting_currency']} | "
            f"**Total IS:** {_num(tca['total_implementation_shortfall_base'])} ({tca['total_is_bps_of_notional']:+.2f} bps of notional)",
            f"- **Delay:** {_num(tca['total_delay_cost_base'])} | **Market Impact:** {_num(tca['total_market_impact_cost_base'])} | "
            f"**Opportunity:** {_num(tca['total_opportunity_cost_base'])} | **Explicit Fees:** {_num(tca['total_explicit_cost_base'])}",
            "",
            "## 8. Reconciliation Checks & Data-Quality Exceptions",
            "",
            "| Check | Status | Residual | Tolerance |",
            "|---|---|---:|---:|",
        ]
    )
    for chk in rec["checks"]:
        badge = "PASS" if chk["passed"] else "FAIL"
        lines.append(
            f"| `{chk['check_name']}` | **{badge}** | `{chk['residual']:.2e}` | `{chk['tolerance']:.2e}` |"
        )

    if rec["exceptions"]:
        lines.extend(["", "### Data-Quality Exceptions", ""])
        for ex in rec["exceptions"]:
            lines.append(
                f"- **[{ex['severity']}] `{ex['code']}`** ({ex.get('trade_date') or 'PERIOD'} / `{ex.get('entity_id') or 'PORTFOLIO'}`): {ex['message']}"
            )

    return "\n".join(lines) + "\n"


def export_csv(report: Dict[str, Any]) -> str:
    """Export daily position contributions with full position/trade provenance as CSV."""
    buf = io.StringIO()
    fieldnames = [
        "trade_date",
        "position_id",
        "strategy_id",
        "instrument_id",
        "sector",
        "local_currency",
        "start_weight",
        "end_weight",
        "local_price_contribution",
        "fx_total_contribution",
        "trading_contribution",
        "income_contribution",
        "gross_contribution",
        "fee_drag_contribution",
        "net_contribution",
        "gross_pnl_base",
        "fees_base",
        "net_pnl_base",
        "trade_ids",
    ]
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for row in report["contributions"]["daily_positions"]:
        out = {k: row.get(k) for k in fieldnames if k != "trade_ids"}
        out["trade_ids"] = ";".join(row.get("trade_ids", []))
        writer.writerow(out)
    return buf.getvalue()


def export_html(report: Dict[str, Any]) -> str:
    """Render a standalone zero-CDN HTML report with inline SVG waterfall chart."""
    from pae.web.dashboard import render_report_html

    return render_report_html(report)


def write_report_bundle(report: Dict[str, Any], output_dir: Path) -> Dict[str, str]:
    """Write JSON, Markdown, CSV, and HTML exports for an AttributionReport."""
    output_dir.mkdir(parents=True, exist_ok=True)
    vid = report["metadata"]["version_id"]
    paths: Dict[str, str] = {}
    for ext, fn in (
        ("json", export_json),
        ("md", export_markdown),
        ("csv", export_csv),
        ("html", export_html),
    ):
        p = output_dir / f"{vid}.{ext}"
        p.write_text(fn(report), encoding="utf-8")
        paths[ext] = str(p)
    return paths
