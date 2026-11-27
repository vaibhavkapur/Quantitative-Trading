"""Standalone zero-CDN HTML/SVG attribution dashboard and report viewer."""

from __future__ import annotations

import html
from typing import Any, Dict, List


def _esc(val: Any) -> str:
    return html.escape(str(val))


def _svg_bar_chart(items: List[Dict[str, Any]], label_key: str, value_key: str, title: str) -> str:
    if not items:
        return f"<p class='muted'>No data for {_esc(title)}.</p>"
    width = 640
    row_h = 30
    height = 44 + len(items) * row_h
    max_abs = max((abs(float(x[value_key])) for x in items), default=1e-6)
    max_abs = max(max_abs, 1e-6)
    zero_x = 340
    scale = 200.0 / max_abs

    bars = [
        f"<text x='10' y='22' font-size='13' font-weight='600' fill='#0f172a'>{_esc(title)}</text>",
        f"<line x1='{zero_x}' y1='32' x2='{zero_x}' y2='{height - 8}' stroke='#94a3b8' stroke-dasharray='3,3'/>",
    ]
    for i, item in enumerate(items):
        y = 38 + i * row_h
        val = float(item[value_key])
        bar_w = abs(val) * scale
        x = zero_x if val >= 0 else zero_x - bar_w
        color = "#0d9488" if val >= 0 else "#dc2626"
        label = _esc(item[label_key])
        pct_str = f"{val * 100.0:+.3f}%"
        bars.append(
            f"<text x='10' y='{y + 15}' font-size='12' fill='#334155'>{label}</text>"
            f"<rect x='{x:.1f}' y='{y + 3}' width='{max(bar_w, 1.5):.1f}' height='16' rx='3' fill='{color}'/>"
            f"<text x='555' y='{y + 15}' font-size='11' fill='#0f172a'>{pct_str}</text>"
        )
    return (
        f"<svg viewBox='0 0 {width} {height}' width='100%' style='max-width:{width}px;background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;'>"
        + "".join(bars)
        + "</svg>"
    )


def render_report_html(report: Dict[str, Any]) -> str:
    """Render an AttributionReport dictionary into a rich standalone HTML page."""
    meta = report["metadata"]
    ret = report["returns"]
    wf = report["waterfall"]
    contrib = report["contributions"]
    br = report["brinson"]
    fac = report["factors"]
    tca = report["execution_costs"]
    rec = report["reconciliation"]

    strat_svg = _svg_bar_chart(
        contrib["by_strategy"], "group_key", "linked_net_contribution", "Strategy Net Contribution (Carino-Linked)"
    )
    inst_svg = _svg_bar_chart(
        contrib["by_instrument"], "group_key", "linked_net_contribution", "Instrument Net Contribution (Carino-Linked)"
    )

    waterfall_items = [
        {"label": "Paper Pre-IS", "val": wf["theoretical_paper_return_before_is"]},
        {"label": "Implicit TCA Drag", "val": -wf["implicit_execution_cost_drag"]},
        {"label": "Gross Return", "val": ret["cumulative_gross_return"]},
        {"label": "Explicit Trade Fees", "val": -wf["explicit_trade_cost_drag"]},
        {"label": "Other Fees & Comp.", "val": -wf["other_fees_and_compounding_drag"]},
        {"label": "Net Return", "val": ret["cumulative_net_return"]},
        {"label": "Explicit Residual", "val": rec["linked_net_residual"]},
    ]
    wf_svg = _svg_bar_chart(waterfall_items, "label", "val", "Gross-to-Net & Execution Cost Waterfall")

    status_color = {
        "RECONCILED": "#15803d",
        "RECONCILED_WITH_WARNINGS": "#b45309",
        "DEGRADED_ZERO_NAV": "#b91c1c",
        "UNRECONCILED": "#b91c1c",
    }.get(rec["status"], "#334155")

    exc_rows = "".join(
        f"<tr><td><strong>{_esc(e['severity'])}</strong></td><td><code>{_esc(e['code'])}</code></td>"
        f"<td>{_esc(e.get('trade_date') or 'PERIOD')}</td><td>{_esc(e['message'])}</td></tr>"
        for e in rec["exceptions"]
    ) or "<tr><td colspan='4' class='muted'>No data-quality exceptions.</td></tr>"

    brinson_rows = "".join(
        f"<tr><td>{_esc(s['sector'])}</td><td>{s['avg_portfolio_weight']*100:.2f}%</td>"
        f"<td>{s['avg_benchmark_weight']*100:.2f}%</td><td>{s['allocation_effect']*10000:+.2f} bps</td>"
        f"<td>{s['selection_effect']*10000:+.2f} bps</td><td>{s['interaction_effect']*10000:+.2f} bps</td>"
        f"<td><strong>{s['total_effect']*10000:+.2f} bps</strong></td></tr>"
        for s in br["by_sector"]
    ) or "<tr><td colspan='7' class='muted'>Brinson sector data unavailable.</td></tr>"

    factor_rows = "".join(
        f"<tr><td>{_esc(f['factor_name'])}</td><td>{f['beta']:+.4f}</td>"
        f"<td>{f['t_stat']:.2f}</td><td>{f['linked_contribution']*100:+.4f}%</td></tr>"
        if f["t_stat"] is not None
        else f"<tr><td>{_esc(f['factor_name'])}</td><td>{f['beta']:+.4f}</td><td>N/A</td><td>{f['linked_contribution']*100:+.4f}%</td></tr>"
        for f in fac["factors"]
    ) or "<tr><td colspan='4' class='muted'>Factor regression unavailable or insufficient observations.</td></tr>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>PAE Report — {_esc(meta['version_id'])}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; background: #f8fafc; color: #0f172a; }}
  .banner {{ background: #fef3c7; border: 1px solid #f59e0b; padding: 10px 14px; border-radius: 8px; margin-bottom: 18px; font-size: 13px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-bottom: 20px; }}
  .card {{ background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px; }}
  .card .label {{ font-size: 12px; color: #64748b; text-transform: uppercase; }}
  .card .value {{ font-size: 22px; font-weight: 700; margin-top: 4px; }}
  .charts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 16px; margin-bottom: 22px; }}
  table {{ width: 100%; border-collapse: collapse; background: #ffffff; border: 1px solid #e2e8f0; margin-bottom: 20px; font-size: 13px; }}
  th, td {{ padding: 8px 10px; border-bottom: 1px solid #e2e8f0; text-align: left; }}
  th {{ background: #f1f5f9; }}
  .muted {{ color: #64748b; font-style: italic; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 999px; color: #ffffff; font-weight: 600; font-size: 12px; background: {status_color}; }}
</style>
</head>
<body>
  <div class="banner">
    <strong>{_esc(meta['environment_mode'])}</strong> — Order execution is disabled. Gross and Net returns are strictly separated.
    Factor correlation does not prove causation. Methodology: <code>{_esc(meta['methodology_version'])}</code>.
  </div>
  <h1>Performance Attribution Report <span class="badge">{_esc(rec['status'])}</span></h1>
  <p>Version: <code>{_esc(meta['version_id'])}</code> | Portfolio: <code>{_esc(meta['portfolio_id'])}</code> |
     Period: {_esc(meta['period_start'])} to {_esc(meta['period_end'])} | Currency: {_esc(meta['reporting_currency'])}</p>

  <div class="grid">
    <div class="card"><div class="label">Cumulative Gross Return</div><div class="value">{ret['cumulative_gross_return']*100:+.4f}%</div></div>
    <div class="card"><div class="label">Cumulative Net Return</div><div class="value">{ret['cumulative_net_return']*100:+.4f}%</div></div>
    <div class="card"><div class="label">Total Explicit Fee Drag</div><div class="value">{ret['cumulative_fee_drag']*10000:+.2f} bps</div></div>
    <div class="card"><div class="label">Implementation Shortfall</div><div class="value">{tca['total_is_bps_of_notional']:+.2f} bps</div></div>
    <div class="card"><div class="label">Linked Net Residual</div><div class="value">{rec['linked_net_residual']*10000:+.4f} bps</div></div>
  </div>

  <div class="charts">
    <div>{wf_svg}</div>
    <div>{strat_svg}</div>
    <div>{inst_svg}</div>
  </div>

  <h2>Brinson Sector Attribution ({_esc(br['model'])} — {_esc(br['status'])})</h2>
  <table>
    <thead><tr><th>Sector</th><th>Port Wt</th><th>Bench Wt</th><th>Allocation</th><th>Selection</th><th>Interaction</th><th>Total</th></tr></thead>
    <tbody>{brinson_rows}</tbody>
  </table>

  <h2>Statistical Risk-Factor Attribution ({_esc(fac['status'])})</h2>
  <p class="muted">{_esc(fac['disclaimer'])}</p>
  <table>
    <thead><tr><th>Factor</th><th>Beta</th><th>t-stat</th><th>Linked Contribution</th></tr></thead>
    <tbody>{factor_rows}</tbody>
  </table>

  <h2>Data-Quality &amp; Reconciliation Exceptions</h2>
  <table>
    <thead><tr><th>Severity</th><th>Code</th><th>Date</th><th>Message</th></tr></thead>
    <tbody>{exc_rows}</tbody>
  </table>
</body>
</html>
"""


def render_dashboard_index(reports: List[Dict[str, Any]], health: Dict[str, Any]) -> str:
    """Render main dashboard listing stored report versions with links to HTML/JSON views."""
    rows = "".join(
        f"<tr><td><code>{_esc(r['version_id'])}</code></td><td>{_esc(r['portfolio_id'])}</td>"
        f"<td>v{r['version_number']}</td><td>{_esc(r['period_start'])} &rarr; {_esc(r['period_end'])}</td>"
        f"<td>{_esc(r['reconciliation_status'])}</td><td><strong>{_esc(r['status'])}</strong></td>"
        f"<td><a href='/reports/{_esc(r['version_id'])}/html'>View Dashboard</a> | "
        f"<a href='/api/v1/reports/{_esc(r['version_id'])}'>JSON</a></td></tr>"
        for r in reports
    ) or "<tr><td colspan='7'>No reports stored yet. POST /api/v1/attribution/run to generate one.</td></tr>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Performance Attribution Engine Dashboard</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; background: #f8fafc; color: #0f172a; }}
  .banner {{ background: #e0f2fe; border: 1px solid #0284c7; padding: 10px 14px; border-radius: 8px; margin-bottom: 18px; }}
  table {{ width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #e2e8f0; }}
  th, td {{ padding: 10px; border-bottom: 1px solid #e2e8f0; text-align: left; font-size: 13px; }}
  th {{ background: #f1f5f9; }}
</style>
</head>
<body>
  <div class="banner">
    <strong>Performance Attribution Engine</strong> | Methodology: <code>{_esc(health['methodology_version'])}</code> |
    Environment: <code>{_esc(health['environment_mode'])}</code> | Order Execution: <strong>DISABLED</strong> |
    Audit Chain Verified: <strong>{_esc(health['audit_chain_valid'])}</strong>
  </div>
  <h1>Versioned Attribution Reports</h1>
  <table>
    <thead><tr><th>Version ID</th><th>Portfolio</th><th>Version</th><th>Period</th><th>Reconciliation</th><th>Workflow</th><th>Links</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</body>
</html>
"""
