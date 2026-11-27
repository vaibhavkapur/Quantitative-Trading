#!/usr/bin/env python3
"""Run reproducible walk-forward comparison across baselines and constrained allocators."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dcae.domain import EngineConfig  # noqa: E402
from dcae.evaluation import WalkForwardEvaluator  # noqa: E402
from dcae.fixtures import generate_walkforward_strategies  # noqa: E402


def render_markdown(report: dict) -> str:
    w = report["evaluation_window"]
    lines = [
        "# Walk-Forward Evaluation Report",
        "",
        f"> **Disclaimer:** {report['disclaimer']}",
        "",
        "## Evaluation Configuration",
        "",
        f"- **Data source:** `{report['provenance']['source']}` (`{report['provenance']['entitlement']}`, seed `{report['provenance']['seed']}`)",
        f"- **Period:** `{w['start_utc']}` to `{w['end_utc']}` ({w['evaluated_periods']} trading days, warmup {w['warmup_periods']} days)",
        f"- **Rebalance cadence:** every {w['rebalance_every']} trading days ({w['rebalance_count']} rebalances, lookback {w['lookback_window']} days)",
        f"- **Initial paper capital:** \\${w['initial_nav_usd']:,.2f} USD",
        "- **Identical cost model:** all allocators pay the exact same per-strategy transaction cost schedule and cash reserve rules.",
        "",
        "## Comparative Results (Net of Transaction Costs)",
        "",
        "| Allocator | Final NAV (USD) | Cum. Return | Ann. Return | Ann. Vol | Sharpe (Diag.) | Max Drawdown | Avg 1-Way Turnover | Total Costs (USD) | Avg Cash | Fallbacks | Verifier Violations | Failed Steps | Stress Steps |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, s in report["summaries"].items():
        lines.append(
            f"| `{name}` | \\${s['final_nav_usd']:,.2f} | {s['cumulative_return']:.2%} | "
            f"{s['annualized_return']:.2%} | {s['annualized_volatility']:.2%} | "
            f"{s['sharpe_diagnostic']:.2f} | {s['max_drawdown']:.2%} | "
            f"{s['avg_one_way_turnover']:.2%} | \\${s['total_transaction_cost_usd']:,.2f} | "
            f"{s['avg_cash_weight']:.2%} | {s['fallback_count']} | "
            f"{s['verifier_violation_count']} | {s['failed_step_count']} | {s['stress_active_steps']} |"
        )
    lines.extend(
        [
            "",
            "## Observations & Honest Reporting",
            "",
            "- Baselines (`equal_weight`, `inverse_volatility`, `risk_budget`) do not enforce turnover, ADV liquidity, group exposure, or portfolio volatility constraints; any resulting hard constraint violations are counted transparently under **Verifier Violations**.",
            "- `constrained_mv` and `constrained_mv_stressed` enforce all constraints independently verified at every rebalance step (zero hard verifier violations when feasible, safe fallback otherwise).",
            "- During the injected correlated drawdown window (days 90-110), automatic drawdown stress triggers on `constrained_mv` (`stress_active_steps`), while `constrained_mv_stressed` keeps correlation stress permanently active.",
            "- These metrics reflect synthetic test series only and are **not** evidence of live trading performance.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    cfg = EngineConfig.from_json_file(str(ROOT / "config" / "default_config.json"))
    strategies, provenance = generate_walkforward_strategies(seed=20260409, n_days=180)
    evaluator = WalkForwardEvaluator(cfg)
    report = evaluator.evaluate(
        strategies=strategies,
        provenance=provenance,
        initial_nav_usd=10_000_000.0,
        warmup_periods=30,
        rebalance_every=5,
        lookback_window=60,
    )

    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "walkforward_evaluation.json"
    md_path = reports_dir / "walkforward_evaluation.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"Wrote {json_path} and {md_path}")
    for name, s in report["summaries"].items():
        print(
            f"  {name:24s} ret={s['cumulative_return']:+.2%} vol={s['annualized_volatility']:.2%} "
            f"mdd={s['max_drawdown']:.2%} cost=${s['total_transaction_cost_usd']:,.0f} "
            f"viol={s['verifier_violation_count']} fb={s['fallback_count']}"
        )


if __name__ == "__main__":
    main()
