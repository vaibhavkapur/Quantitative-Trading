"""Command-line interface for the Optimal Execution Research Engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence

from optimal_execution.analytics import sensitivity_to_polars, strategy_comparison_to_polars
from optimal_execution.contracts import (
    ExecutionEngineError,
    ScheduleRequest,
    StrategyType,
)
from optimal_execution.io import (
    build_run_metadata,
    get_logger,
    load_json_file,
    load_scenario,
    log_event,
    mc_summary_to_dict,
    replay_from_manifest,
    save_json_report,
    save_polars_csv,
    save_replay_manifest,
    simulation_result_to_dict,
)
from optimal_execution.schedules import (
    compare_analytic_and_numeric_schedules,
    generate_schedule,
)
from optimal_execution.sensitivity import (
    DEFAULT_LIQUIDITY_REGIMES,
    LiquidityRegime,
    compute_efficient_frontier,
    run_sensitivity_analysis,
)
from optimal_execution.simulator import simulate_deterministic, simulate_monte_carlo


def _default_requests(risk_aversion_lambda: float, target_pov: float) -> List[ScheduleRequest]:
    return [
        ScheduleRequest(strategy=StrategyType.TWAP),
        ScheduleRequest(strategy=StrategyType.VWAP),
        ScheduleRequest(
            strategy=StrategyType.PARTICIPATION,
            target_participation_rate=target_pov,
        ),
        ScheduleRequest(
            strategy=StrategyType.ALMGREN_CHRISS_ANALYTIC,
            risk_aversion_lambda=risk_aversion_lambda,
        ),
        ScheduleRequest(
            strategy=StrategyType.ALMGREN_CHRISS_NUMERIC,
            risk_aversion_lambda=risk_aversion_lambda,
            enforce_capacity_in_optimizer=False,
        ),
    ]


def cmd_demo(args: argparse.Namespace) -> int:
    logger = get_logger()
    scenario = load_scenario(Path(args.scenario))
    requests = _default_requests(args.risk_aversion, args.target_pov)
    extra_cfg = {
        "schedule_requests": [
            {
                "strategy": r.strategy.value,
                "risk_aversion_lambda": r.risk_aversion_lambda,
                "target_participation_rate": r.target_participation_rate,
                "enforce_capacity_in_optimizer": r.enforce_capacity_in_optimizer,
            }
            for r in requests
        ]
    }
    metadata = build_run_metadata(scenario, extra_config=extra_cfg)
    log_event(
        logger,
        "starting_demo_run",
        run_id=metadata.run_id,
        scenario_id=scenario.scenario_id,
        seed=scenario.seed,
        num_mc_paths=scenario.num_monte_carlo_paths,
    )

    det_results = []
    mc_summaries = []
    for req in requests:
        schedule = generate_schedule(scenario, req)
        det = simulate_deterministic(scenario, schedule)
        mc, _ = simulate_monte_carlo(scenario, schedule)
        det_results.append(det)
        mc_summaries.append(mc)
        log_event(
            logger,
            "strategy_evaluated",
            run_id=metadata.run_id,
            strategy=req.strategy.value,
            completion_rate=round(det.metrics.completion_rate, 6),
            det_total_is_bps=round(det.metrics.total_is_bps, 4),
            mc_mean_is_bps=round(mc.mean_total_is_bps, 4),
            mc_std_is_bps=round(mc.std_total_is_bps, 4),
            capacity_capped_slices=det.diagnostics.capacity_capped_slices,
        )

    comparison_df = strategy_comparison_to_polars(det_results, mc_summaries)
    frontier = compute_efficient_frontier(scenario)
    ac_cmp = compare_analytic_and_numeric_schedules(
        scenario,
        risk_aversion_lambda=args.risk_aversion,
        enforce_capacity_on_numeric=False,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = save_json_report(
        out_dir / "baseline_demo_results.json",
        metadata,
        {
            "scenario": scenario.to_dict(),
            "deterministic_results": [simulation_result_to_dict(r) for r in det_results],
            "monte_carlo_summaries": [mc_summary_to_dict(m) for m in mc_summaries],
            "efficient_frontier": frontier,
            "analytic_vs_numeric": ac_cmp,
        },
    )
    csv_path = save_polars_csv(out_dir / "baseline_demo_summary.csv", comparison_df)
    manifest_path = save_replay_manifest(
        out_dir / "baseline_replay_manifest.json",
        scenario,
        requests,
        metadata,
    )

    print(f"Run ID: {metadata.run_id}")
    print(comparison_df)
    print(f"Saved JSON report: {json_path}")
    print(f"Saved CSV summary: {csv_path}")
    print(f"Saved replay manifest: {manifest_path}")
    return 0


def cmd_sensitivity(args: argparse.Namespace) -> int:
    logger = get_logger()
    scenario = load_scenario(Path(args.scenario))
    if args.grid:
        grid_cfg = load_json_file(Path(args.grid))
        regimes = [
            LiquidityRegime(
                name=r["name"],
                liquidity_multiplier=float(r["liquidity_multiplier"]),
                spread_multiplier=float(r["spread_multiplier"]),
                volatility_multiplier=float(r["volatility_multiplier"]),
                impact_multiplier=float(r["impact_multiplier"]),
            )
            for r in grid_cfg["regimes"]
        ]
        lambdas = [float(x) for x in grid_cfg["lambdas"]]
        strategies = [StrategyType(s) for s in grid_cfg["strategies"]]
        target_pov = float(grid_cfg.get("target_participation_rate", 0.12))
    else:
        regimes = list(DEFAULT_LIQUIDITY_REGIMES)
        lambdas = [0.0, 1e-7, 1e-6, 1e-5, 1e-4]
        strategies = [
            StrategyType.TWAP,
            StrategyType.VWAP,
            StrategyType.PARTICIPATION,
            StrategyType.ALMGREN_CHRISS_ANALYTIC,
        ]
        target_pov = 0.12

    metadata = build_run_metadata(
        scenario,
        extra_config={
            "regimes": [r.name for r in regimes],
            "lambdas": lambdas,
            "strategies": [s.value for s in strategies],
        },
    )
    log_event(
        logger,
        "starting_sensitivity_analysis",
        run_id=metadata.run_id,
        regimes=len(regimes),
        lambdas=len(lambdas),
    )

    points = run_sensitivity_analysis(
        base_scenario=scenario,
        regimes=regimes,
        lambdas=lambdas,
        strategies=strategies,
        target_participation_rate=target_pov,
        num_mc_paths=args.mc_paths,
    )
    df = sensitivity_to_polars(points)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = save_json_report(
        out_dir / "sensitivity_results.json",
        metadata,
        {"points": [p.__dict__ for p in points]},
    )
    csv_path = save_polars_csv(out_dir / "sensitivity_grid.csv", df)
    print(f"Run ID: {metadata.run_id} ({len(points)} sensitivity points)")
    print(df.head(12))
    print(f"Saved JSON report: {json_path}")
    print(f"Saved CSV grid: {csv_path}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    scenario = load_scenario(Path(args.scenario))
    lambdas = [float(x) for x in args.lambdas.split(",")]
    comparisons = [
        compare_analytic_and_numeric_schedules(
            scenario,
            risk_aversion_lambda=lam,
            enforce_capacity_on_numeric=args.enforce_capacity,
            rel_l2_tolerance=args.tolerance,
        )
        for lam in lambdas
    ]
    all_ok = all(c["within_tolerance"] for c in comparisons)
    if args.output:
        metadata = build_run_metadata(
            scenario,
            extra_config={"lambdas": lambdas, "tolerance": args.tolerance},
        )
        save_json_report(Path(args.output), metadata, {"comparisons": comparisons, "all_within_tolerance": all_ok})
    print(json.dumps({"all_within_tolerance": all_ok, "comparisons": comparisons}, indent=2))
    return 0 if all_ok else 2


def cmd_validate(args: argparse.Namespace) -> int:
    scenario = load_scenario(Path(args.scenario))
    print(
        json.dumps(
            {
                "valid": True,
                "scenario_id": scenario.scenario_id,
                "schema_version": scenario.schema_version,
                "horizon_slices": scenario.order.horizon_slices,
                "parent_quantity": scenario.order.quantity,
                "total_market_volume": scenario.market.total_market_volume,
            },
            indent=2,
        )
    )
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    scenario, det_results, mc_summaries, metadata = replay_from_manifest(Path(args.manifest))
    df = strategy_comparison_to_polars(det_results, mc_summaries)
    print(f"Replayed Run ID: {metadata.run_id} (scenario={scenario.scenario_id}, seed={scenario.seed})")
    print(df)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="optimal-execution",
        description="Optimal Execution Research Engine CLI (offline research prototype).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_demo = sub.add_parser("demo", help="Run baseline multi-strategy comparison and save reports.")
    p_demo.add_argument("--scenario", required=True, help="Path to scenario JSON file.")
    p_demo.add_argument("--output-dir", default="reports", help="Directory for output artifacts.")
    p_demo.add_argument("--risk-aversion", type=float, default=1e-6, help="Almgren-Chriss lambda.")
    p_demo.add_argument("--target-pov", type=float, default=0.12, help="POV participation rate.")
    p_demo.set_defaults(func=cmd_demo)

    p_sens = sub.add_parser("sensitivity", help="Run liquidity and urgency sensitivity analysis.")
    p_sens.add_argument("--scenario", required=True, help="Path to base scenario JSON file.")
    p_sens.add_argument("--grid", default=None, help="Optional sensitivity grid JSON config.")
    p_sens.add_argument("--output-dir", default="reports", help="Directory for output artifacts.")
    p_sens.add_argument("--mc-paths", type=int, default=None, help="Override MC paths per grid point.")
    p_sens.set_defaults(func=cmd_sensitivity)

    p_cmp = sub.add_parser("compare", help="Compare analytic vs numeric Almgren-Chriss schedules.")
    p_cmp.add_argument("--scenario", required=True, help="Path to scenario JSON file.")
    p_cmp.add_argument("--lambdas", default="0.0,1e-7,1e-6,1e-5", help="Comma-separated lambda values.")
    p_cmp.add_argument("--tolerance", type=float, default=1e-4, help="Relative L2 tolerance.")
    p_cmp.add_argument("--enforce-capacity", action="store_true", help="Apply capacity bounds to numeric solver.")
    p_cmp.add_argument("--output", default=None, help="Optional JSON output path.")
    p_cmp.set_defaults(func=cmd_compare)

    p_val = sub.add_parser("validate", help="Validate a scenario JSON contract.")
    p_val.add_argument("--scenario", required=True, help="Path to scenario JSON file.")
    p_val.set_defaults(func=cmd_validate)

    p_rep = sub.add_parser("replay", help="Replay a previous run from its manifest.")
    p_rep.add_argument("--manifest", required=True, help="Path to replay manifest JSON.")
    p_rep.set_defaults(func=cmd_replay)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ExecutionEngineError, FileNotFoundError) as exc:
        print(json.dumps({"error": type(exc).__name__, "detail": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
