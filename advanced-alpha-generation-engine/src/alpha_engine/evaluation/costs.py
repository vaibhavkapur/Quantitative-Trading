"""Cost-aware signal-to-position research, turnover smoothing, and cost sensitivity sweeps."""

from __future__ import annotations

import numpy as np
import polars as pl

from alpha_engine.config import CostModelConfig
from alpha_engine.contracts.models import CostSensitivityPoint


class SignalToPositionResearcher:
    """Translates neutralized signals into dollar-neutral, capped, turnover-aware weights and evaluates net PnL."""

    def __init__(self, config: CostModelConfig) -> None:
        self.config = config

    def build_weights(
        self,
        neutralized_signal_df: pl.DataFrame,
        *,
        signal_col: str = "neutralized_value",
        apply_smoothing: bool = True,
    ) -> pl.DataFrame:
        """Map daily neutralized signals into dollar-neutral capped portfolio weights with turnover tracking."""
        ordered = neutralized_signal_df.filter(pl.col(signal_col).is_finite()).sort(
            ["event_ts_utc", "symbol"]
        )
        if ordered.height == 0:
            return pl.DataFrame(
                schema={
                    "event_ts_utc": pl.Datetime("us", "UTC"),
                    "available_at_utc": pl.Datetime("us", "UTC"),
                    "symbol": pl.String,
                    "sector": pl.String,
                    "weight": pl.Float64,
                    "prev_weight": pl.Float64,
                    "delta_weight": pl.Float64,
                }
            )

        cfg = self.config
        prev_weights: dict[str, float] = {}
        weight_rows: list[dict] = []

        for (dt,), grp in ordered.group_by(["event_ts_utc"], maintain_order=True):
            syms = grp["symbol"].to_list()
            scores = grp[signal_col].to_numpy().astype(np.float64)
            sectors = grp["sector"].to_list() if "sector" in grp.columns else ["ALL"] * grp.height
            avails = grp["available_at_utc"].to_list()

            # Step 1: Dollar-neutral raw target weights with gross exposure = 1.0
            centered = scores - float(np.mean(scores))
            gross = float(np.sum(np.abs(centered)))
            if gross > 1e-12:
                target = centered / gross
            else:
                target = np.zeros_like(centered)

            # Clip single-name weight and re-center / re-normalize
            target = np.clip(target, -cfg.max_single_weight, cfg.max_single_weight)
            target = target - float(np.mean(target))
            gross_clipped = float(np.sum(np.abs(target)))
            if gross_clipped > 1e-12:
                target = target / gross_clipped

            target_map = {s: float(w) for s, w in zip(syms, target, strict=True)}

            # Step 2: Turnover smoothing + no-trade buffer for continuing symbols
            cur_weights: dict[str, float] = {}
            alpha = cfg.turnover_smoothing_alpha if apply_smoothing else 1.0
            buf = cfg.no_trade_buffer if apply_smoothing else 0.0

            for s in syms:
                w_prev = prev_weights.get(s, 0.0)
                w_tgt = target_map[s]
                if not prev_weights or not apply_smoothing:
                    cur_weights[s] = w_tgt
                elif abs(w_tgt - w_prev) < buf:
                    cur_weights[s] = w_prev
                else:
                    cur_weights[s] = alpha * w_tgt + (1.0 - alpha) * w_prev

            # Re-center slightly to maintain strict dollar neutrality after smoothing
            if cur_weights:
                mean_w = float(np.mean(list(cur_weights.values())))
                for s in syms:
                    cur_weights[s] -= mean_w
                gross_now = float(sum(abs(v) for v in cur_weights.values()))
                if gross_now > 1e-12:
                    for s in syms:
                        cur_weights[s] /= gross_now

            # Emit rows for active symbols
            for s, sec, av in zip(syms, sectors, avails, strict=True):
                w_prev = prev_weights.get(s, 0.0)
                w_now = cur_weights[s]
                weight_rows.append(
                    {
                        "event_ts_utc": dt,
                        "available_at_utc": av,
                        "symbol": s,
                        "sector": sec,
                        "weight": w_now,
                        "prev_weight": w_prev,
                        "delta_weight": w_now - w_prev,
                    }
                )

            # Account for liquidated positions (e.g. delisted/exited symbols) so turnover is never undercounted!
            active_set = set(syms)
            for exited_sym, old_w in prev_weights.items():
                if exited_sym not in active_set and abs(old_w) > 1e-12:
                    weight_rows.append(
                        {
                            "event_ts_utc": dt,
                            "available_at_utc": avails[0],
                            "symbol": exited_sym,
                            "sector": "EXITED",
                            "weight": 0.0,
                            "prev_weight": old_w,
                            "delta_weight": -old_w,
                        }
                    )

            prev_weights = cur_weights

        return pl.DataFrame(weight_rows).sort(["event_ts_utc", "symbol"])

    def evaluate_cost_sensitivity(
        self,
        weights_df: pl.DataFrame,
        bars_df: pl.DataFrame,
        *,
        cost_grid_bps: list[float] | None = None,
    ) -> tuple[list[CostSensitivityPoint], float]:
        """Evaluate gross & net PnL, Sharpe, drawdown, and turnover across a grid of trading costs."""
        grid = cost_grid_bps if cost_grid_bps is not None else self.config.cost_grid_bps
        if weights_df.height == 0:
            empty_points = [
                CostSensitivityPoint(
                    cost_bps=bps,
                    gross_annualized_return=0.0,
                    net_annualized_return=0.0,
                    gross_sharpe=0.0,
                    net_sharpe=0.0,
                    avg_daily_turnover=0.0,
                    total_cost_drag_bps=0.0,
                    max_drawdown=0.0,
                )
                for bps in grid
            ]
            return empty_points, 0.0

        # Compute next-day return realized strictly AFTER signal formation at t: ret_{t->t+1}
        next_rets = (
            bars_df.sort(["symbol", "event_ts_utc"])
            .with_columns(
                (
                    pl.col("adjusted_close").shift(-1).over("symbol") / pl.col("adjusted_close")
                    - 1.0
                ).alias("next_1d_return")
            )
            .select("symbol", "event_ts_utc", "next_1d_return")
        )

        joined = weights_df.join(next_rets, on=["symbol", "event_ts_utc"], how="left").with_columns(
            pl.col("next_1d_return").fill_null(0.0)
        )

        daily = (
            joined.group_by("event_ts_utc")
            .agg(
                (pl.col("weight") * pl.col("next_1d_return")).sum().alias("gross_ret"),
                (0.5 * pl.col("delta_weight").abs().sum()).alias("one_way_turnover"),
                (pl.col("delta_weight").abs().pow(1.5).sum()).alias("impact_term"),
            )
            .sort("event_ts_utc")
        )

        # Exclude initial portfolio ramp day from steady-state average turnover when n > 5
        turnover_series = daily["one_way_turnover"].to_numpy()
        steady_turnover = (
            float(np.mean(turnover_series[1:])) if len(turnover_series) > 5 else float(np.mean(turnover_series))
        )
        gross_rets = daily["gross_ret"].to_numpy()
        impact_terms = daily["impact_term"].to_numpy()
        ann = float(self.config.annualization_days)

        gross_ann_ret = float(np.mean(gross_rets) * ann)
        gross_vol = float(np.std(gross_rets, ddof=1) * np.sqrt(ann)) if len(gross_rets) > 1 else 0.0
        gross_sharpe = gross_ann_ret / gross_vol if gross_vol > 1e-12 else 0.0

        points: list[CostSensitivityPoint] = []
        for bps in grid:
            # Linear spread + convex impact scaling linearly with cost grid relative to base spread
            impact_scale = (bps / max(self.config.base_spread_bps, 1.0)) if bps > 0 else 0.0
            daily_cost = (
                turnover_series * (bps / 10_000.0)
                + impact_terms * (self.config.impact_coeff_bps * impact_scale / 10_000.0)
            )
            net_rets = gross_rets - daily_cost
            net_ann_ret = float(np.mean(net_rets) * ann)
            net_vol = float(np.std(net_rets, ddof=1) * np.sqrt(ann)) if len(net_rets) > 1 else 0.0
            net_sharpe = net_ann_ret / net_vol if net_vol > 1e-12 else 0.0

            cum = np.cumprod(1.0 + net_rets)
            running_max = np.maximum.accumulate(cum)
            drawdowns = (cum / np.maximum(running_max, 1e-12)) - 1.0
            max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

            points.append(
                CostSensitivityPoint(
                    cost_bps=float(bps),
                    gross_annualized_return=round(gross_ann_ret, 6),
                    net_annualized_return=round(net_ann_ret, 6),
                    gross_sharpe=round(gross_sharpe, 4),
                    net_sharpe=round(net_sharpe, 4),
                    avg_daily_turnover=round(steady_turnover, 6),
                    total_cost_drag_bps=round(float(np.sum(daily_cost) * 10_000.0), 2),
                    max_drawdown=round(max_dd, 6),
                )
            )

        return points, steady_turnover
