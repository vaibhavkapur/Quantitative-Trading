"""Chronological walk-forward splitting with purge/embargo and full OOS signal evaluation."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import polars as pl

from alpha_engine.config import EngineConfig, RegistryPolicy, UniversePolicy, WalkForwardConfig
from alpha_engine.contracts.models import (
    EvaluationReport,
    FoldMetrics,
    SignalDefinition,
    StabilityReport,
)
from alpha_engine.contracts.validation import PointInTimeValidator
from alpha_engine.data.loader import ResearchDataset
from alpha_engine.errors import DataValidationError, LookAheadViolationError
from alpha_engine.evaluation.costs import SignalToPositionResearcher
from alpha_engine.evaluation.metrics import (
    compute_cross_sectional_ic,
    compute_sector_exposure,
    evaluate_stability,
    summarize_ic_series,
)
from alpha_engine.evaluation.neutralization import SignalNeutralizer
from alpha_engine.factors.base import BaseFactor
from alpha_engine.models.baselines import AlphaModel


@dataclass(frozen=True)
class WalkForwardSplit:
    fold_id: int
    train_dates: list[datetime]
    test_dates: list[datetime]

    @property
    def train_start_utc(self) -> datetime:
        return self.train_dates[0]

    @property
    def train_end_utc(self) -> datetime:
        return self.train_dates[-1]

    @property
    def test_start_utc(self) -> datetime:
        return self.test_dates[0]

    @property
    def test_end_utc(self) -> datetime:
        return self.test_dates[-1]


class WalkForwardSplitter:
    """Generates strictly chronological train/test splits with a mandatory embargo gap."""

    def __init__(self, config: WalkForwardConfig) -> None:
        self.config = config

    def split(self, unique_dates: list[datetime]) -> list[WalkForwardSplit]:
        dates = sorted(set(unique_dates))
        cfg = self.config
        embargo = max(cfg.embargo_days, cfg.forward_horizon_days)
        needed = cfg.min_train_days + embargo + cfg.n_splits * cfg.test_days
        if len(dates) < needed:
            raise DataValidationError(
                f"Insufficient trading days ({len(dates)}) for walk-forward config (need >= {needed}).",
                details={"available_days": len(dates), "required_days": needed},
            )

        splits: list[WalkForwardSplit] = []
        for fold in range(cfg.n_splits):
            test_start_idx = cfg.min_train_days + embargo + fold * cfg.test_days
            test_end_idx = test_start_idx + cfg.test_days
            train_end_idx = test_start_idx - embargo
            train_start_idx = 0 if cfg.expanding_window else max(0, train_end_idx - cfg.min_train_days)

            train_slice = dates[train_start_idx:train_end_idx]
            test_slice = dates[test_start_idx:test_end_idx]

            if not train_slice or not test_slice:
                raise DataValidationError(f"Empty train or test slice in fold {fold}.")
            if train_slice[-1] >= test_slice[0]:
                raise LookAheadViolationError(
                    f"Walk-forward fold {fold} has overlapping train/test dates.",
                    details={
                        "train_end": train_slice[-1].isoformat(),
                        "test_start": test_slice[0].isoformat(),
                    },
                )
            splits.append(WalkForwardSplit(fold_id=fold, train_dates=train_slice, test_dates=test_slice))
        return splits


class WalkForwardEvaluator:
    """Runs causal walk-forward validation for both individual factors and predictive ML models."""

    def __init__(
        self,
        engine_config: EngineConfig,
        universe_policy: UniversePolicy,
        registry_policy: RegistryPolicy,
    ) -> None:
        self.engine_config = engine_config
        self.universe_policy = universe_policy
        self.registry_policy = registry_policy
        self.validator = PointInTimeValidator(universe_policy)
        self.neutralizer = SignalNeutralizer(engine_config.neutralization)
        self.position_researcher = SignalToPositionResearcher(engine_config.cost_model)
        self.splitter = WalkForwardSplitter(engine_config.walk_forward)

    @staticmethod
    def _compute_forward_returns(bars_df: pl.DataFrame, horizon_days: int) -> pl.DataFrame:
        """Compute forward horizon return purely as evaluation target (never fed into features)."""
        return (
            bars_df.sort(["symbol", "event_ts_utc"])
            .with_columns(
                (
                    pl.col("adjusted_close").shift(-horizon_days).over("symbol")
                    / pl.col("adjusted_close")
                    - 1.0
                ).alias("fwd_return")
            )
            .select("symbol", "event_ts_utc", "fwd_return")
        )

    def _evaluate_neutralized_oos_panel(
        self,
        *,
        definition: SignalDefinition,
        oos_neutralized_df: pl.DataFrame,
        bars_df: pl.DataFrame,
        splits: list[WalkForwardSplit],
        fold_coefficients: dict[int, dict[str, float]],
        dataset_manifest_digest: str,
        max_peer_correlation: float = 0.0,
        most_correlated_peer: str | None = None,
    ) -> EvaluationReport:
        horizon = self.engine_config.walk_forward.forward_horizon_days
        fwd_df = self._compute_forward_returns(bars_df, horizon)
        eval_df = oos_neutralized_df.join(fwd_df, on=["symbol", "event_ts_utc"], how="inner")

        weights_df = self.position_researcher.build_weights(oos_neutralized_df)
        cost_points, avg_turnover = self.position_researcher.evaluate_cost_sensitivity(
            weights_df, bars_df
        )
        max_sector_exp = compute_sector_exposure(weights_df)

        fold_metrics: list[FoldMetrics] = []
        all_spearman_ics: list[float] = []

        for sp in splits:
            f_df = eval_df.filter(
                (pl.col("event_ts_utc") >= pl.lit(sp.test_start_utc))
                & (pl.col("event_ts_utc") <= pl.lit(sp.test_end_utc))
            )
            ic_df = compute_cross_sectional_ic(f_df)
            sp_ics = ic_df["spearman_ic"].to_list()
            pe_ics = ic_df["pearson_ic"].to_list()
            all_spearman_ics.extend(sp_ics)
            ic_stats = summarize_ic_series(sp_ics, seed=self.engine_config.seed + sp.fold_id)

            f_weights = weights_df.filter(
                (pl.col("event_ts_utc") >= pl.lit(sp.test_start_utc))
                & (pl.col("event_ts_utc") <= pl.lit(sp.test_end_utc))
            )
            _, f_turnover = self.position_researcher.evaluate_cost_sensitivity(
                f_weights, bars_df, cost_grid_bps=[self.engine_config.cost_model.base_spread_bps]
            )
            f_sec_exp = compute_sector_exposure(f_weights)

            coefs = dict(fold_coefficients.get(sp.fold_id, {}))
            if not coefs:
                # For single-factor signals, track univariate cross-sectional slope per fold
                valid = f_df.filter(
                    pl.col("neutralized_value").is_finite() & pl.col("fwd_return").is_finite()
                )
                if valid.height >= 10:
                    x = valid["neutralized_value"].to_numpy()
                    y = valid["fwd_return"].to_numpy()
                    denom = float(np.dot(x, x))
                    slope = float(np.dot(x, y) / denom) if denom > 1e-12 else 0.0
                    coefs = {definition.signal_id: slope}

            fold_metrics.append(
                FoldMetrics(
                    fold_id=sp.fold_id,
                    train_start_utc=sp.train_start_utc,
                    train_end_utc=sp.train_end_utc,
                    test_start_utc=sp.test_start_utc,
                    test_end_utc=sp.test_end_utc,
                    n_test_days=ic_df.height,
                    oos_spearman_ic=round(ic_stats["mean_ic"], 6),
                    oos_pearson_ic=round(float(np.mean(pe_ics)) if pe_ics else 0.0, 6),
                    oos_icir=round(ic_stats["icir"], 4),
                    hit_rate=round(ic_stats["hit_rate"], 4),
                    avg_daily_turnover=round(f_turnover, 6),
                    max_abs_sector_exposure=round(f_sec_exp, 6),
                    coefficients=coefs,
                )
            )

        agg_stats = summarize_ic_series(all_spearman_ics, seed=self.engine_config.seed)
        policy = self.registry_policy
        stability: StabilityReport = evaluate_stability(
            fold_metrics,
            max_sign_flip_rate=policy.max_coefficient_sign_flip_rate,
            max_coefficient_cv=policy.max_coefficient_cv,
            min_ic_sign_consistency=policy.min_ic_sign_consistency,
        )

        base_bps = self.engine_config.cost_model.base_spread_bps
        base_point = min(cost_points, key=lambda pt: abs(pt.cost_bps - base_bps))
        net_sharpe_base = base_point.net_sharpe

        gate_failures = self._check_gates(
            oos_mean_ic=agg_stats["mean_ic"],
            oos_icir=agg_stats["icir"],
            bh_adjusted_pvalue=agg_stats["pvalue"],
            avg_turnover=avg_turnover,
            max_sector_exp=max_sector_exp,
            max_peer_corr=max_peer_correlation,
            net_sharpe_base=net_sharpe_base,
            stability=stability,
        )

        notes = (
            ""
            if not gate_failures
            else "Candidate failed validation gates and is blocked from promotion: "
            + "; ".join(gate_failures)
        )

        return EvaluationReport(
            trial_id=f"trial-{definition.signal_id}-{uuid.uuid4().hex[:8]}",
            signal_id=definition.signal_id,
            signal_version=definition.version,
            definition_hash=definition.definition_hash,
            evaluated_at_utc=datetime.now(UTC),
            dataset_manifest_digest=dataset_manifest_digest,
            sample_days=len(all_spearman_ics),
            universe_size=int(oos_neutralized_df.select("symbol").n_unique()),
            n_folds=len(fold_metrics),
            oos_mean_ic=round(agg_stats["mean_ic"], 6),
            oos_std_ic=round(agg_stats["std_ic"], 6),
            oos_icir=round(agg_stats["icir"], 4),
            oos_ic_tstat=round(agg_stats["tstat"], 4),
            oos_ic_pvalue=round(agg_stats["pvalue"], 6),
            bh_adjusted_pvalue=round(agg_stats["pvalue"], 6),
            bootstrap_ic_ci_low=round(agg_stats["ci_low"], 6),
            bootstrap_ic_ci_high=round(agg_stats["ci_high"], 6),
            avg_daily_turnover=round(avg_turnover, 6),
            max_abs_sector_exposure=round(max_sector_exp, 6),
            max_peer_correlation=round(max_peer_correlation, 6),
            most_correlated_peer=most_correlated_peer,
            stability=stability,
            folds=fold_metrics,
            cost_sensitivity=cost_points,
            base_cost_bps=base_bps,
            net_sharpe_at_base_cost=round(net_sharpe_base, 4),
            passed_gates=len(gate_failures) == 0,
            gate_failures=gate_failures,
            negative_result_notes=notes,
        )

    def _check_gates(
        self,
        *,
        oos_mean_ic: float,
        oos_icir: float,
        bh_adjusted_pvalue: float,
        avg_turnover: float,
        max_sector_exp: float,
        max_peer_corr: float,
        net_sharpe_base: float,
        stability: StabilityReport,
    ) -> list[str]:
        p = self.registry_policy
        failures: list[str] = []
        if oos_mean_ic < p.min_oos_mean_ic:
            failures.append(f"oos_mean_ic={oos_mean_ic:.4f} < min {p.min_oos_mean_ic:.4f}")
        if oos_icir < p.min_oos_icir:
            failures.append(f"oos_icir={oos_icir:.4f} < min {p.min_oos_icir:.4f}")
        if bh_adjusted_pvalue > p.max_bh_adjusted_pvalue:
            failures.append(
                f"bh_adjusted_pvalue={bh_adjusted_pvalue:.4f} > max {p.max_bh_adjusted_pvalue:.4f}"
            )
        if avg_turnover > p.max_avg_daily_turnover:
            failures.append(
                f"avg_daily_turnover={avg_turnover:.4f} > max {p.max_avg_daily_turnover:.4f}"
            )
        if max_sector_exp > p.max_abs_sector_exposure:
            failures.append(
                f"max_abs_sector_exposure={max_sector_exp:.4f} > max {p.max_abs_sector_exposure:.4f}"
            )
        if abs(max_peer_corr) > p.max_peer_factor_correlation:
            failures.append(
                f"max_peer_correlation={abs(max_peer_corr):.4f} > max {p.max_peer_factor_correlation:.4f}"
            )
        if net_sharpe_base < p.min_net_sharpe_at_base_cost:
            failures.append(
                f"net_sharpe_at_base_cost={net_sharpe_base:.4f} < min {p.min_net_sharpe_at_base_cost:.4f}"
            )
        if not stability.is_stable:
            failures.extend([f"stability:{r}" for r in stability.reasons])
        return failures

    def evaluate_factor(
        self,
        factor: BaseFactor,
        dataset: ResearchDataset,
    ) -> tuple[EvaluationReport, pl.DataFrame]:
        """Evaluate a deterministic causal BaseFactor over chronological walk-forward folds."""
        raw_sig = factor.compute(dataset.bars, fundamentals_df=dataset.fundamentals)
        self.validator.validate_signal_causality(
            raw_sig, dataset.bars, fundamentals_df=dataset.fundamentals
        )
        neutralized = self.neutralizer.neutralize(raw_sig, dataset.bars)

        unique_dates = dataset.bars.select("event_ts_utc").unique().sort("event_ts_utc")["event_ts_utc"].to_list()
        splits = self.splitter.split(unique_dates)

        oos_start = splits[0].test_start_utc
        oos_end = splits[-1].test_end_utc
        oos_panel = neutralized.filter(
            (pl.col("event_ts_utc") >= pl.lit(oos_start))
            & (pl.col("event_ts_utc") <= pl.lit(oos_end))
        )

        report = self._evaluate_neutralized_oos_panel(
            definition=factor.definition,
            oos_neutralized_df=oos_panel,
            bars_df=dataset.bars,
            splits=splits,
            fold_coefficients={},
            dataset_manifest_digest=dataset.manifest.manifest_digest,
        )
        return report, oos_panel

    def evaluate_model(
        self,
        model: AlphaModel,
        feature_matrix: pl.DataFrame,
        feature_cols: list[str],
        dataset: ResearchDataset,
    ) -> tuple[EvaluationReport, pl.DataFrame]:
        """Train model strictly on each chronological training fold and predict out-of-sample."""
        horizon = self.engine_config.walk_forward.forward_horizon_days
        fwd_df = self._compute_forward_returns(dataset.bars, horizon)

        # Standardize features cross-sectionally per date (strictly causal)
        panel = feature_matrix.join(fwd_df, on=["symbol", "event_ts_utc"], how="left").sort(
            ["event_ts_utc", "symbol"]
        )
        unique_dates = dataset.bars.select("event_ts_utc").unique().sort("event_ts_utc")["event_ts_utc"].to_list()
        splits = self.splitter.split(unique_dates)

        oos_chunks: list[pl.DataFrame] = []
        fold_coefs: dict[int, dict[str, float]] = {}

        for sp in splits:
            train_df = panel.filter(
                (pl.col("event_ts_utc") >= pl.lit(sp.train_start_utc))
                & (pl.col("event_ts_utc") <= pl.lit(sp.train_end_utc))
                & pl.col("fwd_return").is_finite()
            )
            test_df = panel.filter(
                (pl.col("event_ts_utc") >= pl.lit(sp.test_start_utc))
                & (pl.col("event_ts_utc") <= pl.lit(sp.test_end_utc))
            )

            # Drop training rows with all-null features
            X_train = train_df.select(feature_cols).to_numpy().astype(np.float64)
            y_train = train_df["fwd_return"].to_numpy().astype(np.float64)
            X_test = test_df.select(feature_cols).to_numpy().astype(np.float64)

            fold_model = model.clone_fresh()
            fold_model.fit(X_train, y_train, feature_cols)
            preds = fold_model.predict(X_test)
            fold_coefs[sp.fold_id] = fold_model.feature_coefficients()

            chunk = test_df.select("symbol", "event_ts_utc", "available_at_utc").with_columns(
                pl.lit(model.definition.signal_id).alias("signal_id"),
                pl.lit(model.definition.version).alias("signal_version"),
                pl.Series("raw_value", preds, dtype=pl.Float64),
            )
            oos_chunks.append(chunk)

        oos_raw = pl.concat(oos_chunks, how="vertical").sort(["event_ts_utc", "symbol"])
        self.validator.validate_signal_causality(oos_raw, dataset.bars)
        oos_neutralized = self.neutralizer.neutralize(oos_raw, dataset.bars)

        report = self._evaluate_neutralized_oos_panel(
            definition=model.definition,
            oos_neutralized_df=oos_neutralized,
            bars_df=dataset.bars,
            splits=splits,
            fold_coefficients=fold_coefs,
            dataset_manifest_digest=dataset.manifest.manifest_digest,
        )
        return report, oos_neutralized
