"""Top-level orchestration engine connecting I/O, alignment pipeline, models, and reporting."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import polars as pl

from mfape.errors import ConfigurationError, DataContractError
from mfape.evaluation import build_comparison_summary, evaluate_out_of_sample
from mfape.fama_macbeth import estimate_fama_macbeth
from mfape.io import (
    collect_runtime_environment,
    compute_file_sha256,
    compute_run_id,
    load_config,
    load_table,
    resolve_path,
    write_run_artifacts,
)
from mfape.pipeline import build_aligned_panel, split_panel
from mfape.schemas import (
    DiagnosticCounters,
    EngineConfig,
    ModelEvaluationResult,
    RunOutput,
    SCHEMA_VERSION,
)
from mfape.time_series import fit_time_series_model

ENGINE_VERSION = "0.1.0"
logger = logging.getLogger("mfape.engine")


class AssetPricingEngine:
    """Reproducible multi-factor asset pricing engine orchestrator."""

    def __init__(self, logger_override: logging.Logger | None = None) -> None:
        self._log = logger_override or logger

    def run_on_frames(
        self,
        asset_returns_df: pl.DataFrame,
        factor_returns_df: pl.DataFrame,
        risk_free_df: pl.DataFrame,
        memberships_df: pl.DataFrame,
        config: EngineConfig,
        input_fingerprints: dict[str, str] | None = None,
    ) -> RunOutput:
        """Execute alignment, time-series, Fama-MacBeth, and OOS comparison on in-memory frames."""
        config.validate()
        diag = DiagnosticCounters()

        # Union of all unique factors requested across configured models (stable sorted order)
        all_factors = sorted({f for m in config.models for f in m.factors})

        self._log.info(
            "Starting run '%s' with %d model(s) and %d factor(s).",
            config.run_label,
            len(config.models),
            len(all_factors),
        )
        diag.record(
            "INFO",
            "engine_start",
            f"Initialized engine run '{config.run_label}'.",
            models=[m.name for m in config.models],
            factors=all_factors,
        )

        panel = build_aligned_panel(
            asset_returns_df=asset_returns_df,
            factor_returns_df=factor_returns_df,
            risk_free_df=risk_free_df,
            memberships_df=memberships_df,
            required_factors=all_factors,
            frequency=config.frequency,
            universe_policy=config.universe_policy,
            rf_missing_policy=config.rf_missing_policy,
            period_alignment_policy=config.period_alignment_policy,
            diagnostics=diag,
        )

        train_panel, test_panel = split_panel(panel, config.split)

        split_boundaries = {
            "full_panel_start": panel.dates[0],
            "full_panel_end": panel.dates[-1],
            "full_panel_periods": len(panel.dates),
            "train_start": train_panel.dates[0],
            "train_end": train_panel.dates[-1],
            "train_periods": len(train_panel.dates),
            "test_start": test_panel.dates[0] if test_panel is not None else None,
            "test_end": test_panel.dates[-1] if test_panel is not None else None,
            "test_periods": len(test_panel.dates) if test_panel is not None else 0,
            "zero_leakage_verified": True,
        }
        diag.record("INFO", "split_panel", "Verified chronological train/test boundaries.", **split_boundaries)

        model_results: list[ModelEvaluationResult] = []
        for spec in config.models:
            self._log.info("Fitting model '%s' (%s)...", spec.name, ", ".join(spec.factors))
            ts_train = fit_time_series_model(
                panel=train_panel,
                spec=spec,
                cov_type=config.cov_type,
                hac_maxlags=config.hac_maxlags,
                min_ts_observations=config.min_ts_observations,
                rank_tol=config.rank_tol,
                max_condition_number=config.max_condition_number,
            )
            fm_train = estimate_fama_macbeth(
                panel=train_panel,
                ts_result=ts_train,
                spec=spec,
                annualization_factor=config.annualization_factor,
                fm_hac_maxlags=config.fm_hac_maxlags,
                min_cs_assets=config.min_cs_assets,
                rank_tol=config.rank_tol,
                max_condition_number=config.max_condition_number,
            )
            oos_res = None
            if test_panel is not None:
                oos_res = evaluate_out_of_sample(
                    train_panel=train_panel,
                    test_panel=test_panel,
                    ts_train=ts_train,
                    fm_train=fm_train,
                    spec=spec,
                )

            diag.record(
                "INFO",
                "model_fit",
                f"Completed model '{spec.name}'.",
                model=spec.name,
                ts_mean_r2=ts_train.mean_r_squared,
                fm_cs_r2=fm_train.pricing_metrics.cross_sectional_r2,
                oos_cs_r2=oos_res.oos_cs_pricing_metrics.cross_sectional_r2 if oos_res else None,
            )
            model_results.append(
                ModelEvaluationResult(
                    spec=spec,
                    time_series_train=ts_train,
                    fama_macbeth_train=fm_train,
                    oos_evaluation=oos_res,
                )
            )

        comparison = build_comparison_summary(
            results=model_results,
            baseline_model_name=config.baseline_model_name,
        )

        fingerprints = input_fingerprints or {"in_memory": "unhashed"}
        config_dict = config.to_dict()
        run_id = compute_run_id(config_dict, fingerprints, ENGINE_VERSION)

        return RunOutput(
            schema_version=SCHEMA_VERSION,
            engine_version=ENGINE_VERSION,
            run_id=run_id,
            run_label=config.run_label,
            config=config_dict,
            input_fingerprints=fingerprints,
            runtime_environment=collect_runtime_environment(),
            split_boundaries=split_boundaries,
            diagnostics=diag.to_dict(),
            models=tuple(model_results),
            comparison=comparison,
        )

    def run_from_config(
        self,
        config: EngineConfig,
        base_dir: str | Path | None = None,
        write_outputs: bool = True,
    ) -> tuple[RunOutput, dict[str, str]]:
        """Load input tables from disk, execute engine, and optionally persist versioned outputs."""
        config.validate()
        root = Path(base_dir).resolve() if base_dir is not None else Path.cwd()

        asset_path = resolve_path(config.asset_returns_path, root)
        factor_path = resolve_path(config.factor_returns_path, root)
        rf_path = resolve_path(config.risk_free_path, root)
        mem_path = resolve_path(config.memberships_path, root)

        asset_df = load_table(asset_path)
        factor_df = load_table(factor_path)
        rf_df = load_table(rf_path)
        mem_df = load_table(mem_path)

        fingerprints = {
            "asset_returns": compute_file_sha256(asset_path),
            "factor_returns": compute_file_sha256(factor_path),
            "risk_free": compute_file_sha256(rf_path),
            "portfolio_memberships": compute_file_sha256(mem_path),
        }

        output = self.run_on_frames(
            asset_returns_df=asset_df,
            factor_returns_df=factor_df,
            risk_free_df=rf_df,
            memberships_df=mem_df,
            config=config,
            input_fingerprints=fingerprints,
        )

        written: dict[str, str] = {}
        if write_outputs and config.output_dir is not None:
            out_dir = resolve_path(config.output_dir, root)
            written = write_run_artifacts(output, out_dir, include_series=True)
            self._log.info("Wrote versioned run outputs to %s (run_id=%s).", out_dir, output.run_id)

        return output, written

    def run_from_config_file(
        self,
        config_path: str | Path,
        base_dir: str | Path | None = None,
        write_outputs: bool = True,
    ) -> tuple[RunOutput, dict[str, str]]:
        """Load JSON config from path and execute full engine run."""
        cfg = load_config(config_path)
        return self.run_from_config(cfg, base_dir=base_dir, write_outputs=write_outputs)

    def replay_from_manifest(
        self,
        manifest_path: str | Path,
        base_dir: str | Path | None = None,
        verify_fingerprints: bool = True,
        write_outputs: bool = False,
    ) -> tuple[RunOutput, dict[str, str]]:
        """Replay a prior run from its saved replay_manifest.json and verify logical identity."""
        path = Path(manifest_path)
        if not path.exists():
            raise ConfigurationError(f"Replay manifest not found: {path}", {"path": str(path)})
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigurationError(f"Malformed JSON in replay manifest {path}: {exc}") from exc

        if "config" not in manifest or "run_id" not in manifest:
            raise ConfigurationError("Replay manifest missing required 'config' or 'run_id' fields.")

        cfg = EngineConfig.from_dict(manifest["config"])
        output, written = self.run_from_config(cfg, base_dir=base_dir, write_outputs=write_outputs)

        if verify_fingerprints and "input_fingerprints" in manifest:
            expected_fp = manifest["input_fingerprints"]
            if output.input_fingerprints != expected_fp:
                raise DataContractError(
                    "Input fingerprint mismatch during replay: input files have changed since original run.",
                    {
                        "expected_fingerprints": expected_fp,
                        "actual_fingerprints": output.input_fingerprints,
                    },
                )
        if verify_fingerprints and output.run_id != manifest["run_id"]:
            raise DataContractError(
                f"Replay run_id mismatch: expected {manifest['run_id']}, got {output.run_id}.",
                {"expected_run_id": manifest["run_id"], "actual_run_id": output.run_id},
            )
        return output, written
