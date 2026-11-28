"""Boundary, volatility-shock, and operational resilience tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from volatility_forecasting.cli import main as cli_main
from volatility_forecasting.config import (
    DataConfig,
    EngineConfig,
    EvaluationConfig,
    FeatureConfig,
    ModelSpec,
    OutputConfig,
    SplitConfig,
    load_config,
)
from volatility_forecasting.errors import ConfigurationError, UnsupportedOptionError
from volatility_forecasting.features import build_causal_dataset
from volatility_forecasting.fixtures import generate_multi_regime_fixture
from volatility_forecasting.pipeline import evaluate_models_chronologically


def test_volatility_shock_stability_and_regime_classification() -> None:
    """Inject an extreme 8x volatility shock in the OOS test window and verify stability."""
    ret_df, bar_df = generate_multi_regime_fixture(
        n_days=260, bars_per_day=13, seed=99, shock_day_index=205
    )
    # Amplify the shock window dramatically to simulate an extreme market stress event
    shock_dates = set(ret_df["session_date"][205:215].to_list())
    ret_shocked = ret_df.with_columns(
        pl.when(pl.col("session_date").is_in(list(shock_dates)))
        .then(pl.col("log_return") * 3.5)
        .otherwise(pl.col("log_return"))
        .alias("log_return")
    )

    # Use daily returns so the amplified shock hits daily_rv directly
    dataset, _ = build_causal_dataset(ret_shocked, horizons=(1, 5))

    cfg = EngineConfig(
        version="1.0",
        run_name="shock_boundary",
        seed=42,
        annualization_factor=252,
        expected_bars_per_session=13,
        data=DataConfig(returns_path="unused.csv"),
        features=FeatureConfig(horizons=(1, 5), primary_horizon=5),
        splits=SplitConfig(scheme="expanding", n_splits=1, min_train_size=180, test_size=60),
        models=(
            ModelSpec(name="naive", type="naive"),
            ModelSpec(name="rolling_22d", type="rolling_historical", params={"window": 22}),
            ModelSpec(name="ewma", type="ewma", params={"decay": 0.94}),
            ModelSpec(name="garch11", type="garch11", params={"backend": "scipy"}),
            ModelSpec(name="ridge_har", type="ridge_har", params={"alpha": 1.0}),
            ModelSpec(name="hgbr", type="hgbr", params={"max_iter": 50}),
        ),
        evaluation=EvaluationConfig(strict_nonnegative=True),
        output=OutputConfig(reports_dir="reports"),
    )

    _, comp_df, intervals_df, regime_df, _, _ = evaluate_models_chronologically(
        dataset, cfg, horizon_days=5
    )

    # All predictions finite, strictly nonnegative, no numerical explosion
    preds = intervals_df["pred_var"].to_numpy()
    assert np.all(np.isfinite(preds))
    assert np.all(preds >= 0.0)
    assert np.all(comp_df["qlike"].to_numpy() >= 0.0)
    assert np.all(np.isfinite(comp_df["qlike"].to_numpy()))

    # Shock regime must be detected and reported in regime diagnostics
    regimes_present = set(regime_df["regime"].to_list())
    assert "shock" in regimes_present

    # EWMA / GARCH forecasts during shock regime should exceed their low/normal forecasts
    ewma_reg = regime_df.filter(pl.col("model") == "ewma")
    shock_vol = ewma_reg.filter(pl.col("regime") == "shock")["mean_pred_vol"].item()
    normal_vol = ewma_reg.filter(pl.col("regime") == "normal")["mean_pred_vol"].item()
    assert shock_vol > normal_vol


def test_operational_missing_and_malformed_configs(tmp_path: Path) -> None:
    # Missing config file
    with pytest.raises(ConfigurationError, match="not found"):
        load_config(tmp_path / "does_not_exist.yaml")

    # Malformed YAML
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("models: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Invalid YAML"):
        load_config(bad_yaml)

    # Unsupported model type
    bad_model_cfg = tmp_path / "bad_model.yaml"
    bad_model_cfg.write_text(
        """
run_name: test
data:
  returns_path: x.csv
models:
  - name: mystery
    type: quantum_vol_net
""",
        encoding="utf-8",
    )
    with pytest.raises(UnsupportedOptionError, match="Unsupported model type"):
        load_config(bad_model_cfg)

    # Unsupported split scheme
    bad_split_cfg = tmp_path / "bad_split.yaml"
    bad_split_cfg.write_text(
        """
run_name: test
data:
  returns_path: x.csv
splits:
  scheme: random_kfold
models:
  - name: naive
    type: naive
""",
        encoding="utf-8",
    )
    with pytest.raises(UnsupportedOptionError, match="Unsupported split scheme"):
        load_config(bad_split_cfg)


def test_cli_end_to_end_and_error_reporting(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # 1. Generate fixtures via CLI
    fix_dir = tmp_path / "fixtures"
    rc = cli_main(["generate-fixtures", "--out-dir", str(fix_dir)])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "ok"

    # 2. Validate valid data via CLI
    rc_val = cli_main(
        [
            "validate-data",
            "--returns",
            str(fix_dir / "synthetic_multi_regime_returns.csv"),
            "--intraday",
            str(fix_dir / "synthetic_multi_regime_intraday.csv"),
        ]
    )
    assert rc_val == 0
    out_val = json.loads(capsys.readouterr().out)
    assert out_val["returns_rows"] == 420

    # 3. Validate invalid data via CLI -> explicit non-zero exit code + structured JSON error
    rc_bad = cli_main(
        ["validate-data", "--returns", str(fix_dir / "invalid_missing_column_returns.csv")]
    )
    assert rc_bad == 2
    err_payload = json.loads(capsys.readouterr().err)
    assert err_payload["status"] == "error"
    assert err_payload["error_type"] == "DataContractError"

    # 4. Missing config file via CLI -> exit code 2, no silent success
    rc_missing = cli_main(["run", "--config", str(tmp_path / "missing.yaml")])
    assert rc_missing == 2
    err_missing = json.loads(capsys.readouterr().err)
    assert err_missing["error_type"] == "ConfigurationError"
