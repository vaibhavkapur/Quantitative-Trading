"""Operational resilience and CLI tests (missing config, malformed inputs, replay CLI)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cross_asset_factor_engine.cli import main as cli_main
from cross_asset_factor_engine.config import load_config
from cross_asset_factor_engine.errors import ConfigurationError


def test_missing_config_raises_configuration_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError) as exc_info:
        load_config(tmp_path / "nonexistent.yaml")
    assert "not found" in str(exc_info.value)


def test_malformed_and_inconsistent_configs_rejected(tmp_path: Path) -> None:
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("schema_version: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        load_config(bad_yaml)

    bad_vol = tmp_path / "bad_vol.yaml"
    bad_vol.write_text(
        """
schema_version: "1.0.0"
portfolio:
  target_volatility: 0.20
  volatility_cap: 0.10
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError) as exc_info:
        load_config(bad_vol)
    assert "volatility_cap" in str(exc_info.value)

    zero_lag = tmp_path / "zero_lag.yaml"
    zero_lag.write_text(
        """
schema_version: "1.0.0"
portfolio:
  execution_lag_days: 0
""",
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError):
        load_config(zero_lag)


def test_cli_validate_run_and_verify_replay(
    project_root: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # 1. validate command succeeds on valid_small
    rc_val = cli_main(["validate", "--dataset", str(project_root / "data/fixtures/valid_small")])
    assert rc_val == 0
    val_out = json.loads(capsys.readouterr().out)
    assert val_out["status"] == "VALID"
    assert val_out["n_assets"] == 4

    # 2. run command succeeds on reference_fixture.yaml
    rc_run = cli_main(
        [
            "run",
            "--config",
            str(project_root / "config/reference_fixture.yaml"),
            "--output-dir",
            str(tmp_path / "reports"),
            "--project-root",
            str(project_root),
        ]
    )
    assert rc_run == 0
    run_out = json.loads(capsys.readouterr().out)
    assert run_out["status"] == "OK"
    assert Path(run_out["report_dir"]).exists()

    # 3. verify-replay command succeeds
    rc_rep = cli_main(
        [
            "verify-replay",
            "--config",
            str(project_root / "config/reference_fixture.yaml"),
            "--project-root",
            str(project_root),
        ]
    )
    assert rc_rep == 0
    rep_out = json.loads(capsys.readouterr().out)
    assert rep_out["equivalent"] is True

    # 4. Missing config fails cleanly with exit code 1 and structured JSON error (no silent success)
    rc_err = cli_main(["run", "--config", str(tmp_path / "missing.yaml")])
    assert rc_err == 1
    err_out = json.loads(capsys.readouterr().err)
    assert err_out["status"] == "ERROR"
    assert err_out["error"]["code"] == "CONFIGURATION_ERROR"
