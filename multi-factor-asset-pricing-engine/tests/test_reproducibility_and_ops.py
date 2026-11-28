"""Reproducibility, replay, CLI, and operational failure handling tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mfape.cli import main as cli_main
from mfape.engine import AssetPricingEngine
from mfape.errors import ConfigurationError, DataContractError, UnsupportedInputError
from mfape.io import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_reproducibility_and_replay_from_manifest(tmp_path: Path) -> None:
    cfg_raw = json.loads((PROJECT_ROOT / "config" / "default_config.json").read_text(encoding="utf-8"))
    cfg_raw["output_dir"] = str(tmp_path / "run1")
    cfg1_path = tmp_path / "cfg1.json"
    cfg1_path.write_text(json.dumps(cfg_raw), encoding="utf-8")

    cfg_raw_2 = dict(cfg_raw)
    cfg_raw_2["output_dir"] = str(tmp_path / "run2")
    cfg2_path = tmp_path / "cfg2.json"
    cfg2_path.write_text(json.dumps(cfg_raw_2), encoding="utf-8")

    engine = AssetPricingEngine()
    out1, written1 = engine.run_from_config_file(cfg1_path, base_dir=PROJECT_ROOT, write_outputs=True)
    out2, written2 = engine.run_from_config_file(cfg2_path, base_dir=PROJECT_ROOT, write_outputs=True)

    # Identical logical outputs and identical content-addressed run_id
    assert out1.run_id == out2.run_id
    assert out1.input_fingerprints == out2.input_fingerprints
    assert out1.comparison == out2.comparison
    assert out1.models == out2.models

    # Verify all artifact files were created
    for key in ("result_json", "replay_manifest", "comparison_csv", "summary_md"):
        assert Path(written1[key]).exists()
        assert Path(written2[key]).exists()

    # Replay from saved manifest matches run_id and comparison exactly
    replayed, _ = engine.replay_from_manifest(
        written1["replay_manifest"],
        base_dir=PROJECT_ROOT,
        verify_fingerprints=True,
        write_outputs=False,
    )
    assert replayed.run_id == out1.run_id
    assert replayed.comparison == out1.comparison

    # Tampering with saved input fingerprint triggers explicit DataContractError on replay
    manifest_path = Path(written1["replay_manifest"])
    tampered = json.loads(manifest_path.read_text(encoding="utf-8"))
    tampered["input_fingerprints"]["risk_free"] = "0" * 64
    tampered_path = tmp_path / "tampered_manifest.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(DataContractError, match="fingerprint mismatch"):
        engine.replay_from_manifest(tampered_path, base_dir=PROJECT_ROOT, verify_fingerprints=True)


def test_operational_error_handling_missing_and_unsupported_inputs(tmp_path: Path) -> None:
    # Missing config file
    with pytest.raises(ConfigurationError, match="not found"):
        load_config(tmp_path / "does_not_exist.json")

    # Unsupported config extension
    yaml_cfg = tmp_path / "config.yaml"
    yaml_cfg.write_text("run_label: test\n", encoding="utf-8")
    with pytest.raises(UnsupportedInputError, match="Unsupported configuration"):
        load_config(yaml_cfg)

    # Malformed JSON
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Malformed JSON"):
        load_config(bad_json)

    # Unsupported cov_type
    cfg_raw = json.loads((PROJECT_ROOT / "config" / "reference_config.json").read_text(encoding="utf-8"))
    cfg_raw["cov_type"] = "BOOTSTRAP_INVALID"
    bad_cov_path = tmp_path / "bad_cov.json"
    bad_cov_path.write_text(json.dumps(cfg_raw), encoding="utf-8")
    with pytest.raises(UnsupportedInputError, match="Unsupported cov_type"):
        load_config(bad_cov_path)

    # Baseline model missing from models list
    cfg_raw2 = json.loads((PROJECT_ROOT / "config" / "reference_config.json").read_text(encoding="utf-8"))
    cfg_raw2["baseline_model_name"] = "NON_EXISTENT_BASE"
    bad_base_path = tmp_path / "bad_base.json"
    bad_base_path.write_text(json.dumps(cfg_raw2), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="baseline_model_name"):
        load_config(bad_base_path)


def test_cli_validate_run_replay_and_error_exit_codes(tmp_path: Path, capsys) -> None:
    # 1. validate valid config
    rc = cli_main(["validate", "--config", str(PROJECT_ROOT / "config" / "reference_config.json")])
    assert rc == 0
    out_text = capsys.readouterr().out
    assert '"status": "valid"' in out_text

    # 2. validate missing config returns non-zero with structured JSON on stderr (no silent success)
    rc_err = cli_main(["validate", "--config", str(tmp_path / "missing.json")])
    assert rc_err == 1
    err_text = capsys.readouterr().err
    err_json = json.loads(err_text)
    assert err_json["error_code"] == "CONFIGURATION_ERROR"

    # 3. run + replay via CLI
    cfg_raw = json.loads((PROJECT_ROOT / "config" / "reference_config.json").read_text(encoding="utf-8"))
    out_dir = tmp_path / "cli_out"
    cfg_raw["output_dir"] = str(out_dir)
    cfg_path = tmp_path / "cli_cfg.json"
    cfg_path.write_text(json.dumps(cfg_raw), encoding="utf-8")

    rc_run = cli_main(["run", "--config", str(cfg_path), "--base-dir", str(PROJECT_ROOT)])
    assert rc_run == 0
    run_payload = json.loads(capsys.readouterr().out)
    assert run_payload["status"] == "ok"

    manifest = out_dir / "replay_manifest.json"
    assert manifest.exists()
    rc_rep = cli_main(["replay", "--manifest", str(manifest), "--base-dir", str(PROJECT_ROOT)])
    assert rc_rep == 0
    rep_payload = json.loads(capsys.readouterr().out)
    assert rep_payload["status"] == "replayed_ok"
    assert rep_payload["run_id"] == run_payload["run_id"]
