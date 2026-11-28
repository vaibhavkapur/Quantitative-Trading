"""End-to-end pipeline, promotion gating, replay reproducibility, and CLI tests."""

from __future__ import annotations

import json
from pathlib import Path

from robustness_lab.cli import main as cli_main
from robustness_lab.io_adapters import load_policy_config
from robustness_lab.models import PromotionDecision
from robustness_lab.pipeline import compare_reports_logical_equality, evaluate_from_directory
from robustness_lab.promotion import evaluate_promotion


def test_end_to_end_known_signal_promotes_and_replays_deterministically(
    fixtures_dir: Path, default_policy_path: Path, tmp_path: Path
) -> None:
    exp_dir = fixtures_dir / "valid" / "known_signal_experiment"
    r1 = evaluate_from_directory(exp_dir, default_policy_path, output_dir=tmp_path, report_basename="known")
    r2 = evaluate_from_directory(exp_dir, default_policy_path)

    assert r1["verdict"]["decision"] == PromotionDecision.PROMOTE.value
    assert r1["verdict"]["evidence_complete"] is True
    assert r1["counters"]["n_trials_logged"] == 9
    assert r1["counters"]["n_trials_failed_or_rejected"] == 1
    assert (tmp_path / "known.json").exists()
    assert (tmp_path / "known.md").exists()

    matched, diffs = compare_reports_logical_equality(r1, r2)
    assert matched is True
    assert diffs == []


def test_end_to_end_null_cherry_and_insufficient_reject(
    fixtures_dir: Path, default_policy_path: Path
) -> None:
    null_rep = evaluate_from_directory(
        fixtures_dir / "valid" / "null_overfit_experiment", default_policy_path
    )
    assert null_rep["verdict"]["decision"] == PromotionDecision.REJECT.value

    cherry_rep = evaluate_from_directory(
        fixtures_dir / "valid" / "cherry_picked_experiment", default_policy_path
    )
    assert cherry_rep["verdict"]["decision"] == PromotionDecision.REJECT.value

    insuff_rep = evaluate_from_directory(
        fixtures_dir / "boundary" / "insufficient_sample_experiment", default_policy_path
    )
    assert insuff_rep["verdict"]["decision"] == PromotionDecision.REJECT.value
    assert insuff_rep["verdict"]["evidence_complete"] is False
    assert len(insuff_rep["verdict"]["missing_evidence"]) >= 3


def test_promotion_strictly_rejects_when_any_evidence_is_incomplete(default_policy_path: Path) -> None:
    policy = load_policy_config(default_policy_path)
    verdict = evaluate_promotion(
        candidate_trial_id="cand_partial",
        policy=policy,
        integrity=None,
        dsr=None,
        pbo=None,
        bootstrap=None,
        walk_forward=None,
        perturbation=None,
        outliers=None,
    )
    assert verdict.decision == PromotionDecision.REJECT
    assert verdict.evidence_complete is False
    assert len(verdict.missing_evidence) == 7


def test_cli_evaluate_verify_replay_and_error_exit_codes(
    fixtures_dir: Path, default_policy_path: Path, tmp_path: Path
) -> None:
    # 1. Valid registry verification
    rc_ok = cli_main(
        ["verify-registry", "--ledger", str(fixtures_dir / "valid" / "known_signal_experiment" / "registry.jsonl")]
    )
    assert rc_ok == 0

    # 2. Tampered registry verification -> non-zero exit code (no silent success)
    rc_bad = cli_main(
        ["verify-registry", "--ledger", str(fixtures_dir / "invalid" / "tampered_registry.jsonl")]
    )
    assert rc_bad == 2

    # 3. Evaluate & write baseline JSON, then replay against it
    rc_eval = cli_main(
        [
            "evaluate",
            "--experiment-dir",
            str(fixtures_dir / "valid" / "known_signal_experiment"),
            "--policy",
            str(default_policy_path),
            "--output-dir",
            str(tmp_path),
            "--report-name",
            "cli_run",
        ]
    )
    assert rc_eval == 0
    report_json = tmp_path / "cli_run.json"
    assert report_json.exists()
    parsed = json.loads(report_json.read_text(encoding="utf-8"))
    assert parsed["verdict"]["decision"] == "PROMOTE"

    rc_replay = cli_main(
        [
            "replay",
            "--experiment-dir",
            str(fixtures_dir / "valid" / "known_signal_experiment"),
            "--policy",
            str(default_policy_path),
            "--baseline-report",
            str(report_json),
        ]
    )
    assert rc_replay == 0

    # 4. Missing policy config -> non-zero exit code
    rc_missing = cli_main(
        [
            "evaluate",
            "--experiment-dir",
            str(fixtures_dir / "valid" / "known_signal_experiment"),
            "--policy",
            str(tmp_path / "missing_policy.json"),
        ]
    )
    assert rc_missing == 2
