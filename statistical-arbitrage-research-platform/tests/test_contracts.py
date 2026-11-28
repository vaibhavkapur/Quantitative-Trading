"""Data contract, invalid fixture, and operational error handling tests."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from statarb.cli import main as cli_main
from statarb.config import load_config, validate_config_dict
from statarb.contracts import load_and_validate_inputs, validate_borrow, validate_prices
from statarb.exceptions import (
    ConfigurationError,
    ContractValidationError,
    MissingDataError,
    UnsupportedInputError,
)
from statarb.fixtures import write_all_fixtures


@pytest.fixture(scope="module")
def fixtures_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("fixtures")
    write_all_fixtures(root)
    return root


def test_valid_reference_and_universe_fixtures_pass_contracts(fixtures_dir: Path) -> None:
    _, _, ref_rep = load_and_validate_inputs(
        fixtures_dir / "reference_pair_prices.csv",
        fixtures_dir / "reference_pair_borrow.csv",
    )
    assert ref_rep.prices_rows > 0
    assert ref_rep.borrow_rows > 0
    assert len(ref_rep.symbols) == 4

    _, _, uni_rep = load_and_validate_inputs(
        fixtures_dir / "synthetic_universe_prices.parquet",
        fixtures_dir / "synthetic_universe_borrow.parquet",
    )
    assert uni_rep.corporate_action_events >= 2
    assert len(uni_rep.symbols) == 14


def test_missing_price_raises_or_drops_symbol(fixtures_dir: Path) -> None:
    missing_df = pl.read_csv(fixtures_dir / "invalid" / "missing_prices.csv", try_parse_dates=True)

    with pytest.raises(MissingDataError, match="null/non-finite"):
        validate_prices(missing_df, missing_price_policy="raise")

    cleaned, dropped, _ = validate_prices(missing_df, missing_price_policy="drop_symbol")
    assert "REF_Y" in dropped
    assert "REF_Y" not in cleaned["symbol"].unique().to_list()


def test_duplicate_keys_raise_contract_error(fixtures_dir: Path) -> None:
    dup_df = pl.read_csv(fixtures_dir / "invalid" / "duplicate_keys.csv", try_parse_dates=True)
    with pytest.raises(ContractValidationError, match="duplicate"):
        validate_prices(dup_df)


def test_negative_price_raises_contract_error(fixtures_dir: Path) -> None:
    neg_df = pl.read_csv(fixtures_dir / "invalid" / "negative_price.csv", try_parse_dates=True)
    with pytest.raises(ContractValidationError, match="strictly positive"):
        validate_prices(neg_df)


def test_unadjusted_split_caught_by_corporate_action_validator(fixtures_dir: Path) -> None:
    bad_split = pl.read_csv(
        fixtures_dir / "invalid" / "unadjusted_split_prices.csv", try_parse_dates=True
    )
    with pytest.raises(ContractValidationError, match="Corporate action inconsistency"):
        validate_prices(bad_split, check_corporate_actions=True)


def test_missing_required_column_and_unsupported_format(tmp_path: Path) -> None:
    bad_df = pl.DataFrame({"date": ["2023-01-02"], "symbol": ["A"]})
    with pytest.raises(ContractValidationError, match="missing required columns"):
        validate_prices(bad_df)

    bad_borrow = pl.DataFrame({"date": ["2023-01-02"], "symbol": ["A"]})
    with pytest.raises(ContractValidationError, match="missing required columns"):
        validate_borrow(bad_borrow)

    unsupported_file = tmp_path / "prices.xlsx"
    unsupported_file.write_text("dummy", encoding="utf-8")
    with pytest.raises(UnsupportedInputError):
        load_and_validate_inputs(unsupported_file, unsupported_file)

    with pytest.raises(MissingDataError):
        load_and_validate_inputs(tmp_path / "does_not_exist.csv", tmp_path / "does_not_exist.csv")


def test_operational_config_errors_and_cli_nonzero_exit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Missing config file
    with pytest.raises(ConfigurationError, match="not found"):
        load_config(tmp_path / "missing.toml")

    # Unsupported config extension
    bad_ext = tmp_path / "cfg.yaml"
    bad_ext.write_text("seed: 1\n", encoding="utf-8")
    with pytest.raises(UnsupportedInputError):
        load_config(bad_ext)

    # Unknown keys rejected (extra="forbid")
    with pytest.raises(ConfigurationError):
        validate_config_dict(
            {
                "data": {"prices_path": "a.csv", "borrow_path": "b.csv"},
                "unknown_section": {"foo": 1},
            }
        )

    # Invalid signal thresholds rejected
    with pytest.raises(ConfigurationError):
        validate_config_dict(
            {
                "data": {"prices_path": "a.csv", "borrow_path": "b.csv"},
                "signals": {"z_entry": 3.0, "z_exit": 0.5, "z_stop": 2.0},
            }
        )

    # CLI must return non-zero exit code with explicit diagnostic on missing config
    rc = cli_main(["run", "--config", str(tmp_path / "missing.toml")])
    assert rc == 2
    err = capsys.readouterr().err
    assert "ConfigurationError" in err
