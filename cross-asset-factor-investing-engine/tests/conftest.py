"""Shared pytest fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from cross_asset_factor_engine import EngineConfig, MarketDataset, load_config, load_dataset_from_dir

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = PROJECT_ROOT / "data" / "fixtures"


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def small_dataset() -> MarketDataset:
    return load_dataset_from_dir(FIXTURES_DIR / "valid_small", base_currency="USD")


@pytest.fixture(scope="session")
def small_config() -> EngineConfig:
    return load_config(PROJECT_ROOT / "config" / "reference_fixture.yaml")


@pytest.fixture(scope="session")
def small_oracle() -> dict[str, Any]:
    return json.loads((FIXTURES_DIR / "valid_small" / "expected_reference.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def multi_year_dataset() -> MarketDataset:
    return load_dataset_from_dir(FIXTURES_DIR / "synthetic_multi_year", base_currency="USD")


@pytest.fixture(scope="session")
def default_config() -> EngineConfig:
    return load_config(PROJECT_ROOT / "config" / "default.yaml")
