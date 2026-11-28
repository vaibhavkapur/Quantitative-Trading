"""Shared pytest fixtures and helpers."""

from __future__ import annotations

from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = PROJECT_ROOT / "config" / "default_policy.json"
FIXTURES_DIR = PROJECT_ROOT / "data" / "fixtures"


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def default_policy_path() -> Path:
    return POLICY_PATH


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES_DIR
