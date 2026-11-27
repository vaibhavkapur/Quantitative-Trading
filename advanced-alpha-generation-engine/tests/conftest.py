"""Shared deterministic fixtures for unit, integration, and scenario tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from alpha_engine.config import (
    EngineConfig,
    RegistryPolicy,
    UniversePolicy,
    load_engine_config,
    load_registry_policy,
    load_universe_policy,
)
from alpha_engine.data.fixtures import (
    SyntheticDatasetBundle,
    generate_synthetic_pit_dataset,
    save_dataset_fixtures,
)
from alpha_engine.data.loader import PointInTimeDataLoader, ResearchDataset

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def engine_config() -> EngineConfig:
    return load_engine_config(REPO_ROOT / "config" / "engine_config.json")


@pytest.fixture(scope="session")
def universe_policy() -> UniversePolicy:
    return load_universe_policy(REPO_ROOT / "config" / "universe_policy.json")


@pytest.fixture(scope="session")
def registry_policy() -> RegistryPolicy:
    return load_registry_policy(REPO_ROOT / "config" / "registry_policy.json")


@pytest.fixture(scope="session")
def synthetic_bundle() -> SyntheticDatasetBundle:
    return generate_synthetic_pit_dataset(seed=42)


@pytest.fixture(scope="session")
def saved_fixtures_dir(tmp_path_factory: pytest.TempPathFactory, synthetic_bundle: SyntheticDatasetBundle) -> Path:
    d = tmp_path_factory.mktemp("pit_fixtures")
    save_dataset_fixtures(synthetic_bundle, d)
    return d


@pytest.fixture(scope="session")
def research_dataset(saved_fixtures_dir: Path, universe_policy: UniversePolicy) -> ResearchDataset:
    return PointInTimeDataLoader(saved_fixtures_dir, universe_policy).load()
