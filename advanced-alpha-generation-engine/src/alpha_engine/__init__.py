"""Advanced Alpha Generation Engine — causal research and evaluation framework."""

from alpha_engine.catalog import AlphaCatalog, PaperSignalExporter
from alpha_engine.config import (
    EngineConfig,
    RegistryPolicy,
    UniversePolicy,
    load_engine_config,
    load_registry_policy,
    load_universe_policy,
)
from alpha_engine.pipeline import AlphaResearchPipeline, PipelineResult

__version__ = "0.1.0"

__all__ = [
    "AlphaCatalog",
    "AlphaResearchPipeline",
    "EngineConfig",
    "PaperSignalExporter",
    "PipelineResult",
    "RegistryPolicy",
    "UniversePolicy",
    "__version__",
    "load_engine_config",
    "load_registry_policy",
    "load_universe_policy",
]
