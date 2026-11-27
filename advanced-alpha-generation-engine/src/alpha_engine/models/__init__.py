"""Statistical and machine learning alpha baselines with isolated optional dependencies."""

from alpha_engine.models.baselines import (
    AlphaModel,
    EqualRankCompositeBaseline,
    GradientBoostedAlphaBaseline,
    RidgeAlphaBaseline,
    XGBoostAlphaBaseline,
)

__all__ = [
    "AlphaModel",
    "EqualRankCompositeBaseline",
    "GradientBoostedAlphaBaseline",
    "RidgeAlphaBaseline",
    "XGBoostAlphaBaseline",
]
