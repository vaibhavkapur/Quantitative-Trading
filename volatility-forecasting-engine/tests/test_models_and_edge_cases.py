"""Unit tests for statistical & ML volatility baselines, constant returns, and negative forecast rejection."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from volatility_forecasting.config import (
    DataConfig,
    EngineConfig,
    EvaluationConfig,
    FeatureConfig,
    ModelSpec,
    OutputConfig,
    SplitConfig,
)
from volatility_forecasting.errors import NegativeForecastError, UnsupportedOptionError
from volatility_forecasting.features import build_causal_dataset
from volatility_forecasting.fixtures import (
    generate_constant_returns_fixture,
    generate_multi_regime_fixture,
)
from volatility_forecasting.models import (
    BaseVolatilityModel,
    EWMAModel,
    GARCH11Model,
    HGBRVolModel,
    NaivePersistenceModel,
    RidgeHARModel,
    RollingHistoricalModel,
    create_model,
)
from volatility_forecasting.pipeline import evaluate_models_chronologically


class _BrokenNegativeModel(BaseVolatilityModel):
    """Test double that intentionally emits negative variance forecasts."""

    def fit(self, train_df: pl.DataFrame, horizon: int) -> "_BrokenNegativeModel":
        self.is_fitted = True
        return self

    def _predict_raw_variance(
        self,
        eval_df: pl.DataFrame,
        horizon: int,
        context_df: pl.DataFrame | None = None,
    ) -> np.ndarray:
        preds = np.full(eval_df.height, 1e-4, dtype=np.float64)
        preds[0] = -1e-5
        return preds


class _BrokenNaNModel(BaseVolatilityModel):
    """Test double that intentionally emits NaN forecasts."""

    def fit(self, train_df: pl.DataFrame, horizon: int) -> "_BrokenNaNModel":
        self.is_fitted = True
        return self

    def _predict_raw_variance(
        self,
        eval_df: pl.DataFrame,
        horizon: int,
        context_df: pl.DataFrame | None = None,
    ) -> np.ndarray:
        preds = np.full(eval_df.height, 1e-4, dtype=np.float64)
        preds[-1] = float("nan")
        return preds


def test_all_baseline_models_fit_predict_nonnegative() -> None:
    ret_df, bar_df = generate_multi_regime_fixture(n_days=220, bars_per_day=13, seed=7)
    dataset, _ = build_causal_dataset(ret_df, bar_df, horizons=(1, 5), expected_bars_per_session=13)
    valid = dataset.filter(pl.col("target_rv_h5").is_not_null())
    train_df = valid[:150]
    test_df = valid[155:195]

    specs = [
        ModelSpec(name="naive", type="naive", params={"lookback": 1}),
        ModelSpec(name="rolling", type="rolling_historical", params={"window": 22}),
        ModelSpec(name="ewma", type="ewma", params={"decay": 0.94}),
        ModelSpec(name="garch11", type="garch11", params={"backend": "scipy"}),
        ModelSpec(name="ridge_har", type="ridge_har", params={"alpha": 1.0, "use_log_target": True}),
        ModelSpec(name="hgbr", type="hgbr", params={"max_iter": 40, "max_depth": 3}),
    ]

    for spec in specs:
        model = create_model(spec, seed=42)
        model.fit(train_df, horizon=5)
        preds = model.predict_variance(test_df, horizon=5, context_df=valid[:155])
        assert preds.shape == (test_df.height,)
        assert np.all(np.isfinite(preds))
        assert np.all(preds >= 0.0), f"Model {spec.name} produced negative predictions"


def test_all_models_handle_constant_returns_cleanly() -> None:
    const_df = generate_constant_returns_fixture(n_days=80, constant_return=0.0)
    dataset, _ = build_causal_dataset(const_df, horizons=(1, 5))
    valid = dataset.filter(pl.col("target_rv_h5").is_not_null())
    train_df = valid[:50]
    test_df = valid[55:70]

    models = [
        NaivePersistenceModel(),
        RollingHistoricalModel(params={"window": 10}),
        EWMAModel(params={"decay": 0.94}),
        GARCH11Model(params={"backend": "scipy"}),
        RidgeHARModel(),
        HGBRVolModel(params={"max_iter": 20}),
    ]

    for model in models:
        model.fit(train_df, horizon=5)
        preds = model.predict_variance(test_df, horizon=5, context_df=valid[:55])
        assert np.all(np.isfinite(preds))
        np.testing.assert_allclose(preds, np.zeros(test_df.height), atol=1e-12)


def test_ewma_exact_analytical_recursion() -> None:
    df = pl.DataFrame(
        {
            "asset_id": ["A"] * 5,
            "timestamp": [
                "2024-01-02T16:00:00",
                "2024-01-03T16:00:00",
                "2024-01-04T16:00:00",
                "2024-01-05T16:00:00",
                "2024-01-08T16:00:00",
            ],
            "log_return": [0.01, 0.02, 0.01, 0.03, 0.0],
        }
    )
    dataset, _ = build_causal_dataset(df, horizons=(1,))
    decay = 0.8
    model = EWMAModel(params={"decay": decay})
    model.fit(dataset[:3], horizon=1)

    # Hand-calculated state after train[:3] (rv = [1e-4, 4e-4, 1e-4], mean = 2e-4):
    s = 2e-4
    for v in [1e-4, 4e-4, 1e-4]:
        s = decay * s + (1.0 - decay) * v
    expected_step3 = decay * s + (1.0 - decay) * 9e-4
    expected_step4 = decay * expected_step3 + (1.0 - decay) * 0.0

    preds = model.predict_variance(dataset[3:], horizon=1)
    np.testing.assert_allclose(preds, [expected_step3, expected_step4], rtol=1e-12, atol=1e-15)


def test_garch11_analytical_horizon_averaging() -> None:
    omega, alpha, beta = 1e-5, 0.10, 0.80
    phi = alpha + beta
    uncond = omega / (1.0 - phi)
    one_step = 4e-4
    h = 5
    # Direct sum of k=1..5: uncond + phi^(k-1) * (one_step - uncond)
    manual_avg = np.mean([uncond + (phi ** (k - 1)) * (one_step - uncond) for k in range(1, h + 1)])
    calc = GARCH11Model._horizon_avg_forecast(one_step, uncond, phi, h)
    assert calc == pytest.approx(manual_avg, rel=1e-12)


def test_negative_and_nan_forecast_explicit_rejection() -> None:
    ret_df, bar_df = generate_multi_regime_fixture(n_days=140, bars_per_day=13, seed=11)
    dataset, _ = build_causal_dataset(ret_df, bar_df, horizons=(1, 5), expected_bars_per_session=13)
    valid = dataset.filter(pl.col("target_rv_h5").is_not_null())

    # Direct model check
    bad_neg = _BrokenNegativeModel(name="broken_neg").fit(valid[:80], horizon=5)
    with pytest.raises(NegativeForecastError, match="negative variance forecasts"):
        bad_neg.predict_variance(valid[85:100], horizon=5, strict_nonnegative=True)

    bad_nan = _BrokenNaNModel(name="broken_nan").fit(valid[:80], horizon=5)
    with pytest.raises(NegativeForecastError, match="non-finite variance forecasts"):
        bad_nan.predict_variance(valid[85:100], horizon=5, strict_nonnegative=True)

    # Pipeline-level check
    cfg = EngineConfig(
        version="1.0",
        run_name="neg_test",
        seed=42,
        annualization_factor=252,
        expected_bars_per_session=13,
        data=DataConfig(returns_path="unused.csv"),
        features=FeatureConfig(horizons=(1, 5), primary_horizon=5),
        splits=SplitConfig(scheme="expanding", n_splits=1, min_train_size=80, test_size=20),
        models=(ModelSpec(name="naive", type="naive"),),
        evaluation=EvaluationConfig(strict_nonnegative=True),
        output=OutputConfig(reports_dir="reports"),
    )
    with pytest.raises(NegativeForecastError):
        evaluate_models_chronologically(
            dataset,
            cfg,
            horizon_days=5,
            custom_models=[_BrokenNegativeModel(name="broken_neg")],
        )


def test_optional_arch_backend_clear_error_when_missing() -> None:
    try:
        import arch  # noqa: F401
        pytest.skip("Optional 'arch' package is installed in this environment.")
    except ImportError:
        pass

    ret_df, _ = generate_multi_regime_fixture(n_days=60, bars_per_day=4, seed=3)
    dataset, _ = build_causal_dataset(ret_df, horizons=(1,))
    model = GARCH11Model(params={"backend": "arch"})
    with pytest.raises(UnsupportedOptionError, match="optional 'arch' dependency"):
        model.fit(dataset[:50], horizon=1)
