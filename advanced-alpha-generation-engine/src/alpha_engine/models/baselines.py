"""Statistical and ML predictive baselines for causal cross-sectional alpha research.

Optional external dependencies (such as XGBoost) are isolated behind a clean adapter
protocol so core deterministic baselines run cleanly even without boosting extras.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from alpha_engine.contracts.models import SignalDefinition
from alpha_engine.errors import OptionalDependencyError


class AlphaModel(ABC):
    """Canonical predictive model interface for walk-forward training and inference."""

    @property
    @abstractmethod
    def definition(self) -> SignalDefinition:
        """Return versioned SignalDefinition for this predictive model."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Sequence[str]) -> AlphaModel:
        """Fit deterministically on training window only."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cross-sectional alpha scores on out-of-sample rows."""

    @abstractmethod
    def feature_coefficients(self) -> dict[str, float]:
        """Return signed feature weights/sensitivities for stability diagnostics."""

    @abstractmethod
    def clone_fresh(self) -> AlphaModel:
        """Return an unfitted copy with identical hyperparameters and seed."""


class EqualRankCompositeBaseline(AlphaModel):
    """Simple statistical baseline: equal-weighted standardized composite of selected features."""

    def __init__(
        self,
        *,
        feature_signs: dict[str, float] | None = None,
        signal_id: str = "stat_equal_composite",
        version: str = "1.0.0",
    ) -> None:
        self.feature_signs = feature_signs or {}
        self._feature_names: list[str] = []
        self._means: np.ndarray | None = None
        self._stds: np.ndarray | None = None
        self._weights: np.ndarray | None = None
        self._def = SignalDefinition(
            signal_id=signal_id,
            name="Equal-Weight Statistical Composite",
            version=version,
            family="statistical_ml",
            description="Deterministic z-scored equal-weight composite baseline across causal factors.",
            parameters={"feature_signs": self.feature_signs},
            lookback_bars=60,
            horizon_bars=5,
            inputs_required=["feature_matrix"],
        )

    @property
    def definition(self) -> SignalDefinition:
        return self._def

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Sequence[str]) -> EqualRankCompositeBaseline:
        self._feature_names = list(feature_names)
        self._means = np.nanmean(X, axis=0)
        self._stds = np.nanstd(X, axis=0)
        self._stds = np.where(self._stds < 1e-12, 1.0, self._stds)
        signs = np.array(
            [float(self.feature_signs.get(name, 1.0)) for name in self._feature_names],
            dtype=np.float64,
        )
        self._weights = signs / max(1, len(self._feature_names))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._means is not None and self._stds is not None and self._weights is not None
        cleaned = np.where(np.isfinite(X), X, self._means)
        z = (cleaned - self._means) / self._stds
        return z @ self._weights

    def feature_coefficients(self) -> dict[str, float]:
        if self._weights is None:
            return {}
        return {name: float(w) for name, w in zip(self._feature_names, self._weights, strict=True)}

    def clone_fresh(self) -> EqualRankCompositeBaseline:
        return EqualRankCompositeBaseline(
            feature_signs=dict(self.feature_signs),
            signal_id=self._def.signal_id,
            version=self._def.version,
        )


class RidgeAlphaBaseline(AlphaModel):
    """Regularized linear ML baseline (StandardScaler + Ridge) with deterministic fitting."""

    def __init__(
        self,
        *,
        alpha: float = 50.0,
        signal_id: str = "ml_ridge_baseline",
        version: str = "1.0.0",
    ) -> None:
        self.alpha = alpha
        self._feature_names: list[str] = []
        self._pipeline: Pipeline | None = None
        self._col_medians: np.ndarray | None = None
        self._def = SignalDefinition(
            signal_id=signal_id,
            name=f"Ridge Linear Baseline (alpha={alpha})",
            version=version,
            family="statistical_ml",
            description="L2-regularized cross-sectional Ridge regression trained strictly on walk-forward history.",
            parameters={"alpha": alpha, "model": "sklearn.linear_model.Ridge"},
            lookback_bars=60,
            horizon_bars=5,
            inputs_required=["feature_matrix"],
        )

    @property
    def definition(self) -> SignalDefinition:
        return self._def

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Sequence[str]) -> RidgeAlphaBaseline:
        self._feature_names = list(feature_names)
        self._col_medians = np.nanmedian(X, axis=0)
        self._col_medians = np.where(np.isfinite(self._col_medians), self._col_medians, 0.0)
        X_clean = np.where(np.isfinite(X), X, self._col_medians)
        mask = np.isfinite(y)
        self._pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=self.alpha, random_state=42)),
            ]
        )
        self._pipeline.fit(X_clean[mask], y[mask])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._pipeline is not None and self._col_medians is not None
        X_clean = np.where(np.isfinite(X), X, self._col_medians)
        return np.asarray(self._pipeline.predict(X_clean), dtype=np.float64)

    def feature_coefficients(self) -> dict[str, float]:
        if self._pipeline is None:
            return {}
        coefs = self._pipeline.named_steps["ridge"].coef_
        return {name: float(c) for name, c in zip(self._feature_names, coefs, strict=True)}

    def clone_fresh(self) -> RidgeAlphaBaseline:
        return RidgeAlphaBaseline(
            alpha=self.alpha,
            signal_id=self._def.signal_id,
            version=self._def.version,
        )


class GradientBoostedAlphaBaseline(AlphaModel):
    """Regularized scikit-learn HistGradientBoosting baseline (zero external native runtime deps)."""

    def __init__(
        self,
        *,
        max_iter: int = 40,
        max_depth: int = 2,
        learning_rate: float = 0.05,
        l2_regularization: float = 15.0,
        seed: int = 42,
        signal_id: str = "ml_gbdt_baseline",
        version: str = "1.0.0",
    ) -> None:
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.l2_regularization = l2_regularization
        self.seed = seed
        self._feature_names: list[str] = []
        self._model: HistGradientBoostingRegressor | None = None
        self._signed_sensitivities: dict[str, float] = {}
        self._col_medians: np.ndarray | None = None
        self._def = SignalDefinition(
            signal_id=signal_id,
            name="Regularized GBDT Baseline (sklearn)",
            version=version,
            family="statistical_ml",
            description="Shallow L2-regularized histogram gradient-boosted tree baseline using scikit-learn.",
            parameters={
                "max_iter": max_iter,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "l2_regularization": l2_regularization,
                "seed": seed,
            },
            lookback_bars=60,
            horizon_bars=5,
            inputs_required=["feature_matrix"],
        )

    @property
    def definition(self) -> SignalDefinition:
        return self._def

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Sequence[str]) -> GradientBoostedAlphaBaseline:
        self._feature_names = list(feature_names)
        self._col_medians = np.nanmedian(X, axis=0)
        self._col_medians = np.where(np.isfinite(self._col_medians), self._col_medians, 0.0)
        X_clean = np.where(np.isfinite(X), X, self._col_medians)
        mask = np.isfinite(y)
        X_fit = X_clean[mask]
        y_fit = y[mask]

        model = HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            l2_regularization=self.l2_regularization,
            min_samples_leaf=20,
            random_state=self.seed,
        )
        model.fit(X_fit, y_fit)
        self._model = model

        # Directional partial sensitivity per feature (standardized correlation with model output)
        preds = np.asarray(model.predict(X_fit), dtype=np.float64)
        sens: dict[str, float] = {}
        for j, name in enumerate(self._feature_names):
            col = X_fit[:, j]
            if np.std(col) > 1e-12 and np.std(preds) > 1e-12:
                sens[name] = float(np.corrcoef(col, preds)[0, 1])
            else:
                sens[name] = 0.0
        self._signed_sensitivities = sens
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None or self._col_medians is None:
            raise RuntimeError("GradientBoostedAlphaBaseline must be fit before predict.")
        X_clean = np.where(np.isfinite(X), X, self._col_medians)
        return np.asarray(self._model.predict(X_clean), dtype=np.float64)

    def feature_coefficients(self) -> dict[str, float]:
        return dict(self._signed_sensitivities)

    def clone_fresh(self) -> GradientBoostedAlphaBaseline:
        return GradientBoostedAlphaBaseline(
            max_iter=self.max_iter,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            l2_regularization=self.l2_regularization,
            seed=self.seed,
            signal_id=self._def.signal_id,
            version=self._def.version,
        )


class XGBoostAlphaBaseline(AlphaModel):
    """Isolated optional XGBoost gradient-boosted baseline with conservative regularization."""

    def __init__(
        self,
        *,
        n_estimators: int = 40,
        max_depth: int = 2,
        learning_rate: float = 0.05,
        reg_lambda: float = 15.0,
        subsample: float = 0.8,
        seed: int = 42,
        signal_id: str = "ml_xgboost_baseline",
        version: str = "1.0.0",
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.subsample = subsample
        self.seed = seed
        self._feature_names: list[str] = []
        self._model = None
        self._signed_importances: dict[str, float] = {}
        self._col_medians: np.ndarray | None = None
        self._def = SignalDefinition(
            signal_id=signal_id,
            name="XGBoost Regularized Baseline",
            version=version,
            family="statistical_ml",
            description="Shallow L2-regularized gradient-boosted tree baseline isolated behind optional xgboost dependency.",
            parameters={
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "reg_lambda": reg_lambda,
                "subsample": subsample,
                "seed": seed,
            },
            lookback_bars=60,
            horizon_bars=5,
            inputs_required=["feature_matrix"],
        )

    @property
    def definition(self) -> SignalDefinition:
        return self._def

    @staticmethod
    def _import_xgboost():
        try:
            import xgboost as xgb  # type: ignore[import-not-found]
        except Exception as exc:
            raise OptionalDependencyError(
                "Optional xgboost runtime unavailable (missing package or native libomp runtime). "
                "Install with `uv sync --extra boosting` (and `brew install libomp` on macOS).",
                details={
                    "optional_extra": "boosting",
                    "package": "xgboost",
                    "cause": str(exc)[:240],
                },
            ) from exc
        return xgb

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Sequence[str]) -> XGBoostAlphaBaseline:
        xgb = self._import_xgboost()
        self._feature_names = list(feature_names)
        self._col_medians = np.nanmedian(X, axis=0)
        self._col_medians = np.where(np.isfinite(self._col_medians), self._col_medians, 0.0)
        X_clean = np.where(np.isfinite(X), X, self._col_medians)
        mask = np.isfinite(y)
        X_fit = X_clean[mask]
        y_fit = y[mask]

        model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            reg_lambda=self.reg_lambda,
            subsample=self.subsample,
            colsample_bytree=0.8,
            random_state=self.seed,
            n_jobs=1,
            tree_method="hist",
        )
        model.fit(X_fit, y_fit)
        self._model = model

        # Combine tree gain importance with correlation sign of predictions vs feature
        preds = np.asarray(model.predict(X_fit), dtype=np.float64)
        raw_imp = np.asarray(model.feature_importances_, dtype=np.float64)
        signed: dict[str, float] = {}
        for j, name in enumerate(self._feature_names):
            col = X_fit[:, j]
            if np.std(col) > 1e-12 and np.std(preds) > 1e-12:
                corr = float(np.corrcoef(col, preds)[0, 1])
                sign = 1.0 if corr >= 0 else -1.0
            else:
                sign = 1.0
            signed[name] = float(sign * raw_imp[j])
        self._signed_importances = signed
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None or self._col_medians is None:
            raise RuntimeError("XGBoostAlphaBaseline must be fit before predict.")
        X_clean = np.where(np.isfinite(X), X, self._col_medians)
        return np.asarray(self._model.predict(X_clean), dtype=np.float64)

    def feature_coefficients(self) -> dict[str, float]:
        return dict(self._signed_importances)

    def clone_fresh(self) -> XGBoostAlphaBaseline:
        return XGBoostAlphaBaseline(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            reg_lambda=self.reg_lambda,
            subsample=self.subsample,
            seed=self.seed,
            signal_id=self._def.signal_id,
            version=self._def.version,
        )
