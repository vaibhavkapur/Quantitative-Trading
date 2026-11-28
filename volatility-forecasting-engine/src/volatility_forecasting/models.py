"""Pure domain volatility forecasting models (isolated from I/O)."""

from __future__ import annotations

from abc import ABC, abstractmethod
import math
from typing import Any

import numpy as np
import polars as pl
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from volatility_forecasting.config import ModelSpec
from volatility_forecasting.contracts import validate_nonnegative_forecasts
from volatility_forecasting.errors import (
    ModelFitError,
    UnsupportedOptionError,
)
from volatility_forecasting.features import CAUSAL_FEATURE_COLUMNS


class BaseVolatilityModel(ABC):
    """Abstract base class for causal daily-variance forecasting models."""

    def __init__(self, name: str, params: dict[str, Any] | None = None, seed: int = 42) -> None:
        self.name = name
        self.params = dict(params or {})
        self.seed = int(seed)
        self.is_fitted = False
        self.fit_metadata: dict[str, Any] = {}

    @abstractmethod
    def fit(self, train_df: pl.DataFrame, horizon: int) -> "BaseVolatilityModel":
        """Fit model parameters strictly on the provided training window."""

    @abstractmethod
    def _predict_raw_variance(
        self,
        eval_df: pl.DataFrame,
        horizon: int,
        context_df: pl.DataFrame | None = None,
    ) -> np.ndarray:
        """Return 1D daily realized-variance predictions aligned with eval_df."""

    def predict_variance(
        self,
        eval_df: pl.DataFrame,
        horizon: int,
        context_df: pl.DataFrame | None = None,
        *,
        strict_nonnegative: bool = True,
    ) -> np.ndarray:
        """Predict daily variance with contract enforcement against negative/NaN forecasts."""
        if not self.is_fitted:
            raise ModelFitError(f"Model '{self.name}' must be fitted before calling predict.")
        if eval_df.height == 0:
            return np.array([], dtype=np.float64)
        raw = self._predict_raw_variance(eval_df, horizon, context_df=context_df)
        if strict_nonnegative:
            return validate_nonnegative_forecasts(raw, model_name=self.name, quantity="variance")
        return np.asarray(raw, dtype=np.float64)


class NaivePersistenceModel(BaseVolatilityModel):
    """Naive benchmark forecasting future variance as the trailing k-day realized variance."""

    def __init__(self, name: str = "naive", params: dict[str, Any] | None = None, seed: int = 42):
        super().__init__(name=name, params=params, seed=seed)
        self.lookback = int(self.params.get("lookback", 1))
        if self.lookback <= 0:
            raise ModelFitError("NaivePersistenceModel lookback must be >= 1.")
        self._train_mean_rv = 0.0

    def fit(self, train_df: pl.DataFrame, horizon: int) -> "NaivePersistenceModel":
        if train_df.height == 0:
            raise ModelFitError(f"Model '{self.name}' received empty training data.")
        self._train_mean_rv = float(np.mean(train_df["daily_rv"].to_numpy()))
        self.fit_metadata = {"lookback": self.lookback, "train_mean_rv": self._train_mean_rv}
        self.is_fitted = True
        return self

    def _predict_raw_variance(
        self,
        eval_df: pl.DataFrame,
        horizon: int,
        context_df: pl.DataFrame | None = None,
    ) -> np.ndarray:
        if self.lookback == 1:
            return np.maximum(0.0, eval_df["rv_1d"].to_numpy().astype(np.float64))
        if self.lookback == 5 and "rv_5d" in eval_df.columns:
            return np.maximum(0.0, eval_df["rv_5d"].to_numpy().astype(np.float64))
        if self.lookback == 22 and "rv_22d" in eval_df.columns:
            return np.maximum(0.0, eval_df["rv_22d"].to_numpy().astype(np.float64))
        return _causal_rolling_rv(eval_df, context_df, window=self.lookback)


class RollingHistoricalModel(BaseVolatilityModel):
    """Rolling historical variance model averaging trailing W days of realized variance."""

    def __init__(
        self, name: str = "rolling_historical", params: dict[str, Any] | None = None, seed: int = 42
    ):
        super().__init__(name=name, params=params, seed=seed)
        self.window = int(self.params.get("window", 22))
        if self.window <= 0:
            raise ModelFitError("RollingHistoricalModel window must be >= 1.")

    def fit(self, train_df: pl.DataFrame, horizon: int) -> "RollingHistoricalModel":
        if train_df.height == 0:
            raise ModelFitError(f"Model '{self.name}' received empty training data.")
        self.fit_metadata = {
            "window": self.window,
            "train_mean_rv": float(np.mean(train_df["daily_rv"].to_numpy())),
        }
        self.is_fitted = True
        return self

    def _predict_raw_variance(
        self,
        eval_df: pl.DataFrame,
        horizon: int,
        context_df: pl.DataFrame | None = None,
    ) -> np.ndarray:
        if self.window == 22 and "rv_22d" in eval_df.columns:
            return np.maximum(0.0, eval_df["rv_22d"].to_numpy().astype(np.float64))
        if self.window == 5 and "rv_5d" in eval_df.columns:
            return np.maximum(0.0, eval_df["rv_5d"].to_numpy().astype(np.float64))
        return _causal_rolling_rv(eval_df, context_df, window=self.window)


class EWMAModel(BaseVolatilityModel):
    """RiskMetrics-style EWMA recursive realized-variance filter.

    Recursion at origin t (strictly causal):
        v_hat_t = decay * v_hat_{t-1} + (1 - decay) * daily_rv_t
    Because EWMA is a random-walk variance model (IGARCH with omega=0),
    the h-step average forecast equals v_hat_t.
    """

    def __init__(self, name: str = "ewma", params: dict[str, Any] | None = None, seed: int = 42):
        super().__init__(name=name, params=params, seed=seed)
        self.decay = float(self.params.get("decay", 0.94))
        if not (0.0 < self.decay < 1.0):
            raise ModelFitError(f"EWMA decay must be in (0, 1), got {self.decay}.")
        self._init_var_by_asset: dict[str, float] = {}
        self._global_init_var = 0.0

    def fit(self, train_df: pl.DataFrame, horizon: int) -> "EWMAModel":
        if train_df.height == 0:
            raise ModelFitError(f"Model '{self.name}' received empty training data.")
        self._init_var_by_asset = {}
        all_rv = train_df["daily_rv"].to_numpy().astype(np.float64)
        self._global_init_var = float(max(0.0, np.mean(all_rv)))

        for (asset_id,), grp in train_df.group_by(["asset_id"], maintain_order=True):
            rv_seq = grp.sort("timestamp")["daily_rv"].to_numpy().astype(np.float64)
            state = float(max(0.0, np.mean(rv_seq[: min(22, len(rv_seq))])))
            for val in rv_seq:
                state = self.decay * state + (1.0 - self.decay) * float(max(0.0, val))
            self._init_var_by_asset[str(asset_id)] = state

        self.fit_metadata = {
            "decay": self.decay,
            "global_init_var": self._global_init_var,
        }
        self.is_fitted = True
        return self

    def _predict_raw_variance(
        self,
        eval_df: pl.DataFrame,
        horizon: int,
        context_df: pl.DataFrame | None = None,
    ) -> np.ndarray:
        preds = np.empty(eval_df.height, dtype=np.float64)
        eval_indexed = eval_df.with_row_index("_row_idx")

        for (asset_id,), grp in eval_indexed.group_by(["asset_id"], maintain_order=True):
            aid = str(asset_id)
            grp_sorted = grp.sort("timestamp")
            row_idxs = grp_sorted["_row_idx"].to_numpy()
            eval_ts = grp_sorted["timestamp"].to_list()
            eval_rv = grp_sorted["daily_rv"].to_numpy().astype(np.float64)

            if context_df is not None and context_df.height > 0:
                ctx_asset = context_df.filter(pl.col("asset_id") == aid).sort("timestamp")
                ctx_rv = ctx_asset["daily_rv"].to_numpy().astype(np.float64)
                ctx_ts = ctx_asset["timestamp"].to_list()
                # Filter context strictly earlier than first eval timestamp
                first_eval_ts = eval_ts[0]
                pre_rv = [float(v) for ts, v in zip(ctx_ts, ctx_rv) if ts < first_eval_ts]
                if pre_rv:
                    state = float(max(0.0, np.mean(pre_rv[: min(22, len(pre_rv))])))
                    for v in pre_rv:
                        state = self.decay * state + (1.0 - self.decay) * max(0.0, v)
                else:
                    state = self._init_var_by_asset.get(aid, self._global_init_var)
            else:
                state = self._init_var_by_asset.get(aid, self._global_init_var)

            out_vals = np.empty(len(eval_rv), dtype=np.float64)
            for i, v in enumerate(eval_rv):
                state = self.decay * state + (1.0 - self.decay) * max(0.0, float(v))
                out_vals[i] = max(0.0, state)
            preds[row_idxs] = out_vals

        return preds


class GARCH11Model(BaseVolatilityModel):
    """Stationary Gaussian QML GARCH(1,1) variance model with exact h-step forecast.

    Model specification:
        sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2
        omega > 0, alpha >= 0, beta >= 0, alpha + beta < 1

    Supports:
      - backend="scipy": Deterministic self-contained Gaussian QML via L-BFGS-B.
      - backend="arch": Optional adapter delegating parameter estimation to the
        `arch` package if installed.
    """

    def __init__(self, name: str = "garch11", params: dict[str, Any] | None = None, seed: int = 42):
        super().__init__(name=name, params=params, seed=seed)
        self.backend = str(self.params.get("backend", "scipy"))
        if self.backend not in {"scipy", "arch"}:
            raise UnsupportedOptionError(
                f"Unsupported GARCH11 backend '{self.backend}'. Supported: ['arch', 'scipy']."
            )
        self.variance_floor = float(self.params.get("variance_floor", 1e-10))
        self._params_by_asset: dict[str, dict[str, float]] = {}
        self._global_params: dict[str, float] = {
            "omega": 0.0,
            "alpha": 0.0,
            "beta": 0.0,
            "uncond_var": 0.0,
            "last_var": 0.0,
            "rv_scale": 1.0,
        }

    @staticmethod
    def _fit_single_scipy(returns: np.ndarray, daily_rv: np.ndarray) -> dict[str, float]:
        r = np.asarray(returns, dtype=np.float64)
        rv = np.asarray(daily_rv, dtype=np.float64)
        sample_var = float(np.mean(r * r))
        mean_rv = float(np.mean(np.maximum(rv, 0.0)))

        # Constant returns edge case: exact zero variance
        if sample_var <= 1e-16 and mean_rv <= 1e-16:
            return {
                "omega": 0.0,
                "alpha": 0.0,
                "beta": 0.0,
                "uncond_var": 0.0,
                "last_var": 0.0,
                "rv_scale": 1.0,
            }

        # Use daily_rv as innovation proxy if returns are constant while intraday RV > 0
        innov = r * r if sample_var > 1e-16 else np.maximum(rv, 0.0)
        scale = float(np.mean(innov))
        z2 = innov / scale  # O(1) variance-normalized squared innovations
        n = len(z2)

        def neg_loglik(theta: np.ndarray) -> float:
            omega, alpha, beta = float(theta[0]), float(theta[1]), float(theta[2])
            if omega <= 1e-8 or alpha < 0.0 or beta < 0.0 or (alpha + beta) >= 0.998:
                return 1e12
            sig2 = np.empty(n, dtype=np.float64)
            sig2[0] = 1.0
            for t in range(1, n):
                sig2[t] = omega + alpha * z2[t - 1] + beta * sig2[t - 1]
                if sig2[t] <= 1e-10:
                    return 1e12
            return float(0.5 * np.sum(np.log(sig2) + z2 / sig2))

        best_res = None
        starts = [
            np.array([0.05, 0.08, 0.87], dtype=np.float64),
            np.array([0.10, 0.12, 0.78], dtype=np.float64),
            np.array([0.02, 0.05, 0.93], dtype=np.float64),
        ]
        bounds = [(1e-6, 1.0), (1e-6, 0.50), (1e-6, 0.98)]

        for x0 in starts:
            res = minimize(
                neg_loglik,
                x0=x0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 250, "ftol": 1e-10},
            )
            if best_res is None or res.fun < best_res.fun:
                best_res = res

        assert best_res is not None
        omega_s, alpha, beta = (
            float(best_res.x[0]),
            float(best_res.x[1]),
            float(best_res.x[2]),
        )
        if alpha + beta >= 0.998:
            shrink = 0.995 / (alpha + beta)
            alpha *= shrink
            beta *= shrink
            omega_s = max(1e-6, 1.0 - alpha - beta)

        omega = omega_s * scale
        uncond_var = omega / max(1e-6, 1.0 - alpha - beta)

        # Compute terminal conditional variance at end of training window
        sig2 = scale
        for t in range(n):
            sig2 = omega + alpha * float(innov[t]) + beta * sig2

        # Ratio aligning return-squared variance scale to realized-variance scale (fit on train only)
        rv_scale = float(np.clip(mean_rv / max(sample_var, 1e-16), 0.25, 4.0)) if sample_var > 1e-16 else 1.0
        return {
            "omega": omega,
            "alpha": alpha,
            "beta": beta,
            "uncond_var": uncond_var,
            "last_var": max(0.0, sig2),
            "rv_scale": rv_scale,
        }

    @staticmethod
    def _fit_single_arch(returns: np.ndarray, daily_rv: np.ndarray) -> dict[str, float]:
        try:
            from arch import arch_model  # type: ignore[import-not-found]
        except ImportError as exc:
            raise UnsupportedOptionError(
                "GARCH11 backend='arch' requires the optional 'arch' dependency. "
                "Install it with `pip install -e '.[arch]'` or use backend='scipy'."
            ) from exc

        r = np.asarray(returns, dtype=np.float64)
        rv = np.asarray(daily_rv, dtype=np.float64)
        sample_var = float(np.mean(r * r))
        mean_rv = float(np.mean(np.maximum(rv, 0.0)))
        if sample_var <= 1e-16 and mean_rv <= 1e-16:
            return {
                "omega": 0.0,
                "alpha": 0.0,
                "beta": 0.0,
                "uncond_var": 0.0,
                "last_var": 0.0,
                "rv_scale": 1.0,
            }

        # Rescale returns by 100 for numerical stability as recommended by arch
        am = arch_model(r * 100.0, mean="Zero", vol="GARCH", p=1, q=1, rescale=False)
        res = am.fit(disp="off", show_warning=False)
        omega = float(res.params["omega"]) / 10000.0
        alpha = float(res.params["alpha[1]"])
        beta = float(res.params["beta[1]"])
        phi = min(alpha + beta, 0.998)
        uncond_var = omega / max(1e-6, 1.0 - phi)
        cond_vol = res.conditional_volatility / 100.0
        last_var = float(cond_vol[-1] ** 2)
        rv_scale = float(np.clip(mean_rv / max(sample_var, 1e-16), 0.25, 4.0)) if sample_var > 1e-16 else 1.0
        return {
            "omega": omega,
            "alpha": alpha,
            "beta": beta,
            "uncond_var": uncond_var,
            "last_var": max(0.0, last_var),
            "rv_scale": rv_scale,
        }

    def fit(self, train_df: pl.DataFrame, horizon: int) -> "GARCH11Model":
        if train_df.height == 0:
            raise ModelFitError(f"Model '{self.name}' received empty training data.")
        fitter = self._fit_single_arch if self.backend == "arch" else self._fit_single_scipy
        self._params_by_asset = {}
        self._global_params = fitter(
            train_df["log_return"].to_numpy(),
            train_df["daily_rv"].to_numpy(),
        )
        for (asset_id,), grp in train_df.group_by(["asset_id"], maintain_order=True):
            grp_sorted = grp.sort("timestamp")
            self._params_by_asset[str(asset_id)] = fitter(
                grp_sorted["log_return"].to_numpy(),
                grp_sorted["daily_rv"].to_numpy(),
            )
        self.fit_metadata = {
            "backend": self.backend,
            "params_by_asset": self._params_by_asset,
        }
        self.is_fitted = True
        return self

    @staticmethod
    def _horizon_avg_forecast(one_step_var: float, uncond_var: float, phi: float, horizon: int) -> float:
        if horizon <= 1:
            return max(0.0, one_step_var)
        if abs(1.0 - phi) < 1e-9:
            return max(0.0, one_step_var)
        weight = (1.0 - (phi**horizon)) / (float(horizon) * (1.0 - phi))
        return max(0.0, uncond_var + weight * (one_step_var - uncond_var))

    def _predict_raw_variance(
        self,
        eval_df: pl.DataFrame,
        horizon: int,
        context_df: pl.DataFrame | None = None,
    ) -> np.ndarray:
        preds = np.empty(eval_df.height, dtype=np.float64)
        eval_indexed = eval_df.with_row_index("_row_idx")

        for (asset_id,), grp in eval_indexed.group_by(["asset_id"], maintain_order=True):
            aid = str(asset_id)
            p = self._params_by_asset.get(aid, self._global_params)
            omega = p["omega"]
            alpha = p["alpha"]
            beta = p["beta"]
            phi = alpha + beta
            uncond_var = p["uncond_var"]
            rv_scale = p["rv_scale"]

            grp_sorted = grp.sort("timestamp")
            row_idxs = grp_sorted["_row_idx"].to_numpy()
            eval_ts = grp_sorted["timestamp"].to_list()
            eval_r2 = (grp_sorted["log_return"].to_numpy().astype(np.float64)) ** 2
            eval_rv = grp_sorted["daily_rv"].to_numpy().astype(np.float64)

            if uncond_var <= 1e-16 and omega <= 1e-16:
                preds[row_idxs] = 0.0
                continue

            if context_df is not None and context_df.height > 0:
                ctx_asset = context_df.filter(pl.col("asset_id") == aid).sort("timestamp")
                ctx_ts = ctx_asset["timestamp"].to_list()
                ctx_r2 = (ctx_asset["log_return"].to_numpy().astype(np.float64)) ** 2
                first_eval_ts = eval_ts[0]
                sig2 = uncond_var
                for ts, r2_val in zip(ctx_ts, ctx_r2):
                    if ts < first_eval_ts:
                        sig2 = omega + alpha * float(r2_val) + beta * sig2
            else:
                sig2 = p["last_var"]

            out_vals = np.empty(len(eval_r2), dtype=np.float64)
            for i in range(len(eval_r2)):
                innov = float(eval_r2[i]) if eval_r2[i] > 0.0 else float(max(0.0, eval_rv[i]))
                one_step = omega + alpha * innov + beta * sig2
                sig2 = one_step
                avg_h = self._horizon_avg_forecast(one_step, uncond_var, phi, horizon)
                out_vals[i] = max(0.0, avg_h * rv_scale)
            preds[row_idxs] = out_vals

        return preds


class _SklearnLogVarianceModel(BaseVolatilityModel):
    """Base wrapper for scikit-learn models trained strictly on causal features."""

    def __init__(
        self,
        name: str,
        estimator: Any,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        feature_columns: tuple[str, ...] = CAUSAL_FEATURE_COLUMNS,
    ) -> None:
        super().__init__(name=name, params=params, seed=seed)
        self._estimator = estimator
        self.feature_columns = feature_columns
        self.use_log_target = bool(self.params.get("use_log_target", True))
        self.variance_floor = float(self.params.get("variance_floor", 1e-10))
        self._constant_target: float | None = None
        self._smearing_factor: float = 1.0

    def _extract_X(self, df: pl.DataFrame) -> np.ndarray:
        missing = [c for c in self.feature_columns if c not in df.columns]
        if missing:
            raise ModelFitError(f"Model '{self.name}' missing required feature columns: {missing}.")
        X = df.select(list(self.feature_columns)).to_numpy().astype(np.float64)
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    def fit(self, train_df: pl.DataFrame, horizon: int) -> "_SklearnLogVarianceModel":
        target_col = f"target_rv_h{horizon}"
        if target_col not in train_df.columns:
            raise ModelFitError(f"Training frame missing target column '{target_col}'.")
        valid = train_df.filter(pl.col(target_col).is_not_null())
        if valid.height == 0:
            raise ModelFitError(f"Model '{self.name}' has no non-null training targets.")

        X = self._extract_X(valid)
        y = valid[target_col].to_numpy().astype(np.float64)

        if float(np.ptp(y)) <= 1e-16:
            self._constant_target = max(0.0, float(np.mean(y)))
            self._smearing_factor = 1.0
            self.fit_metadata = {
                "constant_target": self._constant_target,
                "n_train": int(valid.height),
            }
            self.is_fitted = True
            return self

        self._constant_target = None
        if self.use_log_target:
            y_fit = np.log(np.maximum(y, self.variance_floor))
            self._estimator.fit(X, y_fit)
            y_hat_log = self._estimator.predict(X)
            resid = np.clip(y_fit - y_hat_log, -5.0, 5.0)
            # Duan smearing estimator fit strictly on training residuals
            self._smearing_factor = float(np.clip(np.mean(np.exp(resid)), 0.5, 2.0))
        else:
            self._estimator.fit(X, y)
            self._smearing_factor = 1.0

        self.fit_metadata = {
            "n_train": int(valid.height),
            "use_log_target": self.use_log_target,
            "smearing_factor": self._smearing_factor,
        }
        self.is_fitted = True
        return self

    def _predict_raw_variance(
        self,
        eval_df: pl.DataFrame,
        horizon: int,
        context_df: pl.DataFrame | None = None,
    ) -> np.ndarray:
        if self._constant_target is not None:
            return np.full(eval_df.height, self._constant_target, dtype=np.float64)
        X = self._extract_X(eval_df)
        pred = np.asarray(self._estimator.predict(X), dtype=np.float64)
        if self.use_log_target:
            pred_clipped = np.clip(pred, -30.0, 10.0)
            return self._smearing_factor * np.exp(pred_clipped)
        return np.maximum(0.0, pred)


class RidgeHARModel(_SklearnLogVarianceModel):
    """Regularized HAR-RV linear baseline (StandardScaler + Ridge inside Pipeline)."""

    def __init__(self, name: str = "ridge_har", params: dict[str, Any] | None = None, seed: int = 42):
        p = dict(params or {})
        alpha = float(p.get("alpha", 1.0))
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=alpha, random_state=seed)),
            ]
        )
        super().__init__(name=name, estimator=pipe, params=p, seed=seed)


class HGBRVolModel(_SklearnLogVarianceModel):
    """Non-linear gradient-boosted trees baseline via scikit-learn HistGradientBoostingRegressor."""

    def __init__(self, name: str = "hgbr", params: dict[str, Any] | None = None, seed: int = 42):
        p = dict(params or {})
        est = HistGradientBoostingRegressor(
            max_iter=int(p.get("max_iter", 120)),
            max_depth=int(p.get("max_depth", 4)),
            learning_rate=float(p.get("learning_rate", 0.05)),
            min_samples_leaf=int(p.get("min_samples_leaf", 10)),
            l2_regularization=float(p.get("l2_regularization", 1.0)),
            random_state=seed,
        )
        super().__init__(name=name, estimator=est, params=p, seed=seed)


def _causal_rolling_rv(
    eval_df: pl.DataFrame,
    context_df: pl.DataFrame | None,
    window: int,
) -> np.ndarray:
    """Compute strictly causal trailing W-day rolling mean RV for arbitrary window W."""
    preds = np.empty(eval_df.height, dtype=np.float64)
    eval_indexed = eval_df.with_row_index("_row_idx")
    for (asset_id,), grp in eval_indexed.group_by(["asset_id"], maintain_order=True):
        aid = str(asset_id)
        grp_sorted = grp.sort("timestamp")
        row_idxs = grp_sorted["_row_idx"].to_numpy()
        eval_ts = grp_sorted["timestamp"].to_list()
        eval_rv = grp_sorted["daily_rv"].to_numpy().astype(np.float64)

        pre_rv: list[float] = []
        if context_df is not None and context_df.height > 0:
            ctx = context_df.filter(pl.col("asset_id") == aid).sort("timestamp")
            first_ts = eval_ts[0]
            for ts, val in zip(ctx["timestamp"].to_list(), ctx["daily_rv"].to_numpy()):
                if ts < first_ts:
                    pre_rv.append(float(max(0.0, val)))

        combined = np.array(pre_rv + [max(0.0, float(v)) for v in eval_rv], dtype=np.float64)
        offset = len(pre_rv)
        out = np.empty(len(eval_rv), dtype=np.float64)
        for i in range(len(eval_rv)):
            end = offset + i + 1
            start = max(0, end - window)
            out[i] = float(np.mean(combined[start:end]))
        preds[row_idxs] = out
    return preds


def create_model(spec: ModelSpec, seed: int = 42) -> BaseVolatilityModel:
    """Factory instantiating a typed volatility model from a ModelSpec."""
    mtype = spec.type
    if mtype == "naive":
        return NaivePersistenceModel(name=spec.name, params=spec.params, seed=seed)
    if mtype == "rolling_historical":
        return RollingHistoricalModel(name=spec.name, params=spec.params, seed=seed)
    if mtype == "ewma":
        return EWMAModel(name=spec.name, params=spec.params, seed=seed)
    if mtype == "garch11":
        return GARCH11Model(name=spec.name, params=spec.params, seed=seed)
    if mtype == "ridge_har":
        return RidgeHARModel(name=spec.name, params=spec.params, seed=seed)
    if mtype == "hgbr":
        return HGBRVolModel(name=spec.name, params=spec.params, seed=seed)
    raise UnsupportedOptionError(f"Unsupported model type: '{mtype}'.")
