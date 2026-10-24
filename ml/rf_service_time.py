"""Random Forest Service Time Predictor.

Refactors the original empirical Random Forest model from main.py and app.py
into a reusable predictor that estimates individual task durations and derives
second-order service variability (C_s^2) for the G/G/k queueing model.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def generate_synthetic_workload(
    n_samples: int = 1500,
    noise_level: float = 3.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic cloud tasks with CPU, RAM, WorkloadType, and actual execution duration.

    Features:
        CPU: Number of allocated virtual cores (1 to 16)
        RAM: Memory allocation in GB (2 to 64)
        Type: Workload category (1=Web/Read, 2=Compute, 3=Database/Batch)
    Target:
        Actual_Time: Stochastic execution duration in seconds
    """
    rng = np.random.RandomState(seed)

    cpu_req = rng.randint(1, 17, size=n_samples)
    ram_req = rng.randint(2, 65, size=n_samples)
    workload_type = rng.randint(1, 4, size=n_samples)

    # Underlying physics model representing cloud service execution
    base_time = (cpu_req * 1.5) + (ram_req * 0.4) + (workload_type * 4.5)
    stochastic_noise = rng.normal(0, noise_level, size=n_samples)
    actual_time = np.clip(base_time + stochastic_noise, 0.5, None)

    return pd.DataFrame({
        "CPU": cpu_req,
        "RAM": ram_req,
        "Type": workload_type,
        "Actual_Time": actual_time,
    })


class RFServiceTimePredictor:
    """Random Forest predictor for cloud service execution durations.

    Predicts task execution times from hardware request features and calculates
    estimated service rate mu = 1 / E[S] and service variability C_s^2 for G/G/k queueing.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 12,
        random_state: int = 42,
    ):
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.random_state = int(random_state)
        self.feature_names = ["CPU", "RAM", "Type"]
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        self.is_fitted = False
        self.training_metrics: Dict[str, float] = {}
        self.upper_residual_quantile: float = 0.0
        self.calibration_quantile: float = 0.95

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> Dict[str, float]:
        """Fit Random Forest on feature matrix X and duration target y."""
        if isinstance(X, pd.DataFrame):
            # Ensure consistent feature column order
            cols = [c for c in self.feature_names if c in X.columns]
            if len(cols) != len(self.feature_names):
                raise ValueError(f"X must contain feature columns {self.feature_names}.")
            X_mat = X[cols].to_numpy(dtype=float)
        else:
            X_mat = np.asarray(X, dtype=float)
        if X_mat.ndim != 2 or X_mat.shape[1] != len(self.feature_names):
            raise ValueError(f"X must have {len(self.feature_names)} feature columns.")

        y_vec = np.asarray(y, dtype=float).ravel()
        if len(y_vec) != len(X_mat):
            raise ValueError("X and y must have the same number of rows")
        if len(y_vec) == 0 or not np.all(np.isfinite(X_mat)) or not np.all(np.isfinite(y_vec)):
            raise ValueError("training data must be non-empty and contain only finite values")
        if np.any(y_vec <= 0):
            raise ValueError("runtime targets must be strictly positive")

        self.model.fit(X_mat, y_vec)
        self.is_fitted = True

        # Compute training goodness-of-fit metrics
        preds = self.model.predict(X_mat)
        self.training_metrics = {
            "MAE": float(mean_absolute_error(y_vec, preds)),
            "MSE": float(mean_squared_error(y_vec, preds)),
            "R2": float(r2_score(y_vec, preds)),
        }
        return self.training_metrics

    def predict(self, X: pd.DataFrame | np.ndarray | Sequence[Any]) -> np.ndarray:
        """Predict execution durations for incoming task requests."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() before predict().")

        if isinstance(X, pd.DataFrame):
            cols = [c for c in self.feature_names if c in X.columns]
            if len(cols) == len(self.feature_names):
                X_mat = X[cols].to_numpy(dtype=float)
            else:
                raise ValueError(f"X must contain feature columns {self.feature_names}.")
        else:
            X_mat = np.asarray(X, dtype=float)
            if X_mat.ndim == 1:
                X_mat = X_mat.reshape(1, -1)

        if X_mat.ndim != 2 or X_mat.shape[1] != len(self.feature_names):
            raise ValueError(f"X must have {len(self.feature_names)} feature columns.")
        if not np.all(np.isfinite(X_mat)):
            raise ValueError("X contains missing or non-finite task features.")
        preds = np.asarray(self.model.predict(X_mat), dtype=float)
        preds = np.nan_to_num(preds, nan=0.1, posinf=0.1, neginf=0.1)
        return np.clip(preds, 0.1, None)

    def calibrate(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray, quantile: float = 0.95) -> float:
        """Fit a one-sided residual offset on a disjoint calibration split."""
        if not 0.0 < quantile < 1.0:
            raise ValueError("quantile must be strictly between 0 and 1")
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calibration.")
        actual = np.asarray(y, dtype=float).ravel()
        pred = self.predict(X)
        if actual.size != pred.size:
            raise ValueError("calibration features and targets have different lengths")
        valid = np.isfinite(actual) & np.isfinite(pred)
        residuals = actual[valid] - pred[valid]
        if residuals.size == 0:
            self.upper_residual_quantile = 0.0
        else:
            # 'higher' is conservative for finite calibration samples. For a
            # single sample, use its residual; this is not a coverage guarantee.
            self.upper_residual_quantile = max(0.0, float(np.quantile(residuals, quantile, method="higher")))
        self.calibration_quantile = float(quantile)
        return self.upper_residual_quantile

    def predict_upper(self, X: pd.DataFrame | np.ndarray | Sequence[Any]) -> np.ndarray:
        """Return max(0, prediction) + max(0, calibrated residual quantile)."""
        prediction = np.asarray(self.predict(X), dtype=float)
        prediction = np.nan_to_num(prediction, nan=0.0, posinf=0.0, neginf=0.0)
        return np.maximum(0.0, prediction) + max(0.0, self.upper_residual_quantile)

    def estimate_service_moments(
        self,
        X: pd.DataFrame | np.ndarray | Sequence[Any],
    ) -> Tuple[float, float, float, float]:
        """Predict execution durations and compute service moments for G/G/k queueing.

        Returns
        -------
        Tuple[float, float, float, float]
            (mu_rate, mean_service, var_service, cs2)
            Where:
                mu_rate = 1 / E[S]
                mean_service = E[S]
                var_service = Var(S)
                cs2 = Var(S) / (E[S]^2)
        """
        predictions = self.predict(X)
        mean_s = float(np.mean(predictions))
        var_s = float(np.var(predictions, ddof=1)) if len(predictions) > 1 else 0.0

        if mean_s <= 0:
            raise ValueError("Predicted mean service duration must be strictly positive.")

        mu_rate = 1.0 / mean_s
        cs2 = float(var_s / (mean_s ** 2))
        return mu_rate, mean_s, var_s, cs2


def train_rf_predictor(
    n_samples: int = 1500,
    noise_level: float = 3.0,
    seed: int = 42,
) -> Tuple[RFServiceTimePredictor, Dict[str, float]]:
    """Convenience helper to generate training workload and fit the RF predictor."""
    df = generate_synthetic_workload(n_samples=n_samples, noise_level=noise_level, seed=seed)
    X = df[["CPU", "RAM", "Type"]]
    y = df["Actual_Time"]

    predictor = RFServiceTimePredictor(n_estimators=100, max_depth=12, random_state=seed)
    metrics = predictor.fit(X, y)
    return predictor, metrics


def train_calibrate_test_predictor(
    data: Optional[pd.DataFrame] = None,
    n_samples: int = 1500,
    noise_level: float = 3.0,
    seed: int = 42,
    quantile: float = 0.95,
    train_data: Optional[pd.DataFrame] = None,
    calibration_data: Optional[pd.DataFrame] = None,
    validation_data: Optional[pd.DataFrame] = None,
    test_data: Optional[pd.DataFrame] = None,
) -> Tuple[RFServiceTimePredictor, Dict[str, float], Dict[str, np.ndarray]]:
    """Train, calibrate, validate, and test on disjoint data.

    Default split proportions are 60% train, 15% calibration, 10% validation,
    and 15% test. Explicit dataframes may be supplied for independent streams.
    Test metrics are never used to fit the model or calibrate its residual offset.
    """
    required = ["CPU", "RAM", "Type", "Actual_Time"]

    def clean(frame: pd.DataFrame) -> pd.DataFrame:
        if any(c not in frame.columns for c in required):
            raise ValueError(f"data must contain columns {required}")
        clean_frame = frame[required].replace([np.inf, -np.inf], np.nan).dropna()
        clean_frame = clean_frame[clean_frame["Actual_Time"] > 0]
        return clean_frame

    explicit = [train_data, calibration_data, validation_data, test_data]
    if any(part is not None for part in explicit):
        if any(part is None for part in explicit):
            raise ValueError("provide all four train, calibration, validation, and test dataframes")
        train_df, cal_df, val_df, test_df = (clean(part) for part in explicit if part is not None)
        if any(len(part) == 0 for part in (train_df, cal_df, val_df, test_df)):
            raise ValueError("all four data splits must contain at least one valid row")
    else:
        df = data.copy() if data is not None else generate_synthetic_workload(n_samples, noise_level, seed)
        df = clean(df)
        if len(df) < 8:
            raise ValueError("at least eight valid rows are required for train/calibration/validation/test splits")
        indices = np.arange(len(df))
        train_idx, remaining_idx = train_test_split(indices, test_size=0.4, random_state=seed)
        cal_val_idx, test_idx = train_test_split(remaining_idx, test_size=0.375, random_state=seed + 1)
        cal_idx, val_idx = train_test_split(cal_val_idx, test_size=0.4, random_state=seed + 2)
        train_df = df.iloc[train_idx]
        cal_df = df.iloc[cal_idx]
        val_df = df.iloc[val_idx]
        test_df = df.iloc[test_idx]

    predictor = RFServiceTimePredictor(n_estimators=100, max_depth=12, random_state=seed)
    feature_cols = ["CPU", "RAM", "Type"]
    predictor.fit(train_df[feature_cols], train_df["Actual_Time"])
    predictor.calibrate(cal_df[feature_cols], cal_df["Actual_Time"], quantile=quantile)
    val_pred = predictor.predict(val_df[feature_cols])
    val_upper = predictor.predict_upper(val_df[feature_cols])
    y_val = val_df["Actual_Time"].to_numpy(dtype=float)
    test_pred = predictor.predict(test_df[feature_cols])
    test_upper = predictor.predict_upper(test_df[feature_cols])
    y_test = test_df["Actual_Time"].to_numpy(dtype=float)
    result = {
        "MAE": float(mean_absolute_error(y_test, test_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, test_pred))),
        "upper_coverage": float(np.mean(y_test <= test_upper)),
        "validation_MAE": float(mean_absolute_error(y_val, val_pred)),
        "validation_RMSE": float(np.sqrt(mean_squared_error(y_val, val_pred))),
        "validation_upper_coverage": float(np.mean(y_val <= val_upper)),
        "calibration_quantile": float(quantile),
        "upper_residual_quantile": float(predictor.upper_residual_quantile),
        "n_train": float(len(train_df)),
        "n_calibration": float(len(cal_df)),
        "n_validation": float(len(val_df)),
        "n_test": float(len(test_df)),
    }
    arrays = {"test_actual": y_test, "test_prediction": test_pred, "test_upper": test_upper,
              "validation_actual": y_val, "validation_prediction": val_pred, "validation_upper": val_upper}
    return predictor, result, arrays
