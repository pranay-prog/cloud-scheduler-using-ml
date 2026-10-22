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

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> Dict[str, float]:
        """Fit Random Forest on feature matrix X and duration target y."""
        if isinstance(X, pd.DataFrame):
            # Ensure consistent feature column order
            cols = [c for c in self.feature_names if c in X.columns]
            if len(cols) == len(self.feature_names):
                X_mat = X[cols].values
            else:
                X_mat = X.values
        else:
            X_mat = np.asarray(X, dtype=float)

        y_vec = np.asarray(y, dtype=float).ravel()

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
                X_mat = X[cols].values
            else:
                X_mat = X.values
        else:
            X_mat = np.asarray(X, dtype=float)
            if X_mat.ndim == 1:
                X_mat = X_mat.reshape(1, -1)

        preds = self.model.predict(X_mat)
        return np.clip(preds, 0.1, None)

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
