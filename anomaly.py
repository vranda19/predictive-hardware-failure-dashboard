"""Isolation Forest Hardware Anomaly Detection Module.

Implements unsupervised anomaly detection for hardware telemetry streams.
Extracts multi-sensor feature vectors, imputes missing edge readings,
trains an Isolation Forest ensemble (Liu et al., 2008), normalizes anomaly
decision scores into a calibrated [0.0, 1.0] severity interval, and extracts
top contributing feature attributions for root-cause explainability.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("models.anomaly")

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "saved"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "isolation_forest.joblib"

# Primary sensor features monitored across hardware entities
FEATURE_COLUMNS = [
    "cpu_usage_pct",
    "memory_usage_pct",
    "temperature_c",
    "reallocated_sectors",
    "pending_sectors",
    "read_error_rate",
    "disk_temp_c",
    "load_avg_1m",
]

# Imputation defaults when sensor values are missing/null on edge devices
FEATURE_DEFAULTS = {
    "cpu_usage_pct": 20.0,
    "memory_usage_pct": 45.0,
    "temperature_c": 50.0,
    "reallocated_sectors": 0.0,
    "pending_sectors": 0.0,
    "read_error_rate": 0.0,
    "disk_temp_c": 35.0,
    "load_avg_1m": 1.0,
}


class IsolationForestAnomalyDetector:
    """Unsupervised Isolation Forest detector for multi-sensor hardware telemetry."""

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = 0.05,
        max_samples: Union[int, str] = 256,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_samples = max_samples
        self.random_state = random_state

        self.model: Optional[IsolationForest] = None
        self.scaler = StandardScaler()
        self.feature_columns: List[str] = list(FEATURE_COLUMNS)
        self.baseline_stats: Dict[str, Dict[str, float]] = {}
        self.is_fitted: bool = False
        self.trained_at: Optional[str] = None
        self.train_samples_count: int = 0

    def _prepare_matrix(
        self, data: Union[pd.DataFrame, List[Dict[str, Any]], np.ndarray]
    ) -> np.ndarray:
        """Converts heterogeneous records into a clean numerical matrix with imputation."""
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        elif isinstance(data, np.ndarray):
            arr = np.nan_to_num(data, nan=0.0)
            return arr
        else:
            raise TypeError(f"Unsupported data format: {type(data)}")

        # Ensure all expected feature columns exist and fill missing values
        df_feats = pd.DataFrame(index=df.index)
        for col in self.feature_columns:
            fill_val = (
                self.baseline_stats.get(col, {}).get("mean")
                if self.is_fitted and col in self.baseline_stats
                else FEATURE_DEFAULTS.get(col, 0.0)
            )
            if col in df.columns:
                df_feats[col] = pd.to_numeric(df[col], errors="coerce").fillna(fill_val)
            else:
                df_feats[col] = fill_val

        matrix = df_feats[self.feature_columns].to_numpy(dtype=float, copy=True)
        return matrix

    def fit(
        self,
        data: Union[pd.DataFrame, List[Dict[str, Any]]],
        feature_columns: Optional[List[str]] = None,
    ) -> "IsolationForestAnomalyDetector":
        """Fits the Isolation Forest on nominal or historical operational telemetry."""
        if feature_columns:
            self.feature_columns = list(feature_columns)

        X = self._prepare_matrix(data)
        if len(X) == 0:
            raise ValueError("Cannot train IsolationForest on empty dataset.")

        # Compute baseline descriptive stats for z-score attribution
        self.baseline_stats = {}
        for idx, col in enumerate(self.feature_columns):
            vals = X[:, idx]
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals))
            self.baseline_stats[col] = {
                "mean": round(mean_val, 4),
                "std": round(std_val if std_val > 1e-6 else 1.0, 4),
                "min": round(float(np.min(vals)), 4),
                "max": round(float(np.max(vals)), 4),
            }

        # Fit Scaler and Isolation Forest
        X_scaled = self.scaler.fit_transform(X)

        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_samples=min(self.max_samples, len(X)) if isinstance(self.max_samples, int) else self.max_samples,
            random_state=self.random_state,
            n_jobs=1,
        )
        self.model.fit(X_scaled)

        self.is_fitted = True
        self.trained_at = datetime.now(timezone.utc).isoformat()
        self.train_samples_count = len(X)
        logger.info(
            "IsolationForest fitted on %d samples (contamination=%.3f, estimators=%d)",
            self.train_samples_count,
            self.contamination,
            self.n_estimators,
        )
        return self

    def score_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluates a single telemetry sample.

        Returns:
            Dict containing:
            - is_anomaly (bool): Whether sample is classified as an outlier
            - anomaly_score (float): Calibrated score [0.0 - 1.0] where 1.0 = extreme anomaly
            - raw_score (float): Decision function output from scikit-learn
            - top_contributing_features (list): Ranked list of features driving the anomaly
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model is not fitted yet. Call fit() or load() before scoring.")

        X = self._prepare_matrix([sample])
        X_scaled = self.scaler.transform(X)

        # raw_score: positive = normal, negative = anomalous
        raw_score = float(self.model.decision_function(X_scaled)[0])
        # prediction: 1 = inlier/normal, -1 = outlier/anomalous
        pred = int(self.model.predict(X_scaled)[0])
        is_anomaly = bool(pred == -1)

        # Calibrated anomaly score [0.0, 1.0]:
        # raw_score typically falls in [-0.4, 0.4]. An offset sigmoid maps to [0, 1]
        calibrated_score = float(np.clip(1.0 / (1.0 + np.exp(6.0 * raw_score)), 0.0, 1.0))

        # Calculate feature attribution via z-scores relative to baseline
        attributions = []
        for idx, col in enumerate(self.feature_columns):
            val = float(X[0, idx])
            stats = self.baseline_stats.get(col, {"mean": 0.0, "std": 1.0})
            mean_val = stats["mean"]
            std_val = max(stats["std"], 1e-4)
            z_score = abs(val - mean_val) / std_val
            attributions.append({
                "feature": col,
                "current_value": round(val, 2),
                "baseline_mean": mean_val,
                "z_deviation": round(float(z_score), 2),
            })

        # Rank by highest deviation
        attributions.sort(key=lambda x: x["z_deviation"], reverse=True)

        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": round(calibrated_score, 4),
            "raw_decision_score": round(raw_score, 4),
            "top_contributing_features": attributions[:3],
        }

    def predict_batch(
        self, data: Union[pd.DataFrame, List[Dict[str, Any]]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Batch prediction returning binary labels (1=anomaly, 0=normal) and anomaly scores."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model is not fitted yet.")

        X = self._prepare_matrix(data)
        X_scaled = self.scaler.transform(X)

        raw_scores = self.model.decision_function(X_scaled)
        preds = self.model.predict(X_scaled)

        # Map: 1 -> 0 (normal), -1 -> 1 (anomaly)
        binary_anomalies = np.where(preds == -1, 1, 0)
        calibrated_scores = np.clip(1.0 / (1.0 + np.exp(6.0 * raw_scores)), 0.0, 1.0)

        return binary_anomalies, calibrated_scores

    def save(self, filepath: Optional[Union[Path, str]] = None) -> Path:
        """Serializes the fitted detector to disk."""
        target_path = Path(filepath) if filepath else DEFAULT_MODEL_PATH
        target_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_columns": self.feature_columns,
            "baseline_stats": self.baseline_stats,
            "n_estimators": self.n_estimators,
            "contamination": self.contamination,
            "max_samples": self.max_samples,
            "random_state": self.random_state,
            "trained_at": self.trained_at,
            "train_samples_count": self.train_samples_count,
            "is_fitted": self.is_fitted,
        }
        joblib.dump(payload, str(target_path))
        logger.info("Detector successfully saved to: %s", target_path)
        return target_path

    def load(self, filepath: Optional[Union[Path, str]] = None) -> "IsolationForestAnomalyDetector":
        """Deserializes detector state from disk."""
        target_path = Path(filepath) if filepath else DEFAULT_MODEL_PATH
        if not target_path.exists():
            raise FileNotFoundError(f"Model file not found at: {target_path}")

        payload = joblib.load(str(target_path))
        self.model = payload["model"]
        self.scaler = payload["scaler"]
        self.feature_columns = payload["feature_columns"]
        self.baseline_stats = payload["baseline_stats"]
        self.n_estimators = payload.get("n_estimators", 100)
        self.contamination = payload.get("contamination", 0.05)
        self.max_samples = payload.get("max_samples", 256)
        self.random_state = payload.get("random_state", 42)
        self.trained_at = payload.get("trained_at")
        self.train_samples_count = payload.get("train_samples_count", 0)
        self.is_fitted = payload.get("is_fitted", True)

        logger.info("Detector successfully loaded from: %s", target_path)
        return self


# Global singleton instance for runtime inference
_detector_instance: Optional[IsolationForestAnomalyDetector] = None


def get_anomaly_detector(auto_train: bool = True) -> IsolationForestAnomalyDetector:
    """Returns the application singleton anomaly detector, bootstrapping if necessary."""
    global _detector_instance
    if _detector_instance is not None and _detector_instance.is_fitted:
        return _detector_instance

    detector = IsolationForestAnomalyDetector()

    # Try loading pre-trained artifact
    if DEFAULT_MODEL_PATH.exists():
        try:
            detector.load(DEFAULT_MODEL_PATH)
            _detector_instance = detector
            return _detector_instance
        except Exception as e:
            logger.warning("Failed to load existing model file (%s), re-training: %s", DEFAULT_MODEL_PATH, e)

    if auto_train:
        # Train on synthetic baseline or existing SQLite telemetry
        from data.synthetic_generator import generate_synthetic_telemetry

        logger.info("Bootstrapping baseline IsolationForest model on nominal telemetry...")
        baseline_df = generate_synthetic_telemetry(n_samples=800, anomaly_ratio=0.0, random_seed=42)
        detector.fit(baseline_df)
        detector.save(DEFAULT_MODEL_PATH)

    _detector_instance = detector
    return _detector_instance
