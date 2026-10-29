import tempfile
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

from models.anomaly import IsolationForestAnomalyDetector
from data.synthetic_generator import generate_synthetic_telemetry


@pytest.fixture
def nominal_dataset():
    return generate_synthetic_telemetry(n_samples=200, anomaly_ratio=0.0, random_seed=42)


def test_detector_initialization():
    detector = IsolationForestAnomalyDetector(n_estimators=50, contamination=0.05)
    assert not detector.is_fitted
    assert detector.n_estimators == 50
    assert detector.contamination == 0.05


def test_detector_fit_and_persistence(nominal_dataset):
    with tempfile.TemporaryDirectory() as tmpdir:
        model_file = Path(tmpdir) / "test_iforest.joblib"

        detector = IsolationForestAnomalyDetector(n_estimators=30, random_state=42)
        detector.fit(nominal_dataset)

        assert detector.is_fitted
        assert detector.train_samples_count == len(nominal_dataset)
        assert len(detector.baseline_stats) > 0

        # Save model
        saved_path = detector.save(model_file)
        assert saved_path.exists()

        # Load into new instance
        new_detector = IsolationForestAnomalyDetector()
        new_detector.load(model_file)
        assert new_detector.is_fitted
        assert new_detector.n_estimators == 30


def test_detector_scoring_nominal_vs_anomalous(nominal_dataset):
    detector = IsolationForestAnomalyDetector(n_estimators=50, random_state=42)
    detector.fit(nominal_dataset)

    # 1. Nominal sample
    nominal_sample = {
        "cpu_usage_pct": 20.0,
        "memory_usage_pct": 45.0,
        "temperature_c": 48.0,
        "reallocated_sectors": 0,
        "pending_sectors": 0,
        "read_error_rate": 0.0,
        "disk_temp_c": 35.0,
        "load_avg_1m": 1.2,
    }
    nom_result = detector.score_sample(nominal_sample)
    assert 0.0 <= nom_result["anomaly_score"] <= 1.0
    assert not nom_result["is_anomaly"]
    assert nom_result["raw_decision_score"] > 0

    # 2. Extreme anomalous sample
    anomalous_sample = {
        "cpu_usage_pct": 99.8,
        "memory_usage_pct": 98.5,
        "temperature_c": 96.0,
        "reallocated_sectors": 45,
        "pending_sectors": 12,
        "read_error_rate": 80.0,
        "disk_temp_c": 62.0,
        "load_avg_1m": 14.5,
    }
    anom_result = detector.score_sample(anomalous_sample)
    assert anom_result["is_anomaly"]
    assert anom_result["anomaly_score"] > nom_result["anomaly_score"]
    assert len(anom_result["top_contributing_features"]) > 0


def test_detector_missing_feature_imputation(nominal_dataset):
    detector = IsolationForestAnomalyDetector(n_estimators=30, random_state=42)
    detector.fit(nominal_dataset)

    # Incomplete sample (e.g. thermal sensor missing on Windows, disk SMART not queried)
    incomplete_sample = {
        "cpu_usage_pct": 32.0,
        "memory_usage_pct": 52.0,
        "temperature_c": None,
        "reallocated_sectors": None,
    }
    result = detector.score_sample(incomplete_sample)
    assert 0.0 <= result["anomaly_score"] <= 1.0
