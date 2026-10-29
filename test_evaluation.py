import tempfile
from pathlib import Path
import pytest
import pandas as pd

from models.anomaly import IsolationForestAnomalyDetector
from data.synthetic_generator import generate_synthetic_telemetry
from evaluation.evaluator import evaluate_anomaly_detector, benchmark_edge_system


def test_evaluate_anomaly_detector():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create small test dataset
        csv_file = Path(tmpdir) / "test_benchmark.csv"
        df = generate_synthetic_telemetry(n_samples=150, anomaly_ratio=0.2, random_seed=42)
        df.to_csv(csv_file, index=False)

        detector = IsolationForestAnomalyDetector(n_estimators=30, random_state=42)
        detector.fit(df)

        results = evaluate_anomaly_detector(detector, benchmark_csv_path=csv_file)

        assert results["total_test_samples"] == 150
        assert "metrics" in results
        assert 0.0 <= results["metrics"]["accuracy"] <= 1.0
        assert 0.0 <= results["metrics"]["f1_score"] <= 1.0
        assert "confusion_matrix" in results
        cm = results["confusion_matrix"]
        assert (cm["true_positives"] + cm["false_positives"] + cm["true_negatives"] + cm["false_negatives"]) == 150


def test_benchmark_edge_system():
    with tempfile.TemporaryDirectory() as tmpdir:
        bench_db = Path(tmpdir) / "bench.db"
        results = benchmark_edge_system(db_path=str(bench_db))

        assert "sqlite_batch_write_100_ms" in results
        assert results["sqlite_batch_write_100_ms"] >= 0.0
        assert "sqlite_query_100_records_ms" in results
        assert "process_overhead" in results
        assert results["process_overhead"]["rss_memory_mb"] > 0
