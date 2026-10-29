"""Evaluation Harness & Performance Benchmarking Engine.

Performs quantitative assessment according to university capstone requirements:
1. ML Classification Metrics on Labeled Ground-Truth Hardware Failure Benchmark:
   - Accuracy, Precision, Recall, F1-Score, MCC, Confusion Matrix
2. Edge System Operational Latencies & Resource Overhead:
   - SQLite batch write latency (ms)
   - SQLite query retrieval latency (ms)
   - Chevtchenko sliding window feature extraction latency (ms)
   - Isolation Forest inference latency per sample (ms)
   - Telemetry reader sampling latency (ms)
   - Host CPU & RSS Memory footprint of the monitoring process
"""

from datetime import datetime, timezone
import json
import logging
import sys
from pathlib import Path

# Ensure project root is in python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import psutil

from models.anomaly import IsolationForestAnomalyDetector, get_anomaly_detector
from features.windowing import compute_window_features, SlidingWindowBuffer
from storage.db import (
    get_db_connection,
    init_db,
    insert_telemetry_batch,
    get_latest_telemetry,
)

logger = logging.getLogger("evaluation.evaluator")


def evaluate_anomaly_detector(
    detector: IsolationForestAnomalyDetector,
    benchmark_csv_path: Union[Path, str] = "data/synthetic/benchmark_test_labeled.csv",
) -> Dict[str, Any]:
    """Evaluates Isolation Forest on labeled test benchmark dataset.

    Calculates genuine Accuracy, Precision, Recall, F1-Score, and MCC without fabricating results.
    """
    path = Path(benchmark_csv_path)
    if not path.exists():
        from data.synthetic_generator import save_benchmark_datasets
        logger.info("Benchmark test dataset not found. Generating...")
        save_benchmark_datasets()

    df = pd.read_csv(path)
    if "label" not in df.columns:
        raise ValueError("Benchmark dataset must contain ground-truth 'label' column (0 or 1).")

    y_true = df["label"].to_numpy(dtype=int)

    # Measure inference timing
    t0 = time.perf_counter()
    y_pred, anomaly_scores = detector.predict_batch(df)
    total_inference_time = time.perf_counter() - t0
    avg_latency_ms = (total_inference_time / len(df)) * 1000.0

    # Compute confusion matrix elements
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    total = len(y_true)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Matthews Correlation Coefficient (MCC)
    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = float((tp * tn - fp * fn) / denom) if denom > 0 else 0.0

    # Breakdown by anomaly type if available
    breakdown_by_type: Dict[str, Dict[str, Any]] = {}
    if "anomaly_type" in df.columns:
        for a_type in df["anomaly_type"].unique():
            mask = df["anomaly_type"] == a_type
            sub_true = y_true[mask]
            sub_pred = y_pred[mask]
            sub_total = int(len(sub_true))
            sub_detected = int(np.sum(sub_pred == 1))
            breakdown_by_type[str(a_type)] = {
                "sample_count": sub_total,
                "detected_count": sub_detected,
                "detection_rate": round(sub_detected / sub_total, 4) if sub_total > 0 else 0.0,
            }

    results = {
        "dataset_name": path.name,
        "total_test_samples": total,
        "ground_truth_anomalies": int(np.sum(y_true == 1)),
        "ground_truth_nominals": int(np.sum(y_true == 0)),
        "confusion_matrix": {
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
        },
        "metrics": {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "mcc": round(mcc, 4),
        },
        "performance": {
            "total_inference_time_seconds": round(total_inference_time, 4),
            "avg_latency_ms_per_sample": round(avg_latency_ms, 3),
            "throughput_samples_per_second": round(total / total_inference_time, 1) if total_inference_time > 0 else 0.0,
        },
        "breakdown_by_failure_mode": breakdown_by_type,
    }
    return results


def benchmark_edge_system(db_path: str = "data/telemetry_benchmark.db") -> Dict[str, Any]:
    """Measures live database I/O latency, feature engineering overhead, and process footprint."""
    from collector.readers import read_system_metrics

    test_db = Path(db_path)
    if test_db.exists():
        test_db.unlink()
    init_db(test_db)

    # 1. Hardware Reader Latency
    t0 = time.perf_counter()
    metrics_sample = read_system_metrics()
    reader_latency_ms = (time.perf_counter() - t0) * 1000.0

    # 2. SQLite Batch Write Latency (100 records)
    records_to_write = []
    now_iso = datetime.now(timezone.utc).isoformat()
    for i in range(100):
        records_to_write.append({
            "timestamp": now_iso,
            "cpu_usage_pct": 25.0 + (i % 20),
            "memory_usage_pct": 50.0 + (i % 10),
            "temperature_c": 48.0,
            "reallocated_sectors": 0,
            "pending_sectors": 0,
            "read_error_rate": 0.0,
            "disk_temp_c": 35.0,
            "disk_status": "PASSED",
            "load_avg_1m": 1.2,
            "load_avg_5m": 1.1,
            "load_avg_15m": 1.0,
            "entity_class": "CPU" if i % 2 == 0 else "RAM",
            "host_id": "bench-node",
        })

    t0 = time.perf_counter()
    written_count = insert_telemetry_batch(records_to_write, db_path=test_db)
    db_write_latency_ms = (time.perf_counter() - t0) * 1000.0
    write_latency_per_record_ms = db_write_latency_ms / max(1, written_count)

    # 3. SQLite Query Retrieval Latency (latest 100 records)
    t0 = time.perf_counter()
    queried = get_latest_telemetry(limit=100, db_path=test_db)
    db_query_latency_ms = (time.perf_counter() - t0) * 1000.0

    # 4. Chevtchenko Sliding Window Feature Extraction (window_size=12)
    win_buffer = SlidingWindowBuffer(window_size=12, stride=1)
    for r in records_to_write[:12]:
        win_buffer.append(r)
    full_win = win_buffer.get_current_window()

    t0 = time.perf_counter()
    features = compute_window_features(full_win)
    feat_latency_ms = (time.perf_counter() - t0) * 1000.0

    # 5. Process Memory & CPU Footprint
    proc = psutil.Process()
    mem_info = proc.memory_info()
    rss_mb = mem_info.rss / (1024 * 1024)
    vms_mb = mem_info.vms / (1024 * 1024)
    cpu_percent = proc.cpu_percent(interval=0.1)

    # Clean up benchmark DB
    if test_db.exists():
        test_db.unlink()

    return {
        "hardware_reader_latency_ms": round(reader_latency_ms, 2),
        "sqlite_batch_write_100_ms": round(db_write_latency_ms, 2),
        "sqlite_write_per_record_ms": round(write_latency_per_record_ms, 3),
        "sqlite_query_100_records_ms": round(db_query_latency_ms, 2),
        "sliding_window_feature_extraction_ms": round(feat_latency_ms, 2),
        "sliding_window_features_count": len(features),
        "process_overhead": {
            "rss_memory_mb": round(rss_mb, 2),
            "vms_memory_mb": round(vms_mb, 2),
            "cpu_utilization_pct": round(cpu_percent, 2),
        },
    }


def run_full_evaluation() -> Dict[str, Any]:
    """Runs end-to-end evaluation and returns consolidated benchmark report."""
    detector = get_anomaly_detector(auto_train=True)
    ml_eval = evaluate_anomaly_detector(detector)
    sys_bench = benchmark_edge_system()

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_architecture": "Isolation Forest (scikit-learn)",
        "model_parameters": {
            "n_estimators": detector.n_estimators,
            "contamination": detector.contamination,
            "max_samples": detector.max_samples,
            "random_state": detector.random_state,
            "feature_columns": detector.feature_columns,
        },
        "machine_learning_evaluation": ml_eval,
        "system_performance_benchmark": sys_bench,
    }
    return report


# Helper type
from typing import Union
Union_Path_str = Union[Path, str]

if __name__ == "__main__":
    report = run_full_evaluation()
    print("\n" + "=" * 60)
    print("CAPSTONE TECHNICAL EVALUATION BENCHMARK RESULTS")
    print("=" * 60)
    print(json.dumps(report, indent=2))
