"""Synthetic Hardware Telemetry & Failure Scenario Generator.

Generates reproducible synthetic hardware telemetry datasets for controlled
model training, baseline benchmarking, and validation of anomaly detection and
risk scoring engines.

Generates realistic telemetry matching the edge SQLite schema:
- Nominal operating baseline (cyclic CPU, stable RAM, normal temperatures, healthy SMART)
- Stress & failure injection modes:
  1. Thermal runaway & cooling degradation
  2. Memory exhaustion / linear leak
  3. SMART storage mechanical/surface degradation
  4. CPU runaway starvation & load surge
"""

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd


def generate_synthetic_telemetry(
    n_samples: int = 1200,
    sampling_interval_seconds: int = 10,
    anomaly_ratio: float = 0.12,
    host_id: str = "synthetic-node-01",
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generates synthetic time-series telemetry with ground-truth failure/anomaly labels.

    Args:
        n_samples: Number of chronological samples to synthesize.
        sampling_interval_seconds: Frequency between consecutive observations.
        anomaly_ratio: Proportion of observations containing synthetic failure signatures.
        host_id: Virtual edge node identifier.
        random_seed: Seed for reproducibility.

    Returns:
        DataFrame conforming to the telemetry schema plus 'label' (0=normal, 1=anomaly)
        and 'anomaly_type'.
    """
    rng = np.random.RandomState(random_seed)
    base_time = datetime.now(timezone.utc) - timedelta(seconds=n_samples * sampling_interval_seconds)

    records: List[Dict] = []

    # State variables for temporal drift
    cur_cpu = 18.0
    cur_ram = 45.0
    cur_temp = 48.0
    cur_reallocated = 0
    cur_pending = 0
    cur_read_err = 0.0
    cur_disk_temp = 36.0

    # Determine anomaly blocks (continuous episodes of failure)
    n_anomaly_samples = int(n_samples * anomaly_ratio)
    anomaly_episode_len = max(15, n_anomaly_samples // 4)

    # Pick starting indices for 4 distinct anomaly episodes
    anomaly_starts = {
        int(n_samples * 0.20): "thermal_runaway",
        int(n_samples * 0.45): "memory_leak",
        int(n_samples * 0.70): "smart_degradation",
        int(n_samples * 0.88): "cpu_starvation",
    }

    active_anomaly: Optional[str] = None
    anomaly_countdown = 0

    for i in range(n_samples):
        ts = (base_time + timedelta(seconds=i * sampling_interval_seconds)).isoformat()

        # Check if an anomaly episode begins
        if i in anomaly_starts:
            active_anomaly = anomaly_starts[i]
            anomaly_countdown = anomaly_episode_len

        is_anomaly = 1 if (active_anomaly is not None and anomaly_countdown > 0) else 0
        anomaly_type = active_anomaly if is_anomaly else "nominal"

        # 1. Base nominal dynamics (random walk with mean reversion)
        diurnal = 5.0 * math.sin(2 * math.pi * i / 200)
        cpu_target = 22.0 + diurnal + rng.normal(0, 3.5)
        cur_cpu = np.clip(cur_cpu + 0.2 * (cpu_target - cur_cpu) + rng.normal(0, 2.0), 2.0, 85.0)

        ram_target = 48.0 + rng.normal(0, 1.0)
        cur_ram = np.clip(cur_ram + 0.1 * (ram_target - cur_ram) + rng.normal(0, 0.4), 25.0, 80.0)

        temp_target = 46.0 + (cur_cpu * 0.25) + rng.normal(0, 1.0)
        cur_temp = np.clip(cur_temp + 0.15 * (temp_target - cur_temp), 35.0, 75.0)

        # Baseline disk SMART
        disk_status = "PASSED"
        disk_temp_target = 34.0 + (cur_temp * 0.1)
        cur_disk_temp = np.clip(cur_disk_temp + 0.05 * (disk_temp_target - cur_disk_temp), 28.0, 50.0)

        # 2. Inject Anomaly Signatures
        if is_anomaly:
            step_in_episode = anomaly_episode_len - anomaly_countdown

            if active_anomaly == "thermal_runaway":
                # Temperature climbs rapidly towards junction max
                cur_temp = min(96.5, 58.0 + (step_in_episode * 2.5) + rng.normal(0, 1.2))
                cur_cpu = min(98.0, 75.0 + (step_in_episode * 1.5) + rng.normal(0, 2.0))
                cur_disk_temp = min(62.0, cur_disk_temp + 0.8)

            elif active_anomaly == "memory_leak":
                # Linear ramp towards 99%
                cur_ram = min(99.4, 55.0 + (step_in_episode * 2.8) + rng.normal(0, 0.3))
                cur_cpu = cur_cpu + rng.normal(0, 2.0)

            elif active_anomaly == "smart_degradation":
                # Reallocated and pending sectors accumulate
                cur_reallocated = min(80, cur_reallocated + rng.choice([0, 1, 2]))
                cur_pending = min(45, cur_pending + rng.choice([0, 1, 3]))
                cur_read_err = min(120.0, cur_read_err + rng.exponential(1.5))
                cur_disk_temp = min(58.0, 42.0 + (step_in_episode * 0.8))
                if cur_reallocated > 25 or cur_pending > 15:
                    disk_status = "FAILED"

            elif active_anomaly == "cpu_starvation":
                # CPU pegged at 98-100%
                cur_cpu = np.clip(96.0 + rng.normal(2.0, 1.0), 92.0, 100.0)
                cur_temp = min(88.0, 65.0 + (step_in_episode * 1.2))

            anomaly_countdown -= 1
            if anomaly_countdown <= 0:
                active_anomaly = None

        # Load averages tied to CPU
        load_1m = max(0.1, (cur_cpu / 100.0) * 8.0 + rng.normal(0, 0.2))
        load_5m = max(0.1, (cur_cpu / 100.0) * 6.5 + rng.normal(0, 0.15))
        load_15m = max(0.1, (cur_cpu / 100.0) * 5.0 + rng.normal(0, 0.1))

        # Assign entity class cyclically
        entity_class = "CPU" if i % 3 == 0 else ("RAM" if i % 3 == 1 else "HDD")

        record = {
            "timestamp": ts,
            "cpu_usage_pct": round(float(cur_cpu), 2),
            "memory_usage_pct": round(float(cur_ram), 2),
            "temperature_c": round(float(cur_temp), 2),
            "reallocated_sectors": int(cur_reallocated) if entity_class == "HDD" or cur_reallocated > 0 else None,
            "pending_sectors": int(cur_pending) if entity_class == "HDD" or cur_pending > 0 else None,
            "read_error_rate": round(float(cur_read_err), 2) if entity_class == "HDD" or cur_read_err > 0 else None,
            "disk_temp_c": round(float(cur_disk_temp), 2) if entity_class == "HDD" else None,
            "disk_status": disk_status if entity_class == "HDD" else None,
            "load_avg_1m": round(float(load_1m), 2),
            "load_avg_5m": round(float(load_5m), 2),
            "load_avg_15m": round(float(load_15m), 2),
            "entity_class": entity_class,
            "host_id": host_id,
            "label": is_anomaly,
            "anomaly_type": anomaly_type,
        }
        records.append(record)

    return pd.DataFrame(records)


def save_benchmark_datasets(output_dir: Path | str = "data/synthetic") -> Tuple[Path, Path]:
    """Generates and writes baseline and labeled benchmark evaluation datasets."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Baseline Nominal Dataset (Training reference, label=0)
    baseline_df = generate_synthetic_telemetry(
        n_samples=1000,
        anomaly_ratio=0.0,
        random_seed=101,
        host_id="baseline-training-node",
    )
    baseline_file = out_path / "baseline_nominal.csv"
    baseline_df.to_csv(baseline_file, index=False)

    # 2. Labeled Benchmark Evaluation Dataset (Testing reference, 15% mixed anomalies)
    benchmark_df = generate_synthetic_telemetry(
        n_samples=1500,
        anomaly_ratio=0.15,
        random_seed=202,
        host_id="benchmark-test-node",
    )
    benchmark_file = out_path / "benchmark_test_labeled.csv"
    benchmark_df.to_csv(benchmark_file, index=False)

    return baseline_file, benchmark_file


if __name__ == "__main__":
    b_file, t_file = save_benchmark_datasets()
    print(f"[+] Baseline training dataset saved: {b_file}")
    print(f"[+] Labeled benchmark evaluation dataset saved: {t_file}")
