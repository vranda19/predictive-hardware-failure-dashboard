#!/usr/bin/env python3
"""Cross-platform Task Runner for Predictive Hardware Failure Dashboard.

Works seamlessly across Windows, Linux, and macOS.
Usage:
    python run.py init-db
    python run.py collect [--once]
    python run.py api [--port 8000]
    python run.py test
    python run.py train-model
    python run.py evaluate
    python run.py generate-synthetic-data
    python run.py export-parquet [--output data/parquet/telemetry.parquet]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def init_db():
    print("[*] Running database migration and initialization...")
    from storage.init_db import run_init
    run_init()


def run_collector(once: bool = False, interval: float | None = None):
    print(f"[*] Launching Telemetry Collector (once={once}, interval={interval})...")
    cmd = [sys.executable, str(PROJECT_ROOT / "collector" / "agent.py")]
    if once:
        cmd.append("--once")
    if interval:
        cmd.extend(["--interval", str(interval)])
    subprocess.run(cmd)


def run_api(host: str = "127.0.0.1", port: int = 8000, reload: bool = True):
    print(f"[*] Starting FastAPI server on http://{host}:{port} (reload={reload})...")
    print(f"[*] Interactive Dashboard available at: http://{host}:{port}/dashboard")
    print(f"[*] OpenAPI Specification available at: http://{host}:{port}/docs")
    import uvicorn
    uvicorn.run("api.app:app", host=host, port=port, reload=reload)


def run_tests():
    print("[*] Running comprehensive test suite with pytest...")
    cmd = [sys.executable, "-m", "pytest", "tests", "-v"]
    res = subprocess.run(cmd)
    sys.exit(res.returncode)


def train_model():
    print("[*] Retraining Isolation Forest Anomaly Detection Model...")
    from models.anomaly import get_anomaly_detector
    from data.synthetic_generator import generate_synthetic_telemetry
    from storage.db import get_all_telemetry_df

    df = get_all_telemetry_df()
    detector = get_anomaly_detector(auto_train=False)

    if len(df) < 50:
        print(f"[*] SQLite telemetry has {len(df)} records. Augmenting with 1000 nominal baseline samples...")
        synth = generate_synthetic_telemetry(n_samples=1000, anomaly_ratio=0.0, random_seed=42)
        detector.fit(synth)
    else:
        print(f"[*] Training directly on {len(df)} historical telemetry records...")
        detector.fit(df)

    saved_path = detector.save()
    print(f"[+] Model successfully trained and serialized to: {saved_path}")


def run_evaluate():
    print("[*] Running Capstone Quantitative Evaluation Harness...")
    from evaluation.evaluator import run_full_evaluation

    report = run_full_evaluation()
    print("\n" + "=" * 65)
    print("CAPSTONE TECHNICAL EVALUATION BENCHMARK RESULTS")
    print("=" * 65)
    print(json.dumps(report, indent=2))
    return report


def generate_synthetic_data():
    print("[*] Generating synthetic baseline and labeled evaluation datasets...")
    from data.synthetic_generator import save_benchmark_datasets

    b_file, t_file = save_benchmark_datasets()
    print(f"[+] Baseline training dataset saved to: {b_file}")
    print(f"[+] Labeled benchmark evaluation dataset saved to: {t_file}")


def export_parquet(output_path: str = "data/parquet/telemetry.parquet"):
    print(f"[*] Exporting SQLite telemetry to {output_path}...")
    from storage.db import export_to_parquet
    count = export_to_parquet(output_path)
    print(f"[+] Successfully exported {count} records to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Predictive Hardware Failure Dashboard Task Runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # init-db
    subparsers.add_parser("init-db", help="Initialize SQLite telemetry schema")

    # collect
    collect_parser = subparsers.add_parser("collect", help="Run edge telemetry collector")
    collect_parser.add_argument("--once", action="store_true", help="Perform single sample and exit")
    collect_parser.add_argument("--interval", type=float, default=None, help="Sampling interval in seconds")

    # api
    api_parser = subparsers.add_parser("api", help="Launch FastAPI web service and dashboard")
    api_parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind host")
    api_parser.add_argument("--port", type=int, default=8000, help="Bind port")
    api_parser.add_argument("--no-reload", action="store_true", help="Disable auto-reloading")

    # test
    subparsers.add_parser("test", help="Run unit and integration tests")

    # train-model
    subparsers.add_parser("train-model", help="Train Isolation Forest anomaly model")

    # evaluate
    subparsers.add_parser("evaluate", help="Execute evaluation benchmark & print metrics")

    # generate-synthetic-data
    subparsers.add_parser("generate-synthetic-data", help="Generate synthetic test datasets")

    # export-parquet
    parquet_parser = subparsers.add_parser("export-parquet", help="Export telemetry table to Parquet")
    parquet_parser.add_argument("--output", type=str, default="data/parquet/telemetry.parquet")

    args = parser.parse_args()

    if args.command == "init-db":
        init_db()
    elif args.command == "collect":
        run_collector(once=args.once, interval=args.interval)
    elif args.command == "api":
        run_api(host=args.host, port=args.port, reload=not args.no_reload)
    elif args.command == "test":
        run_tests()
    elif args.command == "train-model":
        train_model()
    elif args.command == "evaluate":
        run_evaluate()
    elif args.command == "generate-synthetic-data":
        generate_synthetic_data()
    elif args.command == "export-parquet":
        export_parquet(output_path=args.output)


if __name__ == "__main__":
    main()
