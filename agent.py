"""Telemetry Collector ("Data-Collection Agent").

A robust, platform-agnostic background service designed for edge/on-device deployment.
Samples hardware telemetry every 10-15 seconds, buffers records into Chevtchenko-style
sliding windows, persists data to SQLite, and broadcasts updates to subscribers (e.g. WebSocket).
"""

import argparse
from datetime import datetime, timezone
import logging
import platform
import socket
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Ensure project root is in python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configs import load_config
from collector.readers import read_system_metrics, read_disk_smart
from features.windowing import SlidingWindowBuffer, compute_window_features
from storage.db import init_db, insert_telemetry_batch
from storage.models import TelemetryRecord

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger("collector.agent")


class TelemetryCollector:
    """Edge telemetry collection agent."""

    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path)
        self.live_mode: bool = self.config.get("mode", {}).get("live_mode", True)

        collector_cfg = self.config.get("collector", {})
        self.interval_seconds: float = float(collector_cfg.get("interval_seconds", 10))

        # Host ID resolution
        configured_host = collector_cfg.get("host_id", "auto")
        if not configured_host or configured_host == "auto":
            self.host_id: str = socket.gethostname() or platform.node() or "edge-device-01"
        else:
            self.host_id = configured_host

        self.smartctl_bin: str = collector_cfg.get("smartctl", {}).get("binary_path", "smartctl")
        self.smartctl_device: str = collector_cfg.get("smartctl", {}).get("device", "auto")

        storage_cfg = self.config.get("storage", {})
        self.db_path: str = storage_cfg.get("db_path", "data/telemetry.db")

        # Sliding window parameters (Chevtchenko et al.)
        win_cfg = self.config.get("windowing", {})
        self.window_size: int = int(win_cfg.get("window_size", 12))
        self.stride: int = int(win_cfg.get("stride", 1))

        # Separate sliding window buffers per entity class
        self.window_buffers: Dict[str, SlidingWindowBuffer] = {
            "CPU": SlidingWindowBuffer(window_size=self.window_size, stride=self.stride),
            "RAM": SlidingWindowBuffer(window_size=self.window_size, stride=self.stride),
            "HDD": SlidingWindowBuffer(window_size=self.window_size, stride=self.stride),
        }

        # Subscribers for live updates (e.g., API WebSockets)
        self.subscribers: List[Callable[[List[Dict[str, Any]]], None]] = []
        self._running: bool = False

        # Ensure database is initialized
        init_db(self.db_path)
        logger.info(
            "TelemetryCollector initialized (host_id=%s, interval=%.1fs, live_mode=%s, window_size=%d, stride=%d)",
            self.host_id,
            self.interval_seconds,
            self.live_mode,
            self.window_size,
            self.stride,
        )

    def subscribe(self, callback: Callable[[List[Dict[str, Any]]], None]) -> None:
        """Register a callback for streaming live telemetry rows."""
        self.subscribers.append(callback)

    def collect_once(self) -> List[Dict[str, Any]]:
        """Collects a single round of telemetry across CPU, RAM, and HDD entities.

        Persists rows to SQLite, updates sliding window buffers, and notifies subscribers.
        """
        now_iso = datetime.now(timezone.utc).isoformat()

        if not self.live_mode:
            logger.info("Synthetic mode stub: live_mode is false. No live samples collected.")
            return []

        # Read hardware telemetry
        sys_metrics = read_system_metrics()
        smart_metrics = read_disk_smart(self.smartctl_bin, self.smartctl_device)

        records_to_insert: List[Dict[str, Any]] = []

        # 1. CPU Entity Record
        cpu_record = {
            "timestamp": now_iso,
            "cpu_usage_pct": sys_metrics["cpu_usage_pct"],
            "memory_usage_pct": sys_metrics["memory_usage_pct"],
            "temperature_c": sys_metrics["temperature_c"],
            "reallocated_sectors": None,
            "pending_sectors": None,
            "read_error_rate": None,
            "disk_temp_c": None,
            "disk_status": None,
            "load_avg_1m": sys_metrics["load_avg_1m"],
            "load_avg_5m": sys_metrics["load_avg_5m"],
            "load_avg_15m": sys_metrics["load_avg_15m"],
            "entity_class": "CPU",
            "host_id": self.host_id,
        }
        records_to_insert.append(cpu_record)

        # 2. RAM Entity Record
        ram_record = {
            "timestamp": now_iso,
            "cpu_usage_pct": sys_metrics["cpu_usage_pct"],
            "memory_usage_pct": sys_metrics["memory_usage_pct"],
            "temperature_c": sys_metrics["temperature_c"],
            "reallocated_sectors": None,
            "pending_sectors": None,
            "read_error_rate": None,
            "disk_temp_c": None,
            "disk_status": None,
            "load_avg_1m": sys_metrics["load_avg_1m"],
            "load_avg_5m": sys_metrics["load_avg_5m"],
            "load_avg_15m": sys_metrics["load_avg_15m"],
            "entity_class": "RAM",
            "host_id": self.host_id,
        }
        records_to_insert.append(ram_record)

        # 3. HDD Entity Record
        hdd_record = {
            "timestamp": now_iso,
            "cpu_usage_pct": sys_metrics["cpu_usage_pct"],
            "memory_usage_pct": sys_metrics["memory_usage_pct"],
            "temperature_c": sys_metrics["temperature_c"],
            "reallocated_sectors": smart_metrics["reallocated_sectors"],
            "pending_sectors": smart_metrics["pending_sectors"],
            "read_error_rate": smart_metrics["read_error_rate"],
            "disk_temp_c": smart_metrics["disk_temp_c"],
            "disk_status": smart_metrics["disk_status"],
            "load_avg_1m": sys_metrics["load_avg_1m"],
            "load_avg_5m": sys_metrics["load_avg_5m"],
            "load_avg_15m": sys_metrics["load_avg_15m"],
            "entity_class": "HDD",
            "host_id": self.host_id,
        }
        records_to_insert.append(hdd_record)

        # Validate with TelemetryRecord Pydantic models
        validated_records = [TelemetryRecord(**rec).model_dump() for rec in records_to_insert]

        # Insert to SQLite in a single transaction
        insert_telemetry_batch(validated_records, self.db_path)

        # Append to sliding window buffers
        for rec in validated_records:
            e_class = rec["entity_class"]
            if e_class in self.window_buffers:
                emitted_window = self.window_buffers[e_class].append(rec)
                if emitted_window is not None:
                    features = compute_window_features(emitted_window)
                    logger.debug(
                        "Sliding window emitted for %s (size=%d, features_extracted=%d)",
                        e_class,
                        len(emitted_window),
                        len(features),
                    )

        # Broadcast to subscribers
        for sub in self.subscribers:
            try:
                sub(validated_records)
            except Exception as e:
                logger.error("Error in subscriber callback: %s", e)

        logger.info(
            "Collected sample: CPU=%.1f%%, RAM=%.1f%%, Temp=%s, Disks=%s (host=%s)",
            sys_metrics["cpu_usage_pct"],
            sys_metrics["memory_usage_pct"],
            f"{sys_metrics['temperature_c']}C" if sys_metrics["temperature_c"] is not None else "null",
            smart_metrics["disk_status"] or "null",
            self.host_id,
        )

        return validated_records

    def run(self) -> None:
        """Main edge collection loop."""
        self._running = True
        logger.info("Starting edge telemetry collector loop (interval: %.1fs)...", self.interval_seconds)

        try:
            while self._running:
                start_t = time.time()
                try:
                    self.collect_once()
                except Exception as e:
                    logger.exception("Unexpected error during telemetry collection: %s", e)

                elapsed = time.time() - start_t
                sleep_time = max(0.0, self.interval_seconds - elapsed)
                time.sleep(sleep_time)
        except KeyboardInterrupt:
            logger.info("Collector stopped by user interrupt.")
        finally:
            self._running = False
            logger.info("Collector loop terminated.")

    def stop(self) -> None:
        """Gracefully stop collection loop."""
        self._running = False


def main():
    parser = argparse.ArgumentParser(description="Edge Telemetry Collector Agent")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--once", action="store_true", help="Collect single sample and exit")
    parser.add_argument("--interval", type=float, default=None, help="Sampling interval override")
    args = parser.parse_args()

    collector = TelemetryCollector(config_path=args.config)
    if args.interval:
        collector.interval_seconds = args.interval

    if args.once:
        print("[*] Performing single telemetry collection...")
        records = collector.collect_once()
        print(f"[+] Successfully captured and persisted {len(records)} entity telemetry records:")
        for r in records:
            print(f"    - [{r['entity_class']}] CPU: {r['cpu_usage_pct']}%, RAM: {r['memory_usage_pct']}%, Temp: {r['temperature_c']}, SMART status: {r['disk_status']}")
    else:
        collector.run()


if __name__ == "__main__":
    main()
