"""Hardware Failure Risk Assessment & Alert Engine.

Combines real-time telemetry, Chevtchenko-style sliding window statistical
features (slopes, peak-to-peak amplitude), and Isolation Forest anomaly scores
to synthesize multi-subsystem risk indicators across CPU, RAM, and Storage (SMART).
Generates actionable engineering alerts with targeted remediation recommendations.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from storage.db import insert_alert, insert_anomaly_event

logger = logging.getLogger("models.risk_engine")

# Deduplication cache to prevent repeating the exact same alert within N seconds
_RECENT_ALERTS_CACHE: Dict[str, float] = {}
DEDUP_WINDOW_SECONDS = 60.0


class HardwareRiskEngine:
    """Multi-subsystem hardware failure risk assessment and alert generation engine."""

    def __init__(
        self,
        cpu_temp_warn_threshold: float = 75.0,
        cpu_temp_crit_threshold: float = 88.0,
        cpu_usage_warn_threshold: float = 85.0,
        cpu_usage_crit_threshold: float = 95.0,
        mem_usage_warn_threshold: float = 80.0,
        mem_usage_crit_threshold: float = 92.0,
        disk_temp_warn_threshold: float = 50.0,
        anomaly_warn_threshold: float = 0.55,
        anomaly_crit_threshold: float = 0.75,
    ):
        self.cpu_temp_warn = cpu_temp_warn_threshold
        self.cpu_temp_crit = cpu_temp_crit_threshold
        self.cpu_usage_warn = cpu_usage_warn_threshold
        self.cpu_usage_crit = cpu_usage_crit_threshold
        self.mem_usage_warn = mem_usage_warn_threshold
        self.mem_usage_crit = mem_usage_crit_threshold
        self.disk_temp_warn = disk_temp_warn_threshold
        self.anomaly_warn = anomaly_warn_threshold
        self.anomaly_crit = anomaly_crit_threshold

    def evaluate_cpu_risk(
        self,
        record: Dict[str, Any],
        window_features: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """Calculates CPU & Thermal risk score (0-100) and candidate alerts."""
        risk_score = 0.0
        alerts: List[Dict[str, Any]] = []

        cpu_usage = float(record.get("cpu_usage_pct") or 0.0)
        temp_c = record.get("temperature_c")
        load_1m = record.get("load_avg_1m")

        # 1. CPU Usage component (0 - 40 points)
        if cpu_usage >= self.cpu_usage_crit:
            risk_score += 40.0
            alerts.append({
                "severity": "CRITICAL",
                "entity_class": "CPU",
                "metric_name": "cpu_usage_pct",
                "metric_value": cpu_usage,
                "threshold": self.cpu_usage_crit,
                "message": f"Critical CPU saturation: {cpu_usage:.1f}% exceeds {self.cpu_usage_crit}% threshold.",
                "remediation": "Identify runaway processes with psutil/top; evaluate thread pool bottlenecks.",
            })
        elif cpu_usage >= self.cpu_usage_warn:
            risk_score += 25.0
            alerts.append({
                "severity": "WARNING",
                "entity_class": "CPU",
                "metric_name": "cpu_usage_pct",
                "metric_value": cpu_usage,
                "threshold": self.cpu_usage_warn,
                "message": f"Elevated CPU utilization: {cpu_usage:.1f}% exceeds warning threshold.",
                "remediation": "Monitor compute load distribution across available hardware cores.",
            })
        else:
            risk_score += (cpu_usage / 100.0) * 20.0

        # 2. Thermal component (0 - 45 points)
        if temp_c is not None:
            temp_val = float(temp_c)
            if temp_val >= self.cpu_temp_crit:
                risk_score += 45.0
                alerts.append({
                    "severity": "CRITICAL",
                    "entity_class": "CPU",
                    "metric_name": "temperature_c",
                    "metric_value": temp_val,
                    "threshold": self.cpu_temp_crit,
                    "message": f"Critical thermal junction temperature: {temp_val:.1f}°C (risk of thermal shutdown).",
                    "remediation": "Inspect cooling airflow, fan PWM profile, and thermal paste application.",
                })
            elif temp_val >= self.cpu_temp_warn:
                risk_score += 25.0
                alerts.append({
                    "severity": "WARNING",
                    "entity_class": "CPU",
                    "metric_name": "temperature_c",
                    "metric_value": temp_val,
                    "threshold": self.cpu_temp_warn,
                    "message": f"Elevated processor temperature: {temp_val:.1f}°C exceeds warning boundary.",
                    "remediation": "Verify ambient thermal conditions and chassis intake filters.",
                })
            else:
                risk_score += max(0.0, (temp_val - 35.0) / 40.0) * 15.0

        # 3. Sliding Window dynamic trend (0 - 15 points)
        if window_features:
            temp_slope = window_features.get("temperature_c_slope", 0.0)
            if temp_slope > 0.5:  # Climbing faster than 0.5 C/interval
                risk_score += 15.0
                alerts.append({
                    "severity": "WARNING",
                    "entity_class": "CPU",
                    "metric_name": "temperature_c_slope",
                    "metric_value": temp_slope,
                    "threshold": 0.5,
                    "message": f"Rapid thermal escalation detected: slope {temp_slope:+.2f}°C/sample.",
                    "remediation": "Proactively throttle heavy background worker tasks.",
                })

        return min(100.0, risk_score), alerts

    def evaluate_ram_risk(
        self,
        record: Dict[str, Any],
        window_features: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """Calculates RAM/Memory risk score (0-100) and candidate alerts."""
        risk_score = 0.0
        alerts: List[Dict[str, Any]] = []

        mem_usage = float(record.get("memory_usage_pct") or 0.0)

        # 1. Raw Usage level (0 - 60 points)
        if mem_usage >= self.mem_usage_crit:
            risk_score += 60.0
            alerts.append({
                "severity": "CRITICAL",
                "entity_class": "RAM",
                "metric_name": "memory_usage_pct",
                "metric_value": mem_usage,
                "threshold": self.mem_usage_crit,
                "message": f"Critical memory pressure: {mem_usage:.1f}% (risk of Out-Of-Memory termination).",
                "remediation": "Evict volatile caches and restart memory-hungry daemon processes.",
            })
        elif mem_usage >= self.mem_usage_warn:
            risk_score += 35.0
            alerts.append({
                "severity": "WARNING",
                "entity_class": "RAM",
                "metric_name": "memory_usage_pct",
                "metric_value": mem_usage,
                "threshold": self.mem_usage_warn,
                "message": f"Elevated memory consumption: {mem_usage:.1f}% exceeds warning threshold.",
                "remediation": "Review system memory allocation and look for unreleased buffer pools.",
            })
        else:
            risk_score += (mem_usage / 100.0) * 30.0

        # 2. Memory leak trajectory via window slope (0 - 40 points)
        if window_features:
            mem_slope = window_features.get("memory_usage_pct_slope", 0.0)
            if mem_slope > 0.8:
                risk_score += 40.0
                alerts.append({
                    "severity": "WARNING",
                    "entity_class": "RAM",
                    "metric_name": "memory_usage_pct_slope",
                    "metric_value": mem_slope,
                    "threshold": 0.8,
                    "message": f"Sustained memory accumulation trend: slope {mem_slope:+.2f}%/sample.",
                    "remediation": "Inspect application heap for potential memory leaks.",
                })

        return min(100.0, risk_score), alerts

    def evaluate_disk_risk(
        self, record: Dict[str, Any]
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """Calculates Storage SMART risk score (0-100) and candidate alerts."""
        risk_score = 0.0
        alerts: List[Dict[str, Any]] = []

        reallocated = record.get("reallocated_sectors")
        pending = record.get("pending_sectors")
        read_error = record.get("read_error_rate")
        disk_temp = record.get("disk_temp_c")
        disk_status = record.get("disk_status")

        # 1. SMART Overall Status
        if disk_status == "FAILED":
            risk_score += 85.0
            alerts.append({
                "severity": "CRITICAL",
                "entity_class": "HDD",
                "metric_name": "disk_status",
                "metric_value": 1.0,
                "threshold": 0.0,
                "message": "SMART health diagnostic status returned FAILED.",
                "remediation": "Immediate replacement required. Backup data to external volume.",
            })

        # 2. Reallocated Sectors (Attribute 5)
        if reallocated is not None:
            realloc_val = int(reallocated)
            if realloc_val > 20:
                risk_score += 70.0
                alerts.append({
                    "severity": "CRITICAL",
                    "entity_class": "HDD",
                    "metric_name": "reallocated_sectors",
                    "metric_value": float(realloc_val),
                    "threshold": 20.0,
                    "message": f"Critical bad sector count: {realloc_val} reallocated sectors.",
                    "remediation": "Storage medium surface degrading. Schedule device retirement.",
                })
            elif realloc_val > 0:
                risk_score += 35.0
                alerts.append({
                    "severity": "WARNING",
                    "entity_class": "HDD",
                    "metric_name": "reallocated_sectors",
                    "metric_value": float(realloc_val),
                    "threshold": 0.0,
                    "message": f"Warning: {realloc_val} reallocated sectors identified.",
                    "remediation": "Monitor sector growth rate; ensure automated snapshots are active.",
                })

        # 3. Pending Sectors (Attribute 197)
        if pending is not None:
            pending_val = int(pending)
            if pending_val > 10:
                risk_score += 50.0
                alerts.append({
                    "severity": "CRITICAL",
                    "entity_class": "HDD",
                    "metric_name": "pending_sectors",
                    "metric_value": float(pending_val),
                    "threshold": 10.0,
                    "message": f"High unstable pending sector count: {pending_val}.",
                    "remediation": "Perform full surface scan and backup critical disk partitions.",
                })
            elif pending_val > 0:
                risk_score += 25.0
                alerts.append({
                    "severity": "WARNING",
                    "entity_class": "HDD",
                    "metric_name": "pending_sectors",
                    "metric_value": float(pending_val),
                    "threshold": 0.0,
                    "message": f"Warning: {pending_val} pending sectors waiting for remapping.",
                    "remediation": "Run non-destructive disk check to attempt sector reallocation.",
                })

        # 4. Disk Temperature
        if disk_temp is not None:
            d_temp = float(disk_temp)
            if d_temp >= self.disk_temp_warn:
                risk_score += 25.0
                alerts.append({
                    "severity": "WARNING",
                    "entity_class": "HDD",
                    "metric_name": "disk_temp_c",
                    "metric_value": d_temp,
                    "threshold": self.disk_temp_warn,
                    "message": f"Elevated drive temperature: {d_temp:.1f}°C.",
                    "remediation": "Check drive bay ventilation and ambient enclosure temperatures.",
                })

        return min(100.0, risk_score), alerts

    def assess_comprehensive_risk(
        self,
        latest_record: Dict[str, Any],
        window_features: Optional[Dict[str, float]] = None,
        anomaly_score: float = 0.0,
        is_anomaly: bool = False,
        db_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Synthesizes overall hardware failure risk across all subsystems.

        Returns comprehensive assessment dict and dispatches new alerts to SQLite.
        """
        import time

        now_time = time.time()
        cpu_risk, cpu_alerts = self.evaluate_cpu_risk(latest_record, window_features)
        ram_risk, ram_alerts = self.evaluate_ram_risk(latest_record, window_features)
        disk_risk, disk_alerts = self.evaluate_disk_risk(latest_record)

        all_candidate_alerts = cpu_alerts + ram_alerts + disk_alerts

        # Anomaly Detection Alert
        if anomaly_score >= self.anomaly_crit:
            all_candidate_alerts.append({
                "severity": "CRITICAL",
                "entity_class": latest_record.get("entity_class", "SYSTEM"),
                "metric_name": "anomaly_score",
                "metric_value": round(anomaly_score, 4),
                "threshold": self.anomaly_crit,
                "message": f"Severe statistical anomaly detected (anomaly score: {anomaly_score:.2f}).",
                "remediation": "Inspect top deviating features and check system logs for anomalous spikes.",
            })
        elif anomaly_score >= self.anomaly_warn:
            all_candidate_alerts.append({
                "severity": "WARNING",
                "entity_class": latest_record.get("entity_class", "SYSTEM"),
                "metric_name": "anomaly_score",
                "metric_value": round(anomaly_score, 4),
                "threshold": self.anomaly_warn,
                "message": f"Statistical outlier detected in sensor streams (score: {anomaly_score:.2f}).",
                "remediation": "Review telemetry trend graphs for early signs of hardware instability.",
            })

        # Weighted composite risk
        weighted_risk = (0.35 * disk_risk) + (0.35 * cpu_risk) + (0.30 * ram_risk)
        # Scale with anomaly score
        anomaly_scaled_risk = anomaly_score * 100.0
        final_risk = min(100.0, max(weighted_risk, 0.7 * weighted_risk + 0.3 * anomaly_scaled_risk))

        # Categorize risk level
        if final_risk >= 80.0:
            risk_level = "CRITICAL"
        elif final_risk >= 60.0:
            risk_level = "MODERATE"
        elif final_risk >= 40.0:
            risk_level = "LOW"
        else:
            risk_level = "NOMINAL"

        # Actionable recommendations
        recommendations: List[str] = []
        if disk_risk > 50.0:
            recommendations.append("Priority 1: Storage integrity degraded — clone disk or trigger immediate off-site backup.")
        if cpu_risk > 50.0:
            recommendations.append("Priority 2: Processor thermal/load overload — check chassis fan PWM and CPU heat sink.")
        if ram_risk > 50.0:
            recommendations.append("Priority 3: Memory capacity exhaustion — audit background process footprint.")
        if not recommendations:
            recommendations.append("All hardware subsystems operating within expected nominal specifications.")

        # Persist alerts to DB if path provided with deduplication
        dispatched_alerts: List[Dict[str, Any]] = []
        if db_path:
            for alert in all_candidate_alerts:
                dedup_key = f"{alert['entity_class']}_{alert['metric_name']}_{alert['severity']}"
                last_dispatched = _RECENT_ALERTS_CACHE.get(dedup_key, 0.0)

                if (now_time - last_dispatched) > DEDUP_WINDOW_SECONDS:
                    alert_id = insert_alert(alert, db_path=db_path)
                    alert["id"] = alert_id
                    _RECENT_ALERTS_CACHE[dedup_key] = now_time
                    dispatched_alerts.append(alert)

            # Also log anomaly event if anomalous
            if is_anomaly or anomaly_score > 0.4:
                try:
                    insert_anomaly_event(
                        {
                            "timestamp": latest_record.get("timestamp"),
                            "entity_class": latest_record.get("entity_class", "CPU"),
                            "anomaly_score": anomaly_score,
                            "is_anomaly": 1 if is_anomaly else 0,
                            "feature_summary": json.dumps({
                                "cpu": latest_record.get("cpu_usage_pct"),
                                "ram": latest_record.get("memory_usage_pct"),
                                "temp": latest_record.get("temperature_c"),
                            }),
                            "host_id": latest_record.get("host_id", "edge-node"),
                        },
                        db_path=db_path,
                    )
                except Exception as e:
                    logger.warning("Error persisting anomaly event: %s", e)

        return {
            "composite_risk_score": round(final_risk, 1),
            "risk_level": risk_level,
            "component_risks": {
                "cpu_risk": round(cpu_risk, 1),
                "ram_risk": round(ram_risk, 1),
                "disk_risk": round(disk_risk, 1),
            },
            "anomaly_score": round(anomaly_score, 4),
            "is_anomaly": is_anomaly,
            "recommendations": recommendations,
            "active_alerts_count": len(all_candidate_alerts),
            "candidate_alerts": all_candidate_alerts,
        }


# Singleton Risk Engine
_risk_engine_instance: Optional[HardwareRiskEngine] = None


def get_risk_engine() -> HardwareRiskEngine:
    global _risk_engine_instance
    if _risk_engine_instance is None:
        _risk_engine_instance = HardwareRiskEngine()
    return _risk_engine_instance
