"""Platform-agnostic metric readers for CPU, memory, load, temperature, and disk SMART attributes.

Gracefully handles hardware and OS differences (e.g. Windows vs Linux, missing smartmontools,
unavailable thermal sensors). Missing readings are logged once and returned as None/null without crashing.
"""

import json
import logging
import platform
import shutil
import subprocess
from typing import Any, Dict, Optional, Tuple

import psutil

logger = logging.getLogger(__name__)

# State flags to prevent warning log spamming in tight loops
_TEMP_SENSOR_WARNED = False
_SMARTCTL_WARNED = False
_LOADAVG_WARNED = False


def read_system_metrics() -> Dict[str, Any]:
    """Reads real-time system metrics: CPU, memory, temperature, and load average."""
    global _TEMP_SENSOR_WARNED, _LOADAVG_WARNED

    # 1. CPU Usage Percentage
    cpu_usage = float(psutil.cpu_percent(interval=None))

    # 2. Memory Usage Percentage
    mem = psutil.virtual_memory()
    mem_usage = float(mem.percent)

    # 3. System Load Average (1m, 5m, 15m)
    load_1m: Optional[float] = None
    load_5m: Optional[float] = None
    load_15m: Optional[float] = None
    try:
        if hasattr(psutil, "getloadavg"):
            load_tuple = psutil.getloadavg()
            # On Windows psutil may return (0.0, 0.0, 0.0) or raise AttributeError
            if load_tuple and any(x > 0 for x in load_tuple):
                load_1m, load_5m, load_15m = (
                    float(load_tuple[0]),
                    float(load_tuple[1]),
                    float(load_tuple[2]),
                )
            elif platform.system() != "Windows":
                load_1m, load_5m, load_15m = (
                    float(load_tuple[0]),
                    float(load_tuple[1]),
                    float(load_tuple[2]),
                )
    except Exception as e:
        if not _LOADAVG_WARNED:
            logger.info("System load average unavailable on this platform: %s", e)
            _LOADAVG_WARNED = True

    # 4. Temperature Readings (Celsius)
    temp_c: Optional[float] = None
    try:
        if hasattr(psutil, "sensors_temperatures"):
            temps = psutil.sensors_temperatures()
            if temps:
                # Find CPU or thermal zone temperature
                all_currents = []
                for sensor_name, entries in temps.items():
                    for entry in entries:
                        if entry.current is not None and entry.current > 0:
                            all_currents.append(entry.current)
                if all_currents:
                    temp_c = float(max(all_currents))
            else:
                if not _TEMP_SENSOR_WARNED:
                    logger.info("Thermal sensors not exposed by OS/hardware.")
                    _TEMP_SENSOR_WARNED = True
        else:
            if not _TEMP_SENSOR_WARNED:
                logger.info("psutil.sensors_temperatures not supported on %s.", platform.system())
                _TEMP_SENSOR_WARNED = True
    except Exception as e:
        if not _TEMP_SENSOR_WARNED:
            logger.warning("Error reading thermal sensors: %s", e)
            _TEMP_SENSOR_WARNED = True

    return {
        "cpu_usage_pct": round(cpu_usage, 2),
        "memory_usage_pct": round(mem_usage, 2),
        "load_avg_1m": round(load_1m, 2) if load_1m is not None else None,
        "load_avg_5m": round(load_5m, 2) if load_5m is not None else None,
        "load_avg_15m": round(load_15m, 2) if load_15m is not None else None,
        "temperature_c": round(temp_c, 2) if temp_c is not None else None,
    }


def read_disk_smart(
    binary_path: str = "smartctl", device: str = "auto"
) -> Dict[str, Any]:
    """Reads disk SMART attributes using smartctl JSON output.

    Gracefully falls back to None values if smartctl is missing or fails.
    """
    global _SMARTCTL_WARNED

    default_result: Dict[str, Any] = {
        "reallocated_sectors": None,
        "pending_sectors": None,
        "read_error_rate": None,
        "disk_temp_c": None,
        "disk_status": None,
    }

    # Verify smartctl executable is accessible
    resolved_bin = shutil.which(binary_path) or binary_path
    try:
        # Check if smartctl is executable
        cmd_scan = [resolved_bin, "--scan", "-j"]
        scan_res = subprocess.run(cmd_scan, capture_output=True, text=True, timeout=5)
        if scan_res.returncode != 0 and scan_res.returncode != 1:  # smartctl uses bitmasks
            raise FileNotFoundError(f"smartctl scan returned code {scan_res.returncode}")

        target_device = device
        if device == "auto":
            scan_data = json.loads(scan_res.stdout) if scan_res.stdout else {}
            devices = scan_data.get("devices", [])
            if devices:
                target_device = devices[0].get("name", "/dev/sda")
            else:
                target_device = "/dev/sda" if platform.system() != "Windows" else "/dev/pd0"

        cmd_info = [resolved_bin, "-a", "-j", target_device]
        info_res = subprocess.run(cmd_info, capture_output=True, text=True, timeout=5)
        if not info_res.stdout:
            return default_result

        smart_json = json.loads(info_res.stdout)

        # Extract overall health status
        smart_status_obj = smart_json.get("smart_status", {})
        disk_status = "PASSED" if smart_status_obj.get("passed", False) else "FAILED"
        if "passed" not in smart_status_obj:
            disk_status = "UNKNOWN"

        # Extract temperature
        disk_temp = None
        temp_obj = smart_json.get("temperature", {})
        if "current" in temp_obj:
            disk_temp = float(temp_obj["current"])

        # Extract ATA SMART attributes
        reallocated = None
        pending = None
        read_error = None

        ata_table = smart_json.get("ata_smart_attributes", {}).get("table", [])
        for attr in ata_table:
            attr_id = attr.get("id")
            raw_val = attr.get("raw", {}).get("value")
            if attr_id == 5:  # Reallocated Sectors Count
                reallocated = int(raw_val) if raw_val is not None else None
            elif attr_id == 197:  # Current Pending Sector Count
                pending = int(raw_val) if raw_val is not None else None
            elif attr_id == 1:  # Raw Read Error Rate
                read_error = float(raw_val) if raw_val is not None else None
            elif attr_id == 194 and disk_temp is None:  # Disk Temperature
                disk_temp = float(raw_val) if raw_val is not None else None

        # NVMe attributes fallback
        nvme_health = smart_json.get("nvme_smart_health_information_log", {})
        if nvme_health:
            if disk_temp is None and "temperature" in nvme_health:
                disk_temp = float(nvme_health["temperature"])
            if disk_status == "UNKNOWN":
                critical_warning = nvme_health.get("critical_warning", 0)
                disk_status = "PASSED" if critical_warning == 0 else "FAILED"

        return {
            "reallocated_sectors": reallocated,
            "pending_sectors": pending,
            "read_error_rate": read_error,
            "disk_temp_c": disk_temp,
            "disk_status": disk_status,
        }

    except (FileNotFoundError, subprocess.SubprocessError, json.JSONDecodeError, OSError) as e:
        if not _SMARTCTL_WARNED:
            logger.info("smartctl not available or error querying disk SMART: %s", e)
            _SMARTCTL_WARNED = True
        return default_result
