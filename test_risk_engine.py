import tempfile
from pathlib import Path
import pytest

from models.risk_engine import HardwareRiskEngine
from storage.db import init_db, get_alerts


def test_cpu_risk_scoring():
    engine = HardwareRiskEngine()

    # Nominal CPU
    nom_record = {"cpu_usage_pct": 25.0, "temperature_c": 45.0, "load_avg_1m": 1.2}
    nom_risk, nom_alerts = engine.evaluate_cpu_risk(nom_record)
    assert nom_risk < 40.0
    assert len(nom_alerts) == 0

    # Critical Thermal & CPU saturation
    crit_record = {"cpu_usage_pct": 98.0, "temperature_c": 92.0, "load_avg_1m": 12.0}
    crit_risk, crit_alerts = engine.evaluate_cpu_risk(crit_record)
    assert crit_risk >= 70.0
    assert len(crit_alerts) >= 2
    assert any(a["severity"] == "CRITICAL" for a in crit_alerts)


def test_ram_risk_and_leak_slope():
    engine = HardwareRiskEngine()

    # Normal RAM
    nom_record = {"memory_usage_pct": 45.0}
    nom_risk, nom_alerts = engine.evaluate_ram_risk(nom_record)
    assert nom_risk < 40.0
    assert len(nom_alerts) == 0

    # Memory Leak slope from sliding window
    leak_record = {"memory_usage_pct": 85.0}
    win_features = {"memory_usage_pct_slope": 1.2}
    leak_risk, leak_alerts = engine.evaluate_ram_risk(leak_record, window_features=win_features)
    assert leak_risk > 60.0
    assert len(leak_alerts) >= 2


def test_disk_smart_risk_scoring():
    engine = HardwareRiskEngine()

    # Healthy SMART
    healthy_disk = {
        "reallocated_sectors": 0,
        "pending_sectors": 0,
        "read_error_rate": 0.0,
        "disk_temp_c": 34.0,
        "disk_status": "PASSED",
    }
    h_risk, h_alerts = engine.evaluate_disk_risk(healthy_disk)
    assert h_risk == 0.0
    assert len(h_alerts) == 0

    # Degraded SMART
    bad_disk = {
        "reallocated_sectors": 35,
        "pending_sectors": 12,
        "read_error_rate": 45.0,
        "disk_temp_c": 52.0,
        "disk_status": "FAILED",
    }
    b_risk, b_alerts = engine.evaluate_disk_risk(bad_disk)
    assert b_risk >= 80.0
    assert len(b_alerts) >= 2


def test_comprehensive_risk_assessment_and_alert_persistence():
    engine = HardwareRiskEngine()

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_risk.db"
        init_db(db_path)

        record = {
            "timestamp": "2026-10-29T12:00:00Z",
            "cpu_usage_pct": 96.0,
            "memory_usage_pct": 94.0,
            "temperature_c": 91.0,
            "reallocated_sectors": 25,
            "pending_sectors": 5,
            "read_error_rate": 2.0,
            "disk_temp_c": 42.0,
            "disk_status": "PASSED",
            "entity_class": "CPU",
            "host_id": "test-host",
        }

        assessment = engine.assess_comprehensive_risk(
            latest_record=record,
            window_features={},
            anomaly_score=0.85,
            is_anomaly=True,
            db_path=str(db_path),
        )

        assert assessment["risk_level"] in ("MODERATE", "CRITICAL")
        assert assessment["composite_risk_score"] > 60.0
        assert len(assessment["recommendations"]) > 0

        # Verify alerts were saved to SQLite
        saved_alerts = get_alerts(db_path=db_path)
        assert len(saved_alerts) > 0
