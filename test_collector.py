import pytest
from collector.readers import read_system_metrics, read_disk_smart
from collector.agent import TelemetryCollector


def test_read_system_metrics_structure():
    metrics = read_system_metrics()
    assert "cpu_usage_pct" in metrics
    assert "memory_usage_pct" in metrics
    assert "load_avg_1m" in metrics
    assert "load_avg_5m" in metrics
    assert "load_avg_15m" in metrics
    assert "temperature_c" in metrics
    assert 0.0 <= metrics["cpu_usage_pct"] <= 100.0
    assert 0.0 <= metrics["memory_usage_pct"] <= 100.0


def test_read_disk_smart_fallback():
    # Calling with nonexistent binary path should return fallback dict with None, not crash
    smart = read_disk_smart(binary_path="nonexistent_binary_xyz_123")
    assert smart["reallocated_sectors"] is None
    assert smart["pending_sectors"] is None
    assert smart["disk_status"] is None


def test_collector_once_execution(tmp_path):
    db_file = tmp_path / "collector_test.db"
    from storage.db import init_db
    init_db(db_file)
    collector = TelemetryCollector()
    collector.db_path = str(db_file)

    records = collector.collect_once()
    assert len(records) == 3

    classes = {r["entity_class"] for r in records}
    assert classes == {"CPU", "RAM", "HDD"}
    for r in records:
        assert r["host_id"] is not None
        assert "timestamp" in r
