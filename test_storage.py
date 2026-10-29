import tempfile
from pathlib import Path
import pytest
from pydantic import ValidationError

from storage.db import (
    init_db,
    insert_telemetry,
    insert_telemetry_batch,
    get_latest_telemetry,
    get_telemetry_history,
    get_telemetry_count,
    export_to_parquet,
)
from storage.models import TelemetryRecord


@pytest.fixture
def temp_db(tmp_path):
    db_path = tmp_path / "test_telemetry.db"
    init_db(db_path)
    return db_path


def test_schema_initialization(temp_db):
    assert temp_db.exists()
    assert get_telemetry_count(temp_db) == 0


def test_insert_and_query_single_record(temp_db):
    rec = TelemetryRecord(
        timestamp="2026-09-27T01:00:00Z",
        cpu_usage_pct=45.5,
        memory_usage_pct=62.1,
        temperature_c=55.0,
        load_avg_1m=1.2,
        load_avg_5m=1.1,
        load_avg_15m=0.9,
        entity_class="CPU",
        host_id="test-node-01",
    )
    row_id = insert_telemetry(rec, temp_db)
    assert row_id == 1
    assert get_telemetry_count(temp_db) == 1

    latest = get_latest_telemetry(limit=1, db_path=temp_db)
    assert len(latest) == 1
    assert latest[0]["cpu_usage_pct"] == 45.5
    assert latest[0]["entity_class"] == "CPU"
    assert latest[0]["host_id"] == "test-node-01"


def test_insert_batch_and_entity_filter(temp_db):
    records = [
        {
            "timestamp": "2026-09-27T01:00:01Z",
            "cpu_usage_pct": 50.0,
            "memory_usage_pct": 70.0,
            "entity_class": "CPU",
            "host_id": "test-node",
        },
        {
            "timestamp": "2026-09-27T01:00:02Z",
            "cpu_usage_pct": 50.0,
            "memory_usage_pct": 70.0,
            "entity_class": "RAM",
            "host_id": "test-node",
        },
        {
            "timestamp": "2026-09-27T01:00:03Z",
            "cpu_usage_pct": 50.0,
            "memory_usage_pct": 70.0,
            "reallocated_sectors": 0,
            "pending_sectors": 0,
            "read_error_rate": 0.0,
            "disk_status": "PASSED",
            "entity_class": "HDD",
            "host_id": "test-node",
        },
    ]
    inserted = insert_telemetry_batch(records, temp_db)
    assert inserted == 3
    assert get_telemetry_count(temp_db) == 3

    hdd_records = get_latest_telemetry(entity_class="HDD", db_path=temp_db)
    assert len(hdd_records) == 1
    assert hdd_records[0]["disk_status"] == "PASSED"


def test_invalid_entity_class_rejected():
    with pytest.raises(ValidationError):
        TelemetryRecord(
            cpu_usage_pct=10.0,
            memory_usage_pct=20.0,
            entity_class="GPU",  # Invalid entity class
            host_id="test",
        )


def test_export_to_parquet(temp_db):
    insert_telemetry(
        {
            "timestamp": "2026-09-27T01:00:00Z",
            "cpu_usage_pct": 12.0,
            "memory_usage_pct": 34.0,
            "entity_class": "CPU",
            "host_id": "node-1",
        },
        temp_db,
    )
    pq_path = temp_db.parent / "test_export.parquet"
    count = export_to_parquet(pq_path, temp_db)
    assert count == 1
    assert pq_path.exists()
