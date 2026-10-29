import os
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone, timedelta
import pandas as pd

from storage.models import TelemetryRecord

DEFAULT_DB_PATH = Path("data/telemetry.db")
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def get_db_connection(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Creates a configured SQLite connection with WAL mode enabled for edge concurrency."""
    target_path = Path(db_path) if db_path else DEFAULT_DB_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(target_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    # High-performance edge pragmas
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA busy_timeout = 5000;")
    return conn


from contextlib import contextmanager


@contextmanager
def db_session(db_path: Path | str | None = None):
    """Context manager that guarantees the SQLite connection is closed on exit."""
    conn = get_db_connection(db_path)
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path: Path | str | None = None) -> None:
    """Initializes the SQLite database with the telemetry schema and indexes."""
    target_path = Path(db_path) if db_path else DEFAULT_DB_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)

    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema_sql = f.read()

    with db_session(target_path) as conn:
        conn.executescript(schema_sql)
        conn.commit()


SCHEMA_FIELDS = [
    "timestamp",
    "cpu_usage_pct",
    "memory_usage_pct",
    "temperature_c",
    "reallocated_sectors",
    "pending_sectors",
    "read_error_rate",
    "disk_temp_c",
    "disk_status",
    "load_avg_1m",
    "load_avg_5m",
    "load_avg_15m",
    "entity_class",
    "host_id",
]


def _normalize_record_dict(record: TelemetryRecord | Dict[str, Any]) -> Dict[str, Any]:
    """Ensures all schema fields are present with None default to satisfy SQLite named bindings."""
    if isinstance(record, TelemetryRecord):
        data = record.model_dump()
    else:
        data = dict(record)

    data.pop("id", None)
    normalized = {field: data.get(field, None) for field in SCHEMA_FIELDS}
    if not normalized.get("timestamp"):
        normalized["timestamp"] = datetime.now(timezone.utc).isoformat()
    return normalized


def insert_telemetry(record: TelemetryRecord | Dict[str, Any], db_path: Path | str | None = None) -> int:
    """Inserts a single telemetry record into the database."""
    data = _normalize_record_dict(record)

    sql = """
        INSERT INTO telemetry (
            timestamp, cpu_usage_pct, memory_usage_pct, temperature_c,
            reallocated_sectors, pending_sectors, read_error_rate, disk_temp_c, disk_status,
            load_avg_1m, load_avg_5m, load_avg_15m, entity_class, host_id
        ) VALUES (
            :timestamp, :cpu_usage_pct, :memory_usage_pct, :temperature_c,
            :reallocated_sectors, :pending_sectors, :read_error_rate, :disk_temp_c, :disk_status,
            :load_avg_1m, :load_avg_5m, :load_avg_15m, :entity_class, :host_id
        );
    """
    with db_session(db_path) as conn:
        cursor = conn.execute(sql, data)
        conn.commit()
        return cursor.lastrowid


def insert_telemetry_batch(
    records: List[TelemetryRecord | Dict[str, Any]], db_path: Path | str | None = None
) -> int:
    """Inserts multiple telemetry records within a single transaction."""
    if not records:
        return 0

    prepared_data = [_normalize_record_dict(r) for r in records]

    sql = """
        INSERT INTO telemetry (
            timestamp, cpu_usage_pct, memory_usage_pct, temperature_c,
            reallocated_sectors, pending_sectors, read_error_rate, disk_temp_c, disk_status,
            load_avg_1m, load_avg_5m, load_avg_15m, entity_class, host_id
        ) VALUES (
            :timestamp, :cpu_usage_pct, :memory_usage_pct, :temperature_c,
            :reallocated_sectors, :pending_sectors, :read_error_rate, :disk_temp_c, :disk_status,
            :load_avg_1m, :load_avg_5m, :load_avg_15m, :entity_class, :host_id
        );
    """
    with db_session(db_path) as conn:
        cursor = conn.executemany(sql, prepared_data)
        conn.commit()
        return cursor.rowcount


def get_latest_telemetry(
    entity_class: Optional[str] = None,
    host_id: Optional[str] = None,
    limit: int = 10,
    db_path: Path | str | None = None,
) -> List[Dict[str, Any]]:
    """Retrieves the latest telemetry records, optionally filtered by entity_class or host_id."""
    query = "SELECT * FROM telemetry WHERE 1=1"
    params: List[Any] = []

    if entity_class:
        query += " AND entity_class = ?"
        params.append(entity_class)
    if host_id:
        query += " AND host_id = ?"
        params.append(host_id)

    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    with db_session(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]


def get_telemetry_history(
    hours: float = 1.0,
    entity_class: Optional[str] = None,
    host_id: Optional[str] = None,
    limit: int = 5000,
    db_path: Path | str | None = None,
) -> List[Dict[str, Any]]:
    """Retrieves telemetry within the last N hours."""
    cutoff_time = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()

    query = "SELECT * FROM telemetry WHERE timestamp >= ?"
    params: List[Any] = [cutoff_time]

    if entity_class:
        query += " AND entity_class = ?"
        params.append(entity_class)
    if host_id:
        query += " AND host_id = ?"
        params.append(host_id)

    query += " ORDER BY timestamp ASC LIMIT ?"
    params.append(limit)

    with db_session(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]


def get_telemetry_count(db_path: Path | str | None = None) -> int:
    """Returns total count of telemetry records in SQLite."""
    with db_session(db_path) as conn:
        row = conn.execute("SELECT COUNT(*) as count FROM telemetry;").fetchone()
        return row["count"] if row else 0


def export_to_parquet(output_filepath: Path | str, db_path: Path | str | None = None) -> int:
    """Exports SQLite telemetry table to an optimized Parquet file for downstream ML."""
    out_path = Path(output_filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with db_session(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM telemetry ORDER BY id ASC;", conn)

    if not df.empty:
        df.to_parquet(out_path, index=False, engine="pyarrow")
    return len(df)


def get_all_telemetry_df(db_path: Path | str | None = None) -> pd.DataFrame:
    """Retrieves all telemetry records as a pandas DataFrame."""
    with db_session(db_path) as conn:
        return pd.read_sql_query("SELECT * FROM telemetry ORDER BY id ASC;", conn)


# =========================================================================
# Alert Storage Operations
# =========================================================================

def insert_alert(alert: Dict[str, Any], db_path: Path | str | None = None) -> int:
    """Inserts a new hardware risk or metric alert into SQLite."""
    now_iso = alert.get("timestamp") or datetime.now(timezone.utc).isoformat()
    sql = """
        INSERT INTO alerts (
            timestamp, entity_class, severity, metric_name,
            metric_value, threshold, message, remediation, acknowledged
        ) VALUES (
            :timestamp, :entity_class, :severity, :metric_name,
            :metric_value, :threshold, :message, :remediation, :acknowledged
        );
    """
    params = {
        "timestamp": now_iso,
        "entity_class": alert.get("entity_class", "SYSTEM"),
        "severity": alert.get("severity", "WARNING"),
        "metric_name": alert.get("metric_name", "unknown"),
        "metric_value": alert.get("metric_value"),
        "threshold": alert.get("threshold"),
        "message": alert.get("message", "Hardware warning detected"),
        "remediation": alert.get("remediation"),
        "acknowledged": int(alert.get("acknowledged", 0)),
    }
    with db_session(db_path) as conn:
        cursor = conn.execute(sql, params)
        conn.commit()
        return cursor.lastrowid


def get_alerts(
    severity: Optional[str] = None,
    acknowledged: Optional[int] = None,
    limit: int = 50,
    db_path: Path | str | None = None,
) -> List[Dict[str, Any]]:
    """Retrieves alerts, optionally filtered by severity or acknowledgment status."""
    query = "SELECT * FROM alerts WHERE 1=1"
    params: List[Any] = []

    if severity:
        query += " AND severity = ?"
        params.append(severity)
    if acknowledged is not None:
        query += " AND acknowledged = ?"
        params.append(int(acknowledged))

    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    with db_session(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]


def acknowledge_alert(alert_id: int, db_path: Path | str | None = None) -> bool:
    """Marks an alert as acknowledged."""
    sql = "UPDATE alerts SET acknowledged = 1 WHERE id = ?;"
    with db_session(db_path) as conn:
        cursor = conn.execute(sql, (alert_id,))
        conn.commit()
        return cursor.rowcount > 0


def get_alert_counts(db_path: Path | str | None = None) -> Dict[str, int]:
    """Returns total, active (unacknowledged), and critical counts."""
    with db_session(db_path) as conn:
        total = conn.execute("SELECT COUNT(*) as cnt FROM alerts;").fetchone()["cnt"]
        active = conn.execute("SELECT COUNT(*) as cnt FROM alerts WHERE acknowledged = 0;").fetchone()["cnt"]
        critical = conn.execute("SELECT COUNT(*) as cnt FROM alerts WHERE severity = 'CRITICAL' AND acknowledged = 0;").fetchone()["cnt"]
        return {"total": total, "active": active, "critical": critical}


# =========================================================================
# Anomaly Events Storage Operations
# =========================================================================

def insert_anomaly_event(event: Dict[str, Any], db_path: Path | str | None = None) -> int:
    """Logs an anomaly detection event emitted by the ML inference engine."""
    now_iso = event.get("timestamp") or datetime.now(timezone.utc).isoformat()
    sql = """
        INSERT INTO anomaly_events (
            timestamp, entity_class, anomaly_score, is_anomaly, feature_summary, host_id
        ) VALUES (
            :timestamp, :entity_class, :anomaly_score, :is_anomaly, :feature_summary, :host_id
        );
    """
    params = {
        "timestamp": now_iso,
        "entity_class": event.get("entity_class", "CPU"),
        "anomaly_score": float(event.get("anomaly_score", 0.0)),
        "is_anomaly": 1 if event.get("is_anomaly") else 0,
        "feature_summary": event.get("feature_summary"),
        "host_id": event.get("host_id", "edge-node"),
    }
    with db_session(db_path) as conn:
        cursor = conn.execute(sql, params)
        conn.commit()
        return cursor.lastrowid


def get_latest_anomalies(
    limit: int = 20,
    entity_class: Optional[str] = None,
    db_path: Path | str | None = None,
) -> List[Dict[str, Any]]:
    """Retrieves recent anomaly events."""
    query = "SELECT * FROM anomaly_events WHERE 1=1"
    params: List[Any] = []

    if entity_class:
        query += " AND entity_class = ?"
        params.append(entity_class)

    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    with db_session(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

