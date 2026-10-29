-- Telemetry Time-Series Edge Storage Schema
-- Mission 1 Contract for Predictive Hardware Failure Dashboard

CREATE TABLE IF NOT EXISTS telemetry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,                     -- UTC ISO-8601 formatted string (e.g. 2026-09-27T01:45:00.123456Z)
    cpu_usage_pct REAL NOT NULL,                 -- CPU usage percentage (0.0 - 100.0)
    memory_usage_pct REAL NOT NULL,              -- Memory usage percentage (0.0 - 100.0)
    temperature_c REAL,                          -- System or component temperature in Celsius (nullable)
    reallocated_sectors INTEGER,                 -- SMART 5: Reallocated Sectors Count (nullable)
    pending_sectors INTEGER,                     -- SMART 197: Current Pending Sector Count (nullable)
    read_error_rate REAL,                        -- SMART 1: Raw Read Error Rate (nullable)
    disk_temp_c REAL,                            -- SMART 194: Disk Temperature in Celsius (nullable)
    disk_status TEXT,                            -- SMART Health assessment: 'PASSED', 'FAILED', or 'UNKNOWN'
    load_avg_1m REAL,                            -- 1-minute load average (nullable on Windows)
    load_avg_5m REAL,                            -- 5-minute load average (nullable on Windows)
    load_avg_15m REAL,                           -- 15-minute load average (nullable on Windows)
    entity_class TEXT NOT NULL CHECK(entity_class IN ('HDD', 'RAM', 'CPU')),
    host_id TEXT NOT NULL                        -- Machine/Node identifier
);

-- Optimize for fast temporal queries and entity-sliced retrieval
CREATE INDEX IF NOT EXISTS idx_telemetry_timestamp ON telemetry (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_telemetry_entity_ts ON telemetry (entity_class, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_telemetry_host_ts ON telemetry (host_id, timestamp DESC);

-- Alerts table for automated threshold and risk violation tracking
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,                     -- UTC ISO-8601
    entity_class TEXT NOT NULL,                 -- 'HDD', 'RAM', 'CPU', or 'SYSTEM'
    severity TEXT NOT NULL CHECK(severity IN ('INFO', 'WARNING', 'CRITICAL')),
    metric_name TEXT NOT NULL,                  -- e.g. 'cpu_usage_pct', 'temperature_c', 'reallocated_sectors', 'composite_risk'
    metric_value REAL,                          -- Value that triggered the alert
    threshold REAL,                             -- Threshold breached
    message TEXT NOT NULL,                      -- Human-readable alert summary
    remediation TEXT,                           -- Prescribed action/remediation step
    acknowledged INTEGER NOT NULL DEFAULT 0     -- 0 = active, 1 = acknowledged
);

CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts (severity, timestamp DESC);

-- Anomaly events table logged by the Isolation Forest detector
CREATE TABLE IF NOT EXISTS anomaly_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,                     -- UTC ISO-8601
    entity_class TEXT NOT NULL,
    anomaly_score REAL NOT NULL,                 -- Calibrated score (0.0 - 1.0)
    is_anomaly INTEGER NOT NULL,                 -- 1 if anomaly detected, 0 if normal
    feature_summary TEXT,                        -- JSON summary of contributing features
    host_id TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_anomaly_timestamp ON anomaly_events (timestamp DESC);

