"""FastAPI Backend Application for Predictive Hardware Failure Monitoring.

Core Endpoints:
- GET /health: Health check, system status, record counts, active alerts, and ML detector status
- GET /telemetry/latest: Latest telemetry records (filterable by entity_class)
- GET /telemetry/history: Telemetry records over the past N hours
- GET /window/latest: Current Chevtchenko sliding window & extracted statistical features
- POST /telemetry/collect: Trigger an immediate on-demand collection sample
- GET /risk/assessment: Multi-subsystem failure risk scores (CPU, RAM, Storage) & recommendations
- GET /anomalies/latest: Recent anomaly events and feature attributions
- GET /alerts: Generated alerts (active and historical, filterable by severity and acknowledgment)
- POST /alerts/{alert_id}/acknowledge: Acknowledge active hardware alert
- GET /model/status: ML model state, hyperparameters, and feature list
- POST /model/train: Retrain anomaly detection model on available telemetry
- GET /evaluation/metrics: Quantitative benchmark metrics (Accuracy, Precision, Recall, F1, MCC, latencies)
- WebSocket /ws/telemetry: Real-time telemetry, anomaly score, and risk streaming to connected clients
- Static Web Dashboard: Mounted at root (/) and (/dashboard)
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import platform
import socket
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from configs import load_config
from collector.agent import TelemetryCollector
from features.windowing import compute_window_features
from storage.db import (
    get_db_connection,
    get_latest_telemetry,
    get_telemetry_history,
    get_telemetry_count,
    init_db,
    get_alerts,
    acknowledge_alert,
    get_alert_counts,
    get_latest_anomalies,
)
from storage.models import TelemetryRecord
from models.anomaly import get_anomaly_detector, IsolationForestAnomalyDetector
from models.risk_engine import get_risk_engine, HardwareRiskEngine
from evaluation.evaluator import run_full_evaluation

logger = logging.getLogger("api.app")

# System startup time for uptime metric
START_TIME = time.time()
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"

# Cache for latest evaluated state
latest_evaluated_risk: Dict[str, Any] = {}
latest_anomaly_result: Dict[str, Any] = {}


class ConnectionManager:
    """Manages active WebSocket connections and broadcasts telemetry updates."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("WebSocket client connected. Total clients: %d", len(self.active_connections))

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info("WebSocket client disconnected. Total clients: %d", len(self.active_connections))

    async def broadcast_json(self, data: Any):
        """Broadcasts payload to all connected clients; removes stale connections."""
        for connection in list(self.active_connections):
            try:
                await connection.send_json(data)
            except Exception as e:
                logger.warning("Failed to send WebSocket message, dropping client: %s", e)
                self.disconnect(connection)


manager = ConnectionManager()
collector_instance: Optional[TelemetryCollector] = None
collector_task: Optional[asyncio.Task] = None


async def background_collector_loop(collector: TelemetryCollector, interval: float):
    """Background async worker that collects telemetry, runs ML inference, and broadcasts."""
    global latest_evaluated_risk, latest_anomaly_result
    logger.info("Embedded collector loop started (interval: %.1fs)", interval)

    config = load_config()
    db_path = config.get("storage", {}).get("db_path", "data/telemetry.db")
    detector = get_anomaly_detector(auto_train=True)
    risk_engine = get_risk_engine()

    try:
        while True:
            records = collector.collect_once()
            if records:
                # 1. Evaluate Anomaly Detection on newest record
                primary_sample = records[0]
                try:
                    anomaly_res = detector.score_sample(primary_sample)
                    latest_anomaly_result = anomaly_res
                except Exception as e:
                    logger.warning("Error running anomaly detector: %s", e)
                    anomaly_res = {"is_anomaly": False, "anomaly_score": 0.05, "top_contributing_features": []}

                # 2. Extract sliding window features for dynamic slope analysis
                win_features = {}
                if "CPU" in collector.window_buffers and collector.window_buffers["CPU"].get_current_window():
                    win_features = compute_window_features(collector.window_buffers["CPU"].get_current_window())

                # 3. Evaluate comprehensive multi-subsystem failure risk
                try:
                    risk_assessment = risk_engine.assess_comprehensive_risk(
                        latest_record=primary_sample,
                        window_features=win_features,
                        anomaly_score=anomaly_res.get("anomaly_score", 0.0),
                        is_anomaly=anomaly_res.get("is_anomaly", False),
                        db_path=db_path,
                    )
                    latest_evaluated_risk = risk_assessment
                except Exception as e:
                    logger.warning("Error evaluating comprehensive risk: %s", e)
                    risk_assessment = {
                        "composite_risk_score": 15.0,
                        "risk_level": "NOMINAL",
                        "component_risks": {"cpu_risk": 15.0, "ram_risk": 15.0, "disk_risk": 0.0},
                    }

                # 4. Broadcast live frame over WebSockets
                await manager.broadcast_json({
                    "event": "telemetry_update",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "records": records,
                    "anomaly_detection": anomaly_res,
                    "risk_assessment": risk_assessment,
                })

            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        logger.info("Embedded collector loop received cancellation signal.")
    except Exception as e:
        logger.exception("Error in embedded collector loop: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI: initializes DB, models, and background collector."""
    global collector_instance, collector_task

    config = load_config()
    db_path = config.get("storage", {}).get("db_path", "data/telemetry.db")
    init_db(db_path)

    # Initialize Isolation Forest Detector singleton
    get_anomaly_detector(auto_train=True)

    # Initialize embedded collector
    collector_instance = TelemetryCollector()
    interval = collector_instance.interval_seconds

    # Start background task for collection + live WebSocket broadcast
    collector_task = asyncio.create_task(background_collector_loop(collector_instance, interval))
    logger.info("FastAPI service initialized with embedded telemetry collector and ML engine.")

    yield

    # Shutdown
    if collector_task:
        collector_task.cancel()
        try:
            await collector_task
        except asyncio.CancelledError:
            pass
    logger.info("FastAPI service shut down.")


app = FastAPI(
    title="Predictive Hardware Failure Dashboard API",
    description="Edge telemetry ingestion, ML anomaly detection, risk assessment, and real-time streaming API",
    version="0.3.0",
    lifespan=lifespan,
)

# Enable CORS for frontend
cfg = load_config()
cors_origins = cfg.get("api", {}).get("cors_origins", ["*"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins if cors_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Dashboard Static Assets if directory exists
if DASHBOARD_DIR.exists():
    app.mount("/dashboard", StaticFiles(directory=str(DASHBOARD_DIR), html=True), name="dashboard")


@app.get("/", summary="Dashboard Web UI Redirect")
def get_dashboard_root():
    """Serves the main interactive React Dashboard index page."""
    index_file = DASHBOARD_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Predictive Hardware Failure API active. Visit /docs for Swagger documentation."}


@app.get("/health", summary="Service Health and System Status")
def health_check() -> Dict[str, Any]:
    """Returns the operational status of the edge telemetry API, database, active alerts, and host."""
    config = load_config()
    db_path = config.get("storage", {}).get("db_path", "data/telemetry.db")
    total_records = get_telemetry_count(db_path)
    alert_counts = get_alert_counts(db_path)
    uptime = time.time() - START_TIME
    detector = get_anomaly_detector(auto_train=False)

    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "host_id": collector_instance.host_id if collector_instance else socket.gethostname(),
        "platform": f"{platform.system()} {platform.release()}",
        "live_mode": config.get("mode", {}).get("live_mode", True),
        "db_path": db_path,
        "total_records": total_records,
        "active_alerts": alert_counts.get("active", 0),
        "critical_alerts": alert_counts.get("critical", 0),
        "model_fitted": detector.is_fitted if detector else False,
        "active_websocket_clients": len(manager.active_connections),
        "uptime_seconds": round(uptime, 2),
        "mission_phase": "Mission 3 / Phase III Completed (ML Anomaly Detection, Risk Engine, Dashboard)",
    }


@app.get("/telemetry/latest", summary="Retrieve latest telemetry")
def get_latest(
    entity_class: Optional[str] = Query(
        None, description="Filter by hardware entity: 'HDD', 'RAM', or 'CPU'"
    ),
    limit: int = Query(10, ge=1, le=100, description="Max number of records to return"),
) -> Dict[str, Any]:
    """Fetches the most recent telemetry samples recorded by the edge collector."""
    if entity_class and entity_class not in ("HDD", "RAM", "CPU"):
        raise HTTPException(status_code=400, detail="entity_class must be one of: 'HDD', 'RAM', 'CPU'")

    config = load_config()
    db_path = config.get("storage", {}).get("db_path", "data/telemetry.db")
    rows = get_latest_telemetry(entity_class=entity_class, limit=limit, db_path=db_path)
    return {
        "count": len(rows),
        "entity_class_filter": entity_class,
        "records": rows,
    }


@app.get("/telemetry/history", summary="Retrieve telemetry history")
def get_history(
    hours: float = Query(1.0, gt=0, le=168.0, description="Lookback window in hours (default 1.0)"),
    entity_class: Optional[str] = Query(
        None, description="Filter by hardware entity: 'HDD', 'RAM', or 'CPU'"
    ),
    limit: int = Query(5000, ge=1, le=10000, description="Maximum number of historical records"),
) -> Dict[str, Any]:
    """Fetches historical telemetry within the last N hours."""
    if entity_class and entity_class not in ("HDD", "RAM", "CPU"):
        raise HTTPException(status_code=400, detail="entity_class must be one of: 'HDD', 'RAM', 'CPU'")

    config = load_config()
    db_path = config.get("storage", {}).get("db_path", "data/telemetry.db")
    rows = get_telemetry_history(
        hours=hours, entity_class=entity_class, limit=limit, db_path=db_path
    )
    return {
        "hours": hours,
        "entity_class_filter": entity_class,
        "count": len(rows),
        "records": rows,
    }


@app.get("/window/latest", summary="Extract sliding window and statistical features")
def get_latest_window(
    entity_class: str = Query("CPU", description="Entity class: 'HDD', 'RAM', or 'CPU'"),
    window_size: int = Query(12, ge=2, le=120, description="Window size W"),
) -> Dict[str, Any]:
    """Demonstrates sliding window feature extraction (Chevtchenko et al.) on current telemetry."""
    if entity_class not in ("HDD", "RAM", "CPU"):
        raise HTTPException(status_code=400, detail="entity_class must be one of: 'HDD', 'RAM', 'CPU'")

    config = load_config()
    db_path = config.get("storage", {}).get("db_path", "data/telemetry.db")
    rows = get_latest_telemetry(entity_class=entity_class, limit=window_size, db_path=db_path)
    # Reverse to chronological order
    rows = list(reversed(rows))

    if not rows:
        return {"entity_class": entity_class, "window_size": 0, "features": {}, "samples": []}

    features = compute_window_features(rows)
    return {
        "entity_class": entity_class,
        "sample_count": len(rows),
        "window_size_requested": window_size,
        "statistical_features": features,
        "raw_samples": rows,
    }


@app.post("/telemetry/collect", summary="Trigger manual collection cycle")
def trigger_collection() -> Dict[str, Any]:
    """Manually triggers a collection cycle on demand."""
    if not collector_instance:
        raise HTTPException(status_code=500, detail="Collector not initialized")
    records = collector_instance.collect_once()
    return {
        "status": "success",
        "collected_records": len(records),
        "records": records,
    }


# =========================================================================
# ML Anomaly Detection & Risk Assessment Endpoints
# =========================================================================

@app.get("/risk/assessment", summary="Multi-Subsystem Hardware Failure Risk Assessment")
def get_risk_assessment() -> Dict[str, Any]:
    """Returns real-time risk scores for CPU, RAM, Disk, and overall composite score."""
    global latest_evaluated_risk

    if latest_evaluated_risk:
        return latest_evaluated_risk

    # Compute on demand if empty
    config = load_config()
    db_path = config.get("storage", {}).get("db_path", "data/telemetry.db")
    latest = get_latest_telemetry(limit=1, db_path=db_path)
    if not latest:
        return {
            "composite_risk_score": 0.0,
            "risk_level": "NOMINAL",
            "component_risks": {"cpu_risk": 0.0, "ram_risk": 0.0, "disk_risk": 0.0},
            "recommendations": ["No telemetry recorded yet."],
            "active_alerts_count": 0,
        }

    detector = get_anomaly_detector(auto_train=True)
    anomaly_res = detector.score_sample(latest[0])
    risk_engine = get_risk_engine()

    return risk_engine.assess_comprehensive_risk(
        latest_record=latest[0],
        anomaly_score=anomaly_res.get("anomaly_score", 0.0),
        is_anomaly=anomaly_res.get("is_anomaly", False),
        db_path=db_path,
    )


@app.get("/anomalies/latest", summary="Retrieve latest anomaly events")
def get_anomalies(
    limit: int = Query(20, ge=1, le=100, description="Max anomalies to return"),
    entity_class: Optional[str] = Query(None, description="Filter by subsystem"),
) -> Dict[str, Any]:
    """Retrieves logged statistical anomaly events from the Isolation Forest."""
    config = load_config()
    db_path = config.get("storage", {}).get("db_path", "data/telemetry.db")
    anomalies = get_latest_anomalies(limit=limit, entity_class=entity_class, db_path=db_path)
    return {
        "count": len(anomalies),
        "latest_score": latest_anomaly_result.get("anomaly_score", 0.0),
        "latest_is_anomaly": latest_anomaly_result.get("is_anomaly", False),
        "latest_top_features": latest_anomaly_result.get("top_contributing_features", []),
        "events": anomalies,
    }


@app.get("/alerts", summary="Retrieve hardware failure risk alerts")
def get_alerts_endpoint(
    severity: Optional[str] = Query(None, description="Filter by severity ('CRITICAL', 'WARNING', 'INFO')"),
    acknowledged: Optional[int] = Query(None, description="0=active, 1=acknowledged"),
    limit: int = Query(50, ge=1, le=200, description="Max alerts"),
) -> Dict[str, Any]:
    """Fetches system risk alerts generated by the threshold and anomaly rules."""
    config = load_config()
    db_path = config.get("storage", {}).get("db_path", "data/telemetry.db")
    alert_rows = get_alerts(severity=severity, acknowledged=acknowledged, limit=limit, db_path=db_path)
    counts = get_alert_counts(db_path)
    return {
        "total_alerts": counts["total"],
        "active_alerts": counts["active"],
        "critical_alerts": counts["critical"],
        "alerts": alert_rows,
    }


@app.post("/alerts/{alert_id}/acknowledge", summary="Acknowledge an active alert")
def acknowledge_alert_endpoint(alert_id: int) -> Dict[str, Any]:
    """Marks a specific alert as reviewed/acknowledged."""
    config = load_config()
    db_path = config.get("storage", {}).get("db_path", "data/telemetry.db")
    success = acknowledge_alert(alert_id, db_path=db_path)
    if not success:
        raise HTTPException(status_code=404, detail=f"Alert id {alert_id} not found")
    return {"status": "success", "alert_id": alert_id, "acknowledged": True}


@app.get("/model/status", summary="Machine Learning Model Metadata & Hyperparameters")
def get_model_status() -> Dict[str, Any]:
    """Returns architecture, training sample count, and parameters of the Isolation Forest detector."""
    detector = get_anomaly_detector(auto_train=True)
    return {
        "model_type": "IsolationForest",
        "is_fitted": detector.is_fitted,
        "n_estimators": detector.n_estimators,
        "contamination": detector.contamination,
        "max_samples": detector.max_samples,
        "random_state": detector.random_state,
        "trained_at": detector.trained_at,
        "train_samples_count": detector.train_samples_count,
        "feature_columns": detector.feature_columns,
        "baseline_stats": detector.baseline_stats,
    }


@app.post("/model/train", summary="Retrain Anomaly Detection Model")
def retrain_model() -> Dict[str, Any]:
    """Triggers retraining of the Isolation Forest model on available telemetry or nominal baseline."""
    detector = get_anomaly_detector(auto_train=False)
    config = load_config()
    db_path = config.get("storage", {}).get("db_path", "data/telemetry.db")

    from storage.db import get_all_telemetry_df
    df = get_all_telemetry_df(db_path=db_path)

    if len(df) < 50:
        # Augment with synthetic nominal baseline
        from data.synthetic_generator import generate_synthetic_telemetry
        logger.info("Database has <%d records. Bootstrapping with synthetic nominal baseline.", len(df))
        synth = generate_synthetic_telemetry(n_samples=800, anomaly_ratio=0.0, random_seed=42)
        detector.fit(synth)
    else:
        detector.fit(df)

    detector.save()
    return {
        "status": "success",
        "samples_trained": detector.train_samples_count,
        "trained_at": detector.trained_at,
        "n_estimators": detector.n_estimators,
    }


@app.get("/evaluation/metrics", summary="Capstone Quantitative Performance Benchmark")
def get_evaluation_metrics() -> Dict[str, Any]:
    """Returns genuine evaluation metrics tested against labeled failure scenarios and latency benchmarks."""
    report = run_full_evaluation()
    return report


# =========================================================================
# WebSocket Real-Time Stream
# =========================================================================

@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    """WebSocket endpoint streaming live telemetry frames to frontend/dashboards."""
    await manager.connect(websocket)
    try:
        config = load_config()
        db_path = config.get("storage", {}).get("db_path", "data/telemetry.db")
        latest = get_latest_telemetry(limit=3, db_path=db_path)

        # Initial frame
        await websocket.send_json({
            "event": "connected",
            "message": "Connected to edge telemetry real-time stream",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "latest_snapshot": latest,
            "risk_assessment": latest_evaluated_risk,
            "anomaly_detection": latest_anomaly_result,
        })

        while True:
            data = await websocket.receive_text()
            if data.strip().lower() == "ping":
                await websocket.send_json({"event": "pong", "timestamp": datetime.now(timezone.utc).isoformat()})
            else:
                try:
                    payload = json.loads(data)
                    if payload.get("action") == "get_latest":
                        req_entity = payload.get("entity_class")
                        records = get_latest_telemetry(entity_class=req_entity, limit=5, db_path=db_path)
                        await websocket.send_json({"event": "snapshot", "records": records})
                except Exception:
                    await websocket.send_json({"event": "ack", "received": data})
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.warning("WebSocket error: %s", e)
        manager.disconnect(websocket)
