import pytest
from fastapi.testclient import TestClient
from api.app import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "timestamp" in data
    assert "host_id" in data
    assert "live_mode" in data
    assert "total_records" in data
    assert "uptime_seconds" in data
    assert "model_fitted" in data
    assert "active_alerts" in data


def test_telemetry_latest_endpoint(client):
    response = client.get("/telemetry/latest?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert "count" in data
    assert "records" in data
    assert isinstance(data["records"], list)


def test_telemetry_history_endpoint(client):
    response = client.get("/telemetry/history?hours=1")
    assert response.status_code == 200
    data = response.json()
    assert "hours" in data
    assert data["hours"] == 1.0
    assert "records" in data


def test_window_latest_endpoint(client):
    response = client.get("/window/latest?entity_class=CPU&window_size=5")
    assert response.status_code == 200
    data = response.json()
    assert data["entity_class"] == "CPU"
    assert "statistical_features" in data


def test_risk_assessment_endpoint(client):
    response = client.get("/risk/assessment")
    assert response.status_code == 200
    data = response.json()
    assert "composite_risk_score" in data
    assert "risk_level" in data
    assert "component_risks" in data
    assert "cpu_risk" in data["component_risks"]
    assert "ram_risk" in data["component_risks"]
    assert "disk_risk" in data["component_risks"]
    assert "recommendations" in data


def test_anomalies_latest_endpoint(client):
    response = client.get("/anomalies/latest")
    assert response.status_code == 200
    data = response.json()
    assert "count" in data
    assert "latest_score" in data
    assert "events" in data


def test_alerts_endpoint(client):
    response = client.get("/alerts")
    assert response.status_code == 200
    data = response.json()
    assert "total_alerts" in data
    assert "active_alerts" in data
    assert "alerts" in data
    assert isinstance(data["alerts"], list)


def test_model_status_endpoint(client):
    response = client.get("/model/status")
    assert response.status_code == 200
    data = response.json()
    assert data["model_type"] == "IsolationForest"
    assert "is_fitted" in data
    assert "n_estimators" in data
    assert "feature_columns" in data


def test_model_train_endpoint(client):
    response = client.post("/model/train")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["samples_trained"] > 0


def test_evaluation_metrics_endpoint(client):
    response = client.get("/evaluation/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "machine_learning_evaluation" in data
    assert "system_performance_benchmark" in data
    assert "metrics" in data["machine_learning_evaluation"]
    assert "accuracy" in data["machine_learning_evaluation"]["metrics"]


def test_dashboard_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200


def test_websocket_telemetry(client):
    with client.websocket_connect("/ws/telemetry") as websocket:
        initial_msg = websocket.receive_json()
        assert initial_msg["event"] == "connected"
        assert "latest_snapshot" in initial_msg

        websocket.send_text("ping")
        pong_msg = websocket.receive_json()
        assert pong_msg["event"] == "pong"
