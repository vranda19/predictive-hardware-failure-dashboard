# Capstone Mid-Term Progress & Phase III Evaluation Report
## Predictive Hardware Failure Dashboard Using Machine Learning

**Academic Evaluation Session**: Capstone Phase III / Mid-Term Review  
**Project Title**: Edge-Ready Predictive Hardware Failure Monitoring, Anomaly Detection, and Risk Assessment System  
**System Status**: Working Prototype Implemented, Empirically Benchmarked, and Verified  
**Host Environment Tested**: Windows 11 AMD64, Python 3.12.10, SQLite 3 (WAL Mode), FastAPI 0.110, React 18  

---

## 1. Executive Summary

This project delivers a cross-platform, edge-ready hardware failure prediction and health monitoring system. The solution continuously samples hardware telemetry (CPU, RAM, storage SMART attributes, system load, and thermal indicators), extracts rolling temporal statistics via Chevtchenko-style sliding window buffers, identifies multi-sensor statistical anomalies using an unsupervised **Isolation Forest** ensemble, and evaluates subsystem failure risk via a multi-factor risk assessment engine. 

All components—including the background data collector, SQLite edge storage engine, machine learning pipeline, FastAPI asynchronous backend, and real-time React 18 dashboard—have been fully implemented, integrated, and verified against genuine hardware telemetry and reproducible failure injection benchmarks.

> [!IMPORTANT]
> **Scientific Integrity & Boundary Condition Note**: In accordance with academic guidelines, this system explicitly treats anomaly detection as a statistical indicator of operational deviation and pre-failure distress, rather than a deterministic guarantee of physical hardware failure. Real-world mechanical and solid-state hardware degradations depend on stochastic environmental and wear-out factors; the model provides proactive risk probabilities to enable timely remediation before catastrophic outage.

---

## 2. System Architecture & Data Flow

### 2.1 High-Level Architectural Decomposition

The system is decomposed into four discrete tiers: Edge Telemetry Ingestion, Preprocessing & Feature Engineering, Analytics & Storage Core, and Interface & Presentation.

```mermaid
graph TD
    subgraph "Tier 1: Hardware & Edge Collection"
        H1["Physical Host Telemetry<br/>(psutil, thermal sensors)"] --> CA["TelemetryCollector Agent<br/>(Background Daemon / Task)"]
        H2["Storage Controllers<br/>(smartctl / NVMe SMART)"] --> CA
        H3["Synthetic Benchmark Generator<br/>(Failure Injection Modes)"] -.-> CA
    end

    subgraph "Tier 2: Preprocessing & Sliding Window"
        CA --> SWB["SlidingWindowBuffer<br/>(W=12, S=1 Circular Buffer)"]
        SWB --> FE["compute_window_features<br/>(Mean, Std, Min, Max, Slope)"]
    end

    subgraph "Tier 3: Storage & Analytical Core"
        CA --> DB[("SQLite Database<br/>(WAL Mode, Indexed)")]
        FE --> RE["HardwareRiskEngine<br/>(Weighted Composite Risk)"]
        FE --> IF["IsolationForest Detector<br/>(100 Trees, Contamination=0.05)"]
        IF --> RE
        RE --> AL["Alert Dispatcher<br/>(Deduplicated Persistence)"]
        AL --> DB
    end

    subgraph "Tier 4: Service & Presentation"
        DB --> API["FastAPI Asynchronous Backend<br/>(REST Endpoints + WebSocket)"]
        RE --> API
        API --> WS["WebSocket Stream<br/>(/ws/telemetry)"]
        API --> UI["React 18 Dashboard<br/>(Gauges, Area Charts, Alerts)"]
        WS --> UI
    end
```

### 2.2 Telemetry Ingestion & Scoring Workflow

```mermaid
sequenceDiagram
    autonumber
    participant HW as Hardware Telemetry / OS
    participant Col as Telemetry Collector
    participant Win as Sliding Window Buffer
    participant ML as Isolation Forest Model
    participant Risk as Risk Assessment Engine
    participant DB as SQLite Storage (WAL)
    participant API as FastAPI Server
    participant UI as React Dashboard

    loop Every Sampling Interval (10 seconds)
        Col->>HW: Sample CPU%, RAM%, SMART attributes, Load Avg
        Col->>DB: Batch Insert validated TelemetryRecord
        Col->>Win: Push sample into circular buffer (W=12)
        Win-->>Col: Emit window statistical vector (slope, volatility)
        Col->>ML: Evaluate sample vector (score_sample)
        ML-->>Col: Anomaly score [0, 1] + Z-score feature attributions
        Col->>Risk: assess_comprehensive_risk(sample, window, anomaly_score)
        Risk->>DB: Persist candidate alerts (deduplicated)
        Risk-->>Col: Composite risk score & categorical risk level
        Col->>API: Broadcast live payload over WebSocket
        API->>UI: Real-time JSON stream update
        UI->>UI: Re-render animated circular dials, SVG charts & alert tables
    end
```

---

## 3. Technology Stack & Module Inventory

| Module / Component | Technologies Employed | Architectural Role | Current Status |
|---|---|---|---|
| **Collector Daemon** | Python 3.12, `psutil`, `subprocess`, `socket` | Platform-agnostic background service gathering system metrics and disk SMART attributes. | **Fully Implemented & Verified** |
| **Preprocessing & Windowing** | NumPy 1.26, Pandas 2.2, `collections.deque` | Chevtchenko et al. circular buffer emitting $W=12$ rolling windows and 7 statistical indicators per metric. | **Fully Implemented & Verified** |
| **Storage Engine** | SQLite 3, WAL Mode, PyArrow (Parquet) | Low-overhead ACID relational storage with compound indexes on `(timestamp, entity_class, host_id)`. | **Fully Implemented & Verified** |
| **Anomaly Detection Model** | Scikit-Learn 1.9, `joblib`, `StandardScaler` | Unsupervised Isolation Forest (100 estimators, $c=0.05$) mapping raw decision boundaries to calibrated $[0, 1]$ severity. | **Fully Implemented & Verified** |
| **Risk Assessment Engine** | Python, Heuristic Rule Tables, Z-Score | Multi-subsystem scoring weighting disk integrity (35%), thermal/CPU (35%), and memory leakage (30%). | **Fully Implemented & Verified** |
| **Backend API** | FastAPI 0.110, Uvicorn, WebSockets | Asynchronous RESTful API exposing telemetry queries, model metadata, alert management, and live streaming. | **Fully Implemented & Verified** |
| **Frontend Dashboard** | React 18, Babel Standalone, Vanilla CSS | Responsive dark-mode dashboard with circular gauges, SVG area charts, alert acknowledgments, and WebSocket feed. | **Fully Implemented & Verified** |
| **Evaluation Harness** | Pytest 9.0, Scikit-Learn Metrics | Automated benchmark assessing confusion matrix, precision, recall, F1, MCC, I/O latency, and RAM footprint. | **Fully Implemented & Verified** |

---

## 4. Methodology and Subsystem Implementation

### 4.1 Telemetry Collection Pipeline
The collector agent (`collector/agent.py`) executes a sampling cycle every 10 seconds. It captures three distinct hardware entity domains per cycle:
1. **CPU Entity**: Total processor utilization (`psutil.cpu_percent`), 1m/5m/15m system load averages, and thermal junction temperatures.
2. **RAM Entity**: Total virtual memory utilization percentage (`psutil.virtual_memory().percent`), allocated memory, and swap metrics.
3. **Storage (HDD/SSD) Entity**: Non-destructive SMART queries via `smartctl` (ATA Attributes 5, 197, 1, 194, and NVMe health logs). If `smartctl` or thermal sensors are unavailable on the target host (e.g., standard Windows non-administrator terminals), the reader gracefully falls back to `null` values without throwing exceptions.

### 4.2 Sliding Window Feature Pipeline (Chevtchenko et al., arXiv:2310.14949)
Raw point-in-time metrics are noisy and cannot capture progressive wear trends. The system implements a stateful circular buffer (`features/windowing.py`):
- **Window Length ($W$)**: 12 samples (corresponding to 2 minutes of historical context at 10s intervals).
- **Stride ($S$)**: 1 sample step.
- **Extracted Feature Vector**: Mean ($\mu$), Standard Deviation ($\sigma$), Minimum, Maximum, Median, Peak-to-Peak Amplitude ($PTP$), and Linear Trend Slope ($\beta = \frac{\sum (t - \bar{t})(x - \bar{x})}{\sum (t - \bar{t})^2}$). Positive memory slopes indicate active memory leaks, while positive temperature slopes identify thermal runaway.

### 4.3 Machine Learning Anomaly Detection Model
Located in `models/anomaly.py`:
- **Model**: `sklearn.ensemble.IsolationForest`.
- **Feature Space (8 Features)**: `cpu_usage_pct`, `memory_usage_pct`, `temperature_c`, `reallocated_sectors`, `pending_sectors`, `read_error_rate`, `disk_temp_c`, `load_avg_1m`.
- **Pre-processing**: Missing value imputation using baseline column medians followed by `StandardScaler` feature normalization.
- **Score Calibration**: Scikit-learn's decision function $s(x) \in [-0.5, 0.5]$ is mapped to an intuitive probability interval $[0.0, 1.0]$ via a calibrated logistic transform:
  $$\text{Score}_{\text{calibrated}} = \frac{1}{1 + e^{6 \cdot s(x)}}$$
- **Root-Cause Explainability**: Evaluates per-feature Z-score deviation against fitted baseline statistics ($Z_j = \frac{|x_j - \mu_j|}{\sigma_j}$) to return the top 3 contributing signals for each detected anomaly.

### 4.4 Risk Assessment & Alert Engine
Located in `models/risk_engine.py`:
- Evaluates individual component risk ratings ($0 - 100\%$):
  - **CPU Risk**: Scaled by threshold breaches ($75^\circ\text{C}$ warning, $88^\circ\text{C}$ critical; $85\%$ load warning, $95\%$ critical) plus sliding window thermal slope.
  - **RAM Risk**: Scaled by memory pressure ($80\%$ warning, $92\%$ critical) plus memory accumulation trajectory slope ($\beta > 0.8$).
  - **Disk SMART Risk**: Scaled by reallocated sector increments ($>0$ warning, $>20$ critical), pending sector growth, and drive health status.
- **Composite Risk Index**:
  $$\text{Risk}_{\text{composite}} = \max\left(0.35 \cdot R_{\text{disk}} + 0.35 \cdot R_{\text{cpu}} + 0.30 \cdot R_{\text{ram}}, \; 0.70 \cdot R_{\text{weighted}} + 0.30 \cdot (100 \cdot \text{Score}_{\text{anomaly}})\right)$$
- **Alert Dispatch & Deduplication**: Generates structured alert records (`INFO`, `WARNING`, `CRITICAL`) with specific engineering remediation instructions. An in-memory timestamp cache prevents alert flooding by enforcing a 60-second deduplication cooldown per unique alert condition.

---

## 5. Database Schema & Storage Design

The edge storage engine (`storage/db.py`, `storage/schema.sql`) operates over SQLite 3 configured with **Write-Ahead Logging (WAL)** and normal synchronous mode to provide concurrent read-write access without lock contention.

### 5.1 Tables DDL Summary

```sql
-- Core Telemetry Time-Series Table
CREATE TABLE IF NOT EXISTS telemetry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    cpu_usage_pct REAL NOT NULL,
    memory_usage_pct REAL NOT NULL,
    temperature_c REAL,
    reallocated_sectors INTEGER,
    pending_sectors INTEGER,
    read_error_rate REAL,
    disk_temp_c REAL,
    disk_status TEXT,
    load_avg_1m REAL,
    load_avg_5m REAL,
    load_avg_15m REAL,
    entity_class TEXT NOT NULL CHECK(entity_class IN ('HDD', 'RAM', 'CPU')),
    host_id TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_telemetry_timestamp ON telemetry (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_telemetry_entity_ts ON telemetry (entity_class, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_telemetry_host_ts ON telemetry (host_id, timestamp DESC);

-- Automated Hardware Risk Alerts Table
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    entity_class TEXT NOT NULL,
    severity TEXT NOT NULL CHECK(severity IN ('INFO', 'WARNING', 'CRITICAL')),
    metric_name TEXT NOT NULL,
    metric_value REAL,
    threshold REAL,
    message TEXT NOT NULL,
    remediation TEXT,
    acknowledged INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts (severity, timestamp DESC);

-- Machine Learning Anomaly Detection Events Log
CREATE TABLE IF NOT EXISTS anomaly_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    entity_class TEXT NOT NULL,
    anomaly_score REAL NOT NULL,
    is_anomaly INTEGER NOT NULL,
    feature_summary TEXT,
    host_id TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_anomaly_timestamp ON anomaly_events (timestamp DESC);
```

---

## 6. Dataset and Experimental Setup

To adhere strictly to university scientific standards and avoid inventing data, the project establishes a rigorous dual-dataset protocol:

### 6.1 Real System Host Telemetry
- **Source**: Local physical workstation node (Hostname: `Suryansh`, Windows 11 AMD64).
- **Sampling Interval**: 10.0 seconds.
- **Observations Collected in Database**: Over 175 real observations persisted across CPU, RAM, and storage entities.
- **Characteristics**: Reflects actual operating background workloads (CPU dynamic variance 10%–45%, physical RAM utilization 65%–74%, platform sensor availability).

### 6.2 Controlled Synthetic Benchmark Dataset (`data/synthetic_generator.py`)
Because physical destructive failure testing on personal workstation hardware is non-viable, a controlled failure scenario generator produces reproducible, labeled benchmark sets for model validation:
- **Baseline Training Set (`baseline_nominal.csv`)**: 1,000 nominal observations ($100\%$ normal operating regime, ground-truth label $y = 0$).
- **Benchmark Evaluation Set (`benchmark_test_labeled.csv`)**: 1,500 labeled sequential observations containing $15\%$ ground-truth anomalies ($N_{\text{anomaly}} = 224$, $N_{\text{nominal}} = 1276$).
- **Injected Failure Modes**:
  1. *Thermal Runaway*: Rapid junction temperature climb from $58^\circ\text{C}$ to $96.5^\circ\text{C}$ with accompanying load throttle.
  2. *Memory Leak*: Linear RAM allocation ramp from $55\%$ to $99.4\%$ over consecutive windows.
  3. *SMART Degradation*: Accumulation of reallocated sectors (0 to 80), pending sectors (0 to 45), read error rate explosion, and SMART status transition to `FAILED`.
  4. *CPU Starvation*: Processor utilization pegged at $96\% - 100\%$ with elevated load averages.

---

## 7. Automated Testing and Validation

A 33-test automated regression and integration test suite was created and executed using `pytest`.

### 7.1 Test Execution Matrix

| Test Suite File | Test Scenario Description | Expected Outcome | Actual Result | Status |
|---|---|---|---|---|
| `tests/test_collector.py` | Host telemetry reader structure | Keys match schema, values within valid ranges | Valid dictionary returned | **PASSED** |
| `tests/test_collector.py` | SMART reader graceful fallback | Return null defaults without subprocess crash | Null dictionary returned | **PASSED** |
| `tests/test_collector.py` | Single collection cycle execution | 3 entities (CPU, RAM, HDD) persisted to DB | 3 records inserted | **PASSED** |
| `tests/test_storage.py` | SQLite schema and indexes initialization | Tables and compound indices created | Schema active | **PASSED** |
| `tests/test_storage.py` | Single and batch record insertion | Records stored with autoincrement ID | IDs returned correctly | **PASSED** |
| `tests/test_storage.py` | Check constraint on entity_class | Reject values not in ('HDD','RAM','CPU') | `IntegrityError` raised | **PASSED** |
| `tests/test_storage.py` | Parquet export pipeline | SQLite table serialized to Parquet format | Non-empty file written | **PASSED** |
| `tests/test_windowing.py`| Circular sliding window buffer | Emit window of length $W=12$ every $S=1$ step | Emits expected array | **PASSED** |
| `tests/test_windowing.py`| 3D sliding window array generator | Output shape $(N, W, F)$ for sequential models | Exact shape returned | **PASSED** |
| `tests/test_windowing.py`| Window statistical feature computation | Calculate mean, std, min, max, slope | Correct dictionary keys | **PASSED** |
| `tests/test_model.py` | Isolation Forest initialization | Parameters set, `is_fitted` initially False | Correct properties | **PASSED** |
| `tests/test_model.py` | Model fitting and serialization | Save to `.joblib` and restore state | Identical inference output | **PASSED** |
| `tests/test_model.py` | Scoring nominal vs extreme stress | Anomaly score for stress > nominal | Stress marked anomaly | **PASSED** |
| `tests/test_model.py` | Missing feature imputation | Handle null sensors via median substitution | Output score in $[0, 1]$ | **PASSED** |
| `tests/test_risk_engine.py` | CPU thermal and load risk scoring | Critical temperature triggers critical risk | Risk score $\ge 70$ | **PASSED** |
| `tests/test_risk_engine.py` | RAM memory leak slope scoring | Positive slope increases risk score | Risk score elevated | **PASSED** |
| `tests/test_risk_engine.py` | Storage SMART degradation scoring | Reallocated sectors trigger critical alert | Alert persisted | **PASSED** |
| `tests/test_risk_engine.py` | Comprehensive risk and alert deduplication | Avoid duplicate alerts within 60s cooldown | Deduplication verified | **PASSED** |
| `tests/test_evaluation.py` | Anomaly detector evaluation on benchmark | Compute confusion matrix and valid metrics | Sum matches sample count | **PASSED** |
| `tests/test_evaluation.py` | Edge system performance benchmark | Return non-zero I/O and latency metrics | Latencies recorded | **PASSED** |
| `tests/test_api.py` (12 tests) | Endpoints: `/health`, `/telemetry/*`, `/window/latest`, `/risk/assessment`, `/anomalies/latest`, `/alerts`, `/model/*`, `/evaluation/metrics`, `/ws/telemetry`, `/` | HTTP 200 with contract JSON; WebSocket ping/pong | All 12 endpoints verified | **PASSED** |

**Total Test Results: 33 passed in 4.61s (100% Pass Rate).**

---

## 8. Quantitative Performance Benchmarks

All performance figures reported below were measured empirically on the live system using `evaluation/evaluator.py`:

### 8.1 Machine Learning Classification Performance

Tested on the labeled evaluation dataset (`benchmark_test_labeled.csv`, $N = 1,500$ samples, $N_{\text{anomaly}} = 224$, $N_{\text{nominal}} = 1,276$):

| Metric | Real Empirical Value | Interpretation |
|---|---|---|
| **Overall Accuracy** | **92.07%** ($0.9207$) | Proportion of correct predictions across all classes |
| **Precision** | **75.36%** ($0.7536$) | True positive rate among all flagged anomalies ($TP / (TP + FP)$) |
| **Recall (Sensitivity)** | **69.64%** ($0.6964$) | Proportion of ground-truth failures detected ($TP / (TP + FN)$) |
| **F1-Score** | **72.39%** ($0.7239$) | Harmonic mean of Precision and Recall |
| **Matthews Correlation (MCC)** | **0.6784** | High correlation metric resilient to class imbalance |
| **Inference Latency per Sample** | **0.012 ms** ($12.0\,\mu\text{s}$) | Capable of processing over $86,000$ samples per second |

#### Empirical Confusion Matrix

| Actual \ Predicted | Predicted Normal ($y=0$) | Predicted Anomaly ($y=1$) | Total |
|---|---|---|---|
| **Actual Normal ($y=0$)** | **$TN = 1,225$** | $FP = 51$ | $1,276$ |
| **Actual Anomaly ($y=1$)** | $FN = 68$ | **$TP = 156$** | $224$ |
| **Total** | $1,293$ | $207$ | $1,500$ |

#### Failure Mode Sensitivity Breakdown

| Failure Scenario Mode | Total Injected Samples | Anomalies Detected by Isolation Forest | Detection Rate |
|---|---|---|---|
| **Thermal Runaway** | 56 | 56 | **100.0%** |
| **CPU Starvation & Overload** | 56 | 56 | **100.0%** |
| **Memory Leak Progression** | 56 | 37 | **66.07%** |
| **SMART Degradation** | 56 | 7 | **12.5%** *(addressed below)* |
| **Nominal Baseline** | 1,276 | 51 (false alarms) | $4.0\%$ False Positive Rate |

> [!NOTE]
> **Scientific Finding regarding SMART Degradation**: Isolation Forest operates in multi-dimensional Euclidean space. When SMART error counters begin drifting from 0 to small integers (e.g. 1 to 5), the absolute variance is small compared to percentage fluctuations in CPU and RAM. This causes the pure ML detector to detect early SMART drift at only 12.5% sensitivity until counters become extreme. This empirical finding validates our hybrid design: the **HardwareRiskEngine** applies explicit threshold rules that catch SMART sector reallocations immediately on first occurrence, compensating for the unsupervised ML limitation.

### 8.2 Edge System Latency & Resource Overhead

| Pipeline Operation | Measured Latency | Academic Assessment |
|---|---|---|
| **Hardware Sensor Query (`read_system_metrics`)** | $104.04\text{ ms}$ | Negligible overhead for a 10s sampling cycle ($<1.1\%$ duty cycle) |
| **SQLite Batch Insert (100 Records)** | $13.70\text{ ms}$ | Average of $0.137\text{ ms}$ per row with WAL mode |
| **SQLite Query Retrieval (100 Records)** | $2.70\text{ ms}$ | Sub-3ms query speed for instantaneous dashboard loading |
| **Sliding Window Feature Extraction (70 features)** | $3.74\text{ ms}$ | Fast rolling statistical calculation |
| **Process RSS Memory Footprint** | $177.12\text{ MB}$ | Suitable for edge gateways, IPCs, and low-power servers |
| **Idle Background CPU Utilization** | $0.00\% - 0.20\%$ | Minimal impact on host system workload |

---

## 9. Visual Evidence of Working Prototype

Genuine screenshots of the operational prototype were captured during headless browser automated verification:

| Artifact / Screenshot | Primary Content & Verification Focus | File Location |
|---|---|---|
| **Dashboard Overview** | Real-time circular gauges, CPU/RAM utilization area charts, active host indicators (`Suryansh`), Chevtchenko sliding window panel. | `docs/screenshots/dashboard_initial_1793271005996.png` |
| **Alerts & Remediation Tab** | Active hardware warning register, observed values vs thresholds, actionable remediation advice, acknowledge action. | `docs/screenshots/alerts_tab_content_1793271029523.png` |
| **Evaluation & Benchmark Tab** | Live empirical confusion matrix, accuracy ($92.1\%$), precision ($75.4\%$), F1 ($72.4\%$), edge latencies, failure mode sensitivity breakdown. | `docs/screenshots/evaluation_tab_content_1793271038424.png` |
| **WebSocket Stream Inspector** | Terminal feed showing continuous JSON stream broadcast from `/ws/telemetry`. | `docs/screenshots/websocket_tab_content_1793271049263.png` |
| **Post-Collection State** | Updated record counter and verified response after manual on-demand collection sample trigger. | `docs/screenshots/dashboard_final_1793271088524.png` |

---

## 10. Technical Challenges, Limitations, and Mitigation Strategies

1. **Host Sensor Heterogeneity (Windows vs Linux)**:
   - *Limitation*: Under Windows, `psutil.sensors_temperatures()` is not natively exposed by standard Win32 APIs without third-party kernel drivers (e.g. LibreHardwareMonitor/OpenHardwareMonitor). Additionally, `smartctl` requires administrator privileges and installation of `smartmontools`.
   - *Mitigation*: Implemented non-blocking sensor fallback with singleton warning state flags (`collector/readers.py`). Null readings are safely imputed during model scoring via median substitution (`models/anomaly.py`).
2. **SMART Counter Sensitivity in Unsupervised Models**:
   - *Limitation*: Unsupervised tree partitions struggle with rare discrete step changes (e.g., sector reallocation from 0 to 1).
   - *Mitigation*: Built a hybrid architecture where the rule-based `HardwareRiskEngine` acts alongside the Isolation Forest, guaranteeing immediate alert generation whenever critical SMART thresholds are breached.
3. **False Positive Management**:
   - *Limitation*: Temporary bursty workloads (e.g., software compilation or file compression) cause temporary CPU and RAM spikes that resemble anomalies.
   - *Mitigation*: Sliding window slope analysis prevents immediate alert escalation on brief spikes, requiring sustained elevation before triggering critical failure alerts. Deduplication intervals prevent alert spam.

---

## 11. Capstone Milestone Tracker & Phase IV/V Roadmap

```mermaid
gantt
    title Capstone Project Timeline & Milestone Progress
    dateFormat  YYYY-MM-DD
    section Phase I: Scaffold & Storage
    Edge Telemetry Collector & SQLite WAL       :done, p1, 2026-09-01, 2026-09-20
    Sliding Window Pipeline (Chevtchenko)       :done, p2, 2026-09-21, 2026-10-05
    section Phase II: Datasets & Scenarios
    Synthetic Failure Generator & Benchmarks    :done, p3, 2026-10-06, 2026-10-18
    section Phase III: ML & Prototype (Current)
    Isolation Forest Anomaly Detector           :done, p4, 2026-10-19, 2026-10-25
    Hardware Risk Assessment Engine             :done, p5, 2026-10-25, 2026-10-28
    FastAPI Endpoints & WebSocket Server        :done, p6, 2026-10-27, 2026-10-29
    React 18 Dashboard & Live Gauges            :done, p7, 2026-10-28, 2026-10-29
    Quantitative Benchmarking & 33 Tests        :done, p8, 2026-10-29, 2026-10-29
    section Phase IV: Supervised & Deep Models
    LSTM-CNN Hybrid Sequence Predictor         :active, p9, 2026-11-01, 2026-11-20
    Remaining Useful Life (RUL) Regression      :p10, 2026-11-21, 2026-12-05
    section Phase V: Edge Deployment
    Cross-Platform Daemon Packaging (Windows/Linux):p11, 2026-12-06, 2026-12-20
    Final Capstone Defense & Demonstration      :p12, 2026-12-21, 2026-12-31
```

### Detailed Milestone Status Table

| Milestone | Deliverable | Target Phase | Status | Technical Evidence |
|---|---|---|---|---|
| **M1.1** | Edge Telemetry Collector Agent | Phase I | **Completed** | `collector/agent.py`, `collector/readers.py` |
| **M1.2** | SQLite WAL Storage & Parquet Exporter | Phase I | **Completed** | `storage/db.py`, `storage/schema.sql` |
| **M1.3** | Chevtchenko Sliding Window Pipeline | Phase I | **Completed** | `features/windowing.py` (W=12, S=1) |
| **M2.1** | Synthetic Telemetry & Failure Scenarios | Phase II | **Completed** | `data/synthetic_generator.py` |
| **M3.1** | Isolation Forest Anomaly Detection Model | Phase III | **Completed** | `models/anomaly.py` (92.1% Accuracy) |
| **M3.2** | Multi-Factor Hardware Risk Engine | Phase III | **Completed** | `models/risk_engine.py` (CPU, RAM, Disk) |
| **M3.3** | FastAPI Backend Integration & WebSockets | Phase III | **Completed** | `api/app.py` (12 REST routes + `/ws/telemetry`) |
| **M3.4** | Interactive React 18 Dashboard | Phase III | **Completed** | `dashboard/` (Gauges, Area Charts, Alerts) |
| **M3.5** | Automated Test Suite (33 Tests) | Phase III | **Completed** | 100% pass rate in `pytest tests -v` |
| **M3.6** | Empirical Benchmarking Harness | Phase III | **Completed** | `evaluation/evaluator.py` |
| **M4.1** | Supervised Failure Forecasting (LSTM-CNN) | Phase IV | Planned | Sequence modeling over 3D window tensors |
| **M4.2** | Remaining Useful Life (RUL) Estimation | Phase IV | Planned | Regression target estimation on Backblaze dataset |
| **M5.1** | Standalone Windows/Linux Daemon Service | Phase V | Planned | NSSM / systemd deployment scripts |

---

## 12. Verification & Reproducibility Guide for the Project Guide

To verify the working implementation on any evaluation machine:

### 1. Execute the Automated Test Suite (33 Tests)
```bash
python run.py test
# or: pytest tests -v
```

### 2. Run the Quantitative Evaluation Benchmark
```bash
python run.py evaluate
```
*Outputs the empirical confusion matrix, accuracy ($92.1\%$), precision ($75.4\%$), recall ($69.6\%$), F1 ($72.4\%$), MCC ($0.678$), and edge system latencies.*

### 3. Launch the Integrated Server and Dashboard
```bash
python run.py api --port 8000
```
- Open **Dashboard UI**: [http://127.0.0.1:8000/dashboard](http://127.0.0.1:8000/dashboard)
- Open **Interactive API Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Open **Health Endpoint**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
