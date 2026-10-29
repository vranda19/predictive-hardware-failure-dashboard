# Edge Sentinel: Predictive Hardware Failure Dashboard

Edge-ready predictive hardware monitoring, anomaly detection, failure risk forecasting, and real-time operations dashboard.

[![Tests](https://img.shields.io/badge/tests-33%20passed-brightgreen.svg)](tests/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](pyproject.toml)
[![Architecture](https://img.shields.io/badge/FastAPI-0.110-teal.svg)](api/app.py)
[![ML](https://img.shields.io/badge/scikit--learn-Isolation%20Forest-orange.svg)](models/anomaly.py)
[![UI](https://img.shields.io/badge/Frontend-React%2018%20SPA-61dafb.svg)](dashboard/)

---

## 🎯 6-Phase Capstone Progression

| Phase | Description | Status |
|---|---|---|
| **Mission 1** | **Scaffold, Telemetry Schema, Collector Agent, SQLite WAL Storage, FastAPI Skeleton, Sliding Window Pipeline** | **COMPLETE** |
| **Mission 2** | **Synthetic Failure Dataset Generation (10-Day Horizon) & Stress Injection Engine** | **COMPLETE** |
| **Mission 3** | **Machine Learning Anomaly Detection (Isolation Forest), Multi-Factor Risk Assessment Engine, React 18 Dashboard** | **COMPLETE** |
| **Mission 4** | **Model Evaluation Harness (Empirical Confusion Matrix, Accuracy, Precision, Recall, F1, MCC, Edge Latency Profiling)** | **COMPLETE** |
| **Mission 5** | Remaining Useful Life (RUL) Regression & Supervised Lead-Time Prediction (LSTM-CNN Sequence Model) | Planned |
| **Mission 6** | Autonomous Edge Deployment Daemon (systemd / Windows NSSM Service) | Planned |

---

## 📁 Repository Layout

```
predictive hardware failure dashboard/
├── api/                   # FastAPI backend & WebSocket streaming service
│   ├── __init__.py
│   └── app.py            # REST endpoints, WebSocket manager, static UI mount
├── collector/             # On-device background telemetry collector
│   ├── __init__.py
│   ├── agent.py          # Continuous collection daemon & window dispatcher
│   └── readers.py        # psutil system metrics & smartctl SMART readers
├── configs/               # Global and subsystem configuration
│   ├── __init__.py
│   └── config.yaml       # Sampling intervals, hosts, DB paths, CORS origins
├── dashboard/             # Interactive React 18 Single-Page Application
│   ├── index.html        # Host page linking React 18, Babel, and styles
│   ├── app.jsx           # React component tree, gauges, area charts, tabs
│   └── styles.css        # Cyber-dark glassmorphism design system
├── data/                  # SQLite storage and synthetic datasets
│   ├── telemetry.db      # SQLite WAL database (telemetry, alerts, anomaly_events)
│   └── synthetic/        # Labeled evaluation and nominal baseline datasets
│       ├── baseline_nominal.csv
│       └── benchmark_test_labeled.csv
├── docs/                  # Documentation and visual evidence
│   └── screenshots/      # Prototype screenshots of working dashboard
├── evaluation/            # Quantitative evaluation and latency benchmarking
│   ├── __init__.py
│   └── evaluator.py      # Confusion matrix, F1, MCC, and system profiling
├── features/              # Feature engineering & sliding windows (Chevtchenko et al.)
│   ├── __init__.py
│   └── windowing.py      # Circular buffer and rolling statistical extraction
├── models/                # Machine learning models and risk engine
│   ├── __init__.py
│   ├── anomaly.py        # Isolation Forest detector with Z-score attribution
│   ├── risk_engine.py    # Multi-subsystem failure risk evaluator & alert dispatch
│   └── saved/            # Serialized model artifacts (.joblib)
├── storage/               # SQLite edge storage engine, DDL schema, and migration
│   ├── __init__.py
│   ├── db.py             # WAL queries, alert transactions, Parquet export
│   ├── init_db.py        # Database migration script
│   ├── models.py         # Pydantic v2 data models
│   └── schema.sql        # DDL definitions and composite indexes
├── tests/                 # Automated test suite (33 Pytest tests)
│   ├── test_api.py
│   ├── test_collector.py
│   ├── test_evaluation.py
│   ├── test_model.py
│   ├── test_risk_engine.py
│   ├── test_storage.py
│   └── test_windowing.py
├── CAPSTONE_EVALUATION.md # Comprehensive university evaluation report
├── Makefile               # Task automation
├── pyproject.toml         # Packaging configuration
├── requirements.txt       # Core project dependencies
└── run.py                 # Cross-platform CLI task runner
```

---

## ⏱️ Sliding Window Pipeline (Chevtchenko et al., arXiv:2310.14949)

Located in `features/windowing.py`:
1. **`SlidingWindowBuffer`**: An edge-ready circular memory buffer emitting windows of length $W=12$ with stride $S=1$.
2. **`compute_window_features`**: Computes rolling statistical indicators per window:
   - Rolling Mean ($\mu$) & Volatility ($\sigma$)
   - Minimum, Maximum, and Median
   - Peak-to-Peak Amplitude ($PTP$)
   - Linear Trend Slope ($\beta$) for early thermal runaway and memory leak detection.
3. **`extract_sliding_windows`**: Offline 3D sequence array extractor (`[N_windows, W, N_features]`) ready for Phase 5 deep learning models.

---

## 🧪 Empirical Evaluation & Performance Benchmarks

All metrics are measured on genuine datasets using `run.py evaluate`:

### 1. Classification Metrics (Benchmark Test Set: $N = 1,500$, $15\%$ Failure Injection)

| Metric | Real Empirical Value | Formula / Details |
|---|---|---|
| **Accuracy** | **92.07%** ($0.9207$) | $(TP + TN) / (TP + TN + FP + FN)$ |
| **Precision** | **75.36%** ($0.7536$) | $TP / (TP + FP)$ |
| **Recall (Sensitivity)** | **69.64%** ($0.6964$) | $TP / (TP + FN)$ |
| **F1-Score** | **72.39%** ($0.7239$) | $2 \cdot \frac{P \cdot R}{P + R}$ |
| **Matthews Correlation (MCC)** | **0.6784** | Class-balanced correlation coefficient |
| **Inference Latency** | **0.012 ms** ($12\,\mu\text{s}$) | $>86,000$ samples/sec throughput |

### 2. Edge System Profiling

- **Hardware Sampling Latency**: $104\text{ ms}$ (psutil + platform readers)
- **SQLite Batch Insert (100 Rows)**: $13.7\text{ ms}$ ($0.137\text{ ms}$ per row with WAL)
- **SQLite Query Latency (100 Rows)**: $2.7\text{ ms}$
- **Sliding Window Feature Extraction (70 features)**: $3.74\text{ ms}$
- **Collector Memory Overhead (RSS)**: $177.1\text{ MB}$

---

## 🚀 Quickstart & Execution Instructions

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Initialize Database Schema
```bash
python run.py init-db
```

### 3. Generate Benchmark Datasets
```bash
python run.py generate-synthetic-data
```

### 4. Train Anomaly Detection Model
```bash
python run.py train-model
```

### 5. Run the Automated Test Suite (33 Tests)
```bash
python run.py test
# or: pytest tests -v
```

### 6. Run the Technical Evaluation Benchmark
```bash
python run.py evaluate
```

### 7. Launch the FastAPI Service & React Dashboard
```bash
python run.py api --port 8000
```

- **Interactive Dashboard**: Open [http://127.0.0.1:8000/dashboard](http://127.0.0.1:8000/dashboard) or [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Interactive Swagger API Docs**: Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **System Health Endpoint**: Open [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

## 🌐 API Endpoints Reference

| HTTP Method | Route | Description |
|---|---|---|
| `GET` | `/` | Web UI Dashboard redirect |
| `GET` | `/dashboard` | Interactive React 18 single-page dashboard |
| `GET` | `/health` | Service health, active alerts count, and model status |
| `GET` | `/telemetry/latest` | Most recent telemetry samples filtered by subsystem |
| `GET` | `/telemetry/history` | Historical telemetry records within lookback hours |
| `GET` | `/window/latest` | Sliding window contents and extracted statistical features |
| `POST` | `/telemetry/collect` | Triggers an immediate on-demand collection cycle |
| `GET` | `/risk/assessment` | Real-time multi-subsystem failure risk scores and recommendations |
| `GET` | `/anomalies/latest` | Recent detected anomaly events and feature Z-score attributions |
| `GET` | `/alerts` | Active and historical hardware risk alerts |
| `POST` | `/alerts/{id}/acknowledge` | Marks a specific alert as acknowledged |
| `GET` | `/model/status` | Model parameters, training metadata, and baseline stats |
| `POST` | `/model/train` | Retrains Isolation Forest model on available telemetry |
| `GET` | `/evaluation/metrics` | Quantitative evaluation report (confusion matrix, F1, latencies) |
| `WebSocket` | `/ws/telemetry` | Real-time bi-directional streaming feed for frontend subscribers |

---

## 📸 Prototype Screenshots

Genuine screenshots captured from the running prototype are stored in `docs/screenshots/`:
- `docs/screenshots/dashboard_initial_1793271005996.png`: Dashboard overview with circular gauges and real-time CPU/RAM charts.
- `docs/screenshots/alerts_tab_content_1793271029523.png`: Alert register and actionable remediation guidance.
- `docs/screenshots/evaluation_tab_content_1793271038424.png`: Capstone evaluation metrics, confusion matrix, and sensitivity breakdown.
- `docs/screenshots/websocket_tab_content_1793271049263.png`: Live WebSocket stream terminal feed.

For complete academic capstone evaluation documentation, see [CAPSTONE_EVALUATION.md](CAPSTONE_EVALUATION.md).
