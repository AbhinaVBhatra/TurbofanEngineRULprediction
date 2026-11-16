# Predictive Maintenance of Turbofan Engines: Remaining Useful Life (RUL) Prediction and Maintenance Decision Support using NASA C-MAPSS

[![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.162.0-009688.svg)](https://fastapi.tiangolo.com/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.7.0-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.15.0+cpu-EE4C2C.svg)](https://pytorch.org/)
[![Tests Passing](https://img.shields.io/badge/PyTest-12%20Passed-success.svg)](https://pytest.org/)

A complete, production-grade, reproducible College Industrial Training Report (ITR) project and live web application developed by **Abhinav Bhatra** (B.Tech Computer Science with Artificial Intelligence, JECRC Foundation) under the mentorship of **LinuxWorld Informatics Pvt. Ltd.**

---

## 📌 Executive Overview & Framing

- **Business Problem**: In commercial aviation and gas turbine operations, unexpected mechanical breakdowns cost between **$100,000 and $1,000,000+ per event** in unscheduled repairs, flight groundings, and passenger compensation. Conversely, rigid calendar-based overhauls discard **25% to 40% of usable engine life** prematurely.
- **Data Problem**: Given multivariate time-series readings from 21 thermodynamic sensors and 3 flight settings across operational cycles, accurately predict Remaining Useful Life (RUL).
- **Analytical Solution**: A leakage-free ML pipeline featuring piecewise-linear RUL targets (capped at 125 cycles following crack-propagation physics), operational regime clustering for multi-condition environments (FD004), Gradient Boosted Trees (LightGBM), PyTorch recurrent networks (GRU), TreeSHAP interpretability, and 80% prediction intervals.
- **Quantified Financial Impact**: Simulating a fleet of 100 turbofans proves that model-driven maintenance at optimal threshold $T^* = 10$ cycles eliminates 100% of catastrophic failures while slashing fleet maintenance expenditure by **79.0% ($15.80 Million saved vs Run-to-Failure)** and **26.14% ($1.49 Million saved vs fixed-interval overhaul)**.
- **CRITICAL TRANSPARENCY NOTICE ON SIMULATED DATA**:
  *The NASA C-MAPSS dataset consists of high-fidelity simulated thermodynamic degradation trajectories generated via the Commercial Modular Aero-Propulsion System Simulation code developed at NASA Glenn Research Center. It represents simulated physics with noise, NOT actual revenue airline flight telemetry.*

---

## 📂 Project Directory Structure

```text
ptm_project/
├── data/
│   ├── raw/                      # Downloaded official NASA C-MAPSS files (FD001 - FD004)
│   ├── processed/                # Normalized datasets with RUL labels and engineered features
│   └── sample_engine.csv         # 196-cycle test engine for live UI testing
├── notebooks/
│   ├── 01_eda.py                 # Lifetime distribution, invariant sensors, degradation curves
│   ├── 02_features.py            # Regime clustering, rolling stats (99 feats), 3D sequence tensors
│   ├── 03_models.py              # Multi-model benchmark (Ridge, RF, LightGBM, PyTorch GRU)
│   ├── 04_explain_uncertainty.py # TreeSHAP attributions & dual-quantile prediction intervals
│   └── 05_decision_cost.py       # Threshold sweep & 100-engine fleet cost simulation
├── src/
│   ├── data.py                   # Automated NASA download, extraction, validation, and labeling
│   ├── features.py               # Sensor cleaning, regime clustering, rolling stats, windowing
│   ├── models.py                 # Estimators (Ridge, RF, LightGBM, PyTorch GRU)
│   ├── evaluate.py               # RMSE, MAE, NASA asymmetric scoring function
│   ├── explain.py                # TreeSHAP wrapper and quantile prediction intervals
│   ├── cost.py                   # Cost modeling (unplanned vs planned vs waste life) & ROI
│   └── inference.py              # Zero-skew inference engine shared with FastAPI backend
├── artifacts/                    # Saved models, scalers, clusterers, feature configs
├── app/
│   ├── main.py                   # FastAPI REST API serving real models and simulation endpoints
│   ├── static/
│   │   ├── css/style.css         # Modern, responsive industrial dashboard styling
│   │   └── js/app.js             # Dynamic charts (Chart.js) driven strictly by live API calls
│   └── templates/
│       └── index.html            # Single-page multi-view application (Fleet, Detail, Upload, Sim, Cost)
├── tests/
│   ├── test_pipeline.py          # Data ingestion and feature consistency unit tests
│   └── test_api.py               # API tests: verifies dynamic predictions, no skew, validation
├── results/
│   ├── figures/                  # 18 high-resolution publication plots
│   ├── tables/                   # Tabular summaries (CSV)
│   └── metrics.json              # Canonical machine-readable metrics from code execution
├── report/
│   ├── figures/                  # Synchronized high-resolution report plots
│   ├── generate_report.py        # Automated report compiler script
│   ├── report.md                 # 16-chapter exhaustive academic training report
│   └── report.docx               # Formatted Word document with embedded figures and tables
├── docs/
│   ├── CHANGELOG_AND_ISSUES.md   # Running engineering log of bugs, fixes, and decisions
│   └── viva_prep.md              # 25 in-depth technical viva questions & answers
├── Dockerfile                    # Containerization specification for lightweight deployment
├── requirements.txt              # Pinned production and training dependencies
├── README.md                     # Complete project documentation and reproduction instructions
└── run_all.sh                    # End-to-end execution script (data -> train -> eval -> docs)
```

---

## 🚀 Quickstart & Reproduction

### 1. Environment Setup
```bash
# Clone and enter directory
cd ptm_project

# Install pinned dependencies
pip install -r requirements.txt
```

### 2. End-to-End Reproduction (Single Command)
To download data, preprocess features, train models, evaluate uncertainty, run cost simulations, execute the PyTest suite, and generate `report.docx`:
```bash
# Linux / macOS / Git Bash
bash run_all.sh

# Or on Windows PowerShell:
python src/data.py
python notebooks/01_eda.py
python notebooks/02_features.py
python notebooks/03_models.py
python notebooks/04_explain_uncertainty.py
python notebooks/05_decision_cost.py
pytest tests/ -v
python report/generate_report.py
```

### 3. Launching the Live Web Application
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Open your browser to: **`http://localhost:8000`**

---

## 📊 Empirical Results Summary

### Multi-Model Benchmark on FD001 (100 Test Engines)
*All metrics from code execution saved in `results/metrics.json`:*

| Model Architecture | Input Features | Test RMSE (Cycles) | Test MAE (Cycles) | NASA Score ($S$) | % Early (Safe) | % Late (Hazardous) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression (Ridge)** | 99 Engineered Feats | 19.908 | 15.176 | 728.63 | 45.0% | 55.0% |
| **Random Forest** | 99 Engineered Feats | 18.474 | 13.343 | 624.83 | 47.0% | 53.0% |
| **LightGBM Regressor (Live)** | **99 Engineered Feats** | **17.160** | **12.327** | **505.77** | **44.0%** | **56.0%** |
| **PyTorch GRU (2-layer)** | **30-cycle 3D Tensor** | **15.223** | **11.036** | **385.39** | **55.0%** | **45.0%** |

*Literature Benchmark Reference (Rule 9)*:
- *Heimes (2008) Challenge Winner*: RMSE = 16.98, Score = 436.5
- *Ramasso (2014) RVM Baseline*: RMSE = 19.82, Score = 1152.0
- *Zheng et al. (2017) CNN/LSTM*: RMSE = 16.14, Score = 338.0

### Financial Impact (100-Engine Fleet Lifecycle)
- **Run-to-Failure Total Cost**: $20,000,000 (100 unplanned breakdowns)
- **Optimal Fixed-Schedule (130 cycles)**: $5,686,600 (1 unplanned failure + $1.53M in wasted component life)
- **Predictive Maintenance ($T^* = 10\text{c}$)**: **$4,200,000 (0 unplanned failures + $200k wasted life)**
- **Net Cost Avoidance**: **$15,800,000 saved vs Run-to-Failure (79.0%)** | **$1,486,600 saved vs Fixed-Schedule (26.14%)**

---

## 🧪 Quality Assurance & Verification Proofs

Run the automated test suite:
```bash
pytest tests/ -v
```
All **12 out of 12 tests pass**, proving:
1. **Zero Train/Serve Skew**: Offline inference on `sample_engine.csv` matches API `POST /predict` output to $< 10^{-4}$ cycles.
2. **Truly Dynamic Predictions**: Perturbing temperature sensors dynamically changes predicted RUL, proving numbers are **never hardcoded**.
3. **Input Validation**: Uploading files with $< 30$ cycles or missing columns returns structured HTTP 422 errors.

---

## 🐳 Docker Deployment

Build and run using the lightweight container specification:
```bash
# Build image
docker build -t turbofan-pdm:latest .

# Run container
docker run -p 8000:8000 turbofan-pdm:latest
```
Access the dashboard at `http://localhost:8000`.

---

## 🎓 Academic Viva Voce Defense

A dedicated **25-question viva voce preparation guide** is available at:
👉 **[docs/viva_prep.md](file:///c:/Users/abhin/.gemini/antigravity/scratch/IndustrialTrainingProject/ptm_project/docs/viva_prep.md)**

Covers: crack propagation physics, the Paris law rationale for the 125-cycle cap, the NASA asymmetric scoring formula, leakage prevention via engine grouping, TreeSHAP thermodynamic mapping, pinball loss for quantile intervals, and real-world project limitations.
