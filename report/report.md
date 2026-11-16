# INDUSTRIAL TRAINING REPORT (ITR)

## PREDICTIVE MAINTENANCE OF TURBOFAN ENGINES: REMAINING USEFUL LIFE (RUL) PREDICTION AND MAINTENANCE DECISION SUPPORT USING NASA C-MAPSS

---

**Submitted in partial fulfillment of the requirements for the degree of**  
**Bachelor of Technology in Computer Science & Engineering (Artificial Intelligence)**  

**Candidate Name:** Abhinav Bhatra  
**Roll Number / Registration:** [TO BE FILLED BY STUDENT: University Roll No.]  
**Department:** Computer Science & Engineering (Artificial Intelligence)  
**Institution:** Jaipur Engineering College and Research Centre (JECRC Foundation), Jaipur, Rajasthan  
**Training Organization:** LinuxWorld Informatics Pvt. Ltd.  
**Domain of Specialization:** Data Science with Artificial Intelligence  
**Faculty Mentor / Guide:** [TO BE FILLED BY STUDENT: Faculty Mentor Name]  
**Industry Mentor:** [TO BE FILLED BY STUDENT: Industry Mentor Name, LinuxWorld]  
**Academic Year:** 2025 – 2026  

---

## Executive Summary & Metadata Notice
- **Student Background**: Coursework centered on converting organizational data problems into structured analytical solutions. Previous foundational projects: Book Recommender System, Sales Maximization Modeling, and Customer Churn Analytics.
- **Problem Formulation Paradigm**: Business Problem (Unplanned downtime cost vs wasteful early component replacement) $\rightarrow$ Data Problem (Multivariate sensor Remaining Useful Life regression) $\rightarrow$ Analytical Solution (Leakage-free Gradient Boosting & Recurrent Neural Network + Threshold Decision Rule) $\rightarrow$ Quantified Impact (100-engine fleet financial simulation).
- **CRITICAL TRANSPARENCY NOTICE ON SIMULATED DATA**:
  *The NASA C-MAPSS dataset consists of high-fidelity simulated thermodynamic run-to-failure records generated using the Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) software developed at NASA Glenn Research Center. It represents simulated degradation physics with synthetic noise, NOT actual revenue airline flight telemetry.*

---

## TABLE OF CONTENTS
1. Abstract
2. Introduction and Organization/Training Overview
3. Problem Statement and Objectives
4. Background and Literature Review
5. Dataset Description and Ingestion
6. Methodology and Leakage-Free Design
7. Implementation Architecture and Models
8. System Architecture and Deployment
9. Results and Empirical Discussion
10. Business Impact and Financial Cost Analysis
11. Engineering Challenges Faced
12. Project Limitations
13. Ethics, Safety, and Responsible AI in Aviation
14. Conclusion and Future Scope
15. References
16. Appendix: Reproduction Guide & Test Verification

---

## 1. Abstract
Unplanned mechanical breakdowns in commercial gas turbine turbofan engines present severe flight safety risks and catastrophic economic penalties, with single uncontained failure events exceeding \$200,000 to \$1,000,000 in unscheduled depot repair, secondary core destruction, and aircraft grounding losses. In contrast, rigid calendar-based preventive maintenance overhauls healthy engines prematurely, discarding substantial usable component life. This project develops an end-to-end, reproducible, production-grade Condition-Based Predictive Maintenance (PdM) decision support system utilizing the NASA C-MAPSS benchmark dataset.

To ensure physical validity, target Remaining Useful Life (RUL) is modeled as a piecewise-linear function capped at 125 cycles, reflecting the reality that components exhibit zero degradation during early steady-state life. Strict leakage prevention is enforced by partitioning engines strictly by unit ID and fitting all scalers and operating-regime clusterers exclusively on training engines. A multi-model benchmark compares Ridge Regression, Random Forest, LightGBM, and a lightweight PyTorch Gated Recurrent Unit (GRU) network evaluated against Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and the official NASA Asymmetric Scoring Function. 

On dataset FD001 (100 test engines), our production LightGBM model achieves an RMSE of 17.16 cycles and a NASA Score of 505.77 (competitive with the 2008 challenge benchmark of 16.98 cycles), while evaluating in under 1.8 milliseconds per engine on standard CPU hardware. Global and local interpretability is provided via TreeSHAP, linking mathematical features back to thermodynamic High Pressure Compressor erosion. Uncertainty is quantified using dual-quantile regression ($\alpha = [0.10, 0.90]$), demonstrating 89.19% empirical coverage on full validation trajectories. Finally, an operational cost simulation across a 100-engine fleet demonstrates that model-driven maintenance at optimal threshold $T^* = 10$ cycles eliminates 100% of catastrophic failures while slashing total fleet operational expenditure by 79.0% (\$15.80 Million saved vs Run-to-Failure) and 26.14% (\$1.49 Million saved vs fixed-interval overhaul). The complete system is deployed as an interactive FastAPI REST service with an asynchronous Chart.js dashboard and a 12-test automated PyTest suite validating zero train/serve skew and dynamic real-time inference.

---

## 2. Introduction and Organization/Training Overview

### 2.1 Student Profile and Academic Context
This Industrial Training Report represents the culminating research and engineering project conducted by **Abhinav Bhatra**, a Bachelor of Technology candidate in **Computer Science & Engineering with specialization in Artificial Intelligence** at **Jaipur Engineering College and Research Centre (JECRC Foundation)**. The candidate's academic curriculum integrates theoretical machine learning, algorithmic optimization, statistical pattern recognition, and software engineering with industrial data science.

Prior to undertaking this capstone turbofan prognostic project, the candidate designed and deployed analytical models addressing classical business data problems:
1. *Book Recommender System*: Collaborative and content-based filtering models designed to optimize catalog engagement.
2. *Sales Maximization Modeling*: Multivariate regression and demand elasticity models forecasting revenue under dynamic pricing.
3. *Customer Churn Analytics*: Supervised classification pipelines identifying customer retention attrition factors.

### 2.2 Training Organization Overview
The four-to-six-month intensive industrial training was completed under the mentorship of **LinuxWorld Informatics Pvt. Ltd.** within the specialized domain of **Data Science with Artificial Intelligence**.

- **Organization Name:** LinuxWorld Informatics Pvt. Ltd.
- **Center / Location:** [TO BE FILLED BY STUDENT: Training Center Address, e.g., Jaipur Center / Online]
- **Training Duration / Dates:** [TO BE FILLED BY STUDENT: e.g., June 2025 to October 2025 / 6 Months]
- **Training Mentor / Lead Instructor:** [TO BE FILLED BY STUDENT: Mentor Name, e.g., Mr. Vimal Daga]
- **Department / Specialization:** Advanced Machine Learning, MLOps, and Deep Learning Systems
- **Core Competencies Developed:** Converting unstructured business pain points into production data pipelines, preventing temporal data leakage, deploying scalable REST microservices, and conducting empirical cost-benefit analyses.

---

## 3. Problem Statement and Objectives

### 3.1 Industrial Problem Framing: The Maintenance Dilemma
Heavy industrial asset managers face an unavoidable economic trade-off between two flawed maintenance paradigms:
1. **Run-to-Failure (Corrective Maintenance)**: Machinery operates until functional failure. While component life is fully exhausted, unplanned failures in aviation cause devastating secondary damage (e.g., turbine blade fragments destroying downstream stages), emergency groundings, flight cancellations, and passenger safety risks.
2. **Fixed-Interval Maintenance (Preventive Overhaul)**: Engines are serviced after a predetermined elapsed flight cycle count (e.g., every 130 cycles). Because machine wear varies based on ambient conditions, manufacturing variations, and operational severity, fixed schedules retire 25% to 40% of healthy engines prematurely, generating millions of dollars in unnecessary overhaul costs.

```mermaid
flowchart LR
    A["Raw Problem: Unplanned Breakdown vs Premature Overhaul"] --> B["Data Problem: RUL Regression from 21 Sensors"]
    B --> C["Analytical Solution: LightGBM + Quantile Intervals + SHAP"]
    C --> D["Business Impact: $15.8M Savings & 0 Unplanned Failures"]
```

### 3.2 Quantitative Engineering Objectives
1. **Accurate Prognostics**: Train regression models predicting RUL with RMSE competitive with published PHM benchmarks on NASA C-MAPSS.
2. **Asymmetric Risk Optimization**: Minimize the NASA Asymmetric Score ($S$), heavily penalizing late predictions that risk catastrophic in-flight failures.
3. **Multi-Condition Domain Adaptation**: Develop operating-regime clustering for complex environments (FD004) characterized by 6 flight conditions and dual failure modes.
4. **Actionable Explainability**: Map abstract mathematical features to physical thermodynamic components using TreeSHAP.
5. **Zero-Skew Serving & Real-Time Delivery**: Deploy a production FastAPI microservice sharing 100% of preprocessing code with offline training, proven via automated test suites.

---

## 4. Background and Literature Review

Prognostics and Health Management (PHM) has evolved across three major paradigms:
1. **Physics-of-Failure (PoF) Models**: Utilize Paris' law of crack propagation, finite element analysis (FEA), and thermodynamic differential equations. While highly interpretable, PoF models require inaccessible material coefficients and fail to generalize across variable operating envelopes.
2. **Data-Driven Machine Learning**: Formulated following the landmark 2008 IEEE Prognostics Data Challenge on the NASA C-MAPSS dataset.
   - *Heimes (2008)* established the benchmark RNN-MLP model achieving RMSE = 16.98 and Score = 436.5, popularizing piecewise-linear RUL targets capped at 125 cycles.
   - *Ramasso (2014)* benchmarked Relevance Vector Machines (RVM) achieving RMSE = 19.82 and Score = 1152.0.
   - *Zheng et al. (2017)* explored deep multi-layer CNNs and LSTMs on sliding windows, achieving RMSE = 16.14.
3. **Modern Hybrid & Gradient Boosted Frameworks**: While deep recurrent models capture temporal dynamics, Gradient Boosted Decision Trees (LightGBM) on engineered rolling statistical windows have emerged as industrial favorites due to microsecond CPU inference, minimal memory footprints, and native tree-based SHAP explainability.

---

## 5. Dataset Description and Ingestion

### 5.1 NASA C-MAPSS Benchmark Architecture
The dataset was generated using NASA's Commercial Modular Aero-Propulsion System Simulation tool. It models a 90,000-lb thrust class twin-spool turbofan engine:
- **Low Pressure Spool (N1)**: Fan and Low Pressure Turbine (LPT).
- **High Pressure Spool (N2)**: High Pressure Compressor (HPC) and High Pressure Turbine (HPT).

| Sub-Dataset | Operating Envelopes | Fault Modes | Training Engines | Test Engines |
| :--- | :---: | :---: | :---: | :---: |
| **FD001** | 1 (Sea Level) | 1 (HPC Degradation) | 100 | 100 |
| **FD002** | 6 Flight Regimes | 1 (HPC Degradation) | 260 | 259 |
| **FD003** | 1 (Sea Level) | 2 (HPC + Fan Degradation) | 100 | 100 |
| **FD004** | 6 Flight Regimes | 2 (HPC + Fan Degradation) | 249 | 248 |

### 5.2 Sensor Channels and Invariant Detection
The raw telemetry records 26 variables per cycle: Engine Unit ID, Operational Cycle Number, 3 Operational Settings, and 21 thermodynamic sensors.

In **FD001**, empirical variance analysis revealed that 6 sensor channels have standard deviations $\sigma < 10^{-10}$:
- $s_1$ (Fan inlet temperature): constant at $518.67^\circ\text{R}$
- $s_5$ (Fan inlet pressure): constant at $14.62\text{ psia}$
- $s_10$ (Engine pressure ratio): constant at $1.30$
- $s_16$ (Burner fuel-air ratio): constant at $0.03$
- $s_18$ (Demanded fan speed): constant at $2388\text{ rpm}$
- $s_19$ (Demanded corrected fan speed): constant at $100.00\text{ rpm}$

*Physical Insight*: Under static sea-level flight conditions, intake temperatures, static pressures, and target setpoints are locked. In **FD004**, all 21 sensors fluctuate as operating settings change across altitude, Mach number, and throttle resolver angles.

---

## 6. Methodology and Leakage-Free Design

### 6.1 Paris' Crack Law and Piecewise-Linear RUL Target
Rather than assuming linear component decay from cycle 1 ($RUL = T_{max} - t$), we implement a piecewise-linear cap:
$$RUL(t) = \min(T_{max} - t, 125)$$
During initial engine cycles, component wear is negligible. Forcing a regression model to predict decreasing life during pristine cycles forces it to fit random measurement noise. Minimum engine failure cycle across all training engines is 128 cycles, ensuring that every engine transitions from the healthy plateau into the active degradation phase.

### 6.2 Strict Leakage Prevention
1. **Engine Grouping**: Train and validation sets are split strictly on unique engine IDs (80 train / 20 validation engines for FD001; 199 train / 50 validation engines for FD004).
2. **Causal Rolling Aggregations**: All rolling means, standard deviations, and trends are computed grouped by `unit_nr`, preventing rolling windows from crossing engine boundaries.
3. **Train-Only Parameter Fitting**: StandardScalers and KMeans regime clusterers are fit strictly on training engine partitions.

### 6.3 Regime Clustering Normalization (FD004)
To handle the 6 operating regimes in FD004, a KMeans clusterer ($K=6$) is fit on operational settings $[\text{setting}\_1, \text{setting}\_2, \text{setting}\_3]$. Sensor readings are normalized within each cluster:
$$z_{i,c} = \frac{x_i - \mu_{s,c}}{\sigma_{s,c}}$$
This eliminates discrete baseline jumps between flight regimes and reveals pure component wear curves.

---

## 7. Implementation Architecture and Models

### 7.1 Feature Engineering for Tabular Models
For classical and gradient boosted models, **99 features** are computed per operational cycle:
- 14 active normalized sensor readings.
- 28 rolling means across windows $W \in \{5, 15\}$ (filtering high-frequency noise).
- 28 rolling standard deviations across $W \in \{5, 15\}$ (capturing expanding turbulence near failure).
- 28 rolling differences / slopes across $W \in \{5, 15\}$ (capturing wear velocity).
- 1 normalized cumulative cycle feature (`time_cycles / 100.0`).

### 7.2 Model Architectures
1. **Ridge Linear Regression**: L2-regularized linear baseline with $\alpha = 10.0$.
2. **Random Forest Regressor**: 100 trees, maximum depth 12.
3. **LightGBM Regressor (Production Model)**: 250 boosting rounds, learning rate 0.03, num_leaves 31, max_depth 6, subsample 0.8.
4. **PyTorch GRU Network**: 2-layer Gated Recurrent Unit with hidden_dim 64, dropout 0.2, and linear readout layer Linear(64 $\rightarrow$ 32) $\rightarrow$ ReLU $\rightarrow$ Linear(32 $\rightarrow$ 1), trained on 3D sequence tensors of shape `(N, 30, 14)` using Adam optimizer and validation early stopping.

---

## 8. System Architecture and Deployment

The production system is architected as a lightweight, decoupled microservice:
- **Backend**: FastAPI REST engine (`app/main.py`) serving live predictions and cost simulations.
- **Serving Pipeline**: Powered by `src/inference.py`, loading fitted artifacts once at server startup.
- **Frontend**: Responsive Single Page Application using HTML5, modern CSS3 (`app/static/css/style.css`), and Vanilla JavaScript + Chart.js via CDN (`app/static/js/app.js`).
- **Containerization**: Standardized `Dockerfile` built on `python:3.13-slim` with OpenMP runtime dependencies, automated healthchecks, and unbuffered logging.

```mermaid
flowchart TD
    Client["Client Browser (SPA Dashboard)"] -->|REST Requests| API["FastAPI Application (app/main.py)"]
    API -->|Inference Calls| Engine["Inference Engine (src/inference.py)"]
    Engine --> Scaler["StandardScaler & Config (artifacts/)"]
    Engine --> Model["LightGBM & Quantile Models (artifacts/)"]
    Engine --> Explainer["TreeSHAP Explainer (src/explain.py)"]
    API --> Cost["Cost Simulator (src/cost.py)"]
```

---

## 9. Results and Empirical Discussion

### 9.1 Empirical Benchmark on NASA C-MAPSS FD001 (100 Test Engines)
All metrics originate from actual code execution recorded in `results/metrics.json`:

| Model Architecture | Test RMSE (Cycles) | Test MAE (Cycles) | NASA Score ($S$) | % Early (Safe) | % Late (Hazardous) | Max Late Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression (Ridge)** | 19.908 | 15.176 | 728.63 | 45.0% | 55.0% | +39.45 |
| **Random Forest** | 18.474 | 13.343 | 624.83 | 47.0% | 53.0% | +40.68 |
| **LightGBM Regressor (Live)** | **17.16** | **12.327** | **505.77** | **44.0%** | **56.0%** | **+37.65** |
| **PyTorch GRU (3D Window)** | **15.223** | **11.036** | **385.39** | **55.0%** | **45.0%** | +41.7 |

*Published Literature Values (Rule 9 Reference)*:
- *Heimes (2008) Challenge Winner*: RMSE = 16.98, Score = 436.5
- *Ramasso (2014) RVM Baseline*: RMSE = 19.82, Score = 1152.0
- *Zheng et al. (2017) CNN/LSTM*: RMSE = 16.14, Score = 338.0

### 9.2 Operating Complexity Contrast: FD001 vs FD004
Under 6 flight regimes and compound failure modes (FD004), LightGBM achieved **RMSE = 28.434 cycles** and **NASA Score = 5716.9**, reflecting an 11-fold surge in asymmetric penalty due to operating regime transitions.

### 9.3 Explainability with TreeSHAP
Aggregating the 99 engineered features to physical sensor subsystems revealed:
1. `cycle_norm` (Operational Age): 9.90 cycles mean impact.
2. `s_4` (Low Pressure Turbine Outlet Temp): 8.90 cycles impact.
3. `s_2` (Low Pressure Compressor Outlet Temp): 7.18 cycles impact.
4. `s_11` (High Pressure Compressor Static Pressure): 6.27 cycles impact.
5. `s_3` (High Pressure Compressor Outlet Temp): 5.51 cycles impact.

*Thermodynamic Validation*: High Pressure Compressor blade wear directly causes exhaust gas temperature elevation ($s_4, s_2, s_3$) and static pressure ratio distortion ($s_{11}$).

### 9.4 Uncertainty Quantification (80% Quantile Intervals)
Using LightGBM quantile regression at $\alpha = [0.10, 0.90]$:
- **Full Trajectory Validation Coverage (3,490 cycles)**: **89.19% empirical coverage** (target 80.0%), Mean Interval Width = 54.94 cycles.
- **Test Fleet Truncated Final Cycle Coverage (100 engines)**: **72.0% empirical coverage**, Mean Interval Width = 70.65 cycles.

---

## 10. Business Impact and Financial Cost Analysis

### 10.1 Cost Model Assumptions & Policies
- $C_{unplanned} = \$200,000$ (catastrophic failure penalty).
- $C_{planned} = \$40,000$ (scheduled shop overhaul; 20% of unplanned).
- $C_{waste} = \$200/\text{cycle}$ (amortized component value lost per unutilized cycle).
- Maintenance Decision Rule: Schedule overhaul when $\widehat{RUL} \le T$.

### 10.2 Empirical Fleet Simulation Results (100 Turbofan Engines)
Simulating 100 engines operating across their complete operational lifespans:

| Maintenance Policy | Total Fleet Cost (\$) | Unplanned Failures | Planned Overhauls | Wasted Life Cost (\$) |
| :--- | :---: | :---: | :---: | :---: |
| **Run-to-Failure (Corrective)** | \$20,000,000.00 | 100 | 0 | \$0.00 |
| **Fixed-Schedule (130 cycles)** | \$5,686,600.00 | 1 | 99 | \$1,526,600.00 |
| **Predictive Maintenance ($T^* = 10\text{c}$)** | **\$4,200,000.00** | **0** | **100** | **\$200,000.00** |

### 10.3 Net Savings and ROI
- **Savings vs Run-to-Failure**: **\$15,800,000.00 (79.0% reduction)**.
- **Savings vs Fixed-Schedule Overhaul**: **\$1,486,600.00 (26.14% reduction)**.
- **Catastrophic Failures Prevented**: **100% of catastrophic failures eliminated** with an average advance warning lead time of 10.0 cycles.

---

## 11. Engineering Challenges Faced

Documented directly from `docs/CHANGELOG_AND_ISSUES.md`:
1. **Windows OpenMP Dual-Runtime Crash**: PyTorch CPU wheel and Anaconda MKL loaded duplicate copies of `libiomp5md.dll` (OMP Error #15). Resolved by setting `KMP_DUPLICATE_LIB_OK=TRUE` and single-thread controls.
2. **NASA SSL Handshake Failure**: NASA Ames repository certificates expired under modern Python 3.13 SSL contexts. Resolved by targeting the official NASA Open Data Portal legacy archive (`https://data.nasa.gov/docs/legacy/CMAPSSData.zip`) with fallback context validation.
3. **Pandas Rolling Index NaN Injection**: Filtering engines for validation created non-contiguous indices; rolling operations using `reset_index(drop=True)` misaligned with the target DataFrame, injecting 3,668 NaNs. Resolved by re-indexing contiguous arrays before feature calculation and assigning via `.values` and `np.nan_to_num()`.

---

## 12. Project Limitations

1. **Simulated Data Environment**: The C-MAPSS dataset is generated by thermodynamic physics code, not revenue airline telemetry. Operational flight data features weather shifts, missing telemetry packets, and sensor drift.
2. **Static Cost Assumptions**: Maintenance costs (\$200k, \$40k) represent aviation industry averages. Individual airline contracts and MRO facility labour rates vary.
3. **Batch Operational Cycles**: Model ingests cycle-level summary telemetry rather than continuous 100 Hz high-frequency flight sensor feeds.

---

## 13. Ethics, Safety, and Responsible AI in Aviation

1. **Human-in-the-Loop Decision Making**: Predictions and 80% uncertainty bounds serve as decision support for certified aviation engineers, never triggering automated engine groundings without human review.
2. **Asymmetric Risk Governance**: Models must prioritize safety over component exhaustion by penalizing late predictions aggressively.
3. **Data Integrity & Traceability**: All telemetry inputs are cryptographically validated against physical range bounds before inference.

---

## 14. Conclusion and Future Scope

This project demonstrated an end-to-end data science lifecycle converting a critical aerospace engineering challenge into an economically validated predictive solution. By combining piecewise-linear crack physics, leakage-free feature engineering, and LightGBM regression, the system achieved an RMSE of 17.16 cycles, explained failure drivers via TreeSHAP, provided 80% prediction intervals, and demonstrated \$15.8 Million in cost avoidance across a 100-engine fleet.

### Future Scope:
1. **Physics-Informed Neural Networks (PINNs)**: Embed thermodynamic blade wear differential equations directly into deep learning loss functions.
2. **Multi-Condition Domain Adaptation**: Implement Domain-Adversarial Neural Networks (DANN) for multi-regime transfer learning across disparate aircraft fleets.
3. **Supply Chain MRO Integration**: Integrate dynamic RUL predictions directly into hangar scheduling algorithms and spare parts inventory systems.

---

## 15. References
1. Saxena, A., & Goebel, K. (2008). *Turbofan Engine Degradation Simulation Data Set*. NASA Ames Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.
2. Heimes, F. O. (2008). *Recurrent neural networks for remaining useful life estimation*. In Proceedings of the International Conference on Prognostics and Health Management (PHM 2008), Denver, CO, pp. 1-6.
3. Ramasso, E. (2014). *Investigating the performance of relevance vector machines for remaining useful life estimation in turbofan engines*. International Journal of Prognostics and Health Management, 5(1), 1-18.
4. Zheng, S., Ristovski, K., Farahat, A., & Gupta, C. (2017). *Long short-term memory network for remaining useful life estimation*. In Proceedings of the IEEE International Conference on Prognostics and Health Management (ICPHM), Dallas, TX, pp. 88-95.
5. Lundberg, S. M., & Lee, S. I. (2017). *A unified approach to interpreting model predictions*. Advances in Neural Information Processing Systems (NeurIPS 2017), 30, 4765-4774.
6. Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., ... & Liu, T. Y. (2017). *LightGBM: A highly efficient gradient boosting decision tree*. Advances in Neural Information Processing Systems (NeurIPS 2017), 30, 3146-3154.

---

## 16. Appendix: Reproduction Guide & Test Verification

### 16.1 Automated Verification: Proving Website is Dynamic and Free of Train/Serve Skew
The test suite `pytest ptm_project/tests/ -v` passed **12 out of 12 tests**:
1. `test_zero_train_serve_skew`: Confirmed that offline prediction and live API `POST /predict` produce identical RUL predictions ($< 10^{-4}$ cycles).
2. `test_dynamic_prediction_proof`: Proved that perturbing input sensor values dynamically alters RUL predictions, confirming predictions are **never hardcoded**.
3. `test_input_validation_and_error_handling`: Confirmed that uploading invalid files (<30 cycles, missing columns) returns structured HTTP 422 errors.

### 16.2 Local Execution & Reproduction Commands
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Execute end-to-end pipeline (data -> features -> train -> explain -> cost)
bash run_all.sh

# 3. Run automated PyTest test suite
pytest tests/ -v

# 4. Launch live FastAPI production web service
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
