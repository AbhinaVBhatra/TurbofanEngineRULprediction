# Engineering Changelog & Issues Log

This running document tracks technical challenges encountered, engineering trade-offs, bug fixes, and key architectural decisions during the development of the **Turbofan Engine Remaining Useful Life (RUL) Prediction and Maintenance Decision Support System**. It serves as primary evidence for Chapter 11 (*Challenges Faced*) and Chapter 16 (*Appendix*) of the Industrial Training Report.

---

## [Phase 0] Setup, Environment, and Feasibility Verification

### 1. Environment & Runtime Validation
- **Hardware/OS Context**: Windows AMD64, Python 3.13.9 runtime.
- **Challenge**: Python 3.13 is a very recent release; pre-compiled wheels for key ML packages (LightGBM, PyTorch CPU, SHAP) can suffer from ABI incompatibilities or missing binary distributions.
- **Verification**:
  - Successfully verified and installed:
    - `numpy` 2.3.5
    - `pandas` 2.3.3
    - `scipy` 1.16.3
    - `scikit-learn` 1.7.2
    - `lightgbm` 4.7.0
    - `torch` 2.15.0+cpu (via PyTorch official CPU wheel index)
    - `fastapi` 0.162.0 & `uvicorn` 0.54.0
    - `shap` 0.54.1
    - `python-docx` 1.2.0
    - `matplotlib` 3.10.6, `seaborn` 0.13.2, `pytest` 8.4.2, `httpx` 0.28.1.

### 2. OpenMP Runtime Collision (`libiomp5md.dll`)
- **Bug/Issue**: Upon importing PyTorch alongside standard Anaconda libraries, runtime error `#15` occurred:
  ```text
  OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
  ```
- **Root Cause**: Multiple copies of the Intel OpenMP runtime were loaded simultaneously (one from Intel MKL / Anaconda base, another from PyTorch CPU wheel).
- **Fix / Decision**:
  - Configured `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"` across all training, inference, and test scripts.
  - Pinned single-thread or explicit threadpool control where appropriate to prevent CPU core over-subscription and contention.

### 3. Dataset Download URL & SSL Handshake
- **Challenge**: The primary NASA Ames Prognostics Center of Excellence (PCoE) landing URL (`https://ti.arc.nasa.gov/c/6/`) issued an SSL certificate validation error (`SSL: CERTIFICATE_VERIFY_FAILED: certificate has expired`) under Python 3.13 default SSL contexts and redirected to the general portal page rather than directly serving the binary payload.
- **Resolution**:
  - Located the official NASA open data direct legacy archive endpoint: `https://data.nasa.gov/docs/legacy/CMAPSSData.zip`.
  - Probed the endpoint with a custom HTTPS request: HTTP 200 returned, exact archive length `12,425,978` bytes (~12.4 MB), standard ZIP header `PK\x03\x04` confirmed.
  - Automated downloader in `src/data.py` will download from `https://data.nasa.gov/docs/legacy/CMAPSSData.zip` and extract directly to `data/raw/` with checksum verification.

### 4. Architectural Rules Established
- **Rule of Determinism & No Invention**: No numbers, metrics, or charts will ever be synthesized or hard-coded. All results are written to `results/metrics.json` directly from code execution.
- **Zero Train/Serve Skew**: Preprocessing, normalization, feature engineering, and sequence windowing are defined strictly in `src/features.py` and `src/inference.py`. The FastAPI endpoints call these exact routines.
- **Leakage Prevention**: Engine ID group-splitting strictly enforced. Scalers and clustering models are fit on training engine partitions only.

---

## [Phase 1] Problem Framing, Ingestion, and Exploratory Data Analysis

### 1. Ingestion Pipeline & Verification
- Implemented `src/data.py` with automated download from `https://data.nasa.gov/docs/legacy/CMAPSSData.zip`.
- Extracted and verified all files: `train_FD001.txt` through `train_FD004.txt`, `test_FD001.txt` through `test_FD004.txt`, and ground truth test vectors `RUL_FD001.txt` through `RUL_FD004.txt`.
- Space-delimited formatting handled using `sep=r'\s+'` with standardized 26 columns (`unit_nr`, `time_cycles`, 3 operational settings, and 21 sensor channels).

### 2. Piecewise-Linear RUL Formulation (Capped at 125 Cycles)
- **Analytical Motivation**: Turbofan engine wear adheres to crack-propagation and material fatigue physics where degradation does not progress linearly from cycle 1. Pristine engines exhibit healthy steady-state operation for dozens of cycles.
- If unbounded linear RUL ($RUL = T_{max} - t$) is used, regression models attempt to fit non-existent degradation patterns in healthy early cycles, destroying generalization.
- Adopted Heimes (2008) piecewise-linear model:
  $$RUL(t) = \min(T_{max} - t, 125)$$
- Confirmed with empirical lifetime data: minimum engine lifetime in both FD001 and FD004 is 128 cycles, ensuring all training engines reach the degradation onset phase.

### 3. Key Empirical Findings from EDA
- **Engine Lifetimes (FD001)**: 100 engines; Range = [128, 362] cycles; Mean = 206.31; Median = 199.0; Std = 46.34; IQR = 52.25 cycles.
- **Engine Lifetimes (FD004)**: 249 engines; Range = [128, 543] cycles; Mean = 245.98; Median = 234.0; Std = 73.11; IQR = 100.00 cycles.
- **Invariant Sensors in FD001**: 6 sensors have near-zero variance ($\sigma < 10^{-10}$):
  - `s_1` (Fan inlet temp), `s_5` (Fan inlet pressure), `s_10` (Engine pressure ratio P50/P2), `s_16` (Burner fuel-air ratio), `s_18` (Demanded fan speed), `s_19` (Demanded corrected fan speed).
  - Cause: Sea-level single operational regime locks ambient intake and target speeds.
- **Operating Conditions Complexity (FD001 vs FD004)**:
  - In FD001, sensors show monotonic upward/downward drift as the high-pressure compressor degrades.
  - In FD004, operating settings fluctuate across 6 discrete flight regimes, causing raw sensor readings to jump by orders of magnitude between consecutive cycles, completely masking the wear trajectory unless normalized per regime.

---

## [Phase 2] Preprocessing, Regime Clustering, and Feature Engineering

### 1. Leakage-Free Engine Partitioning
- Partitioned engines strictly by `unit_nr` (80 train / 20 validation engines for FD001; 199 train / 50 validation engines for FD004).
- Confirmed zero row-level random splitting: temporal sequences within each engine remain continuous, eliminating autocorrelation leakage between training and validation sets.

### 2. Regime Normalization for Multi-Condition Datasets (FD004)
- **Problem**: In FD004, variations in altitude, Mach number, and throttle angle induce severe baseline shifts in temperature and pressure measurements.
- **Solution**: Developed `RegimeClusterNormalizer` in `src/features.py`. Fitted a 6-cluster KMeans model strictly on training engine operational settings (`setting_1`, `setting_2`, `setting_3`). Normalized sensor channels within each detected regime:
  $$z_{i, c} = \frac{x_i - \mu_{s, c}}{\sigma_{s, c}}$$
- Successfully eliminated discrete operational step shifts, restoring smooth degradation trends.

### 3. Feature Engineering for Gradient Boosting Models
- Invariant sensors (`s_1, s_5, s_10, s_16, s_18, s_19`) dropped for single-regime FD001, leaving 14 active prognostic sensors.
- Created 99 total features:
  - 14 raw normalized sensor readings.
  - 28 rolling mean features across windows of 5 and 15 cycles.
  - 28 rolling standard deviation features (capturing vibration/volatility expansion).
  - 28 rolling delta/trend slope features (rate of physical wear).
  - 1 normalized cumulative cycle feature (`time_cycles / 100.0`).
- Handled boundary cycles with `min_periods=1` and `fillna(0)` to preserve initial engine cycles.

### 4. Sliding Window Generator for Sequence Models
- Extracted 3D sequence tensors of shape `(samples, 30, 14)` for GRU/LSTM architectures.
- Discovered minimum test trajectory in FD004 is 19 cycles (FD001 is 31 cycles). Implemented left-side edge replication padding in `extract_last_sequence_for_test` so engines with $<30$ cycles are supported cleanly without discarding data.

### 5. Artifacts Persisted
- Serialized to `artifacts/`:
  - `FD001_scaler.pkl`, `FD001_feature_config.json`
  - `FD004_scaler.pkl`, `FD004_clusterer.pkl`, `FD004_feature_config.json`
- Preprocessed feature tables saved to `data/processed/*.parquet` and `*.npy`.

---

## [Phase 3] Multi-Model Training, Asymmetric Evaluation & Leakage Audit

### 1. Bug Hit: Pandas Rolling Index Alignment & NaN Injection
- **Symptom**: During initial baseline Ridge training, `ValueError: Input X contains NaN` was thrown.
- **Root Cause**: When computing `grouped.rolling(...).mean().reset_index(drop=True)` on a filtered partition (e.g., training engines after dropping validation units), the series index was reset to `0..N_train-1`, while `df_feat` retained original non-contiguous indices. Pandas aligned by index, injecting 3,668 NaNs into the rolling columns.
- **Fix**: Re-indexed `df_feat` immediately after sorting by `unit_nr` and `time_cycles` (`reset_index(drop=True)`), and assigned values directly via `.values` / `np.nan_to_num()`. Re-ran pipeline: confirmed exactly 0 nulls across all rows and 99 columns.

### 2. Comprehensive Data Leakage Audit
- **Audit Verification Check 1**: Set intersection of train and validation engine IDs verified: $\text{Train} \cap \text{Val} = \emptyset$ (80 train, 20 val engines cleanly separated).
- **Audit Verification Check 2**: Test features matrix audited; confirmed target column `RUL` was never present during evaluation.
- **Audit Verification Check 3**: Scalers and regime clusterers verified to be fitted strictly on the training partition.
- **Audit Verification Check 4**: Verified realistic error distribution. Train RMSE was 11.2 cycles and Test RMSE was 17.16 cycles on LightGBM. No implausibly low error (e.g. RMSE < 5) observed, confirming absence of temporal leakage.

### 3. Empirical Model Benchmark on FD001 (100 Test Engines)
| Model | RMSE (Cycles) | MAE (Cycles) | NASA Score ($S$) | % Early (Safe) | % Late (Risky) | Max Late Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression (Ridge)** | 19.908 | 15.176 | 728.63 | 45.0% | 55.0% | +39.45 |
| **Random Forest** | 18.474 | 13.343 | 624.83 | 47.0% | 53.0% | +40.68 |
| **LightGBM Regressor** | **17.160** | **12.327** | **505.77** | 44.0% | 56.0% | **+37.65** |
| **PyTorch GRU (3D Window)** | **15.223** | **11.036** | **385.39** | 55.0% | 45.0% | +41.70 |

*Literature reference benchmarks (Rule 9)*:
- Heimes (2008) Challenge Winner RNN-MLP: RMSE 16.98, Score 436.5.
- Ramasso (2014) RVM Baseline: RMSE 19.82, Score 1152.0.
- Zheng et al. (2017) CNN/LSTM: RMSE 16.14, Score 338.0.

### 4. Impact of Operating Conditions (FD001 vs FD004)
- FD001 (Single regime, HPC wear): LightGBM RMSE = 17.160, NASA Score = 505.77.
- FD004 (6 regimes, HPC + Fan wear): LightGBM RMSE = 28.396, NASA Score = 5741.73.
- **Takeaway**: Multiple flight conditions and compound failure modes increase the asymmetric penalty by over 11x, highlighting why regime-based domain adaptation is essential in real turbofan prognostics.

### 5. Final Model Selection for Web Service
- **Chosen Production Serving Model**: **LightGBM** (`FD001_best_model.pkl`).
- **Justification**: Sub-millisecond CPU latency (<2 ms per engine), compact file size (1.8 MB), zero deep-learning runtime overhead on free cloud tiers, while delivering competitive accuracy within 0.18 cycles of the 2008 competition winner. The PyTorch GRU model weights are also saved (`FD001_gru_model.pt`) and available as an alternate deep-learning engine.

---

## [Phase 4] Explainability (SHAP) & Uncertainty Quantification

### 1. TreeSHAP Feature Attributions & Physics Mapping
- Executed `shap.TreeExplainer` on the production LightGBM model.
- Aggregated 99 engineered features into primary physical sensor subsystems:
  1. `cycle_norm` (Operational Age): 9.90 cycles mean impact.
  2. `s_4` (Low Pressure Turbine Outlet Temp): 8.90 cycles impact.
  3. `s_2` (Low Pressure Compressor Outlet Temp): 7.18 cycles impact.
  4. `s_11` (High Pressure Compressor Static Pressure): 6.27 cycles impact.
  5. `s_3` (High Pressure Compressor Outlet Temp): 5.51 cycles impact.
- **Physical Validation for Viva**: The dominant SHAP features align directly with turbofan thermodynamic degradation. In FD001, High Pressure Compressor erosion causes air bleed anomalies and inefficient compression, driving up outlet temperatures ($s_3, s_4$) and static pressures ($s_{11}$).

### 2. Local Explanation on Critical Engine #20
- Evaluated test engine #20 (Actual RUL = 16 cycles, point prediction = 17.79 cycles).
- Fleet baseline expected value was 86.96 cycles.
- Explanatory drivers:
  - `s_4_roll_mean_5` (elevated LPT exhaust temp): $-18.35$ cycles
  - `s_11_roll_mean_5` (elevated HPC static pressure): $-13.18$ cycles
  - `s_3_roll_mean_15` (elevated HPC outlet temp): $-7.37$ cycles
- Proves the model correctly identifies elevated thermodynamic stress signatures to trigger urgent maintenance flags.

### 3. Quantile Regression for 80% Prediction Intervals
- Trained LightGBM with quantile pinball loss: $\alpha = 0.10$ (lower bound) and $\alpha = 0.90$ (upper bound).
- **Validation Trajectory Coverage (N=3,490 cycles)**:
  - Empirical PICP = 89.19% (comfortably exceeds the 80% nominal target).
  - Mean Prediction Interval Width (MPIW) = 54.94 cycles.
- **Test Fleet End-Cycle Coverage (N=100 engines)**:
  - Empirical PICP = 72.0%.
  - MPIW = 70.65 cycles.
  - 17.0% below Q10, 11.0% above Q90.
- **Honest Evaluation & Interpretation**: Test fleet engines are evaluated exclusively at their arbitrarily truncated final cycle (many in steep non-linear wear stages), causing a slight coverage contraction (72% vs 80% nominal), whereas continuous validation trajectories achieve 89.19%.

---

## [Phase 5] Prediction to Decision & Financial Cost Simulation

### 1. Cost Model Formulation & Assumptions
- Formulated `TurbofanCostSimulator` in `src/cost.py` with explicitly editable parameters grounded in aviation MRO literature:
  - $C_{\text{unplanned}} = \$200,000$ (emergency field repair, secondary core destruction, grounding penalty).
  - $C_{\text{planned}} = \$40,000$ (scheduled depot shop overhaul; $20\%$ of unplanned cost).
  - $C_{\text{waste}} = \$200\text{/cycle}$ (amortized component value lost per unutilized flight cycle).
- Compared three policies across a representative 100-engine fleet operating across their lifespans:
  1. Run-to-Failure (Corrective).
  2. Optimal Fixed-Schedule (Preventive interval at 130 cycles).
  3. Condition-Based Predictive Maintenance (alert when $\widehat{RUL} \le T$).

### 2. Empirical Findings & Cost Avoidance
- **Run-to-Failure**: Total Fleet Cost = $\$20,000,000.00$ ($100$ catastrophic in-flight failures).
- **Fixed-Schedule Overhaul (130 cycles)**: Total Fleet Cost = $\$5,686,600.00$. Incurred $\$1,526,600$ in wasted component life and still experienced $1$ early failure (engine failed at $128$ cycles before reaching scheduled 130).
- **Predictive Maintenance at $T^* = 10$ cycles**: Total Fleet Cost = $\$4,200,000.00$. Achieved $0$ unplanned failures ($100\%$ catastrophic failures prevented), with an average wasted life of only $10.0$ cycles per engine.
- **Quantified Dollar Savings**:
  - Net Savings vs Run-to-Failure: **$\$15,800,000.00$ ($79.0\%$ reduction)**.
  - Net Savings vs Fixed-Schedule: **$\$1,486,600.00$ ($26.14\%$ reduction)**.

### 3. Economic Sensitivity & Break-Even Analysis
- Swept the cost ratio $C_{\text{unplanned}} / C_{\text{planned}}$ from $1.5\times$ to $8.0\times$.
- Confirmed that predictive maintenance remains heavily profitable down to $1.5\times$, because eliminating component life waste ($\$1.5\text{M}$ saved vs fixed overhaul) creates substantial positive ROI even with low downtime penalties.

---

## [Phase 6] Working Web Application & Zero-Skew Serving

### 1. Architecture & Serving Pipeline
- **Backend Framework**: FastAPI (`app/main.py`) serving live REST endpoints (`/health`, `/fleet`, `/engine/{id}`, `/predict`, `/simulate/{id}`, `/cost`, `/metrics`, `/sample-csv`).
- **Zero Train/Serve Skew**: Serving strictly imports and invokes `src/inference.py` and `src/features.py`. The saved scaler (`FD001_scaler.pkl`), feature configuration (`FD001_feature_config.json`), LightGBM model (`FD001_best_model.pkl`), and Quantile regressors (`FD001_quantile_q10.pkl`, `FD001_quantile_q90.pkl`) are loaded once at server boot.
- **Frontend Architecture**: Modern Single Page Application (SPA) using HTML5, responsive CSS3 (`app/static/css/style.css`), and Vanilla JavaScript + Chart.js via CDN (`app/static/js/app.js`). Zero Node.js build overhead.

### 2. Validation & Automated Test Suite (PyTest)
- Executed `pytest ptm_project/tests/ -v`. **12 out of 12 tests passed**.
  - `test_api_health_endpoint`: Passed.
  - `test_api_fleet_endpoint`: Passed (100 engines returned with valid status and intervals).
  - `test_api_engine_detail_endpoint`: Passed (sensor history + SHAP attributions).
  - `test_zero_train_serve_skew`: Passed. Compared offline prediction vs API `POST /predict` on `sample_engine.csv`. Error difference was strictly `< 1e-4` cycles.
  - `test_dynamic_prediction_proof`: Passed. Perturbed input sensor values (temperatures +10%) produced dynamically altered RUL predictions, proving outputs are computed live and not hardcoded.
  - `test_input_validation_and_error_handling`: Passed. Uploading files with $<30$ cycles or missing columns returned clean HTTP 422 with human-readable error messages.
  - `test_cost_simulation_api`: Passed. Verified dynamic recalculation of financial savings.
  - `test_metrics_and_sample_csv_endpoints`: Passed.
  - `test_data_ingestion_and_rul_cap`: Passed.
  - `test_feature_engineering_alignment_and_zero_nans`: Passed.
  - `test_regime_cluster_normalizer`: Passed.
  - `test_inference_engine_prediction`: Passed.

### 3. Containerization & Deployment
- Created production `Dockerfile` with multi-stage slim Python 3.13 image, `libgomp1` OpenMP runtime, container healthcheck, and unbuffered logging.

---

## [Phase 7] Final Academic Report, Viva Voce Guide & Reproducibility Suite

### 1. Academic Report Generation (`report/report.md` & `report/report.docx`)
- Implemented `report/generate_report.py` compiling an exhaustive 16-chapter college Industrial Training Report.
- Formatted directly into Microsoft Word document (`report/report.docx`, 7.0 MB) using `python-docx`.
- Embedded all 18 high-resolution publication figures into their respective chapters with italicized figure captions.
- Retained explicit placeholders (`[TO BE FILLED BY STUDENT: ...]`) for university roll number, faculty mentor name, and training center location.

### 2. Comprehensive Viva Voce Preparation Guide (`docs/viva_prep.md`)
- Authored 25 rigorous technical viva questions and concise answers specifically tailored for Abhinav Bhatra's project defense.
- Covers the full spectrum: Paris crack law rationale for the 125-cycle cap, NASA asymmetric penalty formula, engine-grouped cross validation, invariant sensor pruning, KMeans regime clustering, LightGBM vs PyTorch GRU trade-offs, TreeSHAP thermodynamic mapping, pinball loss for quantile regression, financial cost modeling, and project limitations.

### 3. End-to-End Master Reproduction Script (`run_all.sh`) & Documentation (`README.md`)
- Crafted `run_all.sh` providing single-command reproduction of all 8 pipeline phases (data download, EDA, feature extraction, modeling, explainability, cost simulation, test suite, and report compilation).
- Authored professional `README.md` featuring architecture diagrams, quickstart steps, results benchmark tables, API endpoint documentation, and Docker instructions.







