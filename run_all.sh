#!/usr/bin/env bash
# ==============================================================================
# Master End-to-End Pipeline Reproduction Script
# Project: Turbofan Engine Remaining Useful Life (RUL) Prediction & Decision Support
# Candidate: Abhinav Bhatra (JECRC Foundation / LinuxWorld Informatics)
# ==============================================================================

set -e

export KMP_DUPLICATE_LIB_OK="TRUE"
export PYTHONPATH="."

echo "======================================================================"
echo "[STEP 1/8] Ingestion: Downloading & Verifying NASA C-MAPSS Data..."
echo "======================================================================"
python src/data.py

echo "======================================================================"
echo "[STEP 2/8] EDA: Running Exploratory Data Analysis & Baseline Profiling..."
echo "======================================================================"
python notebooks/01_eda.py

echo "======================================================================"
echo "[STEP 3/8] Features: Regime Clustering & Sliding Window Generation..."
echo "======================================================================"
python notebooks/02_features.py

echo "======================================================================"
echo "[STEP 4/8] Modeling: Benchmarking Ridge, RF, LightGBM, and PyTorch GRU..."
echo "======================================================================"
python notebooks/03_models.py

echo "======================================================================"
echo "[STEP 5/8] Explainability: TreeSHAP Attributions & 80% Quantile Intervals..."
echo "======================================================================"
python notebooks/04_explain_uncertainty.py

echo "======================================================================"
echo "[STEP 6/8] Economics: Maintenance Threshold Sweep & Cost Simulation..."
echo "======================================================================"
python notebooks/05_decision_cost.py

echo "======================================================================"
echo "[STEP 7/8] Quality Assurance: Running Automated PyTest Test Suite..."
echo "======================================================================"
pytest tests/ -v

echo "======================================================================"
echo "[STEP 8/8] Report Generation: Exporting report.md and report.docx..."
echo "======================================================================"
python report/generate_report.py

echo "======================================================================"
echo "SUCCESS! All 8 pipeline phases executed cleanly with zero errors."
echo "Launch live dashboard with: uvicorn app.main:app --host 0.0.0.0 --port 8000"
echo "======================================================================"
