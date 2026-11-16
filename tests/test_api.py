"""
test_api.py - PyTest Test Suite for FastAPI Endpoints, Zero Train/Serve Skew, and Dynamic Inference Proofs
"""

import os
import sys
import io
import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from app.main import app, startup_event
from src.inference import TurbofanInferenceEngine


@pytest.fixture(scope="module")
def client():
    """Initializes FastAPI test client and triggers server startup event."""
    startup_event()
    with TestClient(app) as test_client:
        yield test_client


def test_api_health_endpoint(client):
    """Verifies that the API reports healthy status and model readiness."""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "C-MAPSS" in data["dataset_notice"]


def test_api_fleet_endpoint(client):
    """Verifies that the /fleet endpoint returns all 100 test engines with valid predictions."""
    res = client.get("/fleet")
    assert res.status_code == 200
    data = res.json()
    assert data["summary"]["total_engines"] == 100
    assert len(data["fleet"]) == 100

    first_eng = data["fleet"][0]
    assert "engine_id" in first_eng
    assert "predicted_rul" in first_eng
    assert "status" in first_eng
    assert first_eng["status"] in ["HEALTHY", "WARNING", "CRITICAL"]


def test_api_engine_detail_endpoint(client):
    """Verifies that /engine/{id} returns sensor history and TreeSHAP attributions."""
    res = client.get("/engine/1")
    assert res.status_code == 200
    data = res.json()
    assert data["engine_id"] == 1
    assert "shap_explanation" in data
    assert len(data["shap_explanation"]["top_features"]) > 0
    assert "history" in data
    assert len(data["history"]["cycles"]) > 0


def test_zero_train_serve_skew(client):
    """
    CRITICAL PROOF: Verifies zero train/serve skew by comparing
    API POST /predict output against offline inference on the identical CSV input.
    Both must produce identical predictions to 4 decimal places.
    """
    sample_path = Path(__file__).resolve().parent.parent / "data" / "sample_engine.csv"
    assert sample_path.exists(), "Sample engine CSV missing"

    sample_df = pd.read_csv(sample_path)

    # 1. Compute offline prediction
    engine = TurbofanInferenceEngine()
    offline_res = engine.predict_engine(sample_df)

    # 2. Compute API prediction via HTTP POST
    with open(sample_path, "rb") as f:
        response = client.post("/predict", files={"file": ("sample_engine.csv", f, "text/csv")})

    assert response.status_code == 200
    api_res = response.json()

    # Assert exact numerical equivalence
    assert abs(api_res["predicted_rul"] - offline_res["predicted_rul"]) < 1e-4, \
        f"Train/Serve Skew Detected! Offline: {offline_res['predicted_rul']}, API: {api_res['predicted_rul']}"
    assert api_res["status"] == offline_res["status"]
    assert api_res["prediction_interval"] == offline_res["prediction_interval"]


def test_dynamic_prediction_proof(client):
    """
    CRITICAL PROOF: Proves that predictions are computed dynamically at request time
    and NOT hardcoded. Perturbing input sensor values MUST yield a different RUL prediction.
    """
    sample_path = Path(__file__).resolve().parent.parent / "data" / "sample_engine.csv"
    original_df = pd.read_csv(sample_path)

    # Base prediction
    csv_bytes_orig = io.BytesIO()
    original_df.to_csv(csv_bytes_orig, index=False)
    csv_bytes_orig.seek(0)
    res_orig = client.post("/predict", files={"file": ("orig.csv", csv_bytes_orig, "text/csv")})
    pred_orig = res_orig.json()["predicted_rul"]

    # Perturbed input: artificially simulate severe wear by raising temperatures by +10%
    perturbed_df = original_df.copy()
    perturbed_df["s_4"] = perturbed_df["s_4"] * 1.10
    perturbed_df["s_2"] = perturbed_df["s_2"] * 1.10

    csv_bytes_pert = io.BytesIO()
    perturbed_df.to_csv(csv_bytes_pert, index=False)
    csv_bytes_pert.seek(0)
    res_pert = client.post("/predict", files={"file": ("pert.csv", csv_bytes_pert, "text/csv")})
    pred_pert = res_pert.json()["predicted_rul"]

    assert pred_pert != pred_orig, "Model output is static! Predictions did not change under perturbed inputs."
    print(f"[TEST PROOF] Original RUL: {pred_orig} -> Perturbed RUL: {pred_pert} (Dynamic Inference Confirmed)")


def test_input_validation_and_error_handling(client):
    """Verifies that invalid uploads return clear human-readable error messages with HTTP 422."""
    # 1. Less than 30 cycles
    short_df = pd.DataFrame({
        "time_cycles": list(range(1, 15)),
        **{f"s_{i}": [500.0] * 14 for i in range(1, 22)}
    })
    buf = io.BytesIO()
    short_df.to_csv(buf, index=False)
    buf.seek(0)
    res = client.post("/predict", files={"file": ("short.csv", buf, "text/csv")})
    assert res.status_code == 422
    assert "minimum of 30 cycles" in res.json()["detail"].lower()

    # 2. Missing required columns
    bad_cols_df = pd.DataFrame({"time_cycles": list(range(1, 35)), "random_col": [1.0] * 34})
    buf2 = io.BytesIO()
    bad_cols_df.to_csv(buf2, index=False)
    buf2.seek(0)
    res2 = client.post("/predict", files={"file": ("bad_cols.csv", buf2, "text/csv")})
    assert res2.status_code == 422
    assert "missing required columns" in res2.json()["detail"].lower()


def test_cost_simulation_api(client):
    """Verifies that POST /cost dynamically computes financial savings."""
    payload = {
        "cost_unplanned": 250000.0,
        "cost_planned": 50000.0,
        "cost_waste_per_cycle": 250.0,
        "threshold": 15.0
    }
    res = client.post("/cost", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "savings" in data
    assert data["savings"]["savings_vs_run_to_failure"] > 0


def test_metrics_and_sample_csv_endpoints(client):
    """Verifies that /metrics and /sample-csv serve real files."""
    res_m = client.get("/metrics")
    assert res_m.status_code == 200
    assert "dataset_fd001" in res_m.json()

    res_csv = client.get("/sample-csv")
    assert res_csv.status_code == 200
    assert "text/csv" in res_csv.headers["content-type"]
    assert len(res_csv.content) > 100
