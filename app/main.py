"""
main.py - FastAPI Production Web Application for Turbofan RUL Prediction & Decision Support

Features:
- Sub-millisecond live inference serving the real saved LightGBM and Quantile models.
- Shared preprocessing pipeline from src/features.py and src/inference.py (Zero Train/Serve Skew).
- Real-time endpoints for fleet monitoring, engine deep dive, file upload, cycle replay, and cost simulation.
- Single Page Application (SPA) frontend served via FastAPI static mounting & Jinja2.
"""

import os
import sys
import json
import io
import pandas as pd
from src.data import ALL_COLS
from pathlib import Path
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Query
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

# Ensure OpenMP fix
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.inference import TurbofanInferenceEngine
from src.cost import TurbofanCostSimulator, calculate_cost_savings

app = FastAPI(
    title="NASA C-MAPSS Turbofan RUL & Decision Support System",
    description="Production Predictive Maintenance Web Application for Jet Engine Health Monitoring",
    version="1.0.0"
)

# Mount static assets and Jinja2 templates
app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT / "app" / "static")), name="static")
templates = Jinja2Templates(directory=str(PROJECT_ROOT / "app" / "templates"))

# Global singletons
INFERENCE_ENGINE: Optional[TurbofanInferenceEngine] = None
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"


@app.on_event("startup")
def startup_event():
    """Initializes the production inference engine at server boot."""
    global INFERENCE_ENGINE
    print("[STARTUP] Booting FastAPI Turbofan Prognostics Service...")
    INFERENCE_ENGINE = TurbofanInferenceEngine(
        artifacts_dir=PROJECT_ROOT / "artifacts",
        data_dir=DATA_DIR
    )
    print("[STARTUP] Service ready for live inference.")


# =====================================================================
# Request / Response Schemas
# =====================================================================

class CostSimulationRequest(BaseModel):
    cost_unplanned: float = Field(200000.0, description="Cost of unplanned catastrophic breakdown in USD")
    cost_planned: float = Field(40000.0, description="Cost of planned depot shop overhaul in USD")
    cost_waste_per_cycle: float = Field(200.0, description="Value of discarded component life per cycle in USD")
    threshold: float = Field(15.0, description="Predictive maintenance decision trigger threshold in cycles")


# =====================================================================
# API Endpoints
# =====================================================================

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    """Renders the main industrial training project dashboard."""
    return templates.TemplateResponse(name = "index.html", request=request, context={"title": "Turbofan RUL Decision Support Dashboard"})


@app.get("/health")
def health_check():
    """Health check verifying model loading and runtime readiness."""
    return {
        "status": "healthy",
        "service": "Turbofan Predictive Maintenance Decision Support",
        "model_loaded": INFERENCE_ENGINE is not None,
        "runtime": f"Python {sys.version.split()[0]} (FastAPI + LightGBM)",
        "dataset_notice": "High-fidelity simulated data (NASA Glenn C-MAPSS)"
    }


@app.get("/fleet")
def get_fleet_overview():
    """
    Returns real-time point RUL predictions, 80% confidence intervals,
    and operational statuses for all test fleet engines.
    """
    if INFERENCE_ENGINE is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized.")
    try:
        fleet_data = INFERENCE_ENGINE.get_fleet_summary()
        critical_count = sum(1 for e in fleet_data if e["status"] == "CRITICAL")
        warning_count = sum(1 for e in fleet_data if e["status"] == "WARNING")
        healthy_count = sum(1 for e in fleet_data if e["status"] == "HEALTHY")
        avg_rul = round(sum(e["predicted_rul"] for e in fleet_data) / len(fleet_data), 1)

        return {
            "summary": {
                "total_engines": len(fleet_data),
                "critical": critical_count,
                "warning": warning_count,
                "healthy": healthy_count,
                "avg_fleet_rul": avg_rul
            },
            "fleet": fleet_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fleet inference failed: {str(e)}")


@app.get("/engine/{engine_id}")
def get_engine_detail(engine_id: int):
    """
    Returns complete sensor history, latest point prediction, 80% interval,
    and dynamic local TreeSHAP feature attributions for a selected engine.
    """
    if INFERENCE_ENGINE is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized.")
    try:
        test_df = INFERENCE_ENGINE.get_test_fleet_data()
        engine_df = test_df[test_df["unit_nr"] == engine_id]
        if engine_df.empty:
            raise HTTPException(status_code=404, detail=f"Engine #{engine_id} not found in fleet.")

        result = INFERENCE_ENGINE.predict_engine(engine_df)
        result["engine_id"] = engine_id
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Engine #{engine_id} prediction failed: {str(e)}")


@app.post("/predict")
async def predict_custom_engine(file: UploadFile = File(...)):
    """
    Accepts an uploaded CSV file containing an engine's operational history.
    Validates schema and minimum cycles, executes zero-skew preprocessing,
    and runs real model inference at request time.
    """
    if INFERENCE_ENGINE is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized.")

    if not file.filename.endswith((".csv", ".txt")):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a .csv or .txt file.")

    try:
        contents = await file.read()
        # Handle comma or whitespace delimited uploads
        try:
            df = pd.read_csv(io.BytesIO(contents))
            if len(df.columns) <= 2:
                df = pd.read_csv(io.BytesIO(contents), sep=r"\s+", header=None)
                if len(df.columns) == 26:
                    df.columns = ALL_COLS
        except Exception:
            df = pd.read_csv(io.BytesIO(contents), sep=r"\s+", header=None)
            if len(df.columns) == 26:
                df.columns = ALL_COLS

        # Execute prediction pipeline with validation
        result = INFERENCE_ENGINE.predict_engine(df)
        result["filename"] = file.filename
        return result

    except ValueError as ve:
        raise HTTPException(status_code=422, detail=f"Validation Error: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error on uploaded file: {str(e)}")


@app.get("/simulate/{engine_id}")
def simulate_engine(engine_id: int, cycle: Optional[int] = Query(None)):
    """
    Steps through an engine's history cycle-by-cycle, returning live RUL and sensor readings.
    """
    if INFERENCE_ENGINE is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized.")
    try:
        return INFERENCE_ENGINE.simulate_engine_step(engine_id, target_cycle=cycle)
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/cost")
def compute_cost_analysis(req: CostSimulationRequest):
    """
    Dynamic financial cost simulator: accepts editable cost parameters and maintenance threshold T,
    re-running cost simulation models against real empirical predictions.
    """
    try:
        simulator = TurbofanCostSimulator(
            cost_unplanned=req.cost_unplanned,
            cost_planned=req.cost_planned,
            cost_waste_per_cycle=req.cost_waste_per_cycle,
            default_threshold=req.threshold
        )

        # 1. Run-to-Failure (100 engines)
        rtf = simulator.evaluate_run_to_failure(num_engines=100)

        # 2. Optimal Fixed-Schedule (from training lifespans)
        proc_dir = DATA_DIR / "processed"
        train_feat = pd.read_parquet(proc_dir / "FD001_train_features.parquet")
        val_feat = pd.read_parquet(proc_dir / "FD001_val_features.parquet")
        all_trajectories = pd.concat([train_feat, val_feat], ignore_index=True)
        all_lifetimes = all_trajectories.groupby("unit_nr")["time_cycles"].max().values

        fixed_res = simulator.evaluate_fixed_schedule(all_lifetimes, interval_cycles=130)

        # 3. Model-Driven Predictive Maintenance Lifecycle
        with open(PROJECT_ROOT / "artifacts" / "FD001_feature_config.json") as f:
            feat_cols = json.load(f)["feature_cols"]

        pdm = simulator.simulate_full_lifecycle(
            df_trajectories=val_feat,
            model=INFERENCE_ENGINE.model,
            feature_cols=feat_cols,
            threshold=req.threshold
        )
        scale_100 = 100.0 / pdm["num_engines"]
        pdm_scaled = {
            "policy": pdm["policy"],
            "threshold": req.threshold,
            "planned_overhauls": int(pdm["planned_overhauls"] * scale_100),
            "unplanned_failures": int(pdm["unplanned_failures"] * scale_100),
            "pct_unplanned_prevented": pdm["pct_unplanned_prevented"],
            "avg_wasted_cycles": pdm["avg_wasted_cycles_per_engine"],
            "avg_lead_time": pdm["avg_advance_warning_lead_time"],
            "planned_cost": round(pdm["planned_cost"] * scale_100, 2),
            "unplanned_cost": round(pdm["unplanned_cost"] * scale_100, 2),
            "wasted_life_cost": round(pdm["wasted_life_cost"] * scale_100, 2),
            "total_cost": round(pdm["total_cost"] * scale_100, 2),
            "avg_cost_per_engine": round((pdm["total_cost"] * scale_100) / 100.0, 2)
        }

        # Savings
        savings = calculate_cost_savings(
            rtf_cost=rtf["total_cost"],
            fixed_cost=fixed_res["total_cost"],
            pdm_cost=pdm_scaled["total_cost"]
        )

        return {
            "inputs": {
                "cost_unplanned": req.cost_unplanned,
                "cost_planned": req.cost_planned,
                "cost_waste_per_cycle": req.cost_waste_per_cycle,
                "threshold": req.threshold
            },
            "run_to_failure": rtf,
            "fixed_schedule": fixed_res,
            "predictive_maintenance": pdm_scaled,
            "savings": savings
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cost simulation failed: {str(e)}")


@app.get("/metrics")
def get_canonical_metrics():
    """Reads and returns the canonical results/metrics.json file directly."""
    metrics_path = RESULTS_DIR / "metrics.json"
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="metrics.json not found.")
    with open(metrics_path, "r") as f:
        data = json.load(f)
    return data


@app.get("/sample-csv")
def download_sample_csv():
    """Serves the sample engine CSV file for user testing."""
    sample_path = DATA_DIR / "sample_engine.csv"
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail="sample_engine.csv not found.")
    return FileResponse(
        path=str(sample_path),
        filename="sample_engine_31.csv",
        media_type="text/csv"
    )
