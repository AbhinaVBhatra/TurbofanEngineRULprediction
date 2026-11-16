"""
explain.py - Model Explainability (TreeSHAP) and Uncertainty Quantification (Quantile Intervals)

Components:
1. TreeSHAP Global & Local Attribution:
   - Quantifies individual sensor and rolling feature contributions to RUL prediction.
   - Aggregates engineered features back to physical sensor subsystems (e.g. HPC, LPT, Bleed).
   - Generates local feature contributions for individual engines in real-time for the web app.
2. Quantile Regression (80% Prediction Intervals):
   - Trains LightGBM models with pinball/quantile loss at alpha=0.10 and alpha=0.90.
   - Evaluates honest empirical coverage probability (PICP) and interval width (MPIW).
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import shap
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from lightgbm import LGBMRegressor

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    from src.data import SENSOR_DESCRIPTIONS
except ImportError:
    from ptm_project.src.data import SENSOR_DESCRIPTIONS


def get_tree_explainer(model: Any) -> shap.TreeExplainer:
    """Returns a TreeExplainer for the gradient boosting model."""
    return shap.TreeExplainer(model)


def compute_global_shap_importance(
    model: Any,
    X: np.ndarray,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Computes global mean absolute SHAP values across all samples.
    Returns a sorted DataFrame of feature importances.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    df_importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap
    }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)

    return df_importance


def aggregate_shap_by_sensor(
    df_feat_shap: pd.DataFrame,
    sensor_cols: List[str]
) -> pd.DataFrame:
    """
    Aggregates the 99 engineered features (raw, rolling mean, std, diff)
    into their parent physical sensor subsystems for clear engineering interpretability.
    """
    sensor_impacts = {}

    for s in sensor_cols:
        # Sum SHAP importance of raw sensor and all its rolling statistics
        matching_rows = df_feat_shap[df_feat_shap["feature"].str.startswith(s)]
        total_impact = float(matching_rows["mean_abs_shap"].sum())
        desc = SENSOR_DESCRIPTIONS.get(s, s)
        sensor_impacts[s] = {
            "sensor": s,
            "description": desc,
            "total_shap_impact": total_impact
        }

    # Add cycle norm if present
    cycle_row = df_feat_shap[df_feat_shap["feature"] == "cycle_norm"]
    if not cycle_row.empty:
        sensor_impacts["cycle_norm"] = {
            "sensor": "cycle_norm",
            "description": "Engine Cumulative Operating Age",
            "total_shap_impact": float(cycle_row["mean_abs_shap"].values[0])
        }

    df_sensors = pd.DataFrame(list(sensor_impacts.values()))
    df_sensors.sort_values(by="total_shap_impact", ascending=False, inplace=True)
    df_sensors.reset_index(drop=True, inplace=True)
    return df_sensors


def explain_single_engine(
    model: Any,
    explainer: shap.TreeExplainer,
    feature_vector: np.ndarray,
    feature_names: List[str],
    top_n: int = 6
) -> Dict[str, Any]:
    """
    Computes local SHAP explanation for a single engine cycle snapshot.
    Returns base value, predicted value, and top driving features with directions.
    Used by the FastAPI GET /engine/{id} endpoint at request time.
    """
    # Ensure 2D shape (1, num_features)
    if feature_vector.ndim == 1:
        feature_vector = feature_vector.reshape(1, -1)

    shap_vals = explainer.shap_values(feature_vector)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
    shap_vals_1d = shap_vals[0]

    base_value = float(explainer.expected_value) if not isinstance(explainer.expected_value, np.ndarray) else float(explainer.expected_value[0])
    pred_value = float(model.predict(feature_vector)[0])

    # Rank features by absolute impact
    indices = np.argsort(np.abs(shap_vals_1d))[::-1][:top_n]

    top_features = []
    for idx in indices:
        feat_name = feature_names[idx]
        val = float(feature_vector[0, idx])
        impact = float(shap_vals_1d[idx])
        top_features.append({
            "feature": feat_name,
            "value": round(val, 3),
            "shap_impact": round(impact, 3),
            "direction": "Decreases RUL (Degraded)" if impact < 0 else "Increases RUL (Healthy)"
        })

    return {
        "base_value": round(base_value, 2),
        "predicted_rul": round(pred_value, 2),
        "top_features": top_features
    }


# =====================================================================
# Quantile Regression for Uncertainty Intervals
# =====================================================================

def train_quantile_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    random_state: int = 42
) -> Tuple[LGBMRegressor, LGBMRegressor]:
    """
    Trains two LightGBM quantile regression models for uncertainty estimation:
    - Lower bound: alpha = 0.10 (10th percentile)
    - Upper bound: alpha = 0.90 (90th percentile)
    Together, they produce an 80% prediction interval [Q10, Q90].
    """
    print("[UNCERTAINTY] Training Lower Quantile Regressor (alpha=0.10)...")
    q10 = LGBMRegressor(
        objective="quantile",
        alpha=0.10,
        n_estimators=200,
        learning_rate=0.03,
        num_leaves=31,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1
    )
    q10.fit(X_train, y_train, eval_set=[(X_val, y_val)])

    print("[UNCERTAINTY] Training Upper Quantile Regressor (alpha=0.90)...")
    q90 = LGBMRegressor(
        objective="quantile",
        alpha=0.90,
        n_estimators=200,
        learning_rate=0.03,
        num_leaves=31,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1
    )
    q90.fit(X_train, y_train, eval_set=[(X_val, y_val)])

    return q10, q90


def evaluate_prediction_intervals(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray
) -> Dict[str, float]:
    """
    Evaluates prediction intervals honestly:
    - PICP (Prediction Interval Coverage Probability): fraction of ground-truth values falling inside [lower, upper]
    - MPIW (Mean Prediction Interval Width): average width of the interval
    - Below fraction & Above fraction
    """
    y_true = np.asarray(y_true)
    y_lower = np.asarray(y_lower)
    y_upper = np.asarray(y_upper)

    # Ensure lower <= upper
    y_lower_adj = np.minimum(y_lower, y_upper)
    y_upper_adj = np.maximum(y_lower, y_upper)

    inside = (y_true >= y_lower_adj) & (y_true <= y_upper_adj)
    picp = float(np.mean(inside) * 100.0)
    mpiw = float(np.mean(y_upper_adj - y_lower_adj))
    pct_below = float(np.mean(y_true < y_lower_adj) * 100.0)
    pct_above = float(np.mean(y_true > y_upper_adj) * 100.0)

    return {
        "picp_coverage_pct": round(picp, 2),
        "target_coverage_pct": 80.0,
        "mpiw_cycles": round(mpiw, 2),
        "pct_below_lower": round(pct_below, 2),
        "pct_above_upper": round(pct_above, 2)
    }
