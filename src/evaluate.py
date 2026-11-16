"""
evaluate.py - Prognostics Evaluation Metrics & NASA Asymmetric Scoring Function

Includes:
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- NASA Asymmetric Scoring Function (PHM 2008 Data Challenge Benchmark)

Physical and Operational Justification of NASA Score:
-----------------------------------------------------
In turbofan maintenance, the cost and risk of overestimating vs underestimating
Remaining Useful Life are radically asymmetric:
1. Early Prediction (d = y_pred - y_true < 0):
   The model underestimates remaining life. The engine is flagged for maintenance slightly
   earlier than strictly necessary. Consequence: Minor loss of usable component lifespan
   (wasteful overhaul), but safety is preserved.
2. Late Prediction (d = y_pred - y_true > 0):
   The model overestimates remaining life. The engine continues operating when it should have
   been serviced. Consequence: Severe catastrophic in-flight failure, severe secondary compressor
   damage, emergency shutdown, passenger hazard, or forced landing.

The NASA PHM 2008 scoring function captures this asymmetry using an exponential penalty:
   d = y_pred - y_true
   s_i = exp(-d / 13) - 1   if d < 0  (Early prediction, gradual penalty)
   s_i = exp( d / 10) - 1   if d >= 0 (Late prediction, aggressive exponential penalty)
   Score S = sum(s_i)
"""

import numpy as np
import pandas as pd
from typing import Dict, Union, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Root Mean Squared Error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Mean Absolute Error."""
    return float(mean_absolute_error(y_true, y_pred))


def compute_nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes the official NASA Ames PHM 2008 asymmetric prognostics scoring metric.

    Parameters:
    -----------
    y_true : np.ndarray
        Ground truth RUL values.
    y_pred : np.ndarray
        Predicted RUL values.

    Returns:
    --------
    total_score : float
        Sum of asymmetric exponential penalties across all evaluation engines.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    diff = y_pred - y_true

    scores = np.where(
        diff < 0,
        np.exp(-diff / 13.0) - 1.0,
        np.exp(diff / 10.0) - 1.0
    )
    return float(np.sum(scores))


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes comprehensive prognostics performance suite: RMSE, MAE, and NASA Score.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    rmse = compute_rmse(y_true, y_pred)
    mae = compute_mae(y_true, y_pred)
    nasa_score = compute_nasa_score(y_true, y_pred)

    diff = y_pred - y_true
    pct_early = float(np.mean(diff < 0) * 100.0)
    pct_late = float(np.mean(diff > 0) * 100.0)
    max_late_error = float(np.max(diff[diff > 0])) if np.any(diff > 0) else 0.0
    max_early_error = float(np.min(diff[diff < 0])) if np.any(diff < 0) else 0.0

    return {
        "rmse": round(rmse, 3),
        "mae": round(mae, 3),
        "nasa_score": round(nasa_score, 2),
        "pct_early": round(pct_early, 1),
        "pct_late": round(pct_late, 1),
        "max_late_cycles": round(max_late_error, 2),
        "max_early_cycles": round(max_early_error, 2)
    }


if __name__ == "__main__":
    # Test asymmetry demonstration
    y_true_mock = np.array([50.0, 50.0])
    y_pred_mock = np.array([40.0, 60.0])  # One 10 cycles early, one 10 cycles late
    print("Test Asymmetric Penalties for 10-cycle error:")
    print("10 cycles early penalty:", np.exp(10 / 13) - 1)
    print("10 cycles late penalty: ", np.exp(10 / 10) - 1)
    print("Metrics output:", evaluate_predictions(y_true_mock, y_pred_mock))
