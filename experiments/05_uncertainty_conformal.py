"""
05_uncertainty_conformal.py - Calibrated Uncertainty & Conformal Prediction

Evaluates:
1. Baseline Pinball Quantile Regression (alpha=0.10, 0.90 for nominal 80% interval).
2. Interval Coverage (PICP) and Interval Width (MPIW) on Validation vs Test Sets.
3. Rigorous analysis of coverage deterioration under test censoring.
4. Split Conformal Prediction (Residual & Conformalized Quantile Regression - CQR).
   Calibrates prediction intervals using the validation set to achieve guaranteed coverage on test.

Outputs:
- results/paper/05_uncertainty_conformal.json
- results/paper/05_uncertainty_conformal.csv
- Appends to paper/number_provenance.csv
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from lightgbm import LGBMRegressor

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from experiments.provenance_tracker import log_number

RESULTS_DIR = PROJECT_ROOT / "results" / "paper"
PROC_DIR = PROJECT_ROOT / "data" / "processed"
ART_DIR = PROJECT_ROOT / "artifacts"


def compute_interval_metrics(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> dict:
    """Computes Prediction Interval Coverage Probability (PICP) and Mean Prediction Interval Width (MPIW)."""
    y_true = np.asarray(y_true)
    lower = np.asarray(lower)
    upper = np.asarray(upper)

    covered = (y_true >= lower) & (y_true <= upper)
    picp = float(np.mean(covered) * 100.0)
    width = float(np.mean(upper - lower))

    return {
        "picp_percent": round(picp, 2),
        "mpiw_cycles": round(width, 2),
        "total_samples": len(y_true),
        "covered_samples": int(np.sum(covered))
    }


def run_uncertainty_experiments():
    print("=" * 70)
    print("EXPERIMENT 5: Calibrated Uncertainty & Conformal Prediction")
    print("=" * 70)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Load tabular features
    with open(ART_DIR / "FD001_feature_config.json") as f:
        cfg = json.load(f)
    feature_cols = cfg["feature_cols"]

    train_feat = pd.read_parquet(PROC_DIR / "FD001_train_features.parquet")
    val_feat = pd.read_parquet(PROC_DIR / "FD001_val_features.parquet")
    test_feat = pd.read_parquet(PROC_DIR / "FD001_test_features.parquet")

    X_train = train_feat[feature_cols].values
    y_train = train_feat["RUL"].values

    X_val = val_feat[feature_cols].values
    y_val = val_feat["RUL"].values

    test_last = test_feat.groupby("unit_nr").last().reset_index()
    X_test_last = test_last[feature_cols].values
    y_test_true = np.load(PROC_DIR / "FD001_y_test_true.npy")

    # Load point prediction model
    point_model = joblib.load(ART_DIR / "FD001_best_model.pkl")
    y_pred_val_point = np.clip(point_model.predict(X_val), 0, 125)
    y_pred_test_point = np.clip(point_model.predict(X_test_last), 0, 125)

    # -------------------------------------------------------------
    # 2. Train Quantile Regression Models (alpha=0.10 and alpha=0.90)
    # -------------------------------------------------------------
    print("\n--- 1. Training Pinball Quantile Regression Models ---")
    q_low_model = LGBMRegressor(
        objective="quantile", alpha=0.10, n_estimators=200,
        learning_rate=0.05, num_leaves=31, random_state=42, n_jobs=-1, verbose=-1
    )
    q_high_model = LGBMRegressor(
        objective="quantile", alpha=0.90, n_estimators=200,
        learning_rate=0.05, num_leaves=31, random_state=42, n_jobs=-1, verbose=-1
    )

    q_low_model.fit(X_train, y_train)
    q_high_model.fit(X_train, y_train)

    # Predictions on Validation Set
    val_lower_raw = np.clip(q_low_model.predict(X_val), 0, 125)
    val_upper_raw = np.clip(q_high_model.predict(X_val), 0, 125)
    # Ensure lower <= upper
    val_lower = np.minimum(val_lower_raw, val_upper_raw)
    val_upper = np.maximum(val_lower_raw, val_upper_raw)

    # Predictions on Test Set (Final Observed Cycle)
    test_lower_raw = np.clip(q_low_model.predict(X_test_last), 0, 125)
    test_upper_raw = np.clip(q_high_model.predict(X_test_last), 0, 125)
    test_lower = np.minimum(test_lower_raw, test_upper_raw)
    test_upper = np.maximum(test_lower_raw, test_upper_raw)

    # Evaluate uncalibrated quantile intervals
    val_uncal = compute_interval_metrics(y_val, val_lower, val_upper)
    test_uncal = compute_interval_metrics(y_test_true, test_lower, test_upper)

    print(f"Validation Uncalibrated Quantiles (Nominal 80%): PICP = {val_uncal['picp_percent']}% | MPIW = {val_uncal['mpiw_cycles']} cycles")
    print(f"Test Set Uncalibrated Quantiles   (Nominal 80%): PICP = {test_uncal['picp_percent']}% | MPIW = {test_uncal['mpiw_cycles']} cycles")
    print(f"Coverage Gap (Validation -> Test): {val_uncal['picp_percent'] - test_uncal['picp_percent']:.1f}% drop under test censoring.")

    # -------------------------------------------------------------
    # 3. Split Conformal Prediction Calibration
    # -------------------------------------------------------------
    print("\n--- 2. Split Conformal Prediction Calibration ---")
    alpha_target = 0.20  # For 80% target coverage (1 - alpha = 0.80)
    n_val = len(y_val)

    # Method A: Residual Split Conformal (Constant-width conformal band)
    val_residuals = np.abs(y_val - y_pred_val_point)
    k_res = int(np.ceil((n_val + 1) * (1.0 - alpha_target)))
    q_res = float(np.sort(val_residuals)[min(k_res - 1, n_val - 1)])

    test_lower_conf_res = np.clip(y_pred_test_point - q_res, 0, 125)
    test_upper_conf_res = np.clip(y_pred_test_point + q_res, 0, 125)
    test_conf_res_metrics = compute_interval_metrics(y_test_true, test_lower_conf_res, test_upper_conf_res)
    print(f"Residual Conformal Band (q={q_res:.1f}c): Test PICP = {test_conf_res_metrics['picp_percent']}% | MPIW = {test_conf_res_metrics['mpiw_cycles']} cycles")

    # Method B: Conformalized Quantile Regression (CQR, Romano et al. 2019)
    # Nonconformity score: s_i = max(lower_i - y_i, y_i - upper_i)
    cqr_scores = np.maximum(val_lower - y_val, y_val - val_upper)
    k_cqr = int(np.ceil((n_val + 1) * (1.0 - alpha_target)))
    q_cqr = float(np.sort(cqr_scores)[min(k_cqr - 1, n_val - 1)])

    test_lower_cqr = np.clip(test_lower - q_cqr, 0, 125)
    test_upper_cqr = np.clip(test_upper + q_cqr, 0, 125)
    test_cqr_metrics = compute_interval_metrics(y_test_true, test_lower_cqr, test_upper_cqr)
    print(f"CQR Calibrated Quantiles (q={q_cqr:.1f}c): Test PICP = {test_cqr_metrics['picp_percent']}% | MPIW = {test_cqr_metrics['mpiw_cycles']} cycles")

    # Also evaluate 90% Nominal Conformal Band (alpha = 0.10)
    alpha_90 = 0.10
    k_90 = int(np.ceil((n_val + 1) * (1.0 - alpha_90)))
    q_cqr_90 = float(np.sort(cqr_scores)[min(k_90 - 1, n_val - 1)])
    test_lower_cqr_90 = np.clip(test_lower - q_cqr_90, 0, 125)
    test_upper_cqr_90 = np.clip(test_upper + q_cqr_90, 0, 125)
    test_cqr_90_metrics = compute_interval_metrics(y_test_true, test_lower_cqr_90, test_upper_cqr_90)
    print(f"CQR Calibrated (Nominal 90%, q={q_cqr_90:.1f}c): Test PICP = {test_cqr_90_metrics['picp_percent']}% | MPIW = {test_cqr_90_metrics['mpiw_cycles']} cycles")

    # Compile Summary Table
    results_records = [
        {
            "method": "Uncalibrated Quantiles (Val)",
            "split": "Validation",
            "nominal_coverage_percent": 80.0,
            "empirical_picp_percent": val_uncal["picp_percent"],
            "mpiw_cycles": val_uncal["mpiw_cycles"],
            "calibration_adjustment_q": 0.0,
            "notes": "Raw pinball loss on full degradation engines"
        },
        {
            "method": "Uncalibrated Quantiles (Test)",
            "split": "Test",
            "nominal_coverage_percent": 80.0,
            "empirical_picp_percent": test_uncal["picp_percent"],
            "mpiw_cycles": test_uncal["mpiw_cycles"],
            "calibration_adjustment_q": 0.0,
            "notes": "Suffers undercoverage due to test censoring shift"
        },
        {
            "method": "Split Conformal Residual Band",
            "split": "Test",
            "nominal_coverage_percent": 80.0,
            "empirical_picp_percent": test_conf_res_metrics["picp_percent"],
            "mpiw_cycles": test_conf_res_metrics["mpiw_cycles"],
            "calibration_adjustment_q": round(q_res, 2),
            "notes": "Constant width conformal band around point prediction"
        },
        {
            "method": "CQR Calibrated Quantiles (80%)",
            "split": "Test",
            "nominal_coverage_percent": 80.0,
            "empirical_picp_percent": test_cqr_metrics["picp_percent"],
            "mpiw_cycles": test_cqr_metrics["mpiw_cycles"],
            "calibration_adjustment_q": round(q_cqr, 2),
            "notes": "Adaptive width conformalized quantile regression"
        },
        {
            "method": "CQR Calibrated Quantiles (90%)",
            "split": "Test",
            "nominal_coverage_percent": 90.0,
            "empirical_picp_percent": test_cqr_90_metrics["picp_percent"],
            "mpiw_cycles": test_cqr_90_metrics["mpiw_cycles"],
            "calibration_adjustment_q": round(q_cqr_90, 2),
            "notes": "Adaptive width calibrated to 90% confidence"
        }
    ]

    df_out = pd.DataFrame(results_records)
    csv_path = RESULTS_DIR / "05_uncertainty_conformal.csv"
    df_out.to_csv(csv_path, index=False)

    json_path = RESULTS_DIR / "05_uncertainty_conformal.json"
    with open(json_path, "w") as f:
        json.dump(results_records, f, indent=2)

    # Log into provenance registry
    log_number("Uncertainty_Val_Uncal_PICP", "FD001", "LightGBM_Quantile", "PICP", f"{val_uncal['picp_percent']:.1f}", "%", "experiments/05_uncertainty_conformal.py", "results/paper/05_uncertainty_conformal.json", "Validation uncalibrated coverage")
    log_number("Uncertainty_Val_Uncal_MPIW", "FD001", "LightGBM_Quantile", "MPIW", f"{val_uncal['mpiw_cycles']:.1f}", "cycles", "experiments/05_uncertainty_conformal.py", "results/paper/05_uncertainty_conformal.json", "Validation uncalibrated width")
    log_number("Uncertainty_Test_Uncal_PICP", "FD001", "LightGBM_Quantile", "PICP", f"{test_uncal['picp_percent']:.1f}", "%", "experiments/05_uncertainty_conformal.py", "results/paper/05_uncertainty_conformal.json", "Test uncalibrated coverage (coverage drop)")
    log_number("Uncertainty_Test_Uncal_MPIW", "FD001", "LightGBM_Quantile", "MPIW", f"{test_uncal['mpiw_cycles']:.1f}", "cycles", "experiments/05_uncertainty_conformal.py", "results/paper/05_uncertainty_conformal.json", "Test uncalibrated width")
    log_number("Uncertainty_Test_CQR80_PICP", "FD001", "CQR_LightGBM", "PICP", f"{test_cqr_metrics['picp_percent']:.1f}", "%", "experiments/05_uncertainty_conformal.py", "results/paper/05_uncertainty_conformal.json", "Test calibrated CQR 80% coverage")
    log_number("Uncertainty_Test_CQR80_MPIW", "FD001", "CQR_LightGBM", "MPIW", f"{test_cqr_metrics['mpiw_cycles']:.1f}", "cycles", "experiments/05_uncertainty_conformal.py", "results/paper/05_uncertainty_conformal.json", "Test calibrated CQR 80% width")
    log_number("Uncertainty_Test_CQR90_PICP", "FD001", "CQR_LightGBM", "PICP", f"{test_cqr_90_metrics['picp_percent']:.1f}", "%", "experiments/05_uncertainty_conformal.py", "results/paper/05_uncertainty_conformal.json", "Test calibrated CQR 90% coverage")
    log_number("Uncertainty_Test_CQR90_MPIW", "FD001", "CQR_LightGBM", "MPIW", f"{test_cqr_90_metrics['mpiw_cycles']:.1f}", "cycles", "experiments/05_uncertainty_conformal.py", "results/paper/05_uncertainty_conformal.json", "Test calibrated CQR 90% width")

    print(f"\n[SAVED] Uncertainty outputs saved to {csv_path.name} and {json_path.name}")


if __name__ == "__main__":
    run_uncertainty_experiments()
