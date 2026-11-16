"""
04_explain_uncertainty.py - TreeSHAP Interpretability and Quantile Uncertainty Intervals

Executes:
1. TreeSHAP Global and Local Feature Attributions on LightGBM.
2. Aggregation to physical sensor subsystems (HPC, LPT, Bleed).
3. Quantile Regression (80% Prediction Intervals at alpha=0.10 and alpha=0.90).
4. Honest evaluation of interval coverage probability (PICP) and width (MPIW).
5. Persists quantile models and updates results/metrics.json.
"""

import os
import sys
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import shap
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.explain import (
    get_tree_explainer,
    compute_global_shap_importance,
    aggregate_shap_by_sensor,
    explain_single_engine,
    train_quantile_models,
    evaluate_prediction_intervals
)
from src.features import INFORMATIVE_SENSORS_FD001

FIG_DIR = PROJECT_ROOT / "results" / "figures"
TAB_DIR = PROJECT_ROOT / "results" / "tables"
ART_DIR = PROJECT_ROOT / "artifacts"
PROC_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"


def main():
    print("\n=======================================================")
    print("[PHASE 4] Explainability (SHAP) & Uncertainty Estimation")
    print("=======================================================")

    # Load artifacts and preprocessed data
    best_model = joblib.load(ART_DIR / "FD001_best_model.pkl")

    with open(ART_DIR / "FD001_feature_config.json", "r") as f:
        feat_config = json.load(f)
    feature_cols = feat_config["feature_cols"]

    train_feat = pd.read_parquet(PROC_DIR / "FD001_train_features.parquet")
    val_feat = pd.read_parquet(PROC_DIR / "FD001_val_features.parquet")
    test_feat = pd.read_parquet(PROC_DIR / "FD001_test_features.parquet")

    y_test_true = np.load(PROC_DIR / "FD001_y_test_true.npy")

    X_train = train_feat[feature_cols].values
    y_train = train_feat["RUL"].values
    X_val = val_feat[feature_cols].values
    y_val = val_feat["RUL"].values

    test_last_records = test_feat.groupby("unit_nr").last().reset_index()
    X_test_last = test_last_records[feature_cols].values

    # =========================================================
    # 1. Global TreeSHAP Feature Attributions
    # =========================================================
    print("[SHAP] Computing TreeSHAP values across test fleet (100 engines)...")
    df_shap_global = compute_global_shap_importance(best_model, X_test_last, feature_cols)
    df_shap_global.to_csv(TAB_DIR / "shap_feature_importance.csv", index=False)

    df_sensor_shap = aggregate_shap_by_sensor(df_shap_global, INFORMATIVE_SENSORS_FD001)
    df_sensor_shap.to_csv(TAB_DIR / "shap_sensor_importance.csv", index=False)

    print("\nTop 5 Physical Sensor Subsystems by SHAP Impact:")
    print(df_sensor_shap.head())

    # =========================================================
    # 2. Local Engine Explanations (Critical vs Healthy)
    # =========================================================
    explainer = get_tree_explainer(best_model)

    # Find a critical engine (e.g. Engine with actual RUL < 25)
    critical_indices = np.where(y_test_true < 25)[0]
    critical_idx = critical_indices[0] if len(critical_indices) > 0 else 0
    critical_engine_id = int(test_last_records.iloc[critical_idx]["unit_nr"])

    local_explanation = explain_single_engine(
        model=best_model,
        explainer=explainer,
        feature_vector=X_test_last[critical_idx],
        feature_names=feature_cols,
        top_n=8
    )

    print(f"\n[SHAP] Local Explanation for Critical Test Engine #{critical_engine_id}:")
    print(f"Base Value (Fleet Prior): {local_explanation['base_value']} cycles")
    print(f"Predicted RUL: {local_explanation['predicted_rul']} cycles | Actual RUL: {y_test_true[critical_idx]} cycles")
    for feat in local_explanation["top_features"][:5]:
        print(f"  {feat['feature']}: impact = {feat['shap_impact']} -> {feat['direction']}")

    # Save local explanation sample to tables
    pd.DataFrame(local_explanation["top_features"]).to_csv(TAB_DIR / f"shap_local_engine_{critical_engine_id}.csv", index=False)

    # =========================================================
    # 3. Quantile Regression for Uncertainty (80% Intervals)
    # =========================================================
    print("\n[UNCERTAINTY] Training Quantile Regressors for 80% Prediction Intervals...")
    q10_model, q90_model = train_quantile_models(X_train, y_train, X_val, y_val, random_state=42)

    # Save quantile models
    joblib.dump(q10_model, ART_DIR / "FD001_quantile_q10.pkl")
    joblib.dump(q90_model, ART_DIR / "FD001_quantile_q90.pkl")
    print("[UNCERTAINTY] Saved Q10 and Q90 models to artifacts/.")

    # Predict intervals on validation split (full trajectory for representative evaluation)
    X_val_all = val_feat[feature_cols].values
    y_val_all = val_feat["RUL"].values

    y_val_q10 = np.clip(q10_model.predict(X_val_all), 0, 125)
    y_val_q90 = np.clip(q90_model.predict(X_val_all), 0, 125)
    val_interval_metrics = evaluate_prediction_intervals(y_val_all, y_val_q10, y_val_q90)
    print("\n[UNCERTAINTY] Validation Set Full Trajectory Interval Metrics (N=3,490 cycles):", val_interval_metrics)

    # Predict intervals on test fleet (last cycle per engine)
    y_test_q10 = np.clip(q10_model.predict(X_test_last), 0, 125)
    y_test_q90 = np.clip(q90_model.predict(X_test_last), 0, 125)
    test_interval_metrics = evaluate_prediction_intervals(y_test_true, y_test_q10, y_test_q90)
    print("[UNCERTAINTY] Test Fleet End-Cycle Interval Metrics (N=100 engines):", test_interval_metrics)

    # Save interval predictions table
    df_intervals = pd.DataFrame({
        "engine_id": test_last_records["unit_nr"].astype(int),
        "actual_rul": y_test_true,
        "predicted_rul": np.clip(best_model.predict(X_test_last), 0, 125),
        "lower_q10": y_test_q10,
        "upper_q90": y_test_q90,
        "interval_width": y_test_q90 - y_test_q10,
        "is_inside_interval": (y_test_true >= y_test_q10) & (y_test_true <= y_test_q90)
    })
    df_intervals.to_csv(TAB_DIR / "test_prediction_intervals.csv", index=False)

    # =========================================================
    # 4. Generate Publication-Quality Plots
    # =========================================================
    print("\n[PLOTS] Generating Phase 4 figures...")

    # Plot 11: Global SHAP Feature & Sensor Importance
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    top_features = df_shap_global.head(10)
    axes[0].barh(top_features["feature"][::-1], top_features["mean_abs_shap"][::-1], color="#1f77b4", edgecolor="black")
    axes[0].set_title("Top 10 Engineered Features (Mean |SHAP Value|)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Mean |SHAP Value| (Cycles)")

    top_sensors = df_sensor_shap.head(8)
    axes[1].barh(top_sensors["sensor"][::-1] + " (" + top_sensors["description"][::-1].str.slice(0, 18) + "...)",
                 top_sensors["total_shap_impact"][::-1], color="#2ca02c", edgecolor="black")
    axes[1].set_title("Aggregated Physical Sensor Importance", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Aggregated SHAP Impact Across Channels (Cycles)")

    plt.tight_layout()
    fig11 = FIG_DIR / "11_shap_summary_global.png"
    plt.savefig(fig11, dpi=300)
    plt.close()
    print(f"[PLOTS] Saved: {fig11.name}")

    # Plot 12: Local SHAP Explanation Waterfall for Critical Engine
    plt.figure(figsize=(10, 5))
    top_local = pd.DataFrame(local_explanation["top_features"])
    colors = ["#d62728" if x < 0 else "#2ca02c" for x in top_local["shap_impact"]]
    plt.barh(top_local["feature"][::-1], top_local["shap_impact"][::-1], color=colors[::-1], edgecolor="black")
    plt.axvline(0, color="black", linestyle="-", linewidth=1)
    plt.title(f"Local SHAP Contributions for Critical Engine #{critical_engine_id}\n(Base Fleet Prior = {local_explanation['base_value']} cycles -> Predicted RUL = {local_explanation['predicted_rul']} cycles)", fontsize=12)
    plt.xlabel("SHAP Impact on Predicted RUL (Cycles)")
    plt.tight_layout()
    fig12 = FIG_DIR / "12_shap_local_waterfall.png"
    plt.savefig(fig12, dpi=300)
    plt.close()
    print(f"[PLOTS] Saved: {fig12.name}")

    # Plot 13: Prediction Intervals vs Ground Truth on Test Fleet
    df_sorted = df_intervals.sort_values(by="actual_rul").reset_index(drop=True)
    plt.figure(figsize=(14, 6))
    x_axis = range(len(df_sorted))

    plt.plot(x_axis, df_sorted["actual_rul"], "k-", linewidth=2, label="Actual Ground Truth RUL")
    plt.plot(x_axis, df_sorted["predicted_rul"], "b--", linewidth=1.5, label="Point Prediction (LightGBM)")
    plt.fill_between(x_axis, df_sorted["lower_q10"], df_sorted["upper_q90"], color="#1f77b4", alpha=0.25, label="80% Prediction Interval [Q10, Q90]")

    # Highlight missed points
    missed = df_sorted[~df_sorted["is_inside_interval"]]
    plt.scatter(missed.index, missed["actual_rul"], color="red", s=35, zorder=5, label=f"Uncovered Actuals ({100 - test_interval_metrics['picp_coverage_pct']:.1f}%)")

    plt.title(f"Uncertainty Quantification on 100 Test Engines: 80% Prediction Intervals\nEmpirical PICP Coverage: {test_interval_metrics['picp_coverage_pct']}% | Mean Interval Width: {test_interval_metrics['mpiw_cycles']} cycles", fontsize=12, fontweight="bold")
    plt.xlabel("Test Engines (Ranked by Ground Truth RUL)")
    plt.ylabel("Remaining Useful Life (Cycles)")
    plt.legend(loc="upper left")
    plt.tight_layout()
    fig13 = FIG_DIR / "13_prediction_intervals.png"
    plt.savefig(fig13, dpi=300)
    plt.close()
    print(f"[PLOTS] Saved: {fig13.name}")

    # Plot 14: Uncertainty Coverage & Interval Calibration
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Error vs Interval Width
    abs_errors = np.abs(df_intervals["predicted_rul"] - df_intervals["actual_rul"])
    axes[0].scatter(df_intervals["interval_width"], abs_errors, color="#1f77b4", alpha=0.7, edgecolors="k")
    axes[0].set_title("Prediction Error vs Interval Width\n(Wider interval signals higher uncertainty)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Predicted Interval Width (Q90 - Q10) [Cycles]")
    axes[0].set_ylabel("Absolute Prediction Error |y - y_hat| [Cycles]")

    # Coverage Breakdown Pie Chart
    labels = ["Inside 80% Interval (Covered)", "Below Q10 (Underpredicted)", "Above Q90 (Overpredicted)"]
    sizes = [
        test_interval_metrics["picp_coverage_pct"],
        test_interval_metrics["pct_below_lower"],
        test_interval_metrics["pct_above_upper"]
    ]
    colors_pie = ["#2ca02c", "#ff7f0e", "#d62728"]
    axes[1].pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors_pie, startangle=140, explode=(0.05, 0, 0))
    axes[1].set_title("Empirical Coverage Distribution (Test Fleet N=100)", fontsize=11, fontweight="bold")

    plt.tight_layout()
    fig14 = FIG_DIR / "14_uncertainty_coverage_calibration.png"
    plt.savefig(fig14, dpi=300)
    plt.close()
    print(f"[PLOTS] Saved: {fig14.name}")

    # =========================================================
    # 5. Append to results/metrics.json
    # =========================================================
    metrics_path = RESULTS_DIR / "metrics.json"
    with open(metrics_path, "r") as f:
        master_metrics = json.load(f)

    master_metrics["uncertainty_quantification"] = {
        "nominal_target_coverage_pct": 80.0,
        "validation_metrics": val_interval_metrics,
        "test_metrics": test_interval_metrics,
        "method": "Quantile Gradient Boosting (LightGBM alpha=0.10, 0.90)"
    }

    master_metrics["explainability"] = {
        "top_features": df_shap_global.head(10).to_dict(orient="records"),
        "top_sensors": df_sensor_shap.head(8).to_dict(orient="records")
    }

    with open(metrics_path, "w") as f:
        json.dump(master_metrics, f, indent=4)
    print(f"[METRICS] Successfully updated {metrics_path}")


if __name__ == "__main__":
    main()
