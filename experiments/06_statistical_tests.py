"""
06_statistical_tests.py - Paired Bootstrap Hypothesis Testing & Confidence Intervals

Performs rigorous statistical significance testing for prognostic differences
between LightGBM and PyTorch GRU on NASA C-MAPSS FD001 (N=100 test engines).

Computes:
1. Observed paired differences: Delta_RMSE, Delta_MAE, Delta_NASA_Score.
2. 10,000 paired bootstrap resamples with replacement.
3. 95% Percentile Confidence Intervals: [2.5th, 97.5th percentiles].
4. Bootstrap empirical two-tailed p-values.
5. Wilcoxon Signed-Rank Test on paired absolute prediction errors.

Outputs:
- results/paper/06_statistical_tests.json
- results/paper/06_statistical_tests.csv
- Appends to paper/number_provenance.csv
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from src.models import PyTorchGRUTrainer
from src.evaluate import compute_rmse, compute_mae, compute_nasa_score
from experiments.provenance_tracker import log_number

RESULTS_DIR = PROJECT_ROOT / "results" / "paper"
PROC_DIR = PROJECT_ROOT / "data" / "processed"
ART_DIR = PROJECT_ROOT / "artifacts"


def run_statistical_tests():
    print("=" * 70)
    print("EXPERIMENT 6: Paired Bootstrap Hypothesis Testing & CIs")
    print("=" * 70)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Load test data and ground truth
    y_test_true = np.load(PROC_DIR / "FD001_y_test_true.npy")
    num_engines = len(y_test_true)

    # 2. Generate LightGBM test predictions
    with open(ART_DIR / "FD001_feature_config.json") as f:
        cfg = json.load(f)
    feature_cols = cfg["feature_cols"]

    test_feat = pd.read_parquet(PROC_DIR / "FD001_test_features.parquet")
    test_last = test_feat.groupby("unit_nr").last().reset_index()
    X_test_tab = test_last[feature_cols].values

    lgbm_model = joblib.load(ART_DIR / "FD001_best_model.pkl")
    preds_lgbm = np.clip(lgbm_model.predict(X_test_tab), 0, 125)

    # 3. Generate PyTorch GRU test predictions
    X_test_seq = np.load(PROC_DIR / "FD001_X_test_seq.npy")
    gru_trainer = PyTorchGRUTrainer(input_dim=14, hidden_dim=64, num_layers=2)
    gru_weights = ART_DIR / "FD001_gru_model.pt"
    if gru_weights.exists():
        gru_trainer.load(gru_weights)
    preds_gru = np.clip(gru_trainer.predict(X_test_seq), 0, 125)

    # 4. Point Estimates
    rmse_lgbm = compute_rmse(y_test_true, preds_lgbm)
    rmse_gru = compute_rmse(y_test_true, preds_gru)
    delta_rmse = rmse_lgbm - rmse_gru

    mae_lgbm = compute_mae(y_test_true, preds_lgbm)
    mae_gru = compute_mae(y_test_true, preds_gru)
    delta_mae = mae_lgbm - mae_gru

    score_lgbm = compute_nasa_score(y_test_true, preds_lgbm)
    score_gru = compute_nasa_score(y_test_true, preds_gru)
    delta_score = score_lgbm - score_gru

    print(f"Point Estimates (N={num_engines}):")
    print(f"  RMSE: LGBM = {rmse_lgbm:.3f}, GRU = {rmse_gru:.3f} | Delta = {delta_rmse:+.3f} cycles")
    print(f"  MAE:  LGBM = {mae_lgbm:.3f}, GRU = {mae_gru:.3f} | Delta = {delta_mae:+.3f} cycles")
    print(f"  NASA: LGBM = {score_lgbm:.1f}, GRU = {score_gru:.1f} | Delta = {delta_score:+.1f} score")

    # 5. Paired Wilcoxon Signed-Rank Test on absolute errors
    errors_lgbm = np.abs(y_test_true - preds_lgbm)
    errors_gru = np.abs(y_test_true - preds_gru)
    w_stat, w_pvalue = stats.wilcoxon(errors_lgbm, errors_gru, alternative="two-sided")
    print(f"\nWilcoxon Signed-Rank Test: W-stat = {w_stat:.1f}, p-value = {w_pvalue:.4e}")

    # 6. Paired Non-Parametric Bootstrap (B = 10,000 resamples)
    print("\nRunning 10,000 paired bootstrap resamples...")
    np.random.seed(42)
    B = 10000
    boot_delta_rmse = np.zeros(B)
    boot_delta_mae = np.zeros(B)
    boot_delta_score = np.zeros(B)

    indices = np.arange(num_engines)
    for b in range(B):
        boot_idx = np.random.choice(indices, size=num_engines, replace=True)
        y_b = y_test_true[boot_idx]
        p_lgbm_b = preds_lgbm[boot_idx]
        p_gru_b = preds_gru[boot_idx]

        r_l = compute_rmse(y_b, p_lgbm_b)
        r_g = compute_rmse(y_b, p_gru_b)
        boot_delta_rmse[b] = r_l - r_g

        m_l = compute_mae(y_b, p_lgbm_b)
        m_g = compute_mae(y_b, p_gru_b)
        boot_delta_mae[b] = m_l - m_g

        s_l = compute_nasa_score(y_b, p_lgbm_b)
        s_g = compute_nasa_score(y_b, p_gru_b)
        boot_delta_score[b] = s_l - s_g

    # Compute 95% CIs
    ci_rmse_low, ci_rmse_high = np.percentile(boot_delta_rmse, [2.5, 97.5])
    ci_mae_low, ci_mae_high = np.percentile(boot_delta_mae, [2.5, 97.5])
    ci_score_low, ci_score_high = np.percentile(boot_delta_score, [2.5, 97.5])

    # Empirical two-tailed p-values
    p_rmse = 2.0 * min(np.mean(boot_delta_rmse <= 0), np.mean(boot_delta_rmse >= 0))
    p_mae = 2.0 * min(np.mean(boot_delta_mae <= 0), np.mean(boot_delta_mae >= 0))
    p_score = 2.0 * min(np.mean(boot_delta_score <= 0), np.mean(boot_delta_score >= 0))

    test_records = [
        {
            "metric": "RMSE",
            "lgbm_value": round(rmse_lgbm, 3),
            "gru_value": round(rmse_gru, 3),
            "delta_lgbm_minus_gru": round(delta_rmse, 3),
            "ci_95_lower": round(ci_rmse_low, 3),
            "ci_95_upper": round(ci_rmse_high, 3),
            "bootstrap_p_value": round(p_rmse, 4),
            "statistically_significant": bool(ci_rmse_low > 0 or ci_rmse_high < 0),
            "unit": "cycles"
        },
        {
            "metric": "MAE",
            "lgbm_value": round(mae_lgbm, 3),
            "gru_value": round(mae_gru, 3),
            "delta_lgbm_minus_gru": round(delta_mae, 3),
            "ci_95_lower": round(ci_mae_low, 3),
            "ci_95_upper": round(ci_mae_high, 3),
            "bootstrap_p_value": round(p_mae, 4),
            "statistically_significant": bool(ci_mae_low > 0 or ci_mae_high < 0),
            "unit": "cycles"
        },
        {
            "metric": "NASA_Score",
            "lgbm_value": round(score_lgbm, 2),
            "gru_value": round(score_gru, 2),
            "delta_lgbm_minus_gru": round(delta_score, 2),
            "ci_95_lower": round(ci_score_low, 2),
            "ci_95_upper": round(ci_score_high, 2),
            "bootstrap_p_value": round(p_score, 4),
            "statistically_significant": bool(ci_score_low > 0 or ci_score_high < 0),
            "unit": "score"
        }
    ]

    print("\nStatistical Test Summary (95% Bootstrap CIs):")
    for r in test_records:
        sig = "Significant (p < 0.05)" if r["statistically_significant"] else "Not Significant"
        print(f"  {r['metric']:10s} | Delta: {r['delta_lgbm_minus_gru']:+.3f} | 95% CI: [{r['ci_95_lower']:+.3f}, {r['ci_95_upper']:+.3f}] | p = {r['bootstrap_p_value']:.4f} | {sig}")

    df_test = pd.DataFrame(test_records)
    csv_path = RESULTS_DIR / "06_statistical_tests.csv"
    df_test.to_csv(csv_path, index=False)

    json_path = RESULTS_DIR / "06_statistical_tests.json"
    with open(json_path, "w") as f:
        json.dump({
            "bootstrap_resamples": B,
            "sample_size": num_engines,
            "wilcoxon_signed_rank": {
                "w_statistic": round(float(w_stat), 2),
                "p_value": float(w_pvalue)
            },
            "metrics": test_records
        }, f, indent=2)

    # Log into provenance registry
    for r in test_records:
        m = r["metric"]
        log_number(f"StatTest_Delta_{m}", "FD001", "LGBM_vs_GRU", f"Delta_{m}", f"{r['delta_lgbm_minus_gru']:.3f}", r["unit"], "experiments/06_statistical_tests.py", "results/paper/06_statistical_tests.json", f"Point difference LGBM - GRU")
        log_number(f"StatTest_CI_Low_{m}", "FD001", "LGBM_vs_GRU", f"CI95_Low_{m}", f"{r['ci_95_lower']:.3f}", r["unit"], "experiments/06_statistical_tests.py", "results/paper/06_statistical_tests.json", f"95% CI lower bound")
        log_number(f"StatTest_CI_High_{m}", "FD001", "LGBM_vs_GRU", f"CI95_High_{m}", f"{r['ci_95_upper']:.3f}", r["unit"], "experiments/06_statistical_tests.py", "results/paper/06_statistical_tests.json", f"95% CI upper bound")
        log_number(f"StatTest_pvalue_{m}", "FD001", "LGBM_vs_GRU", f"p_value_{m}", f"{r['bootstrap_p_value']:.4f}", "p-value", "experiments/06_statistical_tests.py", "results/paper/06_statistical_tests.json", f"Bootstrap p-value")

    log_number("StatTest_Wilcoxon_pvalue", "FD001", "LGBM_vs_GRU", "Wilcoxon_p_value", f"{float(w_pvalue):.4e}", "p-value", "experiments/06_statistical_tests.py", "results/paper/06_statistical_tests.json", "Wilcoxon signed-rank test p-value")

    print(f"\n[SAVED] Statistical test outputs saved to {csv_path.name} and {json_path.name}")


if __name__ == "__main__":
    run_statistical_tests()
