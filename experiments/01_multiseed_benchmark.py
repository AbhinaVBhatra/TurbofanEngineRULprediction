"""
01_multiseed_benchmark.py - Multi-Seed Reproducibility & Stability Evaluation

Trains LightGBM and PyTorch GRU on FD001 across 5 distinct random seeds:
[42, 123, 456, 789, 2024]
Computes Mean +/- Standard Deviation for:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- NASA Asymmetric Scoring Function (PHM 2008 Benchmark)

Outputs:
- results/paper/01_multiseed_benchmark.json
- results/paper/01_multiseed_benchmark.csv
- Registers metrics into paper/number_provenance.csv
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from src.models import build_lightgbm, PyTorchGRUTrainer
from src.evaluate import evaluate_predictions
from experiments.provenance_tracker import log_number

RESULTS_DIR = PROJECT_ROOT / "results" / "paper"
PROC_DIR = PROJECT_ROOT / "data" / "processed"
ART_DIR = PROJECT_ROOT / "artifacts"
SEEDS = [42, 123, 456, 789, 2024]


def run_multiseed():
    print("=" * 70)
    print("EXPERIMENT 1: Multi-Seed Benchmark on NASA C-MAPSS FD001")
    print("=" * 70)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Load tabular data for LightGBM
    with open(ART_DIR / "FD001_feature_config.json", "r") as f:
        feat_config = json.load(f)
    feature_cols = feat_config["feature_cols"]

    train_feat = pd.read_parquet(PROC_DIR / "FD001_train_features.parquet")
    test_feat = pd.read_parquet(PROC_DIR / "FD001_test_features.parquet")

    X_train_tab = train_feat[feature_cols].values
    y_train_tab = train_feat["RUL"].values

    test_last_records = test_feat.groupby("unit_nr").last().reset_index()
    X_test_last = test_last_records[feature_cols].values

    # 2. Load sequence data for PyTorch GRU
    X_train_seq = np.load(PROC_DIR / "FD001_X_train_seq.npy")
    y_train_seq = np.load(PROC_DIR / "FD001_y_train_seq.npy")
    X_val_seq = np.load(PROC_DIR / "FD001_X_val_seq.npy")
    y_val_seq = np.load(PROC_DIR / "FD001_y_val_seq.npy")
    X_test_seq = np.load(PROC_DIR / "FD001_X_test_seq.npy")

    y_test_true = np.load(PROC_DIR / "FD001_y_test_true.npy")

    results_data = []

    # -------------------------------------------------------------
    # Evaluate LightGBM across 5 seeds
    # -------------------------------------------------------------
    print("\n--- Benchmarking LightGBM (5 Seeds) ---")
    lgbm_runs = []
    for seed in SEEDS:
        print(f"Training LightGBM (seed={seed})...")
        lgbm = build_lightgbm(
            n_estimators=250,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=6,
            random_state=seed
        )
        lgbm.fit(X_train_tab, y_train_tab)
        preds = np.clip(lgbm.predict(X_test_last), 0, 125)
        metrics = evaluate_predictions(y_test_true, preds)
        metrics["seed"] = seed
        metrics["model"] = "LightGBM"
        lgbm_runs.append(metrics)
        print(f"  Seed {seed} -> RMSE: {metrics['rmse']:.3f}, MAE: {metrics['mae']:.3f}, Score: {metrics['nasa_score']:.2f}")

    # -------------------------------------------------------------
    # Evaluate PyTorch GRU across 5 seeds
    # -------------------------------------------------------------
    print("\n--- Benchmarking PyTorch GRU (5 Seeds) ---")
    gru_runs = []
    for seed in SEEDS:
        print(f"Training PyTorch GRU (seed={seed})...")
        gru_trainer = PyTorchGRUTrainer(
            input_dim=X_train_seq.shape[2],
            hidden_dim=64,
            num_layers=2,
            learning_rate=0.002,
            batch_size=128,
            device="cpu",
            seed=seed
        )
        gru_trainer.fit(
            X_train_seq, y_train_seq,
            X_val_seq, y_val_seq,
            max_epochs=25,
            patience=6
        )
        preds = np.clip(gru_trainer.predict(X_test_seq), 0, 125)
        metrics = evaluate_predictions(y_test_true, preds)
        metrics["seed"] = seed
        metrics["model"] = "PyTorch GRU"
        gru_runs.append(metrics)
        print(f"  Seed {seed} -> RMSE: {metrics['rmse']:.3f}, MAE: {metrics['mae']:.3f}, Score: {metrics['nasa_score']:.2f}")

    all_runs = lgbm_runs + gru_runs
    df_runs = pd.DataFrame(all_runs)
    csv_path = RESULTS_DIR / "01_multiseed_benchmark.csv"
    df_runs.to_csv(csv_path, index=False)
    print(f"\n[SAVED] Per-seed run metrics -> {csv_path}")

    # Compute aggregate statistics
    summary = {}
    for model_name, runs in [("LightGBM", lgbm_runs), ("PyTorch GRU", gru_runs)]:
        rmse_vals = [r["rmse"] for r in runs]
        mae_vals = [r["mae"] for r in runs]
        score_vals = [r["nasa_score"] for r in runs]

        summary[model_name] = {
            "rmse_mean": float(np.mean(rmse_vals)),
            "rmse_std": float(np.std(rmse_vals, ddof=1)),
            "mae_mean": float(np.mean(mae_vals)),
            "mae_std": float(np.std(mae_vals, ddof=1)),
            "score_mean": float(np.mean(score_vals)),
            "score_std": float(np.std(score_vals, ddof=1)),
            "seeds": SEEDS,
            "individual_runs": runs
        }

        # Log into provenance registry
        mod_prefix = "LGBM" if "LightGBM" in model_name else "GRU"
        log_number(f"FD001_{mod_prefix}_RMSE_mean", "FD001", model_name, "RMSE_mean", f"{summary[model_name]['rmse_mean']:.3f}", "cycles", "experiments/01_multiseed_benchmark.py", "results/paper/01_multiseed_benchmark.json", "5-seed mean on FD001 test")
        log_number(f"FD001_{mod_prefix}_RMSE_std", "FD001", model_name, "RMSE_std", f"{summary[model_name]['rmse_std']:.3f}", "cycles", "experiments/01_multiseed_benchmark.py", "results/paper/01_multiseed_benchmark.json", "5-seed std on FD001 test")
        log_number(f"FD001_{mod_prefix}_MAE_mean", "FD001", model_name, "MAE_mean", f"{summary[model_name]['mae_mean']:.3f}", "cycles", "experiments/01_multiseed_benchmark.py", "results/paper/01_multiseed_benchmark.json", "5-seed mean on FD001 test")
        log_number(f"FD001_{mod_prefix}_MAE_std", "FD001", model_name, "MAE_std", f"{summary[model_name]['mae_std']:.3f}", "cycles", "experiments/01_multiseed_benchmark.py", "results/paper/01_multiseed_benchmark.json", "5-seed std on FD001 test")
        log_number(f"FD001_{mod_prefix}_Score_mean", "FD001", model_name, "NASA_Score_mean", f"{summary[model_name]['score_mean']:.2f}", "score", "experiments/01_multiseed_benchmark.py", "results/paper/01_multiseed_benchmark.json", "5-seed mean on FD001 test")
        log_number(f"FD001_{mod_prefix}_Score_std", "FD001", model_name, "NASA_Score_std", f"{summary[model_name]['score_std']:.2f}", "score", "experiments/01_multiseed_benchmark.py", "results/paper/01_multiseed_benchmark.json", "5-seed std on FD001 test")

    json_path = RESULTS_DIR / "01_multiseed_benchmark.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[SAVED] Multi-seed summary -> {json_path}")

    print("\n================== SUMMARY RESULTS ==================")
    for model_name, stats in summary.items():
        print(f"{model_name:12s} | RMSE: {stats['rmse_mean']:.2f} +/- {stats['rmse_std']:.2f} | MAE: {stats['mae_mean']:.2f} +/- {stats['mae_std']:.2f} | NASA Score: {stats['score_mean']:.1f} +/- {stats['score_std']:.1f}")
    print("=====================================================")


if __name__ == "__main__":
    run_multiseed()
