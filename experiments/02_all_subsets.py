"""
02_all_subsets.py - Comprehensive Benchmark Across All 4 NASA C-MAPSS Subsets

Evaluates:
- FD001 (1 condition, 1 fault mode: HPC)
- FD002 (6 conditions, 1 fault mode: HPC)
- FD003 (1 condition, 2 fault modes: HPC + Fan)
- FD004 (6 conditions, 2 fault modes: HPC + Fan)

Models:
1. Ridge Linear Regression (Baseline)
2. Random Forest Regressor (Baseline)
3. LightGBM Regressor (Production Gradient Boosting)
4. PyTorch GRU (Recurrent Neural Network)

Leakage-Free Protocol:
- Train/val split strictly by engine ID.
- Scaler / RegimeClusterNormalizer fit on training split only.
- Ground truth test labels loaded from RUL_FD00X.txt.

Outputs:
- results/paper/02_all_subsets.json
- results/paper/02_all_subsets.csv
- Appends to paper/number_provenance.csv
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from src.data import load_dataset, add_remaining_useful_life
from src.features import (
    INFORMATIVE_SENSORS_FD001,
    RegimeClusterNormalizer,
    build_engineered_features,
    generate_sequence_windows,
    extract_last_sequence_for_test,
    split_engines_train_val
)
from src.models import (
    build_linear_regression,
    build_random_forest,
    build_lightgbm,
    PyTorchGRUTrainer
)
from src.evaluate import evaluate_predictions
from experiments.provenance_tracker import log_number

RESULTS_DIR = PROJECT_ROOT / "results" / "paper"
PROC_DIR = PROJECT_ROOT / "data" / "processed"
ART_DIR = PROJECT_ROOT / "artifacts"
SENSOR_COLS = INFORMATIVE_SENSORS_FD001


def prepare_dataset_if_needed(dataset_id: str):
    """Checks if processed features exist for dataset_id; generates them if missing."""
    train_feat_path = PROC_DIR / f"{dataset_id}_train_features.parquet"
    test_feat_path = PROC_DIR / f"{dataset_id}_test_features.parquet"
    x_seq_path = PROC_DIR / f"{dataset_id}_X_train_seq.npy"
    
    if train_feat_path.exists() and test_feat_path.exists() and x_seq_path.exists():
        print(f"[{dataset_id}] Preprocessed features and sequences already exist on disk.")
        return

    print(f"[{dataset_id}] Preprocessing raw data (Leakage-Free Protocol)...")
    train_df, test_df, rul_true = load_dataset(dataset_id)
    train_df = add_remaining_useful_life(train_df, rul_cap=125)

    # Split train/val by engine unit_nr
    train_split, val_split, train_units, val_units = split_engines_train_val(
        train_df, val_ratio=0.20, random_state=42
    )

    is_multi_regime = dataset_id in ["FD002", "FD004"]

    if is_multi_regime:
        normalizer = RegimeClusterNormalizer(
            n_clusters=6, sensor_cols=SENSOR_COLS, random_state=42
        )
        normalizer.fit(train_split)
        train_split_norm = normalizer.transform(train_split)
        val_split_norm = normalizer.transform(val_split)
        test_df_norm = normalizer.transform(test_df)
    else:
        scaler = StandardScaler()
        scaler.fit(train_split[SENSOR_COLS])
        train_split_norm = train_split.copy()
        val_split_norm = val_split.copy()
        test_df_norm = test_df.copy()
        train_split_norm[SENSOR_COLS] = scaler.transform(train_split[SENSOR_COLS])
        val_split_norm[SENSOR_COLS] = scaler.transform(val_split[SENSOR_COLS])
        test_df_norm[SENSOR_COLS] = scaler.transform(test_df[SENSOR_COLS])

    # Tabular features
    train_feat, feat_cols = build_engineered_features(train_split_norm, sensor_cols=SENSOR_COLS, window_sizes=[5, 15])
    val_feat, _ = build_engineered_features(val_split_norm, sensor_cols=SENSOR_COLS, window_sizes=[5, 15])
    test_feat, _ = build_engineered_features(test_df_norm, sensor_cols=SENSOR_COLS, window_sizes=[5, 15])

    # Sequence windows for GRU
    X_train_seq, y_train_seq, _ = generate_sequence_windows(train_split_norm, feature_cols=SENSOR_COLS, window_size=30)
    X_val_seq, y_val_seq, _ = generate_sequence_windows(val_split_norm, feature_cols=SENSOR_COLS, window_size=30)
    X_test_seq, _ = extract_last_sequence_for_test(test_df_norm, feature_cols=SENSOR_COLS, window_size=30)

    # Save to disk
    train_feat.to_parquet(train_feat_path, index=False)
    val_feat.to_parquet(PROC_DIR / f"{dataset_id}_val_features.parquet", index=False)
    test_feat.to_parquet(test_feat_path, index=False)

    np.save(PROC_DIR / f"{dataset_id}_X_train_seq.npy", X_train_seq)
    np.save(PROC_DIR / f"{dataset_id}_y_train_seq.npy", y_train_seq)
    np.save(PROC_DIR / f"{dataset_id}_X_val_seq.npy", X_val_seq)
    np.save(PROC_DIR / f"{dataset_id}_y_val_seq.npy", y_val_seq)
    np.save(PROC_DIR / f"{dataset_id}_X_test_seq.npy", X_test_seq)
    np.save(PROC_DIR / f"{dataset_id}_y_test_true.npy", rul_true.values)

    with open(ART_DIR / f"{dataset_id}_feature_config.json", "w") as f:
        json.dump({
            "dataset_id": dataset_id,
            "feature_cols": feat_cols,
            "sensor_cols": SENSOR_COLS,
            "num_features": len(feat_cols)
        }, f, indent=2)

    print(f"[{dataset_id}] Preprocessing complete and saved.")


def run_benchmark():
    print("=" * 70)
    print("EXPERIMENT 2: Comprehensive Benchmark Across All 4 C-MAPSS Subsets")
    print("=" * 70)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = []
    summary_dict = {}

    for dataset_id in ["FD001", "FD002", "FD003", "FD004"]:
        print(f"\n>>> PROCESSING & EVALUATING {dataset_id} <<<")
        prepare_dataset_if_needed(dataset_id)

        # Load tabular features
        with open(ART_DIR / f"{dataset_id}_feature_config.json", "r") as f:
            cfg = json.load(f)
        feature_cols = cfg["feature_cols"]

        train_feat = pd.read_parquet(PROC_DIR / f"{dataset_id}_train_features.parquet")
        test_feat = pd.read_parquet(PROC_DIR / f"{dataset_id}_test_features.parquet")

        X_train_tab = train_feat[feature_cols].values
        y_train_tab = train_feat["RUL"].values

        test_last = test_feat.groupby("unit_nr").last().reset_index()
        X_test_last = test_last[feature_cols].values

        # Load sequence data
        X_train_seq = np.load(PROC_DIR / f"{dataset_id}_X_train_seq.npy")
        y_train_seq = np.load(PROC_DIR / f"{dataset_id}_y_train_seq.npy")
        X_val_seq = np.load(PROC_DIR / f"{dataset_id}_X_val_seq.npy")
        y_val_seq = np.load(PROC_DIR / f"{dataset_id}_y_val_seq.npy")
        X_test_seq = np.load(PROC_DIR / f"{dataset_id}_X_test_seq.npy")
        y_test_true = np.load(PROC_DIR / f"{dataset_id}_y_test_true.npy")

        summary_dict[dataset_id] = {}

        # 1. Ridge Baseline
        print(f"  Training Ridge Regression ({dataset_id})...")
        ridge = build_linear_regression(alpha=10.0)
        ridge.fit(X_train_tab, y_train_tab)
        pred_ridge = np.clip(ridge.predict(X_test_last), 0, 125)
        m_ridge = evaluate_predictions(y_test_true, pred_ridge)
        m_ridge["dataset"] = dataset_id
        m_ridge["model"] = "Linear Regression (Ridge)"
        all_results.append(m_ridge)
        summary_dict[dataset_id]["Ridge"] = m_ridge
        print(f"    Ridge -> RMSE: {m_ridge['rmse']:.2f}, MAE: {m_ridge['mae']:.2f}, Score: {m_ridge['nasa_score']:.1f}")

        # 2. Random Forest Baseline
        print(f"  Training Random Forest ({dataset_id})...")
        rf = build_random_forest(n_estimators=100, max_depth=12)
        rf.fit(X_train_tab, y_train_tab)
        pred_rf = np.clip(rf.predict(X_test_last), 0, 125)
        m_rf = evaluate_predictions(y_test_true, pred_rf)
        m_rf["dataset"] = dataset_id
        m_rf["model"] = "Random Forest"
        all_results.append(m_rf)
        summary_dict[dataset_id]["Random Forest"] = m_rf
        print(f"    Random Forest -> RMSE: {m_rf['rmse']:.2f}, MAE: {m_rf['mae']:.2f}, Score: {m_rf['nasa_score']:.1f}")

        # 3. LightGBM
        print(f"  Training LightGBM ({dataset_id})...")
        lgbm = build_lightgbm(n_estimators=250, learning_rate=0.05, num_leaves=31, max_depth=6, random_state=42)
        lgbm.fit(X_train_tab, y_train_tab)
        pred_lgbm = np.clip(lgbm.predict(X_test_last), 0, 125)
        m_lgbm = evaluate_predictions(y_test_true, pred_lgbm)
        m_lgbm["dataset"] = dataset_id
        m_lgbm["model"] = "LightGBM"
        all_results.append(m_lgbm)
        summary_dict[dataset_id]["LightGBM"] = m_lgbm
        print(f"    LightGBM -> RMSE: {m_lgbm['rmse']:.2f}, MAE: {m_lgbm['mae']:.2f}, Score: {m_lgbm['nasa_score']:.1f}")

        # 4. PyTorch GRU
        print(f"  Training PyTorch GRU ({dataset_id})...")
        gru = PyTorchGRUTrainer(
            input_dim=X_train_seq.shape[2],
            hidden_dim=64,
            num_layers=2,
            learning_rate=0.002,
            batch_size=128,
            device="cpu",
            seed=42
        )
        gru.fit(X_train_seq, y_train_seq, X_val_seq, y_val_seq, max_epochs=25, patience=6)
        pred_gru = np.clip(gru.predict(X_test_seq), 0, 125)
        m_gru = evaluate_predictions(y_test_true, pred_gru)
        m_gru["dataset"] = dataset_id
        m_gru["model"] = "PyTorch GRU"
        all_results.append(m_gru)
        summary_dict[dataset_id]["PyTorch GRU"] = m_gru
        print(f"    PyTorch GRU -> RMSE: {m_gru['rmse']:.2f}, MAE: {m_gru['mae']:.2f}, Score: {m_gru['nasa_score']:.1f}")

        # Provenance logging
        for m_name, m_dict in [("Ridge", m_ridge), ("RF", m_rf), ("LGBM", m_lgbm), ("GRU", m_gru)]:
            log_number(f"{dataset_id}_{m_name}_RMSE", dataset_id, m_dict["model"], "RMSE", f"{m_dict['rmse']:.3f}", "cycles", "experiments/02_all_subsets.py", "results/paper/02_all_subsets.json", f"{dataset_id} test RMSE")
            log_number(f"{dataset_id}_{m_name}_MAE", dataset_id, m_dict["model"], "MAE", f"{m_dict['mae']:.3f}", "cycles", "experiments/02_all_subsets.py", "results/paper/02_all_subsets.json", f"{dataset_id} test MAE")
            log_number(f"{dataset_id}_{m_name}_Score", dataset_id, m_dict["model"], "NASA_Score", f"{m_dict['nasa_score']:.2f}", "score", "experiments/02_all_subsets.py", "results/paper/02_all_subsets.json", f"{dataset_id} test NASA Score")

    df_res = pd.DataFrame(all_results)
    csv_path = RESULTS_DIR / "02_all_subsets.csv"
    df_res.to_csv(csv_path, index=False)

    json_path = RESULTS_DIR / "02_all_subsets.json"
    with open(json_path, "w") as f:
        json.dump(summary_dict, f, indent=2)

    print(f"\n[SAVED] Benchmark outputs saved to {csv_path.name} and {json_path.name}")


if __name__ == "__main__":
    run_benchmark()
