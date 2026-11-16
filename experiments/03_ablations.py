"""
03_ablations.py - Systematic Ablation Studies on NASA C-MAPSS

Investigates four distinct architectural and design dimensions:
1. RUL Piecewise Cap Ablation (FD001, LightGBM):
   - Uncapped linear degradation (None)
   - Cap = 100 cycles
   - Cap = 125 cycles (Baseline)
   - Cap = 150 cycles
2. Feature Group Ablation (FD001, LightGBM):
   - Raw prognostic sensors only (14 features)
   - Raw + Rolling Means (28 features)
   - Raw + Rolling Means + Rolling Stds (42 features)
   - Raw + Means + Stds + Slopes/Diffs (56 features)
   - Full engineered feature set (99 features)
3. Operating Regime Clustered Normalization Ablation (FD004, LightGBM):
   - With KMeans (K=6) Regime Normalization
   - Without Regime Normalization (Global Standard Scaling across all regimes)
4. Sequence Window Length Ablation (FD001, PyTorch GRU):
   - Window W = 20 cycles
   - Window W = 30 cycles (Baseline)
   - Window W = 50 cycles

Outputs:
- results/paper/03_ablations.json
- results/paper/03_ablations.csv
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
from src.models import build_lightgbm, PyTorchGRUTrainer
from src.evaluate import evaluate_predictions
from experiments.provenance_tracker import log_number

RESULTS_DIR = PROJECT_ROOT / "results" / "paper"
PROC_DIR = PROJECT_ROOT / "data" / "processed"
ART_DIR = PROJECT_ROOT / "artifacts"
SENSOR_COLS = INFORMATIVE_SENSORS_FD001


def run_ablations():
    print("=" * 70)
    print("EXPERIMENT 3: Systematic Ablation Studies")
    print("=" * 70)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    ablation_records = []
    summary_dict = {}

    # -------------------------------------------------------------
    # 1. RUL Cap Ablation on FD001 (LightGBM)
    # -------------------------------------------------------------
    print("\n--- 1. RUL Piecewise Linear Cap Ablation (FD001, LightGBM) ---")
    summary_dict["rul_cap_ablation"] = {}

    # Load raw FD001
    train_raw, test_raw, rul_true_s = load_dataset("FD001")
    y_test_true = rul_true_s.values
    train_raw = add_remaining_useful_life(train_raw, rul_cap=125)

    # Pre-scale sensors
    train_split, val_split, _, _ = split_engines_train_val(train_raw, val_ratio=0.20, random_state=42)
    scaler = StandardScaler()
    scaler.fit(train_split[SENSOR_COLS])

    train_norm = train_split.copy()
    test_norm = test_raw.copy()
    train_norm[SENSOR_COLS] = scaler.transform(train_split[SENSOR_COLS])
    test_norm[SENSOR_COLS] = scaler.transform(test_raw[SENSOR_COLS])

    # Build features once (features do not depend on RUL cap, only target does)
    train_feat, feat_cols = build_engineered_features(train_norm, sensor_cols=SENSOR_COLS, window_sizes=[5, 15])
    test_feat, _ = build_engineered_features(test_norm, sensor_cols=SENSOR_COLS, window_sizes=[5, 15])
    test_last = test_feat.groupby("unit_nr").last().reset_index()
    X_train_tab = train_feat[feat_cols].values
    X_test_last = test_last[feat_cols].values

    # Compute unclipped ground truth lifetime for each engine to assign custom caps
    max_cycles = train_norm.groupby("unit_nr")["time_cycles"].max()

    cap_options = [None, 100, 125, 150]
    for cap in cap_options:
        cap_label = "Uncapped" if cap is None else f"Cap_{cap}"
        print(f"  Evaluating RUL Target: {cap_label}...")

        # Assign RUL
        train_norm_copy = train_norm.copy()
        # True RUL per row
        true_ruls = []
        for _, row in train_norm_copy.iterrows():
            u = row["unit_nr"]
            c = row["time_cycles"]
            r = max_cycles[u] - c
            if cap is not None:
                r = min(r, cap)
            true_ruls.append(r)
        train_feat["RUL_target"] = true_ruls
        y_train_target = train_feat["RUL_target"].values

        lgbm = build_lightgbm(random_state=42)
        lgbm.fit(X_train_tab, y_train_target)
        clip_max = cap if cap is not None else 500
        preds = np.clip(lgbm.predict(X_test_last), 0, clip_max)

        metrics = evaluate_predictions(y_test_true, preds)
        metrics["ablation_dimension"] = "RUL_Cap"
        metrics["setting"] = cap_label
        metrics["model"] = "LightGBM"
        ablation_records.append(metrics)
        summary_dict["rul_cap_ablation"][cap_label] = metrics
        print(f"    {cap_label:10s} -> RMSE: {metrics['rmse']:.2f}, MAE: {metrics['mae']:.2f}, Score: {metrics['nasa_score']:.1f}")

        log_number(f"FD001_Ablation_Cap_{cap_label}_RMSE", "FD001", "LightGBM", "RMSE", f"{metrics['rmse']:.3f}", "cycles", "experiments/03_ablations.py", "results/paper/03_ablations.json", f"RUL cap ablation: {cap_label}")
        log_number(f"FD001_Ablation_Cap_{cap_label}_Score", "FD001", "LightGBM", "NASA_Score", f"{metrics['nasa_score']:.2f}", "score", "experiments/03_ablations.py", "results/paper/03_ablations.json", f"RUL cap ablation: {cap_label}")

    # -------------------------------------------------------------
    # 2. Feature Group Ablation on FD001 (LightGBM, Cap=125)
    # -------------------------------------------------------------
    print("\n--- 2. Feature Group Ablation (FD001, LightGBM) ---")
    summary_dict["feature_group_ablation"] = {}

    train_feat_proc = pd.read_parquet(PROC_DIR / "FD001_train_features.parquet")
    test_feat_proc = pd.read_parquet(PROC_DIR / "FD001_test_features.parquet")
    test_last_proc = test_feat_proc.groupby("unit_nr").last().reset_index()

    with open(ART_DIR / "FD001_feature_config.json") as f:
        cfg_proc = json.load(f)
    all_feat_cols = cfg_proc["feature_cols"]

    y_train_standard = train_feat_proc["RUL"].values

    raw_cols = [c for c in all_feat_cols if c in SENSOR_COLS]
    roll_mean_cols = [c for c in all_feat_cols if "roll_mean" in c]
    roll_std_cols = [c for c in all_feat_cols if "roll_std" in c]
    diff_cols = [c for c in all_feat_cols if "diff" in c]

    feature_groups = [
        ("Raw Sensors (14)", raw_cols),
        ("Raw + Rolling Mean (28)", raw_cols + roll_mean_cols[:14]),
        ("Raw + Mean + Std (42)", raw_cols + roll_mean_cols[:14] + roll_std_cols[:14]),
        ("Raw + Mean + Std + Diff (56)", raw_cols + roll_mean_cols[:14] + roll_std_cols[:14] + diff_cols[:14]),
        ("Full Engineered Set (99)", all_feat_cols)
    ]

    for grp_name, cols in feature_groups:
        print(f"  Evaluating Feature Group: {grp_name} ({len(cols)} features)...")
        X_tr_sub = train_feat_proc[cols].values
        X_te_sub = test_last_proc[cols].values

        lgbm = build_lightgbm(random_state=42)
        lgbm.fit(X_tr_sub, y_train_standard)
        preds = np.clip(lgbm.predict(X_te_sub), 0, 125)

        metrics = evaluate_predictions(y_test_true, preds)
        metrics["ablation_dimension"] = "Feature_Group"
        metrics["setting"] = grp_name
        metrics["num_features"] = len(cols)
        metrics["model"] = "LightGBM"
        ablation_records.append(metrics)
        summary_dict["feature_group_ablation"][grp_name] = metrics
        print(f"    {grp_name:30s} -> RMSE: {metrics['rmse']:.2f}, MAE: {metrics['mae']:.2f}, Score: {metrics['nasa_score']:.1f}")

        safe_name = grp_name.split("(")[0].strip().replace(" ", "_").replace("+", "plus")
        log_number(f"FD001_Ablation_Feat_{safe_name}_RMSE", "FD001", "LightGBM", "RMSE", f"{metrics['rmse']:.3f}", "cycles", "experiments/03_ablations.py", "results/paper/03_ablations.json", f"Feature group ablation: {grp_name}")
        log_number(f"FD001_Ablation_Feat_{safe_name}_Score", "FD001", "LightGBM", "NASA_Score", f"{metrics['nasa_score']:.2f}", "score", "experiments/03_ablations.py", "results/paper/03_ablations.json", f"Feature group ablation: {grp_name}")

    # -------------------------------------------------------------
    # 3. Operating Regime Normalization Ablation on FD004 (LightGBM)
    # -------------------------------------------------------------
    print("\n--- 3. Operating Regime Normalization Ablation (FD004, LightGBM) ---")
    summary_dict["regime_normalization_ablation"] = {}

    train_fd4_raw, test_fd4_raw, rul_fd4_s = load_dataset("FD004")
    y_test_fd4 = rul_fd4_s.values
    train_fd4_raw = add_remaining_useful_life(train_fd4_raw, rul_cap=125)
    train_fd4_split, val_fd4_split, _, _ = split_engines_train_val(train_fd4_raw, val_ratio=0.20, random_state=42)

    # Setting A: WITH Regime Clustering Normalization (KMeans K=6)
    tr_feat_regime = pd.read_parquet(PROC_DIR / "FD004_train_features.parquet")
    te_feat_regime = pd.read_parquet(PROC_DIR / "FD004_test_features.parquet")
    with open(ART_DIR / "FD004_feature_config.json") as f:
        cols_fd4 = json.load(f)["feature_cols"]
    te_last_regime = te_feat_regime.groupby("unit_nr").last().reset_index()

    lgbm_regime = build_lightgbm(random_state=42)
    lgbm_regime.fit(tr_feat_regime[cols_fd4].values, tr_feat_regime["RUL"].values)
    pred_with_regime = np.clip(lgbm_regime.predict(te_last_regime[cols_fd4].values), 0, 125)
    m_with_regime = evaluate_predictions(y_test_fd4, pred_with_regime)
    m_with_regime["ablation_dimension"] = "Regime_Normalization"
    m_with_regime["setting"] = "With_Regime_Clustering"
    ablation_records.append(m_with_regime)
    summary_dict["regime_normalization_ablation"]["With_Regime_Clustering"] = m_with_regime
    print(f"  With Regime Clustering    -> RMSE: {m_with_regime['rmse']:.2f}, MAE: {m_with_regime['mae']:.2f}, Score: {m_with_regime['nasa_score']:.1f}")

    # Setting B: WITHOUT Regime Clustering (Global StandardScaler across all regimes)
    global_scaler = StandardScaler()
    global_scaler.fit(train_fd4_split[SENSOR_COLS])
    tr_global = train_fd4_split.copy()
    te_global = test_fd4_raw.copy()
    tr_global[SENSOR_COLS] = global_scaler.transform(train_fd4_split[SENSOR_COLS])
    te_global[SENSOR_COLS] = global_scaler.transform(test_fd4_raw[SENSOR_COLS])

    tr_feat_global, _ = build_engineered_features(tr_global, sensor_cols=SENSOR_COLS, window_sizes=[5, 15])
    te_feat_global, _ = build_engineered_features(te_global, sensor_cols=SENSOR_COLS, window_sizes=[5, 15])
    te_last_global = te_feat_global.groupby("unit_nr").last().reset_index()

    lgbm_global = build_lightgbm(random_state=42)
    lgbm_global.fit(tr_feat_global[cols_fd4].values, tr_feat_global["RUL"].values)
    pred_without_regime = np.clip(lgbm_global.predict(te_last_global[cols_fd4].values), 0, 125)
    m_without_regime = evaluate_predictions(y_test_fd4, pred_without_regime)
    m_without_regime["ablation_dimension"] = "Regime_Normalization"
    m_without_regime["setting"] = "Without_Regime_Clustering (Global Standard)"
    ablation_records.append(m_without_regime)
    summary_dict["regime_normalization_ablation"]["Without_Regime_Clustering"] = m_without_regime
    print(f"  Without Regime Clustering -> RMSE: {m_without_regime['rmse']:.2f}, MAE: {m_without_regime['mae']:.2f}, Score: {m_without_regime['nasa_score']:.1f}")

    log_number("FD004_Ablation_WithRegime_RMSE", "FD004", "LightGBM", "RMSE", f"{m_with_regime['rmse']:.3f}", "cycles", "experiments/03_ablations.py", "results/paper/03_ablations.json", "FD004 with regime clustering")
    log_number("FD004_Ablation_WithoutRegime_RMSE", "FD004", "LightGBM", "RMSE", f"{m_without_regime['rmse']:.3f}", "cycles", "experiments/03_ablations.py", "results/paper/03_ablations.json", "FD004 without regime clustering")
    log_number("FD004_Ablation_WithRegime_Score", "FD004", "LightGBM", "NASA_Score", f"{m_with_regime['nasa_score']:.2f}", "score", "experiments/03_ablations.py", "results/paper/03_ablations.json", "FD004 with regime clustering")
    log_number("FD004_Ablation_WithoutRegime_Score", "FD004", "LightGBM", "NASA_Score", f"{m_without_regime['nasa_score']:.2f}", "score", "experiments/03_ablations.py", "results/paper/03_ablations.json", "FD004 without regime clustering")

    # -------------------------------------------------------------
    # 4. Sequence Window Length Ablation for GRU on FD001
    # -------------------------------------------------------------
    print("\n--- 4. Sequence Window Length Ablation (FD001, PyTorch GRU) ---")
    summary_dict["window_length_ablation"] = {}

    train_fd1_norm = train_norm.copy()
    val_fd1_norm = val_split.copy()
    val_fd1_norm[SENSOR_COLS] = scaler.transform(val_split[SENSOR_COLS])
    test_fd1_norm = test_norm.copy()

    for w in [20, 30, 50]:
        print(f"  Evaluating GRU Sequence Window W = {w} cycles...")
        X_tr_w, y_tr_w, _ = generate_sequence_windows(train_fd1_norm, feature_cols=SENSOR_COLS, window_size=w)
        X_val_w, y_val_w, _ = generate_sequence_windows(val_fd1_norm, feature_cols=SENSOR_COLS, window_size=w)
        X_te_w, _ = extract_last_sequence_for_test(test_fd1_norm, feature_cols=SENSOR_COLS, window_size=w)

        gru_w = PyTorchGRUTrainer(
            input_dim=len(SENSOR_COLS),
            hidden_dim=64,
            num_layers=2,
            learning_rate=0.002,
            batch_size=128,
            device="cpu",
            seed=42
        )
        gru_w.fit(X_tr_w, y_tr_w, X_val_w, y_val_w, max_epochs=20, patience=5)
        pred_w = np.clip(gru_w.predict(X_te_w), 0, 125)

        m_w = evaluate_predictions(y_test_true, pred_w)
        m_w["ablation_dimension"] = "Window_Length"
        m_w["setting"] = f"W_{w}"
        m_w["window_length"] = w
        m_w["model"] = "PyTorch GRU"
        ablation_records.append(m_w)
        summary_dict["window_length_ablation"][f"W_{w}"] = m_w
        print(f"    Window W={w:2d} -> RMSE: {m_w['rmse']:.2f}, MAE: {m_w['mae']:.2f}, Score: {m_w['nasa_score']:.1f}")

        log_number(f"FD001_Ablation_Window_{w}_RMSE", "FD001", "PyTorch GRU", "RMSE", f"{m_w['rmse']:.3f}", "cycles", "experiments/03_ablations.py", "results/paper/03_ablations.json", f"Window length ablation: W={w}")
        log_number(f"FD001_Ablation_Window_{w}_Score", "FD001", "PyTorch GRU", "NASA_Score", f"{m_w['nasa_score']:.2f}", "score", "experiments/03_ablations.py", "results/paper/03_ablations.json", f"Window length ablation: W={w}")

    # Save summary
    df_ab = pd.DataFrame(ablation_records)
    csv_path = RESULTS_DIR / "03_ablations.csv"
    df_ab.to_csv(csv_path, index=False)

    json_path = RESULTS_DIR / "03_ablations.json"
    with open(json_path, "w") as f:
        json.dump(summary_dict, f, indent=2)

    print(f"\n[SAVED] Ablation results saved to {csv_path.name} and {json_path.name}")


if __name__ == "__main__":
    run_ablations()
