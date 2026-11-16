"""
02_features.py - Feature Engineering Pipeline Execution & Artifact Generation

Executes preprocessing, regime clustering, rolling statistical extraction,
and sequence window construction for FD001 and FD004.
Saves serialized scalers, clusterers, schemas, and processed datasets.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data import load_dataset, add_remaining_useful_life, SENSOR_COLS, SETTING_COLS
from src.features import (
    INFORMATIVE_SENSORS_FD001,
    RegimeClusterNormalizer,
    build_engineered_features,
    generate_sequence_windows,
    extract_last_sequence_for_test,
    split_engines_train_val,
    save_pipeline_artifacts
)
from sklearn.preprocessing import StandardScaler


def process_fd001():
    print("\n==========================================")
    print("[PIPELINE] Processing Dataset FD001 (Single Regime)")
    print("==========================================")
    train_df, test_df, rul_true = load_dataset("FD001")
    train_df = add_remaining_useful_life(train_df, rul_cap=125)

    # 1. Split training engines strictly by unit_nr (80 train / 20 validation)
    train_split, val_split, train_units, val_units = split_engines_train_val(train_df, val_ratio=0.20, random_state=42)
    print(f"[FD001] Engine Split: {len(train_units)} Train Engines, {len(val_units)} Validation Engines.")

    # 2. Fit Scaler ONLY on Training engines
    scaler = StandardScaler()
    scaler.fit(train_split[INFORMATIVE_SENSORS_FD001])

    # 3. Transform all splits
    train_split_norm = train_split.copy()
    val_split_norm = val_split.copy()
    test_df_norm = test_df.copy()

    train_split_norm[INFORMATIVE_SENSORS_FD001] = scaler.transform(train_split[INFORMATIVE_SENSORS_FD001])
    val_split_norm[INFORMATIVE_SENSORS_FD001] = scaler.transform(val_split[INFORMATIVE_SENSORS_FD001])
    test_df_norm[INFORMATIVE_SENSORS_FD001] = scaler.transform(test_df[INFORMATIVE_SENSORS_FD001])

    # 4. Build rolling engineered features (mean, std, diff over windows 5 and 15)
    train_feat, feat_cols = build_engineered_features(train_split_norm, INFORMATIVE_SENSORS_FD001, window_sizes=[5, 15])
    val_feat, _ = build_engineered_features(val_split_norm, INFORMATIVE_SENSORS_FD001, window_sizes=[5, 15])
    test_feat, _ = build_engineered_features(test_df_norm, INFORMATIVE_SENSORS_FD001, window_sizes=[5, 15])

    print(f"[FD001] Engineered Feature Count: {len(feat_cols)} features created.")

    # 5. Build 3D Sequence Windows for GRU/LSTM (W=30)
    X_train_seq, y_train_seq, _ = generate_sequence_windows(train_split_norm, INFORMATIVE_SENSORS_FD001, window_size=30)
    X_val_seq, y_val_seq, _ = generate_sequence_windows(val_split_norm, INFORMATIVE_SENSORS_FD001, window_size=30)
    X_test_seq, test_unit_ids = extract_last_sequence_for_test(test_df_norm, INFORMATIVE_SENSORS_FD001, window_size=30)

    print(f"[FD001] Sequence Shapes: Train X={X_train_seq.shape}, Val X={X_val_seq.shape}, Test X={X_test_seq.shape}")

    # 6. Save artifacts
    artifacts_dir = PROJECT_ROOT / "artifacts"
    save_pipeline_artifacts(
        artifacts_dir=artifacts_dir,
        dataset_id="FD001",
        scaler=scaler,
        feature_cols=feat_cols,
        sensor_cols=INFORMATIVE_SENSORS_FD001,
        extra_metadata={
            "train_engines": len(train_units),
            "val_engines": len(val_units),
            "test_engines": len(rul_true),
            "window_size": 30
        }
    )

    # 7. Save processed data arrays and dataframes
    proc_dir = PROJECT_ROOT / "data" / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)

    train_feat.to_parquet(proc_dir / "FD001_train_features.parquet", index=False)
    val_feat.to_parquet(proc_dir / "FD001_val_features.parquet", index=False)
    test_feat.to_parquet(proc_dir / "FD001_test_features.parquet", index=False)

    np.save(proc_dir / "FD001_X_train_seq.npy", X_train_seq)
    np.save(proc_dir / "FD001_y_train_seq.npy", y_train_seq)
    np.save(proc_dir / "FD001_X_val_seq.npy", X_val_seq)
    np.save(proc_dir / "FD001_y_val_seq.npy", y_val_seq)
    np.save(proc_dir / "FD001_X_test_seq.npy", X_test_seq)
    np.save(proc_dir / "FD001_y_test_true.npy", rul_true.values)

    print("[FD001] Processing complete and persisted.")


def process_fd004():
    print("\n==========================================")
    print("[PIPELINE] Processing Dataset FD004 (6 Operating Regimes)")
    print("==========================================")
    train_df, test_df, rul_true = load_dataset("FD004")
    train_df = add_remaining_useful_life(train_df, rul_cap=125)

    # 1. Split training engines strictly by unit_nr (80/20)
    train_split, val_split, train_units, val_units = split_engines_train_val(train_df, val_ratio=0.20, random_state=42)
    print(f"[FD004] Engine Split: {len(train_units)} Train Engines, {len(val_units)} Validation Engines.")

    # 2. Fit RegimeClusterNormalizer ONLY on Training engines
    cluster_normalizer = RegimeClusterNormalizer(n_clusters=6, sensor_cols=INFORMATIVE_SENSORS_FD001, random_state=42)
    cluster_normalizer.fit(train_split)

    # 3. Transform all splits
    train_split_norm = cluster_normalizer.transform(train_split)
    val_split_norm = cluster_normalizer.transform(val_split)
    test_df_norm = cluster_normalizer.transform(test_df)

    # Secondary standard scaler on normalized values for stable convergence
    scaler = StandardScaler()
    scaler.fit(train_split_norm[INFORMATIVE_SENSORS_FD001])

    train_split_norm[INFORMATIVE_SENSORS_FD001] = scaler.transform(train_split_norm[INFORMATIVE_SENSORS_FD001])
    val_split_norm[INFORMATIVE_SENSORS_FD001] = scaler.transform(val_split_norm[INFORMATIVE_SENSORS_FD001])
    test_df_norm[INFORMATIVE_SENSORS_FD001] = scaler.transform(test_df_norm[INFORMATIVE_SENSORS_FD001])

    # 4. Build rolling features
    train_feat, feat_cols = build_engineered_features(train_split_norm, INFORMATIVE_SENSORS_FD001, window_sizes=[5, 15])
    val_feat, _ = build_engineered_features(val_split_norm, INFORMATIVE_SENSORS_FD001, window_sizes=[5, 15])
    test_feat, _ = build_engineered_features(test_df_norm, INFORMATIVE_SENSORS_FD001, window_sizes=[5, 15])

    print(f"[FD004] Engineered Feature Count: {len(feat_cols)} features created.")

    # 5. Build 3D Sequence Windows
    X_train_seq, y_train_seq, _ = generate_sequence_windows(train_split_norm, INFORMATIVE_SENSORS_FD001, window_size=30)
    X_val_seq, y_val_seq, _ = generate_sequence_windows(val_split_norm, INFORMATIVE_SENSORS_FD001, window_size=30)
    X_test_seq, test_unit_ids = extract_last_sequence_for_test(test_df_norm, INFORMATIVE_SENSORS_FD001, window_size=30)

    print(f"[FD004] Sequence Shapes: Train X={X_train_seq.shape}, Val X={X_val_seq.shape}, Test X={X_test_seq.shape}")

    # 6. Save artifacts
    artifacts_dir = PROJECT_ROOT / "artifacts"
    save_pipeline_artifacts(
        artifacts_dir=artifacts_dir,
        dataset_id="FD004",
        scaler=scaler,
        clusterer=cluster_normalizer,
        feature_cols=feat_cols,
        sensor_cols=INFORMATIVE_SENSORS_FD001,
        extra_metadata={
            "train_engines": len(train_units),
            "val_engines": len(val_units),
            "test_engines": len(rul_true),
            "window_size": 30
        }
    )

    # 7. Save processed data
    proc_dir = PROJECT_ROOT / "data" / "processed"
    train_feat.to_parquet(proc_dir / "FD004_train_features.parquet", index=False)
    val_feat.to_parquet(proc_dir / "FD004_val_features.parquet", index=False)
    test_feat.to_parquet(proc_dir / "FD004_test_features.parquet", index=False)

    np.save(proc_dir / "FD004_X_train_seq.npy", X_train_seq)
    np.save(proc_dir / "FD004_y_train_seq.npy", y_train_seq)
    np.save(proc_dir / "FD004_X_val_seq.npy", X_val_seq)
    np.save(proc_dir / "FD004_y_val_seq.npy", y_val_seq)
    np.save(proc_dir / "FD004_X_test_seq.npy", X_test_seq)
    np.save(proc_dir / "FD004_y_test_true.npy", rul_true.values)

    print("[FD004] Processing complete and persisted.")


if __name__ == "__main__":
    process_fd001()
    process_fd004()
