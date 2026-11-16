"""
features.py - Leakage-Free Preprocessing, Regime Clustering, and Feature Engineering

Key Design Principles:
1. Zero Data Leakage:
   - Engine trajectories are split by unit_nr (Engine ID), NEVER by row.
   - Scalers, normalizers, and KMeans regime clusterers are fit strictly on training engines.
   - Test sets and unseen production streams use fitted transform() only.
2. Zero Train/Serve Skew:
   - Both offline model training and live REST API inference in app/main.py import
     and execute the exact functions defined here.
3. Physical Grounding:
   - Invariant sensors in static single-regime environments (FD001) are dropped.
   - Multi-regime datasets (FD002/FD004) use KMeans (K=6) clustering on operating settings
     to normalize sensors relative to their flight condition, unmasking component wear.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 14 active prognostic sensors for FD001 (invariant sensors dropped: s_1, s_5, s_10, s_16, s_18, s_19)
INFORMATIVE_SENSORS_FD001 = [
    "s_2", "s_3", "s_4", "s_7", "s_8", "s_9",
    "s_11", "s_12", "s_13", "s_14", "s_15", "s_17", "s_20", "s_21"
]

ALL_21_SENSORS = [f"s_{i}" for i in range(1, 22)]
SETTING_COLS = ["setting_1", "setting_2", "setting_3"]


class RegimeClusterNormalizer(BaseEstimator, TransformerMixin):
    """
    Cluster-based normalizer for multi-condition datasets (FD002 / FD004).
    Fits KMeans(n_clusters=6) on operational settings [setting_1, setting_2, setting_3]
    strictly on training engines, then computes condition-specific mean and std for each sensor.
    This strips out flight condition variance, leaving pure degradation trends.
    """
    def __init__(self, n_clusters: int = 6, sensor_cols: Optional[List[str]] = None, random_state: int = 42):
        self.n_clusters = n_clusters
        self.sensor_cols = sensor_cols if sensor_cols is not None else ALL_21_SENSORS
        self.random_state = random_state
        self.setting_cols = SETTING_COLS
        self.settings_scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.cluster_stats: Dict[int, Dict[str, Tuple[float, float]]] = {}

    def fit(self, X: pd.DataFrame, y=None):
        # 1. Scale settings and fit KMeans
        scaled_settings = self.settings_scaler.fit_transform(X[self.setting_cols])
        clusters = self.kmeans.fit_predict(scaled_settings)

        # 2. Compute mean & std for each sensor within each cluster
        df_temp = X.copy()
        df_temp["_cluster"] = clusters

        self.cluster_stats = {}
        for c in range(self.n_clusters):
            c_data = df_temp[df_temp["_cluster"] == c]
            self.cluster_stats[c] = {}
            for col in self.sensor_cols:
                if len(c_data) > 1:
                    mean_val = float(c_data[col].mean())
                    std_val = float(c_data[col].std())
                    if std_val < 1e-6:
                        std_val = 1.0  # Avoid division by zero for constant channels
                else:
                    mean_val = float(X[col].mean())
                    std_val = float(X[col].std()) or 1.0
                self.cluster_stats[c][col] = (mean_val, std_val)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        scaled_settings = self.settings_scaler.transform(X[self.setting_cols])
        clusters = self.kmeans.predict(scaled_settings)
        X_out["_cluster"] = clusters
        # Ensure float dtype across normalized sensor columns to avoid downcast warnings
        X_out[self.sensor_cols] = X_out[self.sensor_cols].astype(np.float64)

        # Normalize sensors cluster by cluster
        for c in range(self.n_clusters):
            c_mask = (X_out["_cluster"] == c)
            if not c_mask.any():
                continue
            for col in self.sensor_cols:
                mean_val, std_val = self.cluster_stats[c][col]
                X_out.loc[c_mask, col] = (X_out.loc[c_mask, col] - mean_val) / std_val

        return X_out


def build_engineered_features(
    df: pd.DataFrame,
    sensor_cols: List[str],
    window_sizes: List[int] = [5, 15]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Constructs rolling statistical features for classical and tree-based models (LightGBM).
    Computes rolling mean, std, and delta grouped by engine (unit_nr).
    Zero data leakage: rolling statistics never cross engine boundaries.
    """
    # Sort and reset index to ensure contiguous row alignment and eliminate NaN misalignment
    df_feat = df.sort_values(by=["unit_nr", "time_cycles"]).reset_index(drop=True)
    feature_names = list(sensor_cols)

    grouped = df_feat.groupby("unit_nr")

    for w in window_sizes:
        for s in sensor_cols:
            mean_col = f"{s}_roll_mean_{w}"
            std_col = f"{s}_roll_std_{w}"
            diff_col = f"{s}_diff_{w}"

            # Rolling mean & std with min_periods=1 to preserve early cycles
            df_feat[mean_col] = grouped[s].rolling(window=w, min_periods=1).mean().values
            df_feat[std_col] = np.nan_to_num(grouped[s].rolling(window=w, min_periods=1).std().values, nan=0.0)
            # Trend difference over the window
            df_feat[diff_col] = np.nan_to_num(grouped[s].diff(periods=w).values, nan=0.0)

            feature_names.extend([mean_col, std_col, diff_col])

    # Also include cumulative cycles
    df_feat["cycle_norm"] = df_feat["time_cycles"] / 100.0
    feature_names.append("cycle_norm")

    return df_feat, feature_names


def generate_sequence_windows(
    df: pd.DataFrame,
    feature_cols: List[str],
    window_size: int = 30,
    target_col: Optional[str] = "RUL"
) -> Tuple[np.ndarray, Optional[np.ndarray], List[int]]:
    """
    Generates sliding 3D sequences (N, window_size, num_features) for GRU/LSTM deep learning models.
    Supports left-padding for engines with fewer cycles than window_size.

    Returns:
    --------
    X_seq: np.ndarray of shape (num_samples, window_size, len(feature_cols))
    y_seq: np.ndarray of shape (num_samples,) or None
    unit_ids: list of engine IDs corresponding to each window
    """
    X_list = []
    y_list = []
    unit_list = []

    for unit_id, group in df.groupby("unit_nr"):
        group_features = group[feature_cols].values
        group_len = len(group_features)

        if target_col is not None and target_col in group.columns:
            group_targets = group[target_col].values
        else:
            group_targets = None

        if group_len >= window_size:
            for start in range(group_len - window_size + 1):
                end = start + window_size
                X_list.append(group_features[start:end, :])
                if group_targets is not None:
                    y_list.append(group_targets[end - 1])
                unit_list.append(unit_id)
        else:
            # Pad short sequences (replicate earliest cycle at the front)
            pad_len = window_size - group_len
            first_row_repeated = np.repeat(group_features[[0], :], pad_len, axis=0)
            padded = np.vstack([first_row_repeated, group_features])
            X_list.append(padded)
            if group_targets is not None:
                y_list.append(group_targets[-1])
            unit_list.append(unit_id)

    X_seq = np.array(X_list, dtype=np.float32)
    y_seq = np.array(y_list, dtype=np.float32) if y_list else None
    return X_seq, y_seq, unit_list


def extract_last_sequence_for_test(
    test_df: pd.DataFrame,
    feature_cols: List[str],
    window_size: int = 30
) -> Tuple[np.ndarray, List[int]]:
    """
    Extracts exactly one sequence window of length `window_size` representing the final
    observed operating state of each test engine. Used for end-of-history RUL scoring.
    """
    X_list = []
    unit_ids = []

    for unit_id, group in test_df.groupby("unit_nr"):
        group_features = group[feature_cols].values
        group_len = len(group_features)

        if group_len >= window_size:
            # Take the most recent window
            seq = group_features[-window_size:, :]
        else:
            # Pad beginning if history is shorter than window_size
            pad_len = window_size - group_len
            first_row_repeated = np.repeat(group_features[[0], :], pad_len, axis=0)
            seq = np.vstack([first_row_repeated, group_features])

        X_list.append(seq)
        unit_ids.append(unit_id)

    return np.array(X_list, dtype=np.float32), unit_ids


def split_engines_train_val(
    train_df: pd.DataFrame,
    val_ratio: float = 0.20,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, List[int], List[int]]:
    """
    Partitions the training dataframe into train and validation sets STRICTLY by engine ID (unit_nr).
    Guarantees that no engine appears in both partitions, preventing temporal autocorrelation leakage.
    """
    unique_units = np.array(train_df["unit_nr"].unique())
    train_units, val_units = train_test_split(unique_units, test_size=val_ratio, random_state=random_state)

    train_split = train_df[train_df["unit_nr"].isin(train_units)].copy()
    val_split = train_df[train_df["unit_nr"].isin(val_units)].copy()

    return train_split, val_split, train_units.tolist(), val_units.tolist()


def save_pipeline_artifacts(
    artifacts_dir: Path,
    dataset_id: str,
    scaler: Any,
    feature_cols: List[str],
    sensor_cols: List[str],
    clusterer: Optional[Any] = None,
    extra_metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Serializes fitted scalers, clusterers, and feature schemas to artifacts/
    for guaranteed zero-skew inference in the web application.
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    scaler_path = artifacts_dir / f"{dataset_id}_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"[ARTIFACTS] Saved scaler: {scaler_path.name}")

    if clusterer is not None:
        clusterer_path = artifacts_dir / f"{dataset_id}_clusterer.pkl"
        joblib.dump(clusterer, clusterer_path)
        print(f"[ARTIFACTS] Saved clusterer: {clusterer_path.name}")

    config = {
        "dataset_id": dataset_id,
        "sensor_cols": sensor_cols,
        "feature_cols": feature_cols,
        "num_features": len(feature_cols),
        "window_size": 30,
        "rul_cap": 125
    }
    if extra_metadata:
        config.update(extra_metadata)

    config_path = artifacts_dir / f"{dataset_id}_feature_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"[ARTIFACTS] Saved feature configuration: {config_path.name}")
