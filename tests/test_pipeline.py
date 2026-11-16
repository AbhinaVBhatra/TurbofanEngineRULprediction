"""
test_pipeline.py - Unit and Integration Tests for Data Ingestion, Feature Pipeline, and Model Inference
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# Fix sys.path for test runner
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from src.data import load_dataset, add_remaining_useful_life, ALL_COLS
from src.features import (
    build_engineered_features,
    RegimeClusterNormalizer,
    INFORMATIVE_SENSORS_FD001
)
from src.inference import TurbofanInferenceEngine


def test_data_ingestion_and_rul_cap():
    """Verifies that FD001 loads cleanly and RUL is properly capped at 125 cycles."""
    train_df, test_df, rul_true = load_dataset("FD001")
    assert not train_df.empty, "Train dataframe is empty"
    assert not test_df.empty, "Test dataframe is empty"
    assert len(rul_true) == 100, f"Expected 100 test RUL labels, got {len(rul_true)}"

    df_rul = add_remaining_useful_life(train_df, rul_cap=125)
    assert "RUL" in df_rul.columns
    assert df_rul["RUL"].max() <= 125, "RUL exceeded maximum cap of 125"
    assert (df_rul["RUL"] >= 0).all(), "RUL contains negative values"


def test_feature_engineering_alignment_and_zero_nans():
    """Verifies that feature engineering creates exact expected column count with zero NaNs."""
    train_df, _, _ = load_dataset("FD001")
    sample_engine = train_df[train_df["unit_nr"] == 1].copy()

    df_feat, feat_cols = build_engineered_features(sample_engine, INFORMATIVE_SENSORS_FD001, window_sizes=[5, 15])
    assert len(feat_cols) == 99, f"Expected 99 engineered features, got {len(feat_cols)}"
    assert df_feat[feat_cols].isnull().sum().sum() == 0, "Engineered features contain NaNs!"


def test_regime_cluster_normalizer():
    """Verifies that RegimeClusterNormalizer fits 6 clusters and normalizes without NaNs."""
    train_df, _, _ = load_dataset("FD004")
    sample_train = train_df[train_df["unit_nr"].isin([1, 2, 3])].copy()

    normalizer = RegimeClusterNormalizer(n_clusters=6, sensor_cols=INFORMATIVE_SENSORS_FD001, random_state=42)
    normalizer.fit(sample_train)
    transformed = normalizer.transform(sample_train)

    assert "_cluster" in transformed.columns
    assert transformed[INFORMATIVE_SENSORS_FD001].isnull().sum().sum() == 0, "Regime normalizer produced NaNs!"


def test_inference_engine_prediction():
    """Verifies that TurbofanInferenceEngine runs end-to-end inference cleanly."""
    engine = TurbofanInferenceEngine()
    test_df = engine.get_test_fleet_data()
    eng1 = test_df[test_df["unit_nr"] == 1]

    pred_res = engine.predict_engine(eng1)
    assert "predicted_rul" in pred_res
    assert 0 <= pred_res["predicted_rul"] <= 125
    assert pred_res["status"] in ["HEALTHY", "WARNING", "CRITICAL"]
    assert "shap_explanation" in pred_res
    assert len(pred_res["shap_explanation"]["top_features"]) > 0
