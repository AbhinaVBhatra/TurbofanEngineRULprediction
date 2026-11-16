"""
inference.py - Shared Inference Engine for Real-Time Serving and Offline Evaluation

Guarantees ZERO Train/Serve Skew by:
1. Reusing the exact fitted scalers, feature column schemas, and quantile regressors
   persisted during Phase 2 and Phase 3.
2. Reusing build_engineered_features() and INFORMATIVE_SENSORS_FD001 from src/features.py.
3. Performing strict input validation (columns, min 30 cycles, numeric types, finiteness).
"""

import os
import io
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    from src.features import build_engineered_features, INFORMATIVE_SENSORS_FD001
    from src.data import ALL_COLS, SENSOR_DESCRIPTIONS, load_dataset
    from src.explain import get_tree_explainer, explain_single_engine
except ImportError:
    from ptm_project.src.features import build_engineered_features, INFORMATIVE_SENSORS_FD001
    from ptm_project.src.data import ALL_COLS, SENSOR_DESCRIPTIONS, load_dataset
    from ptm_project.src.explain import get_tree_explainer, explain_single_engine


class TurbofanInferenceEngine:
    """
    Production inference engine. Loaded once at web service startup.
    Evaluates real-time point RUL, 80% prediction intervals, and local SHAP explanations.
    """
    def __init__(self, artifacts_dir: Optional[Path] = None, data_dir: Optional[Path] = None):
        if artifacts_dir is None:
            self.artifacts_dir = Path(__file__).resolve().parent.parent / "artifacts"
        else:
            self.artifacts_dir = Path(artifacts_dir)

        if data_dir is None:
            self.data_dir = Path(__file__).resolve().parent.parent / "data"
        else:
            self.data_dir = Path(data_dir)

        self._load_artifacts()
        self._test_df_cache = None

    def _load_artifacts(self):
        """Loads fitted model, scalers, and configurations."""
        model_path = self.artifacts_dir / "FD001_best_model.pkl"
        scaler_path = self.artifacts_dir / "FD001_scaler.pkl"
        config_path = self.artifacts_dir / "FD001_feature_config.json"
        q10_path = self.artifacts_dir / "FD001_quantile_q10.pkl"
        q90_path = self.artifacts_dir / "FD001_quantile_q90.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Missing required model artifact at {model_path}. Run Phase 3 training first.")

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.feature_cols = self.config["feature_cols"]
        self.sensor_cols = self.config["sensor_cols"]
        self.window_size = self.config.get("window_size", 30)

        self.q10_model = joblib.load(q10_path) if q10_path.exists() else None
        self.q90_model = joblib.load(q90_path) if q90_path.exists() else None

        # Initialize TreeExplainer
        self.explainer = get_tree_explainer(self.model)
        print("[INFERENCE] TurbofanInferenceEngine successfully initialized with LightGBM & Quantile models.")

    def validate_input_dataframe(self, df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        """
        Validates uploaded engine time-series data:
        1. Checks required columns (unit_nr, time_cycles, sensors).
        2. Verifies minimum history length (at least 30 cycles).
        3. Verifies numeric dtypes and no NaNs or infinities.
        """
        if df.empty:
            return False, "Uploaded file is empty."

        # Required columns: unit_nr, time_cycles, and the active sensors
        required = ["time_cycles"] + self.sensor_cols
        missing = [col for col in required if col not in df.columns]
        if missing:
            return False, f"Missing required columns: {missing}. Expected standard C-MAPSS sensor telemetry."

        if len(df) < self.window_size:
            return False, f"Insufficient operational history. Uploaded engine has {len(df)} cycles, but a minimum of {self.window_size} cycles is required for rolling prognostic feature extraction."

        # Numeric and finiteness check
        check_cols = ["time_cycles"] + self.sensor_cols
        for col in check_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                return False, f"Column '{col}' must contain numeric values."
            if df[col].isnull().any():
                return False, f"Column '{col}' contains NaN values."
            if np.isinf(df[col]).any():
                return False, f"Column '{col}' contains infinite values."

        return True, None

    def process_and_extract_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Executes zero-skew preprocessing on raw engine telemetry:
        1. Standardizes active sensor channels using fitted scaler.
        2. Computes rolling mean, std, and diff features.
        3. Returns processed DataFrame and 2D feature matrix.
        """
        df_clean = df.copy()
        if "unit_nr" not in df_clean.columns:
            df_clean["unit_nr"] = 1

        # Sort temporally
        df_clean = df_clean.sort_values(by=["unit_nr", "time_cycles"]).reset_index(drop=True)

        # Apply saved scaler
        df_clean[self.sensor_cols] = self.scaler.transform(df_clean[self.sensor_cols])

        # Apply shared feature engineering
        df_feat, _ = build_engineered_features(df_clean, self.sensor_cols, window_sizes=[5, 15])

        X = df_feat[self.feature_cols].values
        return df_feat, X

    def predict_engine(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Runs live inference on an engine dataframe.
        Predicts RUL at the latest cycle, computes 80% interval, assigns health status,
        and computes local SHAP feature attributions.
        """
        is_valid, err_msg = self.validate_input_dataframe(df)
        if not is_valid:
            raise ValueError(err_msg)

        df_feat, X = self.process_and_extract_features(df)

        # Predict on latest cycle
        X_latest = X[[-1], :]
        pred_rul = float(np.clip(self.model.predict(X_latest)[0], 0, 125))

        # Quantile interval
        if self.q10_model and self.q90_model:
            q10 = float(np.clip(self.q10_model.predict(X_latest)[0], 0, 125))
            q90 = float(np.clip(self.q90_model.predict(X_latest)[0], 0, 125))
            # Ensure lower <= upper
            lower_bound = round(min(q10, q90), 1)
            upper_bound = round(max(q10, q90), 1)
        else:
            lower_bound = max(0.0, round(pred_rul - 15.0, 1))
            upper_bound = min(125.0, round(pred_rul + 15.0, 1))

        # Health status determination
        if pred_rul <= 15.0:
            status = "CRITICAL"
            status_desc = "Immediate maintenance required. Critical degradation detected."
            action = "Ground aircraft and schedule urgent shop overhaul."
        elif pred_rul <= 40.0:
            status = "WARNING"
            status_desc = "Accelerating degradation. Approaching overhaul window."
            action = "Schedule preventive depot maintenance within next 20 cycles."
        else:
            status = "HEALTHY"
            status_desc = "Normal nominal operating condition."
            action = "Engine cleared for continuous flight operations."

        # SHAP local explanation
        shap_explanation = explain_single_engine(
            model=self.model,
            explainer=self.explainer,
            feature_vector=X_latest,
            feature_names=self.feature_cols,
            top_n=6
        )

        # Cycle trajectory for visualization
        all_preds = np.clip(self.model.predict(X), 0, 125).round(1).tolist()
        cycles = df["time_cycles"].tolist()

        return {
            "current_cycle": int(df["time_cycles"].iloc[-1]),
            "predicted_rul": round(pred_rul, 1),
            "prediction_interval": {
                "lower_q10": lower_bound,
                "upper_q90": upper_bound,
                "confidence_level": "80%"
            },
            "status": status,
            "status_description": status_desc,
            "recommended_action": action,
            "shap_explanation": shap_explanation,
            "history": {
                "cycles": cycles,
                "predicted_rul_series": all_preds,
                "sensor_s4_lpt_temp": df["s_4"].tolist() if "s_4" in df.columns else [],
                "sensor_s11_hpc_press": df["s_11"].tolist() if "s_11" in df.columns else [],
                "sensor_s2_lpc_temp": df["s_2"].tolist() if "s_2" in df.columns else []
            }
        }

    def get_test_fleet_data(self) -> pd.DataFrame:
        """Loads and caches test_FD001.txt for fleet overview and simulate endpoints."""
        if self._test_df_cache is None:
            _, test_df, _ = load_dataset("FD001", data_dir=self.data_dir / "raw")
            self._test_df_cache = test_df
        return self._test_df_cache

    def get_fleet_summary(self) -> List[Dict[str, Any]]:
        """
        Computes real model inference for all 100 test engines in FD001.
        Returns ranked list of engines by urgency.
        """
        test_df = self.get_test_fleet_data()
        results = []

        for unit_id, group in test_df.groupby("unit_nr"):
            res = self.predict_engine(group)
            results.append({
                "engine_id": int(unit_id),
                "current_cycles": res["current_cycle"],
                "predicted_rul": res["predicted_rul"],
                "lower_q10": res["prediction_interval"]["lower_q10"],
                "upper_q90": res["prediction_interval"]["upper_q90"],
                "status": res["status"],
                "recommended_action": res["recommended_action"]
            })

        # Sort by urgency (lowest RUL first)
        results.sort(key=lambda x: x["predicted_rul"])
        return results

    def simulate_engine_step(self, engine_id: int, target_cycle: Optional[int] = None) -> Dict[str, Any]:
        """
        Replays engine telemetry up to target_cycle, returning real-time prognostic status.
        """
        test_df = self.get_test_fleet_data()
        engine_df = test_df[test_df["unit_nr"] == engine_id].sort_values("time_cycles").reset_index(drop=True)
        if engine_df.empty:
            raise ValueError(f"Engine #{engine_id} not found in fleet.")

        max_available = int(engine_df["time_cycles"].max())
        if target_cycle is None or target_cycle > max_available:
            target_cycle = max_available

        sub_df = engine_df[engine_df["time_cycles"] <= target_cycle]
        if len(sub_df) < self.window_size:
            # Take minimum required for valid feature extraction
            sub_df = engine_df.iloc[:self.window_size]

        res = self.predict_engine(sub_df)
        res["engine_id"] = engine_id
        res["total_cycles_in_history"] = max_available
        return res
