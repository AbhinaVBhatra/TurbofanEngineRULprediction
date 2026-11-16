"""
07_inference_benchmark.py - Computational Profiling: Latency, Throughput & Model Storage

Benchmarks CPU inference performance and storage footprints for:
1. Linear Regression (Ridge)
2. Random Forest Regressor
3. LightGBM Regressor (Production Model)
4. PyTorch GRU (Sequence Model)

Metrics:
- Single-engine inference latency: Mean, Median, 95th percentile, Min, Max (milliseconds)
- Inference throughput (engines / second)
- Model artifact file size (MB)
- Parameter / tree complexity

Outputs:
- results/paper/07_inference_benchmark.json
- results/paper/07_inference_benchmark.csv
- Appends to paper/number_provenance.csv
"""

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from src.models import (
    build_linear_regression,
    build_random_forest,
    build_lightgbm,
    PyTorchGRUTrainer,
    TurbofanGRU
)
from experiments.provenance_tracker import log_number

RESULTS_DIR = PROJECT_ROOT / "results" / "paper"
PROC_DIR = PROJECT_ROOT / "data" / "processed"
ART_DIR = PROJECT_ROOT / "artifacts"


def run_benchmark():
    print("=" * 70)
    print("EXPERIMENT 7: Computational Profiling (Latency, Throughput, Size)")
    print("=" * 70)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Load sample inputs
    with open(ART_DIR / "FD001_feature_config.json") as f:
        cfg = json.load(f)
    feature_cols = cfg["feature_cols"]

    test_feat = pd.read_parquet(PROC_DIR / "FD001_test_features.parquet")
    test_last = test_feat.groupby("unit_nr").last().reset_index()
    X_single_tab = test_last[feature_cols].values[0:1]  # shape: (1, 99)

    X_test_seq = np.load(PROC_DIR / "FD001_X_test_seq.npy")
    X_single_seq = X_test_seq[0:1]  # shape: (1, 30, 14)

    # 2. Models to benchmark
    # Train or load Ridge and RF
    train_feat = pd.read_parquet(PROC_DIR / "FD001_train_features.parquet")
    X_tr = train_feat[feature_cols].values
    y_tr = train_feat["RUL"].values

    ridge = build_linear_regression(alpha=10.0)
    ridge.fit(X_tr, y_tr)
    ridge_path = ART_DIR / "FD001_ridge_benchmark.pkl"
    joblib.dump(ridge, ridge_path)

    rf = build_random_forest(n_estimators=100, max_depth=12)
    rf.fit(X_tr, y_tr)
    rf_path = ART_DIR / "FD001_rf_benchmark.pkl"
    joblib.dump(rf, rf_path)

    # Load LightGBM
    lgbm_path = ART_DIR / "FD001_best_model.pkl"
    lgbm = joblib.load(lgbm_path)

    # Load PyTorch GRU
    gru_path = ART_DIR / "FD001_gru_model.pt"
    gru_trainer = PyTorchGRUTrainer(input_dim=14, hidden_dim=64, num_layers=2)
    if gru_path.exists():
        gru_trainer.load(gru_path)
    gru_model = gru_trainer.model
    gru_model.eval()

    models_info = [
        ("Linear Regression (Ridge)", ridge, "tabular", ridge_path, 1),
        ("Random Forest (100 trees)", rf, "tabular", rf_path, 100),
        ("LightGBM (250 trees)", lgbm, "tabular", lgbm_path, 250),
        ("PyTorch GRU (2-layer)", gru_model, "sequence", gru_path, sum(p.numel() for p in gru_model.parameters()))
    ]

    benchmark_records = []
    WARMUP_RUNS = 25
    BENCHMARK_RUNS = 200

    print(f"\nBenchmarking CPU Single-Engine Latency (Warmup={WARMUP_RUNS}, Trials={BENCHMARK_RUNS})...")

    for model_name, model_obj, input_type, file_path, complexity in models_info:
        file_size_mb = round(file_path.stat().st_size / (1024 * 1024), 3)

        # Warmup
        if input_type == "tabular":
            for _ in range(WARMUP_RUNS):
                _ = model_obj.predict(X_single_tab)
        else:
            t_input = torch.from_numpy(X_single_seq)
            with torch.no_grad():
                for _ in range(WARMUP_RUNS):
                    _ = model_obj(t_input)

        # Timed benchmark
        latencies_ms = []
        if input_type == "tabular":
            for _ in range(BENCHMARK_RUNS):
                t0 = time.perf_counter()
                _ = model_obj.predict(X_single_tab)
                t1 = time.perf_counter()
                latencies_ms.append((t1 - t0) * 1000.0)
        else:
            t_input = torch.from_numpy(X_single_seq)
            with torch.no_grad():
                for _ in range(BENCHMARK_RUNS):
                    t0 = time.perf_counter()
                    _ = model_obj(t_input)
                    t1 = time.perf_counter()
                    latencies_ms.append((t1 - t0) * 1000.0)

        latencies_ms = np.array(latencies_ms)
        mean_lat = float(np.mean(latencies_ms))
        median_lat = float(np.median(latencies_ms))
        p95_lat = float(np.percentile(latencies_ms, 95))
        min_lat = float(np.min(latencies_ms))
        max_lat = float(np.max(latencies_ms))
        throughput = float(1000.0 / mean_lat) if mean_lat > 0 else 0.0

        record = {
            "model": model_name,
            "architecture_type": input_type,
            "model_size_mb": file_size_mb,
            "complexity_metric": complexity,
            "latency_mean_ms": round(mean_lat, 3),
            "latency_median_ms": round(median_lat, 3),
            "latency_p95_ms": round(p95_lat, 3),
            "latency_min_ms": round(min_lat, 3),
            "latency_max_ms": round(max_lat, 3),
            "throughput_engines_per_sec": round(throughput, 1)
        }
        benchmark_records.append(record)
        print(f"  {model_name:28s} | Latency: {mean_lat:6.3f} ms (p95: {p95_lat:6.3f} ms) | Size: {file_size_mb:5.2f} MB | Throughput: {throughput:6.1f} eng/s")

        # Provenance logging
        safe_mod = model_name.split("(")[0].strip().replace(" ", "_")
        log_number(f"Profile_{safe_mod}_Latency_Mean", "FD001", model_name, "Latency_Mean", f"{mean_lat:.3f}", "ms", "experiments/07_inference_benchmark.py", "results/paper/07_inference_benchmark.json", "Single-engine CPU latency mean")
        log_number(f"Profile_{safe_mod}_Latency_p95", "FD001", model_name, "Latency_p95", f"{p95_lat:.3f}", "ms", "experiments/07_inference_benchmark.py", "results/paper/07_inference_benchmark.json", "Single-engine CPU latency 95th percentile")
        log_number(f"Profile_{safe_mod}_FileSize", "FD001", model_name, "FileSize_MB", f"{file_size_mb:.3f}", "MB", "experiments/07_inference_benchmark.py", "results/paper/07_inference_benchmark.json", "Artifact disk size")
        log_number(f"Profile_{safe_mod}_Throughput", "FD001", model_name, "Throughput", f"{throughput:.1f}", "engines/sec", "experiments/07_inference_benchmark.py", "results/paper/07_inference_benchmark.json", "Inference throughput on CPU")

    # Clean up temporary baseline artifacts
    if ridge_path.exists():
        ridge_path.unlink()
    if rf_path.exists():
        rf_path.unlink()

    df_bench = pd.DataFrame(benchmark_records)
    csv_path = RESULTS_DIR / "07_inference_benchmark.csv"
    df_bench.to_csv(csv_path, index=False)

    json_path = RESULTS_DIR / "07_inference_benchmark.json"
    with open(json_path, "w") as f:
        json.dump(benchmark_records, f, indent=2)

    print(f"\n[SAVED] Benchmark outputs saved to {csv_path.name} and {json_path.name}")


if __name__ == "__main__":
    run_benchmark()
