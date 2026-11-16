"""
01_eda.py - Exploratory Data Analysis & Problem Framing
NASA C-MAPSS Turbofan Degradation (FD001 & FD004)

Business Context:
- Unplanned downtime cost in commercial/industrial turbofans is massive ($100k-$1M+ per catastrophic event).
- Routine premature replacement wastes remaining component life (over-maintenance).
- Objective: Predict Remaining Useful Life (RUL) with high precision to trigger condition-based maintenance.
- Key KPIs:
  1. Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) in operational cycles.
  2. NASA Asymmetric Score Function (asymmetric penalty heavily penalizing late predictions).
  3. Cost Avoidance Ratio / Financial Savings vs Run-to-Failure and Fixed-Interval Overhaul.

Simulated Data Notice:
Data originates from NASA C-MAPSS simulation environment (thermodynamic simulation model),
NOT commercial airline telemetry.
"""

import os
import sys
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

# Fix OpenMP collision
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data import load_dataset, add_remaining_useful_life, SENSOR_COLS, SETTING_COLS, SENSOR_DESCRIPTIONS

# Set plot styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["figure.dpi"] = 300

FIG_DIR = PROJECT_ROOT / "results" / "figures"
TAB_DIR = PROJECT_ROOT / "results" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)


def run_eda():
    print("[EDA] Loading FD001 and FD004 datasets...")
    train_fd1, test_fd1, rul_test_fd1 = load_dataset("FD001")
    train_fd4, test_fd4, rul_test_fd4 = load_dataset("FD004")

    # Add RUL labels with standard piecewise cap of 125 cycles
    train_fd1 = add_remaining_useful_life(train_fd1, rul_cap=125)
    train_fd4 = add_remaining_useful_life(train_fd4, rul_cap=125)

    print(f"[EDA] FD001 Train Engines: {train_fd1['unit_nr'].nunique()}, Records: {len(train_fd1)}")
    print(f"[EDA] FD004 Train Engines: {train_fd4['unit_nr'].nunique()}, Records: {len(train_fd4)}")

    # ==========================================
    # 1. Engine Lifetime Distribution Analysis
    # ==========================================
    lifetimes_fd1 = train_fd1.groupby("unit_nr")["max_cycle"].first()
    lifetimes_fd4 = train_fd4.groupby("unit_nr")["max_cycle"].first()

    stats_summary = {
        "FD001": {
            "num_engines": int(len(lifetimes_fd1)),
            "min_cycles": int(lifetimes_fd1.min()),
            "max_cycles": int(lifetimes_fd1.max()),
            "mean_cycles": float(round(lifetimes_fd1.mean(), 2)),
            "median_cycles": float(round(lifetimes_fd1.median(), 2)),
            "std_cycles": float(round(lifetimes_fd1.std(), 2)),
            "iqr_cycles": float(round(lifetimes_fd1.quantile(0.75) - lifetimes_fd1.quantile(0.25), 2))
        },
        "FD004": {
            "num_engines": int(len(lifetimes_fd4)),
            "min_cycles": int(lifetimes_fd4.min()),
            "max_cycles": int(lifetimes_fd4.max()),
            "mean_cycles": float(round(lifetimes_fd4.mean(), 2)),
            "median_cycles": float(round(lifetimes_fd4.median(), 2)),
            "std_cycles": float(round(lifetimes_fd4.std(), 2)),
            "iqr_cycles": float(round(lifetimes_fd4.quantile(0.75) - lifetimes_fd4.quantile(0.25), 2))
        }
    }

    # Save summary table
    df_stats = pd.DataFrame(stats_summary).T
    df_stats.to_csv(TAB_DIR / "lifetime_statistics.csv")
    print("\n[EDA] Engine Lifetime Statistics:")
    print(df_stats)

    # Plot 1: Lifetime distribution comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    sns.histplot(lifetimes_fd1, kde=True, ax=axes[0], color="#1f77b4", bins=20)
    axes[0].set_title(f"FD001 Engine Lifetimes (N={len(lifetimes_fd1)})\nMean={stats_summary['FD001']['mean_cycles']} cycles", fontsize=12)
    axes[0].set_xlabel("Operational Cycles until Failure")
    axes[0].set_ylabel("Engine Count")
    axes[0].axvline(lifetimes_fd1.mean(), color="red", linestyle="--", label=f"Mean ({stats_summary['FD001']['mean_cycles']})")
    axes[0].axvline(lifetimes_fd1.median(), color="green", linestyle=":", label=f"Median ({stats_summary['FD001']['median_cycles']})")
    axes[0].legend()

    sns.histplot(lifetimes_fd4, kde=True, ax=axes[1], color="#ff7f0e", bins=25)
    axes[1].set_title(f"FD004 Engine Lifetimes (N={len(lifetimes_fd4)})\nMean={stats_summary['FD004']['mean_cycles']} cycles", fontsize=12)
    axes[1].set_xlabel("Operational Cycles until Failure")
    axes[1].axvline(lifetimes_fd4.mean(), color="red", linestyle="--", label=f"Mean ({stats_summary['FD004']['mean_cycles']})")
    axes[1].axvline(lifetimes_fd4.median(), color="green", linestyle=":", label=f"Median ({stats_summary['FD004']['median_cycles']})")
    axes[1].legend()

    plt.tight_layout()
    fig_path1 = FIG_DIR / "01_lifetime_distribution.png"
    plt.savefig(fig_path1, dpi=300)
    plt.close()
    print(f"[EDA] Saved: {fig_path1.name}")

    # ==========================================
    # 2. Sensor Variance & Invariant Sensor Detection
    # ==========================================
    var_fd1 = train_fd1[SENSOR_COLS].std()
    var_fd4 = train_fd4[SENSOR_COLS].std()

    var_df = pd.DataFrame({
        "Sensor": SENSOR_COLS,
        "Description": [SENSOR_DESCRIPTIONS[s] for s in SENSOR_COLS],
        "FD001_Std": var_fd1.values,
        "FD004_Std": var_fd4.values
    })
    var_df["FD001_Is_Constant"] = var_df["FD001_Std"] < 1e-4
    var_df["FD004_Is_Constant"] = var_df["FD004_Std"] < 1e-4

    var_df.to_csv(TAB_DIR / "sensor_variance_analysis.csv", index=False)

    constant_sensors_fd1 = var_df[var_df["FD001_Is_Constant"]]["Sensor"].tolist()
    print(f"\n[EDA] Invariant (constant) sensors in FD001 (Std < 1e-4): {constant_sensors_fd1}")
    print(f"[EDA] Invariant sensors in FD004: {var_df[var_df['FD004_Is_Constant']]['Sensor'].tolist()}")

    # ==========================================
    # 3. Sensor Correlation Heatmap
    # ==========================================
    # Informative sensors in FD001
    active_sensors_fd1 = [s for s in SENSOR_COLS if s not in constant_sensors_fd1]

    corr_df = train_fd1[active_sensors_fd1 + ["rul_unbounded", "RUL"]].corr()
    corr_df.to_csv(TAB_DIR / "sensor_rul_correlations.csv")

    plt.figure(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr_df, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, annot=True, fmt=".2f", annot_kws={"size": 7})
    plt.title("FD001 Correlation Heatmap: Active Sensors vs RUL (Unbounded & Capped)", fontsize=14, pad=15)
    plt.tight_layout()
    fig_path3 = FIG_DIR / "03_sensor_correlation_heatmap.png"
    plt.savefig(fig_path3, dpi=300)
    plt.close()
    print(f"[EDA] Saved: {fig_path3.name}")

    # ==========================================
    # 4. Degradation Curves & Sensor Trends vs Cycle
    # ==========================================
    # Select 6 key prognostic sensors: s_2, s_3, s_4, s_7, s_11, s_12
    key_sensors = ["s_2", "s_3", "s_4", "s_7", "s_11", "s_12"]
    sample_engines = [1, 2, 3, 4, 5]

    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=False)
    axes = axes.flatten()

    for idx, s in enumerate(key_sensors):
        ax = axes[idx]
        for engine_id in sample_engines:
            engine_data = train_fd1[train_fd1["unit_nr"] == engine_id]
            ax.plot(engine_data["time_cycles"], engine_data[s], label=f"Engine #{engine_id}", alpha=0.85, linewidth=1.4)
        ax.set_title(f"{s}: {SENSOR_DESCRIPTIONS[s]}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Operational Cycles")
        ax.set_ylabel("Sensor Reading")
        if idx == 0:
            ax.legend(loc="upper left", fontsize=8)

    plt.suptitle("FD001 Degradation Trajectories Across Sample Engines (Run-to-Failure)", fontsize=14, y=0.99)
    plt.tight_layout()
    fig_path2 = FIG_DIR / "02_sensor_degradation_curves.png"
    plt.savefig(fig_path2, dpi=300)
    plt.close()
    print(f"[EDA] Saved: {fig_path2.name}")

    # ==========================================
    # 5. FD001 vs FD004 Operating Conditions & Degradation Obfuscation
    # ==========================================
    # Show why FD004 is harder: 6 operating conditions cause sensor measurements to jump
    # between discrete levels depending on altitude/Mach/throttle, masking degradation!
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Setting scatter: FD001 vs FD004
    axes[0, 0].scatter(train_fd1["setting_1"], train_fd1["setting_2"], c="#1f77b4", alpha=0.3, s=10)
    axes[0, 0].set_title("FD001 Operating Conditions: Single Regime (Sea Level)", fontsize=12)
    axes[0, 0].set_xlabel("Setting 1 (Altitude / Mach)")
    axes[0, 0].set_ylabel("Setting 2 (Throttle Resolver Angle)")

    axes[0, 1].scatter(train_fd4["setting_1"], train_fd4["setting_2"], c="#d62728", alpha=0.3, s=10)
    axes[0, 1].set_title("FD004 Operating Conditions: 6 Distinct Flight Regimes", fontsize=12)
    axes[0, 1].set_xlabel("Setting 1 (Altitude / Mach)")
    axes[0, 1].set_ylabel("Setting 2 (Throttle Resolver Angle)")

    # Raw Sensor comparison on Engine #1 of each
    eng1_fd1 = train_fd1[train_fd1["unit_nr"] == 1]
    eng1_fd4 = train_fd4[train_fd4["unit_nr"] == 1]

    axes[1, 0].plot(eng1_fd1["time_cycles"], eng1_fd1["s_2"], color="#1f77b4", linewidth=1.5)
    axes[1, 0].set_title("FD001 Engine #1: Sensor s_2 (Smooth Monotonic Upward Degradation)", fontsize=12)
    axes[1, 0].set_xlabel("Cycles")
    axes[1, 0].set_ylabel("s_2 LPC Outlet Temp (°R)")

    axes[1, 1].plot(eng1_fd4["time_cycles"], eng1_fd4["s_2"], color="#d62728", linewidth=0.9)
    axes[1, 1].set_title("FD004 Engine #1: Raw s_2 (Degradation Obfuscated by 6 Regimes!)", fontsize=12)
    axes[1, 1].set_xlabel("Cycles")
    axes[1, 1].set_ylabel("s_2 LPC Outlet Temp (°R)")

    plt.tight_layout()
    fig_path4 = FIG_DIR / "04_fd001_vs_fd004_regimes.png"
    plt.savefig(fig_path4, dpi=300)
    plt.close()
    print(f"[EDA] Saved: {fig_path4.name}")

    # ==========================================
    # 6. Piecewise-Linear RUL Justification Plot
    # ==========================================
    plt.figure(figsize=(9, 4.5))
    sample_engine = train_fd1[train_fd1["unit_nr"] == 1]
    plt.plot(sample_engine["time_cycles"], sample_engine["rul_unbounded"], label="Linear Unbounded RUL (Unrealistic pristine degradation)", color="gray", linestyle="--", linewidth=1.5)
    plt.plot(sample_engine["time_cycles"], sample_engine["RUL"], label="Piecewise-Linear Capped RUL ($RUL_{cap}=125$ cycles)", color="#2ca02c", linewidth=2.5)
    plt.axhline(125, color="orange", linestyle=":", label="Cap Threshold (125 cycles)")
    plt.title("Justification of Piecewise-Linear RUL Target Function (Engine #1, Failure at Cycle 192)", fontsize=12)
    plt.xlabel("Operational Cycle")
    plt.ylabel("RUL Target Label")
    plt.legend()
    plt.tight_layout()
    fig_path5 = FIG_DIR / "05_piecewise_rul_justification.png"
    plt.savefig(fig_path5, dpi=300)
    plt.close()
    print(f"[EDA] Saved: {fig_path5.name}")

    print("\n[EDA] Complete. All statistical tables and figures saved.")


if __name__ == "__main__":
    run_eda()
