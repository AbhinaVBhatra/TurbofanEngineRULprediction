"""
05_decision_cost.py - Predictive Maintenance Decision Rule, Threshold Sweep & Financial Cost Simulation

Evaluates:
1. Run-to-Failure Policy ($200,000 per catastrophic breakdown)
2. Fixed-Interval Policy (overhaul scheduled at fixed cycle interval)
3. Predictive Maintenance Policy (overhaul triggered when model RUL <= threshold T)

Quantifies:
- Optimal threshold T* minimizing total fleet operational expenditure
- Missed failures (False Negatives) vs Wasted Life cycles trade-off
- Net dollar cost savings and Return on Investment (ROI)
- Break-even sensitivity analysis across unplanned failure cost ratios
"""

import os
import sys
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.cost import TurbofanCostSimulator, calculate_cost_savings

FIG_DIR = PROJECT_ROOT / "results" / "figures"
TAB_DIR = PROJECT_ROOT / "results" / "tables"
ART_DIR = PROJECT_ROOT / "artifacts"
PROC_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"


def main():
    print("\n=======================================================")
    print("[PHASE 5] Prediction to Decision & Financial Cost Simulation")
    print("=======================================================")

    # 1. Load actual test RULs and LightGBM model predictions
    y_test_true = np.load(PROC_DIR / "FD001_y_test_true.npy")

    best_model = joblib.load(ART_DIR / "FD001_best_model.pkl")
    with open(ART_DIR / "FD001_feature_config.json", "r") as f:
        feat_config = json.load(f)
    feature_cols = feat_config["feature_cols"]

    test_feat = pd.read_parquet(PROC_DIR / "FD001_test_features.parquet")
    X_test_last = test_feat.groupby("unit_nr").last()[feature_cols].values
    y_pred_best = np.clip(best_model.predict(X_test_last), 0, 125)

    num_test_engines = len(y_test_true)

    # Load complete engine trajectories for continuous lifecycle simulation
    val_feat = pd.read_parquet(PROC_DIR / "FD001_val_features.parquet")
    train_feat = pd.read_parquet(PROC_DIR / "FD001_train_features.parquet")
    all_trajectories = pd.concat([train_feat, val_feat], ignore_index=True)
    all_lifetimes = all_trajectories.groupby("unit_nr")["time_cycles"].max().values

    # 2. Initialize Simulator with clearly labeled assumptions
    simulator = TurbofanCostSimulator(
        cost_unplanned=200000.0,     # $200,000 catastrophic failure penalty
        cost_planned=40000.0,        # $40,000 depot overhaul
        cost_waste_per_cycle=200.0,  # $200/cycle component life value
        default_threshold=20.0       # baseline safety trigger
    )

    print("\nCost Model Parameters (Editable):")
    print(f"  Unplanned Failure Cost (C_unplanned): ${simulator.cost_unplanned:,.2f}")
    print(f"  Planned Overhaul Cost (C_planned):   ${simulator.cost_planned:,.2f}")
    print(f"  Wasted Life Cost (C_waste):          ${simulator.cost_waste_per_cycle:,.2f} per cycle")

    # =========================================================================
    # A. Complete Lifecycle Fleet Simulation (100 Turbofan Engines run to failure)
    # =========================================================================
    print("\n-----------------------------------------------------------")
    print("A. Complete Lifecycle Simulation (Fleet of 100 Run-to-Failure Engines)")
    print("-----------------------------------------------------------")

    # Policy 1: Run-to-Failure (Corrective Maintenance)
    rtf_lifecycle = simulator.evaluate_run_to_failure(num_engines=100)
    print(f"Policy 1 (Run-to-Failure): Total Cost = ${rtf_lifecycle['total_cost']:,.2f} | Unplanned Failures = {rtf_lifecycle['unplanned_failures']}")

    # Policy 2: Fixed-Interval Maintenance (Sweep intervals 120-220 cycles)
    fixed_intervals = list(range(120, 240, 10))
    fixed_results = [simulator.evaluate_fixed_schedule(all_lifetimes, interval) for interval in fixed_intervals]
    df_fixed_sweep = pd.DataFrame(fixed_results)
    best_fixed_idx = df_fixed_sweep["total_cost"].idxmin()
    best_fixed = fixed_results[best_fixed_idx]
    best_interval = best_fixed["interval_cycles"]
    print(f"Policy 2 (Optimal Fixed-Schedule at {best_interval} cycles): Total Cost = ${best_fixed['total_cost']:,.2f} | Unplanned Failures = {best_fixed['unplanned_failures']}")

    # Policy 3: Predictive Maintenance (Simulate threshold sweep on full trajectories)
    thresholds = [10, 15, 20, 25, 30, 35, 40]
    lifecycle_sweep = []
    for t in thresholds:
        res = simulator.simulate_full_lifecycle(val_feat, best_model, feature_cols, threshold=t)
        # Scale to 100 engines for identical fleet comparison
        scale_factor = 100.0 / res["num_engines"]
        lifecycle_sweep.append({
            "threshold": t,
            "planned_overhauls": int(res["planned_overhauls"] * scale_factor),
            "unplanned_failures": int(res["unplanned_failures"] * scale_factor),
            "pct_unplanned_prevented": res["pct_unplanned_prevented"],
            "avg_wasted_cycles": res["avg_wasted_cycles_per_engine"],
            "avg_lead_time": res["avg_advance_warning_lead_time"],
            "planned_cost": res["planned_cost"] * scale_factor,
            "unplanned_cost": res["unplanned_cost"] * scale_factor,
            "wasted_life_cost": res["wasted_life_cost"] * scale_factor,
            "total_cost": res["total_cost"] * scale_factor
        })

    df_lifecycle_sweep = pd.DataFrame(lifecycle_sweep)
    df_lifecycle_sweep.to_csv(TAB_DIR / "cost_lifecycle_sweep_results.csv", index=False)

    best_lc_idx = df_lifecycle_sweep["total_cost"].idxmin()
    optimal_lc = lifecycle_sweep[best_lc_idx]
    opt_threshold = optimal_lc["threshold"]

    print(f"Policy 3 (Predictive Maintenance at T*={opt_threshold} cycles):")
    print(f"  Total Cost = ${optimal_lc['total_cost']:,.2f} | Planned = {optimal_lc['planned_overhauls']} | Unplanned = {optimal_lc['unplanned_failures']}")
    print(f"  Average Wasted Cycles per Engine = {optimal_lc['avg_wasted_cycles']:.1f} cycles")
    print(f"  Average Advance Warning Lead Time = {optimal_lc['avg_lead_time']:.1f} cycles")

    # Savings across 100 engines
    savings = calculate_cost_savings(
        rtf_cost=rtf_lifecycle["total_cost"],
        fixed_cost=best_fixed["total_cost"],
        pdm_cost=optimal_lc["total_cost"]
    )
    print("\nNet Financial Savings (100-Engine Fleet):")
    print(f"  Savings vs Run-to-Failure: ${savings['savings_vs_run_to_failure']:,.2f} ({savings['pct_savings_vs_rtf']}%)")
    print(f"  Savings vs Fixed-Schedule: ${savings['savings_vs_fixed_schedule']:,.2f} ({savings['pct_savings_vs_fixed']}%)")

    # Policy comparison table
    df_policy_comparison = pd.DataFrame([
        {
            "Policy": "Run-to-Failure (Corrective)",
            "Unplanned Cost ($)": rtf_lifecycle["unplanned_cost"],
            "Planned Cost ($)": rtf_lifecycle["planned_cost"],
            "Wasted Life Cost ($)": rtf_lifecycle["wasted_life_cost"],
            "Total Fleet Cost ($)": rtf_lifecycle["total_cost"],
            "Unplanned Failures": rtf_lifecycle["unplanned_failures"],
            "Planned Overhauls": rtf_lifecycle["planned_overhauls"]
        },
        {
            "Policy": f"Fixed-Schedule ({best_interval} cycles)",
            "Unplanned Cost ($)": best_fixed["unplanned_cost"],
            "Planned Cost ($)": best_fixed["planned_cost"],
            "Wasted Life Cost ($)": best_fixed["wasted_life_cost"],
            "Total Fleet Cost ($)": best_fixed["total_cost"],
            "Unplanned Failures": best_fixed["unplanned_failures"],
            "Planned Overhauls": best_fixed["planned_overhauls"]
        },
        {
            "Policy": f"Predictive Maintenance (T*={opt_threshold}c)",
            "Unplanned Cost ($)": optimal_lc["unplanned_cost"],
            "Planned Cost ($)": optimal_lc["planned_cost"],
            "Wasted Life Cost ($)": optimal_lc["wasted_life_cost"],
            "Total Fleet Cost ($)": optimal_lc["total_cost"],
            "Unplanned Failures": optimal_lc["unplanned_failures"],
            "Planned Overhauls": optimal_lc["planned_overhauls"]
        }
    ])
    df_policy_comparison.to_csv(TAB_DIR / "policy_comparison.csv", index=False)

    # =========================================================
    # 7. Generate Publication-Quality Plots
    # =========================================================
    print("\n[PLOTS] Generating Phase 5 figures...")

    # Plot 15: Fleet Cost vs Threshold Sweep (Lifecycle)
    plt.figure(figsize=(10, 5))
    plt.plot(df_lifecycle_sweep["threshold"], df_lifecycle_sweep["total_cost"] / 1e6, "b-o", linewidth=2.2, label="Predictive Maintenance Fleet Cost ($M)")
    plt.axvline(opt_threshold, color="green", linestyle="--", linewidth=1.5, label=f"Optimal Threshold T* = {opt_threshold} cycles (Min Cost: ${optimal_lc['total_cost']/1e6:.2f}M)")
    plt.axhline(best_fixed["total_cost"] / 1e6, color="orange", linestyle="--", linewidth=1.5, label=f"Optimal Fixed-Schedule ({best_interval}c): ${best_fixed['total_cost']/1e6:.2f}M")
    plt.axhline(rtf_lifecycle["total_cost"] / 1e6, color="red", linestyle=":", linewidth=1.5, label=f"Run-to-Failure Baseline: ${rtf_lifecycle['total_cost']/1e6:.1f}M")
    plt.title("Total Fleet Expenditure vs Maintenance Decision Threshold T (100 Turbofans)", fontsize=12, fontweight="bold")
    plt.xlabel("Maintenance Decision Trigger Threshold T (Cycles)")
    plt.ylabel("Total Fleet Expenditure ($ Millions)")
    plt.legend()
    plt.tight_layout()
    fig15 = FIG_DIR / "15_cost_vs_threshold_sweep.png"
    plt.savefig(fig15, dpi=300)
    plt.close()
    print(f"[PLOTS] Saved: {fig15.name}")

    # Plot 16: Policy Cost Breakdown Comparison (Bar Chart)
    fig, ax = plt.subplots(figsize=(10, 5))
    policy_labels = ["Run-to-Failure", f"Fixed-Schedule ({best_interval}c)", f"Predictive (T*={opt_threshold}c)"]
    unplanned_bars = [rtf_lifecycle["unplanned_cost"] / 1e6, best_fixed["unplanned_cost"] / 1e6, optimal_lc["unplanned_cost"] / 1e6]
    planned_bars = [rtf_lifecycle["planned_cost"] / 1e6, best_fixed["planned_cost"] / 1e6, optimal_lc["planned_cost"] / 1e6]
    wasted_bars = [rtf_lifecycle["wasted_life_cost"] / 1e6, best_fixed["wasted_life_cost"] / 1e6, optimal_lc["wasted_life_cost"] / 1e6]

    x = np.arange(len(policy_labels))
    width = 0.55

    ax.bar(x, unplanned_bars, width, label="Unplanned Breakdown ($200k/ea)", color="#d62728", edgecolor="black")
    ax.bar(x, planned_bars, width, bottom=unplanned_bars, label="Planned Overhaul ($40k/ea)", color="#1f77b4", edgecolor="black")
    ax.bar(x, wasted_bars, width, bottom=np.array(unplanned_bars) + np.array(planned_bars), label="Wasted Component Life ($200/cyc)", color="#ff7f0e", edgecolor="black")

    ax.set_ylabel("Total Expenditure ($ Millions)", fontsize=11)
    ax.set_title("Operational Expenditure Breakdown Across Maintenance Policies (100-Engine Fleet)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(policy_labels, fontsize=10, fontweight="bold")
    ax.legend(loc="upper right")

    totals = [rtf_lifecycle["total_cost"]/1e6, best_fixed["total_cost"]/1e6, optimal_lc["total_cost"]/1e6]
    for i, v in enumerate(totals):
        ax.text(i, v + 0.35, f"${v:.2f}M", ha="center", fontweight="bold", fontsize=10)

    plt.tight_layout()
    fig16 = FIG_DIR / "16_policy_cost_comparison.png"
    plt.savefig(fig16, dpi=300)
    plt.close()
    print(f"[PLOTS] Saved: {fig16.name}")

    # Plot 17: Operational Trade-off (Wasted Cycles vs Advance Lead Time)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    line1 = ax1.plot(df_lifecycle_sweep["threshold"], df_lifecycle_sweep["avg_wasted_cycles"], "b-s", linewidth=2, label="Avg Wasted Life Cycles (Discarded Early)")
    line2 = ax2.plot(df_lifecycle_sweep["threshold"], df_lifecycle_sweep["pct_unplanned_prevented"], "g-^", linewidth=2, label="% Catastrophic Failures Prevented")

    ax1.set_xlabel("Maintenance Threshold T (Cycles)", fontsize=11)
    ax1.set_ylabel("Average Wasted Cycles per Engine", color="blue", fontsize=11)
    ax2.set_ylabel("% Unplanned Failures Prevented", color="green", fontsize=11)
    ax1.tick_params(axis="y", labelcolor="blue")
    ax2.tick_params(axis="y", labelcolor="green")
    ax2.set_ylim(80, 105)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right")
    plt.title("Pareto Trade-off: Component Utilization vs Safety Margin", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig17 = FIG_DIR / "17_tradeoff_early_vs_missed.png"
    plt.savefig(fig17, dpi=300)
    plt.close()
    print(f"[PLOTS] Saved: {fig17.name}")

    # Plot 18: Break-even & Sensitivity Curve across Cost Ratios
    cost_ratios = np.linspace(1.5, 8.0, 20)
    net_savings_curve = []
    for r in cost_ratios:
        sim_temp = TurbofanCostSimulator(cost_unplanned=40000.0 * r, cost_planned=40000.0, cost_waste_per_cycle=200.0)
        rtf_c = sim_temp.evaluate_run_to_failure(100)["total_cost"]
        # Predictive maintenance at T*=10c
        pdm_c = (100 * 40000.0) + (100 * optimal_lc["avg_wasted_cycles"] * 200.0)
        net_savings_curve.append((rtf_c - pdm_c) / 1e6)

    plt.figure(figsize=(9, 4.5))
    plt.plot(cost_ratios, net_savings_curve, "g-o", linewidth=2)
    plt.axvline(5.0, color="orange", linestyle="--", label="Baseline Ratio ($200k / $40k = 5.0x)")
    plt.axhline(0, color="black", linestyle="-", linewidth=1)
    plt.title("Economic Sensitivity: Net Savings vs Unplanned-to-Planned Cost Ratio", fontsize=12, fontweight="bold")
    plt.xlabel("Cost Ratio (C_unplanned / C_planned)")
    plt.ylabel("Net Cost Avoidance ($ Millions)")
    plt.legend()
    plt.tight_layout()
    fig18 = FIG_DIR / "18_financial_savings_breakdown.png"
    plt.savefig(fig18, dpi=300)
    plt.close()
    print(f"[PLOTS] Saved: {fig18.name}")

    # =========================================================
    # 8. Append to results/metrics.json
    # =========================================================
    metrics_path = RESULTS_DIR / "metrics.json"
    with open(metrics_path, "r") as f:
        master_metrics = json.load(f)

    master_metrics["decision_and_cost_simulation"] = {
        "assumptions": {
            "cost_unplanned_failure_usd": simulator.cost_unplanned,
            "cost_planned_overhaul_usd": simulator.cost_planned,
            "cost_wasted_life_per_cycle_usd": simulator.cost_waste_per_cycle,
            "cost_ratio_unplanned_to_planned": simulator.cost_unplanned / simulator.cost_planned
        },
        "optimal_threshold_cycles": opt_threshold,
        "run_to_failure": rtf_lifecycle,
        "fixed_schedule_optimal": best_fixed,
        "predictive_maintenance_optimal": optimal_lc,
        "financial_savings": savings
    }

    with open(metrics_path, "w") as f:
        json.dump(master_metrics, f, indent=4)
    print(f"\n[METRICS] Successfully appended updated cost metrics to {metrics_path}")


if __name__ == "__main__":
    main()
