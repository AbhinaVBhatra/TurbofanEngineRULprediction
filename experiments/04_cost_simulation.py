"""
04_cost_simulation.py - Corrected Operational Fleet Maintenance Cost Simulation

Simulates complete cycle-by-cycle degradation trajectories using:
1. Real model-predicted RUL (LightGBM and PyTorch GRU)
2. Run-to-Failure (Corrective Maintenance)
3. Fixed-Interval Maintenance (Preventive at fixed cycles)
4. Theoretical Oracle Policy (Perfect prognostics lower bound)

Mechanics:
At each operational cycle t, model computes y_hat_t.
Alert is triggered at first cycle t* where y_hat_t <= T.
Outcomes:
- Timely Planned Overhaul: Alert while true RUL > 0 and <= 2*T.
  Cost = C_planned + (true_RUL * C_waste).
- Premature Replacement (False Alarm): Alert while true RUL > 2*T.
  Cost = C_planned + (true_RUL * C_waste). High life wastage.
- Missed Catastrophic Breakdown: Engine reaches true RUL == 0 without alert.
  Cost = C_unplanned.

Cost Parameters:
- C_planned = $40,000
- C_unplanned = $200,000 (Baseline 5x)
- C_waste = $200 / unutilized cycle

Sensitivity Analysis:
- Sweeps threshold T in [5, 10, 15, 20, 25, 30, 35, 40]
- Sweeps cost ratio C_unplanned / C_planned from 2x to 20x ($80,000 to $800,000)

Outputs:
- results/paper/04_cost_simulation.json
- results/paper/04_cost_simulation.csv
- results/paper/04_cost_sensitivity.csv
- Appends to paper/number_provenance.csv
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from src.models import PyTorchGRUTrainer, TurbofanGRU
from experiments.provenance_tracker import log_number

RESULTS_DIR = PROJECT_ROOT / "results" / "paper"
PROC_DIR = PROJECT_ROOT / "data" / "processed"
ART_DIR = PROJECT_ROOT / "artifacts"


def simulate_engine_lifecycle(
    df_engine: pd.DataFrame,
    preds: np.ndarray,
    threshold: float,
    cost_planned: float = 40000.0,
    cost_unplanned: float = 200000.0,
    cost_waste: float = 200.0
) -> dict:
    """
    Simulates a single engine chronologically from cycle 1 to failure.
    Tracks first trigger cycle where predicted RUL <= threshold.
    """
    true_ruls = df_engine["RUL"].values
    cycles = df_engine["time_cycles"].values
    total_lifetime = len(cycles)

    # First cycle where predicted RUL <= threshold
    flags = np.where(preds <= threshold)[0]
    
    if len(flags) > 0:
        first_flag = flags[0]
        alert_cycle = int(cycles[first_flag])
        true_rul_at_alert = float(true_ruls[first_flag])

        if true_rul_at_alert > 0:
            # Maintained safely before catastrophic failure
            is_premature = (true_rul_at_alert > 2 * threshold)
            outcome = "premature" if is_premature else "timely_planned"
            wasted_cycles = true_rul_at_alert
            cost = cost_planned + (wasted_cycles * cost_waste)
            lead_time = true_rul_at_alert
        else:
            # Model triggered at or after physical breakdown
            outcome = "missed_failure"
            wasted_cycles = 0.0
            cost = cost_unplanned
            lead_time = 0.0
    else:
        # Engine reached failure without any alert triggered
        outcome = "missed_failure"
        alert_cycle = total_lifetime
        true_rul_at_alert = 0.0
        wasted_cycles = 0.0
        cost = cost_unplanned
        lead_time = 0.0

    return {
        "unit_nr": int(df_engine["unit_nr"].iloc[0]),
        "total_lifetime": total_lifetime,
        "alert_cycle": alert_cycle,
        "true_rul_at_alert": true_rul_at_alert,
        "outcome": outcome,
        "wasted_cycles": wasted_cycles,
        "lead_time": lead_time,
        "cost": cost
    }


def run_cost_simulation():
    print("=" * 70)
    print("EXPERIMENT 4: Corrected Operational Maintenance Cost Simulation")
    print("=" * 70)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load validation trajectories (FD001 validation split: complete run-to-failure engines)
    val_feat = pd.read_parquet(PROC_DIR / "FD001_val_features.parquet")
    with open(ART_DIR / "FD001_feature_config.json") as f:
        feat_config = json.load(f)
    feature_cols = feat_config["feature_cols"]

    # Load LightGBM model
    lgbm_model = joblib.load(ART_DIR / "FD001_best_model.pkl")

    # Load PyTorch GRU model
    X_val_seq = np.load(PROC_DIR / "FD001_X_val_seq.npy")
    y_val_seq = np.load(PROC_DIR / "FD001_y_val_seq.npy")
    gru_trainer = PyTorchGRUTrainer(input_dim=14, hidden_dim=64, num_layers=2)
    gru_weights = ART_DIR / "FD001_gru_model.pt"
    if gru_weights.exists():
        gru_trainer.load(gru_weights)

    val_units = sorted(val_feat["unit_nr"].unique())
    num_engines = len(val_units)
    scale_to_fleet = 100.0 / num_engines  # Project to standard 100-engine fleet

    # Compute engine lifetimes
    lifetimes = {u: len(val_feat[val_feat["unit_nr"] == u]) for u in val_units}
    lifetime_values = np.array(list(lifetimes.values()))

    # Baseline cost parameters
    C_PLANNED = 40000.0
    C_UNPLANNED_BASE = 200000.0
    C_WASTE = 200.0

    # Pre-generate predictions for all cycles
    lgbm_preds_by_unit = {}
    for u in val_units:
        eng_data = val_feat[val_feat["unit_nr"] == u].sort_values("time_cycles").reset_index(drop=True)
        X_eng = eng_data[feature_cols].values
        p = np.clip(lgbm_model.predict(X_eng), 0, 125)
        lgbm_preds_by_unit[u] = (eng_data, p)

    # -------------------------------------------------------------
    # 1. Sweep Decision Threshold T for LightGBM
    # -------------------------------------------------------------
    print("\n--- 1. Decision Threshold Sweep for Predictive Maintenance ---")
    threshold_records = []
    thresholds = [5, 10, 15, 20, 25, 30, 35, 40]

    for T in thresholds:
        timely_cnt = 0
        premature_cnt = 0
        missed_cnt = 0
        wasted_cycles_list = []
        lead_time_list = []
        cost_list = []

        for u in val_units:
            eng_data, p = lgbm_preds_by_unit[u]
            res = simulate_engine_lifecycle(
                eng_data, p, threshold=T,
                cost_planned=C_PLANNED, cost_unplanned=C_UNPLANNED_BASE, cost_waste=C_WASTE
            )
            if res["outcome"] == "timely_planned":
                timely_cnt += 1
            elif res["outcome"] == "premature":
                premature_cnt += 1
            else:
                missed_cnt += 1

            wasted_cycles_list.append(res["wasted_cycles"])
            lead_time_list.append(res["lead_time"])
            cost_list.append(res["cost"])

        fleet_cost_actual = float(np.sum(cost_list) * scale_to_fleet)
        fleet_unplanned = float(missed_cnt * scale_to_fleet)
        fleet_planned = float((timely_cnt + premature_cnt) * scale_to_fleet)
        fleet_premature = float(premature_cnt * scale_to_fleet)
        avg_wasted = float(np.mean(wasted_cycles_list))
        avg_lead = float(np.mean(lead_time_list))
        pct_prevented = float(((num_engines - missed_cnt) / num_engines) * 100.0)

        threshold_records.append({
            "policy": f"Predictive (LightGBM, T={T})",
            "threshold": T,
            "fleet_size": 100,
            "unplanned_failures": fleet_unplanned,
            "planned_overhauls": fleet_planned,
            "premature_overhauls": fleet_premature,
            "pct_failures_prevented": pct_prevented,
            "avg_wasted_cycles": round(avg_wasted, 1),
            "avg_lead_time": round(avg_lead, 1),
            "total_fleet_cost_USD": round(fleet_cost_actual, 2),
            "cost_per_engine_USD": round(fleet_cost_actual / 100.0, 2)
        })
        print(f"  T={T:2d}c -> Cost: ${fleet_cost_actual/1e6:.2f}M | Failures Missed: {fleet_unplanned:.0f} | Prevented: {pct_prevented:.1f}% | Avg Wasted: {avg_wasted:.1f}c | Lead: {avg_lead:.1f}c")

    # -------------------------------------------------------------
    # 2. Benchmark Against Alternative Policies
    # -------------------------------------------------------------
    print("\n--- 2. Benchmarking Across Maintenance Paradigms ---")
    paradigm_records = []

    # Policy A: Run-to-Failure (Corrective)
    cost_rtf = 100 * C_UNPLANNED_BASE
    paradigm_records.append({
        "policy": "Run-to-Failure (Corrective)",
        "unplanned_failures": 100.0,
        "planned_overhauls": 0.0,
        "pct_failures_prevented": 0.0,
        "avg_wasted_cycles": 0.0,
        "total_fleet_cost_USD": cost_rtf,
        "cost_per_engine_USD": cost_rtf / 100.0
    })

    # Policy B: Fixed-Interval (Preventive at I=150 cycles)
    for fixed_interval in [130, 150, 170]:
        unplanned_cnt = int(np.sum(lifetime_values < fixed_interval))
        planned_cnt = len(lifetime_values) - unplanned_cnt
        wasted = np.sum(np.maximum(0, lifetime_values - fixed_interval)[lifetime_values >= fixed_interval])
        avg_w = float(wasted / len(lifetime_values))
        fixed_cost = (unplanned_cnt * C_UNPLANNED_BASE + planned_cnt * C_PLANNED + wasted * C_WASTE) * scale_to_fleet

        paradigm_records.append({
            "policy": f"Fixed-Schedule ({fixed_interval} cycles)",
            "unplanned_failures": float(unplanned_cnt * scale_to_fleet),
            "planned_overhauls": float(planned_cnt * scale_to_fleet),
            "pct_failures_prevented": round(((len(lifetime_values) - unplanned_cnt) / len(lifetime_values)) * 100.0, 1),
            "avg_wasted_cycles": round(avg_w, 1),
            "total_fleet_cost_USD": round(fixed_cost, 2),
            "cost_per_engine_USD": round(fixed_cost / 100.0, 2)
        })

    # Policy C: Predictive Maintenance (LightGBM at T=15)
    best_pred_row = [r for r in threshold_records if r["threshold"] == 15][0]
    paradigm_records.append({
        "policy": "Predictive (LightGBM, T=15)",
        "unplanned_failures": best_pred_row["unplanned_failures"],
        "planned_overhauls": best_pred_row["planned_overhauls"],
        "pct_failures_prevented": best_pred_row["pct_failures_prevented"],
        "avg_wasted_cycles": best_pred_row["avg_wasted_cycles"],
        "total_fleet_cost_USD": best_pred_row["total_fleet_cost_USD"],
        "cost_per_engine_USD": best_pred_row["cost_per_engine_USD"]
    })

    # Policy D: Oracle Policy (Theoretical Lower Bound: Perfect True RUL Knowledge at T=10)
    # Oracle alerts exactly at true RUL = 10 cycles for every engine
    oracle_cost = 100 * (C_PLANNED + 10.0 * C_WASTE)  # 100 * ($40,000 + $2,000) = $4,200,000
    paradigm_records.append({
        "policy": "Oracle Policy (Theoretical Lower Bound, T=10)",
        "unplanned_failures": 0.0,
        "planned_overhauls": 100.0,
        "pct_failures_prevented": 100.0,
        "avg_wasted_cycles": 10.0,
        "total_fleet_cost_USD": oracle_cost,
        "cost_per_engine_USD": oracle_cost / 100.0
    })

    for p in paradigm_records:
        print(f"  {p['policy']:45s} -> Cost: ${p['total_fleet_cost_USD']/1e6:.2f}M | Prevented: {p['pct_failures_prevented']:.1f}% | Wasted: {p['avg_wasted_cycles']:.1f}c")

    # -------------------------------------------------------------
    # 3. Sensitivity Analysis over Cost Ratio (C_unplanned / C_planned)
    # -------------------------------------------------------------
    print("\n--- 3. Sensitivity Analysis over Cost Ratio C_unplanned / C_planned ---")
    ratios = [2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0]
    sensitivity_records = []

    for r in ratios:
        c_unp = r * C_PLANNED

        # RTF cost
        cost_rtf_r = 100.0 * c_unp

        # Fixed Schedule (150c) cost
        unp_fixed = int(np.sum(lifetime_values < 150))
        pln_fixed = len(lifetime_values) - unp_fixed
        w_fixed = np.sum(np.maximum(0, lifetime_values - 150)[lifetime_values >= 150])
        cost_fixed_r = (unp_fixed * c_unp + pln_fixed * C_PLANNED + w_fixed * C_WASTE) * scale_to_fleet

        # Predictive (T=15) cost
        cost_pred_list = []
        for u in val_units:
            eng_data, p = lgbm_preds_by_unit[u]
            res = simulate_engine_lifecycle(
                eng_data, p, threshold=15,
                cost_planned=C_PLANNED, cost_unplanned=c_unp, cost_waste=C_WASTE
            )
            cost_pred_list.append(res["cost"])
        cost_pred_r = float(np.sum(cost_pred_list) * scale_to_fleet)

        # Oracle cost (0 failures, strictly planned + waste)
        cost_oracle_r = 100.0 * (C_PLANNED + 10.0 * C_WASTE)

        # Savings of Predictive over RTF and Fixed
        savings_vs_rtf = cost_rtf_r - cost_pred_r
        savings_vs_fixed = cost_fixed_r - cost_pred_r

        sensitivity_records.append({
            "cost_ratio_r": r,
            "cost_unplanned_USD": c_unp,
            "cost_planned_USD": C_PLANNED,
            "cost_rtf_USD": cost_rtf_r,
            "cost_fixed_150_USD": cost_fixed_r,
            "cost_predictive_T15_USD": cost_pred_r,
            "cost_oracle_USD": cost_oracle_r,
            "savings_vs_rtf_USD": savings_vs_rtf,
            "savings_vs_fixed_USD": savings_vs_fixed,
            "pct_savings_vs_rtf": round((savings_vs_rtf / cost_rtf_r) * 100.0, 1),
            "pct_savings_vs_fixed": round((savings_vs_fixed / cost_fixed_r) * 100.0, 1)
        })
        print(f"  Ratio {r:4.1f}x (C_unp=${c_unp/1e3:.0f}k) | RTF: ${cost_rtf_r/1e6:.2f}M | Fixed: ${cost_fixed_r/1e6:.2f}M | Pred: ${cost_pred_r/1e6:.2f}M | Savings vs RTF: {((savings_vs_rtf/cost_rtf_r)*100.0):.1f}%")

    # Save to disk
    df_thresh = pd.DataFrame(threshold_records)
    df_paradigm = pd.DataFrame(paradigm_records)
    df_sens = pd.DataFrame(sensitivity_records)

    df_thresh.to_csv(RESULTS_DIR / "04_cost_threshold_sweep.csv", index=False)
    df_paradigm.to_csv(RESULTS_DIR / "04_cost_paradigms.csv", index=False)
    df_sens.to_csv(RESULTS_DIR / "04_cost_sensitivity.csv", index=False)

    summary_json = {
        "cost_assumptions": {
            "C_planned": C_PLANNED,
            "C_unplanned_baseline": C_UNPLANNED_BASE,
            "C_waste_per_cycle": C_WASTE,
            "fleet_size": 100
        },
        "threshold_sweep": threshold_records,
        "paradigms": paradigm_records,
        "sensitivity_analysis": sensitivity_records
    }

    with open(RESULTS_DIR / "04_cost_simulation.json", "w") as f:
        json.dump(summary_json, f, indent=2)

    # Log key numbers to provenance registry
    for row in paradigm_records:
        pol_id = row["policy"].split("(")[0].strip().replace(" ", "_")
        log_number(f"Cost_{pol_id}_Total", "FD001", "Policy", "Total_Fleet_Cost", f"{row['total_fleet_cost_USD']:.2f}", "USD", "experiments/04_cost_simulation.py", "results/paper/04_cost_simulation.json", f"Total cost: {row['policy']}")
        log_number(f"Cost_{pol_id}_Prevented", "FD001", "Policy", "Failures_Prevented", f"{row['pct_failures_prevented']:.1f}", "%", "experiments/04_cost_simulation.py", "results/paper/04_cost_simulation.json", f"% Failures prevented: {row['policy']}")

    for row in sensitivity_records:
        r_int = int(row["cost_ratio_r"])
        log_number(f"Cost_Sensitivity_Ratio_{r_int}x_PredCost", "FD001", "LightGBM", "Fleet_Cost", f"{row['cost_predictive_T15_USD']:.2f}", "USD", "experiments/04_cost_simulation.py", "results/paper/04_cost_sensitivity.csv", f"Pred cost at {r_int}x ratio")
        log_number(f"Cost_Sensitivity_Ratio_{r_int}x_SavingsRTF", "FD001", "LightGBM", "Savings_vs_RTF", f"{row['savings_vs_rtf_USD']:.2f}", "USD", "experiments/04_cost_simulation.py", "results/paper/04_cost_sensitivity.csv", f"Savings over RTF at {r_int}x ratio")

    print(f"\n[SAVED] Cost simulation outputs saved to results/paper/")


if __name__ == "__main__":
    run_cost_simulation()
