"""
04_cost_simulation_oof.py - Out-Of-Fold (OOF) Maintenance Cost Simulation on NASA C-MAPSS FD001

Executes 5-fold GroupKFold cross-validation by engine ID (unit_nr) over all 100 FD001 training engines.
Every operational cycle of every engine is predicted strictly OUT-OF-FOLD (zero in-sample bias).

Cycle-by-Cycle Simulation Mechanics:
- For each engine i (1 to 100) running from cycle 1 to failure (t_fail):
  Model predicts y_hat_t at each cycle t.
  Alert triggers at first cycle t* where y_hat_t <= T.
- Outcomes:
  1. Timely Planned Overhaul: Alert while true RUL > 0 and <= 2*T.
     Cost = C_planned + (true_RUL * C_waste).
  2. Premature Replacement (False Alarm): Alert while true RUL > 2*T.
     Cost = C_planned + (true_RUL * C_waste).
  3. Missed Catastrophic Breakdown: Engine reaches true RUL == 0 without alert.
     Cost = C_unplanned.

Benchmark Policies (Fleet size = 100):
- Run-to-Failure (Corrective Maintenance)
- Fixed-Interval Maintenance (Preventive at I=130, 150, 170 cycles)
- Predictive Maintenance (LightGBM OOF at T sweep [5, 10, 15, 20, 25, 30, 35, 40])
- Oracle Policy (Theoretical Lower Bound: perfect true RUL knowledge at T=10)

Cost Parameters:
- C_planned = $40,000
- C_unplanned = $200,000 (Baseline 5x)
- C_waste = $200 / unutilized cycle

Sensitivity Analysis:
- Sweeps C_unplanned / C_planned from 2x to 20x ($80,000 to $800,000)

Outputs:
- results/paper/04_cost_threshold_sweep.csv
- results/paper/04_cost_paradigms.csv
- results/paper/04_cost_sensitivity.csv
- results/paper/04_cost_simulation.json
- Updates paper/number_provenance.csv
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from src.data import load_dataset, add_remaining_useful_life
from src.features import INFORMATIVE_SENSORS_FD001, build_engineered_features
from src.models import build_lightgbm
from experiments.provenance_tracker import log_number

RESULTS_DIR = PROJECT_ROOT / "results" / "paper"
SENSOR_COLS = INFORMATIVE_SENSORS_FD001


def generate_oof_predictions(train_df: pd.DataFrame, n_splits: int = 5) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Performs 5-fold GroupKFold by engine unit_nr.
    Returns DataFrame with all cycles, out-of-fold predicted RUL, and feature columns.
    Strictly leakage-free: scalers and LightGBM fit on training folds only.
    """
    gkf = GroupKFold(n_splits=n_splits)
    groups = train_df["unit_nr"].values

    oof_preds = np.zeros(len(train_df), dtype=np.float64)
    all_features_df = None
    feature_cols = None

    print(f"Generating Out-Of-Fold (OOF) predictions across {n_splits} engine folds...")

    for fold, (train_idx, val_idx) in enumerate(gkf.split(train_df, groups=groups), 1):
        train_fold = train_df.iloc[train_idx].copy()
        val_fold = train_df.iloc[val_idx].copy()

        # Scaler fit on training fold only
        scaler = StandardScaler()
        scaler.fit(train_fold[SENSOR_COLS])

        train_fold[SENSOR_COLS] = scaler.transform(train_fold[SENSOR_COLS])
        val_fold[SENSOR_COLS] = scaler.transform(val_fold[SENSOR_COLS])

        # Extract features per engine
        train_feat, fcols = build_engineered_features(train_fold, sensor_cols=SENSOR_COLS, window_sizes=[5, 15])
        val_feat, _ = build_engineered_features(val_fold, sensor_cols=SENSOR_COLS, window_sizes=[5, 15])

        if feature_cols is None:
            feature_cols = fcols

        # Fit LightGBM on training fold
        X_tr = train_feat[feature_cols].values
        y_tr = train_feat["RUL"].values

        lgbm = build_lightgbm(random_state=42)
        lgbm.fit(X_tr, y_tr)

        # Predict out-of-fold
        X_val = val_feat[feature_cols].values
        pred_val = np.clip(lgbm.predict(X_val), 0, 125)
        oof_preds[val_idx] = pred_val

        # Collect features for downstream inspection
        val_feat["oof_pred_RUL"] = pred_val
        val_feat["orig_idx"] = val_idx
        if all_features_df is None:
            all_features_df = val_feat
        else:
            all_features_df = pd.concat([all_features_df, val_feat], axis=0)

        print(f"  Fold {fold}/{n_splits} complete: {len(np.unique(val_fold['unit_nr']))} engines evaluated.")

    all_features_df = all_features_df.sort_values(["unit_nr", "time_cycles"]).reset_index(drop=True)
    return all_features_df, feature_cols


def simulate_single_engine(
    df_engine: pd.DataFrame,
    threshold: float,
    cost_planned: float = 40000.0,
    cost_unplanned: float = 200000.0,
    cost_waste: float = 200.0
) -> dict:
    """
    Simulates chronological lifecycle of an engine from cycle 1 to failure.
    Tracks first trigger cycle where predicted RUL <= threshold.
    """
    true_ruls = df_engine["RUL"].values
    preds = df_engine["oof_pred_RUL"].values
    cycles = df_engine["time_cycles"].values
    total_lifetime = len(cycles)

    flags = np.where(preds <= threshold)[0]

    if len(flags) > 0:
        first_flag = flags[0]
        alert_cycle = int(cycles[first_flag])
        true_rul_at_alert = float(true_ruls[first_flag])

        if true_rul_at_alert > 0:
            is_premature = (true_rul_at_alert > 2 * threshold)
            outcome = "premature" if is_premature else "timely_planned"
            wasted_cycles = true_rul_at_alert
            cost = cost_planned + (wasted_cycles * cost_waste)
            lead_time = true_rul_at_alert
        else:
            outcome = "missed_failure"
            wasted_cycles = 0.0
            cost = cost_unplanned
            lead_time = 0.0
    else:
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


def run_oof_cost_simulation():
    print("=" * 70)
    print("STEP A: Out-Of-Fold (OOF) Maintenance Cost Simulation (All 100 Engines)")
    print("=" * 70)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Load full training dataset
    train_df, _, _ = load_dataset("FD001")
    train_df = add_remaining_useful_life(train_df, rul_cap=125)

    df_oof, feature_cols = generate_oof_predictions(train_df, n_splits=5)

    engines = sorted(df_oof["unit_nr"].unique())
    num_engines = len(engines)  # Exactly 100 engines
    lifetimes = np.array([len(df_oof[df_oof["unit_nr"] == u]) for u in engines])

    # Cost constants
    C_PLANNED = 40000.0
    C_UNPLANNED_BASE = 200000.0
    C_WASTE = 200.0

    # -------------------------------------------------------------
    # 1. Decision Threshold Sweep (T)
    # -------------------------------------------------------------
    print("\n--- 1. Decision Threshold Sweep for Predictive Maintenance (OOF) ---")
    threshold_records = []
    thresholds = [5, 10, 15, 20, 25, 30, 35, 40]

    for T in thresholds:
        timely_cnt = 0
        premature_cnt = 0
        missed_cnt = 0
        wasted_cycles_list = []
        lead_time_list = []
        cost_list = []

        for u in engines:
            eng_data = df_oof[df_oof["unit_nr"] == u].sort_values("time_cycles").reset_index(drop=True)
            res = simulate_single_engine(
                eng_data, threshold=T,
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

        total_cost = float(np.sum(cost_list))
        pct_prevented = float(((num_engines - missed_cnt) / num_engines) * 100.0)
        avg_wasted = float(np.mean(wasted_cycles_list))
        avg_lead = float(np.mean(lead_time_list))

        threshold_records.append({
            "policy": f"Predictive (LightGBM OOF, T={T})",
            "threshold": T,
            "fleet_size": num_engines,
            "unplanned_failures": missed_cnt,
            "planned_overhauls": timely_cnt + premature_cnt,
            "premature_overhauls": premature_cnt,
            "pct_failures_prevented": pct_prevented,
            "avg_wasted_cycles": round(avg_wasted, 2),
            "avg_lead_time": round(avg_lead, 2),
            "total_fleet_cost_USD": round(total_cost, 2),
            "cost_per_engine_USD": round(total_cost / num_engines, 2)
        })
        print(f"  T={T:2d}c -> Cost: ${total_cost/1e6:.3f}M | Missed: {missed_cnt:2d} | Prevented: {pct_prevented:5.1f}% | Avg Wasted: {avg_wasted:5.2f}c | Lead: {avg_lead:5.2f}c")

    # -------------------------------------------------------------
    # 2. Paradigm Comparison
    # -------------------------------------------------------------
    print("\n--- 2. Benchmarking Across Maintenance Paradigms ---")
    paradigm_records = []

    # Policy A: Run-to-Failure (Corrective)
    cost_rtf = float(num_engines * C_UNPLANNED_BASE)
    paradigm_records.append({
        "policy": "Run-to-Failure (Corrective)",
        "unplanned_failures": float(num_engines),
        "planned_overhauls": 0.0,
        "pct_failures_prevented": 0.0,
        "avg_wasted_cycles": 0.0,
        "total_fleet_cost_USD": cost_rtf,
        "cost_per_engine_USD": cost_rtf / num_engines
    })

    # Policy B: Fixed-Interval Maintenance
    for fixed_interval in [130, 150, 170]:
        unplanned_cnt = int(np.sum(lifetimes < fixed_interval))
        planned_cnt = num_engines - unplanned_cnt
        wasted_arr = np.maximum(0, lifetimes - fixed_interval)[lifetimes >= fixed_interval]
        wasted_sum = float(np.sum(wasted_arr))
        avg_w = float(wasted_sum / num_engines)
        fixed_cost = (unplanned_cnt * C_UNPLANNED_BASE) + (planned_cnt * C_PLANNED) + (wasted_sum * C_WASTE)

        paradigm_records.append({
            "policy": f"Fixed-Schedule ({fixed_interval} cycles)",
            "unplanned_failures": float(unplanned_cnt),
            "planned_overhauls": float(planned_cnt),
            "pct_failures_prevented": round(((num_engines - unplanned_cnt) / num_engines) * 100.0, 1),
            "avg_wasted_cycles": round(avg_w, 2),
            "total_fleet_cost_USD": round(fixed_cost, 2),
            "cost_per_engine_USD": round(fixed_cost / num_engines, 2)
        })

    # Policy C: Predictive Maintenance (LightGBM OOF at T=15)
    best_pred_row = [r for r in threshold_records if r["threshold"] == 15][0]
    paradigm_records.append({
        "policy": "Predictive (LightGBM OOF, T=15)",
        "unplanned_failures": float(best_pred_row["unplanned_failures"]),
        "planned_overhauls": float(best_pred_row["planned_overhauls"]),
        "pct_failures_prevented": best_pred_row["pct_failures_prevented"],
        "avg_wasted_cycles": best_pred_row["avg_wasted_cycles"],
        "total_fleet_cost_USD": best_pred_row["total_fleet_cost_USD"],
        "cost_per_engine_USD": best_pred_row["cost_per_engine_USD"]
    })

    # Policy D: Oracle Policy (Theoretical Lower Bound at T=10)
    # Perfect oracle alerts at exactly 10 cycles remaining for all 100 engines
    oracle_cost = float(num_engines * (C_PLANNED + 10.0 * C_WASTE))
    paradigm_records.append({
        "policy": "Oracle Policy (Theoretical Lower Bound, T=10)",
        "unplanned_failures": 0.0,
        "planned_overhauls": float(num_engines),
        "pct_failures_prevented": 100.0,
        "avg_wasted_cycles": 10.0,
        "total_fleet_cost_USD": oracle_cost,
        "cost_per_engine_USD": oracle_cost / num_engines
    })

    for p in paradigm_records:
        print(f"  {p['policy']:45s} -> Cost: ${p['total_fleet_cost_USD']/1e6:.3f}M | Prevented: {p['pct_failures_prevented']:5.1f}% | Wasted: {p['avg_wasted_cycles']:5.2f}c")

    # -------------------------------------------------------------
    # 3. Sensitivity Analysis over Cost Ratio C_unplanned / C_planned
    # -------------------------------------------------------------
    print("\n--- 3. Sensitivity Analysis over Cost Ratio C_unplanned / C_planned ---")
    ratios = [2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0]
    sensitivity_records = []

    for r in ratios:
        c_unp = r * C_PLANNED
        cost_rtf_r = float(num_engines * c_unp)

        # Fixed Schedule (150c)
        unp_fixed = int(np.sum(lifetimes < 150))
        pln_fixed = num_engines - unp_fixed
        w_fixed = float(np.sum(np.maximum(0, lifetimes - 150)[lifetimes >= 150]))
        cost_fixed_r = (unp_fixed * c_unp) + (pln_fixed * C_PLANNED) + (w_fixed * C_WASTE)

        # Predictive Maintenance (T=15)
        cost_pred_list = []
        for u in engines:
            eng_data = df_oof[df_oof["unit_nr"] == u].sort_values("time_cycles").reset_index(drop=True)
            res = simulate_single_engine(
                eng_data, threshold=15,
                cost_planned=C_PLANNED, cost_unplanned=c_unp, cost_waste=C_WASTE
            )
            cost_pred_list.append(res["cost"])
        cost_pred_r = float(np.sum(cost_pred_list))

        # Oracle
        cost_oracle_r = float(num_engines * (C_PLANNED + 10.0 * C_WASTE))

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
        print(f"  Ratio {r:4.1f}x (C_unp=${c_unp/1e3:3.0f}k) | RTF: ${cost_rtf_r/1e6:6.2f}M | Fixed: ${cost_fixed_r/1e6:5.2f}M | Pred: ${cost_pred_r/1e6:5.2f}M | Savings vs RTF: {((savings_vs_rtf/cost_rtf_r)*100.0):5.1f}%")

    # Persist updated results
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
            "fleet_size": 100,
            "validation_scheme": "5-fold GroupKFold Out-Of-Fold (OOF) across all 100 training engines"
        },
        "threshold_sweep": threshold_records,
        "paradigms": paradigm_records,
        "sensitivity_analysis": sensitivity_records
    }

    with open(RESULTS_DIR / "04_cost_simulation.json", "w") as f:
        json.dump(summary_json, f, indent=2)

    # Update number provenance
    for row in paradigm_records:
        pol_id = row["policy"].split("(")[0].strip().replace(" ", "_")
        log_number(f"Cost_OOF_{pol_id}_Total", "FD001", "Policy", "Total_Fleet_Cost", f"{row['total_fleet_cost_USD']:.2f}", "USD", "experiments/04_cost_simulation_oof.py", "results/paper/04_cost_simulation.json", f"OOF Total cost: {row['policy']}")
        log_number(f"Cost_OOF_{pol_id}_Prevented", "FD001", "Policy", "Failures_Prevented", f"{row['pct_failures_prevented']:.1f}", "%", "experiments/04_cost_simulation_oof.py", "results/paper/04_cost_simulation.json", f"OOF % Failures prevented: {row['policy']}")
        log_number(f"Cost_OOF_{pol_id}_Wasted", "FD001", "Policy", "Avg_Wasted_Cycles", f"{row['avg_wasted_cycles']:.2f}", "cycles", "experiments/04_cost_simulation_oof.py", "results/paper/04_cost_simulation.json", f"OOF Avg wasted cycles: {row['policy']}")

    for row in threshold_records:
        t_val = row["threshold"]
        log_number(f"Cost_OOF_T{t_val}_Total", "FD001", "LightGBM_OOF", "Fleet_Cost", f"{row['total_fleet_cost_USD']:.2f}", "USD", "experiments/04_cost_simulation_oof.py", "results/paper/04_cost_threshold_sweep.csv", f"OOF cost at threshold T={t_val}")
        log_number(f"Cost_OOF_T{t_val}_Missed", "FD001", "LightGBM_OOF", "Missed_Failures", f"{row['unplanned_failures']}", "count", "experiments/04_cost_simulation_oof.py", "results/paper/04_cost_threshold_sweep.csv", f"OOF missed failures at T={t_val}")
        log_number(f"Cost_OOF_T{t_val}_Wasted", "FD001", "LightGBM_OOF", "Avg_Wasted_Cycles", f"{row['avg_wasted_cycles']:.2f}", "cycles", "experiments/04_cost_simulation_oof.py", "results/paper/04_cost_threshold_sweep.csv", f"OOF wasted cycles at T={t_val}")

    for row in sensitivity_records:
        r_int = int(row["cost_ratio_r"])
        log_number(f"Cost_OOF_Sens_{r_int}x_PredCost", "FD001", "LightGBM_OOF", "Fleet_Cost", f"{row['cost_predictive_T15_USD']:.2f}", "USD", "experiments/04_cost_simulation_oof.py", "results/paper/04_cost_sensitivity.csv", f"OOF Pred cost at {r_int}x ratio")
        log_number(f"Cost_OOF_Sens_{r_int}x_SavingsRTF", "FD001", "LightGBM_OOF", "Savings_vs_RTF", f"{row['savings_vs_rtf_USD']:.2f}", "USD", "experiments/04_cost_simulation_oof.py", "results/paper/04_cost_sensitivity.csv", f"OOF Savings over RTF at {r_int}x ratio")

    print(f"\n[SAVED] Out-of-fold cost simulation outputs saved and provenance updated.")


if __name__ == "__main__":
    run_oof_cost_simulation()
