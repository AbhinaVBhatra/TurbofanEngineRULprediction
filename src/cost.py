"""
cost.py - Maintenance Decision Rules, Operational Threshold Sweeps, and Financial Cost Simulation

Framework Formulation:
----------------------
In industrial operations, machine learning predictions must translate into concrete,
economically optimal maintenance decisions.

We evaluate three operational maintenance policies:
1. Run-to-Failure (Corrective Maintenance):
   Components operate until functional breakdown. Every failure incurs high unplanned downtime
   cost (emergency field replacement, secondary core damage, flight grounding).
2. Fixed-Interval Maintenance (Preventive Maintenance):
   Components are replaced at a fixed cycle interval regardless of individual health.
   Trades catastrophic failures for significant component life wastage.
3. Predictive Maintenance (Condition-Based Maintenance via Model):
   Components are scheduled for planned shop overhaul when predicted Remaining Useful Life (RUL)
   falls below a safety decision threshold T (RUL_hat <= T).

Financial Cost Model Assumptions (Clearly Labeled & User-Editable):
-------------------------------------------------------------------
- C_unplanned: Cost of catastrophic unplanned failure = $200,000 (baseline aviation industry average)
- C_planned:   Cost of scheduled preventive overhaul in depot = $40,000 (20% of unplanned)
- C_waste:     Cost of discarding unutilized component life = $200 per unutilized cycle
               (derived from $40,000 planned overhaul amortized over 200 normal operational cycles)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional


class TurbofanCostSimulator:
    """
    Simulates operational costs and financial savings across maintenance policies
    using empirical model predictions and ground-truth engine lifetimes.
    """
    def __init__(
        self,
        cost_unplanned: float = 200000.0,
        cost_planned: float = 40000.0,
        cost_waste_per_cycle: float = 200.0,
        default_threshold: float = 25.0
    ):
        self.cost_unplanned = float(cost_unplanned)
        self.cost_planned = float(cost_planned)
        self.cost_waste_per_cycle = float(cost_waste_per_cycle)
        self.default_threshold = float(default_threshold)

    def evaluate_run_to_failure(self, num_engines: int) -> Dict[str, Any]:
        """
        Policy 1: Run-to-Failure (Corrective Maintenance).
        Every engine runs until failure, incurring unplanned breakdown costs.
        """
        total_cost = num_engines * self.cost_unplanned
        return {
            "policy": "Run-to-Failure",
            "num_engines": num_engines,
            "unplanned_failures": num_engines,
            "planned_overhauls": 0,
            "wasted_cycles_total": 0.0,
            "unplanned_cost": total_cost,
            "planned_cost": 0.0,
            "wasted_life_cost": 0.0,
            "total_cost": total_cost,
            "avg_cost_per_engine": total_cost / num_engines
        }

    def evaluate_fixed_schedule(
        self,
        actual_lifetimes: np.ndarray,
        interval_cycles: int = 150
    ) -> Dict[str, Any]:
        """
        Policy 2: Fixed-Interval Maintenance.
        Replace engine at `interval_cycles`. If an engine fails before that interval,
        it incurs unplanned cost. If it reaches interval, it incurs planned cost + wasted life.
        """
        lifetimes = np.asarray(actual_lifetimes)
        num_engines = len(lifetimes)

        failed_early = lifetimes < interval_cycles
        num_unplanned = int(np.sum(failed_early))
        num_planned = num_engines - num_unplanned

        wasted_cycles = np.maximum(0, lifetimes - interval_cycles)
        total_wasted_cycles = float(np.sum(wasted_cycles[~failed_early]))

        unplanned_cost = num_unplanned * self.cost_unplanned
        planned_cost = num_planned * self.cost_planned
        wasted_cost = total_wasted_cycles * self.cost_waste_per_cycle
        total_cost = unplanned_cost + planned_cost + wasted_cost

        return {
            "policy": f"Fixed-Schedule ({interval_cycles} cycles)",
            "interval_cycles": interval_cycles,
            "num_engines": num_engines,
            "unplanned_failures": num_unplanned,
            "planned_overhauls": num_planned,
            "wasted_cycles_total": round(total_wasted_cycles, 1),
            "unplanned_cost": unplanned_cost,
            "planned_cost": planned_cost,
            "wasted_life_cost": wasted_cost,
            "total_cost": total_cost,
            "avg_cost_per_engine": total_cost / num_engines
        }

    def evaluate_predictive_maintenance(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Policy 3: Condition-Based Predictive Maintenance.
        Decision rule: Flag for maintenance if y_pred <= threshold T.

        Outcomes for test engines at final observed cycle:
        - Timely Intervention (True Positive): y_pred <= T. Engine flagged for scheduled maintenance.
          Cost = C_planned + (y_true * C_waste_per_cycle).
        - Missed Failure / Catastrophic (False Negative): y_pred > T and y_true <= 0 (or near failure).
          Cost = C_unplanned.
        - Deferral (Healthy): y_pred > T and y_true > T. Engine continues operating safely.
          (No overhaul cost incurred yet).
        """
        if threshold is None:
            threshold = self.default_threshold

        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        num_engines = len(y_true)

        # Flagged by model
        flagged = (y_pred <= threshold)

        # In test set, engines with actual RUL < 10 are considered dangerously close to imminent failure
        # A missed failure occurs when engine is in critical zone (RUL_true <= 5) but model fails to flag
        imminent_failure_cutoff = 5.0
        missed_failure = (~flagged) & (y_true <= imminent_failure_cutoff)

        num_planned = int(np.sum(flagged))
        num_unplanned = int(np.sum(missed_failure))
        num_safe_deferrals = num_engines - num_planned - num_unplanned

        # Wasted life for engines flagged while still having significant remaining life
        wasted_cycles_vector = np.where(flagged, y_true, 0.0)
        total_wasted_cycles = float(np.sum(wasted_cycles_vector))

        planned_cost = num_planned * self.cost_planned
        unplanned_cost = num_unplanned * self.cost_unplanned
        wasted_cost = total_wasted_cycles * self.cost_waste_per_cycle
        total_cost = planned_cost + unplanned_cost + wasted_cost

        return {
            "policy": f"Predictive Maintenance (T={threshold} cycles)",
            "threshold": threshold,
            "num_engines": num_engines,
            "planned_overhauls": num_planned,
            "unplanned_failures": num_unplanned,
            "safe_deferrals": num_safe_deferrals,
            "wasted_cycles_total": round(total_wasted_cycles, 1),
            "planned_cost": planned_cost,
            "unplanned_cost": unplanned_cost,
            "wasted_life_cost": round(wasted_cost, 2),
            "total_cost": round(total_cost, 2),
            "avg_cost_per_engine": round(total_cost / num_engines, 2)
        }

    def simulate_full_lifecycle(
        self,
        df_trajectories: pd.DataFrame,
        model: Any,
        feature_cols: List[str],
        threshold: float
    ) -> Dict[str, Any]:
        """
        Simulates complete operational lifecycle for engines running from cycle 1 to failure.
        Tracks exactly which cycle triggers the maintenance threshold (RUL_hat <= threshold).
        Outcomes:
        - Planned Overhaul: Model alerts while RUL_true > 0. Incurs C_planned + (RUL_true * C_waste).
        - Unplanned Failure: Model fails to alert before RUL_true == 0. Incurs C_unplanned.
        """
        units = df_trajectories["unit_nr"].unique()
        num_engines = len(units)

        num_planned = 0
        num_unplanned = 0
        wasted_cycles_list = []
        lead_times = []

        for u in units:
            eng_data = df_trajectories[df_trajectories["unit_nr"] == u].sort_values("time_cycles").reset_index(drop=True)
            X = eng_data[feature_cols].values
            preds = np.clip(model.predict(X), 0, 125)
            ruls = eng_data["RUL"].values

            flagged_indices = np.where(preds <= threshold)[0]
            if len(flagged_indices) > 0:
                first_flag = flagged_indices[0]
                true_rul_at_alert = float(ruls[first_flag])
                if true_rul_at_alert > 0:
                    num_planned += 1
                    wasted_cycles_list.append(true_rul_at_alert)
                    lead_times.append(true_rul_at_alert)
                else:
                    num_unplanned += 1
            else:
                num_unplanned += 1

        total_wasted = float(np.sum(wasted_cycles_list)) if wasted_cycles_list else 0.0
        avg_wasted = float(np.mean(wasted_cycles_list)) if wasted_cycles_list else 0.0
        avg_lead = float(np.mean(lead_times)) if lead_times else 0.0

        planned_cost = num_planned * self.cost_planned
        unplanned_cost = num_unplanned * self.cost_unplanned
        wasted_cost = total_wasted * self.cost_waste_per_cycle
        total_cost = planned_cost + unplanned_cost + wasted_cost

        return {
            "policy": f"Predictive Maintenance Lifecycle (T={threshold}c)",
            "threshold": threshold,
            "num_engines": num_engines,
            "planned_overhauls": num_planned,
            "unplanned_failures": num_unplanned,
            "pct_unplanned_prevented": round((num_planned / num_engines) * 100.0, 1),
            "total_wasted_cycles": round(total_wasted, 1),
            "avg_wasted_cycles_per_engine": round(avg_wasted, 1),
            "avg_advance_warning_lead_time": round(avg_lead, 1),
            "planned_cost": planned_cost,
            "unplanned_cost": unplanned_cost,
            "wasted_life_cost": round(wasted_cost, 2),
            "total_cost": round(total_cost, 2),
            "avg_cost_per_engine": round(total_cost / num_engines, 2)
        }

    def sweep_thresholds(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold_range: Optional[List[float]] = None
    ) -> pd.DataFrame:
        """
        Sweeps decision threshold T on test fleet snapshot to evaluate trade-offs between
        missed failures, early overhauls, and financial cost.
        """
        if threshold_range is None:
            threshold_range = list(range(5, 55, 2))

        records = []
        for t in threshold_range:
            res = self.evaluate_predictive_maintenance(y_true, y_pred, threshold=float(t))
            records.append({
                "threshold": t,
                "unplanned_failures": res["unplanned_failures"],
                "planned_overhauls": res["planned_overhauls"],
                "wasted_cycles": res["wasted_cycles_total"],
                "unplanned_cost": res["unplanned_cost"],
                "planned_cost": res["planned_cost"],
                "wasted_life_cost": res["wasted_life_cost"],
                "total_cost": res["total_cost"]
            })

        df_sweep = pd.DataFrame(records)
        return df_sweep


def calculate_cost_savings(
    rtf_cost: float,
    fixed_cost: float,
    pdm_cost: float
) -> Dict[str, Any]:
    """Computes dollar savings, percent reduction, and ROI metrics."""
    savings_vs_rtf = rtf_cost - pdm_cost
    pct_savings_rtf = (savings_vs_rtf / rtf_cost) * 100.0 if rtf_cost > 0 else 0.0

    savings_vs_fixed = fixed_cost - pdm_cost
    pct_savings_fixed = (savings_vs_fixed / fixed_cost) * 100.0 if fixed_cost > 0 else 0.0

    return {
        "savings_vs_run_to_failure": round(savings_vs_rtf, 2),
        "pct_savings_vs_rtf": round(pct_savings_rtf, 2),
        "savings_vs_fixed_schedule": round(savings_vs_fixed, 2),
        "pct_savings_vs_fixed": round(pct_savings_fixed, 2)
    }
