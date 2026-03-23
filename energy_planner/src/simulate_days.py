"""
Multi-day simulation pipeline.

For each day in [start_date, start_date + num_days):
  1. Predict inputs from ML models + APIs
  2. Simulate ground-truth real day
  3. Run optimizer on PREDICTED inputs  -> J_predicted
  4. Run optimizer on REAL inputs       -> J_oracle  (best achievable with perfect info)
  5. Compute prediction MAE / RMSE for PV, Pfixe, Pflex, Cbuy
  6. Online learning update (demand + PV models)
  7. Append one metrics row to CSV

Usage (from repo root):
    python energy_planner/src/simulate_days.py \
        --start-date 2026-03-01 --num-days 14 --seed 42
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from prediction.predict_day import predict_day_inputs
from simulator.generate_data import SimulationConfig, generate_and_save_day
from ingestion.load_predicted_inputs import load_predicted_inputs
from state.load_state import load_current_state
from online_learning.update_models import append_to_history, update_demand_models, update_pv_model
from Predictor_agent.predictor_ppv import (
    WeatherProvider, PhysicalPVPredictor, MLPVPredictor, HybridPVPredictor,
)
from Predictor_demand.predictor_user_demand import DEFAULT_MODEL_PATH as DEMAND_MODEL_PATH

# --- Panel / location constants (same as main.py) ---
P_STC, BETA, NOCT, NB_PANELS = 3000, -0.004, 45, 1
LAT, LON = 48.8566, 2.3522

HISTORIC_CSV = Path("energy_planner/data/historic/donnees_reelles_historique.csv")
DEFAULT_OUTPUT_CSV = Path("energy_planner/data/processed/simulation_metrics.csv")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _compute_objective(
    plan_df: pd.DataFrame,
    cbuy: list,
    csell: list,
    pflex: list,
    C_L: float,
    C_bat: float,
    C_emissions_grid: float,
    C_emissions_PV: float,
) -> float:
    """
    Re-compute the MILP objective value from a solved plan and price/demand inputs.
    Consistent with the cost function in optimizer.py.

    J = sum_t [ (Cbuy[t] + C_em_grid)*Pin[t]
              - Csell[t]*Pgo[t]
              + C_em_PV*PV[t]
              - C_bat*Ebat[t]
              - C_L*Pflex[t]*S[t] ]
    """
    J = 0.0
    for t in range(24):
        row = plan_df.iloc[t]
        J += (cbuy[t] + C_emissions_grid) * row["Pin"]
        J -= csell[t] * row["Pgo"]
        J += C_emissions_PV * row["PV"]
        J -= C_bat * row["Ebat"]
        J -= C_L * pflex[t] * row["S"]
    return J


def _optimizer_available() -> bool:
    try:
        import cplex 
        from optimization.optimizer import optimize 
        return True
    except Exception:
        return False


def _run_opt(predicted_inputs: pd.DataFrame, state: dict) -> pd.DataFrame | None:
    """Run MILP and return 24-row plan DataFrame, or None if CPLEX unavailable."""
    try:
        from optimization.optimizer import optimize
    except Exception:
        return None

    result = optimize(
        C_grid_buy=predicted_inputs["Cbuy_pred_eur_per_kWh"].tolist(),
        C_grid_sell=predicted_inputs["Csell_pred_eur_per_kWh"].tolist(),
        C_L=state["C_L"],
        C_bat=state["C_bat"],
        C_emissions_grid=state["C_emissions_grid"],
        C_emissions_PV=state["C_emissions_PV"],
        P_fixed=predicted_inputs["Pfix_pred_kW"].tolist(),
        P_flex=predicted_inputs["Pflex_pred_kW"].tolist(),
        PV_max=predicted_inputs["PV_pred_kW"].tolist(),
        P_g_max_import=state["P_g_max_import"],
        P_g_max_export=state["P_g_max_export"],
        E_bat_max=state["E_max"],
        P_bat_max=min(state["P_ch_max"], state["P_dis_max"]),
        E_bat_init=state["E_bat_0"],
    )
    if result is None:
        return None
    vals, idx_Pin, idx_Pgo, idx_PV, idx_Pch, idx_Pdis, idx_Ebat, idx_S = result
    rows = [
        {
            "hour": t,
            "Pin":  vals[idx_Pin[t]],
            "Pgo":  vals[idx_Pgo[t]],
            "PV":   vals[idx_PV[t]],
            "Pch":  vals[idx_Pch[t]],
            "Pdis": vals[idx_Pdis[t]],
            "Ebat": vals[idx_Ebat[t]],
            "S":    int(round(vals[idx_S[t]])),
        }
        for t in range(24)
    ]
    return pd.DataFrame(rows)


def _build_oracle_inputs(real_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build an optimizer-ready DataFrame from the real (ground-truth) CSV.
    Same contract as load_predicted_inputs() output.
    """
    return pd.DataFrame({
        "hour":                  real_df["heure"].astype(int).values,
        "PV_pred_kW":            real_df["PV_reel"].astype(float).values,
        "Pfix_pred_kW":          real_df["Pfixe"].astype(float).values,
        "Pflex_pred_kW":         real_df["Pflex_reel"].astype(float).values,
        "Cbuy_pred_eur_per_kWh": real_df["Cbuy_reel"].astype(float).values,
        "Csell_pred_eur_per_kWh":real_df["Csell_reel"].astype(float).values,
    })


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def simulate(
    start_date: date,
    num_days: int,
    seed: int = 42,
    window_days: int = 14,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
) -> pd.DataFrame:
    """
    Run the full pipeline for *num_days* consecutive days starting at *start_date*.
    Returns a DataFrame with one metrics row per day and saves it to *output_csv*.
    """
    cfg = SimulationConfig(seed=seed)

    weather_provider = WeatherProvider(latitude=LAT, longitude=LON)
    phys_predictor   = PhysicalPVPredictor(p_stc=P_STC, beta=BETA, noct=NOCT, nb_panels=NB_PANELS)
    ml_predictor     = MLPVPredictor(historic_real_csv=HISTORIC_CSV)
    hybrid_pv_agent  = HybridPVPredictor(phys_predictor, ml_predictor, weather_provider)

    records = []

    for i in range(num_days):
        run_date = start_date + timedelta(days=i)
        print(f"\n{'='*55}")
        print(f" Day {i+1}/{num_days}  —  {run_date.isoformat()}")
        print(f"{'='*55}")

        # 1. Predict
        predicted = predict_day_inputs(run_date=run_date, pv_agent=hybrid_pv_agent, cfg=cfg)

        # 2. Simulate real + save CSVs
        paths = generate_and_save_day(run_date=run_date, predicted=predicted, cfg=cfg)
        real_df = pd.read_csv(paths["raw_real_csv"])

        # 3. Prediction quality metrics
        mae_pv    = _mae(predicted["PV"].values,           real_df["PV_reel"].values)
        rmse_pv   = _rmse(predicted["PV"].values,          real_df["PV_reel"].values)
        mae_fix   = _mae(predicted["Pfixe_predit"].values,  real_df["Pfixe"].values)
        rmse_fix  = _rmse(predicted["Pfixe_predit"].values, real_df["Pfixe"].values)
        mae_flex  = _mae(predicted["Pflex_predit"].values,  real_df["Pflex_reel"].values)
        rmse_flex = _rmse(predicted["Pflex_predit"].values, real_df["Pflex_reel"].values)
        mae_cbuy  = _mae(predicted["Cbuy_predit"].values,   real_df["Cbuy_reel"].values)
        rmse_cbuy = _rmse(predicted["Cbuy_predit"].values,  real_df["Cbuy_reel"].values)

        # 4 & 5. Optimizer: predicted plan vs oracle
        J_predicted = J_oracle = regret = None
        pred_inputs  = load_predicted_inputs(run_date=run_date)
        oracle_inputs = _build_oracle_inputs(real_df)
        state = load_current_state(run_date=run_date)

        plan_pred   = _run_opt(pred_inputs,   state)
        plan_oracle = _run_opt(oracle_inputs, state)

        if plan_pred is not None and plan_oracle is not None:
            J_predicted = _compute_objective(
                plan_pred,
                pred_inputs["Cbuy_pred_eur_per_kWh"].tolist(),
                pred_inputs["Csell_pred_eur_per_kWh"].tolist(),
                pred_inputs["Pflex_pred_kW"].tolist(),
                state["C_L"], state["C_bat"],
                state["C_emissions_grid"], state["C_emissions_PV"],
            )
            J_oracle = _compute_objective(
                plan_oracle,
                oracle_inputs["Cbuy_pred_eur_per_kWh"].tolist(),
                oracle_inputs["Csell_pred_eur_per_kWh"].tolist(),
                oracle_inputs["Pflex_pred_kW"].tolist(),
                state["C_L"], state["C_bat"],
                state["C_emissions_grid"], state["C_emissions_PV"],
            )
            regret = J_predicted - J_oracle

        # 6. Online learning update
        append_to_history(real_df, HISTORIC_CSV)
        update_demand_models(HISTORIC_CSV, DEMAND_MODEL_PATH, window_days=window_days)
        update_pv_model(HISTORIC_CSV, hybrid_pv_agent, window_days=window_days)

        records.append({
            "date":        run_date.isoformat(),
            "mae_pv":      round(mae_pv,   4),
            "rmse_pv":     round(rmse_pv,  4),
            "mae_pfixe":   round(mae_fix,  4),
            "rmse_pfixe":  round(rmse_fix, 4),
            "mae_pflex":   round(mae_flex, 4),
            "rmse_pflex":  round(rmse_flex,4),
            "mae_cbuy":    round(mae_cbuy, 6),
            "rmse_cbuy":   round(rmse_cbuy,6),
            "J_predicted": round(J_predicted, 4) if J_predicted is not None else None,
            "J_oracle":    round(J_oracle,    4) if J_oracle    is not None else None,
            "regret":      round(regret,      4) if regret      is not None else None,
        })

        print(f"  PV   — MAE: {mae_pv:.4f}  RMSE: {rmse_pv:.4f}")
        print(f"  Pfixe — MAE: {mae_fix:.4f}  RMSE: {rmse_fix:.4f}")
        print(f"  Pflex — MAE: {mae_flex:.4f}  RMSE: {rmse_flex:.4f}")
        print(f"  Cbuy  — MAE: {mae_cbuy:.6f}  RMSE: {rmse_cbuy:.6f}")
        if regret is not None:
            print(f"  J_predicted: {J_predicted:.4f}  J_oracle: {J_oracle:.4f}  regret: {regret:.4f}")

    metrics_df = pd.DataFrame(records)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_csv, index=False)
    print(f"\nMetrics saved to: {output_csv}")
    return metrics_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-day simulation pipeline.")
    p.add_argument("--start-date",   type=str, required=True,  help="YYYY-MM-DD")
    p.add_argument("--num-days",     type=int, default=7)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--window-days",  type=int, default=14,     help="Online learning window (days)")
    p.add_argument("--output-csv",   type=str, default=str(DEFAULT_OUTPUT_CSV))
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    simulate(
        start_date=_parse_date(args.start_date),
        num_days=args.num_days,
        seed=args.seed,
        window_days=args.window_days,
        output_csv=Path(args.output_csv),
    )
