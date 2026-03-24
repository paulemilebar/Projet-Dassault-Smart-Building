import argparse
from datetime import date, datetime

import pandas as pd
from pathlib import Path

from ingestion.load_predicted_inputs import load_predicted_inputs
from state.load_state import load_current_state
from simulator.generate_data import SimulationConfig, generate_and_save_day
from prediction.predict_day import predict_day_inputs
from online_learning.update_models import append_to_history, update_demand_models, update_pv_model
from Predictor_pv.hybrid_predictor_ppv import WeatherProvider, PhysicalPVPredictor, MLPVPredictor, HybridPVPredictor

import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from Predictor_demand.predictor_user_demand import DEFAULT_MODEL_PATH as DEMAND_MODEL_PATH


# --- Panel parameters ---
P_STC = 3000
BETA = -0.004
NOCT = 45
NB_PANELS = 1

# --- Smart building location (Paris) ---
LAT = 48.8566
LON = 2.3522

# --- Paths ---
HISTORIC_CSV_PATH = Path("./energy_planner/data/historic/donnees_reelles_historique.csv")
PV_MODEL_PATH     = Path("./Predictor_pv/models/rf_pv_model.joblib")

# --- Online learning window (in days; 1 day = 24 rows) ---
WINDOW_DAYS = 14


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smart-building daily energy planning pipeline.")
    parser.add_argument(
        "--run-date",
        type=str,
        default=None,
        help="Date in YYYY-MM-DD format. Default: today",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic generation.",
    )
    return parser.parse_args()


def parse_run_date(raw: str | None) -> date:
    if raw is None:
        return date.today()
    return datetime.strptime(raw, "%Y-%m-%d").date()


def _run_optimizer(predicted_inputs: pd.DataFrame, state: dict) -> pd.DataFrame | None:
    """
    Run the MILP optimizer. Returns a 24-row plan DataFrame or None if CPLEX
    is unavailable.
    """
    try:
        from optimization.optimizer import optimize
    except Exception as exc:
        print(f"\nOptimizer skipped: cannot import optimizer/cplex ({exc})")
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
    rows = []
    for t in range(24):
        rows.append(
            {
                "hour": t,
                "Pin": vals[idx_Pin[t]],
                "Pgo": vals[idx_Pgo[t]],
                "PV": vals[idx_PV[t]],
                "Pch": vals[idx_Pch[t]],
                "Pdis": vals[idx_Pdis[t]],
                "Ebat": vals[idx_Ebat[t]],
                "S": int(round(vals[idx_S[t]])),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """
    Daily energy planning pipeline:
      1. Predict tomorrow's inputs (ML models + APIs)
      2. Save predicted CSV; simulate ground-truth real day; save real CSV
      3. Load normalized optimizer inputs
      4. Run MILP optimizer → 24h plan
      5. Append real data to history; retrain demand + PV models (rolling window)
    """
    args = parse_args()
    run_date = parse_run_date(args.run_date)
    cfg = SimulationConfig(seed=args.seed)

    # --- Build PV agent ---
    weather_provider = WeatherProvider(latitude=LAT, longitude=LON)
    phys_predictor = PhysicalPVPredictor(p_stc=P_STC, beta=BETA, noct=NOCT, nb_panels=NB_PANELS)
    ml_predictor = MLPVPredictor(dataset_csv=HISTORIC_CSV_PATH, model_path=PV_MODEL_PATH)
    hybrid_pv_agent = HybridPVPredictor(phys_predictor, ml_predictor, weather_provider)

    # --- 1. Predict ---
    print(f"\n=== Run date: {run_date.isoformat()} ===")
    print("Step 1: predicting day inputs from ML models...")
    predicted = predict_day_inputs(run_date=run_date, pv_agent=hybrid_pv_agent, cfg=cfg)

    # --- 2. Simulate real day + save all CSVs ---
    print("Step 2: simulating real day and saving CSVs...")
    paths = generate_and_save_day(run_date=run_date, predicted=predicted, cfg=cfg)
    for name, path in paths.items():
        print(f"  {name}: {path}")

    # --- 3. Load optimizer inputs ---
    predicted_inputs = load_predicted_inputs(run_date=run_date)
    state = load_current_state(run_date=run_date)
    print("\nLoaded predicted inputs (optimizer contract):")
    print(predicted_inputs.head(3).to_string(index=False))
    print(f"... total rows: {len(predicted_inputs)}")
    print("\nLoaded state:")
    print(state)

    # --- 4. Optimize ---
    plan_df = _run_optimizer(predicted_inputs, state)
    if plan_df is not None:
        print("\nOptimizer plan (first 6 rows):")
        print(plan_df.head(6).to_string(index=False))

    # --- 5. Online learning update ---
    print("\nStep 5: updating models on new real data...")
    real_df = pd.read_csv(paths["raw_real_csv"])
    append_to_history(real_df, HISTORIC_CSV_PATH)
    update_demand_models(HISTORIC_CSV_PATH, DEMAND_MODEL_PATH, window_days=WINDOW_DAYS)
    update_pv_model(HISTORIC_CSV_PATH, hybrid_pv_agent, window_days=WINDOW_DAYS)
    print("Done.")


if __name__ == "__main__":
    main()
