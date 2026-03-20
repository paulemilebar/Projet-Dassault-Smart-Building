import argparse
from datetime import date, datetime

import pandas as pd
from pathlib import Path

from ingestion.load_predicted_inputs import load_predicted_inputs
from state.load_state import load_current_state
from simulator.generate_data import SimulationConfig, generate_and_save_day, simulate_historical_data
from Predictor_pv.hybrid_predictor_ppv import WeatherProvider, PhysicalPVPredictor, MLPVPredictor, HybridPVPredictor
from baseline.calculator import BaselineCalculator

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MVP daily smart-building datasets.")
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
    Exécute l'optimiseur MILP avec les donnees du pipeline.
    Retourne un DataFrame planifie (24 lignes) ou None si non resolu.
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

# Paramètres du panneau
P_STC = 3000  
BETA = -0.004 
NOCT = 45
NB_PANELS = 1

# Coordonnées du smart building (ex: Paris)
LAT = 48.8566
LON = 2.3522

HISTORIC_CSV_PATH = Path("./energy_planner/data/historic/donnees_reelles_historique.csv")
    
    
def main() -> None:
    """ Point d'entrée principal du script. Gère la génération de données, le chargement 
    des entrées et de l'état, et l'exécution de l'optimiseur. Affiche les résultats dans 
    la console."""
    args = parse_args()
    run_date = parse_run_date(args.run_date)
    cfg = SimulationConfig(seed=args.seed)
    
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATASET_PATH = PROJECT_ROOT / "energy_planner" / "data" / "processed" / "synthetic_user_history.csv"
    PV_MODEL_PATH = PROJECT_ROOT / "Predictor_pv" / "models" / "rf_pv_model.joblib"
    weather_provider = WeatherProvider(latitude=LAT, longitude=LON)
    phys_predictor = PhysicalPVPredictor(p_stc=P_STC, beta=BETA, noct=NOCT, nb_panels=NB_PANELS)
    ml_predictor = MLPVPredictor(dataset_csv=DATASET_PATH, model_path=PV_MODEL_PATH,dynamic_retrain=False)
    hybrid_pv_agent = HybridPVPredictor(phys_predictor, ml_predictor, weather_provider)
    paths = generate_and_save_day(run_date=run_date, cfg=cfg, pv_agent=hybrid_pv_agent)
    predicted_inputs = load_predicted_inputs(run_date=run_date)
    state = load_current_state(run_date=run_date)

    print(f"Run date: {run_date.isoformat()}")
    for name, path in paths.items():
        print(f"{name}: {path}")
    print("\nLoaded predicted inputs (optimizer contract):")
    print(predicted_inputs.head(3).to_string(index=False))
    print(f"... total rows: {len(predicted_inputs)}")
    print("\nLoaded state:")
    print(state)
    plan_df = _run_optimizer(predicted_inputs, state)
    if plan_df is not None:
        print("\nOptimizer plan (first 6 rows):")
        print(plan_df.head(6).to_string(index=False))
        calc = BaselineCalculator()
        res_base = calc.compute_grid_only(predicted_inputs, state)
        res_opt = calc.compute_optimizer_performance(plan_df, predicted_inputs, state)

        print("\n" + "="*60)
        print("📊 RÉSULTATS DE LA STRATÉGIE ÉNERGÉTIQUE")
        print("="*60)
        print(f"{'Indicateur':<25} | {'Baseline':<15} | {'Optimiseur':<15}")
        print("-" * 60)
        print(f"{'Coût financier (€)':<25} | {res_base['cost']:<15.2f} | {res_opt['cost']:<15.2f}")
        print(f"{'Emissions (kg CO2)':<25} | {res_base['emissions']:<15.2f} | {res_opt['emissions']:<15.2f}")
        print("-" * 60)
        
        economie = res_base['cost'] - res_opt['cost']
        print(f"💰 Économie réalisée : {economie:.2f} €")
        print(f"🌿 Réduction carbone : {res_base['emissions'] - res_opt['emissions']:.2f} kg CO2")
        print("="*60)


if __name__ == "__main__":
    main()
