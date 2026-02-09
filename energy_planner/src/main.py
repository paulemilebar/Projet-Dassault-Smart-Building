import argparse
from datetime import date, datetime

import pandas as pd

from ingestion.load_predicted_inputs import load_predicted_inputs
from state.load_state import load_current_state
from simulator.generate_data import SimulationConfig, generate_and_save_day


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


def main() -> None:
    args = parse_args()
    run_date = parse_run_date(args.run_date)
    cfg = SimulationConfig(seed=args.seed)
    paths = generate_and_save_day(run_date=run_date, cfg=cfg)
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


if __name__ == "__main__":
    main()
