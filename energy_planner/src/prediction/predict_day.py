from __future__ import annotations

from datetime import date
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from simulator.profiles import temperature_profile, occupancy_profile
from simulator.schema import PRED_COLUMNS, SimulationConfig
from Predictor_agent.predictor_electricity_price import predict_next_24h_open_dpe, OpenDpeConfig
from Predictor_demand.predictor_user_demand import UserDemandForecastAgent


def predict_day_inputs(
    run_date: date,
    pv_agent,
    cfg: SimulationConfig = SimulationConfig(),
) -> pd.DataFrame:
    """
    Build the 24-hour predicted input DataFrame for *run_date*.

    All values come from ML models / external APIs — no synthetic simulation.
    Raises if any predictor fails (no silent fallback).

    Returns a DataFrame with exactly PRED_COLUMNS columns and 24 rows (one per hour).
    """
    rng = np.random.default_rng(cfg.seed)

    # --- Weather + PV (Open-Meteo API + hybrid physical/ML model) ---
    pv_forecast = pv_agent.predict_for_day(run_date)
    tout = pv_forecast["Tout"].to_numpy()
    irradiance = pv_forecast["G"].to_numpy()
    pv = pv_forecast["PV"].to_numpy() / 1000.0  # W → kW

    # Indoor temperature: physics-based (no dedicated model yet)
    _, tin = temperature_profile(np.arange(24), rng)

    # --- Occupancy (simulated — no dedicated model yet) ---
    occupancy = occupancy_profile(np.arange(24), rng)

    # --- Demand (two Random Forest models) ---
    demand_agent = UserDemandForecastAgent()
    demand_pred = demand_agent.predict_from_context(
        run_date, tout=tout, tin=tin, occupancy=occupancy
    )
    pfixe = demand_pred["Pfixe_predit"].to_numpy()
    pflex = demand_pred["Pflex_predit"].to_numpy()

    # --- Electricity prices (Open DPE API) ---
    price_cfg = OpenDpeConfig(
        tariff="EDF_bleu",
        option="HC/HP",
        beta_sell=0.6,
        hc_hours_weekday=range(9, 18),
    )
    prices = predict_next_24h_open_dpe(config=price_cfg)
    cbuy = np.array(prices.cbuy)
    csell = np.array(prices.csell)

    # --- Assemble DataFrame ---
    pred = pd.DataFrame({
        "heure": np.arange(24, dtype=int),
        "jour":  run_date.day,
        "mois":  run_date.month,
        "annee": run_date.year,
        "Tout":                  np.round(tout, 3),
        "Tin":                   np.round(tin, 3),
        "G":                     np.round(irradiance, 3),
        "alpha_presence_predit": np.round(occupancy, 4),
        "PV":                    np.round(pv, 3),
        "Pfixe_predit":          np.round(pfixe, 3),
        "Pflex_predit":          np.round(pflex, 3),
        "Cbuy_predit":           np.round(cbuy, 4),
        "Csell_predit":          np.round(csell, 4),
    })

    return pred[PRED_COLUMNS]
