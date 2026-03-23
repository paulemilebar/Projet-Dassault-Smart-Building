from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from simulator.profiles import (
    build_time_index,
    occupancy_profile,
    temperature_profile,
)
from simulator.schema import PRED_COLUMNS, REAL_COLUMNS, SimulationConfig
from simulator.storage import save_daily_csv

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from Data_Quality_agent.validation import validate_predicted, validate_real

<<<<<<< HEAD
from Predictor_agent.predictor_electricity_price import predict_next_24h_open_dpe, OpenDpeConfig
from Predictor_demand.predictor_user_demand import UserDemandForecastAgent

"""
TODO : For now we generate one day, maybe of interest to generate multiple days ? 
TODO : Need to code some processing functions to create the processed CSV files from the raw data
This module generates synthetic daily datasets for the smart building energy management.
It creates two tables: 
    - "donnees_predites" with predicted values 
    - "donnees_reelles" with actual values.

each table contains 24 rows (one per hour) and specific columns : 
    - REAL_COLUMNS : [
                    'heure', 'jour', 'mois', 'annee', 'tfrigo', 'Tmin', 'Tmax', 
                    'Tout_reel', 'Tin_reel', 'G', 'alpha_presence_reel', 'PV_reel', 
                    'Pfixe', 'Pflex_reel', 'Pin', 'Pgo', 'Ebat', 'S', 'Cbuy_reel', 'Csell_reel'
                    ]

    - PRED_COLUMNS : [
                    'heure', 'jour', 'mois', 'annee', 'Tout', 'Tin', 'G', 'alpha_presence_predit',
                    'PV', 'Pfixe_predit', 'Pflex_predit', 'Cbuy_predit', 'Csell_predit'
                    ]

The generation is based on configurable parameters and includes random noise for realism.
We fix the random seed for reproducibility. The generated data is validated and saved as CSV files.
"""


def generate_predicted_day(run_date: date, pv_agent=None, cfg: SimulationConfig = SimulationConfig()) -> pd.DataFrame:
    """"Generate the donnees_predites table for a given date based on the simulation configuration."""
    rng = np.random.default_rng(cfg.seed)
    df = build_time_index(run_date)
    h = df["heure"].to_numpy()

    # --- NOUVELLE LOGIQUE D'INTÉGRATION DE L'AGENT PV ---
    if pv_agent is not None:
        # L'agent récupère la météo et prédit la production
        pv_forecast = pv_agent.predict_for_day(run_date)
        tout = pv_forecast["Tout"].to_numpy()
        irradiance = pv_forecast["G"].to_numpy()
        # Conversion de Watts en kiloWatts
        pv = pv_forecast["PV"].to_numpy()
        # On a toujours besoin de simuler la température intérieure (Tin)
        _, tin = temperature_profile(h, rng) 
    else:
        # --- ANCIENNE LOGIQUE (Fallback si aucun agent n'est fourni) ---
        tout, tin = temperature_profile(h, rng)
        irradiance = daylight_irradiance_w_m2(h, rng)
        pv = np.clip(cfg.pv_kw_peak * irradiance / 1000.0, 0.0, cfg.pv_kw_peak)

    occupancy = occupancy_profile(h, rng)

    demand_agent = UserDemandForecastAgent()
    demand_pred = demand_agent.predict_from_context(
        run_date,
        tout=tout,
        tin=tin,
        occupancy=occupancy,
    )
    pfixe = demand_pred["Pfixe_predit"].to_numpy()
    pflex = demand_pred["Pflex_predit"].to_numpy()
    
    # use predictor agent for predicting electricity prices
    cfg = OpenDpeConfig(
        tariff="EDF_bleu",
        option="HC/HP",
        beta_sell=0.6,
        hc_hours_weekday=range(9, 18),
    )
    res = predict_next_24h_open_dpe(config=cfg)
    cbuy = res.cbuy
    csell = res.csell

    pred = df.copy()
    pred["Tout"] = np.round(tout, 3)
    pred["Tin"] = np.round(tin, 3)
    pred["G"] = np.round(irradiance, 3)
    pred["alpha_presence_predit"] = np.round(occupancy, 4)
    pred["PV"] = np.round(pv, 3)
    pred["Pfixe_predit"] = np.round(pfixe, 3)
    pred["Pflex_predit"] = np.round(pflex, 3)
    pred["Cbuy_predit"] = np.round(cbuy, 4)
    pred["Csell_predit"] = np.round(csell, 4)

    validate_predicted(pred)
    return pred[PRED_COLUMNS]

=======
>>>>>>> 5c1d6ea930d4b6bf1bef7e3e93b7442fbca97ba8

def _simulate_battery_and_grid(
    demand_kw: np.ndarray,
    pv_kw: np.ndarray,
    cfg: SimulationConfig,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    """ Simulate the battery state of charge (Ebat), grid import (Pin) and export (Pgo)
    based on the demand and PV production."""

    ebat = np.zeros(24, dtype=float)
    pin = np.zeros(24, dtype=float)
    pgo = np.zeros(24, dtype=float)
    s = np.zeros(24, dtype=int)

    e_prev = cfg.ebat_initial_kwh

    for t in range(24):
        net = demand_kw[t] - pv_kw[t]

        if net <= 0:
            surplus = -net
            charge = min(surplus, cfg.pch_max_kw, (cfg.ebat_max_kwh - e_prev) / cfg.eta_ch)
            e_now = min(cfg.ebat_max_kwh, e_prev + charge * cfg.eta_ch)
            sell = max(0.0, surplus - charge)
            pin[t] = 0.0
            pgo[t] = sell
        else:
            discharge = min(net, cfg.pdis_max_kw, e_prev * cfg.eta_dis)
            e_now = max(0.0, e_prev - discharge / cfg.eta_dis)
            import_grid = max(0.0, net - discharge)
            pin[t] = import_grid
            pgo[t] = 0.0
            if import_grid > 1.8:
                s[t] = 1

        ebat[t] = e_now
        e_prev = e_now

    return np.round(pin, 3), np.round(pgo, 3), np.round(ebat, 3), s


def generate_real_day(
    run_date: date,
    predicted: pd.DataFrame,
    cfg: SimulationConfig = SimulationConfig(),
) -> pd.DataFrame:
    """
    Generate the donnees_reelles table for a given date.
    Adds random noise to predicted values and simulates battery/grid behaviour.
    """
    validate_predicted(predicted)
    rng = np.random.default_rng(cfg.seed + 1)
    real = predicted[["heure", "jour", "mois", "annee"]].copy()

    tout_real = predicted["Tout"].to_numpy() + rng.normal(0.0, 0.7, 24)
    tin_real = predicted["Tin"].to_numpy() + rng.normal(0.0, 0.35, 24)
    g_real = np.clip(predicted["G"].to_numpy() + rng.normal(0.0, 35.0, 24), 0.0, None)
    alpha_real = np.clip(
        predicted["alpha_presence_predit"].to_numpy() + rng.normal(0.0, 0.08, 24), 0.0, 1.0
    )
    pv_real = np.clip(cfg.pv_kw_peak * g_real / 1000.0 + rng.normal(0.0, 0.1, 24), 0.0, cfg.pv_kw_peak)

    pfixe = np.clip(predicted["Pfixe_predit"].to_numpy() + rng.normal(0.0, 0.12, 24), 0.1, None)
    pflex_real = np.clip(0.2 + 1.6 * alpha_real + rng.normal(0.0, 0.2, 24), 0.0, None)
    demand = pfixe + pflex_real
    pin, pgo, ebat, s = _simulate_battery_and_grid(demand, pv_real, cfg)

    cbuy_real = np.clip(predicted["Cbuy_predit"].to_numpy() + rng.normal(0.0, 0.01, 24), 0.0, None)
    csell_real = np.clip(cfg.price_beta_sell * cbuy_real + rng.normal(0.0, 0.005, 24), 0.0, None)

    real["tfrigo"] = cfg.tfrigo_c
    real["Tmin"] = cfg.tmin_c
    real["Tmax"] = cfg.tmax_c
    real["Tout_reel"] = np.round(tout_real, 3)
    real["Tin_reel"] = np.round(tin_real, 3)
    real["G"] = np.round(g_real, 3)
    real["alpha_presence_reel"] = np.round(alpha_real, 4)
    real["PV_reel"] = np.round(pv_real, 3)
    real["Pfixe"] = np.round(pfixe, 3)
    real["Pflex_reel"] = np.round(pflex_real, 3)
    real["Pin"] = pin
    real["Pgo"] = pgo
    real["Ebat"] = ebat
    real["S"] = s
    real["Cbuy_reel"] = np.round(cbuy_real, 4)
    real["Csell_reel"] = np.round(csell_real, 4)

    validate_real(real, cfg)
    return real[REAL_COLUMNS]


def generate_and_save_day(
    run_date: date,
    predicted: pd.DataFrame,
    cfg: SimulationConfig = SimulationConfig(),
    root_dir: str | Path = "energy_planner/data",
) -> dict[str, Path]:
    """
    Simulate ground-truth real data from predicted inputs and save all CSVs.

    The predicted DataFrame must be produced upstream by predict_day_inputs()
    (in energy_planner/src/prediction/predict_day.py).
    """
    real = generate_real_day(run_date=run_date, predicted=predicted, cfg=cfg)
    raw_pred, raw_real, proc_pred, proc_real, hist_pred, hist_real = save_daily_csv(
        predicted=predicted,
        real=real,
        run_date=run_date,
        root_dir=root_dir,
    )
    return {
        "raw_predicted_csv": raw_pred,
        "raw_real_csv": raw_real,
        "processed_predicted_csv": proc_pred,
        "processed_real_csv": proc_real,
        "historic_predicted_csv": hist_pred,
        "historic_real_csv": hist_real,
    }


def simulate_historical_data(
    start_date: date,
    num_days: int,
    cfg: SimulationConfig = SimulationConfig(),
    root_dir: str | Path = "energy_planner/data",
    pv_agent=None
) -> None:
    """
    Simule et sauvegarde les données sur une période de plusieurs jours consécutifs.
    Idéal pour générer rapidement un gros volume de données d'entraînement pour le ML.
    """
    from datetime import timedelta
    
    print(f"\n[*] Simulation historique sur {num_days} jours (à partir du {start_date})...")
    
    for i in range(num_days):
        current_date = start_date + timedelta(days=i)
        
        generate_and_save_day(
            run_date=current_date,
            cfg=cfg,
            root_dir=root_dir,
            pv_agent=pv_agent
        )
        