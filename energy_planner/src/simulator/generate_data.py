from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from simulator.profiles import (
    build_time_index,
    daylight_irradiance_w_m2,
    occupancy_profile,
    tariff_profile,
    temperature_profile,
)
from simulator.schema import PRED_COLUMNS, REAL_COLUMNS, SimulationConfig
from simulator.storage import save_daily_csv
from simulator.validation import validate_predicted, validate_real


## TO DO : change that because there is a problem of path when laucing the simulation. QUICK FIX HERE:
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from Predictor_agent.predictor_electricity_price import predict_next_24h_open_dpe, OpenDpeConfig

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


def generate_predicted_day(run_date: date, cfg: SimulationConfig = SimulationConfig()) -> pd.DataFrame:
    """"Generate the donnees_predites table for a given date based on the simulation configuration."""
    rng = np.random.default_rng(cfg.seed)
    df = build_time_index(run_date)
    h = df["heure"].to_numpy()

    tout, tin = temperature_profile(h, rng)
    irradiance = daylight_irradiance_w_m2(h, rng)
    occupancy = occupancy_profile(h, rng)

    pv = np.clip(cfg.pv_kw_peak * irradiance / 1000.0, 0.0, cfg.pv_kw_peak)
    pfixe = 0.8 + 0.03 * np.maximum(0.0, 22.0 - tout) + 0.18 * (tin < cfg.tmin_c)
    pflex = 0.25 + 1.5 * occupancy
    
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
    s = np.zeros(24, dtype=int) # Binary variable indicating if we import from the grid (1) or not (0)

    e_prev = cfg.ebat_initial_kwh

    for t in range(24):
        net = demand_kw[t] - pv_kw[t]

        if net <= 0: ## i.e we have excess PV production that can be used to charge the battery or sell to the grid
            surplus = -net
            ## battery first, grid export second

            # TODO
            # Maybe we can be smarter and sell to the grid during high price hours 
            # instead of charging the battery at maximum power ? 
            # For now we just prioritize the battery for simplicity 

            charge = min(surplus, cfg.pch_max_kw, (cfg.ebat_max_kwh - e_prev) / cfg.eta_ch)
            e_now = min(cfg.ebat_max_kwh, e_prev + charge * cfg.eta_ch)
            sell = max(0.0, surplus - charge)
            pin[t] = 0.0
            pgo[t] = sell
        else: ## we have a net demand that can be met by discharging the battery or importing from the grid
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
    Generate the donnees_reelles table for a given date based on the predicted values and the simulation configuration.
    We add random noise to the predicted values to create the real values
    and we simulate the battery and grid behavior based on the real demand and PV production.
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
    cfg: SimulationConfig = SimulationConfig(),
    root_dir: str | Path = "energy_planner/data",
) -> dict[str, Path]:
    predicted = generate_predicted_day(run_date=run_date, cfg=cfg)
    real = generate_real_day(run_date=run_date, predicted=predicted, cfg=cfg)
    raw_pred, raw_real, proc_pred, proc_real = save_daily_csv(
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
    }
