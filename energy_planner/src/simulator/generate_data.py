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
        