from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

"""
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
REAL_COLUMNS = [
    "heure",
    "jour",
    "mois",
    "annee",
    "tfrigo",
    "Tmin",
    "Tmax",
    "Tout_reel",
    "Tin_reel",
    "G",
    "alpha_presence_reel",
    "PV_reel",
    "Pfixe",
    "Pflex_reel",
    "Pin",
    "Pgo",
    "Ebat",
    "S",
    "Cbuy_reel",
    "Csell_reel",
]

PRED_COLUMNS = [
    "heure",
    "jour",
    "mois",
    "annee",
    "Tout",
    "Tin",
    "G",
    "alpha_presence_predit",
    "PV",
    "Pfixe_predit",
    "Pflex_predit",
    "Cbuy_predit",
    "Csell_predit",
]


@dataclass(frozen=True)
class SimulationConfig:
    seed: int = 42 # Random seed for reproducibility
    
    ## Initial conditions
    ebat_initial_kwh: float = 6.0 # Initial battery energy in kWh
    
    ## Exogenous parameters
    pv_kw_peak: float = 6.0 # installed PV capacity (How much power the PV can produce at maximum irradiance)
    tfrigo_c: float = 4.0 # Fridge temperature in °C

    ## Decision constraints
    ebat_max_kwh: float = 13.5 # Maximum battery capacity in kWh
    pch_max_kw: float = 4.0 # Maximum charging power in kW
    pdis_max_kw: float = 4.0 # Maximum discharging power in kW
    tmin_c: float = 20.0 # Minimum indoor temperature in °C
    tmax_c: float = 25.0 # Maximum indoor temperature in °C
    eta_ch: float = 0.95 # Charging efficiency
    eta_dis: float = 0.95 # Discharging efficiency

    ## Objective parameters
    price_beta_sell: float = 0.6 # Beta parameter for selling price (relative to buying price)


def _build_time_index(run_date: date) -> pd.DataFrame:
    """Build a DataFrame with time-related columns for each hour of the given date."""
    hours = np.arange(24, dtype=int)
    return pd.DataFrame(
        {
            "heure": hours,
            "jour": np.full(24, run_date.day, dtype=int),
            "mois": np.full(24, run_date.month, dtype=int),
            "annee": np.full(24, run_date.year, dtype=int),
        }
    )


def _daylight_irradiance_w_m2(hours: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """ Simulate a daylight irradiance profile with a bell-shaped curve and some noise"""
    center = 13.0
    sigma = 3.0
    profile = np.exp(-0.5 * ((hours - center) / sigma) ** 2)
    irradiance = 900.0 * profile
    irradiance[(hours < 7) | (hours > 19)] = 0.0
    noise = rng.normal(0.0, 25.0, size=hours.shape[0])
    return np.clip(irradiance + noise, 0.0, None)


def _temperature_profile(hours: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """ Simulate outdoor and indoor temperature profiles with daily patterns and some noise"""

    tout = 11.0 + 6.0 * np.sin((hours - 6) * np.pi / 12.0) # Outdoor temperature profile with a peak in the afternoon
    tin = 21.0 + 1.3 * np.sin((hours - 8) * np.pi / 12.0) # Indoor temperature profile with a smaller amplitude and a slight delay
    tout += rng.normal(0.0, 0.5, size=hours.shape[0])
    tin += rng.normal(0.0, 0.25, size=hours.shape[0])
    return tout, tin


def _occupancy_profile(hours: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """ 
    Simulate occupancy with two peaks: one in the morning and one in the evening, plus some noise.
    People are present at night but with lower consumption
    """
    morning = np.exp(-0.5 * ((hours - 8) / 2.2) ** 2)
    evening = np.exp(-0.5 * ((hours - 19) / 2.5) ** 2)
    occupancy = 0.08 + 0.75 * np.maximum(morning, evening)
    occupancy[(hours >= 0) & (hours <= 5)] *= 0.35
    occupancy += rng.normal(0.0, 0.03, size=hours.shape[0])
    return np.clip(occupancy, 0.0, 1.0)



def _tariff_profile(hours: np.ndarray) -> np.ndarray:
    cbuy = np.full(24, 0.17)
    cbuy[(hours >= 7) & (hours < 11)] = 0.23
    cbuy[(hours >= 18) & (hours < 22)] = 0.29
    cbuy[(hours >= 0) & (hours < 6)] = 0.14
    return cbuy


def generate_predicted_day(run_date: date, cfg: SimulationConfig = SimulationConfig()) -> pd.DataFrame:
    """"Generate the donnees_predites table for a given date based on the simulation configuration."""
    rng = np.random.default_rng(cfg.seed)
    df = _build_time_index(run_date)
    h = df["heure"].to_numpy()

    tout, tin = _temperature_profile(h, rng)
    irradiance = _daylight_irradiance_w_m2(h, rng)
    occupancy = _occupancy_profile(h, rng)

    ## pv = pv_kw_peak * irradiance / 1000.0 with some noise and clipping to [0, pv_kw_peak]
    pv = np.clip(cfg.pv_kw_peak * irradiance / 1000.0, 0.0, cfg.pv_kw_peak)
    
    ## profile for both pfixe and pflex
    pfixe = 0.8 + 0.03 * np.maximum(0.0, 22.0 - tout) + 0.18 * (tin < cfg.tmin_c)
    pflex = 0.25 + 1.5 * occupancy

    cbuy = _tariff_profile(h)
    csell = cfg.price_beta_sell * cbuy

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


def _validate_common(df: pd.DataFrame, expected_columns: list[str], table_name: str) -> None:
    """  
    Common validation for both predicted and real data tables:
        - Check for expected columns
        - Check for 24 rows (one per hour)
        - Check for duplicate primary keys (heure, jour, mois, annee)
        - Check for NaN values
        - Check that 'heure' column contains values from 0 to 23 in order
    """
    missing = [c for c in expected_columns if c not in df.columns]
    if missing:
        raise ValueError(f"{table_name}: missing columns {missing}")

    if len(df) != 24:
        raise ValueError(f"{table_name}: expected 24 rows, got {len(df)}")

    key_dupes = df.duplicated(subset=["heure", "jour", "mois", "annee"]).sum()
    if key_dupes:
        raise ValueError(f"{table_name}: duplicate primary keys found ({key_dupes})")

    if df[expected_columns].isna().any().any():
        raise ValueError(f"{table_name}: NaN values found")

    sorted_hours = df["heure"].to_list()
    if sorted_hours != list(range(24)):
        raise ValueError(f"{table_name}: 'heure' must be 0..23 in order")

def validate_predicted(df: pd.DataFrame) -> None:
    _validate_common(df, PRED_COLUMNS, "donnees_predites")

    if (df["G"] < 0).any():
        raise ValueError("donnees_predites: G must be >= 0")
    if (df["PV"] < 0).any():
        raise ValueError("donnees_predites: PV must be >= 0")
    if ((df["alpha_presence_predit"] < 0) | (df["alpha_presence_predit"] > 1)).any():
        raise ValueError("donnees_predites: alpha_presence_predit must be in [0, 1]")
    if (df["Cbuy_predit"] < 0).any() or (df["Csell_predit"] < 0).any():
        raise ValueError("donnees_predites: prices must be >= 0")

def validate_real(df: pd.DataFrame, cfg: SimulationConfig = SimulationConfig()) -> None:
    _validate_common(df, REAL_COLUMNS, "donnees_reelles")

    if (df["G"] < 0).any():
        raise ValueError("donnees_reelles: G must be >= 0")
    if (df["PV_reel"] < 0).any():
        raise ValueError("donnees_reelles: PV_reel must be >= 0")
    if ((df["alpha_presence_reel"] < 0) | (df["alpha_presence_reel"] > 1)).any():
        raise ValueError("donnees_reelles: alpha_presence_reel must be in [0, 1]")
    if (df["Cbuy_reel"] < 0).any() or (df["Csell_reel"] < 0).any():
        raise ValueError("donnees_reelles: prices must be >= 0")
    if ((df["S"] != 0) & (df["S"] != 1)).any():
        raise ValueError("donnees_reelles: S must be binary")
    if (df["Ebat"] < 0).any() or (df["Ebat"] > cfg.ebat_max_kwh + 1e-9).any():
        raise ValueError("donnees_reelles: Ebat out of bounds")


def save_daily_csv(
    predicted: pd.DataFrame,
    real: pd.DataFrame,
    run_date: date,
    root_dir: str | Path = "energy_planner/data",
) -> tuple[Path, Path, Path, Path]:
    validate_predicted(predicted)
    validate_real(real)

    root = Path(root_dir)
    raw_dir = root / "raw"
    processed_dir = root / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    day = run_date.isoformat()

    raw_pred_path = raw_dir / f"donnees_predites_{day}.csv"
    raw_real_path = raw_dir / f"donnees_reelles_{day}.csv"
    proc_pred_path = processed_dir / f"donnees_predites_clean_{day}.csv"
    proc_real_path = processed_dir / f"donnees_reelles_clean_{day}.csv"

    predicted.to_csv(raw_pred_path, index=False)
    real.to_csv(raw_real_path, index=False)

    # MVP: processed equals validated raw data.
    predicted.to_csv(proc_pred_path, index=False)
    real.to_csv(proc_real_path, index=False)

    return raw_pred_path, raw_real_path, proc_pred_path, proc_real_path


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
