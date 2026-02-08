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

