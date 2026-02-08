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
    pv_kw_peak: float = 6.0 # Peak PV generation in kW
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

