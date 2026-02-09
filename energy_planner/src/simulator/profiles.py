from __future__ import annotations

from datetime import date
from typing import Tuple

import numpy as np
import pandas as pd


def build_time_index(run_date: date) -> pd.DataFrame:
    hours = np.arange(24, dtype=int)
    return pd.DataFrame(
        {
            "heure": hours,
            "jour": np.full(24, run_date.day, dtype=int),
            "mois": np.full(24, run_date.month, dtype=int),
            "annee": np.full(24, run_date.year, dtype=int),
        }
    )


def daylight_irradiance_w_m2(hours: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """ Simulate a daylight irradiance profile with a bell-shaped curve and some noise"""
    center = 13.0
    sigma = 3.0
    profile = np.exp(-0.5 * ((hours - center) / sigma) ** 2)
    irradiance = 900.0 * profile
    irradiance[(hours < 7) | (hours > 19)] = 0.0
    noise = rng.normal(0.0, 25.0, size=hours.shape[0])
    return np.clip(irradiance + noise, 0.0, None)


def temperature_profile(hours: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """ Simulate outdoor and indoor temperature profiles with daily patterns and some noise"""
    tout = 11.0 + 6.0 * np.sin((hours - 6) * np.pi / 12.0)
    tin = 21.0 + 1.3 * np.sin((hours - 8) * np.pi / 12.0)
    tout += rng.normal(0.0, 0.5, size=hours.shape[0])
    tin += rng.normal(0.0, 0.25, size=hours.shape[0])
    return tout, tin


def occupancy_profile(hours: np.ndarray, rng: np.random.Generator) -> np.ndarray:
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


def tariff_profile(hours: np.ndarray) -> np.ndarray:
    cbuy = np.full(24, 0.17)
    cbuy[(hours >= 7) & (hours < 11)] = 0.23
    cbuy[(hours >= 18) & (hours < 22)] = 0.29
    cbuy[(hours >= 0) & (hours < 6)] = 0.14
    return cbuy
