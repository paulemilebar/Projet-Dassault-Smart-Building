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
    seed: int = 42
    pv_kw_peak: float = 6.0
    price_beta_sell: float = 0.6
    ebat_max_kwh: float = 13.5
    pch_max_kw: float = 4.0
    pdis_max_kw: float = 4.0
    eta_ch: float = 0.95
    eta_dis: float = 0.95
    ebat_initial_kwh: float = 6.0
    tfrigo_c: float = 4.0
    tmin_c: float = 20.0
    tmax_c: float = 25.0


def _build_time_index(run_date: date) -> pd.DataFrame:
    hours = np.arange(24, dtype=int)
    return pd.DataFrame(
        {
            "heure": hours,
            "jour": np.full(24, run_date.day, dtype=int),
            "mois": np.full(24, run_date.month, dtype=int),
            "annee": np.full(24, run_date.year, dtype=int),
        }
    )


