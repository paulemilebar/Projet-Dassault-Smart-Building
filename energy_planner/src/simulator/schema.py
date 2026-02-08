from __future__ import annotations

from dataclasses import dataclass

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
