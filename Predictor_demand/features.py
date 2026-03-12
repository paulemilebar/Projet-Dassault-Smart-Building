from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "energy_planner" / "src"
SIMULATOR_ROOT = SRC_ROOT / "simulator"

import sys

for candidate in (PROJECT_ROOT, SRC_ROOT, SIMULATOR_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from profiles import build_time_index, occupancy_profile, temperature_profile
from schema import SimulationConfig


@dataclass(frozen=True)
class DemandFeatureConfig:
    t_out_reference_c: float = 22.0
    t_min_c: float = SimulationConfig().tmin_c


def build_demand_feature_frame(
    run_date: date,
    tout: np.ndarray,
    tin: np.ndarray,
    occupancy: np.ndarray,
    *,
    cfg: DemandFeatureConfig = DemandFeatureConfig(),
) -> pd.DataFrame:
    if not (len(tout) == len(tin) == len(occupancy) == 24):
        raise ValueError("tout, tin, occupancy must all contain 24 hourly values.")

    base = build_time_index(run_date).rename(columns={"heure": "hour"})

    frame = base.copy()
    frame = frame.rename(
        columns={
            "jour": "day",
            "mois": "month",
            "annee": "year",
        }
    )
    frame["Tout"] = tout
    frame["Tin"] = tin
    frame["occupancy"] = occupancy
    frame["heating_gap_outdoor"] = np.maximum(0.0, cfg.t_out_reference_c - tout)
    frame["below_tmin_flag"] = (tin < cfg.t_min_c).astype(int)
    return frame


def compute_targets(
    tout: np.ndarray,
    tin: np.ndarray,
    occupancy: np.ndarray,
    *,
    sim_cfg: SimulationConfig = SimulationConfig(),
) -> tuple[np.ndarray, np.ndarray]:
    """ Calcule les cibles de demande fixe (pfix) et flexible (pflex) à partir des profils de température, notre modele doit 
    apprendre à associer les features à ces targets."""
    pfix = 0.8 + 0.03 * np.maximum(0.0, 22.0 - tout) + 0.18 * (tin < sim_cfg.tmin_c)
    pflex = 0.25 + 1.5 * occupancy
    return pfix.astype(float), pflex.astype(float)


def generate_synthetic_demand_history(
    start_date: date,
    num_days: int,
    *,
    base_seed: int = 42,
    feature_cfg: DemandFeatureConfig = DemandFeatureConfig(),
    sim_cfg: SimulationConfig = SimulationConfig(),
) -> pd.DataFrame:
    """ Génère un historique synthétique de demande pour une période donnée. Pour chaque jour,
    génère des profils de température extérieure, intérieure et d'occupation, puis construit un
    DataFrame de caractéristiques horaires et calcule les cibles de demande fixe et flexible.
    Args:
        start_date: Date de début de la période de génération.
        num_days: Nombre de jours à générer.
        base_seed: Graine de base pour la génération aléatoire (pour reproductibilité).
        feature_cfg: Configuration pour le calcul des caractéristiques de demande.
        sim_cfg: Configuration de simulation utilisée pour calculer les cibles de demande.
    Returns:
        Un DataFrame contenant les caractéristiques horaires et les cibles de demande pour chaque jour."""
    rows: list[pd.DataFrame] = []
    for offset in range(num_days):
        day = start_date + timedelta(days=offset)
        rng = np.random.default_rng(base_seed + offset)
        hours = np.arange(24, dtype=int)
        tout, tin = temperature_profile(hours, rng)
        occupancy = occupancy_profile(hours, rng)
        pfix, pflex = compute_targets(tout, tin, occupancy, sim_cfg=sim_cfg)

        frame = build_demand_feature_frame(day, tout, tin, occupancy, cfg=feature_cfg)
        frame["Pfixe"] = np.round(pfix, 6)
        frame["Pflex"] = np.round(pflex, 6)
        rows.append(frame)

    return pd.concat(rows, ignore_index=True)


FEATURE_COLUMNS = [
    "hour",
    "day",
    "month",
    "year",
    "Tout",
    "Tin",
    "occupancy",
    "heating_gap_outdoor",
    "below_tmin_flag",
]
