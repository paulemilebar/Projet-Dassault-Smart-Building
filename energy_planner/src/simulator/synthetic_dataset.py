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

from profiles import build_time_index, occupancy_profile, temperature_profile, daylight_irradiance_w_m2
from schema import SimulationConfig

@dataclass(frozen=True)
class DemandFeatureConfig:
    t_out_reference_c: float = 22.0
    t_min_c: float = SimulationConfig().tmin_c


def build_feature_frame(
    run_date: date,
    tout: np.ndarray,
    tin: np.ndarray,
    occupancy: np.ndarray,
    irradiance: np.ndarray,
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
    frame["G"] = irradiance
    frame["occupancy"] = occupancy
    frame["heating_gap_outdoor"] = np.maximum(0.0, cfg.t_out_reference_c - tout)
    frame["below_tmin_flag"] = (tin < cfg.t_min_c).astype(int)
    return frame


def compute_targets(
    tout: np.ndarray,
    tin: np.ndarray,
    occupancy: np.ndarray,
    irradiance: np.ndarray, 
    *,
    sim_cfg: SimulationConfig = SimulationConfig(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Calcule les cibles de demande (pfix, pflex) et de production (ppv). """
    
    # 1. Cibles de demande
    pfix = 0.8 + 0.03 * np.maximum(0.0, 22.0 - tout) + 0.18 * (tin < sim_cfg.tmin_c)
    pflex = 0.25 + 1.5 * occupancy
    
    # 2. Cible de production PV
    # Facteur de perte due à la température (ex: -0.4% par degré au-dessus de 25°C)
    '''coeff_temp = 1.0 - 0.004 * (tout - 25.0)
    coeff_temp = np.clip(coeff_temp, 0.8, 1.1) 

    # Nouvelle formule physique + un peu de bruit
    bruit = np.random.normal(1.0, 0.02, size=len(irradiance))
    ppv_physique = (sim_cfg.pv_kw_peak * irradiance / 1000.0) * coeff_temp * bruit

    ppv = np.clip(ppv_physique, 0.0, sim_cfg.pv_kw_peak)'''
    ppv = np.clip(sim_cfg.pv_kw_peak * irradiance / 1000.0, 0.0, sim_cfg.pv_kw_peak)
    
    return pfix.astype(float), pflex.astype(float), ppv.astype(float)

def generate_synthetic_history(
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
        irradiance = daylight_irradiance_w_m2(hours, rng)
        pfix, pflex, ppv = compute_targets(tout, tin, occupancy, irradiance, sim_cfg=sim_cfg)
        frame = build_feature_frame(day, tout, tin, occupancy, irradiance, cfg=feature_cfg)
        frame["Pfixe"] = np.round(pfix, 6)
        frame["Pflex"] = np.round(pflex, 6)
        frame["PV"] = np.round(ppv, 3)
        rows.append(frame)

    return pd.concat(rows, ignore_index=True)


FEATURE_COLUMNS = [
    "hour",
    "day",
    "month",
    "year",
    "Tout",
    "Tin",
    "G",
    "PV",
    "occupancy",
    "heating_gap_outdoor",
    "below_tmin_flag",
]


if __name__ == "__main__":
    from pathlib import Path
    import sys
    
    print("\n[*] Exécution directe de synthetic_dataset.py")
    print("[*] Génération de la base de données synthétique (365 jours)...")
    
    # 1. Gestion propre des chemins
    # __file__ = synthetic_dataset.py
    # parent[0] = simulator | parent[1] = src | parent[2] = energy_planner
    CURRENT_FILE = Path(__file__).resolve()
    ENERGY_PLANNER_DIR = CURRENT_FILE.parents[2]
    
    OUTPUT_DIR = ENERGY_PLANNER_DIR / "data" / "processed"
    OUTPUT_FILE = OUTPUT_DIR / "synthetic_user_history.csv"
    
    # On s'assure que le dossier de destination existe
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 2. Paramétrage des dates
    from datetime import date, timedelta
    start_date = date.today() - timedelta(days=100)
    
    # 3. Appel de la fonction (qui est dans ce même fichier, donc pas besoin d'import !)
    df = generate_synthetic_history(start_date=start_date, num_days=100)
    
    # 4. Sauvegarde
    df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"[+] Succès ! {len(df)} lignes générées.")
    print(f"[+] Fichier sauvegardé ici : {OUTPUT_FILE}\n")
