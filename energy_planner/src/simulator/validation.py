from __future__ import annotations

import pandas as pd

from simulator.schema import PRED_COLUMNS, REAL_COLUMNS, SimulationConfig

"""  
Common validation for both predicted and real data tables:
    - Check for expected columns
    - Check for 24 rows (one per hour)
    - Check for duplicate primary keys (heure, jour, mois, annee)
    - Check for NaN values
    - Check that 'heure' column contains values from 0 to 23 in order
"""

def _validate_common(df: pd.DataFrame, expected_columns: list[str], table_name: str) -> None:
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
