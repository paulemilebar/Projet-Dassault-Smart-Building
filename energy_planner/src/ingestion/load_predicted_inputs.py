from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd


# Required columns in the simulator predicted CSV.
_REQUIRED_PREDICTED_COLUMNS = [
    "heure",
    "jour",
    "mois",
    "annee",
    "PV",
    "Pfixe_predit",
    "Pflex_predit",
    "Cbuy_predit",
    "Csell_predit",
]


def _build_predicted_csv_path(run_date: date, data_root: str | Path, processed: bool = True) -> Path:
    """
    Build the expected predicted-data CSV path for a given run date.

    processed=True  -> data/processed/donnees_predites_clean_YYYY-MM-DD.csv
    processed=False -> data/raw/donnees_predites_YYYY-MM-DD.csv
    """
    root = Path(data_root)
    day = run_date.isoformat()
    if processed:
        return root / "processed" / f"donnees_predites_clean_{day}.csv"
    return root / "raw" / f"donnees_predites_{day}.csv"


def _validate_predicted_inputs(df: pd.DataFrame) -> None:
    """Validate the minimum shape/rules required by the optimizer input contract."""
    missing = [c for c in _REQUIRED_PREDICTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required predicted columns: {missing}")

    if len(df) != 24:
        raise ValueError(f"Predicted input must have 24 rows, got {len(df)}")

    hours = df["heure"].tolist()
    if hours != list(range(24)):
        raise ValueError("Column 'heure' must be exactly 0..23 in order")

    numeric_cols = ["PV", "Pfixe_predit", "Pflex_predit", "Cbuy_predit", "Csell_predit"]
    if df[numeric_cols].isna().any().any():
        raise ValueError("Predicted input has NaN values in optimizer-critical columns")


def load_predicted_inputs(
    run_date: date,
    data_root: str | Path = "energy_planner/data",
    prefer_processed: bool = True,
) -> pd.DataFrame:
    """
    Load the next-day predicted inputs and return a normalized optimizer-ready DataFrame.

    Output columns:
    - hour
    - PV_pred_kW
    - Pfix_pred_kW
    - Pflex_pred_kW
    - Cbuy_pred_eur_per_kWh
    - Csell_pred_eur_per_kWh
    """
    primary_path = _build_predicted_csv_path(run_date, data_root, processed=prefer_processed)
    fallback_path = _build_predicted_csv_path(run_date, data_root, processed=not prefer_processed)

    if primary_path.exists():
        source_path = primary_path
    elif fallback_path.exists():
        source_path = fallback_path
    else:
        raise FileNotFoundError(
            f"No predicted file found for {run_date.isoformat()} in {primary_path.parent} or {fallback_path.parent}"
        )

    df = pd.read_csv(source_path)
    _validate_predicted_inputs(df)

    # Keep a strict and simple contract for optimization modules.
    out = pd.DataFrame(
        {
            "hour": df["heure"].astype(int),
            "PV_pred_kW": df["PV"].astype(float),
            "Pfix_pred_kW": df["Pfixe_predit"].astype(float),
            "Pflex_pred_kW": df["Pflex_predit"].astype(float),
            "Cbuy_pred_eur_per_kWh": df["Cbuy_predit"].astype(float),
            "Csell_pred_eur_per_kWh": df["Csell_predit"].astype(float),
        }
    )
    return out
