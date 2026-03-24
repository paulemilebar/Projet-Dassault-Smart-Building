from __future__ import annotations

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Project root on sys.path so top-level packages resolve.
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from Predictor_demand.features import FEATURE_COLUMNS

# Columns produced by generate_real_day() that we need for demand retraining.
_REAL_FEATURE_MAP = {
    "hour":                "heure",
    "day":                 "jour",
    "month":               "mois",
    "year":                "annee",
    "Tout":                "Tout_reel",
    "Tin":                 "Tin_reel",
    "occupancy":           "alpha_presence_reel",
}
_TMIN_DEFAULT = 20.0  # matches SimulationConfig.tmin_c


# ---------------------------------------------------------------------------
# 1. History append
# ---------------------------------------------------------------------------

def append_to_history(new_rows: pd.DataFrame, history_csv: Path) -> None:
    """
    Append *new_rows* (REAL_COLUMNS schema) to the rolling history CSV.
    Creates the file with a header if it does not exist yet.
    """
    history_csv = Path(history_csv)
    history_csv.parent.mkdir(parents=True, exist_ok=True)
    new_rows.to_csv(
        history_csv,
        mode="a",
        header=not history_csv.exists(),
        index=False,
    )


# ---------------------------------------------------------------------------
# 2. Demand model update
# ---------------------------------------------------------------------------

def _build_demand_features(real_df: pd.DataFrame, tmin_c: float = _TMIN_DEFAULT) -> pd.DataFrame:
    """Convert REAL_COLUMNS rows into the demand feature frame expected by the RF bundle."""
    feat = pd.DataFrame()
    for feat_col, real_col in _REAL_FEATURE_MAP.items():
        feat[feat_col] = real_df[real_col].values
    tout = real_df["Tout_reel"].values
    tin = real_df["Tin_reel"].values
    feat["heating_gap_outdoor"] = np.maximum(0.0, 22.0 - tout)
    feat["below_tmin_flag"] = (tin < tmin_c).astype(int)
    return feat[FEATURE_COLUMNS]


def update_demand_models(
    history_csv: Path,
    model_path: Path,
    window_days: int = 14,
    tmin_c: float = _TMIN_DEFAULT,
) -> None:
    """
    Retrain the demand RF bundle on the last *window_days* days of real history.

    Loads the existing bundle to preserve hyperparameters, retrains both
    pfix_model and pflex_model, and saves the updated bundle in-place.

    Args:
        history_csv:  Path to the rolling real-data CSV (REAL_COLUMNS schema).
        model_path:   Path to the joblib bundle to update.
        window_days:  Rolling window size in days (1 day = 24 rows).
        tmin_c:       Indoor temperature threshold used to compute below_tmin_flag.
    """
    history_csv = Path(history_csv)
    model_path = Path(model_path)

    if not history_csv.exists():
        print("[update_demand_models] History CSV not found — skipping.")
        return
    if not model_path.exists():
        print("[update_demand_models] Model bundle not found — skipping.")
        return

    df = pd.read_csv(history_csv)
    n_rows = window_days * 24
    window = df.tail(n_rows).reset_index(drop=True)

    if len(window) < 24:
        print(f"[update_demand_models] Not enough rows ({len(window)}) — skipping.")
        return

    X = _build_demand_features(window, tmin_c=tmin_c)
    y_fix = window["Pfixe"].values
    y_flex = window["Pflex_reel"].values

    bundle = joblib.load(model_path)
    # Reuse existing hyperparameters from the bundle's stored models.
    pfix_params = bundle["pfix_model"].get_params()
    pflex_params = bundle["pflex_model"].get_params()

    pfix_model = RandomForestRegressor(**pfix_params).fit(X, y_fix)
    pflex_model = RandomForestRegressor(**pflex_params).fit(X, y_flex)

    bundle["pfix_model"] = pfix_model
    bundle["pflex_model"] = pflex_model
    joblib.dump(bundle, model_path)
    print(f"[update_demand_models] Retrained on {len(window)} rows ({window_days} days). Bundle saved.")


# ---------------------------------------------------------------------------
# 3. PV model update
# ---------------------------------------------------------------------------

# Mapping from real-history CSV columns → MLPVPredictor feature/target names.
_PV_REAL_COL_MAP = {
    "heure":    "hour",
    "jour":     "day",
    "mois":     "month",
    "annee":    "year",
    "Tout_reel": "Tout",
    "PV_reel":  "PPV",
}


def update_pv_model(
    history_csv: Path,
    pv_predictor,
    window_days: int = 14,
) -> None:
    """
    Force-retrain the ML component of *pv_predictor* (a HybridPVPredictor)
    on the last *window_days* days of real history.

    The real-history CSV uses French column names; a temporary renamed CSV is
    written so MLPVPredictor.train() can read the correct feature/target columns.

    Args:
        history_csv:   Path to the rolling real-data CSV (REAL_COLUMNS schema).
        pv_predictor:  HybridPVPredictor instance (from Predictor_pv).
        window_days:   Rolling window in days used for retraining.
    """
    import tempfile
    from datetime import date as _date

    history_csv = Path(history_csv)
    if not history_csv.exists():
        print("[update_pv_model] History CSV not found — skipping.")
        return

    # Take the rolling window and rename columns to match MLPVPredictor's schema.
    df = pd.read_csv(history_csv)
    window = df.tail(window_days * 24).rename(columns=_PV_REAL_COL_MAP)

    ml = pv_predictor.ml  # MLPVPredictor

    if len(window) < ml.min_samples:
        print(f"[update_pv_model] Not enough PV history ({len(window)} rows) — skipping.")
        return

    # Write renamed window to a temp CSV so train() can read it normally.
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        tmp_path = Path(f.name)
    window.to_csv(tmp_path, index=False)

    # Save originals and override for a forced retrain.
    original_dynamic_retrain = ml.dynamic_retrain
    original_last_train_date = ml.last_train_date
    original_dataset_csv     = ml.dataset_csv
    original_min_samples     = ml.min_samples

    ml.dynamic_retrain = True
    ml.last_train_date = None
    ml.dataset_csv     = tmp_path
    ml.min_samples     = 1  # window already filtered above

    trained = ml.train(current_date=_date.today())

    # Restore all originals and clean up temp file.
    ml.dynamic_retrain = original_dynamic_retrain
    ml.last_train_date = original_last_train_date
    ml.dataset_csv     = original_dataset_csv
    ml.min_samples     = original_min_samples
    tmp_path.unlink(missing_ok=True)

    if trained:
        print(f"[update_pv_model] ML PV model retrained on {window_days}-day window.")
    else:
        print("[update_pv_model] Retraining failed — check history data.")
