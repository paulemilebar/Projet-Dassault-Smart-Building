from __future__ import annotations

from datetime import date
from pathlib import Path
import sys

import pandas as pd

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

from Data_Quality_agent.validation import validate_predicted, validate_real


def save_daily_csv(
    predicted: pd.DataFrame,
    real: pd.DataFrame,
    run_date: date,
    root_dir: str | Path = "energy_planner/data",
) -> tuple[Path, Path, Path, Path]:
    validate_predicted(predicted)
    validate_real(real)

    root = Path(root_dir)
    raw_dir = root / "raw"
    processed_dir = root / "processed"
    historic_dir = root / "historic"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    historic_dir.mkdir(parents=True, exist_ok=True)

    day = run_date.isoformat()
    raw_pred_path = raw_dir / f"donnees_predites_{day}.csv"
    raw_real_path = raw_dir / f"donnees_reelles_{day}.csv"
    proc_pred_path = processed_dir / f"donnees_predites_clean_{day}.csv"
    proc_real_path = processed_dir / f"donnees_reelles_clean_{day}.csv"
    hist_pred_path = historic_dir / "donnees_predites_historique.csv"
    hist_real_path = historic_dir / "donnees_reelles_historique.csv"

    #Pk on stocke les mêmes choses dans raw et processed
    predicted.to_csv(raw_pred_path, index=False)
    real.to_csv(raw_real_path, index=False)
    predicted.to_csv(proc_pred_path, index=False)
    real.to_csv(proc_real_path, index=False)
    predicted.to_csv(hist_pred_path, mode='a', header=not hist_pred_path.exists(), index=False)
    real.to_csv(hist_real_path, mode='a', header=not hist_real_path.exists(), index=False)

    return raw_pred_path, raw_real_path, proc_pred_path, proc_real_path, hist_pred_path, hist_real_path
