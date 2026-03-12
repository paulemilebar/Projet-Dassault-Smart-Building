from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS, build_demand_feature_frame, generate_synthetic_demand_history


DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "user_demand_rf_bundle.joblib"


def load_demand_model_bundle(model_path: str | Path = DEFAULT_MODEL_PATH) -> dict:
    """ Charge le fichier qui contient à la fois les modèles de prédiction pour la demande fixe et flexible,
    ainsi que les features utilisées. 
    :param model_path: Chemin vers le fichier du bundle de modèle.
    :return: Dictionnaire contenant les modèles et les features.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Demand model bundle not found at {path}. "
            "Train it first with Predictor_demand/train_user_demand_model.py."
        )
    bundle = joblib.load(path)
    required = {"pfix_model", "pflex_model", "feature_columns", "metadata"}
    missing = required.difference(bundle)
    if missing:
        raise ValueError(f"Invalid demand model bundle. Missing keys: {sorted(missing)}")
    return bundle


@dataclass
class UserDemandForecastAgent:
    """ Agent de prédiction de la demande utilisateur. 
    Utilise des modèles de ML pour prédire la demande fixe et 
    flexible à partir de features construites à partir de données contextuelles (température, occupation, ...) 
    """
    model_path: str | Path = DEFAULT_MODEL_PATH

    def __post_init__(self) -> None:
        bundle = load_demand_model_bundle(self.model_path)
        self.pfix_model = bundle["pfix_model"]
        self.pflex_model = bundle["pflex_model"]
        self.feature_columns = list(bundle["feature_columns"])
        self.metadata = dict(bundle["metadata"])

    def predict_from_context(
        self,
        run_date: date,
        *,
        tout: np.ndarray,
        tin: np.ndarray,
        occupancy: np.ndarray,
    ) -> pd.DataFrame:
        """ Construit les features à partir des données contextuelles et utilise les modèles pour faire les prédictions de demande fixe et flexible."""
        features = build_demand_feature_frame(run_date, tout, tin, occupancy)
        model_input = features[self.feature_columns]

        pfix = self.pfix_model.predict(model_input)
        pflex = self.pflex_model.predict(model_input)

        return pd.DataFrame(
            {
                "hour": features["hour"].astype(int),
                "Pfixe_predit": np.maximum(pfix, 0.0),
                "Pflex_predit": np.maximum(pflex, 0.0),
            }
        )

    def predict_dataframe(self, feature_frame: pd.DataFrame) -> pd.DataFrame:
        missing = [col for col in self.feature_columns if col not in feature_frame.columns]
        if missing:
            raise ValueError(f"Feature frame is missing columns: {missing}")

        pfix = self.pfix_model.predict(feature_frame[self.feature_columns])
        pflex = self.pflex_model.predict(feature_frame[self.feature_columns])
        return pd.DataFrame(
            {
                "hour": feature_frame["hour"].astype(int),
                "Pfixe_predit": np.maximum(pfix, 0.0),
                "Pflex_predit": np.maximum(pflex, 0.0),
            }
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict next-day fixed and flexible user demand.")
    parser.add_argument("--run-date", type=str, required=True, help="Date in YYYY-MM-DD format.")
    parser.add_argument(
        "--output-csv",
        type=str,
        required=True,
        help="Path to the 24-row forecast CSV to write.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to the saved demand model bundle.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used to generate the synthetic context for the requested day.",
    )
    return parser.parse_args()


def _parse_date(raw: str) -> date:
    return datetime.strptime(raw, "%Y-%m-%d").date()


def main() -> None:
    args = parse_args()
    run_date = _parse_date(args.run_date)
    context_df = generate_synthetic_demand_history(
        start_date=run_date,
        num_days=1,
        base_seed=args.seed,
    )

    agent = UserDemandForecastAgent(model_path=args.model_path)
    prediction = agent.predict_from_context(
        run_date,
        tout=context_df["Tout"].to_numpy(),
        tin=context_df["Tin"].to_numpy(),
        occupancy=context_df["occupancy"].to_numpy(),
    )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction.to_csv(output_path, index=False)
    print(f"Saved demand forecast to: {output_path}")


if __name__ == "__main__":
    main()
