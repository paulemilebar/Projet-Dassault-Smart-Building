import argparse
import joblib
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

# --- CONFIGURATION DES COLONNES ---
PV_FEATURES = ["hour", "day", "month", "year", "Tout", "G"]
PV_TARGET = "PV"

# Chemins par défaut (calculés dynamiquement)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "energy_planner" / "data" / "processed" / "synthetic_user_history.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "Predictor_pv" / "models" / "rf_pv_model_bundle.joblib"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entraînement du modèle de prédiction PV.")
    parser.add_argument("--dataset-path", type=str, default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--model-save-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Proportion des données pour l'entraînement (ex: 0.8 = 80%)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _compute_metrics(y_true: pd.Series, y_pred) -> dict[str, float]:
    """Calcule et retourne les métriques de performance."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset_path)
    model_save_path = Path(args.model_save_path)

    print(f"[*] Chargement des données depuis : {dataset_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"[!] Le fichier {dataset_path} est introuvable.")

    df = pd.read_csv(dataset_path)

    # 1. Vérification de sécurité
    missing_cols = [col for col in PV_FEATURES + [PV_TARGET] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"[!] Colonnes manquantes dans le dataset : {missing_cols}")

    # 2. Séparation Train / Validation (Chronologique pour des séries temporelles)
    # On trie par date au cas où le CSV soit dans le désordre
    df = df.sort_values(by=["year", "month", "day", "hour"]).reset_index(drop=True)
    
    split_index = int(len(df) * args.train_ratio)
    train_df = df.iloc[:split_index]
    valid_df = df.iloc[split_index:]

    print(f"[*] Set d'entraînement : {len(train_df)} lignes")
    print(f"[*] Set de validation  : {len(valid_df)} lignes")

    X_train = train_df[PV_FEATURES]
    Y_train = train_df[PV_TARGET]
    X_valid = valid_df[PV_FEATURES]
    Y_valid = valid_df[PV_TARGET]

    # 3. Entraînement du modèle
    print(f"[*] Entraînement du RandomForestRegressor (n_estimators={args.n_estimators})...")
    model = RandomForestRegressor(
        n_estimators=args.n_estimators, 
        max_depth=args.max_depth,
        random_state=args.seed, 
        n_jobs=-1
    )
    model.fit(X_train, Y_train)

    # 4. Évaluation sur le set de validation
    print("[*] Évaluation du modèle sur les données de validation...")
    pv_pred = model.predict(X_valid)
    metrics = _compute_metrics(Y_valid, pv_pred)

    # 5. Création du Bundle et Sauvegarde
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "feature_columns": PV_FEATURES,
        "target_column": PV_TARGET,
        "metadata": {
            "n_estimators": args.n_estimators,
            "train_size": len(train_df),
            "valid_size": len(valid_df),
            "metrics": metrics,
        },
    }
    joblib.dump(bundle, model_save_path)
    
    print(f"\n[+] Bundle modèle sauvegardé avec succès sous : {model_save_path}")
    print("\n📊 Métriques de validation :")
    for name, value in metrics.items():
        print(f"    {name.upper()}: {value:.4f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[!] Erreur fatale : {e}")