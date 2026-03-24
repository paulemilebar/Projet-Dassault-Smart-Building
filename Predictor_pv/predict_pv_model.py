import pandas as pd
import numpy as np
import joblib
from pathlib import Path

PV_FEATURES = ["hour", "day", "month", "year", "Tout", "G"]

def load_pv_model(model_path: Path):
    """
    Charge le modèle Machine Learning préalablement entraîné et sauvegardé.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"[!] Modèle introuvable : {model_path}\n"
            f"Veuillez lancer 'train_pv_model.py' pour générer le modèle d'abord."
        )
    return joblib.load(model_path)

def predict_pv_production(model, forecast_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prend le modèle chargé et un DataFrame de prévisions météo (sur 24h par exemple),
    et ajoute/met à jour la colonne 'PPV' avec les prédictions.
    """
    # 1. Vérification de sécurité
    missing_cols = [col for col in PV_FEATURES if col not in forecast_df.columns]
    if missing_cols:
        raise ValueError(f"Impossible de prédire. Colonnes manquantes dans les prévisions : {missing_cols}")

    # 2. Extraction des Features
    X_pred = forecast_df[PV_FEATURES]

    # 3. Prédiction
    predictions = model.predict(X_pred)

    # 4. Formatage du résultat
    result_df = forecast_df.copy()
    
    # On utilise np.clip pour s'assurer que le modèle ne sort jamais une puissance négative
    result_df["PV"] = np.round(np.clip(predictions, 0.0, None), 3)
    
    return result_df


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    MODEL_PATH = PROJECT_ROOT / "Predictor_pv" / "models" / "rf_pv_model.joblib"
    
    try:
        # 1. Chargement du modèle
        my_model = load_pv_model(MODEL_PATH)
        
        # 2. Création d'un faux DataFrame de prévision (pour le test)
        dummy_data = {
            "hour": [10, 13, 22],
            "day": [15, 15, 15],
            "month": [6, 6, 6],
            "year": [2024, 2024, 2024],
            "Tout": [22.5, 26.0, 18.0],
            "G": [600.0, 950.0, 0.0]  # Soleil moyen, Fort soleil, Nuit
        }
        df_forecast = pd.DataFrame(dummy_data)
        
        # 3. Prédiction
        df_result = predict_pv_production(my_model, df_forecast)
        
        print("\n[+] Résultats de la prédiction :")
        print(df_result[["hour", "G", "PV"]])
        
    except Exception as e:
        print(f"[!] Erreur : {e}")