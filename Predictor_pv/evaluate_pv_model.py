import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Doit correspondre à ton entraînement
PV_FEATURES = ["hour", "day", "month", "year", "Tout", "G"]
PV_TARGET = "PV"

def evaluate_model(dataset_path: Path, model_path: Path):
    """
    Charge le modèle et le dataset, puis calcule les métriques de performance.
    """
    if not dataset_path.exists():
        raise FileNotFoundError("Le dataset est introuvable. Vérifiez les chemins.")
    if not model_path.exists():
        raise FileNotFoundError("Le modèle est introuvable. Vérifiez les chemins.")

    print(f"[*] Chargement du dataset : {dataset_path.name}")
    df = pd.read_csv(dataset_path)

    print(f"[*] Chargement du modèle : {model_path.name}")
    model = joblib.load(model_path)

    # ---------------------------------------------------------
    # BONNE PRATIQUE ML : Évaluer sur des données non vues.
    # On va prendre les 30 derniers jours du dataset pour le test.
    # ---------------------------------------------------------
    test_days = 30
    test_size = 24 * test_days 
    
    if len(df) > test_size:
        df_test = df.iloc[-test_size:].copy()
        # print(f"[*] Évaluation sur les {test_days} derniers jours ({len(df_test)} heures).")
    else:
        df_test = df.copy()
        # print(f"[*] Évaluation sur l'ensemble du dataset ({len(df_test)} heures).")

    # Préparation des données de test
    X_test = df_test[PV_FEATURES]
    y_true = df_test[PV_TARGET]

    # Prédictions
    y_pred = model.predict(X_test)
    y_pred = np.maximum(0, y_pred) # Pas de production négative

    # ---------------------------------------------------------
    # CALCUL DES MÉTRIQUES
    # ---------------------------------------------------------
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # print("\n" + "="*50)
    # print("📊 RÉSULTATS DE L'ÉVALUATION (Modèle ML PV)")
    # print("="*50)
    print(f"🔹 MAE (Erreur Moyenne Absolue) : {mae:.3f} kW")
    print(f"   -> En moyenne, le modèle se trompe de {mae*1000:.1f} Watts par heure.")
    print(f"🔹 RMSE (Erreur Quadratique)    : {rmse:.3f} kW")
    print(f"🔹 R² (Score de précision)      : {r2:.4f} (1.0 = Parfait)")
    print("="*50)

    # ---------------------------------------------------------
    # ANALYSE DES PIRES ERREURS
    # ---------------------------------------------------------
    df_test['Prediction'] = np.round(y_pred, 3)
    df_test['Erreur_Absolue'] = np.round(np.abs(y_true - y_pred), 3)
    
    pires_erreurs = df_test.sort_values(by='Erreur_Absolue', ascending=False).head(5)
    
    print("\n Les 5 plus grandes erreurs de prédiction sur la période :")
    colonnes_a_afficher = ['month', 'day', 'hour', 'G', PV_TARGET, 'Prediction', 'Erreur_Absolue']
    print(pires_erreurs[colonnes_a_afficher].to_string(index=False))


if __name__ == "__main__":
    # Ajuste ces chemins selon ton arborescence
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    
    DATASET_PATH = PROJECT_ROOT / "energy_planner" / "data" / "processed" / "synthetic_user_history.csv"
    MODEL_PATH = PROJECT_ROOT / "Predictor_pv" / "models" / "rf_pv_model.joblib"
    
    try:
        evaluate_model(DATASET_PATH, MODEL_PATH)
    except Exception as e:
        print(f"[!] Erreur : {e}")