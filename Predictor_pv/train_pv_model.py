import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

# --- CONFIGURATION DES COLONNES ---
# ⚠️ Attention : J'utilise ici les noms de colonnes de ton nouveau dataset synthétique.
# Si tu utilises l'ancien historique, remets ['heure', 'jour', 'mois', 'annee', 'Tout', 'G']
PV_FEATURES = ["hour", "day", "month", "year", "Tout", "G"]
PV_TARGET = "PV"  # Ou "PV_reel" si tu pointes sur l'ancien fichier

def train_pv_model(
    dataset_path: Path, 
    model_save_path: Path, 
    n_estimators: int = 100, 
    random_state: int = 42
) -> RandomForestRegressor:
    """
    Charge le dataset historique, entraîne le modèle Random Forest pour le PV,
    et le sauvegarde sur le disque.
    
    :param dataset_path: Chemin vers le CSV d'entraînement (ex: dataset_entrainement_global.csv).
    :param model_save_path: Chemin où sauvegarder le modèle entraîné (.joblib).
    :return: Le modèle entraîné.
    """
    # print(f"[*] Chargement des données d'entraînement depuis : {dataset_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Le fichier {dataset_path} est introuvable.")

    df = pd.read_csv(dataset_path)

    # 1. Vérification de sécurité
    missing_cols = [col for col in PV_FEATURES + [PV_TARGET] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Colonnes manquantes dans le dataset : {missing_cols}")

    # 2. Préparation des Features (X) et Target (Y)
    X = df[PV_FEATURES]
    Y = df[PV_TARGET]

    # 3. Entraînement du modèle
    print(f"[*] Entraînement du RandomForestRegressor sur {len(df)} échantillons...")
    # L'argument n_jobs=-1 permet d'utiliser tous les cœurs de ton processeur pour aller plus vite
    model = RandomForestRegressor(
        n_estimators=n_estimators, 
        random_state=random_state, 
        n_jobs=-1
    )
    model.fit(X, Y)

    # 4. Sauvegarde du modèle sur le disque
    # On s'assure que le dossier de destination existe
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_save_path)
    
    # print(f"[+] Modèle entraîné et sauvegardé avec succès sous : {model_save_path}")
    
    return model

if __name__ == "__main__":

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATASET_PATH = PROJECT_ROOT / "energy_planner" / "data" / "processed" / "synthetic_user_history.csv"
    MODEL_PATH = PROJECT_ROOT / "Predictor_pv" / "models" / "rf_pv_model.joblib"
    
    try:
        train_pv_model(DATASET_PATH, MODEL_PATH)
    except Exception as e:
        print(f"[!] Erreur lors de l'entraînement : {e}")