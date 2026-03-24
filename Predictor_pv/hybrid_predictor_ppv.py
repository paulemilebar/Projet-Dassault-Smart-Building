import pandas as pd
import requests
import joblib
import numpy as np
from datetime import date, timedelta
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

class WeatherProvider:
    def __init__(self, latitude: float, longitude: float):
        self.latitude = latitude
        self.longitude = longitude

    def get_forecast(self, target_date: date) -> pd.DataFrame:
        """Récupère la météo pour un jour précis."""
        date_str = target_date.strftime("%Y-%m-%d")
        url = (f"https://api.open-meteo.com/v1/forecast?"
               f"latitude={self.latitude}&longitude={self.longitude}&"
               f"hourly=temperature_2m,shortwave_radiation&"
               f"timezone=auto&start_date={date_str}&end_date={date_str}")
        
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        df = pd.DataFrame({
            'time': pd.to_datetime(data['hourly']['time']),
            'Tout': data['hourly']['temperature_2m'],
            'G': data['hourly']['shortwave_radiation']
        })
        
        # ⚠️ MODIFICATION : Traduction en anglais pour matcher avec le dataset ML
        df['hour'] = df['time'].dt.hour
        df['day'] = df['time'].dt.day
        df['month'] = df['time'].dt.month
        df['year'] = df['time'].dt.year
        
        return df


class PhysicalPVPredictor:
    def __init__(self, p_stc: float, beta: float, noct: float, nb_panels: int):
        self.p_stc = p_stc
        self.beta = beta
        self.noct = noct
        self.nb_panels = nb_panels

    def predict(self, forecast_df: pd.DataFrame) -> pd.DataFrame:
        result_df = forecast_df.copy()
        
        T_cell = result_df['Tout'] + (self.noct - 20) / 800 * result_df['G']
        P_panel = self.p_stc * (result_df['G'] / 1000.0) * (1 + self.beta * (T_cell - 25.0))
        P_panel = np.maximum(P_panel, 0)

        P_total = self.nb_panels * P_panel
        
        result_df['PV'] = P_total / 1000.0  # W → kW
        
        return result_df


class MLPVPredictor:
    def __init__(self, dataset_csv: Path, model_path: Path, min_samples: int = 720, delta_train_days: int = 7, dynamic_retrain: bool = False): # <-- Nouvel argument ici
        self.dataset_csv = dataset_csv
        self.model_path = model_path
        self.min_samples = min_samples
        self.delta_train = timedelta(days=delta_train_days)
        
        self.dynamic_retrain = dynamic_retrain # <-- On stocke le choix
        
        self.last_train_date = None
        self.features = ['hour', 'day', 'month', 'year', 'Tout', 'G']
        self.target = 'PV'

    @property
    def is_trained(self) -> bool:
        return self.model_path.exists()

    def train(self, current_date: date) -> bool:
        # --- NOUVEAU COUPE-CIRCUIT ---
        if self.is_trained and not self.dynamic_retrain:
            # Si le modèle existe sur le disque et qu'on a interdit le ré-entraînement,
            # on arrête tout de suite la fonction et on signale que tout va bien.
            if self.last_train_date is None: 
                self.last_train_date = current_date # Juste pour l'initialisation
            return True

        # --- ANCIENNE LOGIQUE DE RÉ-ENTRAÎNEMENT (Maintenue si dynamic_retrain=True) ---
        if self.last_train_date is not None:
            jours_ecoules = current_date - self.last_train_date
            if jours_ecoules < self.delta_train:
                return True

        # Déclenchement de l'entraînement
        if not self.dataset_csv.exists():
            print(f"[ML Agent] ⚠️ Impossible d'entraîner : dataset introuvable.")
            return False

        df = pd.read_csv(self.dataset_csv)
        if len(df) < self.min_samples:
            print(f"[ML Agent] ⚠️ Pas assez de données pour entraîner.")
            return False

        print(f"[ML Agent] Entraînement du modèle RandomForest déclenché (Date : {current_date})...")
        X = df[self.features]
        Y = df[self.target]

        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X, Y)
        
        import joblib
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, self.model_path)
        
        self.last_train_date = current_date
        print(f"[ML Agent] [+] Modèle sauvegardé avec succès ({self.model_path.name})")
        return True

    def predict(self, forecast_df: pd.DataFrame) -> pd.DataFrame:
        # ... (ta fonction predict reste strictement identique) ...
        if not self.is_trained:
            raise RuntimeError("Le modèle ML n'est pas entraîné et aucun fichier .joblib n'existe.")
        
        import joblib
        model = joblib.load(self.model_path)
        
        result_df = forecast_df.copy()
        X_pred = result_df[self.features]
        
        predictions = model.predict(X_pred)
        import numpy as np
        result_df['PV'] = np.round(np.maximum(0, predictions), 3)
        return result_df


class HybridPVPredictor:
    def __init__(self, physical_predictor, ml_predictor, weather_provider):
        self.physical = physical_predictor
        self.ml = ml_predictor
        self.weather = weather_provider

    def predict_for_day(self, target_date: date, mode: str = "ML") -> pd.DataFrame:
        """
        Prédit la production PV pour une journée donnée.
        :param target_date: La date de prédiction.
        :param mode: "ML" pour le Machine Learning, "Physique" pour le modèle mathématique.
        """
        print(f"\n[Hybrid Agent] Récupération de la météo pour le {target_date}...")
        forecast_df = self.weather.get_forecast(target_date)

        # --- LOGIQUE DE CHOIX EXPLICITE ---
        if mode == "ML":
            # Sécurité : on vérifie que le modèle est bien sur le disque
            if not self.ml.is_trained:
                raise RuntimeError("[!] Mode ML demandé, mais aucun fichier .joblib n'a été trouvé. Lance l'entraînement d'abord !")
            
            print("[Hybrid Agent] Utilisation explicite du modèle ML.")
            result_df = self.ml.predict(forecast_df)
            result_df['prediction_mode'] = 'ML'

        elif mode == "Physique":
            print("[Hybrid Agent] Utilisation explicite du modèle Physique.")
            result_df = self.physical.predict(forecast_df)
            result_df['prediction_mode'] = 'Physique'

        else:
            raise ValueError(f"[!] Mode inconnu : '{mode}'. Les options valides sont 'ML' ou 'Physique'.")

        return result_df


# --- EXEMPLE D'UTILISATION ---
if __name__ == "__main__":
    from pathlib import Path
    from datetime import date, timedelta
    
    PROJECT_ROOT = Path(__file__).resolve().parent.parent # Adapte selon ton arborescence

    # Paramètres du panneau
    P_STC = 3000  
    BETA = -0.004 
    NOCT = 45
    NB_PANELS = 1
    
    # Coordonnées du smart building (Paris)
    LAT = 48.8566
    LON = 2.3522
    
    # Nouveaux chemins
    dataset_path = PROJECT_ROOT / "energy_planner" / "data" / "processed" / "synthetic_user_history.csv"
    model_path = PROJECT_ROOT / "Predictor_pv" / "models" / "rf_pv_model.joblib"
    
    # Instanciation
    weather_api = WeatherProvider(latitude=LAT, longitude=LON)
    phys_agent = PhysicalPVPredictor(p_stc=P_STC, beta=BETA, noct=NOCT, nb_panels=NB_PANELS)
    
    # Le ML Agent utilise maintenant les deux chemins (données + modèle) et est figé
    ml_agent = MLPVPredictor(
        dataset_csv=dataset_path, 
        model_path=model_path,
        dynamic_retrain=False # <-- On fige le modèle !
    )
    
    smart_agent = HybridPVPredictor(
        physical_predictor=phys_agent, 
        ml_predictor=ml_agent, 
        weather_provider=weather_api
    )

    # Exécution de la prédiction pour demain
    demain = date.today() + timedelta(days=1)
    
    # --- LA NOUVELLE LOGIQUE DE TEST ---
    
    # 1. On lance la prédiction en mode ML
    previsions_ml = smart_agent.predict_for_day(demain, mode="ML")
    
    # 2. On lance la prédiction en mode Physique
    previsions_phys = smart_agent.predict_for_day(demain, mode="Physique")

    # --- AFFICHAGE COMPARATIF ---
    print("\n--- RÉSULTATS DE LA PRÉDICTION (COMPARATIF) ---")
    
    # On filtre sur les heures de jour (8h - 18h)
    filtre_jour = (previsions_ml['hour'] >= 8) & (previsions_ml['hour'] <= 18)
    journee_ml = previsions_ml[filtre_jour]
    journee_phys = previsions_phys[filtre_jour]
    
    # Création d'un tableau comparatif clair
    df_comparatif = journee_ml[['time', 'Tout', 'G']].copy()
    df_comparatif['PV_ML'] = journee_ml['PV']
    df_comparatif['PV_Physique'] = journee_phys['PV']

    df_comparatif['Ecart (kW)'] = np.round(df_comparatif['PV_ML'] - df_comparatif['PV_Physique'], 3)
    
    print(df_comparatif.to_string(index=False))