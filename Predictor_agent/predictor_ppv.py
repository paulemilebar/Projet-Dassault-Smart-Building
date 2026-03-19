import pandas as pd
import requests
from datetime import date, timedelta
import numpy as np
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
        
        df['heure'] = df['time'].dt.hour
        df['jour'] = df['time'].dt.day
        df['mois'] = df['time'].dt.month
        df['annee'] = df['time'].dt.year
        
        return df


class PhysicalPVPredictor:
    def __init__(self, p_stc: float, beta: float, noct: float, nb_panels: int):
        """
        Initialise l'agent prédicteur basé sur le modèle physique.
        p_stc: Puissance nominale du panneau sous conditions standards (en Watts).
        beta: Coefficient de température de puissance (ex: -0.004 pour -0.4%/°C).
        noct: Nominal Operating Cell Temperature (measured at G=800W/m², Tout=20°C).
        nb_panels: nbre de panneaux pv.
        latitude: Latitude du smart building (pour l'API météo).
        longitude: Longitude du smart building.
        """
        self.p_stc = p_stc
        self.beta = beta
        self.noct = noct
        self.nb_panels = nb_panels

    def predict(self, forecast_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prédit la production PV à partir d'un DataFrame de prévisions météo.
        Attend les colonnes 'Tout' (Température) et 'G' (Irradiance).
        """
        # On travaille sur une copie pour éviter de modifier le DataFrame original
        result_df = forecast_df.copy()
        
        T_cell = result_df['Tout'] + (self.noct - 20) / 800 * result_df['G']
        P_panel = self.p_stc * (result_df['G'] / 1000.0) * (1 + self.beta * (T_cell - 25.0))
        P_panel = np.maximum(P_panel, 0)

        # Puissance totale installation
        P_total = self.nb_panels * P_panel
        
        result_df['PV'] = P_total
        
        return result_df


class MLPVPredictor:
    def __init__(self, historic_real_csv: Path, min_samples: int = 720, delta_train_days: int = 7):
        """
        :param historic_real_csv: Chemin vers l'historique.
        :param min_samples: Nombre de lignes minimum pour le premier entraînement.
        :param delta_train_days: Nombre de jours à attendre avant de réentraîner le modèle.
        """
        self.historic_real_csv = historic_real_csv
        self.min_samples = min_samples
        self.delta_train = timedelta(days=delta_train_days)
        
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False
        self.last_train_date = None  # Mémorise la date du dernier entraînement
        self.features = ['heure', 'jour', 'mois', 'annee', 'Tout', 'G']

    def train(self, current_date: date) -> bool:
        """
        Entraîne le modèle uniquement s'il n'a jamais été entraîné, 
        ou si le délai delta_train est dépassé.
        """
        # --- VERIFICATION DU DELTA TRAIN ---
        if self.is_trained and self.last_train_date is not None:
            jours_ecoules = current_date - self.last_train_date
            if jours_ecoules < self.delta_train:
                return True

        # --- PROCESSUS D'ENTRAINEMENT ---
        if not self.historic_real_csv.exists():
            return False

        df = pd.read_csv(self.historic_real_csv)
        if len(df) < self.min_samples:
            return False

        if 'Tout_reel' in df.columns:
            df = df.rename(columns={'Tout_reel': 'Tout'})

        X = df[self.features]
        Y = df['PV_reel']

        print(f"[ML Agent] Entraînement du modèle déclenché (Date : {current_date})...")
        self.model.fit(X, Y)
        
        # Mise à jour des statuts
        self.is_trained = True
        self.last_train_date = current_date
        return True

    def predict(self, forecast_df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_trained:
            raise RuntimeError("Le modèle ML n'est pas entraîné.")
        
        result_df = forecast_df.copy()
        X_pred = result_df[self.features]
        predictions = self.model.predict(X_pred)
        result_df['PV'] = np.maximum(0, predictions)
        return result_df


class HybridPVPredictor:
    def __init__(self, physical_predictor, ml_predictor, weather_provider):
        self.physical = physical_predictor
        self.ml = ml_predictor
        self.weather = weather_provider

    def predict_for_day(self, target_date: date) -> pd.DataFrame:
        print(f"\n[Hybrid Agent] Récupération de la météo pour le {target_date}...")
        forecast_df = self.weather.get_forecast(target_date)

        # On demande au modèle ML de s'entraîner (il vérifiera lui-même son delta_train)
        self.ml.train(current_date=target_date)

        if self.ml.is_trained:
            print("[Hybrid Agent] Utilisation du modèle ML.")
            result_df = self.ml.predict(forecast_df)
            result_df['prediction_mode'] = 'ML'
        else:
            print("[Hybrid Agent] Utilisation du modèle Physique.")
            result_df = self.physical.predict(forecast_df)
            result_df['prediction_mode'] = 'Physique'

        return result_df


# --- EXEMPLE D'UTILISATION ---
if __name__ == "__main__":
    # Paramètres du panneau
    P_STC = 3000  
    BETA = -0.004 
    NOCT = 45
    NB_PANELS = 1
    
    # Coordonnées du smart building (ex: Paris)
    LAT = 48.8566
    LON = 2.3522
    
    test_historic_path = Path("./energy_planner/data/historic/donnees_reelles_historique.csv")
    
    # Instanciation des composants
    weather_api = WeatherProvider(latitude=LAT, longitude=LON)
    phys_agent = PhysicalPVPredictor(p_stc=P_STC, beta=BETA, noct=NOCT, nb_panels=NB_PANELS)
    ml_agent = MLPVPredictor(historic_real_csv=test_historic_path, min_samples=720, delta_train_days=7)
    
    # L'agent hybride rassemble le tout
    smart_agent = HybridPVPredictor(
        physical_predictor=phys_agent, 
        ml_predictor=ml_agent, 
        weather_provider=weather_api
    )

    # Exécution de la prédiction pour demain
    demain = date.today() + timedelta(days=1)
    previsions = smart_agent.predict_for_day(demain)

    # Affichage des résultats
    print("\n--- RÉSULTATS DE LA PRÉDICTION ---")
    heures_journee = previsions[(previsions['heure'] >= 8) & (previsions['heure'] <= 18)]
    
    # On affiche les colonnes les plus pertinentes
    colonnes_a_afficher = ['time', 'Tout', 'G', 'PV', 'prediction_mode']
    print(heures_journee[colonnes_a_afficher].to_string(index=False))