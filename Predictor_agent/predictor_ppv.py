import pandas as pd
import requests
from datetime import date, timedelta
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

class PhysicalPVPredictor:
    def __init__(self, p_stc: float, beta: float, noct: float, nb_panels: int, latitude: float, longitude: float):
        """
        Initialise l'agent prédicteur basé sur le modèle physique.
        
        :param p_stc: Puissance nominale du panneau sous conditions standards (en Watts).
        :param beta: Coefficient de température de puissance (ex: -0.004 pour -0.4%/°C).
        :param latitude: Latitude du smart building (pour l'API météo).
        :param longitude: Longitude du smart building.
        """
        self.p_stc = p_stc
        self.beta = beta
        self.noct = noct
        self.nb_panels = nb_panels
        self.latitude = latitude
        self.longitude = longitude

    def _fetch_weather_forecast(self, target_date: date) -> pd.DataFrame:
        """
        Récupère les prévisions météo horaires via l'API gratuite Open-Meteo.
        """
        date_str = target_date.strftime("%Y-%m-%d")
        
        # On demande la température à 2m et l'irradiance globale (shortwave_radiation)
        url = (f"https://api.open-meteo.com/v1/forecast?"
               f"latitude={self.latitude}&longitude={self.longitude}&"
               f"hourly=temperature_2m,shortwave_radiation&"
               f"timezone=auto&start_date={date_str}&end_date={date_str}")
        
        response = requests.get(url)
        response.raise_for_status() # Lève une erreur si l'API échoue
        data = response.json()
        
        # Création du DataFrame avec les données récupérées
        df = pd.DataFrame({
            'time': pd.to_datetime(data['hourly']['time']),
            'Tout': data['hourly']['temperature_2m'],
            'G': data['hourly']['shortwave_radiation']
        })
        return df

    def predict_day(self, target_date: date) -> pd.DataFrame:
        """
        Prédit la production PV heure par heure pour le jour donné.
        """
        # 1. Récupération de la météo
        forecast_df = self._fetch_weather_forecast(target_date)
        
        # 2. Application de la formule physique
        # P(t) = P_STC * (G(t) / 1000) * [1 + beta * (Tcell(t) - 25)]
        T_cell = forecast_df['Tout'] + (self.noct - 20) / 800 * forecast_df['G']
        P_panel = self.p_stc * (forecast_df['G'] / 1000.0) * (1 + self.beta * (T_cell - 25.0))
        P_panel = np.maximum(P_panel, 0)

        # Puissance totale installation
        P_total = self.nb_panels * P_panel
        
        forecast_df['PPV'] = P_total
        
        return forecast_df



class MLPVPredictor:
    def __init__(self, historic_real_csv: Path, min_samples: int = 720):
        self.historic_real_csv = historic_real_csv
        self.min_samples = min_samples
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False
        
        # On définit explicitement les variables attendues par le modèle
        self.features = ['heure', 'jour', 'mois', 'annee', 'Tout_reel', 'G']

    def train(self) -> bool:
        """
        Entraîne le modèle en utilisant les colonnes temporelles natives du dataset.
        """
        if not self.historic_real_csv.exists():
            return False

        df = pd.read_csv(self.historic_real_csv)
        
        if len(df) < self.min_samples:
            return False

        # On utilise directement tes colonnes
        # Assure-toi que les noms correspondent exactement à ton CSV
        X = df[self.features]
        Y = df['PV_reel'] # Remplace par le nom exact de ta colonne de production réelle

        self.model.fit(X, Y)
        self.is_trained = True
        return True

    def predict(self, forecast_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prédit la production. Génère les colonnes temporelles si elles sont absentes du forecast.
        """
        if not self.is_trained:
            raise RuntimeError("Le modèle ML n'est pas entraîné. Impossible de prédire.")

        result_df = forecast_df.copy()
        
        # Si le dataframe de prévision (venant de l'API météo) n'a pas tes colonnes, on les crée
        if 'heure' not in result_df.columns:
            result_df['time'] = pd.to_datetime(result_df['time'])
            result_df['heure'] = result_df['time'].dt.hour
            result_df['jour'] = result_df['time'].dt.day
            result_df['mois'] = result_df['time'].dt.month
            result_df['annee'] = result_df['time'].dt.year
            
        # On s'assure de passer les variables dans le même ordre que lors de l'entraînement
        X_pred = result_df[self.features]

        predictions = self.model.predict(X_pred)
        result_df['PPV'] = np.maximum(0, predictions)
        
        return result_df



# --- EXEMPLE D'UTILISATION ---
if __name__ == "__main__":
    # Paramètres du panneau (ex: installation de 3000 Wc, coeff de -0.4% par degré)
    P_STC = 3000  
    BETA = -0.004 
    NOCT = 45
    NB_PANELS = 1
    
    # Coordonnées du smart building (ex: Paris)
    LAT = 48.8566
    LON = 2.3522
    
    # Création de l'agent
    agent = PhysicalPVPredictor(p_stc=P_STC, beta=BETA, noct=NOCT, nb_panels=NB_PANELS, latitude=LAT, longitude=LON)
    
    # Prédiction pour demain
    demain = date.today() + timedelta(days=1)
    
    print(f"--- Prévision de production PV pour le {demain} ---")
    previsions = agent.predict_day(demain)
    
    # Affichage des résultats (heures de jour pour plus de clarté, ex: 8h à 18h)
    heures_journee = previsions.iloc[8:19]
    print(heures_journee.to_string(index=False))
    