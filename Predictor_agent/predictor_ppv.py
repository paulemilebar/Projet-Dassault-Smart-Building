import requests
import numpy as np
from datetime import datetime


class PVForecastAgent:
    """
    Agent de prévision de puissance photovoltaïque horaire
    basé sur données météo Open-Meteo + modèle physique PV.
    """

    def __init__(self,
                 latitude,
                 longitude,
                 P_STC,
                 nb_panels,
                 gamma=-0.004,
                 NOCT=45):

        # Localisation
        self.latitude = latitude
        self.longitude = longitude

        # Paramètres système PV
        self.P_STC = P_STC              # W par panneau
        self.nb_panels = nb_panels
        self.gamma = gamma              # coefficient température
        self.NOCT = NOCT

        # Constantes
        self.G_STC = 1000               # W/m²
        self.T_STC = 25                 # °C


    def _get_weather_forecast(self):

        url = (
            "https://api.open-meteo.com/v1/forecast?"
            f"latitude={self.latitude}&longitude={self.longitude}"
            "&hourly=temperature_2m,shortwave_radiation"
            "&forecast_days=1"
            "&timezone=auto"
        )

        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        temps = np.array(data["hourly"]["temperature_2m"])
        irradiance = np.array(data["hourly"]["shortwave_radiation"])
        timestamps = data["hourly"]["time"]

        return temps, irradiance, timestamps


    def _compute_pv_power(self, temps, irradiance):

        # Température cellule
        T_cell = temps + (self.NOCT - 20) / 800 * irradiance

        # Puissance par panneau
        P_panel = (
            self.P_STC
            * (irradiance / self.G_STC)
            * (1 + self.gamma * (T_cell - self.T_STC))
        )

        # Pas de puissance négative
        P_panel = np.maximum(P_panel, 0)

        # Puissance totale installation
        P_total = self.nb_panels * P_panel

        return P_total

    def run(self):

        temps, irradiance, timestamps = self._get_weather_forecast()

        P_total = self._compute_pv_power(temps, irradiance)

        return {
            "agent": "pv_forecaster",
            "location": {
                "latitude": self.latitude,
                "longitude": self.longitude
            },
            "horizon_hours": len(P_total),
            "resolution": "1h",
            "unit": "W",
            "timestamp_start": timestamps[0],
            "timestamps": timestamps,
            "values": P_total.tolist()
        }

