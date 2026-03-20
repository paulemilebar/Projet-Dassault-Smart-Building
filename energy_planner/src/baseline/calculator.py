import pandas as pd

class BaselineCalculator:
    def __init__(self):
        pass

    def compute_grid_only(self, df: pd.DataFrame, state: dict) -> dict:
        """
        Baseline : Achat total au réseau. 
        On récupère le facteur d'émission depuis 'state'.
        """
        # Demande totale
        demand = df["Pfix_pred_kW"] + df["Pflex_pred_kW"]
        
        # Coût financier
        cost = (demand * df["Cbuy_pred_eur_per_kWh"]).sum()
        
        # Émissions : On va chercher dans 'state' car ce n'est pas dans le DataFrame
        emissions_factor = state.get("C_emissions_grid", 0.05) # 0.05 par défaut si absent
        emissions = (demand * emissions_factor).sum()
        
        return {"cost": cost, "emissions": emissions}

    def compute_optimizer_performance(self, plan_df: pd.DataFrame, inputs_df: pd.DataFrame, state: dict) -> dict:
        """Bilan réel de l'optimiseur."""
        # Coût = (Import * PrixAchat) - (Export * PrixVente)
        cost_import = (plan_df["Pin"] * inputs_df["Cbuy_pred_eur_per_kWh"]).sum()
        revenue_export = (plan_df["Pgo"] * inputs_df["Csell_pred_eur_per_kWh"]).sum()
        
        # Emissions : Grid + PV (on utilise 'state' pour les deux facteurs)
        emissions_grid = (plan_df["Pin"] * state.get("C_emissions_grid", 0.05)).sum()
        emissions_pv = (plan_df["PV"] * state.get("C_emissions_PV", 0.0)).sum()
        
        return {
            "cost": cost_import - revenue_export,
            "emissions": emissions_grid + emissions_pv
        }