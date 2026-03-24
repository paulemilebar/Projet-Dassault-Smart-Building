from __future__ import annotations
import os
from typing import Any
import pandas as pd
from visualization import build_visualization_frame, summarize_dispatch


def build_optimization_summary_payload(
    predicted_inputs: pd.DataFrame,
    plan_df: pd.DataFrame,
    *,
    initial_battery_kwh: float,
    battery_capacity_kwh: float,
) -> dict[str, Any]:
    """
    Build a compact business summary from optimizer inputs and outputs.
    """
    viz_df = build_visualization_frame(
        predicted_inputs,
        plan_df,
        initial_battery_kwh=initial_battery_kwh,
        battery_capacity_kwh=battery_capacity_kwh,
    )
    dispatch_summary = summarize_dispatch(viz_df)

    total_buy_cost = float(
        (viz_df["Pin"] * viz_df["Cbuy_pred_eur_per_kWh"]).sum()
    )
    total_sell_revenue = float(
        (viz_df["Pgo"] * viz_df["Csell_pred_eur_per_kWh"]).sum()
    )
    flex_requested = float(viz_df["Pflex_pred_kW"].sum())
    flex_served = float(viz_df["flex_served_kW"].sum())
    curtailed_flex = float(viz_df["curtailed_flex_kW"].sum())
    total_pv = float(viz_df["PV"].sum())
    self_consumed_pv = float(viz_df["self_consumed_pv_kW"].sum())

    peak_import_idx = viz_df["Pin"].idxmax()
    peak_export_idx = viz_df["Pgo"].idxmax()
    peak_charge_idx = viz_df["Pch"].idxmax()
    peak_discharge_idx = viz_df["Pdis"].idxmax()

    return {
        "totals": {
            "served_demand_kWh": round(dispatch_summary["total_served_demand_kWh"], 3),
            "pv_used_kWh": round(total_pv, 3),
            "pv_self_consumed_kWh": round(self_consumed_pv, 3),
            "grid_import_kWh": round(dispatch_summary["grid_import_kWh"], 3),
            "grid_export_kWh": round(dispatch_summary["grid_export_kWh"], 3),
            "battery_charge_kWh": round(dispatch_summary["battery_charge_kWh"], 3),
            "battery_discharge_kWh": round(dispatch_summary["battery_discharge_kWh"], 3),
            "battery_start_kWh": round(float(initial_battery_kwh), 3),
            "battery_end_kWh": round(float(viz_df["Ebat"].iloc[-1]), 3),
            "battery_peak_kWh": round(dispatch_summary["peak_battery_kWh"], 3),
            "flex_requested_kWh": round(flex_requested, 3),
            "flex_served_kWh": round(flex_served, 3),
            "flex_curtailed_kWh": round(curtailed_flex, 3),
            "energy_purchase_cost_eur": round(total_buy_cost, 3),
            "energy_sale_revenue_eur": round(total_sell_revenue, 3),
            "net_energy_cost_eur": round(total_buy_cost - total_sell_revenue, 3),
        },
        "ratios": {
            "flex_served_pct": round(100.0 * flex_served / max(flex_requested, 1e-9), 1),
            "pv_self_consumed_pct": round(100.0 * self_consumed_pv / max(total_pv, 1e-9), 1),
            "grid_import_share_pct": round(
                100.0 * dispatch_summary["grid_import_kWh"] / max(dispatch_summary["total_served_demand_kWh"], 1e-9),
                1,
            ),
        },
        "peaks": {
            "grid_import": {
                "hour": int(viz_df.loc[peak_import_idx, "hour"]),
                "value_kW": round(float(viz_df.loc[peak_import_idx, "Pin"]), 3),
            },
            "grid_export": {
                "hour": int(viz_df.loc[peak_export_idx, "hour"]),
                "value_kW": round(float(viz_df.loc[peak_export_idx, "Pgo"]), 3),
            },
            "battery_charge": {
                "hour": int(viz_df.loc[peak_charge_idx, "hour"]),
                "value_kW": round(float(viz_df.loc[peak_charge_idx, "Pch"]), 3),
            },
            "battery_discharge": {
                "hour": int(viz_df.loc[peak_discharge_idx, "hour"]),
                "value_kW": round(float(viz_df.loc[peak_discharge_idx, "Pdis"]), 3),
            },
        },
        "regime_hours": dispatch_summary["regime_hours"],
    }


def build_rule_based_summary(payload: dict[str, Any]) -> str:
    """
    Deterministic fallback summary used when LLM access is unavailable.
    """
    totals = payload["totals"]
    ratios = payload["ratios"]
    peaks = payload["peaks"]
    regimes = payload["regime_hours"]
    dominant_regime = max(regimes, key=regimes.get) if regimes else "Balanced"

    return (
        "Resume optimisation sur 24h : "
        f"le batiment sert {totals['served_demand_kWh']:.2f} kWh de demande. "
        f"Seuls {totals['grid_import_kWh']:.2f} kWh sont achetes au reseau, soit "
        f"{ratios['grid_import_share_pct']:.1f}% de la demande servie "
        f"(cout {totals['energy_purchase_cost_eur']:.2f} EUR). "
        f"Le batiment revend {totals['grid_export_kWh']:.2f} kWh "
        f"(revenu {totals['energy_sale_revenue_eur']:.2f} EUR), "
        f"pour un cout net de {totals['net_energy_cost_eur']:.2f} EUR. "
        f"La batterie charge {totals['battery_charge_kWh']:.2f} kWh, decharge "
        f"{totals['battery_discharge_kWh']:.2f} kWh et passe de "
        f"{totals['battery_start_kWh']:.2f} a {totals['battery_end_kWh']:.2f} kWh "
        f"(pic a {totals['battery_peak_kWh']:.2f} kWh). "
        f"La flexibilite servie represente {ratios['flex_served_pct']:.1f}% de la demande flexible. "
        f"Le pic d'achat reseau est atteint a {peaks['grid_import']['hour']:02d}h avec "
        f"{peaks['grid_import']['value_kW']:.2f} kW, et le pic de vente a "
        f"{peaks['grid_export']['hour']:02d}h avec {peaks['grid_export']['value_kW']:.2f} kW. "
        f"Le regime dominant est '{dominant_regime}' sur {regimes.get(dominant_regime, 0)} heures."
    )


def _build_llm_prompt(payload: dict[str, Any]) -> str:
    totals = payload["totals"]
    ratios = payload["ratios"]
    peaks = payload["peaks"]
    regimes = payload["regime_hours"]

    return f"""
Tu es un expert en gestion d'energie autonome pour les smart building.
Redige une explication détaillé en francais, clair et naturel de la gestion des sources d'énergie (PV, Batteries et acaht au fournisseur) pour la demande de l'utilisateur.
Tu dois rester strictement fidele aux chiffres fournis et ne rien inventer. Fais attention aux points et virgules dans les Puissances, j'aimerais que tu arrondisses à l'unité près.
Mentionne explicitement que le cout d'achat ne porte que sur l'energie importee P_in, pas sur toute la demande servie. Mentionne aussi les ventes P_go, la batterie, le PV, le cout net et les heures de pics importantes.

Chiffres a résumer mais en langage naturel. Explique comme si tu étais un expert en énergie:
- Demande servie: {totals["served_demand_kWh"]} kWh
- Achat reseau P_in: {totals["grid_import_kWh"]} kWh
- Vente reseau P_go: {totals["grid_export_kWh"]} kWh
- Cout achat energie: {totals["energy_purchase_cost_eur"]} EUR
- Revenu vente energie: {totals["energy_sale_revenue_eur"]} EUR
- Cout net: {totals["net_energy_cost_eur"]} EUR
- Production PV utilisee: {totals["pv_used_kWh"]} kWh
- PV autoconsomme: {totals["pv_self_consumed_kWh"]} kWh
- Batterie chargee: {totals["battery_charge_kWh"]} kWh
- Batterie dechargee: {totals["battery_discharge_kWh"]} kWh
- Batterie debut: {totals["battery_start_kWh"]} kWh
- Batterie fin: {totals["battery_end_kWh"]} kWh
- Batterie max: {totals["battery_peak_kWh"]} kWh
- Flexibilite servie: {totals["flex_served_kWh"]} kWh
- Flexibilite coupee: {totals["flex_curtailed_kWh"]} kWh
- Taux de service flexibilite: {ratios["flex_served_pct"]} %
- Taux autoconsommation PV: {ratios["pv_self_consumed_pct"]} %
- Part de la demande couverte par achat reseau: {ratios["grid_import_share_pct"]} %
- Pic achat reseau: heure {peaks["grid_import"]["hour"]}, valeur {peaks["grid_import"]["value_kW"]} kW
- Pic vente reseau: heure {peaks["grid_export"]["hour"]}, valeur {peaks["grid_export"]["value_kW"]} kW
- Pic charge batterie: heure {peaks["battery_charge"]["hour"]}, valeur {peaks["battery_charge"]["value_kW"]} kW
- Pic decharge batterie: heure {peaks["battery_discharge"]["hour"]}, valeur {peaks["battery_discharge"]["value_kW"]} kW
- Repartition des regimes horaires: {regimes}
""".strip()


def try_generate_llm_summary(
    payload: dict[str, Any],
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> tuple[str, str]:
    """
    Return (summary_text, source), where source is 'llm' or 'template'.
    """
    try:
        from openai import OpenAI
    except ImportError:
        return build_rule_based_summary(payload), "template_missing_openai_package"

    resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
    model_name = model or os.getenv("OPENAI_MODEL")
    resolved_base_url = base_url or os.getenv("OPENAI_BASE_URL")
    if not resolved_api_key:
        return build_rule_based_summary(payload), "template_missing_api_key"
    if not model_name:
        return build_rule_based_summary(payload), "template_missing_model"

    try:
        client_kwargs: dict[str, Any] = {"api_key": resolved_api_key}
        if resolved_base_url:
            client_kwargs["base_url"] = resolved_base_url
        client = OpenAI(**client_kwargs)
        response = client.responses.create(
            model=model_name,
            input=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "Tu rédiges des résumés de résultats d'optimisation des sources d'energie dans les smart buildings qui sont auto gérés énergétiquement avec des panneaux photovoltaïques, une batterie et un achat au main grid et une revente si surplus. "
                                "Tu es precis, concis et tu n'inventes aucuns chiffres. Fais également attention aux virgules et points et je veux que tu arrondisses à l'unité près."
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": _build_llm_prompt(payload)}],
                },
            ],
        )
    except Exception:
        return build_rule_based_summary(payload), "template_llm_error"
    return response.output_text.strip(), "llm"
