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
    comparison_results: dict[str, Any] | None = None,
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
    dispatch_summary = summarize_dispatch(viz_df, comparison_results=comparison_results)

    total_buy_cost = float((viz_df["Pin"] * viz_df["Cbuy_pred_eur_per_kWh"]).sum())
    total_sell_revenue = float((viz_df["Pgo"] * viz_df["Csell_pred_eur_per_kWh"]).sum())
    flex_requested = float(viz_df["Pflex_pred_kW"].sum())
    flex_served = float(viz_df["flex_served_kW"].sum())
    curtailed_flex = float(viz_df["curtailed_flex_kW"].sum())
    total_pv = float(viz_df["PV"].sum())
    self_consumed_pv = float(viz_df["self_consumed_pv_kW"].sum())

    peak_import_idx = viz_df["Pin"].idxmax()
    peak_export_idx = viz_df["Pgo"].idxmax()
    peak_charge_idx = viz_df["Pch"].idxmax()
    peak_discharge_idx = viz_df["Pdis"].idxmax()

    payload = {
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
                100.0
                * dispatch_summary["grid_import_kWh"]
                / max(dispatch_summary["total_served_demand_kWh"], 1e-9),
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

    comparison = dispatch_summary.get("comparison")
    if comparison:
        baseline_cost = float(comparison["base_cost"])
        optimizer_cost = float(comparison["opt_cost"])
        baseline_co2 = float(comparison["base_co2"])
        optimizer_co2 = float(comparison["opt_co2"])

        payload["comparison"] = {
            "baseline_cost_eur": round(baseline_cost, 3),
            "optimizer_cost_eur": round(optimizer_cost, 3),
            "cost_delta_eur": round(optimizer_cost - baseline_cost, 3),
            "cost_delta_pct": round(
                100.0 * (optimizer_cost - baseline_cost) / max(abs(baseline_cost), 1e-9),
                1,
            ),
            "baseline_co2_kg": round(baseline_co2, 3),
            "optimizer_co2_kg": round(optimizer_co2, 3),
            "co2_delta_kg": round(optimizer_co2 - baseline_co2, 3),
            "co2_delta_pct": round(
                100.0 * (optimizer_co2 - baseline_co2) / max(abs(baseline_co2), 1e-9),
                1,
            ),
        }

    return payload


def build_rule_based_summary(payload: dict[str, Any]) -> str:
    """
    Deterministic fallback summary used when LLM access is unavailable.
    """
    totals = payload["totals"]
    ratios = payload["ratios"]
    peaks = payload["peaks"]
    regimes = payload["regime_hours"]
    comparison = payload.get("comparison")
    dominant_regime = max(regimes, key=regimes.get) if regimes else "Balanced"

    summary = (
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

    if comparison:
        summary += (
            f" Par rapport a la baseline, le cout passe de {comparison['baseline_cost_eur']:.2f} EUR "
            f"a {comparison['optimizer_cost_eur']:.2f} EUR "
            f"({comparison['cost_delta_eur']:.2f} EUR, {comparison['cost_delta_pct']:.1f}%). "
            f"Les emissions passent de {comparison['baseline_co2_kg']:.2f} kgCO2 "
            f"a {comparison['optimizer_co2_kg']:.2f} kgCO2 "
            f"({comparison['co2_delta_kg']:.2f} kgCO2, {comparison['co2_delta_pct']:.1f}%)."
        )

    return summary


def _build_llm_prompt(payload: dict[str, Any]) -> str:
    totals = payload["totals"]
    ratios = payload["ratios"]
    peaks = payload["peaks"]
    regimes = payload["regime_hours"]
    comparison = payload.get("comparison")

    lines = [
        "Tu es un expert en gestion d'energie autonome pour les smart building.",
        "Redige une explication detaillee en francais, claire et naturelle de la gestion des sources d'energie (PV, batteries et achat au fournisseur) pour la demande de l'utilisateur.",
        "Tu dois rester strictement fidele aux chiffres fournis et ne rien inventer. Fais attention aux points et virgules dans les puissances, et arrondis a l'unite pres.",
        "Mentionne explicitement que le cout d'achat ne porte que sur l'energie importee P_in, pas sur toute la demande servie. Mentionne aussi les ventes P_go, la batterie, le PV, le cout net et les heures de pics importantes.",
        "Si des valeurs de comparaison baseline sont fournies, integre-les explicitement pour comparer les resultats actuels de l'optimiseur a cette baseline.",
        "",
        "Chiffres a resumer mais en langage naturel. Explique comme si tu etais un expert en energie:",
        f"- Demande servie: {totals['served_demand_kWh']} kWh",
        f"- Achat reseau P_in: {totals['grid_import_kWh']} kWh",
        f"- Vente reseau P_go: {totals['grid_export_kWh']} kWh",
        f"- Cout achat energie: {totals['energy_purchase_cost_eur']} EUR",
        f"- Revenu vente energie: {totals['energy_sale_revenue_eur']} EUR",
        f"- Cout net: {totals['net_energy_cost_eur']} EUR",
        f"- Production PV utilisee: {totals['pv_used_kWh']} kWh",
        f"- PV autoconsomme: {totals['pv_self_consumed_kWh']} kWh",
        f"- Batterie chargee: {totals['battery_charge_kWh']} kWh",
        f"- Batterie dechargee: {totals['battery_discharge_kWh']} kWh",
        f"- Batterie debut: {totals['battery_start_kWh']} kWh",
        f"- Batterie fin: {totals['battery_end_kWh']} kWh",
        f"- Batterie max: {totals['battery_peak_kWh']} kWh",
        f"- Flexibilite servie: {totals['flex_served_kWh']} kWh",
        f"- Flexibilite coupee: {totals['flex_curtailed_kWh']} kWh",
        f"- Taux de service flexibilite: {ratios['flex_served_pct']} %",
        f"- Taux autoconsommation PV: {ratios['pv_self_consumed_pct']} %",
        f"- Part de la demande couverte par achat reseau: {ratios['grid_import_share_pct']} %",
        f"- Pic achat reseau: heure {peaks['grid_import']['hour']}, valeur {peaks['grid_import']['value_kW']} kW",
        f"- Pic vente reseau: heure {peaks['grid_export']['hour']}, valeur {peaks['grid_export']['value_kW']} kW",
        f"- Pic charge batterie: heure {peaks['battery_charge']['hour']}, valeur {peaks['battery_charge']['value_kW']} kW",
        f"- Pic decharge batterie: heure {peaks['battery_discharge']['hour']}, valeur {peaks['battery_discharge']['value_kW']} kW",
        f"- Repartition des regimes horaires: {regimes}",
    ]

    if comparison:
        lines.extend(
            [
                "- Reference baseline pour comparer les resultats actuels de l'optimiseur:",
                f"  cout baseline: {comparison['baseline_cost_eur']} EUR",
                f"  cout optimiseur: {comparison['optimizer_cost_eur']} EUR",
                f"  ecart cout optimiseur - baseline: {comparison['cost_delta_eur']} EUR ({comparison['cost_delta_pct']} %)",
                f"  emissions baseline: {comparison['baseline_co2_kg']} kgCO2",
                f"  emissions optimiseur: {comparison['optimizer_co2_kg']} kgCO2",
                f"  ecart emissions optimiseur - baseline: {comparison['co2_delta_kg']} kgCO2 ({comparison['co2_delta_pct']} %)",
            ]
        )

    return "\n".join(lines).strip()


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

    client_kwargs: dict[str, Any] = {"api_key": resolved_api_key}
    if resolved_base_url:
        client_kwargs["base_url"] = resolved_base_url
    client = OpenAI(**client_kwargs)

    system_prompt = (
        "Tu rediges des resumes de resultats d'optimisation des sources d'energie dans les smart buildings "
        "qui sont auto geres energetiquement avec des panneaux photovoltaiques, une batterie, un achat au reseau "
        "et une revente si surplus. Tu es precis, concis et tu n'inventes aucun chiffre. "
        "Fais egalement attention aux virgules et points et arrondis a l'unite pres."
    )
    user_prompt = _build_llm_prompt(payload)

    response_error: Exception | None = None
    try:
        response = client.responses.create(
            model=model_name,
            input=[
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_prompt}],
                },
            ],
        )
        output_text = getattr(response, "output_text", "") or ""
        if output_text.strip():
            return output_text.strip(), "llm"
    except Exception as exc:
        response_error = exc

    try:
        chat_response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        message = chat_response.choices[0].message.content
        if isinstance(message, str) and message.strip():
            return message.strip(), "llm_chat"
    except Exception as chat_exc:
        error_message = str(chat_exc)
        if response_error is not None:
            error_message = f"responses_error={response_error}; chat_error={chat_exc}"
        return build_rule_based_summary(payload), f"template_llm_error: {error_message}"

    error_message = str(response_error) if response_error is not None else "empty_llm_response"
    return build_rule_based_summary(payload), f"template_llm_error: {error_message}"
