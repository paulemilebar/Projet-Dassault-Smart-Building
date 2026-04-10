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
    optimizer_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build a compact business summary from optimizer inputs and outputs
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

    if optimizer_state:
        payload["optimization_model"] = {
            "horizon_hours": 24,
            "objective": {
                "sense": "minimize",
                "expression": (
                    "sum_t ((C_grid_buy[t] + C_emissions_grid) * Pin[t]"
                    " - C_grid_sell[t] * Pgo[t]"
                    " + C_emissions_PV * PV[t]"
                    " - C_bat * Ebat[t]"
                    " - C_L * P_flex[t] * S[t])"
                ),
                "notes": [
                    "Le terme constant C_L * P_flex[t] de la puissance flexible demandée par l'utilisateur non servie est omis car il ne change pas l'optimum.",
                    "Le terme -C_bat * Ebat[t] valorise un niveau de batterie eleve sur l'horizon.",
                    "Le cout d'achat ne s'applique qu'a l'energie importee du main grid P_in.",
                ],
            },
            "variables": {
                "Pin": f"Puissance importée du main grid dans [0, {optimizer_state['P_g_max_import']}]",
                "Pgo": f"Puissance exportée (vendue) du main grid dans [0, {optimizer_state['P_g_max_export']}]",
                "PV": "Production Photo Voltaïque utilisée, continue dans [0, PV_max[t]]",
                "Pch": f"Charge de la batterie dans [0, {min(optimizer_state['P_ch_max'], optimizer_state['P_dis_max'])}]",
                "Pdis": f"Décharge de la batterie dans [0, {min(optimizer_state['P_ch_max'], optimizer_state['P_dis_max'])}]",
                "Ebat": f"Energie batterie continue dans [0, {battery_capacity_kwh}]",
                "S": "Variable binaire de service de la charge flexible (1 = servie, 0 = coupee)",
                "A": "Variable binaire d'activation décharge batterie",
                "B": "Variable binaire d'activation charge batterie",
            },
            "constraints": [
                "Equilibre de puissance horaire: PV[t] + Pdis[t] + Pin[t] = P_fixed[t] + S[t] * P_flex[t] + Pch[t] + Pgo[t].",
                f"Etat initial batterie: Ebat[0] = {round(float(initial_battery_kwh), 3)} + Pch[0] - Pdis[0].",
                "Dynamique batterie pour t >= 1: Ebat[t] = Ebat[t-1] + Pch[t] - Pdis[t].",
                "Activation décharge: Pdis[t] <= P_bat_max * A[t].",
                "Activation charge: Pch[t] <= P_bat_max * B[t].",
                f"Disponibilite energie au depart: Pdis[0] <= {round(float(initial_battery_kwh), 3)}.",
                f"Capacite restante au depart: Pch[0] <= {round(float(battery_capacity_kwh - initial_battery_kwh), 3)}.",
                "Disponibilite decharge pour t >= 1: Pdis[t] <= Ebat[t-1].",
                f"Capacite batterie pour t >= 1: Pch[t] + Ebat[t-1] <= {round(float(battery_capacity_kwh), 3)}.",
                "Non simultaneite batterie: A[t] + B[t] <= 1.",
            ],
            "parameters": {
                "C_L": optimizer_state["C_L"],
                "C_bat": optimizer_state["C_bat"],
                "C_emissions_grid": optimizer_state["C_emissions_grid"],
                "C_emissions_PV": optimizer_state["C_emissions_PV"],
                "P_g_max_import": optimizer_state["P_g_max_import"],
                "P_g_max_export": optimizer_state["P_g_max_export"],
                "E_bat_max": battery_capacity_kwh,
                "P_bat_max": min(optimizer_state["P_ch_max"], optimizer_state["P_dis_max"]),
                "E_bat_init": initial_battery_kwh,
            },
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
        "Resumé de l'optimisation sur 24h : "
        f"Le batiment sert au total {totals['served_demand_kWh']:.2f} kWh de puissance demandée par l'utilisateur. "
        f"Seuls {totals['grid_import_kWh']:.2f} kWh sont achetés au réseau, soit "
        f"{ratios['grid_import_share_pct']:.1f}% de la demande servie "
        f"(cout {totals['energy_purchase_cost_eur']:.2f} EUR). "
        f"Le smart building revend {totals['grid_export_kWh']:.2f} kWh "
        f"(revenu {totals['energy_sale_revenue_eur']:.2f} EUR) au main grid, "
        f"pour un cout net de {totals['net_energy_cost_eur']:.2f} EUR. "
        f"Au total, sur 24h, la batterie charge {totals['battery_charge_kWh']:.2f} kWh, decharge "
        f"{totals['battery_discharge_kWh']:.2f} kWh et passe de "
        f"{totals['battery_start_kWh']:.2f} a {totals['battery_end_kWh']:.2f} kWh "
        f"(avec un pic à {totals['battery_peak_kWh']:.2f} kWh). "
        f"La flexibilite servie représente {ratios['flex_served_pct']:.1f}% de la demande flexible. "
        f"Le pic d'achat réseau est atteint a {peaks['grid_import']['hour']:02d}h avec "
        f"{peaks['grid_import']['value_kW']:.2f} kW, et le pic de vente à "
        f"{peaks['grid_export']['hour']:02d}h avec {peaks['grid_export']['value_kW']:.2f} kW. "
        f"Le régime dominant est '{dominant_regime}' sur {regimes.get(dominant_regime, 0)} heures."
    )

    if comparison:
        summary += (
            f" Par rapport a la baseline, le coût passe de {comparison['baseline_cost_eur']:.2f} EUR initialement (sans PV, batteries et optimisation)"
            f"à {comparison['optimizer_cost_eur']:.2f} EUR "
            f"({comparison['cost_delta_eur']:.2f} EUR, {comparison['cost_delta_pct']:.1f}%). "
            f"Les émissions, quant à elles, passent de {comparison['baseline_co2_kg']:.2f} kgCO2 "
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
    optimization_model = payload.get("optimization_model")

    lines = [
        "Tu es un expert en gestion d'energie autonome pour les smart building.",
        "Redige une explication détaillée en francais, claire et naturelle de la gestion des sources d'energie (PV, batteries et achat au fournisseur) pour la demande de l'utilisateur.",
        "Tu dois rester strictement fidele aux chiffres fournis et ne rien inventer. Fais attention aux points et virgules dans les puissances, et arrondis a l'unite pres.",
        "Mentionne explicitement que le cout d'achat ne porte que sur l'energie importee P_in, pas sur toute la demande servie. Mentionne aussi les ventes P_go, la batterie, le PV, le cout net et les heures de pics importantes.",
        "Si des valeurs de comparaison baseline sont fournies, intègre-les explicitement pour comparer les resultats actuels de l'optimiseur a cette baseline afin de souligner les gains en argent et émissions de gaz à effet de serre de cette gestion d'énergie optimisée comparée à une baseline d'un bâtiment classique. DERNIER POINT : Fais le en quelques paragraphes mais sans rajouter de double ** ou de symboles bizarres.",
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
                "- Réference baseline pour comparer les resultats actuels de l'optimiseur:",
                f"  cout baseline: {comparison['baseline_cost_eur']} EUR",
                f"  cout optimiseur: {comparison['optimizer_cost_eur']} EUR",
                f"  ecart cout optimiseur - baseline: {comparison['cost_delta_eur']} EUR ({comparison['cost_delta_pct']} %)",
                f"  emissions baseline: {comparison['baseline_co2_kg']} kgCO2",
                f"  emissions optimiseur: {comparison['optimizer_co2_kg']} kgCO2",
                f"  ecart emissions optimiseur - baseline: {comparison['co2_delta_kg']} kgCO2 ({comparison['co2_delta_pct']} %)",
            ]
        )

    if optimization_model:
        lines.extend(
            [
                "",
                "Tu dois aussi fournir dans ton résumé une explication du modèle d'optimisation. C'est à dire pourquoi l'optimiseur a finalement choisi de renvoyer ces valeurs optimales au regards des contraintes qu'il avait et de la fonction objectif. Modele d'optimisation a expliquer dans le resume:",
                f"- Horizon: {optimization_model['horizon_hours']} heures",
                f"- Fonction objectif: {optimization_model['objective']['sense']} {optimization_model['objective']['expression']}",
                "- Notes sur l'objectif:",
            ]
        )
        lines.extend(f"  {note}" for note in optimization_model["objective"]["notes"])
        lines.append("- Variables de decision:")
        lines.extend(
            f"  {name}: {description}"
            for name, description in optimization_model["variables"].items()
        )
        lines.append("- Contraintes:")
        lines.extend(f"  {constraint}" for constraint in optimization_model["constraints"])
        lines.append("- Parametres du modele:")
        lines.extend(
            f"  {name}: {value}"
            for name, value in optimization_model["parameters"].items()
        )
        lines.extend(
            [
                "",
                "Dans ton resumé, explique aussi pourquoi ces contraintes conduisent aux décisions observees"
                " (imports reseau, export PV, charge/decharge batterie, service ou non de la flexibilite),"
                " toujours sans inventer de chiffres ni de regles absentes de ce modele.",
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
    Return (summary_text, source), where source is 'llm' or 'template'
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
        "Tu rediges des résumés de resultats d'optimisation des sources d'energie dans les smart buildings et explique les décisions observés de l'optimiseur."
        "qui sont auto gérés energetiquement avec des panneaux photovoltaiques, une batterie, un achat au main grid (réseau) "
        "et une revente si surplus. Tu es precis, concis et tu n'inventes aucun chiffre. "
        "Fais egalement attention aux virgules et points et arrondis a l'unite pres. Fais également attention au petites étoiles et à la mise en forme."
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
