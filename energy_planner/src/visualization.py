from __future__ import annotations

import html
import textwrap
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as exc: 
    raise ImportError(
        "Plotly is required for visualization. Install it with `pip install plotly`."
    ) from exc


_PREDICTED_REQUIRED = [
    "hour",
    "PV_pred_kW",
    "Pfix_pred_kW",
    "Pflex_pred_kW",
    "Cbuy_pred_eur_per_kWh",
    "Csell_pred_eur_per_kWh",
]

_PLAN_REQUIRED = ["hour", "Pin", "Pgo", "PV", "Pch", "Pdis", "Ebat", "S"]

_REGIME_STYLES: dict[str, tuple[int, str, str]] = {
    "Grid charge": (0, "#d1495b", "#f6c4cb"),
    "Grid support": (1, "#edae49", "#fde7bd"),
    "Battery support": (2, "#00798c", "#b9e6eb"),
    "Solar charge": (3, "#66a182", "#d6efe2"),
    "PV export": (4, "#2a6f97", "#c8deeb"),
    "Balanced": (5, "#6c757d", "#e3e6e8"),
}

_REGIME_LABELS = {
    "Grid charge": "GC",
    "Grid support": "GS",
    "Battery support": "BS",
    "Solar charge": "SC",
    "PV export": "EX",
    "Balanced": "BA",
}


def _validate_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _classify_regime(row: pd.Series) -> str:
    if row["Pin"] > 1e-9 and row["Pch"] > 1e-9:
        return "Grid charge"
    if row["Pgo"] > 1e-9:
        return "PV export"
    if row["Pdis"] > 1e-9 and row["Pin"] <= 1e-9:
        return "Battery support"
    if row["Pch"] > 1e-9 and row["PV"] >= row["total_demand_kW"] - 1e-9:
        return "Solar charge"
    if row["Pin"] > 1e-9:
        return "Grid support"
    return "Balanced"


def build_visualization_frame(
    predicted_inputs: pd.DataFrame,
    plan_df: pd.DataFrame,
    *,
    initial_battery_kwh: float | None = None,
    battery_capacity_kwh: float | None = None,
) -> pd.DataFrame:
    """
    Merge optimizer inputs and decision variables into a plotting-ready hourly frame.
    """
    _validate_columns(predicted_inputs, _PREDICTED_REQUIRED, "predicted_inputs")
    _validate_columns(plan_df, _PLAN_REQUIRED, "plan_df")

    df = predicted_inputs.merge(plan_df, on="hour", how="inner", validate="one_to_one").copy()
    df = df.sort_values("hour").reset_index(drop=True)

    df["flex_served_kW"] = df["Pflex_pred_kW"] * df["S"]
    df["total_demand_kW"] = df["Pfix_pred_kW"] + df["flex_served_kW"]
    df["total_requested_kW"] = df["Pfix_pred_kW"] + df["Pflex_pred_kW"]
    df["curtailed_flex_kW"] = df["Pflex_pred_kW"] - df["flex_served_kW"]
    df["net_grid_kW"] = df["Pin"] - df["Pgo"]
    df["net_battery_kW"] = df["Pdis"] - df["Pch"]
    df["self_consumed_pv_kW"] = (df["PV"] - df["Pgo"]).clip(lower=0.0)

    cap = float(battery_capacity_kwh) if battery_capacity_kwh is not None else float(df["Ebat"].max())
    cap = max(cap, 1e-9)
    df["battery_soc_pct"] = 100.0 * df["Ebat"] / cap

    if initial_battery_kwh is None:
        first_delta = df.loc[0, "Pch"] - df.loc[0, "Pdis"]
    else:
        first_delta = df.loc[0, "Ebat"] - float(initial_battery_kwh)
    df["battery_delta_kWh"] = df["Ebat"].diff().fillna(first_delta)

    df["regime"] = df.apply(_classify_regime, axis=1)
    df["regime_code"] = df["regime"].map(lambda name: _REGIME_STYLES[name][0]).astype(int)
    return df


def create_dispatch_dashboard(
    viz_df: pd.DataFrame,
    *,
    title: str = "",
    comparison_results: dict[str, Any] | None = None
) -> go.Figure:
    """
    Build an interactive multi-panel dashboard around optimizer decisions.
    """
    required = [
        "hour",
        "total_demand_kW",
        "Pfix_pred_kW",
        "flex_served_kW",
        "PV",
        "Pin",
        "Pgo",
        "Pch",
        "Pdis",
        "Ebat",
        "net_grid_kW",
        "Cbuy_pred_eur_per_kWh",
        "Csell_pred_eur_per_kWh",
        "regime",
        "regime_code",
    ]
    _validate_columns(viz_df, required, "viz_df")

    hours = viz_df["hour"]
    regime_colors = [_REGIME_STYLES[name][1] for name in viz_df["regime"]]
    regime_scale = [
        [code / 5, color]
        for _, (code, color, _) in _REGIME_STYLES.items()
        for color in [color]
    ]

    fig = make_subplots(
        rows=5, # Passé de 4 à 5
        cols=3, # On utilise 3 colonnes pour les compteurs sur la première ligne
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.15, 0.30, 0.18, 0.18, 0.10], # La ligne 1 est plus petite
        specs=[
            [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}], # Ligne des compteurs
            [{"secondary_y": False, "colspan": 3}, None, None],
            [{"secondary_y": True, "colspan": 3}, None, None],
            [{"secondary_y": True, "colspan": 3}, None, None],
            [{"secondary_y": False, "colspan": 3}, None, None],
        ],
        subplot_titles=(
            "Key Performance Indicators", "", "", # Titres pour la ligne 1
            "Demand vs Dispatch",
            "Battery Dynamics",
            "Grid and Price Signals",
            "Operating Regimes",
        ),
    )

    # --- NOUVEAU : AJOUT DES INDICATEURS DE COMPARAISON ---
    if comparison_results:
        # 1. Économie Financière (€)
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=comparison_results.get("opt_cost", 0),
                delta={'reference': comparison_results.get("base_cost", 0), 'relative': True, 'position': "bottom", 'increasing': {'color': "#d1495b"}, 'decreasing': {'color': "#66a182"}},
                title={"text": "Total Cost (€)<br><span style='font-size:0.8em;color:gray'>vs Baseline</span>"},
                number={'prefix': "€", 'font': {'size': 40}},
            ),
            row=1, col=1
        )

        # 2. Émissions CO2 (kg)
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=comparison_results.get("opt_co2", 0),
                delta={'reference': comparison_results.get("base_co2", 0), 'relative': True, 'position': "bottom", 'increasing': {'color': "#d1495b"}, 'decreasing': {'color': "#66a182"}},
                title={"text": "CO2 Emissions (kg)<br><span style='font-size:0.8em;color:gray'>vs Baseline</span>"},
                number={'suffix': " kg", 'font': {'size': 40}},
            ),
            row=1, col=2
        )

        # 3. Taux d'Autoconsommation (%)
        # Calcul : (PV utilisé / PV total) * 100
        pv_total = viz_df["PV"].sum()
        pv_used = (viz_df["PV"] - viz_df["Pgo"]).sum()
        self_suff = (pv_used / pv_total * 100) if pv_total > 0 else 0
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=self_suff,
                title={"text": "Self-Consumption Rate"},
                gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#edae49"}},
                number={'suffix': "%", 'font': {'size': 35}},
            ),
            row=1, col=3
        )

    fig.add_trace(
        go.Scatter(
            x=hours,
            y=viz_df["total_demand_kW"],
            mode="lines+markers",
            name="Served demand",
            line=dict(color="#111111", width=3),
            hovertemplate="Hour %{x}<br>Served demand %{y:.2f} kW<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=hours,
            y=viz_df["Pfix_pred_kW"],
            name="Fixed demand",
            marker_color="#7c7f85",
            opacity=0.65,
            hovertemplate="Hour %{x}<br>Fixed demand %{y:.2f} kW<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=hours,
            y=viz_df["flex_served_kW"],
            name="Flexible demand served",
            marker_color="#c0d6df",
            hovertemplate="Hour %{x}<br>Flexible served %{y:.2f} kW<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=hours,
            y=viz_df["PV"],
            mode="lines",
            name="PV available",
            fill="tozeroy",
            line=dict(color="#f4a261", width=2),
            fillcolor="rgba(244, 162, 97, 0.28)",
            hovertemplate="Hour %{x}<br>PV %{y:.2f} kW<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=hours,
            y=viz_df["Pin"],
            name="Grid import",
            marker_color="#d1495b",
            hovertemplate="Hour %{x}<br>Grid import %{y:.2f} kW<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=hours,
            y=viz_df["Pdis"],
            name="Battery discharge",
            marker_color="#00798c",
            hovertemplate="Hour %{x}<br>Battery discharge %{y:.2f} kW<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=hours,
            y=-viz_df["Pch"],
            name="Battery charge",
            marker_color="#66a182",
            customdata=viz_df["Pch"],
            hovertemplate="Hour %{x}<br>Battery charge %{customdata:.2f} kW<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=hours,
            y=-viz_df["Pgo"],
            name="Grid export",
            marker_color="#2a6f97",
            customdata=viz_df["Pgo"],
            hovertemplate="Hour %{x}<br>Grid export %{customdata:.2f} kW<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=hours,
            y=viz_df["Ebat"],
            mode="lines+markers",
            name="Battery energy",
            line=dict(color="#0d3b66", width=3),
            fill="tozeroy",
            fillcolor="rgba(13, 59, 102, 0.12)",
            hovertemplate="Hour %{x}<br>Battery %{y:.2f} kWh<extra></extra>",
        ),
        row=3,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            x=hours,
            y=viz_df["Pch"],
            name="Charge power",
            marker_color="#8ecae6",
            hovertemplate="Hour %{x}<br>Charge %{y:.2f} kW<extra></extra>",
        ),
        row=3,
        col=1,
        secondary_y=True,
    )
    fig.add_trace(
        go.Bar(
            x=hours,
            y=-viz_df["Pdis"],
            name="Discharge power",
            marker_color="#219ebc",
            customdata=viz_df["Pdis"],
            hovertemplate="Hour %{x}<br>Discharge %{customdata:.2f} kW<extra></extra>",
        ),
        row=3,
        col=1,
        secondary_y=True,
    )

    fig.add_trace(
        go.Bar(
            x=hours,
            y=viz_df["net_grid_kW"],
            name="Net grid exchange",
            marker_color=regime_colors,
            customdata=viz_df["regime"],
            hovertemplate="Hour %{x}<br>Net grid %{y:.2f} kW<br>Regime %{customdata}<extra></extra>",
        ),
        row=4,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=hours,
            y=viz_df["Cbuy_pred_eur_per_kWh"],
            mode="lines+markers",
            name="Buy price",
            line=dict(color="#6d597a", width=2),
            hovertemplate="Hour %{x}<br>Buy price %{y:.3f} eur/kWh<extra></extra>",
        ),
        row=4,
        col=1,
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=hours,
            y=viz_df["Csell_pred_eur_per_kWh"],
            mode="lines",
            name="Sell price",
            line=dict(color="#b56576", width=2, dash="dot"),
            hovertemplate="Hour %{x}<br>Sell price %{y:.3f} eur/kWh<extra></extra>",
        ),
        row=4,
        col=1,
        secondary_y=True,
    )

    fig.add_trace(
        go.Heatmap(
            x=hours,
            y=["Regime"],
            z=[viz_df["regime_code"].tolist()],
            text=[[_REGIME_LABELS[name] for name in viz_df["regime"].tolist()]],
            customdata=[viz_df["regime"].tolist()],
            texttemplate="%{text}",
            textfont=dict(size=10, color="#0f172a"),
            colorscale=regime_scale,
            showscale=False,
            hovertemplate=(
                "Hour %{x}<br>"
                "Mode %{customdata}<br>"
                "Meaning: %{meta}<extra></extra>"
            ),
            meta=[
                [
                    "Buying from the grid and storing part of it in the battery."
                    if name == "Grid charge"
                    else "Using the grid to help cover building demand."
                    if name == "Grid support"
                    else "Using the battery to help cover building demand."
                    if name == "Battery support"
                    else "Solar is covering demand and charging the battery."
                    if name == "Solar charge"
                    else "Solar production is higher than building needs, so surplus is exported."
                    if name == "PV export"
                    else "No dominant behavior; the system is in a neutral state."
                    for name in viz_df["regime"].tolist()
                ]
            ],
        ),
        row=5,
        col=1,
    )

    '''for hour, regime in zip(hours, viz_df["regime"]):
        _, _, fill_color = _REGIME_STYLES[regime]
        fig.add_vrect(
            x0=hour - 0.5,
            x1=hour + 0.5,
            fillcolor=fill_color,
            opacity=0.12,
            line_width=0,
            row=2,
            col=1,
        )'''

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=1220,
        barmode="relative",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.11,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(255,255,255,0.85)",
        ),
        margin=dict(l=70, r=55, t=145, b=210),
        hovermode="x unified",
        font=dict(family="Segoe UI, Arial, sans-serif", size=12),
        paper_bgcolor="#f8fafc",
        plot_bgcolor="#ffffff",
    )

    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(24)),
        ticktext=[f"{hour:02d}:00" for hour in range(24)],
        tickangle=28,
        tickfont=dict(size=11, color="#334155"),
        automargin=True,
        row=5,
        col=1,
    )
    fig.update_xaxes(title_text="Hour of day", row=5, col=1)
    fig.update_xaxes(showgrid=False, row=5, col=1)

    fig.update_yaxes(title_text="kW", row=2, col=1)
    fig.update_yaxes(title_text="Battery energy (kWh)", row=3, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Battery power (kW)", row=3, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Net grid (kW)", row=4, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Price (eur/kWh)", row=4, col=1, secondary_y=True)
    fig.update_yaxes(showticklabels=False, row=5, col=1)

    fig.update_annotations(font=dict(size=16, color="#1f2937"))

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.0,
        y=-0.16,
        showarrow=False,
        align="left",
        text=(
            "<span style='color:#d1495b'><b>GC</b></span> Grid charge: buying from the grid and charging the battery.  |  "
            "<span style='color:#edae49'><b>GS</b></span> Grid support: the grid is helping power the building.  |  "
            "<span style='color:#00798c'><b>BS</b></span> Battery support: the battery is supplying part of the demand.<br>"
            "<span style='color:#66a182'><b>SC</b></span> Solar charge: solar covers demand and charges the battery.  |  "
            "<span style='color:#2a6f97'><b>EX</b></span> PV export: extra solar energy is exported to the grid.  |  "
            "<span style='color:#6c757d'><b>BA</b></span> Balanced: no strong mode dominates in that hour."
        ),
        bgcolor="rgba(255,255,255,0.98)",
        bordercolor="rgba(203,213,225,0.95)",
        borderwidth=1,
        borderpad=10,
        font=dict(size=11, color="#0f172a"),
    )

    return fig


def save_dashboard_html(fig: go.Figure, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs="cdn")
    return path


def _summary_text_to_html(summary_text: str) -> str:
    blocks: list[str] = []
    list_items: list[str] = []

    for raw_line in summary_text.splitlines():
        line = raw_line.strip()
        if not line:
            if list_items:
                blocks.append("<ul>" + "".join(list_items) + "</ul>")
                list_items = []
            continue

        if line.startswith(("- ", "* ")):
            list_items.append(f"<li>{html.escape(line[2:].strip())}</li>")
            continue

        if list_items:
            blocks.append("<ul>" + "".join(list_items) + "</ul>")
            list_items = []
        blocks.append(f"<p>{html.escape(line)}</p>")

    if list_items:
        blocks.append("<ul>" + "".join(list_items) + "</ul>")

    if not blocks:
        return "<p>No summary available.</p>"
    return "".join(blocks)


def save_dashboard_report_html(
    fig: go.Figure,
    output_path: str | Path,
    *,
    summary_text: str,
    summary_source: str | None = None,
    model_name: str | None = None,
    title: str = "Dispatch Dashboard Report",
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    figure_html = fig.to_html(include_plotlyjs="cdn", full_html=False)
    source_bits = []
    if summary_source:
        source_bits.append(f"Source: {html.escape(summary_source)}")
    if model_name:
        source_bits.append(f"Model: {html.escape(model_name)}")
    source_html = " | ".join(source_bits)

    report_html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>
    body {{
      margin: 0;
      background: #f8fafc;
      color: #0f172a;
      font-family: "Segoe UI", Arial, sans-serif;
    }}
    .page {{
      max-width: 1400px;
      margin: 0 auto;
      padding: 24px;
    }}
    .card {{
      background: #ffffff;
      border: 1px solid #cbd5e1;
      border-radius: 16px;
      box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
      overflow: hidden;
    }}
    .summary {{
      margin-top: 20px;
      padding: 24px 28px;
      line-height: 1.65;
    }}
    .summary h2 {{
      margin: 0 0 8px 0;
      font-size: 1.4rem;
    }}
    .meta {{
      margin: 0 0 16px 0;
      color: #475569;
      font-size: 0.95rem;
    }}
    .summary p {{
      margin: 0 0 12px 0;
    }}
    .summary ul {{
      margin: 0 0 12px 20px;
      padding: 0;
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="card">{figure_html}</div>
    <section class="card summary">
      <h2>Resume naturel du dispatch</h2>
      <p class="meta">{html.escape(source_html) if source_html else ""}</p>
      {_summary_text_to_html(summary_text)}
    </section>
  </div>
</body>
</html>
"""
    path.write_text(report_html, encoding="utf-8")
    return path


def add_natural_language_summary(
    fig: go.Figure,
    summary_text: str,
    *,
    source_label: str | None = None,
) -> go.Figure:
    """
    Append a natural-language summary block below the dashboard.
    """
    if not summary_text or not summary_text.strip():
        return fig

    wrapped_lines: list[str] = []
    for raw_line in summary_text.strip().splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.startswith(("- ", "* ", "• ")):
            bullet = stripped[:2]
            content = stripped[2:].strip()
            chunks = textwrap.wrap(content, width=95) or [content]
            wrapped_lines.append(f"{bullet}{chunks[0]}")
            wrapped_lines.extend(f"&nbsp;&nbsp;{chunk}" for chunk in chunks[1:])
        else:
            wrapped_lines.extend(textwrap.wrap(stripped, width=105) or [stripped])

    clean_text = "<br>".join(wrapped_lines)
    if source_label:
        clean_text = (
            f"<b>Natural-language summary</b> "
            f"<span style='color:#64748b'>(source: {source_label})</span><br>{clean_text}"
        )
    else:
        clean_text = f"<b>Summary and explanation from our experts</b><br>{clean_text}"

    summary_line_count = max(len(wrapped_lines) + 1, 5)
    current_height = fig.layout.height or 1220
    current_bottom = fig.layout.margin.b if fig.layout.margin and fig.layout.margin.b is not None else 210

    extra_space = 28 * summary_line_count
    fig.update_layout(height=current_height + extra_space)
    fig.update_layout(margin=dict(b=max(current_bottom, 210) + extra_space))
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.0,
        y=-0.28,
        showarrow=False,
        align="left",
        text=clean_text,
        bgcolor="rgba(255,255,255,0.98)",
        bordercolor="rgba(203,213,225,0.95)",
        borderwidth=1,
        borderpad=12,
        font=dict(size=12, color="#0f172a"),
    )
    return fig


def summarize_dispatch(viz_df: pd.DataFrame) -> dict[str, Any]:
    """
    Return compact metrics that are useful in notebooks or reports.
    """
    _validate_columns(
        viz_df,
        ["PV", "Pin", "Pgo", "Pch", "Pdis", "total_demand_kW", "Ebat", "regime"],
        "viz_df",
    )

    regime_hours = viz_df["regime"].value_counts().sort_index().to_dict()
    return {
        "total_served_demand_kWh": round(float(viz_df["total_demand_kW"].sum()), 3),
        "total_pv_kWh": round(float(viz_df["PV"].sum()), 3),
        "grid_import_kWh": round(float(viz_df["Pin"].sum()), 3),
        "grid_export_kWh": round(float(viz_df["Pgo"].sum()), 3),
        "battery_charge_kWh": round(float(viz_df["Pch"].sum()), 3),
        "battery_discharge_kWh": round(float(viz_df["Pdis"].sum()), 3),
        "peak_battery_kWh": round(float(viz_df["Ebat"].max()), 3),
        "regime_hours": regime_hours,
    }
