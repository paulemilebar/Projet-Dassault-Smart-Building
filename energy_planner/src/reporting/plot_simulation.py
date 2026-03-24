"""
Generate an interactive HTML report from the simulation metrics CSV.

Usage (from repo root):
    python energy_planner/src/reporting/plot_simulation.py \
        --input-csv  energy_planner/data/processed/simulation_metrics.csv \
        --output-html energy_planner/data/processed/simulation_report.html
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Plot builders
# ---------------------------------------------------------------------------

def _prediction_mae_plot(df: pd.DataFrame, fig: go.Figure, row: int, col: int) -> None:
    """MAE per prediction variable over days."""
    variables = [
        ("mae_pv",    "PV",    "#f4a261"),
        ("mae_pfixe", "Pfixe", "#2a9d8f"),
        ("mae_pflex", "Pflex", "#e76f51"),
        ("mae_cbuy",  "Cbuy",  "#264653"),
    ]
    for col_name, label, color in variables:
        if col_name not in df.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=df["date"], y=df[col_name],
                mode="lines+markers",
                name=f"MAE {label}",
                line=dict(color=color),
                marker=dict(size=6),
            ),
            row=row, col=col,
        )


def _prediction_rmse_plot(df: pd.DataFrame, fig: go.Figure, row: int, col: int) -> None:
    """RMSE per prediction variable over days."""
    variables = [
        ("rmse_pv",    "PV",    "#f4a261"),
        ("rmse_pfixe", "Pfixe", "#2a9d8f"),
        ("rmse_pflex", "Pflex", "#e76f51"),
        ("rmse_cbuy",  "Cbuy",  "#264653"),
    ]
    for col_name, label, color in variables:
        if col_name not in df.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=df["date"], y=df[col_name],
                mode="lines+markers",
                name=f"RMSE {label}",
                line=dict(color=color, dash="dot"),
                marker=dict(size=6),
            ),
            row=row, col=col,
        )


def _optimizer_cost_plot(df: pd.DataFrame, fig: go.Figure, row: int, col: int) -> None:
    """J_predicted vs J_oracle per day."""
    if "J_predicted" not in df.columns or df["J_predicted"].isna().all():
        fig.add_annotation(
            text="Optimizer (CPLEX) not available — no cost data",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=14, color="gray"),
            row=row, col=col,
        )
        return

    fig.add_trace(
        go.Bar(
            x=df["date"], y=df["J_predicted"],
            name="J predicted",
            marker_color="#e9c46a",
            opacity=0.85,
        ),
        row=row, col=col,
    )
    fig.add_trace(
        go.Bar(
            x=df["date"], y=df["J_oracle"],
            name="J oracle (perfect info)",
            marker_color="#2a9d8f",
            opacity=0.85,
        ),
        row=row, col=col,
    )


def _regret_plot(df: pd.DataFrame, fig: go.Figure, row: int, col: int) -> None:
    """Daily regret = J_predicted - J_oracle."""
    if "regret" not in df.columns or df["regret"].isna().all():
        return
    fig.add_trace(
        go.Bar(
            x=df["date"], y=df["regret"],
            name="Regret (J_pred − J_oracle)",
            marker_color="#e76f51",
            opacity=0.85,
        ),
        row=row, col=col,
    )
    # Zero line for reference
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=row, col=col)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_report(input_csv: Path, output_html: Path) -> None:
    df = pd.read_csv(input_csv)

    has_optimizer = "J_predicted" in df.columns and not df["J_predicted"].isna().all()
    n_rows = 3 if has_optimizer else 2

    subplot_titles = [
        "Prediction MAE per variable",
        "Prediction RMSE per variable",
    ]
    if has_optimizer:
        subplot_titles += [
            "Optimizer cost: predicted plan vs oracle (perfect info)",
            "Daily regret  =  J_predicted − J_oracle",
        ]

    fig = make_subplots(
        rows=n_rows,
        cols=2 if has_optimizer else 1,
        subplot_titles=subplot_titles,
        shared_xaxes=False,
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    _prediction_mae_plot(df, fig, row=1, col=1)
    _prediction_rmse_plot(df, fig, row=2, col=1)

    if has_optimizer:
        _optimizer_cost_plot(df, fig, row=1, col=2)
        _regret_plot(df, fig, row=2, col=2)

    # Y-axis labels
    fig.update_yaxes(title_text="MAE", row=1, col=1)
    fig.update_yaxes(title_text="RMSE", row=2, col=1)
    if has_optimizer:
        fig.update_yaxes(title_text="Cost (€)", row=1, col=2)
        fig.update_yaxes(title_text="Regret (€)", row=2, col=2)
        fig.update_layout(barmode="group")

    fig.update_layout(
        title=dict(
            text=(
                f"<b>Simulation report</b>  —  {df['date'].iloc[0]} "
                f"→ {df['date'].iloc[-1]}  ({len(df)} days)"
            ),
            font=dict(size=18),
        ),
        height=600 * n_rows,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
    )

    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_html))
    print(f"Report saved to: {output_html}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot simulation metrics as an HTML report.")
    p.add_argument(
        "--input-csv",
        type=str,
        default="energy_planner/data/processed/simulation_metrics.csv",
    )
    p.add_argument(
        "--output-html",
        type=str,
        default="energy_planner/data/processed/simulation_report.html",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_report(Path(args.input_csv), Path(args.output_html))
