from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize held-out user-demand evaluation results.")
    parser.add_argument("--input-csv", type=str, required=True, help="CSV produced by evaluate_user_demand_model.py.")
    parser.add_argument("--output-html", type=str, required=True, help="HTML report path.")
    return parser.parse_args()


def _metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _build_time_label(df: pd.DataFrame) -> pd.Series:
    return (
        df["year"].astype(int).astype(str)
        + "-"
        + df["month"].astype(int).astype(str).str.zfill(2)
        + "-"
        + df["day"].astype(int).astype(str).str.zfill(2)
        + " "
        + df["hour"].astype(int).astype(str).str.zfill(2)
        + ":00"
    )


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_csv)
    df = pd.read_csv(input_path)
    df["time_label"] = _build_time_label(df)

    pfix_metrics = _metrics(df["Pfixe_reel"], df["Pfixe_predit"])
    pflex_metrics = _metrics(df["Pflex_reel"], df["Pflex_predit"])

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.42, 0.42, 0.16],
        specs=[[{"type": "xy"}], [{"type": "xy"}], [{"type": "table"}]],
        subplot_titles=("Fixed Demand: Actual vs Predicted", "Flexible Demand: Actual vs Predicted", "Metrics Summary"),
    )

    fig.add_trace(
        go.Scatter(
            x=df["time_label"],
            y=df["Pfixe_reel"],
            mode="lines",
            name="Pfixe actual",
            line=dict(color="#1d3557", width=3),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["time_label"],
            y=df["Pfixe_predit"],
            mode="lines",
            name="Pfixe predicted",
            line=dict(color="#e76f51", width=2, dash="dash"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["time_label"],
            y=df["Pflex_reel"],
            mode="lines",
            name="Pflex actual",
            line=dict(color="#2a9d8f", width=3),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["time_label"],
            y=df["Pflex_predit"],
            mode="lines",
            name="Pflex predicted",
            line=dict(color="#f4a261", width=2, dash="dash"),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Table(
            header=dict(
                values=["Target", "MAE", "RMSE", "R2"],
                fill_color="#0f172a",
                font=dict(color="white", size=12),
                align="left",
            ),
            cells=dict(
                values=[
                    ["Pfixe", "Pflex"],
                    [f"{pfix_metrics['mae']:.6f}", f"{pflex_metrics['mae']:.6f}"],
                    [f"{pfix_metrics['rmse']:.6f}", f"{pflex_metrics['rmse']:.6f}"],
                    [f"{pfix_metrics['r2']:.6f}", f"{pflex_metrics['r2']:.6f}"],
                ],
                fill_color="#f8fafc",
                align="left",
            ),
        ),
        row=3,
        col=1,
    )

    fig.update_layout(
        title="User Demand Forecast Evaluation",
        template="plotly_white",
        height=980,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=60, r=40, t=90, b=60),
    )
    fig.update_yaxes(title_text="kW", row=1, col=1)
    fig.update_yaxes(title_text="kW", row=2, col=1)
    fig.update_xaxes(title_text="Hour", tickangle=30, row=2, col=1)

    output_path = Path(args.output_html)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    print(f"Saved evaluation report to: {output_path}")


if __name__ == "__main__":
    main()
