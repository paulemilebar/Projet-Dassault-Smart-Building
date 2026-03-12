from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from .features import generate_synthetic_demand_history
from .predictor_user_demand import DEFAULT_MODEL_PATH, UserDemandForecastAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the user-demand forecasting models on held-out synthetic days.")
    parser.add_argument("--start-date", type=str, required=True, help="Held-out evaluation start date in YYYY-MM-DD format.")
    parser.add_argument("--num-days", type=int, default=1, help="Number of held-out days to evaluate.")
    parser.add_argument("--base-seed", type=int, default=1000, help="Seed used to generate held-out synthetic context.")
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH), help="Path to the saved demand model bundle.")
    parser.add_argument("--output-csv", type=str, default="", help="Optional path to save hour-level prediction vs actual rows.")
    return parser.parse_args()


def _parse_date(raw: str) -> date:
    return datetime.strptime(raw, "%Y-%m-%d").date()


def _metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main() -> None:
    args = parse_args()
    start_date = _parse_date(args.start_date)
    end_date = start_date + pd.Timedelta(days=args.num_days - 1)

    held_out = generate_synthetic_demand_history(
        start_date=start_date,
        num_days=args.num_days,
        base_seed=args.base_seed,
    )

    agent = UserDemandForecastAgent(model_path=args.model_path)
    prediction = agent.predict_dataframe(held_out)

    evaluation_df = held_out[["hour", "day", "month", "year", "Pfixe", "Pflex"]].copy()
    evaluation_df = evaluation_df.rename(
        columns={
            "Pfixe": "Pfixe_reel",
            "Pflex": "Pflex_reel",
        }
    )
    evaluation_df["Pfixe_predit"] = prediction["Pfixe_predit"].to_numpy()
    evaluation_df["Pflex_predit"] = prediction["Pflex_predit"].to_numpy()

    metrics = {
        "pfix": _metrics(evaluation_df["Pfixe_reel"], evaluation_df["Pfixe_predit"]),
        "pflex": _metrics(evaluation_df["Pflex_reel"], evaluation_df["Pflex_predit"]),
    }

    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        evaluation_df.to_csv(output_path, index=False)
        print(f"Saved hour-level evaluation rows to: {output_path}")

    print(f"Evaluation window: {start_date.isoformat()} -> {end_date.isoformat()}")
    print("Held-out metrics:")
    for target, target_metrics in metrics.items():
        print(f"  {target}:")
        for name, value in target_metrics.items():
            print(f"    {name}: {value:.6f}")


if __name__ == "__main__":
    main()
